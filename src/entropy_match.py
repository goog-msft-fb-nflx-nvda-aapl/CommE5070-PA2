"""Sampling-parameter selection by entropy-histogram / perplexity matching (Compose & Embellish; lecture 05 s78-79).
Reference = the model's per-token predictive entropy (and NLL) on REAL held-out 32-bar crops (teacher forcing).
For every sampling config: generate N pieces, teacher-force the model over them, and compare the per-token entropy
histogram (Jensen-Shannon divergence) and the mean NLL (perplexity) of the generated tokens with the reference."""
import argparse, itertools, json, os, subprocess, sys
import numpy as np
import torch
import torch.nn.functional as F
import generate as G

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--tok2idx", required=True)
ap.add_argument("--val_crops", required=True, help="json [{song_name, tokens}] real 32-bar crops of held-out songs")
ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--pos_per_beat", type=int, required=True)
ap.add_argument("--temps", type=float, nargs="+", default=[0.8, 0.9, 1.0, 1.1, 1.2])
ap.add_argument("--top_ps", type=float, nargs="+", default=[0.9, 0.95, 0.98, 1.0])
ap.add_argument("--bar_biases", type=float, nargs="+", default=[0.0])
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

dev = "cuda"
t2i = json.load(open(a.tok2idx)); i2t = {v: k for k, v in t2i.items()}
t2i["tempo_ids"] = {i for k, i in t2i.items() if isinstance(i, int) and k.startswith("Tempo_")}
t2i["chord_ids"] = {i for k, i in t2i.items() if isinstance(i, int) and k.startswith("Chord_")}
model, ck = G.load_model(a.ckpt, dev)
ctx = ck["config"]["n_positions"]
BAR, BOS = t2i["Bar"], t2i["<BOS>"]
BINS = np.linspace(0, 6.5, 66)


@torch.no_grad()
def token_stats(seqs):
    """per-token (entropy, NLL of the actual token) under the model, teacher forced; long sequences are cut at bar boundaries
    into chunks of <= ctx-1 tokens with a <BOS> prefix (the training-crop layout)."""
    H, NL = [], []
    for s in seqs:
        s = [t for t in s if t != BOS]
        bars = [i for i, t in enumerate(s) if t == BAR] + [len(s)]
        start = 0
        while start < len(s):
            end = start
            for b in bars:
                if b > start and b - start <= ctx - 2:
                    end = b
            if end <= start:
                end = min(len(s), start + ctx - 2)
            x = torch.tensor([[BOS] + s[start:end]], device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(input_ids=x).logits[0, :-1].float()
            lp = F.log_softmax(lg, -1)
            H.append((-(lp.exp() * lp).sum(-1)).cpu().numpy()); NL.append((-lp.gather(1, x[0, 1:, None])[:, 0]).cpu().numpy())
            start = end
    return np.concatenate(H), np.concatenate(NL)


def hist(h):
    c, _ = np.histogram(np.clip(h, 0, 6.49), bins=BINS)
    return c / c.sum()


def js(p, q):
    m = 0.5 * (p + q)
    kl = lambda x, y: float(np.sum(np.where(x > 0, x * np.log(x / np.maximum(y, 1e-12)), 0)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


val = [[t2i[t] for t in s["tokens"] if t in t2i] for s in json.load(open(a.val_crops))]
Hv, NLv = token_stats(val)
ref = hist(Hv)
print(f"reference (real held-out crops, n={len(val)}): mean entropy {Hv.mean():.4f}, mean NLL {NLv.mean():.4f} (perplexity {np.exp(NLv.mean()):.3f})", flush=True)
rows = [{"reference": True, "n_crops": len(val), "mean_entropy": float(Hv.mean()), "mean_nll": float(NLv.mean())}]
here = os.path.dirname(os.path.abspath(__file__))
for T, p, b in itertools.product(a.temps, a.top_ps, a.bar_biases):
    torch.manual_seed(a.seed)
    typed = {"temps": [T] * 4, "bar_bias": b} if b != 0 else None
    seqs, _ = G.sample(model, [BOS], a.n, 32, t2i, T, 0, p, ctx, typed=typed)
    Hg, NLg = token_stats(seqs)
    row = {"temperature": T, "top_p": p, "bar_bias": b, "n": a.n, "seed": a.seed, "mean_entropy": float(Hg.mean()), "mean_nll": float(NLg.mean()),
           "js_entropy_hist": js(hist(Hg), ref), "abs_dnll": abs(float(NLg.mean() - NLv.mean())), "mean_tokens": float(np.mean([len(s) for s in seqs]))}
    # objective metrics of the same pieces through the unmodified eval_basic
    toks = [{"song_name": f"g{i:02d}", "tokens": [i2t[t] for t in s]} for i, s in enumerate(seqs)]
    tmp = a.out + f".tmp_{T}_{p}_{b}.json"; json.dump(toks, open(tmp, "w"))
    ev = subprocess.run([sys.executable, os.path.join(here, "..", "scripts", "eval_basic.py"), "--tok2idx_path", a.tok2idx, "--sequences_path", tmp,
                         "--output_csv", tmp + ".csv"], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.path.join(here, "..")})
    import re
    for k in ("H1", "H4", "GS"):
        m = re.search(rf"Average {k}: ([\d.]+)", ev.stdout); row[k] = float(m.group(1)) if m else None
    os.remove(tmp); os.path.exists(tmp + ".csv") and os.remove(tmp + ".csv")
    rows.append(row)
    print(f"T={T} p={p} bias={b}: JS {row['js_entropy_hist']:.4f}  d|NLL| {row['abs_dnll']:.3f}  H(sampled) {row['mean_entropy']:.3f}  H1/H4/GS {row['H1']}/{row['H4']}/{row['GS']}", flush=True)
    json.dump(rows, open(a.out, "w"), indent=1)
