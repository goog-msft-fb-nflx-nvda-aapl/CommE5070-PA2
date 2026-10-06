"""Generation for the unconditional GPT-2: modes `uncond` (from <BOS>) and `cont`
(continue tokenized prompts). Writes .mid files and a tokens json that
eval_basic.py can read directly. Pieces that do not reach n_bars within the
context are discarded and resampled (rate is printed)."""
import argparse, json, os, bisect
import numpy as np
import torch
import pretty_midi
from transformers import GPT2Config, GPT2LMHeadModel, StoppingCriteria, StoppingCriteriaList


def detokenize(tokens, p, out_name="PIANO"):
    """TA-style tokens (Bar / Pos_x / Tempo_x | Pitch_x Vel_x Dur_x) -> PrettyMIDI. Assumes 4/4.
    Returns (midi, n_malformed)."""
    bar, i, bad = -1, 0, 0
    notes, tempos = [], []
    while i < len(tokens):
        t = tokens[i]
        if t == "Bar":
            bar += 1; i += 1
        elif t.startswith("Pos_") and bar >= 0:
            beat = 4 * bar + int(t[4:]) / p
            nxt = tokens[i + 1:i + 4]
            if nxt and nxt[0].startswith("Tempo_"):
                tempos.append((beat, int(nxt[0][6:]))); i += 2
            elif len(nxt) == 3 and nxt[0].startswith("Pitch_") and nxt[1].startswith("Vel_") and nxt[2].startswith("Dur_"):
                notes.append((beat, int(nxt[0][6:]), int(nxt[1][4:]), int(nxt[2][4:]) / p)); i += 4
            else:
                bad += 1; i += 1
        else:
            if t not in ("<BOS>", "<EOS>", "<PAD>"):
                bad += 1
            i += 1
    tempos.sort()
    q0 = tempos[0][1] if tempos else 120
    bp = [0.0] + [b for b, _ in tempos if b > 0]
    qs = [q0] + [q for b, q in tempos if b > 0]
    ts = [0.0]
    for k in range(1, len(bp)):
        ts.append(ts[-1] + (bp[k] - bp[k - 1]) * 60.0 / qs[k - 1])
    def b2t(b):
        k = bisect.bisect_right(bp, b) - 1
        return ts[k] + (b - bp[k]) * 60.0 / qs[k]
    midi = pretty_midi.PrettyMIDI(initial_tempo=q0)
    inst = pretty_midi.Instrument(program=0, name=out_name)
    for b, pit, vel, d in notes:
        s, e = b2t(b), b2t(b + d)
        if e > s and 0 <= pit < 128:
            inst.notes.append(pretty_midi.Note(velocity=int(min(127, max(1, vel))), pitch=pit, start=s, end=e))
    midi.instruments.append(inst)
    return midi, bad


class BarStop(StoppingCriteria):
    def __init__(self, bar_id, target):
        self.bar_id, self.target = bar_id, target
    def __call__(self, input_ids, scores, **kw):
        return (input_ids == self.bar_id).sum(1) >= self.target


def load_model(ckpt, dev):
    ck = torch.load(ckpt, map_location="cpu")
    m = GPT2LMHeadModel._from_config(GPT2Config(**ck["config"]), attn_implementation="sdpa")
    m.load_state_dict(ck["model"]); return m.to(dev).eval(), ck


@torch.no_grad()
def sample(model, prompt_ids, n, n_bars, bar_id, pad_id, temperature, top_k, top_p, ctx, max_rounds=20, batch=20):
    """Return n token-id lists, each containing exactly n_bars bars after the prompt's own bars."""
    dev = next(model.parameters()).device
    out, tried = [], 0
    x0 = torch.tensor([prompt_ids], device=dev)
    target = n_bars + 1
    for _ in range(max_rounds):
        b = min(batch, max(n - len(out), 1))
        x = x0.repeat(b, 1)
        g = model.generate(x, do_sample=True, temperature=temperature, top_k=top_k if top_k > 0 else None,
                           top_p=top_p if top_p < 1 else None, max_new_tokens=ctx - x.shape[1],
                           stopping_criteria=StoppingCriteriaList([BarStop(bar_id, target)]),
                           pad_token_id=pad_id, attention_mask=torch.ones_like(x))
        for row in g.tolist():
            tried += 1
            bars = [i for i, t in enumerate(row) if t == bar_id]
            if len(bars) >= target:
                out.append(row[:bars[target - 1]])
                if len(out) == n:
                    return out, tried
    raise RuntimeError(f"only {len(out)}/{n} pieces reached {n_bars} bars in {tried} tries")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["uncond", "cont"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tok2idx", required=True)
    ap.add_argument("--pos_per_beat", type=int, default=4)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--n_bars", type=int, default=32, help="total bars of the final piece")
    ap.add_argument("--prompt_tokens", help="cont: tokenized prompts json ({song_name, tokens})")
    ap.add_argument("--prompt_bars", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="gen")
    a = ap.parse_args()

    dev = "cuda"
    t2i = json.load(open(a.tok2idx)); i2t = {v: k for k, v in t2i.items()}
    model, ck = load_model(a.ckpt, dev)
    ctx = ck["config"]["n_positions"]
    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    jobs = []  # (name, prompt_ids, n)
    if a.mode == "uncond":
        jobs.append((a.tag, [t2i["<BOS>"]], a.n))
    else:
        for s in json.load(open(a.prompt_tokens)):
            toks = [t for t in s["tokens"] if t != "<EOS>"]
            bars = [i for i, t in enumerate(toks) if t == "Bar"]
            toks = toks[:bars[a.prompt_bars]] if len(bars) > a.prompt_bars else toks
            jobs.append((f"{s['song_name']}_{a.tag}", [t2i[t] for t in toks], a.n))
    records = []
    for name, pids, n in jobs:
        n_bars_rel = a.n_bars if a.mode == "uncond" else a.n_bars - sum(1 for t in pids if t == t2i["Bar"])
        seqs, tried = sample(model, pids, n, n_bars_rel, t2i["Bar"], t2i["<PAD>"], a.temperature, a.top_k, a.top_p, ctx)
        print(f"{name}: {len(seqs)} pieces from {tried} samples (reject rate {1 - len(seqs)/tried:.2f})", flush=True)
        for k, ids in enumerate(seqs):
            toks = [i2t[i] for i in ids]
            midi, bad = detokenize(toks, a.pos_per_beat)
            stem = f"{name}_{k:02d}" if n > 1 else name
            midi.write(os.path.join(a.out_dir, stem + ".mid"))
            records.append({"song_name": stem, "tokens": toks, "malformed": bad})
    json.dump(records, open(os.path.join(a.out_dir, f"{a.tag}_tokens.json"), "w"))
    json.dump(vars(a), open(os.path.join(a.out_dir, f"{a.tag}_config.json"), "w"))
    print("malformed total", sum(r["malformed"] for r in records))
