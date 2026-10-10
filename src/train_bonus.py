"""Train the melody -> piano model. --arch A: GPT-2 on bar-interleaved sequences; --arch B: T5 encoder-decoder."""
import argparse, json, math, os, random, time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel, T5Config, T5ForConditionalGeneration
import bonus_lib as bl

p = argparse.ArgumentParser()
p.add_argument("--arch", choices=["A", "B", "C"], required=True, help="C = stage-1 chord predictor (melody -> one chord token per bar)")
p.add_argument("--data", required=True, help="bonus_train.json")
p.add_argument("--tok2idx", required=True)
p.add_argument("--out_dir", required=True)
p.add_argument("--lookahead", type=int, default=1)
p.add_argument("--chord", type=int, default=0, help="1: per-bar chord token in the melody conditioning")
p.add_argument("--melody_w", type=float, default=0.0, help="A: loss weight on melody/condition tokens")
p.add_argument("--ctx", type=int, default=2048, help="A: context; B: decoder max length")
p.add_argument("--enc_ctx", type=int, default=1024)
p.add_argument("--nb_min", type=int, default=16)
p.add_argument("--nb_max", type=int, default=24)
p.add_argument("--n_layer", type=int, default=8)
p.add_argument("--n_embd", type=int, default=512)
p.add_argument("--n_head", type=int, default=8)
p.add_argument("--d_ff", type=int, default=2048)
p.add_argument("--dropout", type=float, default=0.1)
p.add_argument("--init_from", default=None, help="A: our own unconditional checkpoint (vocab is extended)")
p.add_argument("--batch", type=int, default=16)
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--warmup", type=int, default=1000)
p.add_argument("--wd", type=float, default=0.01)
p.add_argument("--label_smoothing", type=float, default=0.0)
p.add_argument("--transpose", type=int, default=5)
p.add_argument("--val_frac", type=float, default=0.05)
p.add_argument("--eval_every", type=int, default=500)
p.add_argument("--save_every", type=int, default=2500)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
os.makedirs(a.out_dir, exist_ok=True)
v = bl.Vocab(json.load(open(a.tok2idx)))
songs = json.load(open(a.data))
names = sorted(s["song"] for s in songs)
val_names = set(random.Random(a.seed).sample(names, round(len(names) * a.val_frac)))
tr = [bl.to_bars(s, v) for s in songs if s["song"] not in val_names]
va = [bl.to_bars(s, v) for s in songs if s["song"] in val_names]
print(f"train songs {len(tr)}, val songs {len(va)}, vocab {v.size}", flush=True)
w_song = np.array([len(s) for s in tr], dtype=np.float64); w_song /= w_song.sum()


def make_example(bars):
    """bars (already transposed/cropped) -> tensors dict; drops trailing bars until it fits."""
    n = len(bars)
    while n > 1:
        if a.arch == "A":
            ids, w = bl.build_A(bars[:n], v, a.lookahead, a.chord, a.melody_w)
            if len(ids) <= a.ctx:
                return ids, w
        else:
            enc, dec = bl.build_C(bars[:n], v) if a.arch == "C" else bl.build_B(bars[:n], v, a.chord)
            if len(enc) <= a.enc_ctx and len(dec) <= a.ctx:
                return enc, dec
        n -= 1
    raise RuntimeError("single bar does not fit")


def train_example():
    s = tr[np.random.choice(len(tr), p=w_song)]
    nb = random.randint(a.nb_min, a.nb_max)
    j = random.randrange(max(1, len(s) - nb + 1))
    bars = s[j:j + nb]
    if a.transpose:
        lo, hi = bl.pitch_range_ok_k(bars, v, a.transpose)
        bars = bl.transpose_bars(bars, random.randint(lo, hi), v)
    return make_example(bars)


def val_examples():
    ex = []
    for s in va:
        for j in range(0, max(1, len(s) - a.nb_min + 1), a.nb_min):
            ex.append(make_example(s[j:j + a.nb_min]))
    return ex


def pad(rows, val):
    n = max(len(r) for r in rows)
    out = np.full((len(rows), n), val, dtype=rows[0].dtype)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return torch.from_numpy(out)


dev = "cuda"
if a.arch == "A":
    if a.init_from:
        ck = torch.load(a.init_from, map_location="cpu")
        cfgd = dict(ck["config"]); cfgd["vocab_size"] = v.size
        cfg = GPT2Config(**cfgd)
        assert cfg.n_positions >= a.ctx
    else:
        cfg = GPT2Config(vocab_size=v.size, n_positions=a.ctx, n_embd=a.n_embd, n_layer=a.n_layer, n_head=a.n_head,
                         resid_pdrop=a.dropout, embd_pdrop=a.dropout, attn_pdrop=a.dropout,
                         bos_token_id=v.BOS, eos_token_id=v.EOS, pad_token_id=v.PAD)
    model = GPT2LMHeadModel._from_config(cfg, attn_implementation="sdpa")
    if a.init_from:
        sd = model.state_dict(); n_copy = 0
        for k, t in ck["model"].items():
            if k in sd and sd[k].shape == t.shape:
                sd[k] = t; n_copy += 1
            elif k in sd and sd[k].shape[1:] == t.shape[1:] and sd[k].shape[0] >= t.shape[0]:
                sd[k][:t.shape[0]] = t; n_copy += 1
        model.load_state_dict(sd); print(f"init_from: copied {n_copy}/{len(sd)} tensors", flush=True)
else:
    cfg = T5Config(vocab_size=v.size, d_model=a.n_embd, d_kv=a.n_embd // a.n_head, d_ff=a.d_ff, num_layers=a.n_layer,
                   num_decoder_layers=a.n_layer, num_heads=a.n_head, dropout_rate=a.dropout,
                   relative_attention_num_buckets=32, relative_attention_max_distance=256,
                   decoder_start_token_id=v.BOS, pad_token_id=v.PAD, eos_token_id=v.EOS)
    model = T5ForConditionalGeneration(cfg)
model = model.to(dev)
n_par = sum(q.numel() for q in model.parameters())
print(f"arch {a.arch} params {n_par/1e6:.1f}M", flush=True)
decay = [q for q in model.parameters() if q.ndim >= 2]; nodecay = [q for q in model.parameters() if q.ndim < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": nodecay, "weight_decay": 0.0}],
                        lr=a.lr, betas=(0.9, 0.95), fused=True)
def lr_at(s):
    if s < a.warmup:
        return a.lr * (s + 1) / a.warmup
    return 0.05 * a.lr + 0.95 * a.lr * 0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, a.steps - a.warmup)))


def loss_of(batch, ls=0.0):
    """returns (summed weighted CE, summed weight)"""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if a.arch == "A":
            x, w = batch
            x, w = x.to(dev), w.to(dev)
            logits = model(input_ids=x).logits[:, :-1].float(); tgt = x[:, 1:]; wt = w[:, 1:]
        else:
            enc, dec = batch
            enc, dec = enc.to(dev), dec.to(dev)
            dec_in = torch.cat([torch.full_like(dec[:, :1], v.BOS), dec[:, :-1]], 1)
            logits = model(input_ids=enc, attention_mask=(enc != v.PAD).long(), decoder_input_ids=dec_in).logits.float()
            tgt = dec; wt = (dec != v.PAD).float()
    ce = F.cross_entropy(logits.reshape(-1, v.size), tgt.reshape(-1), reduction="none", label_smoothing=ls).view_as(wt)
    if a.arch == "C" and not torch.is_grad_enabled():  # teacher-forced top-1 accuracy on annotated (non-N) chord positions
        ann = torch.isin(tgt, torch.tensor(v.chord_ids, device=tgt.device))
        loss_of.acc_hit = getattr(loss_of, "acc_hit", 0) + ((logits.argmax(-1) == tgt) & ann).sum().item()
        loss_of.acc_n = getattr(loss_of, "acc_n", 0) + ann.sum().item()
    return (ce * wt).sum(), wt.sum()


def collate(ex):
    if a.arch == "A":
        return pad([e[0] for e in ex], v.PAD), pad([e[1] for e in ex], 0.0)
    return pad([e[0] for e in ex], v.PAD), pad([e[1] for e in ex], v.PAD)


val_ex = val_examples()
print(f"val examples {len(val_ex)}", flush=True)

@torch.no_grad()
def evaluate():
    model.eval(); tot = cnt = 0.0
    for i in range(0, len(val_ex), 16):
        l, n = loss_of(collate(val_ex[i:i + 16])); tot += l.item(); cnt += n.item()
    model.train(); return tot / cnt if cnt else float('nan')  # nan when val_frac 0

def save(name):
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "args": vars(a), "arch": a.arch}, os.path.join(a.out_dir, name))

prov = bl.provenance([a.data, a.tok2idx])
json.dump({"args": vars(a), "params": n_par, "val_songs": sorted(val_names), "provenance": prov}, open(os.path.join(a.out_dir, "run_config.json"), "w"), indent=1)
log = open(os.path.join(a.out_dir, "log.jsonl"), "a")
best, t0 = 1e9, time.time()
model.train()
for step in range(1, a.steps + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(step - 1)
    l, n = loss_of(collate([train_example() for _ in range(a.batch)]), a.label_smoothing)
    (l / n).backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    if step % 50 == 0:
        print(f"step {step} train_loss {(l/n).item():.4f} gnorm {gn:.2f} {time.time()-t0:.0f}s", flush=True)
    if step % a.eval_every == 0 or step == a.steps:
        val = evaluate()
        rec = {"step": step, "train_loss": (l / n).item(), "val_nll": val, "elapsed_s": time.time() - t0}
        if a.arch == "C":
            rec["val_chord_top1_teacher_forced"] = getattr(loss_of, "acc_hit", 0) / max(getattr(loss_of, "acc_n", 0), 1); loss_of.acc_hit = loss_of.acc_n = 0
        log.write(json.dumps(rec) + "\n"); log.flush(); print("EVAL", rec, flush=True)
        if val < best:
            best = val; save("best.pt")
    if step % a.save_every == 0:
        save(f"step_{step}.pt")
save("last.pt")
print("done, best val_nll", best, flush=True)
