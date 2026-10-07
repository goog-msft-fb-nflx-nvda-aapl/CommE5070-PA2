"""Unconditional training: GPT-2 (random init) on TA-style REMI tokens.
Samples bar-aligned random crops of the training songs, prepends <BOS>, applies
pitch transposition, and keeps a held-out set of songs for validation NLL."""
import argparse, json, math, os, random, time, bisect
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

p = argparse.ArgumentParser()
p.add_argument("--tok2idx", required=True)
p.add_argument("--sequences", required=True)
p.add_argument("--out_dir", required=True)
p.add_argument("--ctx", type=int, default=4096)
p.add_argument("--n_layer", type=int, default=8)
p.add_argument("--n_embd", type=int, default=512)
p.add_argument("--n_head", type=int, default=8)
p.add_argument("--dropout", type=float, default=0.1)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--steps", type=int, default=40000)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--warmup", type=int, default=1000)
p.add_argument("--wd", type=float, default=0.01)
p.add_argument("--label_smoothing", type=float, default=0.0)
p.add_argument("--transpose", type=int, default=5, help="shift range: -t..t+1 semitones, 0 disables")
p.add_argument("--val_frac", type=float, default=0.05)
p.add_argument("--eval_every", type=int, default=500)
p.add_argument("--save_every", type=int, default=2500)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
os.makedirs(a.out_dir, exist_ok=True)
tok2idx = json.load(open(a.tok2idx))
PAD, BOS, EOS = tok2idx["<PAD>"], tok2idx["<BOS>"], tok2idx["<EOS>"]
V = len(tok2idx)
P_LO, P_HI = tok2idx["Pitch_21"], tok2idx["Pitch_108"]
POS0 = tok2idx["Pos_0"]
TEMPO_IDS = np.array([i for k, i in tok2idx.items() if k.startswith("Tempo_")])
BAR = tok2idx["Bar"]
CHORD_IDS = np.array([i for k, i in tok2idx.items() if k.startswith("Chord_")], dtype=np.int64)

# ---- data ----
songs = json.load(open(a.sequences))
names = sorted(s["song_name"] for s in songs)
val_names = set(random.Random(a.seed).sample(names, round(len(names) * a.val_frac)))

class Song:
    def __init__(self, toks):
        self.ids = np.array([tok2idx[t] for t in toks], dtype=np.int64)
        self.bars = np.flatnonzero(self.ids == BAR)
        is_t = np.isin(self.ids, TEMPO_IDS)
        # tempo id in effect before each bar (-1 if none yet)
        last, self.tempo_before = -1, []
        j = 0
        for bpos in self.bars:
            while j < bpos:
                if is_t[j]:
                    last = self.ids[j]
                j += 1
            self.tempo_before.append(last)

tr = [Song(s["tokens"]) for s in songs if s["song_name"] not in val_names]
va = [Song(s["tokens"]) for s in songs if s["song_name"] in val_names]
print(f"train songs {len(tr)}, val songs {len(va)}", flush=True)
bar_w = np.array([len(s.bars) for s in tr], dtype=np.float64); bar_w /= bar_w.sum()

def make_crop(song, j, max_len, end=None):
    """Crop starting at bar j (j=0 keeps the song's own <BOS>). Mid-song crops get
    a <BOS> and, if the first bar has no tempo at Pos_0, the tempo in effect."""
    ids = song.ids
    if j == 0:
        body = ids[: (end if end is not None else len(ids))]
        return body[:max_len]
    start = song.bars[j]
    body = ids[start: (end if end is not None else len(ids))]
    ins = []
    o = 2 if (len(body) > 1 and body[1] in CHORD_IDS) else 1  # a chord token (if present) stays right after Bar
    has_t = len(body) > o + 1 and body[o] == POS0 and body[o + 1] in TEMPO_IDS
    if not has_t and song.tempo_before[j] >= 0:
        ins = [POS0, song.tempo_before[j]]
    out = np.concatenate([[BOS], body[:o], ins, body[o:]]).astype(np.int64)
    return out[:max_len]

def transpose(c, k_max):
    if k_max == 0:
        return c
    m = (c >= P_LO) & (c <= P_HI)
    if not m.any():
        return c
    lo_off, hi_off = c[m].min() - P_LO, c[m].max() - P_LO
    k = random.randint(max(-k_max, -int(lo_off)), min(k_max + 1, (P_HI - P_LO) - int(hi_off)))
    c = c.copy(); c[m] += k
    return c

def collate(crops):
    n = max(len(c) for c in crops)
    x = np.full((len(crops), n), PAD, dtype=np.int64)
    for i, c in enumerate(crops):
        x[i, : len(c)] = c
    x = torch.from_numpy(x)
    y = x.clone(); y[y == PAD] = -100
    return x, y

def train_batch():
    crops = []
    for _ in range(a.batch):
        s = tr[np.random.choice(len(tr), p=bar_w)]
        j = random.randrange(len(s.bars))
        crops.append(transpose(make_crop(s, j, a.ctx), a.transpose))
    return collate(crops)

def build_val():
    crops = []
    for s in va:
        j = 0
        while j < len(s.bars):
            start = 0 if j == 0 else s.bars[j]
            jn = bisect.bisect_right(s.bars, start + a.ctx - 4) - 1
            if jn <= j:
                jn = j + 1
            end = s.bars[jn] if jn < len(s.bars) else None
            crops.append(make_crop(s, j, a.ctx, end))
            j = jn
    return crops
val_crops = build_val()
print(f"val crops {len(val_crops)}, val tokens {sum(len(c) for c in val_crops)}", flush=True)

# ---- model ----
dev = "cuda"
cfg = GPT2Config(vocab_size=V, n_positions=a.ctx, n_embd=a.n_embd, n_layer=a.n_layer, n_head=a.n_head,
                 resid_pdrop=a.dropout, embd_pdrop=a.dropout, attn_pdrop=a.dropout,
                 bos_token_id=BOS, eos_token_id=EOS, pad_token_id=PAD)
model = GPT2LMHeadModel._from_config(cfg, attn_implementation="sdpa").to(dev)
n_par = sum(q.numel() for q in model.parameters())
print(f"params {n_par/1e6:.1f}M, attn={model.config._attn_implementation}", flush=True)
decay = [q for n, q in model.named_parameters() if q.ndim >= 2]
nodecay = [q for n, q in model.named_parameters() if q.ndim < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": nodecay, "weight_decay": 0.0}],
                        lr=a.lr, betas=(0.9, 0.95), fused=True)
def lr_at(s):
    if s < a.warmup:
        return a.lr * (s + 1) / a.warmup
    return 0.05 * a.lr + 0.95 * a.lr * 0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, a.steps - a.warmup)))

def loss_fn(x, y, ls=0.0):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=x).logits
    logits = logits[:, :-1].float(); tgt = y[:, 1:]
    return F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1), ignore_index=-100,
                           label_smoothing=ls, reduction="sum"), (tgt != -100).sum()

@torch.no_grad()
def evaluate():
    model.eval(); tot, cnt = 0.0, 0
    for i in range(0, len(val_crops), 8):
        x, y = collate(val_crops[i:i + 8])
        l, n = loss_fn(x.to(dev), y.to(dev))
        tot += l.item(); cnt += n.item()
    model.train()
    return tot / cnt

def save(name):
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "args": vars(a)}, os.path.join(a.out_dir, name))

import hashlib, transformers
_sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]
json.dump({"args": vars(a), "params": n_par, "val_songs": sorted(val_names),
           "provenance": {"train_uncond.py": _sha(os.path.abspath(__file__)), "tok2idx": _sha(a.tok2idx), "sequences": _sha(a.sequences),
                          "torch": torch.__version__, "transformers": transformers.__version__}},
          open(os.path.join(a.out_dir, "run_config.json"), "w"), indent=1)
log = open(os.path.join(a.out_dir, "log.jsonl"), "a")
best, t0 = 1e9, time.time()
model.train()
for step in range(1, a.steps + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(step - 1)
    x, y = train_batch()
    l, n = loss_fn(x.to(dev), y.to(dev), a.label_smoothing)
    (l / n).backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    if step % 50 == 0:
        print(f"step {step} train_loss {(l/n).item():.4f} gnorm {gn:.2f} lr {lr_at(step-1):.2e} {time.time()-t0:.0f}s", flush=True)
    if step % a.eval_every == 0 or step == a.steps:
        v = evaluate()
        rec = {"step": step, "train_loss": (l / n).item(), "val_nll": v, "elapsed_s": time.time() - t0}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print("EVAL", rec, flush=True)
        if v < best:
            best = v; save("best.pt")
    if step % a.save_every == 0:
        save(f"step_{step}.pt")
save("last.pt")
print("done, best val_nll", best, flush=True)
