"""Alternative token representations trained with the same GPT-2 backbone, for the representation ablation (lecture 05/06):
  midilike : MIDI-like events in musical time: Tempo_x, Vel_v + NoteOn_p, NoteOff_p, Shift_k (k units of 1/pos_per_beat beat); no bar tokens
  cp       : Compound Word: one step per note / bar / tempo event with fields (type, pos, tempo, pitch, vel, dur), summed field
             embeddings as input, one softmax head per field (conditioned on the step type), per-field sampling
Both are derived losslessly from the TA-style REMI tokens (seqs_p*.json) and decoded back to the TA-style tokens, so the
unmodified eval_basic.py scores them exactly like the REMI models. Bars are assumed to be 4 beats (4P units) for MIDI-like."""
import argparse, json, math, os, random, re, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Model

# ------------------------------------------------------------------ shared: parse TA tokens <-> per-bar events
def parse_song(tokens):
    """-> (bars, n_bars): bars[b] = sorted list of events ('T', pos, tempo_value) | ('N', pos, pitch, vel, dur) in TA order"""
    bars, b, i = [], -1, 0
    while i < len(tokens):
        t = tokens[i]
        if t == "Bar":
            bars.append([]); b += 1; i += 1
        elif t.startswith("Pos_") and b >= 0:
            pos = int(t[4:])
            if tokens[i + 1].startswith("Tempo_"):
                bars[b].append(("T", pos, int(tokens[i + 1][6:]))); i += 2
            else:
                bars[b].append(("N", pos, int(tokens[i + 1][6:]), int(tokens[i + 2][4:]), int(tokens[i + 3][4:]))); i += 4
        else:
            i += 1
    return bars


def remi_tokens(bars_events, bos=True, eos=False):
    out = ["<BOS>"] if bos else []
    for ev in bars_events:
        out.append("Bar")
        for e in sorted(ev, key=lambda e: (e[1], 0 if e[0] == "T" else 1, e[2] if e[0] == "N" else 0)):
            out.append(f"Pos_{e[1]}")
            out += [f"Tempo_{e[2]}"] if e[0] == "T" else [f"Pitch_{e[2]}", f"Vel_{e[3]}", f"Dur_{e[4]}"]
    if eos:
        out.append("<EOS>")
    return out


def last_tempo_before(bars, j):
    for b in range(j - 1, -1, -1):
        for e in reversed(bars[b]):
            if e[0] == "T":
                return e[2]
    return None


def crop_events(bars, j, nb, shift=0):
    """bar-aligned crop [j, j+nb) as per-bar event lists; mid-song crops get the tempo in effect at Pos_0 (as in the REMI runs);
    pitches shifted by `shift` semitones"""
    out = []
    for b in range(j, min(j + nb, len(bars))):
        ev = [e if e[0] == "T" else ("N", e[1], e[2] + shift, e[3], e[4]) for e in bars[b]]
        if b == j and j > 0 and not any(e[0] == "T" and e[1] == 0 for e in ev):
            lt = last_tempo_before(bars, j)
            if lt is not None:
                ev.append(("T", 0, lt))
        out.append(ev)
    return out


# ------------------------------------------------------------------ MIDI-like
class MidiLike:
    def __init__(self, P):
        self.P, self.BAR = P, 4 * P
        names = ["<PAD>", "<BOS>", "<EOS>"] + [f"Shift_{k}" for k in range(1, self.BAR + 1)] + [f"On_{p}" for p in range(21, 109)] + \
                [f"Off_{p}" for p in range(21, 109)] + [f"Vel_{v}" for v in range(1, 128, 4)] + [f"Tempo_{t}" for t in range(30, 211, 5)]
        self.t2i = {n: i for i, n in enumerate(names)}; self.i2t = names

    def normalize(self, ev_bars):
        """What a bar-less note-on/off stream can express: (1) bars are 4 beats (events at pos >= 4P move into the following bar),
        (2) simultaneous notes of the same pitch are merged (max velocity, max duration), (3) overlapping notes of the same pitch are truncated at the next onset."""
        flat = {}
        tempos = []
        for b, evs in enumerate(ev_bars):
            for e in evs:
                t = b * self.BAR + e[1]
                if e[0] == "T":
                    tempos.append((t, e[2]))
                else:
                    k = (t, e[2])
                    v, d = flat.get(k, (0, 0)); flat[k] = (max(v, e[3]), max(d, e[4]))
        # (3) a pitch can sound only once at a time: a note ends where the next note of the same pitch starts
        by_pitch = {}
        for (t, p), (v, d) in flat.items():
            by_pitch.setdefault(p, []).append((t, v, d))
        flat = {}
        for p, lst in by_pitch.items():
            lst.sort()
            for i, (t, v, d) in enumerate(lst):
                if i + 1 < len(lst):
                    d = min(d, lst[i + 1][0] - t)
                flat[(t, p)] = (v, d)
        nb = max([t // self.BAR for t in [k[0] for k in flat] + [t for t, _ in tempos]] + [len(ev_bars) - 1]) + 1
        out = [[] for _ in range(nb)]
        for (t, p), (v, d) in flat.items():
            out[t // self.BAR].append(("N", t % self.BAR, p, v, d))
        for t, v in tempos:
            out[t // self.BAR].append(("T", t % self.BAR, v))
        return out

    def encode(self, ev_bars, bos=True, eos=False):
        """per-bar events -> MIDI-like ids (bar b starts at unit b*4P; a note's off time = on + dur); input is normalized first"""
        ev_bars = self.normalize(ev_bars)
        ev = []  # (time, order, token)
        for b, evs in enumerate(ev_bars):
            for e in evs:
                t = b * self.BAR + e[1]
                if e[0] == "T":
                    ev.append((t, 0, 0, 0, f"Tempo_{e[2]}"))
                else:  # at one time step: tempo, then note-offs, then (Vel, NoteOn) pairs by pitch -- each Vel directly before its On
                    ev.append((t + e[4], 1, e[2], 0, f"Off_{e[2]}"))
                    ev.append((t, 2, e[2], 0, f"Vel_{e[3]}")); ev.append((t, 2, e[2], 1, f"On_{e[2]}"))
        ev.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        ids, cur = ([self.t2i["<BOS>"]] if bos else []), 0
        for t, _, _, _, tok in ev:
            while t > cur:
                k = min(t - cur, self.BAR); ids.append(self.t2i[f"Shift_{k}"]); cur += k
            ids.append(self.t2i[tok])
        if eos:
            ids.append(self.t2i["<EOS>"])
        return ids

    def decode(self, ids, n_bars=None, cut=True):
        """ids -> TA-style REMI tokens (first n_bars bars if given); open notes are closed at the end / when re-triggered.
        cut=True: stop reading at n_bars*4P units (generation); cut=False: read everything, only onsets after n_bars are dropped (round-trip tests)"""
        cur, vel, opened, notes, tempos = 0, 1, {}, [], []
        end_t = None if n_bars is None else n_bars * self.BAR
        for i in ids:
            tok = self.i2t[i]
            if tok.startswith("Shift_"):
                cur += int(tok[6:])
                if cut and end_t is not None and cur >= end_t:
                    cur = end_t; break
            elif tok.startswith("Vel_"):
                vel = int(tok[4:])
            elif tok.startswith("On_"):
                p = int(tok[3:])
                if p in opened:
                    s, v = opened.pop(p); notes.append((s, p, v, max(cur - s, 1)))
                opened[p] = (cur, vel)
            elif tok.startswith("Off_"):
                p = int(tok[4:])
                if p in opened:
                    s, v = opened.pop(p); notes.append((s, p, v, max(cur - s, 1)))
            elif tok.startswith("Tempo_"):
                tempos.append((cur, int(tok[6:])))
        for p, (s, v) in opened.items():
            notes.append((s, p, v, max((end_t if end_t is not None else cur) - s, 1)))
        nb = (n_bars if n_bars is not None else (max([n[0] for n in notes] + [0]) // self.BAR + 1))
        bars = [[] for _ in range(nb)]
        for s, p, v, d in notes:
            if s // self.BAR < nb:
                bars[s // self.BAR].append(("N", s % self.BAR, p, v, d))
        for t, v in tempos:
            if t // self.BAR < nb:
                bars[t // self.BAR].append(("T", t % self.BAR, v))
        return remi_tokens(bars)


# ------------------------------------------------------------------ Compound Word
TYPES = ["<PAD>", "<BOS>", "<EOS>", "Bar", "Note", "Tempo"]
T_PAD, T_BOS, T_EOS, T_BAR, T_NOTE, T_TEMPO = range(6)


class CP:
    def __init__(self, P):
        self.P = P; self.MAXPOS = 8 * P; self.MAXDUR = 2 * P * 8
        # field sizes incl. an "ignore" index at the end
        self.sizes = {"type": 6, "pos": self.MAXPOS + 1, "tempo": 37 + 1, "pitch": 88 + 1, "vel": 32 + 1, "dur": self.MAXDUR + 1}
        self.fields = list(self.sizes)

    def ign(self, f):
        return self.sizes[f] - 1

    def step(self, typ, pos=None, tempo=None, pitch=None, vel=None, dur=None):
        return [typ, self.ign("pos") if pos is None else pos, self.ign("tempo") if tempo is None else (tempo - 30) // 5,
                self.ign("pitch") if pitch is None else pitch - 21, self.ign("vel") if vel is None else (vel - 1) // 4,
                self.ign("dur") if dur is None else dur - 1]

    def encode(self, ev_bars, bos=True, eos=False):
        steps = [self.step(T_BOS)] if bos else []
        for evs in ev_bars:
            steps.append(self.step(T_BAR))
            for e in sorted(evs, key=lambda e: (e[1], 0 if e[0] == "T" else 1, e[2] if e[0] == "N" else 0)):
                steps.append(self.step(T_TEMPO, pos=e[1], tempo=e[2]) if e[0] == "T" else self.step(T_NOTE, pos=e[1], pitch=e[2], vel=e[3], dur=e[4]))
        if eos:
            steps.append(self.step(T_EOS))
        return steps

    def decode(self, steps):
        out = []
        for s in steps:
            if s[0] == T_BOS:
                out.append("<BOS>")
            elif s[0] == T_BAR:
                out.append("Bar")
            elif s[0] == T_TEMPO:
                out += [f"Pos_{s[1]}", f"Tempo_{30 + 5 * s[2]}"]
            elif s[0] == T_NOTE:
                out += [f"Pos_{s[1]}", f"Pitch_{s[3] + 21}", f"Vel_{1 + 4 * s[4]}", f"Dur_{s[5] + 1}"]
        return out


class CPModel(nn.Module):
    def __init__(self, cp, d=512, L=8, H=8, ctx=4096, drop=0.3):
        super().__init__()
        self.cp = cp
        self.emb = nn.ModuleDict({"f_" + f: nn.Embedding(n, d) for f, n in cp.sizes.items()})
        self.body = GPT2Model(GPT2Config(vocab_size=1, n_positions=ctx, n_embd=d, n_layer=L, n_head=H, resid_pdrop=drop, embd_pdrop=drop, attn_pdrop=drop))
        self.temb = nn.Embedding(6, 64)
        self.heads = nn.ModuleDict({"f_" + f: nn.Linear(d + (64 if f != "type" else 0), n) for f, n in cp.sizes.items()})

    def embed(self, steps):  # steps: (B,T,6) long
        return sum(self.emb["f_" + f](steps[..., i]) for i, f in enumerate(self.cp.fields))

    def forward(self, steps, past=None, use_cache=False):
        o = self.body(inputs_embeds=self.embed(steps), past_key_values=past, use_cache=use_cache)
        return o.last_hidden_state, o.past_key_values

    def field_logits(self, h, typ):  # typ: true / sampled type of the step to predict
        te = self.temb(typ)
        return {f: (self.heads["f_" + f](h) if f == "type" else self.heads["f_" + f](torch.cat([h, te], -1))) for f in self.cp.fields}


CP_MASK = {"pos": (T_NOTE, T_TEMPO), "tempo": (T_TEMPO,), "pitch": (T_NOTE,), "vel": (T_NOTE,), "dur": (T_NOTE,)}


def cp_loss(model, steps):
    """teacher-forced loss: step t predicts the fields of step t+1; returns total NLL (nats, summed over all fields/steps) and per-field means"""
    h, _ = model(steps[:, :-1])
    tgt = steps[:, 1:]
    lg = model.field_logits(h, tgt[..., 0])
    total, per = 0.0, {}
    valid = tgt[..., 0] != T_PAD
    for i, f in enumerate(model.cp.fields):
        m = valid if f == "type" else (valid & torch.isin(tgt[..., 0], torch.tensor(CP_MASK[f], device=steps.device)))
        ce = F.cross_entropy(lg[f].reshape(-1, lg[f].shape[-1]), tgt[..., i].reshape(-1), reduction="none").view_as(m)
        s = (ce * m).sum(); total = total + s; per[f] = (s.item(), m.sum().item())
    return total, per


# ------------------------------------------------------------------ data
def load_songs(seq_path, val_frac, seed):
    songs = json.load(open(seq_path))
    names = sorted(s["song_name"] for s in songs)
    val = set(random.Random(seed).sample(names, round(len(names) * val_frac)))
    tr = [parse_song(s["tokens"]) for s in songs if s["song_name"] not in val]
    va = [parse_song(s["tokens"]) for s in songs if s["song_name"] in val]
    return tr, va, sorted(val)


def pitch_shift_range(bars, kmax):
    ps = [e[2] for b in bars for e in b if e[0] == "N"]
    return max(-kmax, 21 - min(ps)), min(kmax + 1, 108 - max(ps))


# ------------------------------------------------------------------ training
def make_batch_fn(mode, tr, P, ctx, batch, tmax, seed):
    ml, cp = MidiLike(P), CP(P)
    rng = random.Random(seed)
    w = np.array([len(s) for s in tr], dtype=np.float64); w /= w.sum()

    def one():
        s = tr[np.random.choice(len(tr), p=w)]
        j = rng.randrange(len(s)); nb = rng.randint(24, 96)
        lo, hi = pitch_shift_range(s[j:j + nb], tmax)
        ev = crop_events(s, j, nb, shift=rng.randint(lo, hi) if tmax else 0)
        if mode == "midilike":
            return np.array(ml.encode(ev)[:ctx], dtype=np.int64)
        return np.array(cp.encode(ev)[:ctx], dtype=np.int64)  # (T, 6)

    def get_batch():
        rows = [one() for _ in range(batch)]
        n = max(len(r) for r in rows)
        if mode == "midilike":
            x = np.zeros((batch, n), dtype=np.int64)
            for i, r in enumerate(rows): x[i, :len(r)] = r
        else:
            x = np.zeros((batch, n, 6), dtype=np.int64)
            for i, r in enumerate(rows): x[i, :len(r)] = r  # PAD rows: type 0 (PAD); other fields 0 are masked by type
        return torch.from_numpy(x)
    return get_batch


def val_windows(va, mode, P, ctx):
    ml, cp = MidiLike(P), CP(P)
    out = []
    for s in va:
        for j in range(0, max(1, len(s) - 31), 32):
            ev = crop_events(s, j, 32)
            n_remi = len(remi_tokens(ev, bos=False))
            out.append((np.array((ml if mode == "midilike" else cp).encode(ev)[:ctx], dtype=np.int64), n_remi))
    return out


def build_model(mode, P, a):
    if mode == "midilike":
        ml = MidiLike(P)
        cfg = GPT2Config(vocab_size=len(ml.i2t), n_positions=a.ctx, n_embd=a.n_embd, n_layer=a.n_layer, n_head=a.n_head,
                         resid_pdrop=a.dropout, embd_pdrop=a.dropout, attn_pdrop=a.dropout, bos_token_id=1, eos_token_id=2, pad_token_id=0)
        return GPT2LMHeadModel._from_config(cfg, attn_implementation="sdpa"), cfg.to_dict()
    cp = CP(P)
    return CPModel(cp, a.n_embd, a.n_layer, a.n_head, a.ctx, a.dropout), {"n_embd": a.n_embd, "n_layer": a.n_layer, "n_head": a.n_head, "ctx": a.ctx, "dropout": a.dropout}


def batch_loss(mode, model, x):
    """returns (summed NLL in nats, number of predicted units)"""
    if mode == "midilike":
        lg = model(input_ids=x).logits[:, :-1].float(); tgt = x[:, 1:]
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction="none", ignore_index=0).view_as(tgt)
        return ce.sum(), (tgt != 0).sum()
    tot, _ = cp_loss(model, x)
    return tot, (x[:, 1:, 0] != T_PAD).sum()


def train(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    P = a.pos_per_beat; os.makedirs(a.out_dir, exist_ok=True)
    tr, va, val_names = load_songs(a.sequences, a.val_frac, a.seed)
    print(f"mode {a.mode}: train songs {len(tr)}, val songs {len(va)}", flush=True)
    model, cfgd = build_model(a.mode, P, a); model = model.cuda()
    n_par = sum(q.numel() for q in model.parameters()); print(f"params {n_par/1e6:.1f}M", flush=True)
    decay = [q for q in model.parameters() if q.ndim >= 2]; nod = [q for q in model.parameters() if q.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": nod, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.95), fused=True)
    lr_at = lambda s: a.lr * (s + 1) / a.warmup if s < a.warmup else 0.05 * a.lr + 0.95 * a.lr * 0.5 * (1 + math.cos(math.pi * (s - a.warmup) / max(1, a.steps - a.warmup)))
    batch = make_batch_fn(a.mode, tr, P, a.ctx, a.batch, a.transpose, a.seed)
    vw = val_windows(va, a.mode, P, a.ctx) if va else []

    @torch.no_grad()
    def evaluate():
        model.eval(); tot = n = 0.0
        for arr, n_remi in vw:
            x = torch.from_numpy(arr)[None].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l, _ = batch_loss(a.mode, model, x)
            tot += l.item(); n += n_remi
        model.train(); return tot / n if n else float("nan")  # nats per REMI-equivalent token (comparable with the REMI GPT-2 runs)

    import hashlib
    json.dump({"args": vars(a), "params": n_par, "val_songs": val_names,
               "provenance": {"alt_repr.py": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16], "torch": torch.__version__}},
              open(os.path.join(a.out_dir, "run_config.json"), "w"), indent=1)
    log = open(os.path.join(a.out_dir, "log.jsonl"), "a"); best, t0 = 1e9, time.time()
    def save(name): torch.save({"model": model.state_dict(), "mode": a.mode, "cfg": cfgd, "args": vars(a)}, os.path.join(a.out_dir, name))
    model.train()
    for step in range(1, a.steps + 1):
        for g in opt.param_groups: g["lr"] = lr_at(step - 1)
        x = batch().cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l, n = batch_loss(a.mode, model, x)
        (l / n).backward(); gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % 50 == 0: print(f"step {step} train_loss/unit {(l/n).item():.4f} gnorm {gn:.2f} {time.time()-t0:.0f}s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            v = evaluate(); rec = {"step": step, "train_loss_per_unit": (l / n).item(), "val_nll_per_remi_token": v, "elapsed_s": time.time() - t0}
            log.write(json.dumps(rec) + "\n"); log.flush(); print("EVAL", rec, flush=True)
            if v < best: best = v; save("best.pt")
        if step % a.save_every == 0: save(f"step_{step}.pt")
    save("last.pt"); print("done, best val nll per REMI token", best, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["midilike", "cp"], required=True)
    ap.add_argument("--sequences", required=True); ap.add_argument("--pos_per_beat", type=int, required=True); ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ctx", type=int, default=None); ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--n_layer", type=int, default=8); ap.add_argument("--n_embd", type=int, default=512); ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.3); ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000); ap.add_argument("--wd", type=float, default=0.01); ap.add_argument("--transpose", type=int, default=5)
    ap.add_argument("--val_frac", type=float, default=0.05); ap.add_argument("--eval_every", type=int, default=500); ap.add_argument("--save_every", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.ctx is None: a.ctx = 8192 if a.mode == "midilike" else 4096
    if a.batch is None: a.batch = 4 if a.mode == "midilike" else 8
    train(a)
