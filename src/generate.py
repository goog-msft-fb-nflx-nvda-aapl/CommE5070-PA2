"""Generation for the unconditional GPT-2: modes `uncond` (from <BOS>) and `cont`
(continue tokenized prompts). Writes .mid files and a tokens json that
eval_basic.py can read directly. Pieces that do not reach n_bars within the
context are discarded and resampled (rate is printed)."""
import argparse, json, os, bisect
import numpy as np
import torch
import pretty_midi
from transformers import LlamaConfig, LlamaForCausalLM, GPT2Config, GPT2LMHeadModel, StoppingCriteria, StoppingCriteriaList, LogitsProcessor, LogitsProcessorList


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
            if t not in ("<BOS>", "<EOS>", "<PAD>") and not t.startswith("Chord_"):
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
    """stop a row once (bars in its context + bars already dropped from the context) >= target"""
    def __init__(self, bar_id, target, offsets):
        self.bar_id, self.target, self.offsets = bar_id, target, offsets
    def __call__(self, input_ids, scores, **kw):
        return ((input_ids == self.bar_id).sum(1) + self.offsets) >= self.target


class NoSpecial(LogitsProcessor):
    """never sample <PAD>/<BOS>/<EOS>: a piece always continues until it has the requested bars"""
    def __init__(self, ids):
        self.ids = ids
    def __call__(self, input_ids, scores):
        scores[:, self.ids] = -float("inf")
        return scores


class TypedSampler(LogitsProcessor):
    """Does the whole sampling transform itself (use with generate(temperature=1, top_k=None, top_p=None)).
    The type of the next token follows from the last token of the TA grammar
    (Bar/BOS/Dur/Tempo -> struct: Pos or Bar ; Pos -> pitch (or Tempo) ; Pitch -> vel ; Vel -> dur),
    and each type has its own temperature. bar_bias is added to the Bar logit at struct steps."""
    def __init__(self, v, t_struct, t_pitch, t_vel, t_dur, bar_bias, top_k, top_p, dev):
        self.bar_id, self.bar_bias, self.top_k, self.top_p = v["Bar"], bar_bias, top_k, top_p
        V = max(i for i in v.values() if isinstance(i, int)) + 1
        nxt = torch.zeros(V, dtype=torch.long)  # default 0 = struct (BOS, Bar, Dur, Tempo)
        for k, i in v.items():
            if not isinstance(i, int):
                continue
            if k.startswith("Pos_"): nxt[i] = 1
            elif k.startswith("Pitch_"): nxt[i] = 2
            elif k.startswith("Vel_"): nxt[i] = 3
        self.nxt = nxt.to(dev)
        self.temps = torch.tensor([t_struct, t_pitch, t_vel, t_dur], dtype=torch.float32, device=dev)
    def __call__(self, input_ids, scores):
        ty = self.nxt[input_ids[:, -1]]
        scores = scores.float() / self.temps[ty][:, None]
        scores[:, self.bar_id] += torch.where(ty == 0, self.bar_bias, 0.0)
        if self.top_k > 0:
            kth = torch.topk(scores, self.top_k, dim=-1).values[:, -1:]
            scores = scores.masked_fill(scores < kth, -float("inf"))
        if self.top_p < 1.0:
            sl, si = torch.sort(scores, descending=True, dim=-1)
            cp = torch.softmax(sl, dim=-1).cumsum(-1)
            rm = cp - torch.softmax(sl, dim=-1) > self.top_p
            scores = scores.masked_fill(rm.scatter(1, si, rm), -float("inf"))
        return scores


def load_model(ckpt, dev):
    ck = torch.load(ckpt, map_location="cpu")
    if ck["config"].get("model_type") == "llama":
        m = LlamaForCausalLM._from_config(LlamaConfig(**ck["config"]), attn_implementation="sdpa")
    else:
        m = GPT2LMHeadModel._from_config(GPT2Config(**ck["config"]), attn_implementation="sdpa")
    m.load_state_dict(ck["model"]); return m.to(dev).eval(), ck


def trim_context(seq, max_len, v):
    """sliding window: keep <BOS> + a suffix that starts at a Bar token (tempo in effect re-inserted at Pos_0,
    as in the training crops). Returns (context, number of Bar tokens dropped)."""
    if len(seq) <= max_len:
        return seq, 0
    bars = [i for i, t in enumerate(seq) if t == v["Bar"]]
    for k in range(1, len(bars)):
        tail = seq[bars[k]:]
        if len(tail) + 3 <= max_len:
            o = 2 if (len(tail) > 1 and tail[1] in v["chord_ids"]) else 1  # chord token (if any) stays right after Bar
            has_t = len(tail) > o + 1 and tail[o] == v["Pos_0"] and tail[o + 1] in v["tempo_ids"]
            last_t = [t for t in seq[:bars[k]] if t in v["tempo_ids"]]
            ins = [v["Pos_0"], last_t[-1]] if (not has_t and last_t) else []
            return [seq[0]] + tail[:o] + ins + tail[o:], k
    return None, 0  # a single bar longer than the window: degenerate sample


@torch.no_grad()
def sample(model, prompt_ids, n, n_bars, v, temperature, top_k, top_p, ctx, chunk=1024, max_new=48000, typed=None, greedy=False, num_beams=1):
    """n pieces with exactly n_bars bars (prompt bars included). Context slides in `chunk`-token steps so no
    sample is rejected. Returns (token-id lists, stats)."""
    dev = next(model.parameters()).device
    bar_id, pad = v["Bar"], v["<PAD>"]
    seqs = [list(prompt_ids) for _ in range(n)]
    done = [False] * n; slid = [False] * n; new_tok = 0; restarts = 0
    target = n_bars + 1
    lp = LogitsProcessorList([NoSpecial([v["<PAD>"], v["<BOS>"], v["<EOS>"]])])
    gen_kw = dict(do_sample=True, temperature=temperature, top_k=top_k if top_k > 0 else None, top_p=top_p if top_p < 1 else None)
    if typed is not None:  # own sampling transform; HF warpers off
        lp.append(TypedSampler(v, *typed["temps"], typed["bar_bias"], top_k, top_p, dev))
        gen_kw = dict(do_sample=True, temperature=1.0, top_k=None, top_p=None)
    if greedy:  # deterministic decoding: argmax (num_beams=1) or beam search (applied chunk by chunk)
        gen_kw = dict(do_sample=False, num_beams=num_beams)
    while not all(done) and new_tok < max_new:
        idx = [i for i in range(n) if not done[i]]
        ctxs, offs = [], []
        for i in idx:
            c, o = trim_context(seqs[i], ctx - chunk, v)
            if c is None:  # restart this sample from the prompt (counted)
                seqs[i] = list(prompt_ids); restarts += 1
                c, o = seqs[i], 0
            ctxs.append(c); offs.append(o); slid[i] = slid[i] or o > 0
        L = max(len(c) for c in ctxs)
        x = torch.tensor([[pad] * (L - len(c)) + c for c in ctxs], device=dev)
        att = torch.tensor([[0] * (L - len(c)) + [1] * len(c) for c in ctxs], device=dev)
        g = model.generate(x, attention_mask=att, **gen_kw,
                           max_new_tokens=chunk, logits_processor=lp, pad_token_id=pad,
                           stopping_criteria=StoppingCriteriaList([BarStop(bar_id, target, torch.tensor(offs, device=dev))]))
        new_tok += chunk
        for r, i in enumerate(idx):
            seqs[i] += [t for t in g[r, L:].tolist() if t != pad]
            bars = [j for j, t in enumerate(seqs[i]) if t == bar_id]
            if len(bars) >= target:
                seqs[i] = seqs[i][:bars[target - 1]]; done[i] = True
    if not all(done):
        raise RuntimeError(f"{done.count(False)}/{n} pieces did not reach {n_bars} bars in {max_new} new tokens")
    return seqs, {"ok": n, "slid_window": int(sum(slid)), "restarts": restarts}


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
    ap.add_argument("--t_struct", type=float, default=None, help="temperature for Pos/Bar choice (default: --temperature)")
    ap.add_argument("--t_pitch", type=float, default=None)
    ap.add_argument("--t_vel", type=float, default=None)
    ap.add_argument("--t_dur", type=float, default=None)
    ap.add_argument("--greedy", type=int, default=0, help="1: deterministic decoding (argmax, or beam search with --num_beams > 1)")
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--bar_bias", type=float, default=0.0, help="added to the Bar logit at Pos/Bar choice steps")
    a = ap.parse_args()

    dev = "cuda"
    t2i = json.load(open(a.tok2idx)); i2t = {v: k for k, v in t2i.items()}
    t2i["tempo_ids"] = {i for k, i in t2i.items() if isinstance(i, int) and k.startswith("Tempo_")}
    t2i["chord_ids"] = {i for k, i in t2i.items() if isinstance(i, int) and k.startswith("Chord_")}
    model, ck = load_model(a.ckpt, dev)
    ctx = ck["config"].get("n_positions") or ck["config"]["max_position_embeddings"]
    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    tt = [a.t_struct, a.t_pitch, a.t_vel, a.t_dur]
    typed = None if (all(x is None for x in tt) and a.bar_bias == 0.0) else \
        {"temps": [a.temperature if x is None else x for x in tt], "bar_bias": a.bar_bias}
    jobs = []  # (name, prompt_ids, n)
    if a.mode == "uncond":
        jobs.append((a.tag, [t2i["<BOS>"]], a.n))
    else:
        for s in json.load(open(a.prompt_tokens)):
            toks = [t for t in s["tokens"] if t != "<EOS>"]
            if toks[0] != "<BOS>":
                toks = ["<BOS>"] + toks
            bars = [i for i, t in enumerate(toks) if t == "Bar"]
            toks = toks[:bars[a.prompt_bars]] if len(bars) > a.prompt_bars else toks
            jobs.append((f"{s['song_name']}_{a.tag}", [t2i[t] for t in toks], a.n))
    records = []
    for name, pids, n in jobs:
        seqs, stats = sample(model, pids, n, a.n_bars, t2i, a.temperature, a.top_k, a.top_p, ctx, typed=typed, greedy=bool(a.greedy), num_beams=a.num_beams)
        print(f"{name}: {len(seqs)} pieces, no rejection; {stats}", flush=True)
        for k, ids in enumerate(seqs):
            toks = [i2t[i] for i in ids]
            midi, bad = detokenize(toks, a.pos_per_beat)
            stem = f"{name}_{k:02d}" if n > 1 else name
            midi.write(os.path.join(a.out_dir, stem + ".mid"))
            records.append({"song_name": stem, "tokens": toks, "malformed": bad})
    json.dump(records, open(os.path.join(a.out_dir, f"{a.tag}_tokens.json"), "w"))
    json.dump(vars(a), open(os.path.join(a.out_dir, f"{a.tag}_config.json"), "w"))
    print("malformed total", sum(r["malformed"] for r in records))
