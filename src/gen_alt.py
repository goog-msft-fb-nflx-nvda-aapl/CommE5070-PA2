"""Generation for the MIDI-like and Compound-Word models (alt_repr.py). Same CLI / output files as generate.py `uncond`
(g_XX.mid, <tag>_tokens.json in TA-style tokens for the unmodified eval_basic.py, <tag>_config.json); pieces that do not reach
n_bars inside the context are discarded and resampled (the rate is printed)."""
import argparse, json, os
import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel, StoppingCriteria, StoppingCriteriaList, LogitsProcessor, LogitsProcessorList
import alt_repr as A
import generate as G


def load(ckpt, dev):
    ck = torch.load(ckpt, map_location="cpu"); a = ck["args"]; P = a["pos_per_beat"]
    if ck["mode"] == "midilike":
        m = GPT2LMHeadModel._from_config(GPT2Config(**ck["cfg"]), attn_implementation="sdpa")
    else:
        class NS: pass
        ns = NS(); ns.n_embd, ns.n_layer, ns.n_head, ns.ctx, ns.dropout = a["n_embd"], a["n_layer"], a["n_head"], a["ctx"], a["dropout"]
        m, _ = A.build_model("cp", P, ns)
    m.load_state_dict(ck["model"])
    return m.to(dev).eval(), ck


def filt(lg, T, top_k, top_p):
    """temperature, top-k, top-p on a (B,V) logit matrix -> (B,V) logits with -inf outside the nucleus"""
    lg = lg.float() / T
    if top_k > 0:
        kth = torch.topk(lg, min(top_k, lg.shape[-1]), -1).values[:, -1:]
        lg = lg.masked_fill(lg < kth, -float("inf"))
    if top_p < 1.0:
        sl, si = torch.sort(lg, descending=True, dim=-1)
        pr = torch.softmax(sl, -1); rm = pr.cumsum(-1) - pr > top_p
        lg = lg.masked_fill(rm.scatter(1, si, rm), -float("inf"))
    return lg


class ShiftStop(StoppingCriteria):
    def __init__(self, shift_val, target):
        self.sv, self.target = shift_val, target
    def __call__(self, input_ids, scores, **kw):
        return self.sv[input_ids].sum(1) >= self.target


class Ban(LogitsProcessor):
    def __init__(self, ids): self.ids = ids
    def __call__(self, input_ids, scores):
        scores[:, self.ids] = -float("inf"); return scores


@torch.no_grad()
def gen_midilike(model, P, n, n_bars, T, top_k, top_p, ctx, dev):
    ml = A.MidiLike(P); target = n_bars * 4 * P
    sv = torch.zeros(len(ml.i2t), dtype=torch.long, device=dev)
    for i, t in enumerate(ml.i2t):
        if t.startswith("Shift_"): sv[i] = int(t[6:])
    x = torch.full((n, 1), ml.t2i["<BOS>"], device=dev)
    g = model.generate(x, attention_mask=torch.ones_like(x), do_sample=True, temperature=T, top_k=top_k if top_k > 0 else None,
                       top_p=top_p if top_p < 1 else None, max_new_tokens=ctx - 1, pad_token_id=0,
                       logits_processor=LogitsProcessorList([Ban([0, 1, 2])]), stopping_criteria=StoppingCriteriaList([ShiftStop(sv, target)]))
    out, rej = [], 0
    for row in g.tolist():
        ids = [t for t in row if t != 0]
        if sum(int(ml.i2t[t][6:]) for t in ids if ml.i2t[t].startswith("Shift_")) >= target:
            out.append(ml.decode(ids, n_bars=n_bars, cut=True))
        else:
            rej += 1
    return out, rej


@torch.no_grad()
def gen_cp(model, P, n, n_bars, T, top_k, top_p, bar_bias, ctx, dev):
    cp = A.CP(P); ign = [cp.ign(f) for f in cp.fields]
    steps = torch.tensor([cp.step(A.T_BOS)] * n, device=dev)[:, None, :]
    past, last = None, steps
    nbar = torch.zeros(n, dtype=torch.long, device=dev); done = torch.zeros(n, dtype=torch.bool, device=dev)
    for _ in range(ctx - 1):
        h, past = model(last, past=past, use_cache=True)
        h = h[:, -1]
        tl = model.heads["f_type"](h).float(); tl[:, [A.T_PAD, A.T_BOS, A.T_EOS]] = -float("inf"); tl[:, A.T_BAR] += bar_bias
        typ = torch.multinomial(torch.softmax(filt(tl, T, top_k, top_p), -1), 1)[:, 0]
        lg = model.field_logits(h, typ)
        new = torch.empty(n, 6, dtype=torch.long, device=dev); new[:, 0] = typ
        for i, f in enumerate(cp.fields[1:], 1):
            s = torch.multinomial(torch.softmax(filt(lg[f][:, :-1], T, top_k, top_p), -1), 1)[:, 0]  # never sample the "ignore" index
            use = torch.isin(typ, torch.tensor(A.CP_MASK[f], device=dev))
            new[:, i] = torch.where(use, s, torch.full_like(s, ign[i]))
        steps = torch.cat([steps, new[:, None, :]], 1); last = new[:, None, :]
        nbar += (typ == A.T_BAR).long(); done |= nbar >= n_bars + 1
        if done.all(): break
    out, rej = [], 0
    for r in range(n):
        st = steps[r].tolist(); bars = [i for i, s in enumerate(st) if s[0] == A.T_BAR]
        if len(bars) >= n_bars + 1:
            out.append(cp.decode(st[:bars[n_bars]]))
        else:
            rej += 1
    return out, rej


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["uncond"])
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tok2idx", default=None); ap.add_argument("--pos_per_beat", type=int, default=None)
    ap.add_argument("--out_dir", required=True); ap.add_argument("--n", type=int, default=20); ap.add_argument("--n_bars", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--top_k", type=int, default=0); ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--bar_bias", type=float, default=0.0); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="gen")
    a = ap.parse_args()
    dev = "cuda"; model, ck = load(a.ckpt, dev); P = ck["args"]["pos_per_beat"]; ctx = ck["args"]["ctx"]; mode = ck["mode"]
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    seqs, rejected = [], 0
    for _ in range(30):
        need = a.n - len(seqs)
        if need <= 0: break
        o, r = (gen_midilike(model, P, need, a.n_bars, a.temperature, a.top_k, a.top_p, ctx, dev) if mode == "midilike"
                else gen_cp(model, P, need, a.n_bars, a.temperature, a.top_k, a.top_p, a.bar_bias, ctx, dev))
        seqs += o; rejected += r
    assert len(seqs) >= a.n, f"only {len(seqs)}/{a.n} pieces reached {a.n_bars} bars"
    recs, tot_bad = [], 0
    for k, toks in enumerate(seqs[:a.n]):
        midi, bad = G.detokenize(toks, P); tot_bad += bad
        stem = f"{a.tag}_{k:02d}"; midi.write(os.path.join(a.out_dir, stem + ".mid"))
        recs.append({"song_name": stem, "tokens": toks, "malformed": bad})
    json.dump(recs, open(os.path.join(a.out_dir, f"{a.tag}_tokens.json"), "w")); json.dump(vars(a), open(os.path.join(a.out_dir, f"{a.tag}_config.json"), "w"))
    print(f"{a.tag}: {a.n} pieces, no rejection; {{'ok': {a.n}, 'slid_window': 0, 'restarts': {rejected}}}")
    print("malformed total", tot_bad)
