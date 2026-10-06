"""Shared pieces for the bonus (melody -> piano) models.

Arch A (decoder-only, bar-interleaved):
  <BOS> then, per piano bar i, any melody bars not yet shown up to bar i+L
  (each as  <M> [Chord_x] melody-events ), then  <P> piano-events ;  <EOS> at the end.
  L = lookahead (0: M_i P_i ; 1: M_0 M_1 P_0 M_2 P_1 ...). Loss is on piano events and the
  tag/EOS that terminates a piano segment (the tag after a piano segment tells the
  inference loop when to inject the next melody bar); melody tokens get weight melody_w.
Arch B (T5 encoder-decoder):
  encoder  <BOS> ( Bar [Chord_x] melody-events )*  <EOS>
  decoder  ( Bar piano-events )* <EOS>   (decoder_input = <BOS> + shifted labels)
"""
import hashlib, json, os, random, subprocess
import numpy as np


class Vocab:
    def __init__(self, tok2idx):
        self.t2i = tok2idx
        self.i2t = {v: k for k, v in tok2idx.items()}
        g = tok2idx.__getitem__
        self.PAD, self.BOS, self.EOS = g("<PAD>"), g("<BOS>"), g("<EOS>")
        self.M, self.P, self.BAR = g("<M>"), g("<P>"), g("Bar")
        self.chord_ids = [g(f"Chord_{i}") for i in range(24)]
        self.chord_N = g("Chord_N")
        self.P_LO, self.P_HI = g("Pitch_21"), g("Pitch_108")
        self.size = len(tok2idx)

    def chord_tok(self, label):
        return self.chord_N if label < 0 else self.chord_ids[label]


def to_bars(song, v):
    """song json -> list of {m: int array, p: int array, c: label}"""
    return [{"m": np.array([v.t2i[t] for t in b["m"]], dtype=np.int64),
             "p": np.array([v.t2i[t] for t in b["p"]], dtype=np.int64), "c": b["c"]} for b in song["bars"]]


def transpose_bars(bars, k, v):
    if k == 0:
        return bars
    out = []
    for b in bars:
        nb = {}
        for key in ("m", "p"):
            a = b[key].copy(); msk = (a >= v.P_LO) & (a <= v.P_HI); a[msk] += k; nb[key] = a
        c = b["c"]
        nb["c"] = c if c < 0 else ((c % 12 + k) % 12) + (12 if c >= 12 else 0)
        out.append(nb)
    return out


def pitch_range_ok_k(bars, v, kmax):
    """allowed transposition interval [lo, hi] for these bars (keeps pitches inside Pitch_21..108)"""
    allp = np.concatenate([b[key] for b in bars for key in ("m", "p")] + [np.zeros(0, dtype=np.int64)])
    msk = (allp >= v.P_LO) & (allp <= v.P_HI)
    if not msk.any():
        return -kmax, kmax + 1
    lo_off, hi_off = int(allp[msk].min() - v.P_LO), int(allp[msk].max() - v.P_LO)
    return max(-kmax, -lo_off), min(kmax + 1, (v.P_HI - v.P_LO) - hi_off)


def schedule(n_bars, L):
    """order of segments: list of ('M', i) / ('P', i)"""
    out, nxt = [], 0
    for i in range(n_bars):
        while nxt <= min(i + L, n_bars - 1):
            out.append(("M", nxt)); nxt += 1
        out.append(("P", i))
    return out


def m_segment(b, v, use_chord):
    return [v.M] + ([v.chord_tok(b["c"])] if use_chord else []) + b["m"].tolist()


def build_A(bars, v, L, use_chord, melody_w):
    ids, w = [v.BOS], [0.0]
    prev_p = False
    for kind, i in schedule(len(bars), L):
        if kind == "M":
            seg = m_segment(bars[i], v, use_chord)
            w += [1.0 if prev_p else melody_w] + [melody_w] * (len(seg) - 1)
            ids += seg; prev_p = False
        else:
            ids.append(v.P); w.append(1.0 if prev_p else 0.0)
            p = bars[i]["p"].tolist(); ids += p; w += [1.0] * len(p); prev_p = True
    ids.append(v.EOS); w.append(1.0)
    return np.array(ids, dtype=np.int64), np.array(w, dtype=np.float32)


def build_B(bars, v, use_chord):
    enc = [v.BOS]; dec = []
    for b in bars:
        enc += [v.BAR] + ([v.chord_tok(b["c"])] if use_chord else []) + b["m"].tolist()
        dec += [v.BAR] + b["p"].tolist()
    enc.append(v.EOS); dec.append(v.EOS)
    return np.array(enc, dtype=np.int64), np.array(dec, dtype=np.int64)


def file_sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def provenance(files):
    import torch, transformers
    here = os.path.dirname(os.path.abspath(__file__))
    d = {"code_sha256": {f: file_sha(os.path.join(here, f)) for f in ["bonus_lib.py", "train_bonus.py", "bonus_data.py"]},
         "torch": torch.__version__, "transformers": transformers.__version__}
    d["data_sha256"] = {os.path.basename(f): file_sha(f) for f in files}
    return d
