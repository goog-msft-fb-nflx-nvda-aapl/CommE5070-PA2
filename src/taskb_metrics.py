"""Objective proxies for Task b continuations (no metric is defined by the spec): compares bars 9-32 with the 8-bar prompt, from the token files.
Per file: notes/bar ratio (continuation/prompt), pitch-class histogram cosine (key/harmony consistency), onset-position histogram cosine (rhythm consistency),
out-of-key rate (notes whose pitch class is not among the prompt's 7 most used), repeated-bar rate (continuation bars identical in (Pos,Pitch) content to an earlier bar).
usage: python taskb_metrics.py <dir with taskB_cfg*/cfg*_tokens.json>"""
import glob, json, os, sys
import numpy as np


def split_bars(toks):
    bars, cur = [], None
    for t in toks:
        if t == "Bar":
            cur = []; bars.append(cur)
        elif cur is not None:
            cur.append(t)
    return bars


def notes(bar):
    out, pos = [], None
    for t in bar:
        if t.startswith("Pos_"):
            pos = int(t[4:])
        elif t.startswith("Pitch_") and pos is not None:
            out.append((pos, int(t[6:])))
    return out


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def hist(ns, f, n):
    h = np.zeros(n)
    for p, q in ns:
        h[f(p, q)] += 1
    return h


rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "taskB_cfg*", "cfg*_tokens.json"))):
    cfg = os.path.basename(f).split("_")[0]
    for it in json.load(open(f)):
        bars = [notes(b) for b in split_bars(it["tokens"])]
        P, C = bars[:8], bars[8:32]
        pn, cn = [n for b in P for n in b], [n for b in C for n in b]
        pc_p, pc_c = hist(pn, lambda p, q: q % 12, 12), hist(cn, lambda p, q: q % 12, 12)
        ps_p, ps_c = hist(pn, lambda p, q: p, 512), hist(cn, lambda p, q: p, 512)
        key = set(np.argsort(-pc_p)[:7].tolist())
        seen, rep = [], 0
        for b in bars[:32]:
            s = frozenset(b)
            if len(b) > 0 and s in seen:
                rep += 1
            seen.append(s)
        crep = sum(1 for i, b in enumerate(C) if len(b) > 0 and frozenset(b) in [frozenset(x) for x in bars[8:8 + i]])
        rows.append((it["song_name"], cfg, len(pn) / 8, len(cn) / max(len(C), 1), cos(pc_p, pc_c), cos(ps_p, ps_c),
                     sum(1 for p, q in cn if q % 12 not in key) / max(len(cn), 1), crep / max(len(C), 1)))
print("%-10s %-5s %9s %9s %6s %8s %8s %8s" % ("file", "cfg", "prompt/bar", "cont/bar", "pcCos", "rhythCos", "outKey", "repBar"))
for r in sorted(rows):
    print("%-10s %-5s %9.1f %9.1f %6.3f %8.3f %8.3f %8.3f" % r)
