"""Calibration diagnostic: notes/bar of model continuations vs the real continuation of the same held-out crops.
Prompt = first K bars of each 32-bar real crop of a validation song; compare bars K.. of generation vs real."""
import argparse, json, sys
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--real_crops", required=True)
ap.add_argument("--gen_tokens", required=True)
ap.add_argument("--k", type=int, default=8)
a = ap.parse_args()

def per_bar(tokens, n=32):
    bars = []
    for t in tokens:
        if t == "Bar": bars.append(0)
        elif t.startswith("Pitch_") and bars: bars[-1] += 1
    return bars[:n]
real = {s["song_name"]: per_bar(s["tokens"]) for s in json.load(open(a.real_crops))}
gen = {}
for s in json.load(open(a.gen_tokens)):
    gen[s["song_name"].rsplit("_", 1)[0]] = per_bar(s["tokens"])
common = [k for k in gen if k in real and len(gen[k]) == 32 and len(real[k]) == 32]
R = np.array([real[k] for k in common]); G = np.array([gen[k] for k in common])
print(f"n={len(common)} crops; prompt bars 0..{a.k-1} are identical by construction")
for lo, hi in [(0, a.k), (a.k, a.k + 4), (a.k + 4, a.k + 8), (a.k + 8, 32)]:
    print(f"bars {lo:2d}-{hi-1:2d}: real {R[:, lo:hi].mean():5.1f}  gen {G[:, lo:hi].mean():5.1f}  ratio {G[:, lo:hi].mean()/max(R[:, lo:hi].mean(),1e-9):.2f}")
