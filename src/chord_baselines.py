"""How predictable is the bar chord label from the melody? Baselines on the validation songs' 16-bar windows
(labels = eval_bonus definition; bars with N/unsupported labels are excluded from the denominators)."""
import json, sys, random, collections
import numpy as np
songs = json.load(open(sys.argv[1]))
names = sorted(s["song"] for s in songs)
val = set(random.Random(0).sample(names, round(len(names) * 0.05)))
tr = [s for s in songs if s["song"] not in val]; va = [s for s in songs if s["song"] in val]
cnt = collections.Counter(b["c"] for s in tr for b in s["bars"] if b["c"] >= 0)
maj = cnt.most_common(1)[0][0]
print("train label distribution (top 6):", [(k, round(v / sum(cnt.values()), 3)) for k, v in cnt.most_common(6)])
T = np.zeros((24, 12))
for r in range(12):
    T[r, [r, (r + 4) % 12, (r + 7) % 12]] = 1; T[12 + r, [r, (r + 3) % 12, (r + 7) % 12]] = 1
T /= np.linalg.norm(T, axis=1, keepdims=True)
def pc_hist(b):
    h = np.zeros(12); toks = b["m"]
    for i in range(0, len(toks) - 3, 4):
        if toks[i + 1].startswith("Pitch_"):
            h[int(toks[i + 1][6:]) % 12] += int(toks[i + 3][4:])
    return h
hit = {"majority": 0, "prev_true": 0, "template": 0, "template+prev_true_tiebreak": 0}; n = 0
for s in va:
    bars = s["bars"]
    for j in range(0, max(1, len(bars) - 15), 16):
        prev = None
        for b in bars[j:j + 16]:
            if b["c"] >= 0:
                n += 1
                h = pc_hist(b); est = int(np.argmax(T @ h)) if h.sum() > 0 else -1
                hit["majority"] += b["c"] == maj
                hit["prev_true"] += (prev is not None and prev == b["c"])
                hit["template"] += est == b["c"]
            prev = b["c"]
print(f"n bars {n}: " + ", ".join(f"{k} {v / n:.3f}" for k, v in hit.items() if k != "template+prev_true_tiebreak"))
