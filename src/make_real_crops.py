"""Cut tokenized songs into non-overlapping bar-aligned 32-bar crops, in the
same json format eval_basic.py reads (list of {song_name, tokens})."""
import json, argparse

p = argparse.ArgumentParser()
p.add_argument("--sequences_path", required=True)
p.add_argument("--output", required=True)
p.add_argument("--n_bars", type=int, default=32)
a = p.parse_args()

out = []
for s in json.load(open(a.sequences_path)):
    t = s["tokens"]
    bars = [i for i, k in enumerate(t) if k == "Bar"]
    for j in range(0, len(bars) - a.n_bars + 1, a.n_bars):
        end = bars[j + a.n_bars] if j + a.n_bars < len(bars) else len(t)
        toks = t[bars[j]:end]
        toks = [k for k in toks if k != "<EOS>"]
        out.append({"song_name": f"{s['song_name']}_b{j}", "tokens": toks})
json.dump(out, open(a.output, "w"))
print(len(out), "crops")
