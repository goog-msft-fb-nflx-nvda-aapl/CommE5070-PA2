"""Wrapper around the TA tokenization.py.

Fixes two issues of the TA script (kept unmodified, imported from scripts/):
 1. process_pop909_dataset never passes POS_PER_BEAT to time_to_grid, so onset
    positions are always on a 16-per-beat grid whatever --pos_per_beat says
    (only durations / vocab follow the flag). We set the default of time_to_grid.
 2. Bars with more beats than the Pos vocab covers (beat-annotation anomalies)
    emit Pos_* tokens outside tok2idx. We drop the whole event (Pos + its
    Pitch/Vel/Dur or Tempo) so no orphaned attributes remain.
"""
import argparse, json, os, sys, pathlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import tokenization as tk

ap = argparse.ArgumentParser()
ap.add_argument("--dataset_path", required=True)
ap.add_argument("--tok2idx_output", required=True)
ap.add_argument("--sequences_output", required=True)
ap.add_argument("--pos_per_beat", type=int, default=16)
a = ap.parse_args()

P = a.pos_per_beat
tk.time_to_grid.__defaults__ = (P,)
tok2idx = tk.create_vocabulary(a.dataset_path, 8 * P)
seqs = tk.process_pop909_dataset(a.dataset_path, P, 8 * P)

def drop_oov_events(toks):
    out, i, dropped = [], 0, 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("Pos_"):
            n = 2 if toks[i + 1].startswith("Tempo_") else 4
            if t in tok2idx:
                out.extend(toks[i:i + n])
            else:
                dropped += 1
            i += n
        else:
            out.append(t); i += 1
    return out, dropped

tot_drop = 0
for s in seqs:
    s["tokens"], d = drop_oov_events(s["tokens"])
    tot_drop += d
    assert all(t in tok2idx for t in s["tokens"])
print(f"pos_per_beat={P} vocab={len(tok2idx)} songs={len(seqs)} dropped_oov_events={tot_drop}")
pathlib.Path(a.tok2idx_output).parent.mkdir(parents=True, exist_ok=True)
json.dump(tok2idx, open(a.tok2idx_output, "w"), indent=4)
json.dump(seqs, open(a.sequences_output, "w"))
