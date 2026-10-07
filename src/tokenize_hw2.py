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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--dataset_path", required=True)
ap.add_argument("--tok2idx_output", required=True)
ap.add_argument("--sequences_output", required=True)
ap.add_argument("--pos_per_beat", type=int, default=16)
ap.add_argument("--chord", type=int, default=0, help="1: insert the bar's chord token (Chord_0..23 / Chord_N, eval_bonus definition) after every Bar token")
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
if a.chord:
    import pretty_midi
    import bonus_data as bd
    for k in [f"Chord_{i}" for i in range(24)] + ["Chord_N"]:
        tok2idx[k] = len(tok2idx)
    for s in seqs:
        d = pathlib.Path(a.dataset_path) / s["song_name"]
        midi = pretty_midi.PrettyMIDI(str(d / f"{s['song_name']}.mid"))
        bt, binfo = bd.load_beats(str(d / "beat_midi.txt"))
        first = min(tk.time_to_grid(n.start, bt, binfo, P)[0] for i in midi.instruments for n in i.notes)  # TA's first_note_bar
        labels, _ = bd.bar_chords(str(d / "chord_midi.txt"), str(d / "beat_midi.txt"), binfo[-1][0] + 1)
        out, k = [], 0
        for t in s["tokens"]:
            out.append(t)
            if t == "Bar":
                j = first + k; k += 1
                c = labels[j] if j < len(labels) else -1
                out.append("Chord_N" if c < 0 else f"Chord_{c}")
        s["tokens"] = out
print(f"pos_per_beat={P} vocab={len(tok2idx)} songs={len(seqs)} dropped_oov_events={tot_drop}")
pathlib.Path(a.tok2idx_output).parent.mkdir(parents=True, exist_ok=True)
json.dump(tok2idx, open(a.tok2idx_output, "w"), indent=4)
json.dump(seqs, open(a.sequences_output, "w"))
