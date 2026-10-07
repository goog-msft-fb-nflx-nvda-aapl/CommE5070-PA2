"""Positive control: real PIANO -> bar tokens (bonus_data) -> notes on the prompt beat grid (gen_bonus) -> eval_bonus.
Isolates the cost of quantization + the beat-grid writer from the model."""
import argparse, copy, json, os, sys
import pretty_midi
import bonus_data as bd
import gen_bonus as gb

ap = argparse.ArgumentParser()
ap.add_argument("--prompt_dir", required=True)
ap.add_argument("--prompts", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--pos_per_beat", type=int, default=4)
ap.add_argument("--swap", type=int, default=0, help="1: use the real piano of the NEXT prompt song (plausible piano that ignores this melody)")
a = ap.parse_args()
P = a.pos_per_beat
bd.grid_setup(P)
os.makedirs(a.out_dir, exist_ok=True)
prs = json.load(open(a.prompts))
for k, pr in enumerate(prs):
    d = os.path.join(a.prompt_dir, pr["song"])
    src = prs[(k + 1) % len(prs)] if a.swap else pr
    piano = bd.pick(pretty_midi.PrettyMIDI(os.path.join(a.prompt_dir, src["song"], "piano.mid")), "PIANO")
    bars = bd.bar_segments(piano, src["beat_times"], src["beat_info"], P, src["n_bars"])
    notes, bad = gb.parse_piano(bars, gb.grid_time(pr["beat_times"], pr["beat_info"], P), P)
    mel = pretty_midi.PrettyMIDI(os.path.join(d, "melody.mid"))
    out = pretty_midi.PrettyMIDI()
    mi = copy.deepcopy(mel.instruments[0]); mi.name = "MELODY"
    pi = pretty_midi.Instrument(program=0, name="PIANO"); pi.notes = notes
    out.instruments += [mi, pi]
    out.write(os.path.join(a.out_dir, pr["song"] + ".mid"))
    print(pr["song"], "real piano notes", len(piano.notes), "roundtrip notes", len(notes), "malformed", bad)
