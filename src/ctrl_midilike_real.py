"""Positive control: real 32-bar crops pushed through the MIDI-like normalization (+ encode/decode) and scored by the unmodified eval_basic."""
import json, subprocess, sys, re, os
import alt_repr as A
P = int(sys.argv[1]); ml = A.MidiLike(P)
crops = json.load(open(f"work/crops32_p{P}.json")); out = []
for s in crops:
    toks = [t for t in s["tokens"] if t != "<EOS>"]
    bars = A.parse_song(toks)[:32]
    out.append({"song_name": s["song_name"], "tokens": ml.decode(ml.encode(bars), n_bars=32, cut=True)})
json.dump(out, open("work/ctrl_midilike_real.json", "w"))
ev = subprocess.run([sys.executable, "scripts/eval_basic.py", "--tok2idx_path", f"work/tok2idx_p{P}.json", "--sequences_path", "work/ctrl_midilike_real.json",
                     "--output_csv", "work/ctrl_midilike_real.csv"], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "."})
print(f"real crops through MIDI-like normalization + codec (n={len(out)}, p={P}):", re.findall(r"Average (H1|H4|GS): ([\d.]+)", ev.stdout))
