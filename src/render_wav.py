"""Render every .mid in a folder to .wav (pretty_midi + FluidSynth, bundled TimGM6mb soundfont)."""
import argparse, glob, os
import numpy as np, pretty_midi, soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--midi_dir", required=True)
ap.add_argument("--out_dir", default=None, help="default: same as midi_dir")
ap.add_argument("--fs", type=int, default=22050)
a = ap.parse_args()
out = a.out_dir or a.midi_dir
os.makedirs(out, exist_ok=True)
for f in sorted(glob.glob(os.path.join(a.midi_dir, "*.mid"))):
    audio = pretty_midi.PrettyMIDI(f).fluidsynth(fs=a.fs)
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.9
    sf.write(os.path.join(out, os.path.basename(f)[:-4] + ".wav"), audio, a.fs)
    print("wrote", os.path.basename(f))
