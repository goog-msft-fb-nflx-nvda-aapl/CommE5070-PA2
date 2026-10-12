"""Optional experiment: MidiTok REMI tokenization of POP909 with the time signature corrected from beat_midi.txt (course announcement: the .mid time signatures are mostly wrong).
Per training song: read the .mid (symusic), merge MELODY/BRIDGE/PIANO into one track, shift everything so tick 0 = first annotated beat, rewrite the time signatures from the
per-bar beat counts of beat_midi.txt (TA get_beat_info: pickup bar + beats per bar, denominator 4), tokenize with MidiTok REMI (pitch 21-108, 16 positions per beat, 32 velocities,
tempo tokens, time-signature tokens dropped from the stream because bars/positions already carry the metre). MidiTok token names are renamed to the TA naming
(Bar_None->Bar, Position_k->Pos_k, Velocity_v->Vel_v, Duration_x->Dur_x, PAD/BOS/EOS->TA specials) so that the unmodified eval_basic.py (pitch histograms per bar, onset XOR per bar) and train_uncond.py can be used.
Outputs (--out_dir): tok2idx_mt.json, seqs_mt.json ({song_name, tokens}), real_crops_mt.json (100 random real 32-bar crops, same format, for the real-data reference), tok_config.json.
usage: python miditok_data.py --dataset_path data/POP909/training --out_dir work/mt"""
import argparse, json, os, pathlib, random, sys
import numpy as np
from symusic import Score, Track, Note, Tempo, TimeSignature
from miditok import REMI, TokenizerConfig
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import tokenization as tk

ap = argparse.ArgumentParser()
ap.add_argument("--dataset_path", required=True); ap.add_argument("--out_dir", required=True)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)

cfg = TokenizerConfig(pitch_range=(21, 109), beat_res={(0, 4): 16, (4, 12): 8}, num_velocities=32, use_chords=False, use_programs=False, use_tempos=True,
                      use_time_signatures=True, use_pitchdrum_tokens=False, special_tokens=["PAD", "BOS", "EOS"])
tok = REMI(cfg)


def rename(t):
    if t == "PAD_None": return "<PAD>"
    if t == "BOS_None": return "<BOS>"
    if t == "EOS_None": return "<EOS>"
    if t == "Bar_None": return "Bar"
    k, v = t.split("_", 1)
    return {"Position": "Pos", "Velocity": "Vel", "Duration": "Dur"}.get(k, k) + "_" + v


def sec2tick(sec, tempos, tpq):
    """seconds -> ticks through the piecewise-constant tempo map [(tick, mspq)]"""
    t_sec, tick = 0.0, 0
    for i, (tt, mspq) in enumerate(tempos):
        nxt = tempos[i + 1][0] if i + 1 < len(tempos) else None
        spt = mspq / 1e6 / tpq
        if nxt is not None and t_sec + (nxt - tt) * spt < sec:
            t_sec += (nxt - tt) * spt; continue
        return tt + (sec - t_sec) / spt
    return tick


def convert(song_dir):
    name = song_dir.name
    sc = Score(str(song_dir / f"{name}.mid"))
    tpq = sc.ticks_per_quarter
    bt, bo = [], []
    for line in open(song_dir / "beat_midi.txt"):
        p = line.split()
        if p:
            bt.append(round(float(p[0]), 6)); bo.append(float(p[2]) if len(p) > 2 else 0.0)
    info = tk.get_beat_info(bo)
    tempos = [(int(x.time), int(x.mspq)) for x in sc.tempos] or [(0, 500000)]
    t0 = int(round(sec2tick(bt[0], tempos, tpq)))
    new = Score(tpq)
    tr = Track(program=0, is_drum=False, name="all")
    notes = []
    for t in sc.tracks:
        for n in t.notes:
            s = n.time - t0
            if s < -tpq // 8:
                continue
            notes.append(Note(time=max(0, s), duration=max(1, n.duration), pitch=n.pitch, velocity=n.velocity))
    tr.notes = notes
    new.tracks.append(tr)
    for tt, m in tempos:
        new.tempos.append(Tempo(time=max(0, tt - t0), mspq=m))
    per_bar = {}
    for b, _ in info:
        per_bar[b] = per_bar.get(b, 0) + 1
    tick, last = 0, None
    for b in sorted(per_bar):
        n = per_bar[b]
        if (n, 4) != last:
            new.time_signatures.append(TimeSignature(time=tick, numerator=n, denominator=4)); last = (n, 4)
        tick += n * tpq
    seq = tok(new)
    toks = [rename(t) for t in seq[0].tokens if not t.startswith("TimeSig")]
    return ["<BOS>"] + toks + ["<EOS>"]


songs, bad = [], 0
for d in sorted(x for x in pathlib.Path(a.dataset_path).iterdir() if x.is_dir()):
    try:
        songs.append({"song_name": d.name, "tokens": convert(d)})
    except Exception as e:
        bad += 1; print("skip", d.name, repr(e)[:100], flush=True)
print("songs", len(songs), "skipped", bad, "mean tokens", np.mean([len(s["tokens"]) for s in songs]), flush=True)

names = ["<PAD>", "<BOS>", "<EOS>", "Bar"]
rest = sorted({rename(t) for t in tok.vocab if not t.startswith("TimeSig")} - set(names))
order = {"Pitch": 0, "Pos": 1, "Tempo": 2, "Vel": 3, "Dur": 4}
key = lambda t: (order.get(t.split("_")[0], 9), [float(x) if x.replace(".", "", 1).isdigit() else 0 for x in t.split("_", 1)[1].split(".")])
rest.sort(key=key)
tok2idx = {t: i for i, t in enumerate(names + rest)}
assert all(t in tok2idx for s in songs for t in s["tokens"]), "unknown token"
print("vocab", len(tok2idx), "pitch contiguous:", tok2idx["Pitch_108"] - tok2idx["Pitch_21"] == 87)
json.dump(tok2idx, open(os.path.join(a.out_dir, "tok2idx_mt.json"), "w"))
json.dump(songs, open(os.path.join(a.out_dir, "seqs_mt.json"), "w"))
rng = random.Random(0); crops = []
for _ in range(100):
    s = songs[rng.randrange(len(songs))]["tokens"]
    bars = [i for i, t in enumerate(s) if t == "Bar"]
    if len(bars) < 33:
        continue
    j = rng.randrange(len(bars) - 32)
    crops.append({"song_name": f"real_{len(crops):03d}", "tokens": ["<BOS>"] + s[bars[j]:bars[j + 32]]})
json.dump(crops, open(os.path.join(a.out_dir, "real_crops_mt.json"), "w"))
json.dump({"miditok": "3.0.6", "cfg": {"pitch_range": [21, 109], "beat_res": "{(0,4):16,(4,12):8}", "num_velocities": 32, "tempos": True, "time_signature_tokens_in_stream": False}},
          open(os.path.join(a.out_dir, "tok_config.json"), "w"))
