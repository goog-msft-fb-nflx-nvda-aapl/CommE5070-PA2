"""Structureness Indicator (SI) for a directory of WAVs, using the MusDr / FMP-notebook Python scape-plot code
(MusDr/run_python_scapeplot.py parameters) and the MusDr SI definition: max of the fitness scape plot inside a
timescale band (default bands, seconds: short 3-8, mid 8-15, long 15-128; scape plot sample rate 2 Hz)."""
import argparse, glob, json, os, sys, time
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "MusDr"))
from mueller_audio_tools.scapeplot import compute_fitness_scape_plot, normalization_properties_SSM
from mueller_audio_tools.ssm_features import compute_SM_from_filename, compute_tempo_rel_set

TEMPO_REL = compute_tempo_rel_set(0.5, 2, 7)
SHIFT = np.array(list(range(12)))
BOUNDS = [3, 8, 15, 128]


def si_from_scape(sp, lo, hi, sr=2):
    lo_ts, hi_ts = int(lo * sr) - 1, int(hi * sr)
    return 0.0 if lo_ts >= sp.shape[0] else float(np.max(sp[lo_ts:hi_ts]))


def proc(path):
    t0 = time.time()
    _, _, _, _, S, _ = compute_SM_from_filename(path, tempo_rel_set=TEMPO_REL, shift_set=SHIFT, thresh=0.25, penalty=-2)
    S = normalization_properties_SSM(S)
    sp = compute_fitness_scape_plot(S)[0]
    return os.path.basename(path), [si_from_scape(sp, BOUNDS[i], BOUNDS[i + 1]) for i in range(3)], sp.shape, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.wav_dir, "*.wav")))
    if a.limit:
        files = files[:a.limit]
    with Pool(a.workers) as p:
        res = p.map(proc, files)
    si = np.array([r[1] for r in res])
    out = {"wav_dir": a.wav_dir, "n": len(res), "SI_short_mid_long_mean": si.mean(0).tolist(), "SI_sd": si.std(0).tolist(),
           "per_piece": {r[0]: r[1] for r in res}, "scape_shape_example": list(res[0][2]), "seconds_per_piece_mean": float(np.mean([r[3] for r in res]))}
    json.dump(out, open(a.out_json, "w"), indent=1)
    print(f"{a.wav_dir}: n={len(res)} SI short/mid/long = {np.round(si.mean(0), 4).tolist()} (sd {np.round(si.std(0), 4).tolist()}); {out['seconds_per_piece_mean']:.1f} s/piece")
