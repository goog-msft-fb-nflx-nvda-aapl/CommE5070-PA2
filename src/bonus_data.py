"""Bonus-task data: per-bar MELODY / PIANO token segments + per-bar chord label.

Uses the TA grid logic (scripts/tokenization.py: get_beat_info, time_to_grid with the
grid fix from tokenize_hw2) and the TA chord utilities (scripts/eval_bonus.py), so the
bar / chord definitions are identical to what eval_bonus.py scores.

A song is {"song", "n_bars", "bars": [{"m": [tokens], "p": [tokens], "c": chord_label}]}
where m/p are Pos/Pitch/Vel/Dur token lists of that bar (no Bar token), c in -1 (N / unsupported), 0..23.
"""
import argparse, json, os, sys, pathlib
import numpy as np
import pretty_midi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import tokenization as tk
import eval_bonus as eb


def grid_setup(P):
    tk.time_to_grid.__defaults__ = (P,)


def bar_segments(inst, beat_times, beat_info, P, n_bars):
    """tokens per bar for one instrument; events outside the Pos vocab (8*P) or bar range are dropped."""
    max_pos = 8 * P
    ev = [[] for _ in range(n_bars)]
    for note in inst.notes:
        bar, pos, b_len = tk.time_to_grid(note.start, beat_times, beat_info, P)
        if bar >= n_bars or pos >= max_pos:
            continue
        dur = max(1, min(max_pos * 2, int(((note.end - note.start) / b_len) * P)))
        vel = max(1, note.velocity // 4 * 4 + 1)
        ev[bar].append((pos, note.pitch, vel, dur))
    out = []
    for b in ev:
        toks = []
        for pos, pit, vel, dur in sorted(b, key=lambda e: (e[0], e[1])):
            toks += [f"Pos_{pos}", f"Pitch_{pit}", f"Vel_{vel}", f"Dur_{dur}"]
        out.append(toks)
    return out


def bar_chords(chord_path, beat_path, n_bars):
    """dominant chord label per bar (eval_bonus definition); -1 if N / unsupported / uncovered."""
    times, down = eb.load_gt_beats(beat_path)
    bars = eb.GTBars(times, down)
    chords = eb.load_gt_chords(chord_path)
    lab = []
    for j in range(n_bars):
        if j < bars.n_bars:
            c = eb.dominant_chord(chords, bars.bar_start(j), bars.bar_end(j))
            lab.append(int(c) if c >= 0 else -1)
        else:
            lab.append(-1)
    return lab, bars.n_bars


def load_beats(beat_path):
    bt, bo = [], []
    for line in open(beat_path):
        p = line.split()
        if not p:
            continue
        bt.append(round(float(p[0]), 6)); bo.append(float(p[2]) if len(p) > 2 else 0.0)
    return bt, tk.get_beat_info(bo)


def pick(midi, name):
    for i in midi.instruments:
        if i.name.strip().upper() == name:
            return i
    return None


def build_training(dataset_path, P):
    songs = []
    for d in sorted(pathlib.Path(dataset_path).iterdir()):
        if not d.is_dir():
            continue
        try:
            midi = pretty_midi.PrettyMIDI(str(d / f"{d.name}.mid"))
            mel, pno = pick(midi, "MELODY"), pick(midi, "PIANO")
            if mel is None or pno is None:
                print("skip (missing track)", d.name); continue
            bt, binfo = load_beats(str(d / "beat_midi.txt"))
            n_bars = binfo[-1][0] + 1
            chord, _ = bar_chords(str(d / "chord_midi.txt"), str(d / "beat_midi.txt"), n_bars)
            m = bar_segments(mel, bt, binfo, P, n_bars); p = bar_segments(pno, bt, binfo, P, n_bars)
            songs.append({"song": d.name, "n_bars": n_bars,
                          "bars": [{"m": m[i], "p": p[i], "c": chord[i]} for i in range(n_bars)]})
        except Exception as e:
            print("error", d.name, e)
    return songs


def build_prompts(prompt_dir, P):
    """melody-only bars for prompt folders + the beat info needed to write MIDI back on the prompt's own time grid."""
    out = []
    for d in sorted(pathlib.Path(prompt_dir).iterdir()):
        midi = pretty_midi.PrettyMIDI(str(d / "melody.mid"))
        mel = pick(midi, "MELODY") or midi.instruments[0]
        bt, binfo = load_beats(str(d / "beat_midi.txt"))
        times, down = eb.load_gt_beats(str(d / "beat_midi.txt"))
        n_eval = eb.GTBars(times, down).n_bars
        chord, _ = bar_chords(str(d / "chord_midi.txt"), str(d / "beat_midi.txt"), n_eval)
        m = bar_segments(mel, bt, binfo, P, n_eval)
        out.append({"song": d.name, "n_bars": n_eval, "bars": [{"m": m[i], "p": [], "c": chord[i]} for i in range(n_eval)],
                    "beat_times": bt, "beat_info": binfo})
    return out


def add_bonus_vocab(tok2idx):
    t = dict(tok2idx)
    for k in ["<M>", "<P>"] + [f"Chord_{i}" for i in range(24)] + ["Chord_N"]:
        if k not in t:
            t[k] = len(t)
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", default=None, help="POP909 training dir (not needed with --prompts_only)")
    ap.add_argument("--prompts_only", type=int, default=0)
    ap.add_argument("--prompt_dir", required=True)
    ap.add_argument("--base_tok2idx", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pos_per_beat", type=int, default=4)
    a = ap.parse_args()
    grid_setup(a.pos_per_beat)
    os.makedirs(a.out_dir, exist_ok=True)
    tok2idx = add_bonus_vocab(json.load(open(a.base_tok2idx)))
    json.dump(tok2idx, open(os.path.join(a.out_dir, "tok2idx_bonus.json"), "w"), indent=4)
    pr = build_prompts(a.prompt_dir, a.pos_per_beat)
    if a.prompts_only:
        json.dump(pr, open(os.path.join(a.out_dir, "bonus_prompts.json"), "w"))
        print(f"prompts {len(pr)} (bars {[s['n_bars'] for s in pr]})")
        sys.exit(0)
    tr = build_training(a.dataset_path, a.pos_per_beat)
    json.dump(tr, open(os.path.join(a.out_dir, "bonus_train.json"), "w"))
    json.dump(pr, open(os.path.join(a.out_dir, "bonus_prompts.json"), "w"))
    allt = [t for s in tr for b in s["bars"] for t in b["m"] + b["p"]]
    miss = [t for t in set(allt) if t not in tok2idx]
    print(f"train songs {len(tr)}, prompts {len(pr)} (bars {[s['n_bars'] for s in pr]}), vocab {len(tok2idx)}, tokens not in vocab: {miss[:5]}")
    nb = np.array([len(b["m"]) / 4 for s in tr for b in s["bars"]]); npn = np.array([len(b["p"]) / 4 for s in tr for b in s["bars"]])
    print(f"notes/bar mean melody {nb.mean():.2f}, piano {npn.mean():.2f}; tokens/bar melody {4*nb.mean():.1f}, piano {4*npn.mean():.1f}")
    cl = np.array([b["c"] for s in tr for b in s["bars"]])
    print("chord label dist: N/unsupported %.3f, major %.3f, minor %.3f" % ((cl < 0).mean(), ((cl >= 0) & (cl < 12)).mean(), (cl >= 12).mean()))
