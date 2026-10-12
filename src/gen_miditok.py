"""Optional MidiTok experiment: sample 32-bar pieces from a MidiTok-REMI-token model (same sliding-window sampler as generate.py), write the TA-named token json that the unmodified
eval_basic.py scores, and decode to .mid with MidiTok (best effort; 4/4 assumed because time-signature tokens are not in the stream).
usage: gen_miditok.py --ckpt C --tok2idx tok2idx_mt.json --out_dir D [--n 20] [--temperature 1.0 --top_k 0 --top_p 1.0] [--seed 0] [--tag g]"""
import argparse, json, os
import numpy as np, torch
import generate as G
from miditok import REMI, TokenizerConfig

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--tok2idx", required=True); ap.add_argument("--out_dir", required=True)
ap.add_argument("--n", type=int, default=20); ap.add_argument("--n_bars", type=int, default=32)
ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--top_k", type=int, default=0); ap.add_argument("--top_p", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="g"); ap.add_argument("--no_midi", type=int, default=0)
a = ap.parse_args()
dev = "cuda"
t2i = json.load(open(a.tok2idx)); i2t = {v: k for k, v in t2i.items()}
t2i["tempo_ids"] = {i for k, i in t2i.items() if isinstance(i, int) and k.startswith("Tempo_")}
t2i["chord_ids"] = set()
model, ck = G.load_model(a.ckpt, dev)
ctx = ck["config"].get("n_positions") or ck["config"]["max_position_embeddings"]
os.makedirs(a.out_dir, exist_ok=True)
torch.manual_seed(a.seed); np.random.seed(a.seed)
seqs, stats = G.sample(model, [t2i["<BOS>"]], a.n, a.n_bars, t2i, a.temperature, a.top_k, a.top_p, ctx)
print("pieces", len(seqs), stats, flush=True)
cfg = TokenizerConfig(pitch_range=(21, 109), beat_res={(0, 4): 16, (4, 12): 8}, num_velocities=32, use_chords=False, use_programs=False, use_tempos=True,
                      use_time_signatures=True, use_pitchdrum_tokens=False, special_tokens=["PAD", "BOS", "EOS"])
tok = REMI(cfg)
back = lambda t: {"<PAD>": "PAD_None", "<BOS>": "BOS_None", "<EOS>": "EOS_None", "Bar": "Bar_None"}.get(t) or \
    ({"Pos": "Position", "Vel": "Velocity", "Dur": "Duration"}.get(t.split("_", 1)[0], t.split("_", 1)[0]) + "_" + t.split("_", 1)[1])
recs, nmid = [], 0
for k, ids in enumerate(seqs):
    toks = [i2t[i] for i in ids]
    stem = f"{a.tag}_{k:02d}"
    recs.append({"song_name": stem, "tokens": toks})
    if not a.no_midi:
        try:
            mt = [back(t) for t in toks if t not in ("<BOS>", "<EOS>", "<PAD>")]
            # insert a 4/4 time-signature token after every Bar (MidiTok decoding expects it when time-signature tokens are enabled)
            full = []
            for t in mt:
                full.append(t)
                if t == "Bar_None":
                    full.append("TimeSig_4/4")
            score = tok.decode([tok.vocab[t] for t in full])
            score.dump_midi(os.path.join(a.out_dir, stem + ".mid")); nmid += 1
        except Exception as e:
            print("midi decode failed", stem, repr(e)[:120])
json.dump(recs, open(os.path.join(a.out_dir, f"{a.tag}_tokens.json"), "w"))
json.dump(vars(a), open(os.path.join(a.out_dir, f"{a.tag}_config.json"), "w"))
print("midi written", nmid, "mean tokens", np.mean([len(r["tokens"]) for r in recs]))
