"""Optional BPE experiment: sample 32-bar pieces from a BPE-token model with the same sliding-window sampler as generate.py, expand merged tokens back to the TA
base tokens, write .mid and the base-token json that the unmodified eval_basic.py scores.
usage: gen_bpe.py --ckpt C --tok2idx tok2idx_bpe.json --table bpe_table.json --out_dir D [--n 20] [--temperature 1.0 --top_k 0 --top_p 1.0] [--seed 0] [--tag g]"""
import argparse, json, os
import numpy as np, torch
import generate as G

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--tok2idx", required=True); ap.add_argument("--table", required=True)
ap.add_argument("--out_dir", required=True); ap.add_argument("--n", type=int, default=20); ap.add_argument("--n_bars", type=int, default=32)
ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--top_k", type=int, default=0); ap.add_argument("--top_p", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="g"); ap.add_argument("--pos_per_beat", type=int, default=16)
a = ap.parse_args()
dev = "cuda"
t2i = json.load(open(a.tok2idx)); i2t = {v: k for k, v in t2i.items()}
table = json.load(open(a.table))
t2i["tempo_ids"] = {i for k, i in t2i.items() if isinstance(i, int) and k.startswith("Tempo_")}
t2i["chord_ids"] = set()
model, ck = G.load_model(a.ckpt, dev)
ctx = ck["config"].get("n_positions") or ck["config"]["max_position_embeddings"]
os.makedirs(a.out_dir, exist_ok=True)
torch.manual_seed(a.seed); np.random.seed(a.seed)
seqs, stats = G.sample(model, [t2i["<BOS>"]], a.n, a.n_bars, t2i, a.temperature, a.top_k, a.top_p, ctx)
print("pieces", len(seqs), stats, flush=True)
recs = []
for k, ids in enumerate(seqs):
    toks = []
    for i in ids:
        t = i2t[i]
        toks += table[t] if t in table else [t]
    midi, bad = G.detokenize(toks, a.pos_per_beat)
    stem = f"{a.tag}_{k:02d}"
    midi.write(os.path.join(a.out_dir, stem + ".mid"))
    recs.append({"song_name": stem, "tokens": toks, "malformed": bad, "bpe_len": len(ids)})
json.dump(recs, open(os.path.join(a.out_dir, f"{a.tag}_tokens.json"), "w"))
json.dump(vars(a), open(os.path.join(a.out_dir, f"{a.tag}_config.json"), "w"))
print("malformed total", sum(r["malformed"] for r in recs), "mean bpe len", np.mean([r["bpe_len"] for r in recs]), "mean base len", np.mean([len(r["tokens"]) for r in recs]))
