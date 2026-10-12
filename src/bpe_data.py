"""Optional experiment: BPE-style merging of the TA REMI p16 token sequences (lossless, reversible).
Base tokens are mapped to single characters; HF `tokenizers` learns merges on transposed copies of the TRAINING songs only (POP909 train split, no extra data).
Bar, Tempo_*, <BOS>, <EOS> are isolated by a pre-tokenizer so they never merge (the training crops / sliding window use them).
Transposition (-5..+6) is applied OFFLINE (12 copies per song), because pitch ids inside merged tokens cannot be shifted at training time.
Outputs (--out_dir): tok2idx_bpe.json (base vocab unchanged + merged tokens BPE_k), seqs_bpe.json ({song_name, tokens}), bpe_table.json (merged token -> base token list).
usage: python bpe_data.py --tok2idx work/tok2idx_p16.json --sequences work/seqs_p16.json --out_dir work/bpe --merges 1500"""
import argparse, json, os, re
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, Regex

ap = argparse.ArgumentParser()
ap.add_argument("--tok2idx", required=True); ap.add_argument("--sequences", required=True); ap.add_argument("--out_dir", required=True)
ap.add_argument("--merges", type=int, default=1500); ap.add_argument("--transpose", type=int, default=5)
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
t2i = json.load(open(a.tok2idx)); V = len(t2i)
i2t = {i: t for t, i in t2i.items()}
ch = lambda i: chr(0x100 + i)
songs = json.load(open(a.sequences))
LO, HI = 21, 108


def transposed(toks, k):
    out = []
    for t in toks:
        if t.startswith("Pitch_"):
            t = "Pitch_%d" % (int(t[6:]) + k)
        out.append(t)
    return out


copies = []  # (song_name, k, tokens)
for s in songs:
    ps = [int(t[6:]) for t in s["tokens"] if t.startswith("Pitch_")]
    lo, hi = (min(ps), max(ps)) if ps else (LO, HI)
    for k in range(-a.transpose, a.transpose + 2):
        if lo + k >= LO and hi + k <= HI:
            copies.append((s["song_name"], k, transposed(s["tokens"], k) if k else s["tokens"]))
print("songs", len(songs), "copies", len(copies), flush=True)
texts = ["".join(ch(t2i[t]) for t in toks) for _, _, toks in copies]

iso = [t2i["Bar"], t2i["<BOS>"], t2i["<EOS>"]] + [i for t, i in t2i.items() if t.startswith("Tempo_")]
pat = "[" + "".join(re.escape(ch(i)) for i in iso) + "]"
tok = Tokenizer(models.BPE(unk_token=None))
tok.pre_tokenizer = pre_tokenizers.Split(Regex(pat), behavior="isolated")
trainer = trainers.BpeTrainer(vocab_size=V + a.merges, initial_alphabet=[ch(i) for i in range(V)], special_tokens=[], show_progress=False)
tok.train_from_iterator(texts, trainer)
vocab = tok.get_vocab()
print("learned vocab", len(vocab), flush=True)

# merged token string -> new name; base single chars keep their names
name_of = {}
table = {}
for s_, _ in sorted(vocab.items(), key=lambda kv: kv[1]):
    if len(s_) == 1:
        name_of[s_] = i2t[ord(s_) - 0x100]
    else:
        nm = "BPE_%d" % len(table)
        name_of[s_] = nm; table[nm] = [i2t[ord(c) - 0x100] for c in s_]
new_t2i = dict(t2i)
for nm in table:
    new_t2i[nm] = len(new_t2i)
seqs, n_base, n_bpe = [], 0, 0
for (name, k, toks), text in zip(copies, texts):
    enc = tok.encode(text).tokens
    assert "".join(enc) == text
    seqs.append({"song_name": "%s_t%+d" % (name, k), "tokens": [name_of[e] for e in enc]})
    n_base += len(toks); n_bpe += len(enc)
print("compression: %d -> %d tokens (%.3fx)" % (n_base, n_bpe, n_base / n_bpe), flush=True)
json.dump(new_t2i, open(os.path.join(a.out_dir, "tok2idx_bpe.json"), "w"))
json.dump(table, open(os.path.join(a.out_dir, "bpe_table.json"), "w"))
json.dump(seqs, open(os.path.join(a.out_dir, "seqs_bpe.json"), "w"))
json.dump({"merges": a.merges, "vocab": len(new_t2i), "copies": len(copies), "tokens_base": n_base, "tokens_bpe": n_bpe}, open(os.path.join(a.out_dir, "stats.json"), "w"))
