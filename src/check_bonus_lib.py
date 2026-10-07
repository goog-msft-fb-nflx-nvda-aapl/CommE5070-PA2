"""Positive controls for bonus_lib: sequence layout, loss weights, transposition, schedule."""
import json, sys, numpy as np
import bonus_lib as bl
v = bl.Vocab(json.load(open(sys.argv[2])))
songs = json.load(open(sys.argv[1]))
bars = bl.to_bars(songs[0], v)[20:24]
# schedule
assert bl.schedule(4, 0) == [("M",0),("P",0),("M",1),("P",1),("M",2),("P",2),("M",3),("P",3)]
assert bl.schedule(4, 1) == [("M",0),("M",1),("P",0),("M",2),("P",1),("M",3),("P",2),("P",3)], bl.schedule(4, 1)
for L in (0, 1, 2):
    for chord in (0, 1):
        ids, w = bl.build_A(bars, v, L, chord, 0.0)
        ntoks_p = sum(len(b["p"]) for b in bars)
        # weight-1 tokens = piano events + one terminator per piano segment (next tag or EOS)
        assert int(w.sum()) == ntoks_p + len(bars), (L, chord, w.sum(), ntoks_p)
        # every P segment content is exactly the bar's piano tokens, in order
        txt = [v.i2t[i] for i in ids]
        got = []; cur = None
        for t, wt in zip(txt, w):
            pass
        pi = [i for i, t in enumerate(ids) if t == v.P]
        assert len(pi) == len(bars)
        for k, start in enumerate(pi):
            end = start + 1 + len(bars[k]["p"])
            assert list(ids[start + 1:end]) == bars[k]["p"].tolist()
            assert ids[end] in (v.M, v.P, v.EOS) and w[end] == 1.0
            prev_is_p = k > 0 and not any(i == v.M for i in ids[pi[k - 1]:start])
            assert w[start] == (1.0 if prev_is_p else 0.0)  # tag after a P segment is its terminator (loss); after melody it is forced (no loss)
        # melody tokens: carry loss only if melody_w > 0
        assert ids[0] == v.BOS and ids[-1] == v.EOS
        # lookahead: before P_k, melody bars up to min(k+L, n-1) are present
        for k, start in enumerate(pi):
            n_m = sum(1 for i in ids[:start] if i == v.M)
            assert n_m == min(k + L, len(bars) - 1) + 1, (L, k, n_m)
        if chord:
            assert all(ids[i + 1] in v.chord_ids + [v.chord_N] for i, t in enumerate(ids) if t == v.M)
print("arch A layout / weights / lookahead OK (L=0,1,2, chord on/off)")
enc, dec = bl.build_B(bars, v, 1)
assert enc[0] == v.BOS and enc[-1] == v.EOS and dec[-1] == v.EOS
assert (dec == v.BAR).sum() == len(bars) and (enc == v.BAR).sum() == len(bars)
print("arch B layout OK: enc", len(enc), "dec", len(dec))
# transposition: pitch ids shift by k, non-pitch ids unchanged, chord root rotates, major/minor preserved
tb = bl.transpose_bars(bars, 3, v)
for b0, b1 in zip(bars, tb):
    for key in ("m", "p"):
        a0, a1 = b0[key], b1[key]; msk = (a0 >= v.P_LO) & (a0 <= v.P_HI)
        assert (a1[msk] == a0[msk] + 3).all() and (a1[~msk] == a0[~msk]).all()
    c0, c1 = b0["c"], b1["c"]
    assert (c0 < 0 and c1 < 0) or (c1 % 12 == (c0 % 12 + 3) % 12 and (c1 >= 12) == (c0 >= 12))
lo, hi = bl.pitch_range_ok_k(bars, v, 5)
assert lo <= 0 <= hi - 0 and lo >= -5 and hi <= 6
print("transposition OK, allowed k range", lo, hi)
# detokenized sample for a human eye
print("A (L=1) first 40 tokens:", [v.i2t[i] for i in bl.build_A(bars, v, 1, 1, 0.0)[0][:40]])

# ---- chord modes 2,3,4 (arch A) and 4 (arch B) ----
CH = set(v.chord_ids + [v.chord_N])
for mode in (2, 3, 4):
    for L in (0, 1):
        ids, w = bl.build_A(bars, v, L, mode, 0.0)
        pi = [i for i, t in enumerate(ids) if t == v.P]
        for k, st in enumerate(pi):
            assert ids[st + 1] == v.chord_tok(bars[k]["c"]), (mode, L, k)           # chord token right after the <P> tag
            assert w[st + 1] == (1.0 if mode == 4 else 0.0)                          # predicted: loss; forced: no loss
            end = st + 2 + len(bars[k]["p"])
            assert list(ids[st + 2:end]) == bars[k]["p"].tolist()
        n_m_chord = sum(1 for i, t in enumerate(ids) if t == v.M and ids[i + 1] in CH)
        assert n_m_chord == (len(bars) if mode == 3 else 0), (mode, n_m_chord)       # melody-segment chord only in mode 3
        extra = len(bars) if mode == 4 else 0
        assert int(w.sum()) == sum(len(b["p"]) for b in bars) + len(bars) + extra
print("arch A chord modes 2/3/4 layout + weights OK")
for mode in (2, 3, 4):
    e, d = bl.build_B(bars, v, mode)
    assert sum(int(t in CH) for t in d) == len(bars)
    assert sum(int(t in CH) for t in e) == (len(bars) if mode == 3 else 0)
print("arch B chord modes 2/3/4 layout OK")
