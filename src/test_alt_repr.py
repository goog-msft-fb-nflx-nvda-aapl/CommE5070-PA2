"""Positive controls for alt_repr: lossless codecs (CP exact; MIDI-like exact on normalized data) + how much the MIDI-like normalization changes."""
import json, sys
import alt_repr as A

for P, f in ((4, "work/seqs_p4.json"), (16, "work/seqs_p16.json")):
    songs = json.load(open(f)); ml, cp = A.MidiLike(P), A.CP(P)
    ok_ml = ok_cp = 0; dup = nev = moved = nevents = 0; fail = []
    for s in songs:
        toks = [t for t in s["tokens"] if t != "<EOS>"]
        bars = A.parse_song(toks); assert A.remi_tokens(bars) == toks
        ok_cp += cp.decode(cp.encode(bars)) == toks
        norm = ml.normalize(bars); ref = A.remi_tokens(norm)
        if ml.decode(ml.encode(bars), n_bars=len(norm), cut=False) == ref:
            ok_ml += 1
        elif len(fail) < 3:
            fail.append(s["song_name"])
        n_ev = sum(1 for b in bars for e in b if e[0] == "N"); dup += n_ev - sum(1 for b in norm for e in b if e[0] == "N"); nev += n_ev
        moved += sum(1 for b in bars for e in b if e[1] >= 4 * P); nevents += sum(len(b) for b in bars)
    print(f"p={P}: CP lossless {ok_cp}/{len(songs)} | MIDI-like codec lossless on normalized data {ok_ml}/{len(songs)} {fail} | normalization: "
          f"{100 * dup / nev:.2f}% of notes merged (same pitch & time), {100 * moved / nevents:.2f}% of events re-barred (pos >= 4 beats)")
    ok2 = n2 = 0
    for s in songs[:80]:
        bars = A.parse_song([t for t in s["tokens"] if t != "<EOS>"]); j = len(bars) // 3
        ev = A.crop_events(bars, j, 16, shift=2); norm = ml.normalize(ev); n2 += 1
        ok2 += (ml.decode(ml.encode(ev), n_bars=len(norm), cut=False) == A.remi_tokens(norm)) and (cp.decode(cp.encode(ev)) == A.remi_tokens(ev))
    print(f"   shifted mid-song 16-bar crops (both codecs): {ok2}/{n2}")
    bars = A.parse_song([t for t in songs[0]["tokens"] if t != "<EOS>"]); ev = A.crop_events(bars, 0, 32)
    print(f"   first 32 bars of song {songs[0]['song_name']}: REMI {len(A.remi_tokens(ev))} tokens | MIDI-like {len(ml.encode(ev))} | CP {len(cp.encode(ev))} steps; MIDI-like vocab {len(ml.t2i)}")
