import json
import alt_repr as A
P = 4; songs = json.load(open("work/seqs_p4.json")); s = songs[0]
ml = A.MidiLike(P)
bars = A.parse_song([t for t in s["tokens"] if t != "<EOS>"]); norm = ml.normalize(bars); ref = A.remi_tokens(norm)
got = ml.decode(ml.encode(bars), n_bars=len(norm), cut=False)
print("len ref/got", len(ref), len(got), "bars", len(norm))
i = next((k for k, (a, b) in enumerate(zip(ref, got)) if a != b), None)
print("first mismatch idx", i); print("ref", ref[max(0, i-6):i+10]); print("got", got[max(0, i-6):i+10])
