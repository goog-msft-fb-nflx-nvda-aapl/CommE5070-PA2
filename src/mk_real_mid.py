import json, random, sys
import generate as G
P = int(sys.argv[3]) if len(sys.argv) > 3 else 4
crops = json.load(open(f"work/crops32_p{P}.json")); random.Random(0).shuffle(crops)
n = int(sys.argv[1]); out = sys.argv[2]
for s in crops[:n]:
    m, _ = G.detokenize(s["tokens"], P)
    name = s["song_name"]
    m.write(out + "/" + name + ".mid")
print("wrote", n, "real-crop midis to", out)
