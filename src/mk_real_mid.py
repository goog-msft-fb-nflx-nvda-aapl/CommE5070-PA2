import json, random, sys
import generate as G
crops = json.load(open("work/crops32_p4.json")); random.Random(0).shuffle(crops)
n = int(sys.argv[1]); out = sys.argv[2]
for s in crops[:n]:
    m, _ = G.detokenize(s["tokens"], 4)
    name = s["song_name"]
    m.write(out + "/" + name + ".mid")
print("wrote", n, "real-crop midis to", out)
