"""Aggregate bonus result rows (mean +- sd over seeds per cell)."""
import json, glob, collections, statistics as st, sys
rows = [json.loads(l) for f in sorted(glob.glob(sys.argv[1])) for l in open(f)]
def norm(m):  # rows written before the regex fix stored CA under a garbled key
    return {("CA" if k.endswith("Average CA") else k): v for k, v in m.items()}
g = collections.defaultdict(list)
for r in rows:
    r["metrics"] = norm(r["metrics"]); g[(r["ckpt"].split("/")[-2], r["tag"])].append(r)
print("ceiling CA 0.7625 | wrong-song piano CA 0.0125, OA .846/.384/.649/.464 (p4) ; .006/.845/.383/.661/.450 (p16)")
print("%-24s %-22s n | CA            | OA_P  OA_V  OA_D  OA_IOI | notes/bar malformed | chord-pred acc" % ("run", "cell"))
for (run, tag), rs in sorted(g.items()):
    m = lambda k: st.mean(r["metrics"][k] for r in rs)
    pa = [r.get("pred_chord_acc") for r in rs if r.get("pred_chord_acc") is not None]
    print("%-24s %-22s %d | %.3f +- %.3f | %.3f %.3f %.3f %.3f | %.1f %d | " % (run, tag.split("__", 1)[-1] if "__" in tag else tag, len(rs), m("CA"),
          st.pstdev(r["metrics"]["CA"] for r in rs), m("DP (Pitch-class OA)"), m("DV (Velocity OA)"), m("DD (Duration OA)"),
          m("DIOI (Inter-Onset OA)"), st.mean(r["notes_per_bar"] for r in rs), sum(r["malformed"] for r in rs)) + (f"{st.mean(pa):.3f}" if pa else "-"))
