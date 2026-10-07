"""Aggregate unconditional-generation sweep rows (results/*/results_*.jsonl written by sweep_uncond.py):
mean +- sd over seeds per (run, ckpt, sampling) next to the real-data reference of the same grid."""
import glob, json, collections, statistics as st, sys
REF = {4: (2.118, 2.617, 0.887, 20.6), 16: (2.118, 2.617, 0.962, 20.6)}  # real 32-bar crops: H1, H4, GS, notes/bar
rows = [json.loads(l) for f in sorted(glob.glob(sys.argv[1])) for l in open(f)]
g = collections.defaultdict(list)
for r in rows:
    g[(r["run"], r["ckpt"], r["sampling"], r["pos_per_beat"])].append(r)
print("| run | ckpt | sampling | seeds | H1 | H4 | GS | notes/bar | malformed |\n|---|---|---|---|---|---|---|---|---|")
for (run, ck, samp, pb), rs in sorted(g.items()):
    m = lambda k: st.mean(r[k] for r in rs)
    sd = lambda k: st.pstdev(r[k] for r in rs) if len(rs) > 1 else 0.0
    print(f"| {run} | {ck} | {samp} | {len(rs)} | {m('H1'):.3f} +- {sd('H1'):.3f} | {m('H4'):.3f} +- {sd('H4'):.3f} | {m('GS'):.3f} +- {sd('GS'):.3f} | {m('mean_notes')/32:.1f} | {sum(r['malformed_total'] for r in rs)} |")
for pb, (h1, h4, gs, npb) in REF.items():
    print(f"| REAL 32-bar crops (p={pb}) | - | - | - | {h1} | {h4} | {gs} | {npb} | - |")
