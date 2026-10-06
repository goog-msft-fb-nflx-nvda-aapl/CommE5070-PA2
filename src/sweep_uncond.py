"""Generation sweep for one trained run: checkpoints x sampling configs. For each cell:
generate N 32-bar pieces (generate.py), score with the unmodified eval_basic.py, append one
json row (full config, checkpoint + code hashes, reject stats, H1/H4/GS) to results jsonl.
Resumable: cells already in the results file are skipped."""
import argparse, hashlib, json, os, re, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("--run_dir", required=True)
ap.add_argument("--tok2idx", required=True)
ap.add_argument("--pos_per_beat", type=int, required=True)
ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--out_root", required=True)
ap.add_argument("--results", required=True)
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--sampling_file", default=None, help="json {name: {temperature, top_k, top_p, t_struct, t_pitch, t_vel, t_dur, bar_bias}}")
a = ap.parse_args()

SAMPLING = {  # name: params
    "T1.0_p0.95": {"temperature": 1.0, "top_k": 0, "top_p": 0.95},
    "T1.0_pure": {"temperature": 1.0, "top_k": 0, "top_p": 1.0},
    "T0.9_p0.9": {"temperature": 0.9, "top_k": 0, "top_p": 0.9},
    "T1.2_k5": {"temperature": 1.2, "top_k": 5, "top_p": 1.0},
}
if a.sampling_file:
    SAMPLING = json.load(open(a.sampling_file))
EXTRA = ["t_struct", "t_pitch", "t_vel", "t_dur", "bar_bias"]
here = os.path.dirname(os.path.abspath(__file__))

def quarters(toks):
    per = []
    for x in toks:
        bars = []
        for k in x["tokens"]:
            if k == "Bar": bars.append(0)
            elif k.startswith("Pitch_") and bars: bars[-1] += 1
        if len(bars) >= 32: per.append(bars[:32])
    return [round(sum(sum(q[i * 8:(i + 1) * 8]) for q in per) / 8 / len(per), 2) for i in range(4)] if per else None

sha = lambda f: hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]
code = {f: sha(os.path.join(here, f)) for f in ["generate.py", "sweep_uncond.py"]}
done = set()
if os.path.exists(a.results):
    for l in open(a.results):
        r = json.loads(l); done.add((r["ckpt"], r["sampling"]))
run = os.path.basename(a.run_dir.rstrip("/"))
for ck in a.ckpts:
    for name, sp in SAMPLING.items():
        T, k, pp = sp.get("temperature", 1.0), sp.get("top_k", 0), sp.get("top_p", 1.0)
        if (ck, name) in done:
            continue
        out = os.path.join(a.out_root, run, f"{ck}__{name}")
        cmd = [sys.executable, os.path.join(here, "generate.py"), "uncond", "--ckpt", os.path.join(a.run_dir, ck + ".pt"),
               "--tok2idx", a.tok2idx, "--pos_per_beat", str(a.pos_per_beat), "--out_dir", out, "--n", str(a.n),
               "--temperature", str(T), "--top_k", str(k), "--top_p", str(pp), "--seed", str(a.seed), "--tag", "g"] + sum([[f"--{e}", str(sp[e])] for e in EXTRA if e in sp], [])
        g = subprocess.run(cmd, capture_output=True, text=True)
        m = re.search(r"pieces, no rejection; \{'ok': (\d+), 'slid_window': (\d+), 'restarts': (\d+)\}", g.stdout)
        if g.returncode != 0 or not m:
            print("GEN FAILED", ck, name, g.stderr[-400:], flush=True); continue
        ev = subprocess.run([sys.executable, os.path.join(here, "..", "scripts", "eval_basic.py"), "--tok2idx_path", a.tok2idx,
                             "--sequences_path", os.path.join(out, "g_tokens.json"), "--output_csv", os.path.join(out, "metrics.csv")],
                            capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.path.join(here, "..")})
        h = {k_: float(re.search(rf"Average {k_}: ([\d.]+)", ev.stdout).group(1)) for k_ in ("H1", "H4", "GS")}
        toks = json.load(open(os.path.join(out, "g_tokens.json")))
        row = {"run": run, "ckpt": ck, "ckpt_sha": sha(os.path.join(a.run_dir, ck + ".pt")), "sampling": name, "sampling_params": sp,
               "temperature": T, "top_k": k, "top_p": pp, "n": a.n, "seed": a.seed, "pos_per_beat": a.pos_per_beat,
               "n_slid_window": int(m.group(2)), "n_restarts": int(m.group(3)),
               "mean_tokens": sum(len(t["tokens"]) for t in toks) / len(toks),
               "mean_notes": sum(sum(k.startswith("Pitch_") for k in t["tokens"]) for t in toks) / len(toks),
               "malformed_total": sum(t["malformed"] for t in toks),
               "notes_per_bar_quarters": quarters(toks), **h, "code_sha256": code,
               "train_args": json.load(open(os.path.join(a.run_dir, "run_config.json")))["args"]}
        open(a.results, "a").write(json.dumps(row) + "\n")
        print(f"{run} {ck} {name}: H1 {h['H1']:.3f} H4 {h['H4']:.3f} GS {h['GS']:.3f} slid {row['n_slid_window']}/{a.n} tok {row['mean_tokens']:.0f} bad {row['malformed_total']}", flush=True)
