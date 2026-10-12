"""Report figures from the saved logs / result rows (results/). Palette: validated categorical slots 1-4 (light)."""
import glob, json, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100"}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6})


def rows(pattern):
    return [json.loads(l) for f in sorted(glob.glob(os.path.join(R, pattern))) for l in open(f)]


def fig_val_curves():
    runs = [("u1_p4_8L512", "dropout 0.1, 40k steps", C["blue"]), ("v4_wd0.1", "dropout 0.1, wd 0.1", C["yellow"]),
            ("v1_steps12k", "dropout 0.1, 12k steps", C["aqua"]), ("v2_drop0.3", "dropout 0.3, 20k steps (chosen)", C["orange"])]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for name, label, col in runs:
        L = [json.loads(l) for l in open(os.path.join(R, "runs", name, "log.jsonl"))]
        x = [r["step"] for r in L]; y = [r["val_nll"] for r in L]
        ax.plot(x, y, color=col, lw=2, label=label)
        i = min(range(len(y)), key=lambda k: y[k]); ax.plot([x[i]], [y[i]], "o", color=col, ms=6, mec=SURF, mew=1.5)
    ax.set_xlabel("training step"); ax.set_ylabel("validation NLL per token (45 held-out songs)")
    ax.set_ylim(0.9, 1.7); ax.set_xlim(0, 40000)
    ax.legend(frameon=False, fontsize=8, loc="lower right", title="GPT-2 8L x 512, p=4 tokens (dot = minimum)", title_fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "val_curves.png"), dpi=220); plt.close(fig)


def fig_density():
    q = {"real 32-bar crops": [18.4, 20.1, 21.7, 22.0]}
    typed = {r["sampling"]: r for r in rows("gen_typed/results_v2_typed*.jsonl") if r["seed"] == 0}
    q["v2, pure T=1.0"] = typed["B0_pure"]["notes_per_bar_quarters"]
    q["v2, pure T=1.0 + Bar bias 3"] = typed["B_bias3"]["notes_per_bar_quarters"]
    cols = [INK2, C["blue"], C["orange"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    xs = ["bars 1-8", "bars 9-16", "bars 17-24", "bars 25-32"]
    for (name, v), col in zip(q.items(), cols):
        ax.plot(xs, v, color=col, lw=2, marker="o", ms=6, mec=SURF, mew=1.5, label=name)
        ax.annotate(name, (3, v[3]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax.set_ylabel("notes per bar"); ax.set_ylim(10, 38); ax.set_xlim(-0.2, 4.6)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "density.png"), dpi=220); plt.close(fig)


def fig_bonus_ca():
    def ca(r):
        m = r["metrics"]; return m.get("CA", next((v for k, v in m.items() if k.endswith("Average CA")), None))
    label = {  # results-file stem -> (label, group) ; group 0 = no chord input, 1 = chord labels given
        "bA1_A_L1_nochord": ("A, no chord", 0), "bB1_B_6L_nochord": ("B, no chord", 0),
        "bA7_A_L1_predchord": ("A, chord predicted by the model", 0), "bB5_B_6L_predchord": ("B, chord predicted by the model", 0),
        "bB6_B_6L_decchord_40k_s1greedy": ("two-stage: stage-1 chords -> B (40k)", 0),
        "bA2_A_L1_chord": ("A, chord in melody segment", 1), "bA8_A_L1_forcedchord": ("A, chord in piano segment", 1),
        "bB2_B_6L_chord": ("B, chord in encoder", 1), "bB4_B_6L_decchord": ("B, chord in decoder (20k steps)", 1),
        "bB6_B_6L_decchord_40k": ("B, chord in decoder (40k steps)", 1)}
    items = []
    for f in sorted(glob.glob(os.path.join(R, "gen_bonus", "results_b*.jsonl"))):
        stem = os.path.basename(f)[len("results_"):-len(".jsonl")]
        if stem not in label:
            continue
        v = [ca(json.loads(l)) for l in open(f) if json.loads(l)["tag"].endswith("T0.9_p0.9") and json.loads(l)["tag"].startswith("best__")]
        if v:
            items.append((label[stem][0], st.mean(v), st.pstdev(v), label[stem][1], len(v)))
    items.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, (n, m, s_, g, k) in enumerate(items):
        ax.barh(i, m, xerr=s_, color=C["orange"] if g else C["blue"], height=0.62, error_kw={"ecolor": INK2, "lw": 1, "capsize": 2})
        ax.text(m + s_ + 0.012, i, f"{m:.2f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(items))); ax.set_yticklabels([t[0] for t in items], fontsize=8)
    ax.set_ylim(-0.6, len(items) + 0.2)
    ax.axvline(0.7625, color=INK2, lw=1, ls=(0, (4, 3))); ax.text(0.7625, len(items) - 0.1, "real piano 0.76 ", fontsize=7.5, color=INK2, ha="right", va="bottom")
    ax.axvline(0.0125, color=INK2, lw=1, ls=(0, (4, 3))); ax.text(0.0125, len(items) - 0.1, " wrong-song piano 0.01", fontsize=7.5, color=INK2, ha="left", va="bottom")
    ax.set_xlabel("Chord Accuracy (mean +- sd over 3 seeds)"); ax.set_xlim(0, 0.9); ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C["blue"], label="no chord input"), Patch(color=C["orange"], label="chord labels given")], frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "bonus_ca.png"), dpi=220); plt.close(fig)
    return items


def fig_cfg():
    """FA15 (chord dropout) given-chord CA vs classifier-free-guidance scale, per sampling config + mean of OA pitch; final-model reference lines"""
    pat = "final/results_FA15_A_L1_chordboth_chordemb_drop0.15_full_cfg*.jsonl"
    d = {}
    for r in rows(pat):
        if r.get("cfg") is None:
            continue
        d.setdefault((r["cfg"], r["tag"].split("__")[1]), []).append(r["metrics"])
    scales = sorted({k[0] for k in d})
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    cols = {"T1.0_pure": C["blue"], "T1.0_p0.95": C["aqua"], "T0.9_p0.9": C["orange"], "T1.2_k5": C["yellow"]}
    for t, col in cols.items():
        y = [st.mean(m["CA"] for m in d[(w, t)]) for w in scales]
        ax.plot(scales, y, color=col, lw=2, marker="o", ms=5, mec=SURF, mew=1.2, label=t.replace("_", " "))
    ax.axhline(0.7625, color=INK2, lw=1, ls=(0, (4, 3))); ax.text(6.0, 0.7625 + 0.008, "real piano 0.76", fontsize=7.5, color=INK2, ha="right")
    ax.set_xlabel("guidance scale w (1 = no guidance)"); ax.set_ylabel("Chord Accuracy (mean of 3 seeds)")
    ax.set_ylim(0.4, 1.0); ax.legend(frameon=False, fontsize=8, loc="lower right", title="sampling", title_fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "cfg_curve.png"), dpi=220); plt.close(fig)


if __name__ == "__main__":
    fig_val_curves(); fig_density(); fig_cfg(); it = fig_bonus_ca(); print("figures ->", OUT); [print("  %-42s %.3f +- %.3f (n=%d)" % (t[0], t[1], t[2], t[4])) for t in it]
