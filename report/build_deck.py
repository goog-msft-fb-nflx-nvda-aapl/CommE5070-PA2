"""Builds the 16:9 report deck (python-pptx) from the measured numbers in WORKLOG/results; convert with LibreOffice to PDF.
Placeholders that need the author: CLOUD_LINK, TASKB_BEST (set after listening)."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = lambda n: os.path.join(HERE, "figs", n)
STUDENT = "R13921031"
CLOUD_LINK = "https://drive.google.com/drive/folders/19jENIBlJ3hmoRu9UIrka1VXZJrQdtPsP?usp=sharing"
TASKB_BEST = {"001": None, "002": None, "003": None}   # e.g. "cfg2"; None = listening not done yet

INK, INK2, BLUE, ORANGE, BG, RULE = RGBColor(0x0b, 0x0b, 0x0b), RGBColor(0x52, 0x51, 0x4e), RGBColor(0x2a, 0x78, 0xd6), RGBColor(0xeb, 0x68, 0x34), RGBColor(0xfc, 0xfc, 0xfb), RGBColor(0xe6, 0xe5, 0xe1)
RED = RGBColor(0xc0, 0x1f, 0x1f)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
N = [0]


def slide(title, sub=None):
    s = prs.slides.add_slide(BLANK)
    N[0] += 1
    bg = s.background.fill; bg.solid(); bg.fore_color.rgb = BG
    tb = s.shapes.add_textbox(Inches(0.55), Inches(0.3), Inches(12.2), Inches(0.8)); tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; r = p.add_run(); r.text = title; r.font.size = Pt(28); r.font.bold = True; r.font.color.rgb = INK
    if sub:
        p2 = tf.add_paragraph(); r2 = p2.add_run(); r2.text = sub; r2.font.size = Pt(14); r2.font.color.rgb = INK2
    ln = s.shapes.add_shape(1, Inches(0.55), Inches(1.28), Inches(12.2), Emu(15000)); ln.fill.solid(); ln.fill.fore_color.rgb = BLUE; ln.line.fill.background()
    ft = s.shapes.add_textbox(Inches(0.55), Inches(7.05), Inches(12.2), Inches(0.3)).text_frame
    pr = ft.paragraphs[0]; rr = pr.add_run(); rr.text = f"CommE5070 HW2 · {STUDENT} · {N[0]}"; rr.font.size = Pt(10); rr.font.color.rgb = INK2
    return s


def bullets(s, items, x, y, w, h, size=16, gap=6):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    first = True
    for it in items:
        lvl = 0
        if isinstance(it, tuple):
            lvl, it = it
        p = tf.paragraphs[0] if first else tf.add_paragraph(); first = False
        r = p.add_run(); r.text = ("• " if lvl == 0 else "– ") + it
        r.font.size = Pt(size - 2 * lvl); r.font.color.rgb = INK if lvl == 0 else INK2
        p.level = 0; p.space_after = Pt(gap)
        if lvl:
            p.margin_left = Inches(0.35)
    return tb


def note(s, text, x, y, w, h, size=12, color=INK2, bold=False, link=None):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; r = p.add_run(); r.text = text; r.font.size = Pt(size); r.font.color.rgb = color; r.font.bold = bold
    if link:
        r.hyperlink.address = link


def table(s, data, x, y, w, col_w=None, size=12, row_h=0.36, hl_rows=()):
    rows, cols = len(data), len(data[0])
    t = s.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(row_h * rows)).table
    if col_w:
        tot = sum(col_w)
        for i, cw in enumerate(col_w):
            t.columns[i].width = Inches(w * cw / tot)
    for i, row in enumerate(data):
        t.rows[i].height = Inches(row_h)
        for j, v in enumerate(row):
            c = t.cell(i, j); c.text = str(v); c.margin_left = c.margin_right = Inches(0.06); c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            para = c.text_frame.paragraphs[0]; para.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            f = para.runs[0].font; f.size = Pt(size); f.bold = (i == 0 or i in hl_rows)
            c.fill.solid()
            if i == 0:
                c.fill.fore_color.rgb = BLUE; f.color.rgb = RGBColor(255, 255, 255)
            else:
                c.fill.fore_color.rgb = RGBColor(0xee, 0xf4, 0xfc) if i in hl_rows else (BG if i % 2 else RGBColor(0xf3, 0xf3, 0xf1)); f.color.rgb = INK
    return t


def pic(s, name, x, y, w=None, h=None):
    return s.shapes.add_picture(FIG(name), Inches(x), Inches(y), width=Inches(w) if w else None, height=Inches(h) if h else None)


# ---- 1 title
s = slide("Symbolic music generation on POP909, Transformers trained from scratch",
          "Unconditional 32-bar generation · prompt continuation · melody→piano accompaniment (bonus)")
bullets(s, [f"Student: {STUDENT} (CommE5070, HW2)",
            "All models: Transformers (GPT-2 / T5 / Llama-style decoder) trained from random initialisation on the 896 POP909 training songs only; no pretrained weights, no extra data",
            "Code, README, requirements, checkpoints, generated .mid/.wav: see link below",
            "Everything reported is measured by us with the unmodified TA scripts (eval_basic.py, eval_bonus.py); each result row stores its full config"],
        0.55, 1.6, 12.2, 3, size=18, gap=10)
note(s, "Cloud drive (code + README.md + requirements.txt + checkpoints + generation results):", 0.55, 5.1, 12.2, 0.4, size=16, color=INK, bold=True)
note(s, CLOUD_LINK, 0.55, 5.55, 12.2, 0.5, size=15, color=BLUE, bold=True, link=CLOUD_LINK)
note(s, "Run inference: python infer.py {uncond|continue|bonus} --out_dir OUT  (see README, output path is settable; writes .mid and .wav)", 0.55, 6.3, 12.2, 0.5, size=13)

# ---- 2 overview
s = slide("What was built", "Three tasks, one tokenizer family, two model families")
table(s, [["Task", "Model (from scratch)", "Training", "Inference"],
          ["a. Unconditional, 32 bars", "GPT-2 8L×512×8h, ctx 4096, ~27M params", "20k steps, batch 8, dropout 0.3, AdamW 3e-4 cosine, all 896 songs", "sliding-window sampling, 3 sampling configs (+2 extra)"],
          ["b. Continuation, 8→32 bars", "same checkpoint as Task a (no retraining)", "-", "prompt kept verbatim; 3 configs per prompt"],
          ["Bonus. Melody → piano", "A: GPT-2 bar-interleaved decoder-only;  B: T5 6+6 d512; stage-1 chord predictor T5 4+4", "10k (A) / 40k (B) steps, dropout 0.3", "forced melody, sampled piano; chord given vs melody-only"]],
      0.55, 1.6, 12.2, col_w=[2.2, 3.6, 3.6, 3.0], size=12, row_h=0.8)
bullets(s, ["Representation: the TA's REMI-style tokens (Bar, Pos, Tempo, Pitch, Vel, Dur), grids p4 (16th notes) and p16",
            "Protocol: settings chosen on dev runs (5% of songs held out); reported models retrained on all 896 songs, last checkpoint, no checkpoint picking",
            "Reported numbers: means over 3 generation seeds; training-seed replicates (3 seeds) for the headline models, so the noise level is known"],
        0.55, 5.1, 12.2, 1.8, size=14, gap=5)

# ---- 3 data / tokens
s = slide("Data and tokenization", "Two things that change numbers: the onset grid and the metric references")
bullets(s, ["POP909: 909 pop piano performances (MELODY / BRIDGE / PIANO + beat/chord labels); 896 for training, 13 TA prompts held out",
            "TA REMI tokens per note: Pos_x, Pitch, Vel, Dur (+ Bar, Tempo). Vocab ≈ 257–284 (p4) / 545–572 (p16)",
            "Finding: the TA tokenizer ignores pos_per_beat for onsets, so a 4-per-beat grid silently dropped 12% of events as out-of-vocabulary; our wrapper tokenize_hw2.py quantizes onsets to the grid first (TA script used unmodified)",
            "The Grooving-Similarity reference depends on the grid: real 32-bar crops give GS 0.887 at p4 but 0.962 at p16, so each model is compared with the reference of its own grid",
            "Real-data references (eval_basic, 32-bar crops): H1 2.118, H4 2.617; real density 20.6 notes/bar",
            "Bars/beats come from each song's beat_midi.txt (TA tokenizer); the .mid time-signature fields, which are mostly wrong in POP909, are never read, and MidiTok is not used",
            "Augmentation: transposition −5…+6 semitones; random crops of 32 bars (Task a) or 16–24 bars (bonus)"],
        0.55, 1.6, 12.2, 5.3, size=16, gap=8)

# ---- 4 Task a recipe
s = slide("Task a: model, training and the validation-based choices", "GPT-2 8L×512, dropout 0.3, 20k steps, p16 grid")
pic(s, "val_curves.png", 0.55, 1.5, w=6.3)
bullets(s, ["Dev runs on the p4 grid (45 held-out songs): dropout 0.1 overfits after ≈9k steps; dropout 0.3 keeps improving → final recipe dropout 0.3, 20k steps",
            "Final model F1 retrained on all 896 songs, last checkpoint",
            "Generation: sliding window over bars (no rejection sampling, no restarts); malformed tokens are dropped, counted and reported",
            "Sampling configs were chosen by a protocol (entropy-histogram / perplexity matching on dev), not only by looking at the test metric",
            "Density caveat: pure sampling writes 29.6 notes/bar vs 20.6 in real crops; a Bar-logit bias of 3 lowers it to 23.4"],
        7.0, 1.5, 5.8, 5.2, size=14, gap=8)

# ---- 5 Task a results
s = slide("Task a results: 3 required configs + 2 extra", "20 pieces × 3 seeds each, unmodified eval_basic.py, p16; real: H1 2.118 / H4 2.617 / GS 0.962")
table(s, [["Inference config", "H1", "H4", "GS", "notes/bar", "malformed tok. (60 pcs)"],
          ["real data (reference)", "2.118", "2.617", "0.962", "20.6", "-"],
          ["cfg1: T1.0 pure (submitted)", "2.169 ± 0.039", "2.600 ± 0.031", "0.963 ± 0.005", "29.6", "227"],
          ["cfg2: T1.0 + Bar-logit bias 3", "2.105 ± 0.016", "2.594 ± 0.050", "0.966 ± 0.005", "23.4", "157"],
          ["cfg3: T0.9 top-p 0.9", "2.055 ± 0.040", "2.411 ± 0.065", "0.995 ± 0.002", "29.4", "0"],
          ["extra: T1.0 top-p 0.95", "2.191 ± 0.018", "2.552 ± 0.018", "0.988 ± 0.004", "30.8", "36"],
          ["extra: T1.2 top-k 5", "2.237 ± 0.032", "2.570 ± 0.014", "0.959 ± 0.009", "36.9", "0"]],
      0.55, 1.65, 12.2, col_w=[3.6, 1.6, 1.6, 1.6, 1.2, 2.0], size=13, row_h=0.42, hl_rows=(2,))
bullets(s, ["± = standard deviation over the 3 generation seeds; 20 pieces per seed",
            "Submitted default = cfg1 (pure): closest to real on H4 and GS among the required metrics, and the entropy-matching protocol also picks T = 1.0 untruncated",
            "cfg1 vs cfg2 differences in H4/GS are within seed noise; the Bar bias mainly changes density (29.6 → 23.4, real 20.6)",
            "Training-seed replicates of F1 pure (H1/H4/GS): seed 0 2.169/2.600/0.963, seed 1 2.200/2.646/0.964, seed 2 2.174/2.631/0.966, so training-seed spread of H4 ≈ 0.05",
            "Truncated sampling (top-p 0.9) pushes GS to 0.995, above the real value: more repetitive than real music"],
        0.55, 4.75, 12.2, 2.3, size=13, gap=4)
note(s, "Task a submission: 20 .mid + 20 .wav of cfg1, seed 0 (H1 2.212 / H4 2.633 / GS 0.960 for that exact set).", 0.55, 6.72, 12.2, 0.3, size=12, color=INK2)

# ---- 6 Task a ablations + SI
s = slide("Task a ablations and structureness (SI)", "Same backbone and budget, p4 grid unless stated; 3 seeds × 20 pieces, pure sampling")
table(s, [["variant", "H1", "H4", "GS (p4 real 0.887)", "note"],
          ["REMI p4 (F4)", "2.122", "2.568", "0.919", "reference"],
          ["+ chord tokens (F5)", "2.148", "2.532", "0.924", "no clear change"],
          ["+ label smoothing 0.1 (F6)", "2.012", "2.785", "0.721", "breaks free sampling"],
          ["MIDI-like (note-on/off)", "1.917", "2.141", "0.837", "lower pitch entropy"],
          ["Compound Word", "2.099", "2.566", "0.825", "density closest to real"],
          ["Llama-style RoPE (p16)", "2.142", "2.640", "0.967", "vs GPT-2 2.169/2.600/0.963"],
          ["+ pitch repetition penalty α=1", "2.576", "3.079", "0.965", "H1/H4 far above real"],
          ["+ velocity augmentation (p16)", "2.203", "2.582", "0.957", "no measurable effect"]],
      0.55, 1.6, 7.2, col_w=[3.0, 0.9, 0.9, 1.5, 2.6], size=11, row_h=0.4)
table(s, [["SI (20 pieces)", "short", "mid", "long"],
          ["real crops", "0.369", "0.376", "0.344"],
          ["F1 pure", "0.330", "0.290", "0.225"],
          ["F1 Bar bias 3", "0.319", "0.271", "0.210"],
          ["F1 top-p 0.95", "0.528", "0.517", "0.449"],
          ["F1 T0.9 top-p 0.9", "0.657", "0.692", "0.635"],
          ["RoPE pure", "0.287", "0.257", "0.212"]],
      8.1, 1.6, 4.7, col_w=[2.2, 1, 1, 1], size=11, row_h=0.4)
bullets(s, ["SI uses the MusDr scape-plot code on rendered audio; only 20 pieces per set",
            "Pure sampling is closest to real SI; truncated sampling is far above (repetitive)",
            "Dev validation NLL per REMI token: REMI 0.952, MIDI-like 1.211, Compound Word 1.275"],
        8.1, 4.6, 4.7, 2.3, size=13, gap=6)
note(s, "Training-seed noise (H4 ≈ 0.05) is as large as most differences in this table; read single-run gaps below that as unresolved.", 0.55, 5.35, 7.2, 0.8, size=12, color=INK2)

# ---- 7 Task b
s = slide("Task b: continuing the 3 TA prompts (8 → 32 bars)", "Task-a checkpoint F1, prompt kept verbatim, 3 inference configs per prompt")
table(s, [["prompt", "cfg1: T1.0 pure", "cfg2: T1.0 + Bar bias 3", "cfg3: T0.9 top-p 0.9", "best by listening"],
          ["001", "001_cfg1.mid/.wav", "001_cfg2.mid/.wav", "001_cfg3.mid/.wav", TASKB_BEST["001"] or "(pending)"],
          ["002", "002_cfg1.mid/.wav", "002_cfg2.mid/.wav", "002_cfg3.mid/.wav", TASKB_BEST["002"] or "(pending)"],
          ["003", "003_cfg1.mid/.wav", "003_cfg2.mid/.wav", "003_cfg3.mid/.wav", TASKB_BEST["003"] or "(pending)"]],
      0.55, 1.6, 12.2, col_w=[1.0, 2.6, 2.8, 2.6, 2.2], size=13, row_h=0.45)
pic(s, "density.png", 0.55, 3.55, w=5.4)
bullets(s, ["The three configs differ in how much they deviate from the prompt: the Bar-logit bias keeps note density closest to the prompt's own (dev-checkpoint measurement, notes/bar prompt → cfg1 / cfg2 / cfg3: 001 15.5 → 17.5 / 16.0 / 23.8; 002 22.1 → 27.0 / 21.0 / 26.7; 003 22.8 → 32.1 / 24.7 / 45.0)",
            "Truncated sampling (cfg3) is the densest and most repetitive continuation",
            "No objective metric is defined for this task; the 'best' column is the author's own listening choice (single rater, no listening test)",
            "Figure (dev model, not F1): density per quarter of a generated piece, real vs pure sampling vs Bar bias 3"],
        6.2, 3.5, 6.6, 3.4, size=13, gap=6)

# ---- 8 Bonus architectures
s = slide("Bonus: melody → piano accompaniment, models", "Both options named in the spec, plus a chord-conditioning study")
bullets(s, ["A. Decoder-only, bar-interleaved: <M> melody-bar <P> piano-bar …, lookahead L future melody bars; melody forced at inference, piano sampled; loss on piano tokens only",
            "B. Encoder-decoder (HF T5ForConditionalGeneration from random init, 6+6 layers, d512): encoder = melody, decoder = piano",
            "Chord conditioning variants: none · chord token in melody segment · at start of piano segment · in T5 decoder · in-attention chord embedding (learned embedding of the current bar's chord added to every token of the bar)",
            "Stage-1 chord predictor (T5 4+4) → melody-only two-stage pipeline: predicts one chord per bar from the melody, stage 2 then uses them; no chord file is read",
            "Conditioning dropout + classifier-free guidance (CFG, FA15): during training all chord tokens of an example are replaced by 'no chord' with probability 0.15; at inference logp = logp_uncond + w·(logp_cond − logp_uncond), w = 3 in the submission",
            "TA confirmed that the per-bar chord labels (chord_midi.txt) may be used as model input and that CA is expected to be higher with them; we compare the two settings (chord-given vs melody-only) on the same metric. The chord-given model is the main bonus submission"],
        0.55, 1.55, 12.2, 5.4, size=15, gap=9)

# ---- 9 Bonus chord-given
s = slide("Bonus results with chord labels as input", "10 prompts × 16 bars, 3 seeds × 4 sampling configs; real piano CA 0.7625, wrong-song piano CA 0.0125")
table(s, [["model (chord labels given)", "CA", "OA pitch", "OA vel", "OA dur", "OA IOI"],
          ["B T5 6+6, chord in decoder (F2)", "0.587–0.656", "0.843–0.889", "0.553–0.607", "0.583–0.671", "0.469–0.558"],
          ["A L1, chord in melody seg. (FA2)", "0.567–0.669", "0.875–0.889", "0.524–0.570", "0.668–0.703", "0.487–0.591"],
          ["A L1, chord in both segs (FA11)", "0.494–0.583", "0.880–0.888", "0.511–0.567", "0.650–0.691", "0.435–0.516"],
          ["A L1 + chord embedding (FA12)", "0.629–0.660", "0.885–0.890", "0.552–0.579", "0.666–0.690", "0.472–0.519"],
          ["FA15 (+ chord dropout), w = 1", "0.538–0.637", "0.878–0.889", "0.538–0.562", "0.655–0.699", "0.495–0.551"],
          ["FA15 + CFG w = 3 (submitted)", "0.835–0.883", "0.883–0.890", "0.520–0.572", "0.689–0.708", "0.489–0.520"]],
      0.55, 1.55, 7.4, col_w=[3.0, 1.2, 1.2, 1.2, 1.2, 1.2], size=10, row_h=0.42, hl_rows=(6,))
pic(s, "cfg_curve.png", 8.1, 1.5, w=4.8)
bullets(s, ["Ranges are over the 4 sampling configs of the 3-seed means",
            "CFG raises CA from ≈0.6 to ≈0.88 and saturates at w ≈ 3–4; OA stays in the same range; it uses only the given chords, no metric-based selection",
            "Training-seed replicates (CA, pure/p0.95/T0.9/T1.2): FA12 seed 1 0.67/0.69/0.75/0.64, seed 2 0.58/0.61/0.63/0.63 → training-seed spread ≈ 0.1 CA, as large as many gaps in this table",
            "Assumption check (TA): with chord labels CA 0.84 vs 0.23 melody-only (T0.9 top-p 0.9, seed 0, submitted files) → labels raise CA as expected",
            "CA 0.84–0.88 lies above the 0.7625 real-piano ceiling: generation agrees with the chord labels more strongly than real piano does; that measures agreement with the labels, not better music"],
        0.55, 4.6, 12.2, 2.4, size=12, gap=3)
note(s, "Submitted chord_given set (seed 0): CA 0.844, OA 0.881 / 0.521 / 0.680 / 0.514.", 8.1, 4.1, 4.8, 0.5, size=11)

# ---- 10 Bonus melody-only + negative
s = slide("Bonus results without chord labels, and what did not help", "Melody-only baseline for the chord-given comparison; ranges over 4 sampling configs")
table(s, [["model (no chord file)", "CA", "OA pitch", "OA vel", "OA dur", "OA IOI"],
          ["two-stage: stage-1 chords → F2 (provided)", "0.183–0.231", "0.846–0.848", "0.575–0.622", "0.620–0.671", "0.476–0.528"],
          ["A L1, no chord (FA1)", "0.146–0.179", "0.847–0.860", "0.534–0.572", "0.642–0.675", "0.470–0.533"],
          ["A L1, chord generated in piano seg. (FA7)", "0.121–0.175", "0.844–0.859", "0.510–0.550", "0.661–0.694", "0.444–0.535"],
          ["B T5, no chord (FB1)", "0.131–0.148", "0.829–0.862", "0.549–0.622", "0.636–0.729", "0.427–0.505"],
          ["+ chord aux. loss (FA13 w0.3 / FA14 w1.0)", "0.138–0.173 / 0.135–0.162", "-", "-", "-", "-"],
          ["+ key-aware pitch bias 1 / 2 / 4", "0.173 / 0.179 / 0.140 (ctrl 0.177)", "-", "-", "-", "-"],
          ["two-stage FA15 + CFG w = 3", "0.231 (w=1: 0.183)", "0.842", "0.541", "0.696", "0.496"]],
      0.55, 1.55, 12.2, col_w=[4.2, 2.6, 1.3, 1.3, 1.3, 1.3], size=11, row_h=0.4)
bullets(s, ["Melody alone gives CA ≈ 0.15–0.23 (real-piano ceiling 0.76); the stage-1 chord predictor's free-running chord accuracy is only 0.273, which bounds the two-stage pipeline: better conditioning (CFG) gives +0.05, not more",
            "No resolved effect: chord auxiliary loss, key-aware pitch bias, lookahead 2 (CA 0.21–0.32 with chord labels), A initialised from the unconditional model, melody-token loss, T5 relative-bucket changes, velocity augmentation (Task a)",
            "Metric-guided variants, reported separately and not as the main result: chord-triad logit bias +1/+2/+4 CA 0.748 / 0.871 / 0.921 and best-of-8 reranking CA 0.912–0.925 on the chord-given F2 (they use the labels the metric scores)",
            "Also provided: melody_only set (seed 0): CA 0.231, OA 0.837 / 0.537 / 0.597 / 0.443"],
        0.55, 4.9, 12.2, 2.2, size=12, gap=4)

# ---- 11 caveats
s = slide("Reliability, limits and honest caveats")
bullets(s, ["Noise is measured, not assumed: 3 training-seed replicates of F1, FA12 and F2. Spread: H4 ≈ 0.05, bonus CA ≈ 0.1; gaps smaller than this are not claimed",
            "Prior published numbers: we looked for results under the same TA metric implementation and could not verify any published number as comparable (different metric definitions, grids, splits); the report therefore compares only against real data, controls and our own ablations",
            "Controls: real piano CA 0.7625; wrong-song piano CA 0.0125; round-trip through our tokens keeps OA pitch 0.980 / vel 0.973 / dur 0.870 / IOI 0.745 (p4), so duration and IOI OA are partly capped by the grid",
            "GS depends on the onset grid (real 0.887 at p4, 0.962 at p16): p4 and p16 numbers are never compared across grids",
            "Task a metrics are distribution statistics; they do not measure musical quality. Task b 'best' and all listening impressions come from one rater (the author); there was no listening test",
            "CA is bounded and per-bar: bars labelled N/dim/aug/sus can never match; 160 bars per seed, one bar = 0.006 CA",
            "Chord-given vs melody-only: the TA confirmed chord labels may be used as input; the assumption that labels raise CA holds here (0.84 with labels vs 0.23 without, same sampling, seed 0)"],
        0.55, 1.5, 12.2, 5.5, size=15, gap=9)

# ---- 12 references / repro
s = slide("Reproducibility, code reuse and references")
bullets(s, ["Run: python infer.py uncond|continue|bonus --out_dir OUT (README); seeds fixed; a clean-environment test reproduced the objective metrics of the experiment runs to 4 decimals",
            "Every result row (results/*.jsonl) stores config, checkpoint hash, code hashes, seed",
            "TA scripts (tokenization.py, eval_basic.py, eval_bonus.py) are used unmodified; HF Transformers (GPT2LMHeadModel, T5ForConditionalGeneration, Llama config) with random initialisation; MusDr (Wu et al.) for GS/SI with a one-line librosa fix; pretty_midi, FluidSynth for .wav",
            "Ideas followed (re-implemented by us, no code or weights copied): REMI (Huang & Yi, Pop Music Transformer, 2020); Compound Word Transformer (Hsiao et al., 2021); Compose & Embellish (Wu & Yang, 2023) for entropy matching / structureness; classifier-free guidance (Ho & Salimans, 2022); T5 (Raffel et al., 2020); RoPE (Su et al., 2021); POP909 (Wang et al., 2020)",
            "Released checkpoints: uncond_final.pt (F1), bonus_final.pt (F2) + bonus_stage1.pt (F3), bonus_chord_final.pt (FA15)"],
        0.55, 1.5, 12.2, 5.4, size=15, gap=10)

out = os.path.join(HERE, f"{STUDENT}_report.pptx")
prs.save(out)
print("saved", out, "slides", N[0])
