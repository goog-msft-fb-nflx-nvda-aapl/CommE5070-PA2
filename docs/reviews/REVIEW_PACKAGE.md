# External review package: symbolic music generation from scratch on POP909 (course homework, solo, ~3 weeks)

## 0. Instructions for the reviewing LLM (paste this whole file)

You are reviewing a small-scale research project. Use web search / browsing. Everything below is **measured by us**; nothing is claimed as state of the art. Please:

1. **Benchmark**: find the strongest published or released systems for each of the three tasks below (venues: ISMIR, ICASSP, Interspeech, AAAI, ACL/EMNLP, NeurIPS/ICML/ICLR, KDD, TISMIR, arXiv, GitHub, Hugging Face), and report, with links, their numbers on the *same or closest metrics* (H1/H4 pitch-class entropy, groove similarity GS, structureness indicator SI, chord accuracy CA, overlapped area OA of pitch/velocity/duration/IOI) and the data/protocol they used (POP909 or not, train-from-scratch or pretrained, model size, steps, bars, tokenization). Mark every claim you cannot verify as "unverified". Do not invent numbers.
2. **Position our numbers**: for each table in section 5, say whether our value is plausible, good, mediocre or suspicious *relative to those systems under comparable constraints* (Transformer trained from scratch, POP909 only, ~900 songs, one GPU-day-scale budget). Say explicitly when a comparison is not apples-to-apples (different metric definition, grid, data split).
3. **Audit method and protocol**: list methodological errors, unfair comparisons, leakage risks, over-claims, metric pitfalls (see section 6 for our own known caveats), and anything in the evaluation scripts' behavior that could make a number misleading.
4. **Gaps**: list approaches that are legal under the constraints (section 1) and that the literature suggests would move the numbers, which we have NOT tried (section 4 lists what we tried). Rank the top 5 by expected gain per GPU-hour and say which single number would decide each.
5. **Do not suggest** pretrained weights, extra data, or anything outside the constraints.

Output format: numbered headings 1-5 matching the list above; links for every claim; a final table "our number vs best comparable published number".

## 1. Problem and constraints (from the course spec)

Course CommE5070 (Deep Learning for Music/Audio, NTU), HW2. Dataset **POP909** (909 Mandopop piano performances; MELODY / BRIDGE / PIANO tracks, beat and chord annotations); 896 songs for training, the rest are held out as prompts.

Rules: Transformer-based, **trained from scratch** (no pretrained checkpoints), **POP909 only**, use the whole training set, public code may be reused with citation. Graded entirely on a report.

Tasks:
- **Task a, unconditional**: generate 32-bar pop piano pieces. At least 3 inference configurations x 20 pieces, scored with the TA script `eval_basic.py` (MusDr: H1 = 1-bar pitch-class entropy, H4 = 4-bar pitch-class entropy, GS = groove similarity; SI optional). Closer to the real-data value is better (not larger is better).
- **Task b, continuation**: reuse the Task-a checkpoint (no retraining) to extend three 8-bar TA prompts to 32 bars, 3 inference configs each. No objective metric; judged by listening.
- **Bonus, accompaniment**: given 10 MELODY-track prompts (16 bars, unseen songs), generate the PIANO track. Scored with the TA script `eval_bonus.py`: **CA** (chord accuracy: per bar, the best of 24 major/minor triads by bass-weighted chroma cosine similarity of the generated piano must equal the ground-truth dominant chord of the bar from `chord_midi.txt`; bars with dim/aug/sus/N labels can never match) and **OA** (overlapped area of KDE-smoothed per-bar distributions of pitch class, velocity, duration, inter-onset interval, generated vs ground-truth piano). Two architecture options are named in the spec: (A) interleave melody and piano tokens in one decoder-only model, (B) encoder-decoder (e.g. HF T5).
- Open question for the TA (not answered yet): whether the prompt folder's chord file may be used as *model input* in the bonus. We therefore report chord-given and melody-only (no chord file) results separately.

Compute actually used: up to 3 H200 GPUs on a shared server; all final models train in about 12 to 75 minutes each (10k-40k steps).

## 2. Data, representation, metrics plumbing

- Tokenization: the TA's REMI-style tokenizer (Bar, Pos_x, Tempo, Pitch, Vel, Dur per note), wrapped by us (`tokenize_hw2.py`) because the TA script ignores `pos_per_beat` for onsets (a 4-per-beat grid silently dropped 12% of events as out-of-vocabulary). Two grids: p4 (16th-note grid, vocab ~257-284) and p16 (vocab 545-572). Measured real-data averages: ~4.1 melody notes/bar (16 tokens), ~12.9 piano notes/bar (52 tokens); 20.6 notes/bar for the 32-bar unconditional crops.
- Training crops: random 32-bar (Task a) or 16-24-bar (bonus) crops, transposition augmentation of -5..+6 semitones.
- **Reference values for Task a (real 32-bar crops, unmodified eval_basic.py)**: H1 2.118, H4 2.617, GS 0.887 at p4; the GS reference moves to **0.962 at p16** (the metric depends on the onset grid, so p4 and p16 numbers are compared with their own reference).
- Protocol: settings (steps, dropout) were chosen on dev runs with 5% of songs held out; the reported "final" models are retrained on all 896 songs with those settings and use the last checkpoint (no checkpoint cherry-picking). Generation uses sliding-window decoding (no rejection sampling), fixed seeds (0, 1, 2), 20 pieces per seed for Task a and 10 prompts per seed for the bonus. All numbers are means over 3 generation seeds; each model was trained **once** (one training seed).
- Controls we measured: real piano through the CA estimator (CA-ceiling) = **0.7625**; a piano from a *different* song = CA **0.0125**, OA 0.846 / 0.384 / 0.649 / 0.464 (pitch/vel/dur/IOI); piano re-encoded and decoded through our tokens (round-trip) OA 0.980 / 0.973 / 0.870 / 0.745 at p4 and 0.959 (dur) / 0.935 (IOI) at p16 (so duration and IOI OA are capped by the grid).

## 3. Models and training recipes

- **Unconditional (Task a)**: GPT-2 (HF, random init) 8 layers, d=512, 8 heads, context 4096 tokens, learned absolute positions, dropout 0.3, AdamW lr 3e-4 cosine, 20k steps, batch 8, bf16. ~26-28M parameters.
- **Bonus A (decoder-only)**: same GPT-2 family, bar-interleaved `<M> melody-bar <P> piano-bar ...`, lookahead L = number of future melody bars shown before each piano bar (0, 1, 2), inference loop forces the melody segments and samples the piano segments; loss on piano tokens only. Chord modes: none / chord token in melody segment / chord token forced at the start of the piano segment / chord generated by the model / (new) learned in-attention chord embedding added to every token of the bar.
- **Bonus B (encoder-decoder)**: HF `T5ForConditionalGeneration` from random init, 6+6 layers, d=512, 8 heads, dropout 0.3, 40k steps (encoder = whole melody, decoder = whole piano). Chord modes: none / chord token in encoder / chord token in decoder (forced bar by bar at inference) / chord generated by the decoder.
- **Stage-1 chord predictor (two-stage melody-only pipeline)**: T5 4+4, d=256; predicts one chord token per bar from the melody; the stage-2 model then receives the predicted chords instead of the annotation. This is the only configuration where no chord file is used and chords still help.
- Inference levers we implemented: temperature / top-k / top-p, per-token-type temperature, Bar-logit bias (controls note density), greedy and beam search, entropy-histogram matching of sampling settings (Compose & Embellish style), chord-triad logit bias, best-of-N selection, pitch-only repetition penalty.

## 4. What we tried (all runs have full configs and per-seed rows in the repo)

Task a: REMI p4 vs p16 grid; dropout 0.1 vs 0.3; chord tokens in the sequence; label smoothing 0.1; MIDI-like representation (note-on/off + time-shift, with a verified lossless normalization); Compound Word (per-field heads); RoPE/RMSNorm/SwiGLU decoder instead of GPT-2; sampling grid; Bar-logit bias; greedy/beam; entropy-histogram matching; pitch-only repetition penalty; SI on generated sets.

Bonus: arch A vs B; lookahead 0/1/2; chord input position (melody segment, piano segment, decoder, encoder, generated, in-attention embedding); two-stage melody-only; p4 vs p16 grid; T5 relative-position buckets (32 vs 128) and size (6+6 d512 vs 4+4 d256); initialization of A from our unconditional checkpoint; loss on melody tokens; per-token-type temperatures; best-of-8 selection; chord-triad logit bias; wrong-song / shuffled-chord / round-trip controls.

Not tried: pretrained anything (forbidden); chord-aware auxiliary loss; key-aware logit bias; reranking with a learned scorer; longer training / larger models beyond the sizes above; multiple training seeds per config; human listening tests (only the author listens).

## 5. Results

### 5.1 Task a (unconditional), all 896 songs, last checkpoint, 3 seeds x 20 pieces, unmodified `eval_basic.py`

Real data: p16 reference H1 2.118 / H4 2.617 / GS 0.962; p4 reference H1 2.118 / H4 2.617 / GS 0.887; 20.6 notes/bar.

Final model F1 (GPT-2, p16, dropout 0.3) +- s.d. over seeds:

| sampling | H1 | H4 | GS | notes/bar | malformed tokens (60 pieces) |
|---|---|---|---|---|---|
| T1.0 pure (chosen default) | 2.169 +- 0.039 | 2.600 +- 0.031 | 0.963 +- 0.005 | 29.6 | 227 |
| T1.0 + Bar-logit bias 3 | 2.105 +- 0.016 | 2.594 +- 0.050 | 0.966 +- 0.005 | 23.4 | 157 |
| T1.0 top-p 0.95 | 2.191 +- 0.018 | 2.552 +- 0.018 | 0.988 +- 0.004 | 30.8 | 36 |
| T0.9 top-p 0.9 | 2.055 +- 0.040 | 2.411 +- 0.065 | 0.995 +- 0.002 | 29.4 | 0 |
| T1.2 top-k 5 | 2.237 +- 0.032 | 2.570 +- 0.014 | 0.959 +- 0.009 | 36.9 | 0 |

Selection: pure sampling is chosen because it is closest to real on H4 and GS among the required metrics, closest on SI, and the entropy-histogram matching selects T=1.0 untruncated. Differences between pure and Bar-bias-3 are within seed noise for H4/GS; bias 3 lowers density (23.4 vs 29.6 notes/bar, real 20.6) and is closer on H1.

Ablations (same backbone/budget, p4 grid, 3 seeds x 20 pieces, pure sampling unless stated):

| variant | H1 | H4 | GS | notes/bar | note |
|---|---|---|---|---|---|
| REMI p4 (F4) | 2.122 | 2.568 | 0.919 | 27.6 | reference for this table |
| REMI p4 + Bar bias 3 | 1.999 | 2.545 | 0.891 | 22.1 | GS reaches real 0.887 |
| REMI p4 + chord tokens (F5) | 2.148 | 2.532 | 0.924 | 28.6 | no clear change |
| REMI p4 + label smoothing 0.1 (F6) | 2.012 | 2.785 | 0.721 | 16.0 | breaks free sampling: 28,634 malformed tokens |
| MIDI-like (R1f) | 1.917 | 2.141 | 0.837 | 28.1 | lower pitch entropy; the codec itself is lossless (real crops through the codec: 2.126 / 2.621 / 0.890) |
| Compound Word (R2f) | 2.099 | 2.566 | 0.825 | 19.4 | density and H4 close to real, GS lower |
| RoPE/llama-style decoder, p16 (F7) | 2.142 | 2.640 | 0.967 | 21.6 | vs F1 pure 2.169 / 2.600 / 0.963; one training seed each |
| F1 + pitch-only repetition penalty alpha 1 | 2.576 | 3.079 | 0.965 | n/a | moves H1/H4 far above real; rejected |

Dev-set validation NLL per REMI-equivalent token (45 held-out songs): REMI GPT-2 0.952, MIDI-like 1.211, Compound Word 1.275.

Structureness indicator SI (optional; MusDr scape plots, 20 pieces from seed 0, same code for all sets), short / mid / long:

| set | SI short | mid | long |
|---|---|---|---|
| real (100 crops) | 0.369 | 0.376 | 0.344 |
| F1 pure | 0.330 | 0.290 | 0.225 |
| F1 Bar-bias 3 | 0.319 | 0.271 | 0.210 |
| F1 top-p 0.95 | 0.528 | 0.517 | 0.449 |
| F1 T0.9 top-p 0.9 | 0.657 | 0.692 | 0.635 |
| F7 RoPE pure | 0.287 | 0.257 | 0.212 |

### 5.2 Task b (continuation)

Same F1 checkpoint, 8-bar prompt kept verbatim, 3 configs per prompt (pure T1.0, Bar-bias 3, T0.9 top-p 0.9), exactly 32 bars each. No objective metric is defined by the spec. Measured on the dev checkpoint: continuation density relative to the prompt's own (notes/bar, prompt -> cfg1 / cfg2 / cfg3): prompt 001 15.5 -> 17.5 / 16.0 / 23.8; 002 22.1 -> 27.0 / 21.0 / 26.7; 003 22.8 -> 32.1 / 24.7 / 45.0; the Bar-bias config stays closest to the prompt density. Best config per prompt is chosen by listening (pending).

### 5.3 Bonus (accompaniment), p4 grid, all 896 songs, last checkpoint, 10 prompts x 16 bars, 3 seeds x 4 sampling configs, unmodified `eval_bonus.py`

CA-ceiling (real piano) 0.7625; wrong-song piano CA 0.0125. Ranges are over the 4 sampling configs of the 3-seed means.

Chord given (annotation used as input):

| model | CA | OA pitch | OA vel | OA dur | OA IOI |
|---|---|---|---|---|---|
| B (T5 6+6), chord forced in decoder (F2) | 0.587-0.656 | 0.843-0.889 | 0.553-0.607 | 0.583-0.671 | 0.469-0.558 |
| A, lookahead 1, chord in melody segment (FA2) | 0.567-0.669 | 0.875-0.889 | 0.524-0.570 | 0.668-0.703 | 0.487-0.591 |
| A, L1, chord in both segments, no embedding (FA11, control) | 0.494-0.583 | 0.880-0.888 | 0.511-0.567 | 0.650-0.691 | 0.435-0.516 |
| A, L1, same + in-attention chord embedding (FA12) | 0.629-0.660 | 0.885-0.890 | 0.552-0.579 | 0.666-0.690 | 0.472-0.519 |
| A, lookahead 2, chord in melody segment (FA6) | 0.210-0.317 | 0.853-0.869 | 0.549-0.600 | 0.629-0.680 | 0.463-0.509 |
| B with chord only in the encoder (FB2) | 0.156-0.204 | 0.787-0.866 | 0.527-0.604 | 0.614-0.720 | 0.403-0.524 |
| B, p16 grid (FB8) | 0.442-0.544 | 0.877-0.889 | 0.575-0.612 | 0.686-0.722 | 0.511-0.551 |

Inference-time additions on the chord-given F2 (T0.9 top-p 0.9, 3 seeds): chord-triad logit bias +1 / +2 / +4: CA 0.748 / 0.871 / 0.921; best-of-8 selected by agreement with the conditioning chords: CA 0.912-0.925. These use the same chord labels the metric scores, so they measure selection/guidance against the evaluation labels (CA above the real-piano ceiling 0.7625 shows this) and are reported as separate labeled variants, not as the main result.

No chord file used (melody only):

| model | CA | OA pitch | OA vel | OA dur | OA IOI |
|---|---|---|---|---|---|
| Two-stage: stage-1 chord predictor (free-running chord accuracy 0.273) -> F2 | 0.183-0.231 | 0.846-0.848 | 0.575-0.622 | 0.620-0.671 | 0.476-0.528 |
| A, L1, no chord (FA1) | 0.146-0.179 | 0.847-0.860 | 0.534-0.572 | 0.642-0.675 | 0.470-0.533 |
| A, L1, chord generated as first token of the piano segment (FA7) | 0.121-0.175 | 0.844-0.859 | 0.510-0.550 | 0.661-0.694 | 0.444-0.535 |
| B, no chord (FB1) | 0.131-0.148 | 0.829-0.862 | 0.549-0.622 | 0.636-0.729 | 0.427-0.505 |
| B, chord generated by the decoder (FB5) | 0.127-0.163 | n/a | n/a | n/a | n/a |
| Two-stage with best-of-8 or chord-triad bias (+1/+2/+4) on the stage-1 chords | 0.233-0.256 | 0.820-0.850 | 0.556-0.623 | 0.623-0.676 | 0.483-0.522 |

Other bonus ablations: T5 relative buckets 128 / max distance 1024 (FB9) CA 0.50-0.61 and T5 4+4 d256 (FB10) CA 0.53-0.66, both in the band of F2 (no resolved effect with one training seed); dev ablations (5% of songs held out, best checkpoint, chord forced in the piano segment, p4): A initialized from our unconditional checkpoint CA 0.500-0.575, loss weight 0.2 on melody tokens CA 0.550-0.637; p16 grid for B (FB8) given above.

## 6. Known caveats (our own audit; please check for more)

1. Each model is trained once; "+-" values are generation-seed s.d., not training-seed s.d. Differences of ~0.03 in H1/H4 and ~0.05 in CA between models are not resolved.
2. GS reference depends on the onset grid (real GS 0.887 at p4, 0.962 at p16); the TA script's tokenization ignores the grid for onsets, which we worked around; cross-paper GS numbers may not be comparable.
3. CA in the TA script compares the estimated chord of the generated bar with the *annotation*; the real piano scores only 0.7625, so CA is bounded and bars labelled N/dim/aug/sus can never match. With 10 prompts x 16 bars = 160 bars per seed, one bar is 0.006 CA; seed-to-seed s.d. is around 0.03-0.05.
4. OA of duration and IOI is capped by the token grid (round-trip OA_IOI 0.745 at p4, 0.935 at p16); our p4 numbers for these two are therefore partly a representation ceiling.
5. Chord-given vs melody-only results answer different questions; the permissibility of using the chord file as input is unconfirmed. The melody-only headline is the two-stage pipeline (CA ~0.2), far below the chord-given numbers (~0.6-0.9 with guidance).
6. The Task a metrics are distribution-matching statistics of pitch-class entropy and onset-pattern similarity; they do not measure musical quality. We have no listening-test results beyond the author's own, and the SI comparison uses only 20 pieces per set.
7. Truncated sampling (top-p 0.9) scores GS 0.995 and SI 0.66 vs real 0.887-0.962 and 0.37: strongly repetitive output; entropy-based checks agree that it is far from the data distribution.
8. Best-of-N and chord-triad bias use evaluation-time chord labels; they are metric-guided and are not claimed as model quality.

## 7. Reproducibility

Code, configs, per-run JSONL result rows (config, checkpoint hash, code hashes, seeds), and a clean-environment test of `infer.py` (exact metric reproduction from the released checkpoints) are in the project repository. Software: PyTorch 2.6, Transformers 5.x, pretty_midi, FluidSynth (bundled soundfont) for audio rendering.
