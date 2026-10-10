# WORKLOG — CommE5070 HW2

Deadline 2026-10-26 23:59. Log every experiment here: exact config (token representation, architecture, hyperparams, inference settings), what ran, and the result — not just the finding. See `TODO.md` for the task backlog and `project notes` for GPU/repo setup.

## 2026-10-06 — Setup
- Assignment released (Week 5 announcement). Bootstrapped `project notes`, `RESEARCH_PROMPT.md`, `TODO.md` on Mac. No experiments yet — research pass pending before any training starts.

## 2026-10-06 — Research review + locked plan (round 1: `reponse_round1/{cladue,gemini,kimi,qwen}.md`)

Source reliability (interpretation): research response A is the most specific and sourced (C&E / MusDr / MidiTok details). research response B has non-GitHub/search-redirect URLs and an unverified `pip install musdr` claim — not trusted. research response D recommends CP word + GPT2 via MidiTok, which contradicts MidiTok's own docs (research response C) discouraging CPWord for small-model generation — not adopted. All token-length figures for 32 bars disagree (0.25k–6k) — must be MEASURED on our data before fixing context length.

Facts read directly from the spec/scripts (not from research):
- `eval_basic.py` finds `Bar`, `Pos_*`, `Pitch_*` by name in `tok2idx.json`; extra token types (e.g. `Chord_*`) are ignored by H/GS. So the TA REMI-style tokenizer feeds the eval with no changes; MidiTok would need eval adaptation.
- TA `tokenization.py` merges all tracks (no track identity) and emits no chord tokens (chord file is parsed but unused). Basic tasks only need the merge; bonus needs MELODY/PIANO separated.
- `eval_bonus.py`: generated MIDI needs >=2 tracks, picks the track named `PIANO` (fallback index 1); GT chords/beats come from `prompts/bonus/<song>/{chord_midi,beat_midi}.txt` + `piano.mid`; CA uses a 24-triad bass-weighted estimator, and reports `CA-ceiling` (same estimator on the GT piano) as the upper reference.
- eval H4/GS depend on the token grid, so the "Real Data" row must be computed with the SAME `pos_per_beat` as generated pieces.

Locked decisions:
1. Tokenization: extend the TA `tokenization.py` (REMI-style Bar/Pos/Tempo/Pitch/Vel/Dur), no MidiTok. Final `pos_per_beat` (4 vs 16) decided after measuring sequence length + real-data eval at both (day 1).
2. Model: HF `GPT2LMHeadModel`, random init, ~8L x 512d x 8h (~25-30M params), context sized from measured 32-bar length; SDPA/bf16. Train on bar-aligned random crops, +-5 semitone transposition, hold out a small val split of the 896 for val NLL. Checkpoint chosen by val NLL AND by metrics across several saved checkpoints (C&E/Jazz-Transformer: lowest loss != best sounding).
3. Task a inference configs: sampling-side (checkpoint-free), candidates top-p 0.95 T1.0 / top-k 5 T1.2 (Pop Music Transformer default) / top-p 0.9 T0.9, plus a checkpoint axis; full sweep logged, three reported. Chord-token variant (from `chord_midi.txt`) is the "w/ Chord" ablation if time. Label-smoothing = extra training run, only if time.
4. Task b: same checkpoint, bar-aligned prompt prefix, generate until 32 bars total.
5. Bonus: Option A first (bar-level interleave, melody bar i+1 placed before piano bar i for one-bar lookahead, C&E-style track-switch inference loop); Option B (T5 from scratch) only as a stretch A-vs-B ablation.
6. SI: optional; clone MusDr (not on PyPI), run on 32-bar pieces with `-j`, only after H4/GS pipeline is done.

Open items to check on the server: do `prompts/bonus/<song>/` folders contain chord files usable as conditioning (spec says "conditional chord sequence")? Do POP909 MIDI tracks have names MELODY/BRIDGE/PIANO? MusDr installs on which Python?

## 2026-10-06 — Server setup + data measurements (gsm-gpu2, /home/jtan/comme5070_pa2/)
Env: conda `comme5070_pa2` (py3.10), torch 2.6.0+cu124 (driver 550 = CUDA 12.4; default pip wheel was cu130 and unusable), transformers 5.18.0, pretty_midi, librosa. MusDr cloned (not on PyPI) to `MusDr/`; `eval_basic.py` needs `PYTHONPATH=/home/jtan/comme5070_pa2`. GPU 1 and GPU 2 passed a 4096x4096 matmul smoke test (H200 NVL). GPUs 0/3 untouched (known broken).
Data: 896 training songs; `prompts/basic-b/{001,002,003}/{<id>.mid,beat_midi.txt,chord_midi.txt}`; `prompts/bonus/{001..010}/{melody.mid,piano.mid,beat_midi.txt,chord_midi.txt}`. Training MIDI tracks are named MELODY/BRIDGE/PIANO (all program 0); basic-b prompts contain BRIDGE+PIANO tracks; bonus melody.mid has MELODY only.
Tokenization (TA `tokenization.py`, unmodified, all 896 songs OK, 0 errors), pos_per_beat in {4, 16}: vocab 257 / 545. Sequence length is IDENTICAL for both (one Pos token per note, so the grid only changes vocab): song mean 6833 (max 17037) tokens, 81.8 bars mean, 86.3 tokens/bar mean (p95 117.9). First 32 bars: mean 2594, p50 2556, p95 3622, max 9114 (n=891 songs with >=33 bars) -> context must be 4096. Earlier research estimates (1.6k-6k) were off; this is the measured figure.
Real-data eval_basic (full songs, 896): p=4: H1 2.1341, H4 2.6296, GS 0.9116. p=16: H1 2.1341, H4 2.6296, GS 0.9524. (H unchanged by grid; GS depends on grid.) Config: TA tokenizer defaults, eval_basic.py as provided, GS uses random.sample of up to 1000 bar pairs (unseeded).
Open: still need real-data row on 32-bar crops (generated pieces are 32 bars; GS over a full song is not comparable). pos_per_beat decision pending.

## 2026-10-06 — Real-data baseline on 32-bar crops; pos_per_beat decision
Config: `src/make_real_crops.py` (non-overlapping bar-aligned 32-bar windows of every tokenized training song, <EOS> dropped) -> 1864 crops; `eval_basic.py` as provided (PYTHONPATH=project root), TA tokenizer defaults.
| pos_per_beat | vocab | H1 | H4 | GS | std H4 | std GS | n |
|---|---|---|---|---|---|---|---|
| 4  | 257 | 2.1183 | 2.6168 | 0.9299 | 0.125 | 0.051 | 1864 |
| 16 | 545 | 2.1183 | 2.6168 | 0.9617 | 0.125 | 0.026 | 1864 |
(Full-song numbers above in the previous entry.) With 32 bars, all 496 bar pairs are used, so GS is deterministic here.
Decision: pos_per_beat = 4 (16th-note grid, vocab 257). Reasons (interpretation): same sequence length, smaller vocab/easier softmax over ~2.6k-token pieces, GS has ~2x the per-piece spread so configs are easier to tell apart. Cost: no sub-16th timing/triplets. Real-data reference row for the report = the p=4 crop row (H1 2.118 / H4 2.617 / GS 0.930).

## 2026-10-06 — CORRECTION: TA tokenization.py ignores --pos_per_beat for onsets
Found when `train_uncond.py` hit `KeyError: 'Pos_56'` on the p=4 sequences. Cause (read from `scripts/tokenization.py`): `process_pop909_dataset` calls `time_to_grid(note.start, beat_times, beat_info)` without `POS_PER_BEAT`, so onsets are always on the 16-per-beat grid; only `Dur_*` and the vocab size follow the flag. Measured: the p=4 and p=16 sequence files were identical (max Pos id 184 in both); at p=4, 737,184 of 6,122,322 tokens (12%) were Pos tokens outside the 32-entry Pos vocab, in all 896 songs, and `eval_basic.py` silently dropped them (`if tok in tok2idx`). At p=16 only 14 Pos tokens are OOV (songs 149, 275; bars with >8 beats).
=> The p=4 rows (and the "pos_per_beat = 4" decision, and the "identical lengths" explanation) in the two entries above are INVALID. The p=16 rows are valid apart from the 14 OOV tokens being dropped by eval. (Interpretation of the stray p=16 lengths: they are the true lengths on a 16/beat grid.)
Fix: `src/tokenize_hw2.py` (wrapper; TA script untouched) sets the `time_to_grid` default grid and drops whole events whose Pos is outside the vocab. Re-running real-data baselines at p in {4, 8, 16} (full songs + 32-bar crops); pos_per_beat to be re-decided from those results.

## 2026-10-06 — Corrected real-data baselines + pos_per_beat decision
Config: `src/tokenize_hw2.py` (TA tokenizer + grid fix + OOV-event drop; 14 events dropped in total, same at every p), `src/make_real_crops.py` (non-overlapping 32-bar windows, 1864 crops), unmodified `scripts/eval_basic.py`, 896 training songs. Files on server: `/home/jtan/comme5070_pa2/work/{tok2idx,seqs,crops32}_p{4,8,16}.json`, `real_{seqs,crops32}_p*.csv`.
| pos_per_beat | vocab | 32-bar crops: H1 / H4 / GS | full songs: H1 / H4 / GS |
|---|---|---|---|
| 4  | 257 | 2.1183 / 2.6168 / 0.8872 | 2.1341 / 2.6296 / 0.8683 |
| 8  | 353 | 2.1183 / 2.6168 / 0.9289 | 2.1341 / 2.6296 / 0.9132 |
| 16 | 545 | 2.1183 / 2.6168 / 0.9617 | 2.1341 / 2.6296 / 0.9523 |
(H1/H4 do not depend on the grid; GS does, and rises with finer grids. Full-song GS uses random.sample of <=1000 pairs, unseeded; crop GS is deterministic.)
Decision: main model pos_per_beat = 4 (16th grid, vocab 257): smallest vocab, coarse grid is easier to model. Real-data reference row for the report = p=4 crops row (H1 2.118 / H4 2.617 / GS 0.887). p=16 (TA default grid) kept as an optional representation ablation if GPU time allows. Not yet measured: whether the corrected sequence lengths differ from the earlier (invalid) stats -- re-measured in the next entry.

## 2026-10-06 — Run u1: unconditional GPT-2, p=4 (launched)
Smoke test (100 steps, GPU 1): OK, 27.4M params, ~0.09 s/step at 8x4096 tokens, val NLL 2.654 (step 50) -> 2.583 (step 100). Corrected p=4 32-bar token lengths: mean 2593 / p50 2556 / p95 3622 / max 9114 (unchanged by the grid fix).
Run u1 config (`src/train_uncond.py`, tmux `pa2:u1`, GPU 1, out `work/runs/u1_p4_8L512/`, log `work/runs_u1.log`): tokens = work/{tok2idx,seqs}_p4.json; GPT2LMHeadModel random init, 8 layers x 512 x 8 heads, ctx 4096, dropout 0.1, sdpa, bf16 autocast; 851 train / 45 val songs (val split seed 0, 5%); bar-aligned random crops (<BOS> prepended; tempo token inserted at Pos_0 for mid-song crops), transposition uniform in [-5,+6] semitones (range-clipped to Pitch 21..108); batch 8 crops (<=4096 tokens), 40000 steps, AdamW lr 3e-4 (betas 0.9/0.95, wd 0.01 on matrices, clip 1.0), warmup 1000, cosine to 5%; label smoothing 0; eval val NLL every 500 steps, ckpt every 2500 + best.pt + last.pt.

## 2026-10-06 — Tooling while u1 trains (no GPU used)
- `src/generate.py` (modes uncond/cont; HF generate with temperature/top-k/top-p; stops at the (n_bars+1)-th Bar; pieces not reaching n_bars inside ctx 4096 are discarded + resampled, reject rate printed; writes .mid + eval_basic-format tokens json + config json), `src/render_wav.py` (pretty_midi.fluidsynth with its bundled TimGM6mb soundfont). Not yet run: generation needs the GPU, which u1 occupies (one experiment at a time).
- WAV rendering env check: conda-forge `fluidsynth` + pip `pyfluidsynth soundfile` installed into `comme5070_pa2`; a 4-note test rendered to audio (66150 samples).
- Task b prompts tokenized with the same wrapper (`work/prompts_basicb_p4.json`): each has exactly 8 Bar tokens (508/720/740 tokens), 0 dropped events, vocab identical to training vocab.
- u1 early progress (not a result): val NLL 2.274 @500, 1.669 @1500.

## 2026-10-06 — GPU policy update + parallel ablation runs u2/u3/u4
User instruction: all 4 GPUs may be used (supersedes the one-GPU-at-a-time default for these independent training runs). Reconfirmed 2026-10-06 ~23:58 JST: GPU 0, 2, 3 each passed an 8192x8192 matmul x20 test under `timeout 90` (1.3 / 1.0 / 1.5 s), so the earlier "GPU 0/3 broken" status did not reproduce in this test; still watching run logs for stalls. Server has 128 CPUs; each process capped at OMP/MKL threads = 4.
All runs: same code (`src/train_uncond.py`), same val split (seed 0, 45 songs), same defaults as u1 (ctx 4096, batch 8, 40000 steps, lr 3e-4 cosine, warmup 1000, wd 0.01, dropout 0.1, transposition -5..+6, eval every 500, ckpt every 2500) except the listed change; tmux windows `pa2:u2..u4`, logs `work/runs_<name>.log`, out `work/runs/<name>/`.
| run | GPU | change vs u1 | out dir |
|---|---|---|---|
| u1 | 1 | baseline: p=4, 8L x 512, ls 0 | u1_p4_8L512 |
| u2 | 2 | pos_per_beat 16 (tok2idx/seqs _p16, vocab 545) | u2_p16_8L512 |
| u3 | 3 | label_smoothing 0.1 (train loss smoothed; logged val_nll is unsmoothed) | u3_p4_8L512_ls0.1 |
| u4 | 0 | n_layer 12 | u4_p4_12L512 |
Note for comparison: val NLL is not comparable across u2 (different vocab/grid) and the others; compare via generated-sample metrics (H1/H4/GS vs the real row of the same grid).
u1 progress (not a result): val NLL 1.083 @5000 steps (401 s).

## 2026-10-07 — Round-1 training curves (runs still going; measured from `work/runs/*/log.jsonl`)
Val NLL = per-token NLL on 45 held-out songs (5%), unsmoothed. All runs overfit: val NLL bottoms out early, then rises while train loss keeps falling.
| run | min val NLL @ step | last logged val NLL @ step | train loss @ last |
|---|---|---|---|
| u1 p4 8L512 | 0.9919 @9000 | 1.5509 @25000 | 0.515 |
| u2 p16 8L512 | 1.2655 @9000 (vocab 545, not comparable) | 1.8519 @21000 | 0.509 |
| u3 p4 8L512 ls0.1 | 1.0607 @9000 | 1.3396 @21000 | 1.301 (smoothed) |
| u4 p4 12L512 | 0.9972 @7000 | 1.3573 @14000 | 0.573 |
Interpretation (not yet tested): 851 training songs are memorized after ~10k steps at this size/dropout; more regularization (dropout, weight decay), fewer steps, or smaller model are candidate next-round levers. Runs are left to finish so the 2500-step checkpoints (step_*.pt) give a checkpoint-choice axis for generation metrics; `best.pt` per run = lowest val NLL.
Decision (user, 2026-10-07): bonus work waits until the round-1 results on the 4 GPUs are in; bonus stays in TODO.
Prepared-but-unrun bonus code (syntax-checked only): `src/bonus_data.py` (ran OK on CPU: 896 train songs, 10 prompts x 16 bars, vocab 284; 4.1 melody / 12.9 piano notes per bar; chord labels N/unsupported 10.5%, major 52.9%, minor 36.6%), `src/bonus_lib.py`, `src/train_bonus.py` (arch A GPT-2 interleaved / arch B T5, lookahead, chord, melody_w, init_from flags). Not yet written: bonus generation + MIDI writer on the prompt beat grid. `RESEARCH_PROMPT_R2.md` written for the next deep-research pass.

## 2026-10-07 — Round-1 generation sweep: method fix (rejection sampling dropped)
All four round-1 trainings finished (u1 00:46, u3 00:51, u2 00:52, u4 01:17 JST). `generate.py` verified end to end on u1 best.pt: 4 pieces -> MIDI (32 bars, 0 malformed tokens) -> unmodified eval_basic (H1 2.166 / H4 2.547 / GS 0.941, n=4, indicative only) -> WAV (FluidSynth).
Problem found: the first version rejected pieces that did not reach 32 bars inside the 4096-token context. Measured on u1 (T1.0, top-p 0.95, n=8 each): best.pt (val-best, step 9000) reject rate 0.68 (16 ctx-limit, 1 early-EOS of 25 samples); step_20000 and last.pt 0.20. In the first sweep attempt some cells (e.g. u? best.pt, pure T=1.0) could not produce 20 pieces in 400 tries (0/20 in one case). Rejection also selects shorter/sparser pieces and so biases H/GS, so those partial rows were discarded (files deleted, nothing logged as a result).
Fix (`src/generate.py`): sliding-window generation. Context is trimmed at a Bar boundary to <= ctx-1024 tokens (<BOS> + suffix, tempo in effect re-inserted at Pos_0 exactly like the training crops) and generation continues in 1024-token chunks until the piece has 32 bars; <PAD>/<BOS>/<EOS> are never sampled. No sample is rejected; `n_slid_window` (pieces that needed the slide) is logged. Check: u1 best.pt, T1.0 pure sampling, n=6: all 6 pieces complete, 3456-5743 tokens/piece (real 32-bar mean 2593), 11 malformed tokens total.
Round-1 sweep config (`src/sweep_uncond.py`, tmux windows s1-s4, GPUs 1/2/3/0 for u1/u2/u3/u4; results `work/gen/results_<run>.jsonl`, one json row per cell with full config, ckpt sha, code sha, train args): checkpoints {best, step_20000, last} x sampling {T1.0 top-p0.95, T1.0 pure, T0.9 top-p0.9, T1.2 top-k5}, n=20 pieces/cell, seed 0, 32 bars, scored with unmodified scripts/eval_basic.py (H1/H4/GS) per run's own grid. Reference rows (real, 32-bar crops): p=4 H1 2.118 / H4 2.617 / GS 0.887; p=16 H1 2.118 / H4 2.617 / GS 0.962.

## 2026-10-07 — Round-1 generation results (48 cells; full table `results/round1_table.md`, raw rows `results/round1_u*.jsonl` with full config + hashes)
Config: 20 pieces/cell, 32 bars, seed 0, sliding-window generation (no rejection), unmodified `scripts/eval_basic.py`; train config = round-1 runs u1-u4 (WORKLOG 2026-10-06/07). Reference (real, 32-bar crops, same grid): p=4 H1 2.118 / H4 2.617 / GS 0.887; p=16 H1 2.118 / H4 2.617 / GS 0.962. Real notes/32 bars = 20.6/bar x 32 = 659.
Method notes: two cells (u1 best T0.9_p0.9; u4 best T1.0_p0.95) first failed because a degenerate sample produced a >3k-token bar with no Bar token (cannot slide the window); fix = restart only that sample from the prompt (counted in `n_restarts`, rows from the fixed code carry the field; the first 46 rows predate the field) and a 48k new-token budget. Same code otherwise.
Measured findings:
- Pure sampling (T1.0, no truncation) and T1.2 top-k5 give H4 closest to real (e.g. u1 last pure 2.619 vs 2.617; u4 last pure 2.543); conservative sampling (T0.9 top-p 0.9) pushes H4 down (2.07-2.51) and GS up to 0.95-0.99, above the real 0.887 (u2 vs its own real 0.962 likewise).
- Density is NOT matched: generated pieces have 27-42 notes/bar vs 20.6 real (u1 step_20000 T1.0_p0.95: 29.3; u1 best T0.9_p0.9: 42.0, 5408 tokens/piece, all 20 pieces needed the window). Median Dur token 1 vs real 2 (16th-grid steps). Notes/bar rises with bar index in generated pieces (u1 step_20000: 25.2, 30.1, 30.8, 31.2 per quarter; real 18.4, 20.1, 21.7, 22.0). So H4/GS near real do not imply real-like density; metrics alone are insufficient (listening still pending, I cannot listen).
- u3 (label smoothing 0.1) at T1.0 (top-p 0.95 or pure) emits thousands of malformed tokens per 20 pieces (u3 best T1.0_p0.95: 3721; pure: 7008) and is sparser (16.8 notes/bar, mean Dur 4.1); its H4/GS in those cells (2.86-2.93 / 0.72-0.76) describe malformed token streams, not clean music. With T0.9 top-p 0.9 or T1.2 top-k5 malformed counts drop (3 to 762) and metrics look like the other runs.
- Checkpoint axis (malformed tokens per 20 pieces, best / step_20000 / last): u1 p0.95 5/10/26, pure 53/94/62, k5 0/23/29; u2 p0.95 0/25/32, pure 79/104/216, k5 0/8/79; u4 p0.95 3/17/37, pure 133/63/77, k5 0/40/35. The least-overfit checkpoint (best, step ~7000-9000) has fewer malformed tokens in most cells but is denser/drifts more (see density bullet); the best checkpoint is not better on H4/GS either. No consistent winner.
- Run comparison (interpretation, n=20 per cell, one seed, no confidence intervals yet): no run/ckpt/sampling cell matches the real row on H1, H4 and GS simultaneously; closest on both H4 and GS is u4 (12L) last T1.0_pure (H4 2.543 / GS 0.937 vs real 2.617 / 0.887) and u1 step_20000 T1.0_pure (2.644 / 0.898). Not yet tested: seed variance, SI, listening.
Round 2 launched (tmux v1-v4, GPUs 1/2/3/0), single-factor changes vs u1 (all else = u1 defaults): v1 steps 12000 (cosine finishes before the overfit regime), v2 dropout 0.3 + steps 20000, v3 6L x 384 x 6h + steps 20000, v4 weight decay 0.1 + steps 20000. Run configs now also save code/data sha256 + torch/transformers versions. Hypothesis behind round 2 (untested): more regularization/fewer steps lowers val NLL and may reduce the density inflation.

## 2026-10-07 — Round-2 training results (val NLL on the same 45 held-out songs, p=4 vocab 257; comparable to u1)
| run | change vs u1 (u1: 8L x 512, drop 0.1, wd 0.01, 40k steps, min val 0.9919 @9000, last 1.5509 @25000) | min val NLL @ step | last val NLL @ step |
|---|---|---|---|
| v1 | steps 12000 | 0.9806 @12000 | 0.9806 @12000 |
| v2 | dropout 0.3, steps 20000 | 0.9523 @19500 | 0.9538 @20000 |
| v3 | 6L x 384 x 6h, steps 20000 | 0.9747 @19500 | 0.9763 @20000 |
| v4 | weight decay 0.1, steps 20000 | 0.9877 @9000 | 1.1864 @20000 |
Observation: dropout 0.3 has the lowest val NLL and no overfitting rise by 20k steps (curve still decreasing slowly: 0.961 @15000, 0.955 @17000); weight decay 0.1 alone still overfits (val rises after step 9000). Single seed each (seed 0); differences of ~0.01-0.03 NLL are not separated from seed noise yet.
Launched (2026-10-07): sweeps s5 (GPU1: v2 best+last, v1 best) and s6 (GPU2: v3 best+last, v4 best), same 4 sampling configs / n=20 / seed 0 as round 1 (results_v*.jsonl); trainings v5 (GPU3: dropout 0.3, steps 40000) and v6 (GPU0: dropout 0.5, steps 40000), otherwise u1 defaults.

## 2026-10-07 — Round-2 generation (v1-v4), exposure-bias diagnostic, typed-sampling sweeps
Faithfulness checks: (1) transformers 5.18.0 has top_k/top_p/temperature defaults = None (checked on GenerationConfig and the model's generation_config), so "pure" sampling in all sweeps is untruncated, labels in round 1/2 are accurate. (2) Re-running v2 best with the typed-sampling code path off (B0_pure, seed 0) reproduced the earlier round-2 cell exactly (H1 2.189 / H4 2.576 / GS 0.916 / 30 malformed): same seed + same code => same pieces.
Round-2 sweep (v1 best only for its 12k schedule; v2/v3 best+last; v4 best; 4 sampling configs, n=20, seed 0; rows in `work/gen/results_v*.jsonl`): density stays inflated for every run: 23.5-34.5 notes/bar vs real 20.6 (v3 last pure 23.5 lowest; v2 29.5-31.6; v4 27.5-34.5). Lower val NLL (v2 0.952) did not fix density. Closest to real on H4/GS: v3 best T1.2_k5 (H4 2.627 / GS 0.886), v3 last pure (2.609 / 0.897), v1 best pure (2.668 / 0.903), v2 last pure (2.650 / 0.916).
Exposure-bias diagnostic (`src/density_diag.py`; v2 best, T1.0 pure, n=1 per prompt; prompts = first 8 bars of 101 real 32-bar crops of the 45 validation songs, `work/val_crops32_p4.json`; generation continues to 32 bars): notes/bar of continuation vs the real continuation of the SAME crop: bars 8-11 gen 22.7 vs real 19.6 (x1.16); bars 12-15 25.8 vs 20.0 (x1.29); bars 16-31 28.7 vs 20.9 (x1.37). Bars 0-7 identical by construction (18.1 vs 18.2). => density inflates immediately and grows with the model's own history (exposure bias), also in the continuation task.
Typed sampling (new in `src/generate.py`: per-token-type temperature t_struct (Pos-vs-Bar choice) / t_pitch / t_vel / t_dur and `bar_bias` added to the Bar logit at struct steps; unit-tested on CPU: exact scaling, top-k/top-p). Sweep on v2 best (n=20, seed 0, 32 bars, sliding window; configs in `results/sampling_typed{A,B}.json`, rows `work/gen_typed/`); real ref H1 2.118 / H4 2.617 / GS 0.887, 20.6 notes/bar, 2593 tokens:
| sampling (T=1.0 unless noted) | H1 | H4 | GS | notes/bar (quarters) | tokens | malformed |
|---|---|---|---|---|---|---|
| pure (B0) | 2.189 | 2.576 | 0.916 | 29.8 (25.8/29.9/30.4/33.0) | 3845 | 30 |
| bar_bias 1 | 2.102 | 2.556 | 0.910 | 27.9 | 3615 | 34 |
| bar_bias 2 | 2.156 | 2.633 | 0.890 | 24.9 | 3228 | 38 |
| bar_bias 3 | 2.080 | 2.582 | 0.879 | 22.1 (19.6/23.6/21.7/23.5) | 2865 | 30 |
| t_struct 0.8 | 2.234 | 2.701 | 0.945 | 29.6 | 3822 | 37 |
| t_struct 0.6 | 2.260 | 2.647 | 0.975 | 32.2 | 4156 | 17 |
| t_dur 0.7 | 2.244 | 2.568 | 0.926 | 37.0 | 4774 | 13 |
| t_dur 1.3 | 2.056 | 2.579 | 0.867 | 22.2 | 2883 | 299 |
| t_pitch 0.8 | 2.050 | 2.458 | 0.892 | 34.7 | 4476 | 40 |
Reading (interpretation): the Bar-logit bias is a monotone density knob (29.8 -> 22.1 notes/bar from bias 0 to 3); bias 3 matches real density, GS (0.879 vs 0.887) and token count best, with H4 2.582. Lower structure temperature does not reduce density; lower duration temperature increases it; t_dur 1.3 lowers density but gives 299 malformed tokens. Drift across quarters persists even at bias 3 (19.6 -> 23.5). Single seed; next: bias 4-6 (t3, GPU1) and seeds 1,2 for pure vs bias 3 (t4, GPU2). Training v5 (drop 0.3, 40k) and v6 (drop 0.5, 40k) still running on GPUs 3/0.

## 2026-10-07 — Round-2 long runs, bias/seed results
Training (same val split/vocab as u1, val NLL): v5 (dropout 0.3, 40k steps) min 0.9610 @16500, last 1.0439 @40000 (overfits again after ~16k); v6 (dropout 0.5, 40k) min 1.0358 @29000, last 1.0585. Neither beats v2 (dropout 0.3, 20k steps, min 0.9523 @19500). Current best checkpoint by val NLL: `work/runs/v2_drop0.3/best.pt` (8L x 512 x 8h, dropout 0.3, p=4, 20k-step cosine schedule, step 19500).
Typed sampling on v2 best, n=20 per cell, 32 bars, sliding window, real ref H1 2.118 / H4 2.617 / GS 0.887, 20.6 notes/bar, 2593 tok (configs `results/sampling_typed{C,D}.json`):
| sampling | seed | H1 | H4 | GS | notes/bar | tokens | malformed |
|---|---|---|---|---|---|---|---|
| bias 4 | 0 | 2.019 | 2.592 | 0.875 | 18.6 | 2420 | 38 |
| bias 5 | 0 | 1.698 | 2.226 | 0.913 | 12.2 | 1595 | 18 |
| bias 6 | 0 | 1.525 | 1.965 | 0.944 | 7.6 | 1011 | 35 |
| pure | 0 / 1 / 2 | 2.189 / 2.182 / 2.066 | 2.576 / 2.634 / 2.498 | 0.916 / 0.930 / 0.921 | 29.8 / 30.3 / 29.1 | 3845 / 3912 / 3761 | 30 / 19 / 35 |
| bias 3 | 0 / 1 / 2 | 2.080 / 2.027 / 2.029 | 2.582 / 2.530 / 2.588 | 0.879 / 0.890 / 0.896 | 22.1 / 20.5 / 22.6 | 2865 / 2665 / 2931 | 30 / 36 / 39 |
3-seed means (n=20 pieces each): pure H1 2.146, H4 2.569 (sd 0.068), GS 0.922 (sd 0.007), 29.7 notes/bar; bias 3 H1 2.045, H4 2.567 (sd 0.032), GS 0.888 (sd 0.009), 21.7 notes/bar. Real: H1 2.118, H4 2.617, GS 0.887, 20.6 notes/bar. Reading: bias 3 matches real GS and density; H4 is similar for both and ~0.05 below real; H1 is lower than real with bias (2.045 vs 2.118), above for pure only at 2 of 3 seeds. Bias >=5 makes the music too sparse (notes/bar 12 and 7.6, H4 collapses to 2.2 / 2.0). Not yet done: SI, listening (WAV samples in `listen/` on the Mac: v2best_{pure,bias3}_{0,1,2}.wav).

## 2026-10-07 — Bonus round 1 launched (melody -> piano)
Data (`src/bonus_data.py`, p=4 grid fix applied, run on CPU): 896 training songs, MELODY and PIANO tracks selected by track name (BRIDGE not used), per-bar token segments; 10 bonus prompts x 16 bars (eval_bonus bar count = 16 for all); vocab 284 (= p4 vocab 257 + <M>, <P>, Chord_0..23, Chord_N); 4.1 melody / 12.9 piano notes per bar (16.4 / 51.8 tokens); chord labels (eval_bonus definition: dominant chord of the bar from chord_midi.txt; 24 triads) N/unsupported 10.5%, major 52.9%, minor 36.6%.
Positive controls (`src/check_bonus_lib.py`, CPU, real data): schedule order for lookahead 0/1; weight-1 tokens == piano tokens + one terminator per piano segment; each <P> segment content equals the bar's piano tokens; melody bars present before P_k == min(k+L, n-1)+1; arch B layout (Bar counts match); transposition shifts pitch ids by k and rotates chord roots keeping major/minor. All passed (after fixing one over-strict assertion in the check itself: a <P> tag directly after a piano segment is the terminator and correctly has loss weight 1).
Smoke tests (100 steps, bs16): arch A 0.07 s/step, val NLL 2.49 -> 2.34; arch B (T5 6+6) 0.29 s/step, val NLL 2.59 -> 2.45.
Launched (tmux bA1,bB1,bA2,bB2 on GPUs 1,2,3,0; `src/train_bonus.py`; common: data work/bonus_p4/bonus_train.json, 20000 steps, batch 16, lr 3e-4 cosine, warmup 1000, wd 0.01, dropout 0.3 (carried over from the unconditional result, not re-tuned for bonus), transposition -5..+6 incl. chord roots, crops of 16-24 bars, val = 45 held-out songs (same seed-0 split, non-overlapping 16-bar windows), eval every 500, ckpt every 2500):
| run | arch | chord token in conditioning | lookahead | size | out dir |
|---|---|---|---|---|---|
| bA1 | A GPT-2 interleaved (ctx 2048) | no | 1 | 8L x 512 x 8h | bA1_A_L1_nochord |
| bB1 | B T5 enc-dec (enc ctx 1024, dec ctx 2048) | no | - | 6+6 layers, d512, ff2048, 8h | bB1_B_6L_nochord |
| bA2 | A | yes | 1 | 8L x 512 | bA2_A_L1_chord |
| bB2 | B | yes | - | 6+6 | bB2_B_6L_chord |
Loss: A = piano events + terminator tags (melody weight 0); B = all decoder tokens (piano). Val NLL is per piano-side token, so A and B val NLL are not directly comparable; compare through eval_bonus (CA, OA) once `gen_bonus.py` exists. Whether chord input is allowed is open (chord_midi.txt sits in the prompt folders; the spec's CA is vs "the conditional chord sequence"): the chord runs are labeled ablations until confirmed.

## 2026-10-07 — Bonus: generation pipeline + round-trip ceilings (positive control)
`src/gen_bonus.py`: arch A = segment-injection loop (training-crop layout, sliding window by re-basing the crop at a later bar when the 2048 context would overflow; piano segment ends when the model emits a tag), arch B = encoder-decoder generate (stop at (n+1)-th Bar or EOS); piano tokens -> notes via the prompt's own beat grid (`beat_midi.txt`, interpolated in beat-index space, so seconds match what eval_bonus uses); output MIDI = original MELODY notes from the prompt (copied) + generated PIANO track; scored with the unmodified `scripts/eval_bonus.py`; one json row per run with config + ckpt/code hashes.
Pipeline check (A, early best.pt of bA1, n=10 prompts, NOT a result): runs end to end, 0 truncated bars, eval_bonus runs (CA 0.144 at that early checkpoint).
Round-trip control (`src/roundtrip_bonus.py`: real piano.mid of the 10 prompts -> bonus_data bar tokens -> gen_bonus beat-grid writer -> eval_bonus; 0 malformed tokens; note counts identical to the real track in all 10): what a PERFECT model could score through our tokenization + writer (pooled over 160 bars):
| pos_per_beat | vocab | CA | CA-ceiling (real piano) | OA pitch-class | OA velocity | OA duration | OA IOI |
|---|---|---|---|---|---|---|---|
| 4 | 284 | 0.7562 | 0.7625 | 0.9800 | 0.9733 | 0.8701 | 0.7452 |
| 8 | 380 | 0.7688 | 0.7625 | 0.9800 | 0.9733 | 0.8979 | 0.8160 |
| 16 | 572 | 0.7688 | 0.7625 | 0.9940 | 0.9906 | 0.9586 | 0.9352 |
Reading: the writer/grid mapping is faithful (CA matches the ceiling within 0.006 at p=4, within 0.007 above it at p=8/16); the 16th-note grid (p=4) caps OA_dur at 0.870 and OA_IOI at 0.745 for ANY model, while p=16 allows 0.959 / 0.935. Sequence length is unchanged by the grid (one Pos per note), only vocab grows. => bonus runs at p=16 (and p=8) are required to be fair on OA; bonus data built at `work/bonus_p8`, `work/bonus_p16` (vocab 380 / 572).
Bonus round-1 runs (bA1, bB1, bA2, bB2) use p=4; a p=16 repeat of the better arch is queued for the first free GPU.
Queued (auto-start when the p=4 A runs finish): bA3 = arch A, lookahead 1, no chord, p=16 data (`work/bonus_p16`, vocab 572) on GPU 1; bB3 = arch B 6+6, no chord, p=16 on GPU 3. All other settings as bA1/bB1 (20000 steps, dropout 0.3, batch 16, lr 3e-4).

## 2026-10-07 — Bonus A-runs finished; scored sweeps launched; controls
Training (p=4, val NLL per piano-side token, same 45 held-out songs): bA1 (A, L1, no chord) min 1.0164 @8500, last 1.1681 @20000; bA2 (A, L1, + chord token) min 0.9946 @8500, last 1.1212 @20000. Both overfit after ~8500 steps even with dropout 0.3 (best.pt = step 8500 used below). Chord token lowers val NLL by 0.022 (single seed; not yet tied to CA).
Lower-bound control ("wrong-song piano": real piano of the next prompt song, tokenized+written on this prompt's grid via `roundtrip_bonus.py --swap 1`; ignores this melody; pooled over 160 bars): p=4 CA 0.0125, OA pitch-class 0.8461, velocity 0.3843, duration 0.6493, IOI 0.4639; p=16 CA 0.0063, OA 0.8453 / 0.3834 / 0.6607 / 0.4497. Ceiling (real piano of the same prompt): see previous entry. So CA has a ~0.01 floor and 0.76 ceiling; OA_pitch is already 0.85 for an unrelated piano, OA_velocity is the most discriminating.
Sweeps launched (tmux gA1/gA2, `src/sweep_bonus.sh`; ckpt best; 4 sampling configs {T1.0 pure, T1.0 top-p0.95, T0.9 top-p0.9, T1.2 top-k5} x 3 seeds {0,1,2}; 10 prompts x 16 bars per cell; rows in `work/gen_bonus/results_<run>.jsonl` with full config + hashes). First cell (bA1 best, T1.0 pure, seed 0): CA 0.1313, malformed 12, 16.7 notes/bar.

## 2026-10-07 — Bonus arch A results (p=4; best.pt step 8500; 10 prompts x 16 bars; 3 seeds per cell, mean +- sd of CA across seeds; `src/agg_bonus.py`, rows `work/gen_bonus/results_bA*.jsonl`)
Controls: ceiling CA 0.7625 (real piano), wrong-song piano CA 0.0125 / OA 0.846 / 0.384 / 0.649 / 0.464, round-trip OA ceilings at p=4: 0.980 / 0.973 / 0.870 / 0.745.
| run | sampling | CA | OA pitch-class | OA velocity | OA duration | OA IOI | notes/bar | malformed (3 seeds) |
|---|---|---|---|---|---|---|---|---|
| bA1 (no chord) | T1.0 pure | 0.140 +- 0.031 | 0.841 | 0.471 | 0.610 | 0.459 | 16.6 | 36 |
| bA1 | T1.0 top-p 0.95 | 0.152 +- 0.015 | 0.857 | 0.518 | 0.693 | 0.491 | 16.5 | 0 |
| bA1 | T0.9 top-p 0.9 | 0.119 +- 0.009 | 0.866 | 0.551 | 0.644 | 0.535 | 16.1 | 0 |
| bA1 | T1.2 top-k 5 | 0.156 +- 0.026 | 0.860 | 0.565 | 0.682 | 0.487 | 15.7 | 3 |
| bA2 (+ chord token) | T1.0 pure | 0.552 +- 0.018 | 0.881 | 0.553 | 0.670 | 0.510 | 16.6 | 22 |
| bA2 | T1.0 top-p 0.95 | 0.583 +- 0.047 | 0.883 | 0.566 | 0.605 | 0.469 | 19.8 | 2 |
| bA2 | T0.9 top-p 0.9 | 0.640 +- 0.021 | 0.891 | 0.514 | 0.652 | 0.507 | 17.7 | 0 |
| bA2 | T1.2 top-k 5 | 0.579 +- 0.045 | 0.886 | 0.520 | 0.690 | 0.500 | 16.8 | 4 |
Findings: (1) the per-bar chord token in the conditioning raises CA from 0.12-0.16 to 0.55-0.64 (ceiling 0.76, floor 0.01); OA changes little. This makes the open chord-input question decisive: melody-only gives CA ~0.15. (2) OA velocity (0.47-0.57) and IOI (0.46-0.54) are far below their p=4 round-trip ceilings (0.97 / 0.75) and not far above the wrong-song floor (0.38 / 0.46); OA pitch-class (0.84-0.89) is close to the floor 0.85 as well, so only CA, OA_V, OA_D discriminate. (3) Lower-entropy sampling (T0.9 top-p 0.9) gives the highest CA (0.640) in bA2; seed sd is 0.02-0.05, so only large differences are meaningful.
Note: first 24 rows stored the CA value under a garbled key (regex spanning a newline) -- values are correct, `agg_bonus.py` normalizes; fixed in `gen_bonus.py` for later rows.
Open question for the TA (draft): "In the bonus task, may the per-bar chord labels (`chord_midi.txt`, present in each `prompts/bonus/<id>/` folder) be used as model input? CA is described as matching 'the conditional chord sequence'." Until answered: melody-only (no chord input) is the conservative main setting; chord-conditioned results are reported as a separate, labeled setting.
Launched/queued (all: 20000 steps, batch 16, lr 3e-4 cosine, warmup 1000, wd 0.01, dropout 0.3, transposition -5..+6): bA4 = A, L1, +chord, p=16 (GPU 1, running); bA5 = A, L0, +chord, p=4 (GPU 2, starts when bB1 finishes); bA6 = A, L2, +chord, p=4 (GPU 0, starts when bB2 finishes). bA3 (A, L1, no chord, p=16; training finished, val NLL not comparable across vocab) is being scored (window gA3, same 4 sampling x 3 seeds).

## 2026-10-07 — Bonus arch B (T5 6+6, p=4) training finished
bB1 (no chord): val NLL min 1.0191 @20000 (= last, still decreasing at the end); bB2 (+ chord): 1.0153 @20000 (still decreasing). Unlike arch A (min @8500, then overfit) the T5 runs had not started overfitting at 20000 steps (dropout 0.3). Val NLL is over decoder tokens incl. Bar/EOS, so not directly comparable to arch A's (piano events + terminator). Chord token changes B's val NLL by only 0.004 (A: 0.022). Scoring sweeps launched: gB1 (GPU 2), gB2 (GPU 0), same 4 sampling x 3 seeds, ckpt best (= step 20000). Candidate follow-up (not launched): B with more steps (40000) since val NLL was still falling.

## 2026-10-07 — Bonus B results, chord-shuffle control, new variant bB4
Scoring (best ckpt, 10 prompts x 16 bars, 3 seeds per cell; `agg_bonus.py`; rows `work/gen_bonus/results_*.jsonl`). CA ranges over the 4 sampling configs (mean over 3 seeds per config); OA ranges likewise:
| run | CA range | OA pitch-class | OA velocity | OA duration | OA IOI |
|---|---|---|---|---|---|
| bA1 A p4 no chord | 0.119-0.156 | 0.841-0.866 | 0.471-0.565 | 0.610-0.693 | 0.459-0.535 |
| bA2 A p4 + chord | 0.552-0.640 | 0.881-0.891 | 0.514-0.566 | 0.605-0.690 | 0.469-0.510 |
| bA3 A p16 no chord | 0.133-0.146 | 0.841-0.859 | 0.485-0.550 | 0.642-0.690 | 0.468-0.556 |
| bB1 B p4 no chord | 0.135-0.165 | 0.836-0.860 | 0.581-0.591 | 0.644-0.676 | 0.438-0.509 |
| bB2 B p4 + chord (encoder) | 0.158-0.183 | 0.829-0.852 | 0.538-0.584 | 0.608-0.664 | 0.498-0.517 |
Controls: ceiling CA 0.7625; wrong-song piano CA 0.0125 / OA 0.846 / 0.384 / 0.649 / 0.464 (p4).
Findings: (1) arch A needs the chord token for CA (0.55-0.64 vs 0.12-0.16); (2) arch B with the chord token in the ENCODER does not benefit (CA 0.16-0.18, same as no chord); (3) p=16 vs p=4 for A without chord: no clear difference in CA or OA (bA3 vs bA1) -- the p=16 ceiling advantage (round-trip) is not yet realized by the model; (4) B has the highest OA_velocity (0.58-0.59) of the no-chord models (A 0.47-0.57), OA_IOI/OA_dur within the A range.
Chord-shuffle control (`gen_bonus.py --chord_shuffle 1`: chord labels permuted across the bars of each prompt, CA still scored vs the TRUE chords; ckpt best, T1.0 top-p0.95, 2 seeds, rows `work/gen_bonus_ctrl/`): bA2 (A + chord): CA 0.650 / 0.556 with true chords -> 0.225 / 0.144 with shuffled chords, so A follows the chord input. bB2 (B + chord in encoder): CA 0.156 / 0.125 with true chords and EXACTLY 0.156 / 0.125 with shuffled chords for the same seeds, so B (20000 steps) is effectively not reading the chord tokens (cross-attention has not learned the bar alignment; interpretation).
New variant (`bonus_lib.build_B` chord modes: 1 encoder, 2 decoder (forced), 3 both; layout checked on real data: chord 2 -> 3 forced decoder chord tokens for 3 bars, 0 in the encoder): bB4 = B 6+6, p=4, chord mode 2 (chord token forced after every Bar in the decoder, injected bar by bar at inference), 20000 steps, same hyperparameters as bB1/bB2, GPU 1. Also scoring bA4 (A, p=16, + chord) in window gA4. Still queued/running: bA5 (A L0 + chord), bA6 (A L2 + chord), bB3 (B p16 no chord).

## 2026-10-07 — Bonus lookahead ablation (A + chord token in the melody segment, p=4), chord modes, incident
Lookahead ablation (best ckpt step 8500 in all three; 3 seeds per cell; CA range over the 4 sampling configs, mean over seeds): bA5 L0: CA 0.575-0.710 (T0.9 p0.9 best 0.710), bA2 L1: 0.552-0.640, bA6 L2: 0.294-0.367; OA nearly identical (pitch-class 0.85-0.89, velocity 0.52-0.57, duration 0.62-0.69, IOI 0.45-0.55); val NLL min 0.9953 (L0) / 0.9946 (L1) / 0.9974 (L2). Reading (interpretation): with the chord token in the melody segment, more lookahead puts the matching bar's chord further back relative to the piano bar that has to use it, and CA falls; val NLL does not see this difference. bA4 (A p16 + chord) partially scored before the incident below (p16 cells: CA 0.537-0.560 so far).
New chord modes (`bonus_lib`; layouts + loss weights checked on real data by `check_bonus_lib.py`): arch A: 2 = chord token at the start of the piano segment, forced at inference (given chord, no loss on it); 3 = both melody segment and piano segment; 4 = chord token is the FIRST TOKEN OF THE PIANO SEGMENT AND GENERATED by the model (loss on it; no chord input at inference; trained on the dataset's chord annotations). Arch B: 2 = decoder chord forced bar by bar, 3 = both, 4 = decoder chord generated. `gen_bonus.py` logs `pred_chord_acc` (fraction of annotated bars where the model's own chord token equals the annotation) for mode 4.
Launched: bA7 = A, L1, chord mode 4, p4 (GPU 0); bB5 = B 6+6, chord mode 4, p4 (GPU 2); bB4 (B, chord mode 2) still training (GPU 1); same hyperparameters as the other bonus runs (20000 steps, dropout 0.3).
Incident (no effect on logged numbers): while patching `gen_bonus.py` a bad string replacement corrupted its `if __name__` line and the broken file was copied to the server for ~minutes. Cells of the running bA4/bB3 sweeps that started then failed with SyntaxError and wrote NO output or row; the failed cell dirs were deleted and both sweeps restarted (resumable: cells with output are skipped; cells that completed before the incident keep their original code hash). File repaired and re-verified (syntax check + sweep cells running).

## 2026-10-07 — Bonus: chord modes scored, chord-from-melody baselines, next runs
Scored (best ckpt, 10 prompts, 3 seeds per cell, 4 sampling configs; `agg_bonus.py`; ranges over the sampling configs, mean over seeds):
| run | what | CA | OA_P | OA_V | OA_D | OA_IOI | chord-pred acc |
|---|---|---|---|---|---|---|---|
| bA4 | A p16, chord token in melody segment (given) | 0.527-0.560 | 0.876-0.879 | 0.476-0.543 | 0.606-0.689 | 0.548-0.555 | - |
| bB3 | B p16, no chord | 0.098-0.179 | 0.837-0.863 | 0.555-0.599 | 0.675-0.720 | 0.510-0.558 | - |
| bB4 | B p4, chord token in DECODER after each Bar, forced bar by bar (given) | 0.496-0.658 (T0.9 p0.9: 0.658 +- 0.028) | 0.881-0.893 | 0.599-0.641 | 0.596-0.684 | 0.429-0.551 | - |
| bA7 | A p4, chord = first token of the piano segment, GENERATED (no chord input) | 0.129-0.177 | 0.846-0.858 | 0.517-0.559 | 0.662-0.681 | 0.487-0.540 | 0.142-0.187 |
| bB5 | B p4, chord token in decoder, GENERATED (no chord input) | 0.123-0.173 | 0.846-0.857 | 0.537-0.573 | 0.644-0.686 | 0.472-0.574 | 0.138-0.193 |
Findings: (1) bB4 shows that B can use the chord when the chord token is in the decoder next to the notes (CA up to 0.658, with OA_velocity 0.60-0.64, higher than the A models' 0.48-0.57); the encoder-side chord (bB2) was ignored. (2) Generated chord tokens (bA7, bB5) do not raise melody-only CA (0.12-0.18 = no-chord level); the models' own chord prediction accuracy is 0.14-0.19. (3) p=16 (bA4 vs bA2): CA similar to slightly lower, OA_IOI +~0.05 (0.548-0.555 vs 0.469-0.510), other OA unchanged.
Chord-from-melody baselines (`src/chord_baselines.py`; 3199 annotated bars of 16-bar windows of the 45 validation songs, training-set label distribution near uniform because songs are in all keys): majority label 0.054, previous bar's TRUE chord 0.130, melody pitch-class template (duration-weighted chroma vs 24 triads) 0.164. So predicting the bar chord from the melody is intrinsically hard in this setting; bA7/bB5's 0.14-0.19 is on par with the template heuristic (not evidence that the models under-learn).
Fix: gen_B forced-chord mode used `max_new_tokens = ctx - len(dec) - 1`, which went negative on a dense piece (bB4 T1.0_p0.95 seed 1 crashed); now the remaining bars are left empty and counted in `truncated_bars` (that cell rerun: CA 0.5375, 6 truncated bars).
Launched (all 20000 steps unless noted, dropout 0.3, batch 16, lr 3e-4, p=4): bB6 = B, chord mode 2 (given, decoder), 40000 steps (bB4 val NLL was still decreasing at 20000) GPU 0; bA8 = A L1, chord mode 2 (chord forced at the start of the piano segment) GPU 1; bA9 = bA8 + `--init_from work/runs/v2_drop0.3/best.pt` (our own unconditional checkpoint, vocab extended 257 -> 284) GPU 2. Check: bA9's validation songs are IDENTICAL to v2's (45 songs, same seed split), so no leakage from the unconditional pre-training; bA9 val NLL 0.988 @500 steps vs bA8 2.12 @500 / 1.60 @1000.

## 2026-10-07 — Stage-1 chord predictor (arch C) + two-stage melody-only pipeline
bC1 = arch C (`train_bonus.py --arch C`; T5 4+4 layers, d256, 4 heads, ff1024, dropout 0.3; encoder = whole 16-24-bar melody, decoder = one chord token per bar + EOS; data `work/bonus_p4/bonus_train.json`, same 45 held-out songs; 8000 steps, batch 16, lr 3e-4; trained from scratch on POP909 chord annotations only). Val: min NLL 2.0652 @7000; teacher-forced top-1 chord accuracy (annotated bars, given the previous TRUE chords) 0.333 @6250 / 0.330 last (baselines on the same kind of bars: majority 0.054, previous true chord 0.130, melody pitch-class template 0.164).
Two-stage pipeline (`gen_bonus.py --chord_from bC1/best.pt`, greedy chords from the melody only; stage 2 = bB4 best (B with forced decoder chord, trained with the TRUE chords); chord file NOT read; 10 prompts, 3 seeds, rows `work/gen_bonus/results_bB4_B_6L_decchord_s1greedy.jsonl`): free-running stage-1 chord accuracy on the prompts 0.293; CA 0.202 (T1.0 pure) / 0.227 (top-p 0.95) / 0.229 (T0.9 p0.9) / 0.212 (T1.2 k5) (+- 0.02-0.03); OA pitch-class 0.841-0.858, velocity 0.578-0.619, duration 0.626-0.679, IOI 0.460-0.535.
Reference points (same prompts/metrics): melody-only models CA 0.12-0.18 (bA1, bA3, bB1, bB3, bA7, bB5); given chords CA 0.50-0.66 (bA2, bA5, bB4); ceiling 0.7625; floor 0.01. Reading: a learned melody->chord stage lifts the melody-only pipeline from ~0.15 to ~0.22 CA; the remaining gap to given chords is mostly stage-1 accuracy (0.29).

## 2026-10-07 — bA8 / bA9 scored (A, chord mode 2; best ckpt; 10 prompts; 3 seeds per cell; p=4)
| run | change | CA | OA_P | OA_V | OA_D | OA_IOI |
|---|---|---|---|---|---|---|
| bA8 | A L1, chord token forced at the start of the piano segment (given chords); val NLL min 0.9932 @8500 | 0.575-0.602 | 0.884-0.889 | 0.529-0.566 | 0.647-0.683 | 0.513-0.552 |
| bA9 | bA8 + init from unconditional v2 best (best.pt = step 3000, val NLL 0.9617, then overfits to 1.31) | 0.500-0.575 | 0.878-0.888 | 0.506-0.551 | 0.631-0.679 | 0.454-0.491 |
Reading: at lookahead 1 the chord position (melody segment bA2 0.552-0.640 vs piano segment bA8 0.575-0.602) makes no clear difference; unconditional initialization lowers val NLL (0.962 vs 0.993) but does not improve CA or OA (slightly lower; seed sd 0.02-0.06). Val NLL is therefore not predictive of the task metrics here. Not tested yet: lookahead 2 with the chord in the piano segment (the lookahead degradation seen in bA6 may be a chord-distance effect).

## 2026-10-07 — Task b: prompt continuation, first set (3 TA prompts x 3 inference configs, n=1, seed 0)
Model: `work/runs/v2_drop0.3/best.pt` (8L x 512 x 8h GPT-2, dropout 0.3, p=4 REMI-like tokens, 20k-step schedule, step 19500, val NLL 0.9523). Prompts `data/POP909/prompts/basic-b/{001,002,003}` tokenized with `tokenize_hw2.py` (8 bars each). `generate.py cont --prompt_bars 8 --n_bars 32 --seed 0`, sliding-window generation, <PAD>/<BOS>/<EOS> never sampled. Configs: cfg1 T=1.0 pure; cfg2 T=1.0 + Bar-logit bias 3; cfg3 T=0.9 top-p 0.9. Output dirs `work/taskb/cfg*/` (.mid + .wav + tokens json + config json); WAVs copied to `listen/taskb/` on the Mac.
Checks (tokens): in all 9 pieces the first 8 bars equal the prompt tokens verbatim and the piece has exactly 32 bars. Continuation density (notes/bar in bars 9-32) vs the prompt's own density: 001: prompt 15.5 -> cfg1 20.8 / cfg2 19.6 / cfg3 38.3; 002: 22.1 -> 23.4 / 17.2 / 28.3; 003: 22.8 -> 22.4 / 23.8 / 33.4. cfg3 (T0.9 top-p 0.9) is ~1.3-2.5x denser than the prompt in all three prompts; cfg1/cfg2 stay near the prompt density. Malformed tokens: cfg1 6, cfg2 0, cfg3 0. Note: the MIDI is written from tokens, so the prompt part is the quantized (16th-note grid) version of the prompt, with constant tempo from the Tempo token. Best config per prompt: to be chosen by listening (user), not by metric.
Launched (2026-10-07): bA10 = A L2, chord mode 2 (forced in the piano segment) GPU 1 (tests whether the lookahead-2 CA drop of bA6 is a chord-distance effect); bA11 = A L1, chord mode 2, melody_w 0.2 (loss on melody/condition tokens) GPU 2; bC2 = stage-1 chord predictor 6L x 512 x 8h then bC3 = 4L x 256 with dropout 0.5 (8000 steps each) GPU 3. All else as bA8 / bC1. bB6 (B, chord mode 2, 40000 steps) still training on GPU 0.

## 2026-10-08 — Stage-1 variants, bA10/bA11/bB6 trained
Stage-1 chord predictors (arch C, 8000 steps, same data/val split; teacher-forced top-1 chord accuracy on annotated validation bars, at the min-val-NLL checkpoint / best over steps): bC1 4L x 256, dropout 0.3: 0.324 / 0.333; bC2 6L x 512 x 8h: 0.283 / 0.285; bC3 4L x 256, dropout 0.5: 0.276 / 0.283. The small model stays the best stage 1; bigger or more regularized were worse (single seed).
Training: bA10 (A L2, chord forced in piano segment): min val NLL 0.9993 @8000; bA11 (A L1, chord mode 2, melody_w 0.2): 0.9994 @8000; bB6 min val 0.9883 @31500 last 0.9925 @40000. Scoring sweeps launched (gA10, gA11, gB6; best ckpt, 4 sampling x 3 seeds).

## 2026-10-08 — bA10 / bA11 / bB6 scored (best ckpt, 10 prompts, 3 seeds per cell, p=4; chord given in all three)
| run | what | CA | OA_P | OA_V | OA_D | OA_IOI |
|---|---|---|---|---|---|---|
| bA10 | A L2, chord token forced at the start of the piano segment | 0.535-0.602 | 0.873-0.884 | 0.500-0.537 | 0.614-0.693 | 0.488-0.542 |
| bA11 | A L1, chord forced in piano segment, melody-token loss weight 0.2 | 0.550-0.637 | 0.877-0.883 | 0.518-0.538 | 0.651-0.702 | 0.455-0.521 |
| bB6 | B 6+6, chord forced in decoder, 40000 steps (val NLL 0.9883 @31500) | 0.627-0.692 (T0.9 p0.9: 0.692 +- 0.024) | 0.887-0.895 | 0.570-0.592 | 0.602-0.685 | 0.500-0.533 |
Findings: (1) bA10 vs bA6: with the chord token in the piano segment, lookahead 2 no longer collapses (CA 0.535-0.602 vs 0.294-0.367 with the chord in the melody segment), supporting the chord-distance explanation for the lookahead drop; lookahead 2 is not better than 0/1. (2) Melody-token loss weight 0.2: no clear effect on CA (0.550-0.637 vs bA8 0.575-0.602). (3) bB6 (longer B) is the best chord-given model so far (CA up to 0.692 vs bB4 0.658 at 20k steps, bA8 0.602). (4) OA_V (0.50-0.59), OA_D (0.60-0.70) and OA_IOI (0.45-0.55) hardly move across all A/B variants.

## 2026-10-08 — bB6 typed-temperature sweep and two-stage melody-only (best ckpt, 10 prompts, 3 seeds per cell, p=4)
Base = bB6 (B 6+6, chord forced in the decoder, 40k steps), sampling T=0.9 top-p 0.9 (rows `work/gen_bonus/results_bB6_B_6L_decchord_40k*.jsonl`; `src/sweep_bonus_typed.sh`, typed sampling via `TypedSampler`: t_struct = Pos-vs-Bar/chord-step temperature, t_pitch, t_vel, t_dur; top-p applied after the per-type scaling). Chord given (true chord_midi labels).
| variant | CA | OA_P | OA_V | OA_D | OA_IOI | notes/bar |
|---|---|---|---|---|---|---|
| base (T0.9 p0.9) | 0.692 +- 0.024 | 0.895 | 0.592 | 0.602 | 0.533 | 19.7 |
| t_struct 0.8 | 0.731 +- 0.015 | 0.892 | 0.579 | 0.642 | 0.526 | 18.2 |
| t_vel 0.5 | 0.696 +- 0.008 | 0.884 | 0.541 | 0.630 | 0.489 | 17.9 |
| t_vel 0.3 | 0.646 +- 0.067 | 0.895 | 0.519 | 0.643 | 0.523 | 17.3 |
| t_dur 0.7 | 0.644 +- 0.069 | 0.887 | 0.595 | 0.633 | 0.479 | 16.3 |
| t_pitch 0.8 | 0.677 +- 0.081 | 0.879 | 0.571 | 0.621 | 0.514 | 17.4 |
| t_vel 0.5 + t_dur 0.7 | 0.688 +- 0.055 | 0.885 | 0.530 | 0.660 | 0.495 | 17.2 |
Reading: lowering the structure temperature to 0.8 gives the best CA so far (0.731 vs ceiling 0.7625, floor 0.01); LOWERING the velocity temperature LOWERS OA_V (0.592 -> 0.541 -> 0.519), i.e. the OA_V gap is not sampling noise on the velocity tokens in this direction; lower duration temperature does not help OA_D/OA_IOI. Seed sd is 0.01-0.08, so differences below ~0.05 CA are not resolved.
Two-stage melody-only with bB6 as stage 2 (stage 1 = bC1 greedy, free-running chord accuracy 0.293; chord file not read; same 4 sampling configs): CA 0.246 +- 0.003 (T0.9 p0.9) / 0.242 (T1.0 p0.95) / 0.231 (pure) / 0.206 (T1.2 k5); OA_V 0.600-0.609 (highest OA_V of any config so far), OA_P 0.840-0.855, OA_D 0.589-0.653, OA_IOI 0.489-0.540. (bB4 as stage 2: CA 0.202-0.229.)
Launched next typed variants on bB6 (T0.9 top-p0.9 base, 3 seeds each): t_vel 1.3 / 1.6, t_struct 0.6 / 0.7, t_struct 0.8 + t_vel 1.3, t_dur 1.3; and bB7 = bB6 with dropout 0.2 (B 6+6, chord mode 2, 40000 steps, GPU 0).

## 2026-10-08 — SI (optional metric) pipeline
`src/si_eval.py` = MusDr/FMP Python scape-plot code (`MusDr/mueller_audio_tools`, parameters of `run_python_scapeplot.py`: tempo_rel_set(0.5, 2, 7), 12 transposition shifts, thresh 0.25, penalty -2, 2 Hz) + MusDr's SI definition (max fitness within a timescale band; bands short 3-8 s, mid 8-15 s, long 15-128 s). Only change to the third-party code: `MusDr/mueller_audio_tools/ssm_features.py` line 347 `librosa.load(fn_wav, Fs)` -> `librosa.load(fn_wav, sr=Fs)` (current librosa takes sr as keyword; original kept as `.orig`; must be listed in the README). WAVs: `render_wav.py` (FluidSynth, TimGM6mb, 22050 Hz). Timing (3 workers): ~24 s per 32-bar piece. Test on 3 real 32-bar crops (real MIDI tokens detokenized + rendered the same way as generated pieces): SI short/mid/long 0.408 / 0.404 / 0.369 (n=3, indicative only).
Job `si1` (tmux, CPU, 12 workers): real = 100 random real 32-bar crops (seed 0 shuffle of crops32_p4.json, detokenized + rendered) and v2 best (n=20, seed 0 per set): pure, bar_bias 3, T1.0 top-p0.95, T0.9 top-p0.9, T1.2 top-k5. Results -> `work/si/out/<name>.json`.

## 2026-10-08 — bB6 typed sweep, round 2 (T0.9 top-p0.9 base; 3 seeds; chord given)
| variant | CA | OA_P | OA_V | OA_D | OA_IOI |
|---|---|---|---|---|---|
| base | 0.692 +- 0.024 | 0.895 | 0.592 | 0.602 | 0.533 |
| t_struct 0.6 | 0.671 +- 0.094 | 0.890 | 0.588 | 0.639 | 0.492 |
| t_struct 0.7 | 0.721 +- 0.031 | 0.890 | 0.579 | 0.658 | 0.514 |
| t_struct 0.8 | 0.731 +- 0.015 | 0.892 | 0.579 | 0.642 | 0.526 |
| t_struct 0.8 + t_vel 1.3 | 0.660 +- 0.071 | 0.885 | 0.602 | 0.620 | 0.522 |
| t_vel 1.3 | 0.679 +- 0.033 | 0.876 | 0.613 | 0.602 | 0.516 |
| t_vel 1.6 | 0.694 +- 0.042 | 0.863 | 0.600 | 0.595 | 0.447 |
| t_dur 1.3 | 0.673 +- 0.036 | 0.882 | 0.574 | 0.640 | 0.550 |
(round-1 rows: t_vel 0.3/0.5, t_dur 0.7, t_pitch 0.8, t_vel0.5+t_dur0.7 above.) Reading: no typed variant moves OA_V/OA_D/OA_IOI outside 0.52-0.61 / 0.60-0.66 / 0.45-0.55 (seed sd ~0.01-0.09 in CA); the structure temperature (0.7-0.8) gives the highest CA (0.721-0.731) but the non-monotone sequence (0.6: 0.671, 0.7: 0.721, 0.8: 0.731, 1.0: 0.692) with sd up to 0.094 means differences of ~0.04 are not resolved. t_vel 1.3: OA_V +0.02 vs base (within noise).

## 2026-10-08 — Task a representation/loss ablation runs at the v2 regularization (dropout 0.3, 8L x 512 x 8h, 20000 steps, batch 8, lr 3e-4, ctx 4096, seed 0, same 45 val songs)
Chord-token tokenization (`tokenize_hw2.py --chord 1`): a `Chord_x` token (24 triads / Chord_N, eval_bonus bar-dominant-chord definition from chord_midi.txt) right after every Bar token; vocab 282 (257 + 25); check: stripping the chord tokens reproduces EXACTLY the no-chord sequences for all 896 songs; 9.3% of bars are Chord_N. `train_uncond.py` / `generate.py` keep the chord token directly after Bar when re-inserting the tempo in mid-song crops / sliding windows; `detokenize` ignores Chord_ tokens; eval_basic ignores them as well (only Bar / Pos_ / Pitch_ names are used).
Runs (single change vs v2): w1 = + chord tokens (GPU 1), w2 = pos_per_beat 16 (vocab 545; the earlier p16 run u2 used dropout 0.1) (GPU 2), w3 = label smoothing 0.1 (GPU 3). bB7 (bonus) still training on GPU 0, SI job `si1` on CPU.

## 2026-10-08 — Submission entry point drafted (not yet tested on the server)
`infer.py` (repo root): subcommands `uncond` (20 x 32-bar, default config = v2 best + Bar-bias 3), `continue` (tokenizes the TA basic-b prompts with the same tokenizer, 3 configs x n=1), `bonus` (melody_only / chord_given; reads `vocab/bonus_final.json` for the final checkpoint + flags, not yet written); all call the experiment scripts via subprocess (src/generate.py, src/gen_bonus.py, src/render_wav.py), settable `--out_dir` and `--seed`, `.mid` + `.wav` written. `bonus_data.py --prompts_only` added (inference needs no training data). Checkpoint names expected under `checkpoints/`: uncond_v2_best.pt + the bonus ones chosen later.

## 2026-10-08 — Caveat found: seed-collision in sweep output folders (metrics unaffected)
`sweep_uncond.py` named the output folder `<out_root>/<run>/<ckpt>__<sampling>` without the seed, so a re-run with another seed into the same out_root overwrote the previous seed's .mid/tokens (rows in the results jsonl were written at generation time and remain correct for every seed). Affected: the v2 typed sweeps (out_root `work/gen_typed`, seeds 0 -> 1 -> 2 for `B0_pure` and `B_bias3`): those two folders now hold the SEED-2 pieces (checked via `g_config.json`: seed 2). Consequences: (1) the WAVs in `listen/v2best_{pure,bias3}_{0,1,2}.wav` are seed-2 pieces g_00..g_02; (2) the SI sets `v2best_pure` / `v2best_bias3` (job si1) are the seed-2 sets (20 pieces each), not seed 0; (3) the other round-1/2 folders (`work/gen/*`) hold seed 0 (verified for v2: T1.0_p0.95, T0.9_p0.9, T1.2_k5).
Fix: from now on folders for seed != 0 carry a `__s<seed>` suffix (seed 0 keeps the old names) -- applied before the seed 1/2 iterations of the Task a sweeps started (work/gen_taskA, results `work/gen_taskA/results_<run>_s<seed>.jsonl`). The Task a deliverable pieces and the final SI numbers will be taken from the clean `work/gen_taskA/v2_drop0.3/best__T1.0_pure_barbias3` (seed 0) folder.
Round-3 Task a sweeps launched (5 sampling configs {T1.0 pure, T1.0 top-p 0.95, T0.9 top-p 0.9, T1.2 top-k 5, T1.0 pure + Bar-bias 3} x seeds {0,1,2}, n=20, best ckpt): v2 (GPU 0), w1 chord tokens (GPU 1), w2 p=16 (GPU 2), w3 label smoothing 0.1 (GPU 3), all at dropout 0.3 / 20000 steps. Training val NLL (NOT comparable across vocabularies): w1 0.9714 @19500 (vocab 282 incl. chord tokens), w2 1.2160 @19500 (vocab 545), w3 1.0171 @20000 (unsmoothed NLL; v2 0.9523 at vocab 257).

## 2026-10-08 — Task a: representation / loss ablation at the v2 regularization, 5 sampling configs x 3 seeds (full table `results/taskA_table.md`; rows `results/gen/results_{v2,w1,w2,w3}_s{0,1,2}.jsonl`)
Setup: best ckpt (lowest val NLL) of v2 (p=4, no chord), w1 (+ chord tokens), w2 (p=16), w3 (label smoothing 0.1); all GPT-2 8L x 512 x 8h, dropout 0.3, 20000-step cosine schedule, batch 8 x <=4096 tokens, lr 3e-4, wd 0.01, transposition -5..+6, same 45 held-out songs. Generation: 20 pieces x 32 bars per (config, seed), seeds 0/1/2 (60 pieces per cell), sliding-window sampling, unmodified `scripts/eval_basic.py`; mean +- sd over the 3 seeds. Real reference (1864 non-overlapping 32-bar crops of the training songs): p=4 H1 2.118 / H4 2.617 / GS 0.887; p=16 GS 0.962 (H1/H4 identical); real density 20.6 notes/bar.
| model | sampling | H1 | H4 | GS | notes/bar | malformed |
|---|---|---|---|---|---|---|
| v2 (p4) | pure T1.0 | 2.146 +- 0.057 | 2.569 +- 0.056 | 0.923 +- 0.006 | 29.7 | 84 |
| v2 | pure + Bar bias 3 | 2.045 +- 0.025 | 2.567 +- 0.026 | 0.888 +- 0.007 | 21.7 | 105 |
| v2 | T1.0 top-p 0.95 | 2.160 +- 0.030 | 2.539 +- 0.054 | 0.967 +- 0.008 | 31.2 | 0 |
| v2 | T0.9 top-p 0.9 | 2.147 +- 0.067 | 2.447 +- 0.028 | 0.990 +- 0.001 | 31.0 | 0 |
| v2 | T1.2 top-k 5 | 2.233 +- 0.016 | 2.629 +- 0.019 | 0.894 +- 0.008 | 30.5 | 6 |
| w1 (+chord tokens) | pure | 2.175 +- 0.060 | 2.546 +- 0.034 | 0.914 +- 0.008 | 30.9 | 228 |
| w1 | pure + bias 3 | 2.038 +- 0.089 | 2.523 +- 0.066 | 0.889 +- 0.008 | 21.8 | 84 |
| w2 (p16) | pure | 2.194 +- 0.014 | 2.636 +- 0.030 | 0.963 +- 0.003 | 28.9 | 110 |
| w2 | pure + bias 3 | 2.102 +- 0.028 | 2.634 +- 0.024 | 0.962 +- 0.003 | 22.6 | 129 |
| w2 | T1.0 top-p 0.95 | 2.179 +- 0.013 | 2.533 +- 0.019 | 0.989 +- 0.002 | 27.9 | 0 |
| w2 | T0.9 top-p 0.9 | 2.051 +- 0.057 | 2.393 +- 0.024 | 0.997 +- 0.001 | 29.3 | 5 |
| w2 | T1.2 top-k 5 | 2.254 +- 0.065 | 2.555 +- 0.065 | 0.956 +- 0.008 | 38.1 | 42 |
| w3 (LS 0.1) | pure | 2.050 +- 0.077 | 2.769 +- 0.045 | 0.746 +- 0.012 | 15.3 | 22594 |
| w3 | pure + bias 3 | 1.408 +- 0.033 | 2.175 +- 0.031 | 0.881 +- 0.002 | 5.5 | 7138 |
| w3 | T0.9 top-p 0.9 | 2.110 +- 0.030 | 2.463 +- 0.040 | 0.984 +- 0.005 | 31.4 | 0 |
(all rows in `results/taskA_table.md`.) Findings: (1) Bar-logit bias 3 brings density (21.7-22.6 vs 29-31 notes/bar) and GS to the real values for p=4 (0.888 vs 0.887) and p=16 (0.962 vs 0.962) without changing H4. (2) The p=16 model with bias 3 is the closest to the real row on all three metrics (|dH1| 0.016, |dH4| 0.017, |dGS| 0.000; sum 0.033) vs v2 + bias 3 (0.073 + 0.050 + 0.001 = 0.124) and w1 + bias 3 (0.080 + 0.094 + 0.002). (3) Chord tokens (w1) do not change H4/GS clearly (differences within the seed sd). (4) Label smoothing (w3) breaks free sampling: up to 22,594 malformed tokens per 60 pieces and 5.5-15 notes/bar (T1.0), and only T0.9 top-p 0.9 gives clean output; conservative nucleus sampling (top-p 0.9/0.95) pushes GS up to 0.98-0.997 (repetitive).
Decision (Task a, by closeness to the real data on H1/H4/GS): model w2 (p=16) best checkpoint, T=1.0 pure sampling + Bar-logit bias 3. Task b must reuse this checkpoint (spec), so Task b is regenerated with w2. Selection used the same 3 x 20 pieces that are reported (no held-out selection set); sd across seeds is 0.01-0.03 for the chosen cell.

## 2026-10-08 — Task b regenerated with the Task a checkpoint (w2, p=16)
Model: `work/runs/w2_p16_drop0.3/best.pt` (GPT-2 8L x 512 x 8h, dropout 0.3, pos_per_beat 16, vocab 545, step 19500, val NLL 1.2160). Prompts: `data/POP909/prompts/basic-b/{001,002,003}` tokenized with `tokenize_hw2.py --pos_per_beat 16` (8 bars each, 0 dropped events, vocab identical to the training vocab). `generate.py cont --prompt_bars 8 --n_bars 32 --n 1 --seed 0` with: cfg1 T=1.0 pure; cfg2 T=1.0 + Bar-logit bias 3 (= the Task a config); cfg3 T=0.9 top-p 0.9. Output `work/taskb_w2/cfg*/` (.mid + .wav + tokens + config); WAVs on the Mac in `listen/taskb_w2/` (the earlier v2/p=4 set moved to `listen/_superseded/`).
Checks: all 9 pieces keep the prompt's 8 bars verbatim and have exactly 32 bars. Continuation density vs the prompt's own: 001 (prompt 15.5/bar): cfg1 17.5 / cfg2 16.0 / cfg3 23.8; 002 (22.1): 27.0 / 21.0 / 26.7; 003 (22.8): 32.1 / 24.7 / 45.0. cfg2 (bias 3) stays closest to the prompt density in all three prompts. Malformed tokens: cfg1 1, cfg2 6, cfg3 0. Best config per prompt: to be chosen by listening (user).
`infer.py` now uses `checkpoints/uncond_w2_best.pt` + `vocab/tok2idx_p16.json` at pos_per_beat 16 for uncond and continue (bonus modes keep the p=4 grid and `vocab/tok2idx_p4.json`).

## 2026-10-08 — Bonus at p=16 (finer grid; round-trip ceilings OA_D 0.959 / OA_IOI 0.935 vs 0.870 / 0.745 at p=4)
Launched: bB8 = bB6's p=16 twin (B 6+6, chord forced in the decoder, 40000 steps, dropout 0.3, `work/bonus_p16`, vocab 572; GPU 1) and bC4 = stage-1 chord predictor at p=16 (same config as bC1: 4L x 256, 8000 steps; GPU 2). bB7 (bB6 with dropout 0.2) still training on GPU 0 (val NLL 1.006 @12000 when last checked). SI job `si2` (CPU) running on the clean seed-0 Task a sets + real crops at p=4 and p=16.

## 2026-10-08 — Report figures (draft) and tooling
Report tooling on the Mac: Chrome, LibreOffice, python-pptx, matplotlib available. Figures: `report/make_figures.py` (reads only the saved logs / result rows under `results/`; palette = validated categorical slots 1-4, light surface; the validator reported PASS on lightness/chroma/CVD (worst adjacent dE 9.1 CVD, 22.9 normal) with a contrast WARN for aqua (2.74:1) and yellow (2.11:1) -> those series carry direct labels / a legend and every figure has its numbers in a table). Drafts: `report/figs/val_curves.png` (validation NLL: dropout 0.1 overfits after ~9k steps, dropout 0.3 does not), `density.png` (notes/bar by bar quarter: real vs v2 pure vs v2 + Bar bias 3), `bonus_ca.png` (bonus CA by model, T0.9 top-p 0.9, 3 seeds, grouped by whether chord labels are given). A first version of the bonus figure averaged the two-stage (melody-only) rows into the given-chord bars of the same checkpoint (bB6 showed 0.47 +- 0.22 instead of 0.692 +- 0.024); caught by looking at the render and checking against the logged table; fixed by grouping per results file.

## 2026-10-08 — SI results (optional metric) and the Task a trade-off
SI = MusDr fitness-scape-plot SI (`src/si_eval.py`; bands short 3-8 s, mid 8-15 s, long 15-128 s); mean over pieces (sd over pieces); rows `results/si/*.json`. Real = 100 random real 32-bar crops (rendered the same way: tokens -> MIDI -> FluidSynth): p=4 tokenization 0.369 / 0.376 / 0.344 (sd 0.071 / 0.090 / 0.104), p=16 tokenization 0.370 / 0.375 / 0.343. Generated sets = 20 pieces, seed 0, folders `work/gen_taskA/<run>/best__<sampling>` (clean seed-0 sets):
| model | sampling | SI short / mid / long |
|---|---|---|
| v2 (p4) | pure | 0.378 / 0.372 / 0.331 |
| v2 | pure + Bar bias 3 | 0.316 / 0.281 / 0.218 |
| v2 | T1.0 top-p 0.95 | 0.492 / 0.529 / 0.503 |
| v2 | T0.9 top-p 0.9 | 0.756 / 0.747 / 0.675 |
| v2 | T1.2 top-k 5 | 0.343 / 0.335 / 0.305 |
| w1 (+chord tokens) | pure / pure + bias 3 | 0.378 / 0.326 / 0.249 ; 0.368 / 0.320 / 0.253 |
| w2 (p16) | pure | 0.362 / 0.352 / 0.283 |
| w2 | pure + Bar bias 3 | 0.329 / 0.294 / 0.247 |
| w2 | T1.0 top-p 0.95 | 0.550 / 0.531 / 0.508 |
| w2 | T0.9 top-p 0.9 | 0.726 / 0.704 / 0.651 |
| w2 | T1.2 top-k 5 | 0.470 / 0.455 / 0.410 |
| w3 (LS) | T0.9 top-p 0.9 | 0.673 / 0.734 / 0.700 |
Findings: (1) Conservative sampling (top-p 0.9/0.95) gives SI 0.5-0.75, 1.4-2x the real value, in line with its GS 0.98-0.997 (repetition). (2) Pure sampling is closest to real SI (w2: 0.362 / 0.352 / 0.283 vs real 0.370 / 0.375 / 0.343). (3) The Bar-bias-3 setting that matched H1/H4/GS/density lowers SI (w2: 0.329 / 0.294 / 0.247; long-scale gap 0.096 vs real; per-piece sd 0.10-0.12, n=20, so the SI gap of ~0.08-0.10 at mid/long is ~4 standard errors). Trade-off for the final Task a config: w2 + bias 3 matches H1 (|d| 0.016), H4 (0.017), GS (0.000), density (22.6 vs 20.6 notes/bar) but not SI long (-0.097); w2 pure matches H4 (0.019), GS (0.001) and SI (-0.008 / -0.023 / -0.061) but is denser (28.9 notes/bar) and H1 is +0.076 off. Final choice pending; a finer bias grid for w2 (bias 1, 2, 2.5; 3 seeds, window aw2b on GPU 3) is running to look for a balance.
Bonus training finished: bB7 (bB6 with dropout 0.2): min val NLL 0.9882 @16500 (last 1.0575; bB6 0.9883 @31500); bB8 (bB6's p=16 twin): min 1.2885 @25000 (vocab 572; not comparable); bC4 (p=16 stage-1): best top-1 0.334 @7500, min NLL 2.0608 @7000, top-1 at the min-NLL checkpoint 0.329 (bC1 at p=4: 0.324 / 0.333). Scoring launched: gB7 (GPU 0), gB8 (GPU 1), two-stage p=16 with bC4 + bB8 (s1B8, GPU 2).

## 2026-10-08 — bB7 / bB8 / two-stage p=16 / w2 bias grid scored (10 prompts, 3 seeds per cell for bonus; 20 pieces x 3 seeds for Task a)
Bonus (best ckpt; ranges over the 4 sampling configs, mean over seeds; chord labels given except the two-stage rows):
| run | what | CA | OA_P | OA_V | OA_D | OA_IOI | notes/bar |
|---|---|---|---|---|---|---|---|
| bB6 (reference) | B, chord in decoder, p4, dropout 0.3, 40k | 0.627-0.692 | 0.887-0.895 | 0.570-0.592 | 0.602-0.685 | 0.500-0.533 | 16.6-19.7 |
| bB7 | same, dropout 0.2 | 0.525-0.633 | 0.880-0.891 | 0.545-0.614 | 0.623-0.657 | 0.429-0.480 | 15.5-17.6 |
| bB8 | same as bB6 at p=16 (vocab 572) | 0.560-0.619 | 0.879-0.888 | 0.565-0.602 | 0.663-0.693 | 0.499-0.557 | 12.9-15.7 |
| two-stage p16: bC4 -> bB8 (melody only; free-running chord acc 0.213) | | 0.169-0.202 | 0.835-0.855 | 0.569-0.607 | 0.675-0.711 | 0.509-0.553 | 14.1-15.5 |
Reading: dropout 0.2 does not help bB6's setup (CA lower; val NLL equal); p=16 raises OA_D (+0.02 to +0.06 vs the p=4 B models) and OA_IOI by a little, but CA is lower than at p=4 (0.56-0.62 vs 0.63-0.69), and the p=16 melody-only two-stage pipeline is lower than at p=4 (0.17-0.20 vs 0.21-0.25), with the p=16 stage 1 predicting 0.213 of the prompt bars correctly vs 0.293 (p=4; 160-bar sample, so noisy). Notes/bar from the p=16 model is lower (12.9-15.7) than from the p=4 models (15.5-19.8).
Task a, w2 (p16) bias grid (3 seeds x 20 pieces; real p16: H1 2.118 / H4 2.617 / GS 0.962, 20.6 notes/bar): bias 1: H1 2.179 +- 0.007, H4 2.665 +- 0.003, GS 0.964, 27.1 notes/bar; bias 2: 2.162 +- 0.028 / 2.656 +- 0.015 / 0.962, 25.1; bias 2.5: 2.117 +- 0.007 / 2.647 +- 0.024 / 0.965, 24.5; (bias 3 from the earlier table: 2.102 / 2.634 / 0.962, 22.6; pure: 2.194 / 2.636 / 0.963, 28.9). Density falls monotonically with bias; H4/GS are flat; H1 reaches the real value at bias 2.5. SI for bias 1 / 2 / 2.5 running (window si3).

## 2026-10-10 — Lecture review (lecture05_midi_generation.pdf 94 slides, lecture06_midi_generation_2.pdf 110 slides; text of all slides + images of slides 73-79 of lecture 05)
Relevant content and what we did about it:
- L5 s46-51 token representations: MIDI-like (note-on/off + time-shift, no bar lines) vs REMI (bar + sub-beat position + note duration) vs CP; L5 s62-63: REMI beats MIDI-like for pop piano. Ours: REMI at p=4 / p=16 / with chord tokens measured; MIDI-like and CP-word NOT tried (the HW table example lists a CP-word row) -> TODO.
- L5 s73 training: "training loss around 0.30 gives better result; overfits and generates overly repeating content when loss is nearly 0; keep several checkpoints and listen"; "no need to consider validation data (unless conditional enc-dec)". Ours: our held-out val NLL minimum came at ~7-9k steps (dropout 0.1) and later checkpoints overfit; repetitive output at conservative sampling was measured (GS 0.98-0.997, SI 0.5-0.75). COMPLIANCE: the HW says to use the WHOLE POP909 training set; our experiments held out 45 of 896 songs (5%) for selection. Fix: keep the held-out runs for choosing settings, retrain the FINAL models on all 896 songs with the selected step counts (launched below).
- L5 s74-79 sampling: greedy, beam search, random, top-k, nucleus, temperature; "sampling parameters influence the result A LOT; people often did not describe how they chose them -> unfair comparison"; Compose & Embellish practice: search tau in {1.2,1.3,1.4} x p in {.95,.97,.98,.99} and choose the combination whose generated outputs have the CLOSEST MEAN PERPLEXITY (measured by the model itself) to the validation data; slide 79: match the per-token ENTROPY HISTOGRAM of sampling to that of the validation set (JS divergence 0.057 "just right" vs 0.192 too conservative vs 0.536 too chaotic). Ours: top-k / top-p / temperature / pure measured; greedy and beam NOT run; we have NOT used the entropy-histogram / perplexity-matching protocol -> TODO (principled way to choose the Task a/b/bonus sampling configs and to report how they were chosen).
- L5 s85-91 objective metrics: H, GS, CPI (chord progression irregularity, not required here), SI. L5 s92: MusPy general metrics (not required).
- L6 s11-21 Compound Word Transformer (group pitch/duration/velocity tokens into one step, multiple embeddings + K output heads, per-type sampling; ~3-4x shorter sequences, faster convergence). Ours: not implemented; typed temperatures (per token type) are implemented for REMI sampling (cf. L6 s21).
- L6 s33/s57 conditioning mechanisms: cross-attention (Theme Transformer), in-attention (MuseMorphose: condition embedding added inside the attention input), interleaving prompt (Compose & Embellish). Ours: A = interleaving; B = encoder-decoder cross-attention (T5); in-attention NOT tried (a bar-level chord embedding added to every token embedding of the bar would be the analogue) -> TODO candidate.
- L6 s42 "The condition might be ignored! (powerful decoder / strong information in previous tokens) -> reduce the decoder's power or increase the relevance of the condition": matches our finding that arch B ignored the encoder chord tokens (CA unchanged under shuffled chords) and that moving the chord to the decoder (next to the notes) fixed it; to be cited in the report.
- L6 s49-54 Compose & Embellish: two-stage (lead sheet first, then piano), structure tokens, lead-sheet pre-training help; ours: two-stage melody -> chords (stage 1) -> piano (stage 2) for the melody-only bonus setting. L6 s25-31 token-design issues (token order for multi-track, chords as root/quality, key-relative functional representation, structure tokens): not tried; a key-relative chord representation could help the stage-1 chord predictor (low accuracy 0.29-0.33, songs in all keys).
- Not relevant to this homework: RNN/VAE/GAN/diffusion/piano-roll/MusicXML/ABC/LLM sections.
Plan changes: (1) FINAL models retrained on all 896 songs (F1 = w2 config p16, 20000 steps; F2 = bB6 config, 40000 steps; F3 = stage-1 bC1 config, 8000 steps; all `--val_frac 0`, tmux f1-f3, 2026-10-10; no best.pt without validation, last.pt = end of the cosine schedule is used); (2) entropy-histogram / perplexity-matching selection of sampling configs; (3) greedy row; (4) optional representation ablations MIDI-like and CP-word; (5) optional in-attention chord embedding for the accompaniment model.

## 2026-10-10 — Homework-aligned experiment protocol started (user instruction: experiment setting must follow the homework description)
Alignment rules now applied: (1) every model that appears in a reported table / the submission is trained on ALL 896 POP909 training songs (no held-out songs; `--val_frac 0`), with step counts / dropout chosen on the earlier held-out-validation (dev) runs, checkpoint = last.pt (end of the cosine schedule); (2) POP909 only, from scratch (the bonus init-from-unconditional run bA9 is a dev ablation only); (3) real-data reference rows use all 896 songs; (4) sampling configs are chosen by a documented protocol (entropy-histogram / perplexity matching on the dev model's held-out songs, lecture 05), not by looking at the evaluation metrics only; (5) the Task b model is the Task a checkpoint; (6) bonus main result = melody-only input (chord_midi.txt input reported as a separate, labeled setting until the TA confirms).
Queues (`src/final_runs.sh <gpu> <jobs...>`, tmux q0-q3; all `--val_frac 0 --dropout 0.3`):
- Task a family (GPT-2 8L x 512 x 8h, 20000 steps, batch 8 x <=4096 tokens): F1 = p=16 (running), F4 = p=4, F5 = p=4 + chord tokens, F6 = p=4 + label smoothing 0.1 (q3 chain).
- Bonus arch A (10000 steps, batch 16; dev check `DEV_A2_10k` with a 5% held-out split verifies that the 10k-step schedule ends near its validation minimum; the dev A runs had their minimum at step 8500 of a 20k schedule): FA2 (L1, chord in melody segment), FA1 (no chord), FA8 (chord in piano segment), FA5 (L0), FA6 (L2), FA10 (L2, chord in piano segment), FA7 (predicted chord) (q2 chain, after F3).
- Bonus arch B (T5 6+6): F2 = chord in decoder p4 40k (running), FB1 no chord / FB2 encoder chord / FB5 predicted chord (20k each, q1 chain after F2), FB8 = chord in decoder p=16 40k + FC4 = p=16 stage-1 (q0 chain after F1); F3 = stage-1 chord predictor p4 8000 steps (done).
Entropy matching: `src/entropy_match.py` (reference = the model's per-token entropy / NLL on real held-out 32-bar crops, teacher forced; per config: 20 sampled pieces, same statistics, JS divergence of the entropy histograms, |d NLL|, plus H1/H4/GS from the unmodified eval_basic). Test on w2 (n=4): real held-out mean entropy 1.124 (mean NLL 1.299, perplexity 3.665); T=1.0 pure: mean entropy 1.047, JS 0.0046; T=1.0 top-p 0.9: 0.395, JS 0.104 (too conservative, as in the lecture's example). Full grid on w2 (T 0.8-1.2 x top-p 0.9/0.95/0.98/1.0 x Bar bias 0/3, n=20): window em1 -> `work/entropy_match_w2.json`.
