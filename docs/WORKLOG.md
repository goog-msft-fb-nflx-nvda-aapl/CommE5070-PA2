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

## 2026-10-10 — Experiment: greedy / beam-search decoding (lecture 05 s74), dev model w2 (p16, best ckpt, step 19500)
`generate.py uncond --greedy 1 [--num_beams 3] --n 1 --seed 0` (deterministic: one piece; beam search is applied chunk by chunk inside the sliding window, an approximation), scored with the unmodified `eval_basic.py`. Real reference p16: H1 2.118 / H4 2.617 / GS 0.962.
| decoding | H1 | H4 | GS | notes/bar | distinct bars (of 32) |
|---|---|---|---|---|---|
| greedy (argmax) | 1.657 | 2.303 | 0.9995 | 40.8 | 3 |
| beam search, 3 beams | 1.689 | 1.928 | 1.0000 | 42.0 | 3 |
Reading: deterministic decoding collapses into a loop of ~3 distinct bars (GS ~ 1.0) and is ~2x denser than real data; n = 1 piece each, so only indicative. Configs in `results/decode/`.

## 2026-10-10 — Representation ablation from the lectures: MIDI-like and Compound Word (`src/alt_repr.py`, `src/test_alt_repr.py`)
Both are derived from the TA-style REMI tokens (`work/seqs_p*.json`) and decoded back to the same TA-style tokens, so the unmodified `eval_basic.py` scores them like the REMI models; same GPT-2 backbone family (8L x 512 x 8h, dropout 0.3, lr 3e-4 cosine, warmup 1000, wd 0.01, transposition -5..+6, bar-aligned random crops), dev split = the same 45 held-out songs.
- MIDI-like (musical time, no bar tokens): vocab 264 at p=4 (312 at p=16): `Shift_k` (k units of 1/p beat, k <= 4p), `Vel_v` directly before `On_p`, `Off_p`, `Tempo_x`; bars are assumed to be 4 beats (bar b starts at unit b*4p); decoded back to Bar/Pos/Pitch/Vel/Dur. Because a note-on/off stream cannot express some things, the data is normalized before encoding: (1) events at pos >= 4 beats move into the following bar (1.17% of events), (2) simultaneous same-pitch notes are merged (3.13% of notes at p=4; max velocity, max duration), (3) overlapping same-pitch notes end at the next onset of that pitch. Context 8192 tokens, batch 4 (32 bars ~ 2355 tokens at p=4 for song 001 vs 2655 REMI).
- Compound Word: one step per Bar / Note / Tempo event with fields (type, pos, tempo, pitch, vel, dur); input = sum of per-field embeddings; one softmax head per field, each (except `type`) conditioned on the step's type embedding; loss = sum over fields on the fields that exist for the step type; 689 steps for the first 32 bars of song 001 (REMI 2655 tokens). Context 4096 steps, batch 8.
Positive controls (all 896 songs, p=4 and p=16): CP round trip tokens -> steps -> tokens exact 896/896; MIDI-like exact on the normalized data 896/896; shifted mid-song 16-bar crops (both codecs) 80/80. Two codec bugs were found and fixed by these controls before any training: Vel tokens must sit directly before their own On token (all Vel tokens of a time step had been sorted ahead of all On tokens), and overlapping same-pitch notes need the truncation rule. Validation metric for these runs = NLL per REMI-equivalent token (total model NLL of a 32-bar window / the REMI token count of the same window), comparable with the REMI GPT-2 runs (smoke test 100 steps: MIDI-like 2.83, CP 2.17; REMI GPT-2 had 2.58).
Dev runs launched (tmux alt1, GPU 3, sequential): R1 = MIDI-like p=4 20000 steps, R2 = CP p=4 20000 steps (`work/runs/R1_midilike_p4_dev`, `R2_cp_p4_dev`). Generation script for both: next.
`src/gen_alt.py` (MIDI-like: HF generate with a stopping rule on the cumulative Shift units = n_bars x 4p; CP: step-wise KV-cache loop, type sampled first (Bar-logit bias supported), then every needed field from its type-conditioned head, "ignore" index never sampled; same CLI/output files as `generate.py uncond`; rejected pieces are resampled and counted in `restarts`); `sweep_uncond.py --gen_script gen_alt.py` drives it. Smoke test with throwaway 60-step checkpoints (deleted): both give 32-bar pieces that go through the unmodified eval_basic (numbers meaningless for untrained models). Dev runs R1 (MIDI-like) and R2 (CP) are training (GPU 3, sequential).

## 2026-10-10 — Final Task a model F1 (all 896 songs, p=16, dropout 0.3, 20000 steps, last.pt) + entropy matching on the dev model
F1 sweep (5 sampling configs x 3 seeds x 20 pieces, unmodified eval_basic; real p16 reference H1 2.118 / H4 2.617 / GS 0.962, 20.6 notes/bar; rows `results/final/results_F1_s*.jsonl`):
| sampling | H1 | H4 | GS | notes/bar | malformed |
|---|---|---|---|---|---|
| pure T1.0 | 2.169 +- 0.039 | 2.600 +- 0.031 | 0.963 +- 0.005 | 29.6 | 227 |
| pure + Bar bias 3 | 2.105 +- 0.016 | 2.594 +- 0.050 | 0.966 +- 0.005 | 23.4 | 157 |
| T1.0 top-p 0.95 | 2.191 +- 0.018 | 2.552 +- 0.018 | 0.988 +- 0.004 | 30.8 | 36 |
| T0.9 top-p 0.9 | 2.055 +- 0.040 | 2.411 +- 0.065 | 0.995 +- 0.002 | 29.4 | 0 |
| T1.2 top-k 5 | 2.237 +- 0.032 | 2.570 +- 0.014 | 0.959 +- 0.009 | 36.9 | 0 |
These agree with the dev model w2 (pure 2.194 / 2.636 / 0.963; bias 3 2.102 / 2.634 / 0.962), so the full-data retraining did not change the picture.
Entropy-histogram matching (`src/entropy_match.py`, DEV model w2 best ckpt, reference = the model's per-token entropy on the 101 real 32-bar crops of the 45 held-out songs: mean entropy 1.124, mean NLL 1.299; 20 pieces per config; `results/entropy/entropy_match_w2.json`; 40 configs = T {0.8,0.9,1.0,1.1,1.2} x top-p {0.9,0.95,0.98,1.0} x Bar bias {0,3}). Lowest JS divergence between the entropy histograms: T=1.0 top-p 1.0 (JS 0.0009 for both bias 0 and bias 3; sampled mean entropy 1.121 / 1.109), T=1.1 top-p 0.98 + bias 3 (0.0012; d|NLL| 0.095), T=1.1 top-p 0.98 (0.0013), T=1.2 top-p 0.95 (0.0021; d|NLL| 0.094), T=1.2 top-p 0.95 + bias 3 (0.0035; d|NLL| 0.019, the closest mean NLL). Conservative settings are far off (T 0.8 top-p 0.98: JS 0.126, sampled entropy 0.33; T=1.2 top-p 0.9: JS 0.0115). So the protocol selects untruncated sampling at T=1.0 (histogram criterion) or T=1.2 top-p 0.95 (mean-perplexity criterion) and rules out top-p <= 0.9; the Bar bias leaves the entropy statistics unchanged and is therefore chosen on the density / structure criteria instead (bias 3 matches density 23.4 vs 20.6 notes/bar; SI favors no bias, see the SI entries).
Queued: post1 (window `post1`, GPU 1): the same 5 x 3 sweep for F4 (p4), F5 (p4 + chord tokens), F6 (p4 + label smoothing) as each finishes training; R1/R2 (MIDI-like / CP dev runs) still training on GPU 3.

## 2026-10-10 18:10 JST — queue status, crash and fix
Finished full-data models (all 896 songs, `--val_frac 0`, last.pt): F1 (Task a, p16), F4 (p4), F5 (p4 + chord tokens), F6 (p4 + LS 0.1), F2 (bonus B chord-in-decoder p4, 40000 steps), F3 (stage-1 p4), FA1/FA2/FA5/FA6/FA7/FA8/FA10 (bonus A variants, 10000 steps). Running: FB1 (B no chord, step ~2300/20000), FB8 (B chord-in-decoder p16, step 33200/40000); queued behind them: FB2, FB5, FC4.
Protocol check (dev run `DEV_bA2_10k`, A L1 + chord in melody segment, 10000-step cosine schedule, 5% held-out split): min val NLL 0.9861 (the earlier 20000-step dev schedule had min 0.9946 at step 8500), so the 10000-step A schedule is at least as good as the dev checkpoint it replaces.
Crash found (not a result): the first launch of the MIDI-like (R1) and CP (R2) dev runs stopped with `ValueError: min() arg is an empty sequence` in `pitch_shift_range` when a random crop contained no notes (R1 at ~step 9000 with val NLL/REMI-token 1.525 @8500, R2 at ~step 4500 with 1.534 @4000; intermediate numbers not used). Fix: no shift for note-less crops; R1/R2 restarted from scratch (windows r1, r2, GPUs 2/3, 20000 steps).
Scoring launched on the final models: post1 (Task a sweeps for F4/F5/F6, GPU 1), pb1 (bonus sweeps FA2, FA1, FA8, FA5, last.pt, GPU 2), pb2 (bonus sweeps F2 chord given, F2 two-stage with F3 stage-1, FA6, FA10, FA7, GPU 3); 4 sampling configs x 3 seeds each, 10 prompts, unmodified eval_bonus.py.

## 2026-10-10 19:30 JST — FINAL (full-data) bonus models scored (best = last ckpt; 10 prompts x 16 bars; 3 seeds x 4 sampling configs; ranges over the 4 sampling configs of the 3-seed means; unmodified eval_bonus.py; rows `results/final/results_F*_full*.jsonl`)
Controls: ceiling CA 0.7625 (real piano); wrong-song piano CA 0.0125, OA 0.846 / 0.384 / 0.649 / 0.464; round-trip OA ceilings at p=4: 0.980 / 0.973 / 0.870 / 0.745.
| model (trained on all 896 songs) | chord input | CA | OA_P | OA_V | OA_D | OA_IOI |
|---|---|---|---|---|---|---|
| F2: B (T5 6+6), chord token in decoder, 40k steps | given (true labels) | 0.587-0.656 | 0.843-0.889 | 0.553-0.607 | 0.583-0.671 | 0.469-0.558 |
| F2 + F3 two-stage (stage-1 chords from the melody, free-running chord acc 0.273) | none | 0.183-0.231 | 0.846-0.848 | 0.575-0.622 | 0.620-0.671 | 0.476-0.528 |
| FA1: A L1, no chord | none | 0.146-0.179 | 0.847-0.860 | 0.534-0.572 | 0.642-0.675 | 0.470-0.533 |
| FA7: A L1, chord predicted as first token of the piano segment (chord acc 0.138-0.198) | none | 0.121-0.175 | 0.844-0.859 | 0.510-0.550 | 0.661-0.694 | 0.444-0.535 |
| FA2: A L1, chord in melody segment | given | 0.567-0.669 | 0.875-0.889 | 0.524-0.570 | 0.668-0.703 | 0.487-0.591 |
| FA5: A L0, chord in melody segment | given | 0.544-0.600 | 0.881-0.889 | 0.515-0.553 | 0.639-0.686 | 0.478-0.521 |
| FA6: A L2, chord in melody segment | given | 0.210-0.317 | 0.853-0.869 | 0.549-0.600 | 0.629-0.680 | 0.463-0.509 |
| FA8: A L1, chord forced in piano segment | given | 0.483-0.544 | 0.880-0.887 | 0.516-0.572 | 0.661-0.698 | 0.494-0.528 |
| FA10: A L2, chord forced in piano segment | given | 0.546-0.579 | 0.878-0.890 | 0.562-0.571 | 0.634-0.687 | 0.483-0.498 |
Same picture as the dev runs: (1) CA depends on the chord input (given 0.48-0.67 vs none 0.12-0.23); (2) lookahead 2 with the chord in the melody segment collapses (0.21-0.32) and recovers when the chord sits in the piano segment (0.55-0.58); (3) the two-stage melody-only pipeline is the best no-chord setting (0.18-0.23 vs 0.12-0.18); (4) OA_V 0.51-0.62, OA_D 0.58-0.70, OA_IOI 0.44-0.59 across all models.
Task a, F1 sweep: see the previous entry. Representation ablation: dev runs finished with val NLL per REMI-equivalent token R1 MIDI-like p4 = 1.2111 (best), R2 CP p4 = 1.2752 (best), vs REMI GPT-2 v2 (p4, dropout 0.3) 0.9523 (same 45 held-out songs; REMI measured per token of the REMI sequence, the others per REMI-equivalent token). Full-data versions R1f / R2f launched (GPUs 2/3, 20000 steps, `--val_frac 0`), p=16 bonus scoring pb3 (FB8 given chords + two-stage with FC4) on GPU 0.

## 2026-10-10 20:40 JST — FINAL (full-data) Task a ablation, p=4 models (3 seeds x 20 pieces per cell; unmodified eval_basic; real p=4: H1 2.118 / H4 2.617 / GS 0.887, 20.6 notes/bar; rows `results/final/results_F{4,5,6}_*`)
| model | sampling | H1 | H4 | GS | notes/bar | malformed |
|---|---|---|---|---|---|---|
| F4 (p4, CE) | pure | 2.122 +- 0.030 | 2.568 +- 0.039 | 0.919 +- 0.004 | 27.6 | 115 |
| F4 | pure + Bar bias 3 | 1.999 +- 0.081 | 2.545 +- 0.056 | 0.891 +- 0.009 | 22.1 | 844 |
| F4 | T1.0 top-p 0.95 | 2.123 +- 0.067 | 2.493 +- 0.055 | 0.968 +- 0.004 | 27.7 | 0 |
| F4 | T0.9 top-p 0.9 | 2.051 +- 0.030 | 2.421 +- 0.046 | 0.988 +- 0.007 | 29.1 | 0 |
| F4 | T1.2 top-k 5 | 2.221 +- 0.019 | 2.607 +- 0.009 | 0.895 +- 0.004 | 31.4 | 10 |
| F5 (p4 + chord tokens) | pure | 2.148 +- 0.024 | 2.532 +- 0.062 | 0.924 +- 0.005 | 28.6 | 94 |
| F5 | pure + Bar bias 3 | 2.025 +- 0.005 | 2.491 +- 0.019 | 0.903 +- 0.008 | 22.4 | 79 |
| F5 | T1.0 top-p 0.95 | 2.155 +- 0.021 | 2.511 +- 0.035 | 0.968 +- 0.007 | 31.0 | 139 |
| F5 | T0.9 top-p 0.9 | 2.037 +- 0.043 | 2.358 +- 0.047 | 0.989 +- 0.005 | 28.1 | 0 |
| F5 | T1.2 top-k 5 | 2.191 +- 0.025 | 2.586 +- 0.034 | 0.905 +- 0.003 | 30.0 | 12 |
| F6 (p4, label smoothing 0.1) | pure | 2.012 +- 0.025 | 2.785 +- 0.018 | 0.721 +- 0.014 | 16.0 | 28634 |
| F6 | pure + Bar bias 3 | 1.397 +- 0.036 | 2.232 +- 0.041 | 0.873 +- 0.007 | 5.7 | 8042 |
| F6 | T1.0 top-p 0.95 | 2.030 +- 0.035 | 2.571 +- 0.035 | 0.840 +- 0.009 | 25.1 | 5336 |
| F6 | T0.9 top-p 0.9 | 2.102 +- 0.039 | 2.458 +- 0.005 | 0.991 +- 0.002 | 28.5 | 0 |
| F6 | T1.2 top-k 5 | 2.214 +- 0.019 | 2.604 +- 0.010 | 0.897 +- 0.004 | 30.7 | 377 |
Same pattern as the dev models: Bar bias 3 gives real-like density (22.1-22.4) and, at p=4, real-like GS (0.891 / 0.903 vs 0.887) without a clear change in H4; chord tokens do not change H4/GS clearly (F5 vs F4 within the seed sd); label smoothing breaks free sampling (up to 28,634 malformed tokens in 60 pieces). F1 (p16) vs F4 (p4) at bias 3: H1 2.105 vs 1.999, H4 2.594 vs 2.545, GS 0.966 vs 0.891 (each vs its own real GS 0.962 / 0.887).
Status: R1f / R2f (full-data MIDI-like / CP) trained; their Task a sweeps run in window post2 (GPU 1; MIDI-like: 4 configs, no Bar bias; CP: 5 configs incl. Bar bias); FB1 (B no chord) done, FB2 (encoder chord) training, FB5 (predicted chord) queued; bonus sweeps for FB1/FB2/FB5 queued in pb4 (GPU 2).
Note (20:55 JST): the first launch of the p=16 bonus scoring (window pb3) never ran (a shell variable was lost in the launch command; no rows were produced); relaunched with explicit paths (FB8 given chords, then FB8 + FC4 two-stage).

## 2026-10-10 23:30 JST — FINAL (full-data) representation ablation, bonus p16 and B ablations
Representation ablation, Task a (all 896 songs, 8L x 512 x 8h, dropout 0.3, 20000 steps, p=4 grid, 3 seeds x 20 pieces, unmodified eval_basic; real p4: H1 2.118 / H4 2.617 / GS 0.887, 20.6 notes/bar; rows `results/final/results_R{1,2}f_*`). MIDI-like has no Bar-bias analogue (4 configs).
| representation | sampling | H1 | H4 | GS | notes/bar | malformed |
|---|---|---|---|---|---|---|
| REMI (F4) | pure | 2.122 +- 0.030 | 2.568 +- 0.039 | 0.919 +- 0.004 | 27.6 | 115 |
| REMI (F4) | pure + Bar bias 3 | 1.999 +- 0.081 | 2.545 +- 0.056 | 0.891 +- 0.009 | 22.1 | 844 |
| REMI + chord tokens (F5) | pure | 2.148 +- 0.024 | 2.532 +- 0.062 | 0.924 +- 0.005 | 28.6 | 94 |
| MIDI-like (R1f) | pure | 1.917 +- 0.041 | 2.141 +- 0.071 | 0.837 +- 0.024 | 28.1 | 0 |
| MIDI-like | T1.0 top-p 0.95 | 1.727 +- 0.031 | 1.788 +- 0.029 | 0.883 +- 0.012 | 31.2 | 0 |
| MIDI-like | T0.9 top-p 0.9 | 1.627 +- 0.068 | 1.647 +- 0.073 | 0.940 +- 0.003 | 30.4 | 0 |
| MIDI-like | T1.2 top-k 5 | 1.876 +- 0.094 | 2.029 +- 0.101 | 0.844 +- 0.013 | 30.7 | 0 |
| Compound Word (R2f) | pure | 2.099 +- 0.028 | 2.566 +- 0.022 | 0.825 +- 0.016 | 19.4 | 56 |
| CP | pure + Bar bias 3 | 1.729 +- 0.053 | 2.322 +- 0.027 | 0.869 +- 0.010 | 10.3 | 0 |
| CP | T1.0 top-p 0.95 | 2.154 +- 0.041 | 2.491 +- 0.024 | 0.902 +- 0.012 | 27.3 | 0 |
| CP | T0.9 top-p 0.9 | 2.046 +- 0.031 | 2.277 +- 0.051 | 0.959 +- 0.004 | 33.8 | 0 |
| CP | T1.2 top-k 5 | 2.053 +- 0.017 | 2.396 +- 0.023 | 0.829 +- 0.005 | 20.6 | 408 |
Positive control for the MIDI-like codec (`src/ctrl_midilike_real.py`): the 1864 real 32-bar crops pushed through the MIDI-like normalization + encode/decode and scored by the unmodified eval_basic give H1 2.1259 / H4 2.6211 / GS 0.8903 (real 2.118 / 2.617 / 0.887), so the poor MIDI-like generation scores are due to the model, not the representation codec. Dev validation (NLL per REMI-equivalent token, 45 held-out songs): REMI GPT-2 0.952, MIDI-like 1.211, CP 1.275.
Reading: with the same backbone, size and training budget, REMI is closest to the real data on H1/H4/GS; MIDI-like loses pitch-class entropy (H4 1.65-2.14 vs 2.62) and rhythmic consistency; CP at T=1.0 is close to the real density (19.4 notes/bar) and H4 (2.566) but has lower GS (0.825 vs 0.887), and the REMI-style Bar bias (a logit shift on the CP type head) over-sparsifies it (10.3 notes/bar).
Bonus p=16 and B ablations (full data, last.pt; 10 prompts; 3 seeds per config; ranges over the 4 sampling configs):
| model | chord input | CA | OA_P | OA_V | OA_D | OA_IOI |
|---|---|---|---|---|---|---|
| FB8: B, chord in decoder, p=16, 40k | given | 0.442-0.544 | 0.877-0.889 | 0.575-0.612 | 0.686-0.722 | 0.511-0.551 |
| FB8 + FC4 two-stage (p=16 stage 1, free-running chord acc 0.307) | none | 0.200-0.235 | 0.853-0.861 | 0.579-0.625 | 0.698-0.727 | 0.536-0.578 |
| FB1: B, no chord, 20k | none | 0.131-0.148 | 0.829-0.862 | 0.549-0.622 | 0.636-0.729 | 0.427-0.505 |
| FB2: B, chord in encoder, 20k | given | 0.156-0.204 | 0.787-0.866 | 0.527-0.604 | 0.614-0.720 | 0.403-0.524 |
(FB5 predicted-chord B is still being scored.) p=16 vs p=4 for the chord-given B model (F2 CA 0.587-0.656, OA_D 0.583-0.671, OA_IOI 0.469-0.558): p=16 has higher OA_D (0.686-0.722) and OA_IOI (0.511-0.551) but lower CA (0.442-0.544); the encoder-chord B model (FB2) stays at 0.16-0.20, i.e. it does not use the chord input effectively, as in the dev runs.

## 2026-10-11 00:00 JST — infer.py tested end to end in a CLEAN environment with the final full-data checkpoints
Clean env `hw2_clean` (Python 3.10) built from the README steps only (torch 2.6.0+cu124 wheel, `pip install -r requirements.txt`, conda-forge fluidsynth): imports OK (torch 2.6.0+cu124, transformers 5.18.0, numpy 2.2.6). Package dir `pkg_test/` = infer.py + src + scripts + vocab + requirements + `checkpoints/{uncond_final.pt (F1 last.pt, 110 MB), bonus_final.pt (F2 last.pt, 177 MB), bonus_stage1.pt (F3 last.pt, 30 MB)}`.
Results: `infer.py uncond` (20 pieces, seed 0, default config cfg2 = pure + Bar bias 3), `infer.py continue` (3 prompts x 3 configs), `infer.py bonus --mode melody_only` and `--mode chord_given` all ran to completion and wrote .mid + .wav. One problem found and fixed: `bonus` imports the TA's `scripts/eval_bonus.py` (via `bonus_data.py`), which needs pandas and tqdm at import; they were only in `requirements-eval.txt` and the bonus mode failed in the clean env -> moved to `requirements.txt` (README install steps unchanged otherwise).
Reproducibility check (unmodified TA evaluators on the files written by infer.py vs the experiment-sweep rows with the same config and seed 0): bonus melody-only: CA 0.2313 / OA 0.8370 / 0.5369 / 0.5974 / 0.4426 = sweep row (F2 + F3, T0.9 top-p 0.9, seed 0) identical to 4 decimals; bonus chord-given: CA 0.6750 / 0.8910 / 0.5994 / 0.6396 / 0.4802 = sweep row identical; Task a (20 pieces, pure + Bar bias 3, seed 0): H1 2.1166 / H4 2.6479 / GS 0.9592 = sweep row F1 identical. (eval_basic needs MusDr on PYTHONPATH; the TA provides it.)
Default final configs written into the package: Task a/b model F1 (p16, all 896 songs); Task a default config = pure + Bar bias 3 (pending the user's choice vs pure, see SI/density trade-off); bonus melody_only = F2 + F3 two-stage, T0.9 top-p 0.9 (CA 0.229 +- 0.008 over 3 seeds); bonus chord_given = F2, T0.9 top-p 0.9 (CA 0.656 +- 0.045).

## 2026-10-11 FB5 (arch B, chord mode 4: decoder generates chords, full data, last.pt)
Config: ckpt work/runs/FB5_B_predchord_full/last.pt (sha b7769c91e181670a), seeds 0-2 x 10 prompts, rows in results/final/results_FB5_B_predchord_full.jsonl.
CA mean over 3 seeds: T1.0 pure 0.140; T1.0 top_p0.95 0.127; T0.9 top_p0.9 0.163; T1.2 top_k5 0.129. Predicted-chord accuracy 0.14-0.18.
Measured: comparable to FB2/FB1 band (0.13-0.20), below two-stage (0.18-0.23) and chord-given (0.55-0.67).
Task a decision (user asked to pick the best performer): default switched to pure (cfg1), closer to real on H4 and GS and on dev SI; bias3 kept as ablation.

## 2026-10-11 in-attention chord embedding (arch A), launched
Implementation: `bonus_lib.ChordEmbGPT2` adds a learned embedding of the most recent Chord_x token in the sequence (forward-filled) to every token embedding; train_bonus `--chord_emb 1`; gen_bonus loads it and generates with use_cache=False (the chord is recomputed from input_ids). Smoke test: 60 steps run.
Jobs (full data, `--val_frac 0`, 10000 steps, dropout 0.3, lookahead 1, seed 0, last.pt, GPU 1, tmux `ce`): FA11 = chord 3 (chord token given in melody and piano segments, no embedding; control), FA12 = chord 3 + chord_emb 1. Then sweep_bonus (4 sampling configs x 3 seeds) for both.

## 2026-10-11 in-attention chord embedding scored (arch A, p4, full data, 10000 steps, dropout 0.3, L1, last.pt, seed 0 train; 4 sampling configs x 3 gen seeds x 10 prompts; eval_bonus unmodified; rows results/final/results_FA11_*.jsonl, results_FA12_*.jsonl)
Chord given (annotation), chord mode 3 (chord token in both the melody and the piano segment). FA11 = no embedding (control), FA12 = + `ChordEmbGPT2` (learned embedding of the latest Chord_x token added to every token's embedding). CA-ceiling 0.762.
| sampling | FA11 CA | FA12 CA | FA12 DP / DV / DD / DIOI |
|---|---|---|---|
| T1.0 pure | 0.494 | 0.629 | 0.885 / 0.573 / 0.690 / 0.472 |
| T1.0 top_p 0.95 | 0.579 | 0.660 | 0.885 / 0.558 / 0.666 / 0.519 |
| T0.9 top_p 0.9 | 0.583 | 0.646 | 0.890 / 0.552 / 0.683 / 0.511 |
| T1.2 top_k 5 | 0.546 | 0.631 | 0.886 / 0.579 / 0.677 / 0.492 |
Measured: +0.063 to +0.135 CA over the control per sampling config. FA12 (0.629-0.660) is in the same band as FA2 (0.567-0.669) and F2 (0.587-0.656); OA unchanged within ~0.03. Single training seed each, so the difference between FA12 and FA2/F2 is not resolved.
Round-2 research (round2_research response D, round2_research response B; read, claims not verified by us) turned into the queue below. Qwen: unverifiable citations (one arXiv id looks malformed), generic. Gemini: has T5 relative-position bucket advice (buckets 128, max distance 1024, d256 4+4), best-of-N rerank, pitch-only repetition penalty, RoPE/ALiBi for 4k contexts.
Queue (each logged with config): FB9 = B chord-in-decoder, rel buckets 128 / max dist 1024 (40000 steps); FB10 = B chord-in-decoder d256 4+4 (40000 steps) [tmux rb, GPU 2]; best-of-8 by agreement with the conditioning chords (gen_bonus --best_of 8; given-chord F2 and two-stage F2+F3, T0.9 p0.9, 3 seeds) [tmux bon, GPU 1]. Note: best-of-N uses the chord estimator of the evaluation script against the same chord labels the metric scores (given mode) or against stage-1 predictions (two-stage); this is metric-guided selection and is reported as a separate labeled variant, not the headline.

## 2026-10-11 queue update (no results yet)
- F7 = Task a with a llama-style decoder (RoPE, RMSNorm, SwiGLU, tied embeddings, attention dropout 0.3 only, intermediate 1408; same 8L x 512 x 8h, ctx 4096, p16, 20000 steps, full data, last.pt), then 3-seed x 5-config sweep (`src/run_rope.sh`). Confound: the architecture family changes, not only the position encoding. Queued on GPU 1 behind best-of-8 (tmux `rope`). GPU 0 and 3 are not used (known-broken per project notes).
- Running: FB9 then FB10 (GPU 2, tmux `rb`), best-of-8 (GPU 1, tmux `bon`).
- Still to queue after these: SI on the final F1 sets, pitch-only repetition penalty (Task a / bonus), chord-conditioned logit bias (chord-given only).

## 2026-10-11 best-of-8 and F7 (RoPE) scored
Best-of-8 (gen_bonus --best_of 8, T0.9 top_p0.9, seeds 0-2 x 10 prompts, bonus_final F2 last.pt, p4; rows results/final/results_F2_bestof8.jsonl, results_F2F3_bestof8.jsonl). Candidates are ranked by the fraction of bars whose estimated chord (eval_bonus.estimate_chord) equals the conditioning chord of that bar.
| variant | CA (3 seeds) | DP / DV / DD / DIOI (mean) | 
|---|---|---|
| chord given, best-of-8 | 0.925 / 0.9125 / 0.9125 (mean 0.917; CA-ceiling 0.7625) | 0.886 / 0.578 / 0.699 / 0.558 |
| two-stage (stage-1 chords), best-of-8 | 0.256 / 0.256 / 0.256 | 0.835 / 0.604 / 0.641 / 0.500 |
Single-sample references: F2 given T0.9 p0.9 CA 0.656; F2+F3 two-stage 0.183-0.231. Best-of-8 in given mode selects with the same labels the metric scores, so the number measures selection against the evaluation labels (CA above the real piano's own ceiling 0.7625 confirms that); it is reported as a separate variant. In two-stage mode the selected candidate agrees with the stage-1 chords in 100% of bars, so CA is bounded by stage-1 accuracy (0.256 here).
F7 = Task a llama-style decoder (RoPE), p16, 20000 steps, dropout 0.3, full data, last.pt, 3 seeds x 20 pieces per sampling config (rows results/final/results_F7_rope_p16_full_s{0,1,2}.jsonl); real data H1 2.118 / H4 2.617 / GS 0.887:
| sampling | H1 | H4 | GS | notes/bar (quarters mean) |
|---|---|---|---|---|
| T1.0 pure | 2.142 | 2.640 | 0.967 | 21.6 |
| T1.0 top_p0.95 | 2.107 | 2.580 | 0.974 | 21.5 |
| T0.9 top_p0.9 | 2.073 | 2.563 | 0.977 | 23.6 |
| T1.2 top_k5 | 2.154 | 2.613 | 0.965 | 23.7 |
| T1.0 + bar_bias3 | 2.082 | 2.600 | 0.967 | 19.4 |
F1 (GPT-2 learned absolute positions) pure: 2.169 / 2.600 / 0.963. Differences are small (H1 0.03, H4 0.04, GS 0.004) and not resolved with one training seed; architecture family also differs (RMSNorm, SwiGLU, attention dropout only).

## 2026-10-11 chord-triad logit bias, FB9/FB10, SI on final Task a sets
Chord-triad logit bias (gen_bonus --chord_bias B: +B on the logit of every Pitch_x whose pitch class is in the triad of the current bar's conditioning chord; applied when the previous token is a Pos_x). F2 last.pt (B, chord mode 2), T0.9 top_p0.9, seeds 0-2 x 10 prompts, p4; rows results/final/results_F2_chordbias.jsonl, results_F2F3_chordbias.jsonl (CA-ceiling 0.7625):
| bias | given chords CA | notes/bar | DP/DV/DD/DIOI | two-stage CA |
|---|---|---|---|---|
| 0 (F2 reference) | 0.656 | | | 0.183-0.231 (4 samplings) |
| 1 | 0.748 | 15.3 | 0.884/0.594/0.649/0.469 | 0.233 |
| 2 | 0.871 | 16.4 | 0.892/0.624/0.649/0.514 | 0.250 |
| 4 | 0.921 | 16.5 | 0.895/0.604/0.651/0.516 | 0.248 |
Measured: given-chord CA rises with the bias (0.75-0.92), OA stays within about 0.03-0.06 of the unbiased runs. Two-stage CA is bounded by the stage-1 predictions (0.23-0.25).
FB9 = arch B chord-in-decoder with T5 relative buckets 128 / max distance 1024 (40000 steps, full data, last.pt; results_FB9_*.jsonl); FB10 = same with d256, 4+4 layers, 4 heads, d_ff 1024 (results_FB10_*.jsonl). CA by sampling (T1.0 pure / T1.0 p0.95 / T0.9 p0.9 / T1.2 k5): FB9 0.573 / 0.610 / 0.610 / 0.498; FB10 0.567 / 0.660 / 0.631 / 0.529; reference F2 (32 buckets, 6+6, d512) 0.587-0.656 for its four configs. No resolved difference with one training seed each.
SI of the final Task a sets (seed 0, 20 pieces each, p16 F1 / F7, same SI code as before; results/si_final/*.json; real data 100 crops short/mid/long = 0.369/0.376/0.344):
| set | SI short | mid | long |
|---|---|---|---|
| F1 pure | 0.330 | 0.290 | 0.225 |
| F1 bar_bias 3 | 0.319 | 0.271 | 0.210 |
| F1 T0.9 top_p0.9 | 0.657 | 0.692 | 0.635 |
| F1 T1.0 top_p0.95 | 0.528 | 0.517 | 0.449 |
| F7 (RoPE) pure | 0.287 | 0.257 | 0.212 |
Measured: pure is closest to real among the F1 configs on all three SI scales; truncated sampling (p0.9, p0.95) gives far higher SI than real. This supports the pure default for Task a.

## 2026-10-11 pitch-only repetition penalty, Task a (F1 last.pt, p16, T1.0, 3 seeds x 20 pieces; rows results/final/results_F1_reppen_s{0,1,2}.jsonl; sampling file results/sampling_taska_reppen.json)
Penalty: at pitch-choice steps, logit of pitch p -= alpha x (occurrences of p among the last 128 tokens); `generate.py --rep_alpha`. (First launch had a bug, penalty applied to the wrong token type, results identical to pure sampling; deleted and rerun after the fix.) Real data H1 2.118 / H4 2.617 / GS 0.887; F1 pure 2.169 / 2.600 / 0.963.
| alpha | H1 | H4 | GS | malformed (60 pieces) |
|---|---|---|---|---|
| 0.5 | 2.419 | 2.914 | 0.961 | 219 |
| 1 | 2.576 | 3.079 | 0.965 | 236 |
| 2 | 2.849 | 3.359 | 0.962 | 575 |
Measured: the penalty raises H1 and H4 far above the real values (0.30-0.73 above for H1) with no change in GS, so it moves the pitch entropy away from real data; not used. No planned experiment remains unrun.

## 2026-10-12 External review of REVIEW_PACKAGE.md (three web LLMs; their text is local-only: review_research response B, review_perplexity.md, review_research response D)
Benchmarks: none of the three found a published number measured with the TA scripts. Perplexity says so explicitly and cites only arXiv 2408.01696 (unconditional, POP909, pitch-class entropy/groove consistency, numbers not retrievable) and 2410.08435 (melody-to-accompaniment, POP909, different CA/OA definitions). Gemini's published numbers (CP-Transformer, Pop Music Transformer, PopMAG, AccoMontage) are all marked "unverified" by itself and are not used. Qwen cites PopMAG CA 0.647 / OA 0.602 and D3PIA CA 80.1% / GS 82.1% but compares them as "apples-to-apples" despite different metric definitions (PopMAG is on LMD/FreeMidi/CPMD and not the TA estimator): not used as a benchmark. Statement for the report: no published number was verified under the TA metric implementation.
Points all three agree on: F1 H4/GS close to real (plausible); SI weaker than real; melody-only CA mediocre; best-of-N and triad bias exceed the real-piano ceiling and must stay labeled metric-guided; training-seed variance unmeasured; no blind listening.
Extra points: Perplexity flags notes/bar 29.6 (pure) vs 20.6 real (44% too dense); Qwen/Gemini/Perplexity propose a chord auxiliary loss (untried); Perplexity: multiple training seeds first; Gemini: classifier-free conditioning dropout, velocity/micro-timing augmentation, bar-relative positions; Qwen: mixed-grid training, learned (non-metric) reranker; Perplexity: key-aware bias, next-bar structural objective, blind pairwise listening by classmates.
Queue from the review (run one by one): (1) training-seed replicates (seeds 1, 2) of F1, FA12, F2 [started: tmux sd1 GPU 1 = F1 then FA12, tmux sd2 GPU 2 = F2; `src/seeds_runs.sh`]; (2) chord auxiliary loss for bonus A; (3) conditioning dropout + classifier-free guidance for bonus; (4) velocity/micro-timing augmentation for Task a; (5) learned non-metric reranker; (6) key-aware pitch bias.

## 2026-10-12 chord auxiliary loss (bonus A) implemented, queued
`train_bonus.py --chord_aux W`: a linear head (25 classes: 24 triads + no label) on the final hidden state of every token of the piano segment (the <P> tag and the piano tokens of bar i) is trained to predict bar i's chord; loss = main + W x mean aux CE x (number of main-loss tokens); the head is dropped at inference (gen_bonus ignores `chord_head.*`). The model never receives chord input. Jobs: FA13 (W 0.3) and FA14 (W 1.0): A, lookahead 1, no chord input, full data, 10000 steps, dropout 0.3, seed 0, last.pt; control = FA1 (identical without aux; CA 0.146-0.179). Queued behind the seed replicates on GPU 1 (tmux `aux`, `src/aux_runs.sh`).

## 2026-10-12 training-seed replicates + chord auxiliary loss (results in results/final/)
Configs: same as the seed-0 runs (F1: p16 gpt2 8L×512 20000 steps dropout 0.3; FA12: arch A L1 chord-both + chord_emb 10000 steps; F2: arch B 6+6 chord 2, 40000 steps), training seeds 1 and 2; sampling-seed 0-2 means, full config in each jsonl row.
- F1 pure (H1/H4/GS), train-seed 0: 2.169/2.600/0.963; seed 1: 2.200/2.646/0.964; seed 2: 2.174/2.631/0.966. Bias 3: seed 1 2.109/2.606/0.964, seed 2 2.040/2.571/0.968.
- FA12 given-chord CA (pure / p0.95 / T0.9p0.9 / T1.2k5): seed 1 0.669/0.688/0.754/0.640; seed 2 0.575/0.606/0.627/0.633 (seed 0 earlier: 0.629–0.660 range). Training-seed spread is ~0.1 CA, comparable to differences between some variants.
- F2 (arch B, given chords): seed 1 0.604/0.660/0.725/0.690; seed 2 0.625/0.623/0.696/0.602.
- Chord aux loss, melody-only A L1 (CA pure/p0.95/T0.9/T1.2k5): FA13 (w0.3) 0.165/0.173/0.142/0.138; FA14 (w1.0) 0.150/0.162/0.135/0.158. Control FA1 0.146–0.179. No measurable gain; aux loss does not help melody-only CA.
Next: conditioning dropout + CFG, velocity/micro-timing augmentation, learned reranker, key-aware bias.

## 2026-10-12 conditioning dropout + CFG implemented, queued (FA15)
train_bonus.py `--chord_drop p`: with prob p all chord labels of a training example become Chord_N. gen_bonus.py `--cfg w`: CFG LogitsProcessor (logp_u + w(logp_c - logp_u), u = same prefix with all chord tokens -> Chord_N, no cache). FA15 = FA12 recipe (A L1 chord 3 + chord_emb, 10000 steps, dropout 0.3) + chord_drop 0.15, sweep at cfg 1.0/1.5/2.0/3.0 (tmux cfg, GPU 1, src/cfg_runs.sh). Note: HF passes the processor scores possibly after temperature; guidance is applied to whatever scores arrive (interpretation caveat).

## 2026-10-12 velocity augmentation for Task a queued (F8)
train_uncond.py `--vel_aug 2`: per crop shift all Vel tokens by random k in [-2,2] bins (4 MIDI-vel units/bin), plus ±1-bin jitter on 30% of notes. F8 = F1 protocol (p16, gpt2 8L×512, 20000 steps, dropout 0.3, full data, seed 0) + vel_aug 2; 3-seed sweep; tmux va, GPU 2, src/velaug_runs.sh. Compare against F1 seed 0-2 (H1/H4/GS 2.169/2.600/0.963 pure). Micro-timing augmentation not implemented (grid tokens; would break Pos alignment). FA15 (CFG) running on GPU 1.

## 2026-10-12 results: FA15 (chord dropout 0.15 + CFG) and F8 (velocity aug)
Configs as logged above; means over sampling seeds 0-2 × 10 bonus prompts; full config per row in results/final/results_FA15_*_cfg{1.0,1.5,2.0,3.0}.jsonl and results_F8_velaug2_p16_full_s*.jsonl.
FA15 given-chord CA / OA pitch / vel / dur / IOI (pure; p0.95; T0.9p0.9; T1.2k5 for CA):
- cfg 1.0: CA 0.558/0.598/0.637/0.538; pure OA 0.878/0.562/0.673/0.530
- cfg 1.5: CA 0.677/0.744/0.719/0.704; pure OA 0.884/0.576/0.676/0.541
- cfg 2.0: CA 0.760/0.771/0.744/0.777; pure OA 0.888/0.538/0.718/0.523
- cfg 3.0: CA 0.883/0.835/0.883/0.838; pure OA 0.890/0.526/0.693/0.489
CFG uses only the given chords (no metric-based selection). CA rises monotonically with scale; OA pitch/vel/dur/IOI stay in the same range (vel/IOI drift down a little at 3.0, single prompt set, differences of ~0.03 are within seed noise as seen in training-seed replicates). Without guidance the dropout-trained model is 0.54-0.64, i.e. dropout itself is not the gain. Notes/bar 15-19 (real piano density not re-measured here).
F8 vel_aug 2, Task a (H1/H4/GS, mean of 3 sampling seeds): pure 2.203/2.582/0.957; p0.95 2.212/2.509/0.991; T0.9p0.9 2.171/2.489/0.996; T1.2k5 2.248/2.556/0.955; bias3 2.123/2.573/0.960. Reference F1 pure 2.169/2.600/0.963 (train-seed spread H4 2.600–2.646). Within training-seed spread: no measurable effect.

## 2026-10-12 queued: higher CFG scales, key-aware pitch bias
- cfghi (GPU 1): FA15 sweep at cfg 4.0 and 6.0 (same 4 sampling configs × 3 seeds), src/cfg_hi.sh.
- kb (GPU 2): melody-only key bias on FA1 (A L1, no chord input, 10000 steps): `--key_bias b` adds b to the logit of pitches in the diatonic scale of the key estimated (Krumhansl-Schmuckler) from the melody prompt only; b = 1, 2, 4, T0.9 top_p0.9, seeds 0-2, src/sweep_keybias.sh. Uses no chord labels, so it is not metric-guided. Control: FA1 same sampling without bias.

## 2026-10-12 results: FA15 CFG scales 4/6, melody-only key bias
Configs as queued above; means over 3 sampling seeds × 10 prompts (rows in results/final/). Given-chord CA / pitch / vel / dur / IOI OA.
FA15 cfg 4.0: pure 0.887 (OA 0.881/0.578/0.701/0.493), p0.95 0.896 (0.891/0.546/0.732/0.547), T0.9p0.9 0.896 (0.887/0.528/0.699/0.509), T1.2k5 0.858 (0.882/0.545/0.730/0.507). (One extra pure row 0.844 was written by an earlier version of gen_bonus.py before the key_bias field existed; the pure mean above is the post-edit rows.)
FA15 cfg 6.0: pure 0.863 (0.857/0.512/0.660/0.507), p0.95 0.898, T0.9p0.9 0.875, T1.2k5 0.890. Saturates near 0.88–0.90 from cfg 3–4; at 6 pitch OA starts to drop (0.857). Full curve (pure): cfg1 0.558, 1.5 0.677, 2 0.760, 3 0.883, 4 0.887, 6 0.863.
Key bias (FA1, melody-only, T0.9 p0.9, estimated key from melody prompt only): b=1 CA 0.173, b=2 0.179, b=4 0.140; control FA1 0.177. No effect on CA (OA unchanged: pitch 0.85–0.86). Key bias does not help.

## 2026-10-12 result: two-stage (F3 predicted chords) + FA15 + CFG, melody-only pipeline
FA15 last.pt, chords from F3 (arch C, stage 1, greedy), T0.9 top_p0.9, seeds 0-2 × 10 prompts, scored against real chords: cfg 1.0 CA 0.183 (OA 0.853/0.574/0.678/0.536); cfg 3.0 CA 0.231 (0.842/0.541/0.696/0.496). Guidance gives +0.05 CA but stays at the melody-only level of the earlier two-stage pipeline (~0.23): the bottleneck is stage-1 chord accuracy, not the conditioning strength. Rows: results/final/results_FA15_*_twostage.jsonl.

## 2026-10-12 final deliverable generation started (infer.py, seed 0)
Bonus release config changed: `chord_given` now = FA15 (arch A, chord dropout 0.15, 10000 steps, dropout 0.3) + CFG scale 3.0, T0.9 top_p0.9 (measured CA 0.883, 3 sampling seeds; checkpoint bonus_chord_final.pt); `melody_only` unchanged = F2 + F3 stage-1 (vocab/bonus_final.json). Server checkpoints/ = symlinks to last.pt of F1 (uncond_final), F2 (bonus_final), F3 (bonus_stage1), FA15 (bonus_chord_final). tmux fg (GPU 1) runs src/final_gen.sh: infer.py uncond (cfg1 pure, 20 pieces), continue (3 prompts × 3 configs, F1 checkpoint), bonus melody_only, bonus chord_given -> work/deliver/. Logs work/deliver.*.log.

## 2026-10-12 deliverables generated via infer.py (seed 0) and scored with the TA scripts
work/deliver/: taskA (20 .mid + 20 .wav, cfg1_T1.0_pure, F1), taskB (3 prompts × 3 configs = 9 .mid/.wav), bonus_mo (10, melody_only), bonus_cg (10, chord_given); 49 .mid / 49 .wav in total.
- Task a (eval_basic, 20 pieces, seed 0): H1 2.2122, H4 2.6326, GS 0.9595 (sampling-seed range of F1 pure: H4 ~2.60–2.65).
- Bonus melody_only (F2 + F3, T0.9 p0.9): CA 0.2313 (ceiling 0.7625), OA pitch/vel/dur/IOI 0.8370/0.5369/0.5974/0.4426.
- Bonus chord_given (FA15, cfg 3.0, T0.9 p0.9): CA 0.8438, OA 0.8813/0.5206/0.6803/0.5136.
Note: the chord_given CA exceeds the "CA-ceiling" 0.7625 (ceiling = CA of the real piano accompaniment re-estimated by the same chord estimator): guided generation agrees with the labels more clearly than the real piano does, so CA > ceiling means stronger agreement with the chord labels, not better music; to be said in the report. csv in results/deliver/.

## 2026-10-12 report deck built (local, report/ is not in the repo whitelist)
report/build_deck.py (python-pptx) -> report/r05921008_report.pptx -> LibreOffice PDF `report/r05921008_report.pdf`, 12 slides 16:9, figures from report/make_figures.py (new cfg_curve.png from results/final). All numbers copied from the logged tables (WORKLOG / REVIEW_PACKAGE). Open placeholders: cloud-drive link on slide 1, Task b best-per-prompt (listening) on slide 7; student ID presumed r05921008.

## 2026-10-12 student ID confirmed, upload package assembled
Student ID is R13921031 (r05921008 was only the git e-mail): report renamed R13921031_report.pdf/pptx. Upload package (local): /Users/chun-feitan/Desktop/CommE5070/PA2/upload_package (600 MB): code (README.md, requirements*.txt, infer.py, src/, scripts/, vocab/, patches/, results/, docs/), checkpoints/ (4 weights-only .pt, 414 MB), generated/ (taskA 20 mid+wav, taskB 9 mid+wav in 3 config folders, bonus_melody_only 10, bonus_chord_given 10), R13921031_report.pdf. No training/preprocessed data, no caches. Cloud link still to be created and inserted on slide 1.

## 2026-10-12 TA reply on chord labels
TA (Bo-Rui Chen): per-bar chord labels (chord_midi.txt) may be used as model conditions; higher CA with labels is expected, "compare your two models to see whether the results meet the assumption"; for the submitted files "you can choose the ones that sound better in your opinion". Actions: chord-given (FA15 + CFG w3) is now the main bonus submission, melody-only (F2+F3) kept as the comparison model; assumption check: CA 0.844 vs 0.231 on the submitted seed-0 sets (T0.9 top-p0.9), holds. infer.py bonus default mode = chord_given. Report slides 8-11 updated (caveat bullet replaced). Submitted-file choice is free: the user may swap in other seeds/configs by listening.

## 2026-10-12 course announcement: POP909 time signatures
Announcement: the time signatures in the POP909 .mid files are mostly wrong, so MidiTok users must take the time signature from beat_midi.txt and write it back into the .mid before tokenizing. Not affected: we do not use MidiTok; all tokenization goes through the TA's tokenization.py (bar/beat structure read from each song's beat_midi.txt, line ~110) via our wrapper tokenize_hw2.py, and the bonus pipeline (bonus_data.py, gen_bonus.py) uses beat_midi.txt/chord_midi.txt for bars and grids; no .mid time-signature field is read anywhere in src/ or scripts/ (grep for time_signature/miditok: no hits). The MIDI-like and Compound Word ablations (alt_repr.py) are our own code on the same beat-derived bar grid. A one-line statement is added to the report (slide 3).

## 2026-10-12 MidiTok decision
User decision: MidiTok goes at the very end of TODO (optional, after all deliverables are final). Rationale logged in TODO: same REMI family as the TA tokenizer, no measured evidence of gain, risk of breaking the unmodified eval scripts.

## 2026-10-12 Task b: objective proxies for the continuations (src/taskb_metrics.py, run on the server; the spec defines no Task b metric)
Per file from the token json of the delivered set (seed 0, F1 checkpoint), bars 9-32 vs the 8-bar prompt: notes/bar, pitch-class histogram cosine to the prompt (pcCos), onset-position histogram cosine (rhythCos), out-of-key rate (pitch classes outside the prompt's 7 most used), repeated-bar rate (continuation bar identical in (Pos,Pitch) content to an earlier bar). Note: the prompt's 8th bar ends at Pos_32 and the model completes it, so "prompt/bar" is computed on bars 1-8 of the generated file (differs slightly between configs).
| file | prompt/bar | cont/bar | pcCos | rhythCos | outKey | repBar |
| 001 cfg1 | 16.5 | 28.7 | 0.924 | 0.925 | 0.029 | 0.000 |
| 001 cfg2 | 15.5 | 14.6 | 0.950 | 0.957 | 0.000 | 0.000 |
| 001 cfg3 | 16.1 | 20.9 | 0.902 | 0.856 | 0.042 | 0.417 |
| 002 cfg1 | 23.0 | 31.8 | 0.925 | 0.446 | 0.076 | 0.000 |
| 002 cfg2 | 22.1 | 23.6 | 0.971 | 0.941 | 0.014 | 0.000 |
| 002 cfg3 | 22.1 | 24.6 | 0.988 | 0.909 | 0.002 | 0.000 |
| 003 cfg1 | 23.2 | 36.0 | 0.949 | 0.769 | 0.005 | 0.000 |
| 003 cfg2 | 22.8 | 31.0 | 0.877 | 0.802 | 0.008 | 0.000 |
| 003 cfg3 | 23.2 | 31.8 | 0.943 | 0.790 | 0.000 | 0.458 |
Reading (interpretation, proxies not quality): cfg2 stays closest to the prompt density and rhythm on 001/002; cfg3 loops (42-46% repeated bars on 001/003); cfg1 gets much denser than the prompt (28.7-36 vs 16.5-23) and on 002 drifts in rhythm (0.446). User's listening: cfg1 or cfg3 sounds better than cfg2 (per-prompt choice pending).

## 2026-10-12 Task b picks (author listening, single rater)
Best per prompt: 001 cfg1, 002 cfg3, 003 cfg3 (the author had listened only to the 003 versions at first, then to 001/002). Slide 7 updated incl. the proxy-metric note (cfg3 repeats 42%/46% of bars on 001/003; picks differ from the proxy-metric favourite cfg2). survey_taskb/ (blind A/B/C) + SURVEY_TASKB.md were prepared but not used. PDF in upload_package/ updated; Drive copy to be replaced.

## 2026-10-12 optional items started (results recorded under results/optional/ in the repo)
F9: Task a, F1 protocol (p16, GPT-2, d512, 8 heads, dropout 0.3, 20000 steps, batch 8, all 896 songs, seed 0) with 12 layers (~42M params) instead of 8; 3-seed sweep with the same 5 sampling configs; tmux o12, GPU 1, src/opt_12L.sh. Compare with F1 (H1/H4/GS pure 2.169/2.600/0.963; training-seed spread of H4 ≈ 0.05). Next optional: BPE-style merging, then MidiTok.

## 2026-10-12 optional: BPE-merged tokens (F10) started
Method: src/bpe_data.py learns 1500 BPE merges (HF tokenizers; base tokens as single characters; Bar / Tempo_* / <BOS> / <EOS> isolated so they never merge) on 10,720 offline-transposed copies (-5..+6 semitones, within Pitch 21-108) of the 896 TRAINING songs only; result lossless (asserted join(tokens)==text). Vocab 545 -> 2045, sequence length 73.23M -> 43.28M tokens (1.69x shorter). Training: GPT-2 8L d512, F1 recipe (20000 steps, batch 8, dropout 0.3, ctx 4096, seed 0, --transpose 0 since transposition is in the data), tmux o13, GPU 2, src/opt_bpe.sh. Generation: src/gen_bpe.py (same sliding-window sampler as generate.py, merged tokens expanded back to TA tokens, scored with the unmodified eval_basic.py against tok2idx_p16.json); 4 sampling configs x 3 seeds x 20 pieces (no Bar-bias / typed sampler for this vocabulary). Confounds to state: each step sees 1.69x more music tokens than F1 (same steps/batch/ctx), BPE tokens are only decodable as whole units. Run log: work/opt_bpe.log. Parallel: F9 12-layer on GPU 1 (tmux o12).

## 2026-10-12 optional: MidiTok REMI (F11) started
src/miditok_data.py: MidiTok 3.0.6 REMI on the POP909 training songs. The .mid time signatures (e.g. 2/4 in song 001, constant 90 bpm) are rewritten from beat_midi.txt (per-bar beat counts via the TA get_beat_info incl. the pickup bar, denominator 4; notes/tempo shifted so tick 0 = first annotated beat; MELODY/BRIDGE/PIANO merged into one track as in the TA tokenizer). Config: pitch 21-108, 16 positions per beat (beat_res {(0,4):16,(4,12):8}), 32 velocities, tempo tokens, TimeSig tokens dropped from the stream (bars/positions carry the metre; as in the TA REMI). Token names renamed to the TA naming so train_uncond.py and the unmodified eval_basic.py work. 896 songs, 0 skipped, mean 5968 tokens per song (TA p16: 6833), vocab 477. Real-data reference in this tokenization (100 random real 32-bar crops, eval_basic): H1 2.192, H4 2.647, GS 0.978 (differs from the TA-token reference 2.118/2.617/0.962 because quantization differs; F11 is compared with its own reference). Training: GPT-2 8L d512, F1 recipe (20000 steps, batch 8, dropout 0.3, ctx 4096, transposition -5..+6, seed 0); tmux o14, GPU 1, src/opt_miditok.sh; sampling via src/gen_miditok.py (same sliding-window sampler, .mid decoded with MidiTok, 4/4 assumed); 4 sampling configs x 3 seeds x 20 pieces.
Course announcement context: this is the MidiTok experiment from TODO. Running concurrently: F9 (12 layers, GPU 1), F10 (BPE, GPU 2).
