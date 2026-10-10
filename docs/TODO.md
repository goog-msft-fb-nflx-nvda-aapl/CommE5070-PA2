# TODO — CommE5070 HW2

Deadline: 2026-10-26 23:59. Work one item at a time, log results/decisions in `WORKLOG.md` as you go.

## PRIORITY (added 2026-10-10 after lecture review, see WORKLOG)
- [~] FINAL models retrained on ALL 896 training songs (HW: use the whole training set): F1 (Task a/b, p16 dropout 0.3, 20k), F2 (bonus B chord-in-decoder p4, 40k), F3 (stage-1 chord predictor) running 2026-10-10; then regenerate Task a/b/bonus deliverables from these checkpoints (last.pt)
- [ ] Entropy-histogram / perplexity-matching selection of sampling params (Compose & Embellish; lecture 05 s78-79) for Task a/b/bonus; report how the configs were chosen
- [ ] Greedy decoding row in the Task a table
- [x] Representation ablations from the lectures: MIDI-like, CP-word (HW example table has a CP-word row); optional in-attention chord embedding for the bonus (FA12, see WORKLOG 2026-10-11)
- [ ] Test infer.py end to end in a clean env; package checkpoints + README
- [ ] Choose final Task a config (bias 3 vs pure vs bias grid; SI trade-off), user marks best Task b config per prompt by listening

## 0. Setup
- [x] Research reviewed, plan locked (see WORKLOG 2026-10-06)
- [x] GitHub repo `CommE5070-PA2` created (private, personal account; code only — planning docs + project notes stay local because they mention assistant)
- [x] scp POP909 + scripts to gsm-gpu2, unzipped there
- [x] conda env `comme5070_pa2` + MusDr clone
- [x] tokenization.py run on all 896 songs (p=4,16); eval_basic real-data baseline run

## 1. Basic Task a — Unconditional generation
- Round 2 running 2026-10-07 (v1 12k steps, v2 dropout 0.3, v3 6L384, v4 wd 0.1; GPUs 1/2/3/0). Round 1 (done): u1 baseline, u2 p=16, u3 label smoothing 0.1, u4 12 layers on GPUs 1/2/3/0. All overfit after ~9k steps (WORKLOG). Next: generation sweep on best.pt + step checkpoints (H1/H4/GS vs real 2.118/2.617/0.887), then round 2 (regularization levers).
- [x] Token representation: TA REMI-style via src/tokenize_hw2.py, pos_per_beat=4 (WORKLOG). Bonus tokenizer (MELODY/PIANO) still to do in section 3
- [~] Model + training script done (src/train_uncond.py); run u1 launched 2026-10-06, see WORKLOG — mark done after training + sanity-listen
- [x] Generate samples (round 1: 48 cells, WORKLOG 2026-10-07) — [ ] USER to listen: WAVs for candidate cells still to be rendered; density finding says metrics alone are not enough
- [ ] Define ≥3 inference configurations to compare (sampling method / temperature / loss variant, etc.)
- [ ] For each config: generate 20× 32-bar `.mid`, run `eval_basic.py` (H4 + GS required, SI optional — flagged slow, start early if attempting)
- [ ] Pick best config, convert its 20 pieces to `.wav` for submission

## 2. Basic Task b — Prompt continuation
- [ ] Load Task-a checkpoint (no retraining), generate 24-bar continuations for all 3 TA prompts × 3 inference configs
- [ ] Convert all to `.mid` + `.wav`, listen, mark best config per prompt

## 3. Bonus — Accompaniment generation (MANDATORY; in progress since 2026-10-07)
- [ ] ASK TA (rakechen168@gmail.com): may `chord_midi.txt` chord labels be used as model input for the bonus task? (draft in WORKLOG 2026-10-07; decides main setting: melody-only CA ~0.15 vs chord-conditioned CA ~0.6)
Prepared (CPU-side, 2026-10-07): `src/bonus_data.py` (run OK), `src/bonus_lib.py`, `src/train_bonus.py` (not yet run on GPU), `RESEARCH_PROMPT_R2.md` (to hand to deep research).
- [ ] Hand `RESEARCH_PROMPT_R2.md` to deep research, review the response
- [ ] Smoke-test `train_bonus.py` for arch A and arch B (100 steps each), check shapes/loss masks with a positive control
- [ ] Write bonus generation (`src/gen_bonus.py`): A = segment-injection loop, B = encoder-decoder generate; write MELODY (copied from prompt) + PIANO tracks on the prompt's own beat grid (eval_bonus compares seconds)
- [x] Train A and B (both run), scored with `eval_bonus.py` — see WORKLOG 2026-10-07 (A: bA1-bA9, B: bB1-bB6, stage-1 chord predictor bC1). Still open: bA8/bA9/bB6 scoring, seeds/CI, choose the final bonus config
- [~] Ablations (done: lookahead 0/1/2, chord on/off + where the chord token sits, p4 vs p16, predicted chord, two-stage; open: melody loss weight, init from unconditional (bA9 running), size, transposition, best-of-N): lookahead 0/1/2, chord token on/off (check with TA whether chord input is intended), melody loss weight, init from unconditional checkpoint, size, transposition, best-of-N
- [ ] Generate for all 10 prompts with the best config, `.mid` + `.wav`
- [ ] Log every run with full config + code/data hashes (provenance is saved in run_config.json)

## 4. Report + submission
- [ ] Write report (16:9 PDF, `studentID_report.pdf`, ~10 pages guideline): token rep + architecture, implementation details, inference-config comparison table, bonus task writeup, discussion
- [ ] Prep cloud-drive package: code, `README.md` (TA-runnable inference, settable output path, produces listening samples), `requirements.txt`, checkpoints, all generation results — exclude training/preprocessed data, others' models, cache files
- [ ] Add cloud-drive link to report's first page + NTU COOL comment
- [ ] Make GitHub repo public (after deadline, per standing policy)

## Round 2 queue (2026-10-11)
- [x] FB5 predicted-chord B scored (CA 0.127-0.163)
- [x] In-attention chord embedding, bonus A (FA11 control vs FA12: +0.06 to +0.14 CA)
- [ ] FB9 (T5 buckets 128 / dist 1024), FB10 (d256 4+4) trained + swept
- [ ] Best-of-8 by chord agreement (given and two-stage)
- [ ] F7 RoPE/llama-style Task a model + sweep
- [ ] SI on final F1 sets
- [ ] Pitch-only repetition penalty; chord-conditioned logit bias (chord-given only)
- [ ] Task a final 20 .mid/.wav (pure default), Task b regenerate from F1, bonus 10 .mid/.wav
