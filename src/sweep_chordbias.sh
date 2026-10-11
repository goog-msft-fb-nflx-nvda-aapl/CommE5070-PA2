#!/bin/bash
# chord-triad logit bias on the final bonus model F2 (T0.9 top_p0.9, seeds 0-2, 10 prompts), given chords and two-stage chords; usage: sweep_chordbias.sh
cd /home/jtan/comme5070_pa2
BD=work/bonus_p4
for B in 1 2 4; do
  for SEED in 0 1 2; do
    python src/gen_bonus.py --ckpt work/runs/F2_bB6_full/last.pt --tok2idx $BD/tok2idx_bonus.json --prompts $BD/bonus_prompts.json \
      --prompt_dir data/POP909/prompts/bonus --pos_per_beat 4 --out_dir work/gen_bonus/F2_cbias${B}_given/s$SEED --results work/gen_bonus/results_F2_chordbias.jsonl \
      --tag "last__T0.9_p0.9__cbias${B}_given" --seed $SEED --temperature 0.9 --top_p 0.9 --chord_bias $B 2>&1 | grep -E "^generated|Average CA:|Error|Traceback"
    python src/gen_bonus.py --ckpt work/runs/F2_bB6_full/last.pt --chord_from work/runs/F3_bC1_full/last.pt --tok2idx $BD/tok2idx_bonus.json --prompts $BD/bonus_prompts.json \
      --prompt_dir data/POP909/prompts/bonus --pos_per_beat 4 --out_dir work/gen_bonus/F2F3_cbias${B}_twostage/s$SEED --results work/gen_bonus/results_F2F3_chordbias.jsonl \
      --tag "last__T0.9_p0.9__cbias${B}_twostage" --seed $SEED --temperature 0.9 --top_p 0.9 --chord_bias $B 2>&1 | grep -E "^generated|Average CA:|Error|Traceback"
  done
done
