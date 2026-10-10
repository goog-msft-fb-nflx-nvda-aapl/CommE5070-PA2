#!/bin/bash
# best-of-N (N candidates per prompt, chosen by agreement with the conditioning chords) for the final bonus models; 3 seeds, T0.9 top_p0.9
# usage: sweep_bestofn.sh <N>   (env: GPU via CUDA_VISIBLE_DEVICES)
cd /home/jtan/comme5070_pa2
N=$1
BD=work/bonus_p4
for SEED in 0 1 2; do
  # chord given (the prompt folder's chord labels)
  OUT=work/gen_bonus/F2_bestof${N}_given/s$SEED
  python src/gen_bonus.py --ckpt work/runs/F2_bB6_full/last.pt --tok2idx $BD/tok2idx_bonus.json --prompts $BD/bonus_prompts.json \
    --prompt_dir data/POP909/prompts/bonus --pos_per_beat 4 --out_dir $OUT --results work/gen_bonus/results_F2_bestof${N}.jsonl \
    --tag "last__T0.9_p0.9__bestof${N}_given" --seed $SEED --temperature 0.9 --top_p 0.9 --best_of $N 2>&1 | grep -E "^generated|Average CA:|Error|Traceback"
  # two-stage (chords predicted from the melody by stage 1; no chord file used)
  OUT=work/gen_bonus/F2F3_bestof${N}_twostage/s$SEED
  python src/gen_bonus.py --ckpt work/runs/F2_bB6_full/last.pt --chord_from work/runs/F3_bC1_full/last.pt --tok2idx $BD/tok2idx_bonus.json --prompts $BD/bonus_prompts.json \
    --prompt_dir data/POP909/prompts/bonus --pos_per_beat 4 --out_dir $OUT --results work/gen_bonus/results_F2F3_bestof${N}.jsonl \
    --tag "last__T0.9_p0.9__bestof${N}_twostage" --seed $SEED --temperature 0.9 --top_p 0.9 --best_of $N 2>&1 | grep -E "^generated|Average CA:|Error|Traceback"
done
