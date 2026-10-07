#!/bin/bash
# usage: sweep_bonus.sh <run_dir_name> <ckpt_name> <bonus_dir> <pos_per_beat> [extra gen_bonus args...]
# env SUFFIX (e.g. _s1chords) keeps a variant's outputs/rows apart from the base run.
# 4 sampling configs x 3 seeds; one scored json row per cell in work/gen_bonus/results_<run>.jsonl
RUN=$1; CK=$2; BD=$3; PB=$4; shift 4
cd /home/jtan/comme5070_pa2
for S in "T1.0_pure|--temperature 1.0" "T1.0_p0.95|--temperature 1.0 --top_p 0.95" "T0.9_p0.9|--temperature 0.9 --top_p 0.9" "T1.2_k5|--temperature 1.2 --top_k 5"; do
  NAME=${S%%|*}; ARGS=${S#*|}
  for SEED in 0 1 2; do
    OUT=work/gen_bonus/${RUN}${SUFFIX}/${CK}__${NAME}__s$SEED
    [ -f $OUT/piano_tokens.json ] && continue
    python src/gen_bonus.py --ckpt work/runs/$RUN/$CK.pt --tok2idx $BD/tok2idx_bonus.json --prompts $BD/bonus_prompts.json \
      --prompt_dir data/POP909/prompts/bonus --pos_per_beat $PB --out_dir $OUT --results work/gen_bonus/results_${RUN}${SUFFIX}.jsonl \
      --tag "${CK}__${NAME}" --seed $SEED $ARGS "$@" 2>&1 | grep -E "^generated|Average CA:|Error|Traceback" | sed "s|^|$RUN $CK $NAME s$SEED: |"
  done
done
