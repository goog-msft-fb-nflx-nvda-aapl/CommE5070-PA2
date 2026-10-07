#!/bin/bash
# usage: sweep_bonus_typed.sh <run> <ckpt> <bonus_dir> <pos_per_beat> <variant_name> [gen_bonus args...]
# base sampling T=0.9 top-p 0.9, seeds 0,1,2; outputs/rows under <run>_<variant_name>
RUN=$1; CK=$2; BD=$3; PB=$4; VAR=$5; shift 5
cd /home/jtan/comme5070_pa2
for SEED in 0 1 2; do
  OUT=work/gen_bonus/${RUN}_${VAR}/${CK}__T0.9_p0.9__s$SEED
  [ -f $OUT/piano_tokens.json ] && continue
  python src/gen_bonus.py --ckpt work/runs/$RUN/$CK.pt --tok2idx $BD/tok2idx_bonus.json --prompts $BD/bonus_prompts.json \
    --prompt_dir data/POP909/prompts/bonus --pos_per_beat $PB --out_dir $OUT --results work/gen_bonus/results_${RUN}_${VAR}.jsonl \
    --tag "${CK}__${VAR}" --seed $SEED --temperature 0.9 --top_p 0.9 "$@" 2>&1 | grep -E "^generated|Average CA:|Error|Traceback" | sed "s|^|$RUN $VAR s$SEED: |"
done
