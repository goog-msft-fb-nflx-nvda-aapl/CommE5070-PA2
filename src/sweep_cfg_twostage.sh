#!/bin/bash
# melody-only pipeline: stage-1 chords from F3 (arch C) -> FA15 with CFG; usage: sweep_cfg_twostage.sh <gpu>
cd /home/jtan/comme5070_pa2
source ~/miniconda3/etc/profile.d/conda.sh && conda activate comme5070_pa2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src CUDA_VISIBLE_DEVICES=$1
BD=work/bonus_p4; R=FA15_A_L1_chordboth_chordemb_drop0.15_full
for W in 1.0 3.0; do for SEED in 0 1 2; do
  python src/gen_bonus.py --ckpt work/runs/$R/last.pt --chord_from work/runs/F3_bC1_full/last.pt --tok2idx $BD/tok2idx_bonus.json --prompts $BD/bonus_prompts.json \
   --prompt_dir data/POP909/prompts/bonus --pos_per_beat 4 --out_dir work/gen_bonus/${R}_twostage_cfg$W/s$SEED --results work/gen_bonus/results_${R}_twostage.jsonl \
   --tag "last__T0.9_p0.9__twostage" --seed $SEED --temperature 0.9 --top_p 0.9 --cfg $W 2>&1 | grep -E "^generated|Average CA:|Error|Traceback"
done; done
