#!/bin/bash
# melody-only key-aware pitch bias on the melody-only models FA1 (A L1 no chord) and the two-stage-free baseline; usage: sweep_keybias.sh <gpu> ; bias 1,2,4, 3 seeds, pure and T0.9_p0.9
cd /home/jtan/comme5070_pa2
source ~/miniconda3/etc/profile.d/conda.sh && conda activate comme5070_pa2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src CUDA_VISIBLE_DEVICES=$1
BD=work/bonus_p4
for R in $(ls work/runs | grep -E "^FA1_" | head -1); do
 for B in 1 2 4; do for SEED in 0 1 2; do
  python src/gen_bonus.py --ckpt work/runs/$R/last.pt --tok2idx $BD/tok2idx_bonus.json --prompts $BD/bonus_prompts.json \
   --prompt_dir data/POP909/prompts/bonus --pos_per_beat 4 --out_dir work/gen_bonus/${R}_kbias$B/s$SEED --results work/gen_bonus/results_${R}_keybias.jsonl \
   --tag "last__T0.9_p0.9__kbias$B" --seed $SEED --temperature 0.9 --top_p 0.9 --key_bias $B 2>&1 | grep -E "^generated|Average CA:|Error|Traceback"
 done; done
done
