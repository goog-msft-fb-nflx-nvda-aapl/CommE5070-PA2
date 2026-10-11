#!/bin/bash
# pitch-only repetition penalty on the final Task a model F1 (last.pt), 3 strengths x 3 seeds x 20 pieces; usage: run_reppen.sh <gpu>
cd /home/jtan/comme5070_pa2
source ~/miniconda3/etc/profile.d/conda.sh && conda activate comme5070_pa2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src CUDA_VISIBLE_DEVICES=$1
for S in 0 1 2; do
  python src/sweep_uncond.py --out_root work/gen_final --sampling_file work/cfg/sampling_taska_reppen.json --ckpts last --run_dir work/runs/F1_w2_full_p16 \
    --tok2idx work/tok2idx_p16.json --pos_per_beat 16 --seed $S --results work/gen_final/results_F1_reppen_s$S.jsonl
done
