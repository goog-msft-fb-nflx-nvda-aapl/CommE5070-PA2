#!/bin/bash
# optional: F9 = Task a F1 protocol (p16, 20000 steps, dropout 0.3, full data, seed 0) with 12 layers (d512, 8 heads, ~42M params); 3-seed sweep. usage: opt_12L.sh <gpu>
cd /home/jtan/comme5070_pa2
source ~/miniconda3/etc/profile.d/conda.sh && conda activate comme5070_pa2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src CUDA_VISIBLE_DEVICES=$1
RUN=F9_12L_p16_full
python src/train_uncond.py --tok2idx work/tok2idx_p16.json --sequences work/seqs_p16.json --out_dir work/runs/$RUN --steps 20000 --val_frac 0 --eval_every 2500 --save_every 2500 --dropout 0.3 --n_layer 12 > work/runs_$RUN.log 2>&1
for S in 0 1 2; do
  python src/sweep_uncond.py --out_root work/gen_final --sampling_file work/cfg/sampling_task_a.json --ckpts last --run_dir work/runs/$RUN \
    --tok2idx work/tok2idx_p16.json --pos_per_beat 16 --seed $S --results work/gen_final/results_${RUN}_s$S.jsonl
done
