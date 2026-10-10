#!/bin/bash
# wait for full-data Task a models and run the 5-config x 3-seed sweep on each (last.pt); usage: post_final.sh <gpu>
cd /home/jtan/comme5070_pa2
source ~/miniconda3/etc/profile.d/conda.sh && conda activate comme5070_pa2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$1
for spec in "F4_p4_full work/tok2idx_p4.json 4" "F5_p4_chord_full work/tok2idx_p4_chord.json 4" "F6_p4_ls0.1_full work/tok2idx_p4.json 4"; do
  set -- $spec; RUN=$1
  while [ ! -f work/runs/$RUN/last.pt ]; do sleep 60; done
  for S in 0 1 2; do
    python src/sweep_uncond.py --out_root work/gen_final --sampling_file work/cfg/sampling_task_a.json --ckpts last --run_dir work/runs/$RUN \
      --tok2idx $2 --pos_per_beat $3 --seed $S --results work/gen_final/results_${RUN}_s$S.jsonl
  done
done
