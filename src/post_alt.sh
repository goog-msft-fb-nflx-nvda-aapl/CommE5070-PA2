#!/bin/bash
# wait for the full-data MIDI-like / CP models and run the Task a sweeps with gen_alt.py; usage: post_alt.sh <gpu>
cd /home/jtan/comme5070_pa2
source ~/miniconda3/etc/profile.d/conda.sh && conda activate comme5070_pa2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=src
for spec in "R1f_midilike_p4_full work/cfg/sampling_task_a_nobias.json" "R2f_cp_p4_full work/cfg/sampling_task_a.json"; do
  set -- $spec; RUN=$1
  while [ ! -f work/runs/$RUN/last.pt ]; do sleep 60; done
  for S in 0 1 2; do
    python src/sweep_uncond.py --gen_script gen_alt.py --out_root work/gen_final --sampling_file $2 --ckpts last --run_dir work/runs/$RUN \
      --tok2idx work/tok2idx_p4.json --pos_per_beat 4 --seed $S --results work/gen_final/results_${RUN}_s$S.jsonl
  done
done
