#!/bin/bash
# bonus A melody-only (no chord input) + auxiliary chord-classification loss; control = FA1 (same settings, no aux). usage: aux_runs.sh <gpu>
cd /home/jtan/comme5070_pa2
source ~/miniconda3/etc/profile.d/conda.sh && conda activate comme5070_pa2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src CUDA_VISIBLE_DEVICES=$1
BD=work/bonus_p4
for spec in "FA13_A_L1_nochord_aux0.3_full 0.3" "FA14_A_L1_nochord_aux1.0_full 1.0"; do
  set -- $spec; R=$1
  python src/train_bonus.py --data $BD/bonus_train.json --tok2idx $BD/tok2idx_bonus.json --arch A --steps 10000 --out_dir work/runs/$R --val_frac 0 --eval_every 5000 --save_every 2500 --dropout 0.3 --lookahead 1 --chord 0 --chord_aux $2 > work/runs_$R.log 2>&1
  bash src/sweep_bonus.sh $R last $BD 4 > work/sweep_$R.log 2>&1
done
