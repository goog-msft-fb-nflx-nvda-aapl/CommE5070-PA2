#!/bin/bash
# conditioning dropout + classifier-free guidance, bonus A L1 chord 3 + chord_emb (FA12 recipe) with --chord_drop 0.15; then sweep at cfg 1 (no guidance), 1.5, 2, 3. usage: cfg_runs.sh <gpu>
cd /home/jtan/comme5070_pa2
source ~/miniconda3/etc/profile.d/conda.sh && conda activate comme5070_pa2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src CUDA_VISIBLE_DEVICES=$1
BD=work/bonus_p4; R=FA15_A_L1_chordboth_chordemb_drop0.15_full
python src/train_bonus.py --data $BD/bonus_train.json --tok2idx $BD/tok2idx_bonus.json --arch A --steps 10000 --out_dir work/runs/$R --val_frac 0 --eval_every 5000 --save_every 2500 --dropout 0.3 --lookahead 1 --chord 3 --chord_emb 1 --chord_drop 0.15 > work/runs_$R.log 2>&1
for W in 1.0 1.5 2.0 3.0; do
  SUFFIX=_cfg$W bash src/sweep_bonus.sh $R last $BD 4 --cfg $W > work/sweep_${R}_cfg$W.log 2>&1
done
