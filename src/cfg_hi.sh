#!/bin/bash
# FA15 sweep at larger guidance scales; usage: cfg_hi.sh <gpu>
cd /home/jtan/comme5070_pa2
source ~/miniconda3/etc/profile.d/conda.sh && conda activate comme5070_pa2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src CUDA_VISIBLE_DEVICES=$1
for W in 4.0 6.0; do SUFFIX=_cfg$W bash src/sweep_bonus.sh FA15_A_L1_chordboth_chordemb_drop0.15_full last work/bonus_p4 4 --cfg $W > work/sweep_FA15_cfg$W.log 2>&1; done
