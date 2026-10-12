#!/bin/bash
# optional: BPE-merged REMI p16 (1500 merges, offline transposition copies, 1.69x shorter), GPT-2 8L F1 recipe (20000 steps, batch 8, dropout 0.3, ctx 4096, --transpose 0 because transposition is baked into the data),
# then 4 sampling configs x 3 seeds x 20 pieces scored with the unmodified eval_basic.py. usage: opt_bpe.sh <gpu>
cd /home/jtan/comme5070_pa2
source ~/miniconda3/etc/profile.d/conda.sh && conda activate comme5070_pa2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=$1
RUN=F10_bpe1500_p16_full; B=work/bpe; O=work/gen_final/$RUN
python src/train_uncond.py --tok2idx $B/tok2idx_bpe.json --sequences $B/seqs_bpe.json --out_dir work/runs/$RUN --steps 20000 --val_frac 0 --eval_every 2500 --save_every 2500 --dropout 0.3 --transpose 0 > work/runs_$RUN.log 2>&1
for S in 0 1 2; do
 for CFG in "T1.0_pure|--temperature 1.0" "T1.0_p0.95|--temperature 1.0 --top_p 0.95" "T0.9_p0.9|--temperature 0.9 --top_p 0.9" "T1.2_k5|--temperature 1.2 --top_k 5"; do
  N=${CFG%%|*}; A=${CFG#*|}; D=$O/${N}_s$S
  python src/gen_bpe.py --ckpt work/runs/$RUN/last.pt --tok2idx $B/tok2idx_bpe.json --table $B/bpe_table.json --out_dir $D --seed $S $A > $D.log 2>&1
  python scripts/eval_basic.py --tok2idx_path work/tok2idx_p16.json --sequences_path $D/g_tokens.json --output_csv $D/metrics.csv 2>&1 | grep -E "^Average" | tr '\n' ' ' | sed "s|^|$N s$S: |"; echo
 done
done
