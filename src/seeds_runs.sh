#!/bin/bash
# training-seed replicates of the three headline models (settings identical to F1 / FA12 / F2, only --seed differs); usage: seeds_runs.sh <gpu> <job> [<job> ...]
# jobs: F1 (Task a), FA12 (bonus A + chord embedding), F2 (bonus B chord in decoder)
cd /home/jtan/comme5070_pa2
source ~/miniconda3/etc/profile.d/conda.sh && conda activate comme5070_pa2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src CUDA_VISIBLE_DEVICES=$1; shift
COM="--val_frac 0 --eval_every 5000 --save_every 2500 --dropout 0.3"
BD=work/bonus_p4
for job in "$@"; do
  for SD in 1 2; do
    case $job in
      F1)
        R=F1_w2_full_p16_tseed$SD
        python src/train_uncond.py --tok2idx work/tok2idx_p16.json --sequences work/seqs_p16.json --out_dir work/runs/$R --steps 20000 --seed $SD --val_frac 0 --eval_every 2500 --save_every 2500 --dropout 0.3 > work/runs_$R.log 2>&1
        for G in 0 1 2; do python src/sweep_uncond.py --out_root work/gen_final --sampling_file work/cfg/sampling_task_a.json --ckpts last --run_dir work/runs/$R --tok2idx work/tok2idx_p16.json --pos_per_beat 16 --seed $G --results work/gen_final/results_${R}_s$G.jsonl; done;;
      FA12)
        R=FA12_A_L1_chordboth_chordemb_full_tseed$SD
        python src/train_bonus.py --data $BD/bonus_train.json --tok2idx $BD/tok2idx_bonus.json --arch A --steps 10000 --out_dir work/runs/$R $COM --lookahead 1 --chord 3 --chord_emb 1 --seed $SD > work/runs_$R.log 2>&1
        bash src/sweep_bonus.sh $R last $BD 4 > work/sweep_$R.log 2>&1;;
      F2)
        R=F2_bB6_full_tseed$SD
        python src/train_bonus.py --data $BD/bonus_train.json --tok2idx $BD/tok2idx_bonus.json --arch B --n_layer 6 --steps 40000 --out_dir work/runs/$R $COM --chord 2 --seed $SD > work/runs_$R.log 2>&1
        bash src/sweep_bonus.sh $R last $BD 4 > work/sweep_$R.log 2>&1;;
    esac
  done
done
