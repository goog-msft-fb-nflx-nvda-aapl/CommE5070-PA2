#!/bin/bash
# FULL-DATA runs (all 896 training songs, --val_frac 0): the models behind the reported tables / submission.
# Settings (steps, dropout) were chosen on the earlier held-out-validation runs; checkpoint = last.pt (end of the cosine schedule).
# usage: final_runs.sh <gpu> <job> [<job> ...]   (jobs run sequentially on that GPU)
cd /home/jtan/comme5070_pa2
source ~/miniconda3/etc/profile.d/conda.sh && conda activate comme5070_pa2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src CUDA_VISIBLE_DEVICES=$1; shift
COM="--val_frac 0 --eval_every 5000 --save_every 2500 --dropout 0.3"
U()  { python src/train_uncond.py --tok2idx $2 --sequences $3 --out_dir work/runs/$1 --steps 20000 $COM $4 > work/runs_$1.log 2>&1; }
BA() { python src/train_bonus.py --data work/bonus_p4/bonus_train.json --tok2idx work/bonus_p4/tok2idx_bonus.json --arch A --steps 10000 --out_dir work/runs/$1 $COM $2 > work/runs_$1.log 2>&1; }
BB() { python src/train_bonus.py --data work/bonus_p4/bonus_train.json --tok2idx work/bonus_p4/tok2idx_bonus.json --arch B --n_layer 6 --steps ${3:-20000} --out_dir work/runs/$1 $COM $2 > work/runs_$1.log 2>&1; }
BB16() { python src/train_bonus.py --data work/bonus_p16/bonus_train.json --tok2idx work/bonus_p16/tok2idx_bonus.json --arch B --n_layer 6 --steps 40000 --out_dir work/runs/$1 $COM $2 > work/runs_$1.log 2>&1; }
BC16() { python src/train_bonus.py --data work/bonus_p16/bonus_train.json --tok2idx work/bonus_p16/tok2idx_bonus.json --arch C --n_layer 4 --n_embd 256 --n_head 4 --d_ff 1024 --steps 8000 --out_dir work/runs/$1 $COM > work/runs_$1.log 2>&1; }
wait_for() { while [ ! -f work/runs/$1/last.pt ]; do sleep 30; done; }
for job in "$@"; do
  case $job in
    waitF1) wait_for F1_w2_full_p16;; waitF2) wait_for F2_bB6_full;; waitF3) wait_for F3_bC1_full;;
    F4_p4)        U F4_p4_full work/tok2idx_p4.json work/seqs_p4.json "";;
    F5_p4_chord)  U F5_p4_chord_full work/tok2idx_p4_chord.json work/seqs_p4_chord.json "";;
    F6_p4_ls)     U F6_p4_ls0.1_full work/tok2idx_p4.json work/seqs_p4.json "--label_smoothing 0.1";;
    FA1) BA FA1_A_L1_nochord_full "--lookahead 1 --chord 0";;
    FA11) BA FA11_A_L1_chordboth_full "--lookahead 1 --chord 3";;
    FA12) BA FA12_A_L1_chordboth_chordemb_full "--lookahead 1 --chord 3 --chord_emb 1";;
    FA2) BA FA2_A_L1_chordmel_full "--lookahead 1 --chord 1";;
    FA5) BA FA5_A_L0_chordmel_full "--lookahead 0 --chord 1";;
    FA6) BA FA6_A_L2_chordmel_full "--lookahead 2 --chord 1";;
    FA7) BA FA7_A_L1_predchord_full "--lookahead 1 --chord 4";;
    FA8) BA FA8_A_L1_chordpno_full "--lookahead 1 --chord 2";;
    FA10) BA FA10_A_L2_chordpno_full "--lookahead 2 --chord 2";;
    DEV_A2_10k) # held-out check of the short-schedule protocol (5% validation), NOT a final model
        python src/train_bonus.py --data work/bonus_p4/bonus_train.json --tok2idx work/bonus_p4/tok2idx_bonus.json --arch A --steps 10000 --lookahead 1 --chord 1 --out_dir work/runs/DEV_bA2_10k --dropout 0.3 --eval_every 500 --save_every 2500 > work/runs_DEV_bA2_10k.log 2>&1;;
    FB1) BB FB1_B_nochord_full "--chord 0";;
    FB2) BB FB2_B_encchord_full "--chord 1";;
    FB5) BB FB5_B_predchord_full "--chord 4";;
    FB9)  BB FB9_B_decchord_relpos128_full "--chord 2 --rel_buckets 128 --rel_maxdist 1024" 40000;;
    FB10) BB FB10_B_decchord_small_full "--chord 2 --n_layer 4 --n_embd 256 --n_head 4 --d_ff 1024" 40000;;
    FB8_p16) BB16 FB8_B_decchord_p16_full "--chord 2";;
    FC4_p16) BC16 FC4_C_p16_full;;
  esac
done
