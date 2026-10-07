#!/bin/bash
# SI for: 100 real 32-bar crops + generated sets (given as name=midi_dir pairs). CPU only.
cd /home/jtan/comme5070_pa2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=src CUDA_VISIBLE_DEVICES=""
mkdir -p work/si/real_mid work/si/out
[ -f work/si/out/real.json ] || { python src/mk_real_mid.py 100 work/si/real_mid; mkdir -p work/si/real_wav; python src/render_wav.py --midi_dir work/si/real_mid --out_dir work/si/real_wav > /dev/null 2>&1; python src/si_eval.py --wav_dir work/si/real_wav --out_json work/si/out/real.json --workers 12; }
for pair in "$@"; do
  NAME=${pair%%=*}; DIR=${pair#*=}
  [ -f work/si/out/$NAME.json ] && continue
  mkdir -p work/si/wav_$NAME; python src/render_wav.py --midi_dir $DIR --out_dir work/si/wav_$NAME > /dev/null 2>&1
  python src/si_eval.py --wav_dir work/si/wav_$NAME --out_json work/si/out/$NAME.json --workers 12
done
