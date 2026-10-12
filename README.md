# CommE5070 HW2: symbolic music generation on POP909

Transformers trained from scratch on the POP909 training split only (no pretrained weights, no extra data).

## Layout
- `infer.py`: inference entry point (writes `.mid` + `.wav`, settable output directory)
- `src/`: tokenizer wrapper, models/training, generation, bonus pipeline, evaluation helpers
- `scripts/`: the TA-provided `tokenization.py`, `eval_basic.py`, `eval_bonus.py` (unmodified)
- `vocab/`: token dictionaries (`tok2idx_p4.json` is the one used by the released models)
- `results/`: training logs, run configs, and one json row per generation/evaluation run (config + checkpoint/code hashes + metrics)
- `patches/`: one-line fix for the third-party MusDr code (see below)
- `checkpoints/` (not in git, distributed with the cloud-drive package)

## Install
```bash
conda create -n hw2 python=3.10 -y && conda activate hw2
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# FluidSynth (needed for .wav rendering; no sudo):
conda install -c conda-forge fluidsynth -y        # or: sudo apt-get install fluidsynth libfluidsynth3
```

## Inference
Put the released checkpoints in `checkpoints/` (`uncond_final.pt`, `bonus_final.pt`, `bonus_stage1.pt`, `bonus_chord_final.pt`; distributed with the cloud-drive package), then:
```bash
python infer.py uncond   --out_dir OUT [--n 20] [--seed 0] [--config cfg1_T1.0_pure]   # Basic task a: 20 pieces x 32 bars
python infer.py continue --prompt_dir POP909/prompts/basic-b --out_dir OUT [--seed 0]      # Basic task b: 3 prompts x 3 configs (8 + 24 bars)
python infer.py bonus    --prompt_dir POP909/prompts/bonus   --out_dir OUT --mode chord_given   # Bonus (default): MELODY -> PIANO, chord labels of the prompt folder as input (allowed by the TA)
python infer.py bonus    --prompt_dir POP909/prompts/bonus   --out_dir OUT --mode melody_only   # melody only (no chord file), for comparison
```
Every call writes `.mid` and `.wav` files under `OUT` (and the token files / config json of the run). `--seed` fixes the sampling;
with the same seed, package versions and GPU architecture the pieces are reproduced exactly (checked: the objective metrics of the files written by
`infer.py` equal those of the experiment runs to 4 decimals).
Task a configs: `cfg1_T1.0_pure` (default), `cfg2_T1.0_barbias3`, `cfg3_T0.9_top_p0.9`.
The models are GPT-2 (Task a/b, `pos_per_beat` 16) and a T5 encoder-decoder (bonus, `pos_per_beat` 4), all trained from scratch on the 896 POP909 training songs.

## Training (from scratch, POP909 only)
```bash
python src/tokenize_hw2.py --dataset_path POP909/training --tok2idx_output work/tok2idx_p4.json \
       --sequences_output work/seqs_p4.json --pos_per_beat 4
python src/train_uncond.py --tok2idx work/tok2idx_p4.json --sequences work/seqs_p4.json --out_dir work/runs/v2 \
       --steps 20000 --dropout 0.3
python src/bonus_data.py --dataset_path POP909/training --prompt_dir POP909/prompts/bonus \
       --base_tok2idx work/tok2idx_p4.json --out_dir work/bonus_p4 --pos_per_beat 4
python src/train_bonus.py --arch B --data work/bonus_p4/bonus_train.json --tok2idx work/bonus_p4/tok2idx_bonus.json \
       --out_dir work/runs/bonusB --n_layer 6 --chord 2 --dropout 0.3 --steps 40000
```

## Evaluation
```bash
# MusDr (https://github.com/slSeanWU/MusDr) is cloned next to this repo; scripts/eval_basic.py imports MusDr.musdr
PYTHONPATH=. python scripts/eval_basic.py --tok2idx_path vocab/tok2idx_p4.json --sequences_path OUT/g_tokens.json --output_csv m.csv
python scripts/eval_bonus.py --prompt_dir POP909/prompts/bonus --gen_midi_dir OUT/bonus_melody_only --output_csv b.csv
```
SI (optional) uses the MusDr Python scape-plot code; with current librosa apply `patches/musdr_librosa_sr.patch`
(`cd MusDr && patch -p1 < ../patches/musdr_librosa_sr.patch`), then `python src/si_eval.py --wav_dir WAVS --out_json si.json`.
Extra packages for evaluation: `pip install -r requirements-eval.txt`.

## Third-party code
- TA scripts in `scripts/` (HW2 starter code), used unmodified; `src/tokenize_hw2.py` wraps `scripts/tokenization.py`.
- MusDr (Wu & Yang, ISMIR 2020), used for H / GS (via `eval_basic.py`) and SI; one-line librosa fix in `patches/`.
- Hugging Face `transformers` (`GPT2LMHeadModel`, `T5ForConditionalGeneration`, random initialization).
- Pitch/onset REMI-style token design follows the TA tokenizer (Huang & Yang, Pop Music Transformer).
