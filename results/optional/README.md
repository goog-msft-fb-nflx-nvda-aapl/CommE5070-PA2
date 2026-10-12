# Optional experiments (Task a, same recipe as F1 unless stated; mean of 3 sampling seeds x 20 pieces, unmodified eval_basic.py)
Full configs: run_config.json of each run; per-cell metrics: metrics.csv / results_F9_*.jsonl; scripts: opt_*.sh (src/ has bpe_data.py, gen_bpe.py, miditok_data.py, gen_miditok.py).

| model | sampling | H1 | H4 | GS |
|---|---|---|---|---|
| F1 reference (TA REMI p16, 8L, ~27M; real 2.118/2.617/0.962) | T1.0 pure | 2.169 | 2.600 | 0.963 |
| F9: 12 layers (~42M) | T1.0 pure | 2.185 | 2.651 | 0.957 |
| F9 | T1.0 top-p 0.95 | 2.149 | 2.562 | 0.987 |
| F9 | T0.9 top-p 0.9 | 2.090 | 2.464 | 0.994 |
| F9 | T1.2 top-k 5 | 2.272 | 2.627 | 0.955 |
| F9 | T1.0 + Bar bias 3 | 2.088 | 2.598 | 0.968 |
| F10: BPE 1500 merges (1.69x shorter sequences, vocab 2045) | T1.0 pure | 2.159 | 2.609 | 0.958 |
| F10 | T1.0 top-p 0.95 | 2.050 | 2.429 | 0.984 |
| F10 | T0.9 top-p 0.9 | 1.930 | 2.326 | 0.993 |
| F10 | T1.2 top-k 5 | 1.928 | 2.341 | 0.987 |
| F11: MidiTok REMI (time signature fixed from beat_midi.txt; own real reference 2.192/2.647/0.978) | T1.0 pure | 2.201 | 2.613 | 0.978 |
| F11 | T1.0 top-p 0.95 | 2.194 | 2.501 | 0.993 |
| F11 | T0.9 top-p 0.9 | 2.136 | 2.383 | 0.998 |
| F11 | T1.2 top-k 5 | 2.257 | 2.526 | 0.980 |

Notes: F11 numbers are not comparable with the TA-token rows (different quantization, own real reference). F10 steps see 1.69x more music tokens than F1 steps. Training-seed spread of F1 pure H4 is about 0.05, so none of the pure-sampling differences to F1 is resolved. In F11 the .mid decoding with MidiTok failed in gen_miditok.py (KeyError in decode, token-level eval unaffected); no F11 audio was produced.
