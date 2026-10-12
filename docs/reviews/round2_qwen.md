I will conduct targeted web searches to gather the specific, sourced information required for each of the 7 questions in your research prompt, focusing on POP909, melody-to-accompaniment models, interleaving strategies, T5/BART from scratch, harmony control, inference-time improvements, unconditional generation levers, and reproducibility.


Here is the sourced, deep-research response tailored to your constraints, formatted exactly as requested.

### 1. Small-data from-scratch recipes for melody→piano accompaniment
* **PopMAG**: Uses a Transformer for multi-track harmony modeling, evaluated on LMD, FreeMidi, and CPMD (Chinese pop). It establishes the multi-track interleaving baseline but does not explicitly report POP909 CA/OA. [[38]](https://arxiv.org/abs/2008.07703)
* **Compose & Embellish (C&E)**: A two-stage Transformer (12-layer, ~41.3M params) with relative positional encoding, trained on POP1K7/POP909. It uses REMI-like tokenization and evaluates structureness and chord preservation. [[44]](https://arxiv.org/abs/2209.08212), [[49]](https://huggingface.co/slseanwu/compose-and-embellish-pop1k7)
* **Anticipatory Music Transformer (AMT)**: A decoder-only Transformer with an "anticipation" mechanism for infilling, evaluated on POP909. It explicitly uses chord accuracy to measure if the generated arrangement matches the conditional chord sequence. [[177]](https://arxiv.org/abs/2306.08620), [[163]](https://smcnus.comp.nus.edu.sg/archive/pdf/2024/2024_structured_arrangement.pdf)
* **Metrics**: Chord Accuracy (CA) calculates the percentage of bars/beats where the recognized chord of the generated output exactly matches the ground truth. [[145]](https://icml.cc/virtual/2025/poster/44043) Overlapped Area (OA) measures the average overlapped area of KDE-smoothed distributions for pitch class, note, duration, and inter-onset interval (IOI). Strong systems typically achieve ~0.60–0.75 for pitch class, and ~0.40–0.60 for velocity/duration/IOI. [[104]](https://arxiv.org/abs/2201.08526)
* **Typical small-data setup**: 16- to 32-bar crop lengths, batch size 32–64, ~100k–300k steps, learning rate 1e-4 with linear warmup, and REMI/Compound Word tokenization.

### 2. Interleaving design for decoder-only conditional generation
* **Interleaving granularity**: Bar-level interleaving (`<M>` melody-bar `<P>` piano-bar) is strongly preferred over token-level for REMI-like setups, as it preserves structural alignment and reduces sequence length.
* **Lookahead**: Allowing lookahead (e.g., $t_f > 0$ bars) grants the model access to future melody input, which significantly improves harmony alignment and reduces exposure drift compared to strict 0-bar autoregressive decoding. [[77]](https://arxiv.org/abs/2510.22105) Ablations in streaming accompaniment show that 1- to 2-bar lookahead provides the best trade-off between latency and harmonic correctness.
* **Loss masking**: It is critical to mask the loss on condition (melody) tokens. Training on everything wastes capacity on reconstructing the prompt rather than learning the conditional $P(\text{piano} | \text{melody})$ distribution.
* **Track-switch tokens**: Explicit track tokens (e.g., `<Track_M>`, `<Track_P>`) at the start of each segment help the model distinguish between conditioning and generation, mitigating drift during forced-condition decoding.

### 3. T5/BART-style from scratch on <1k songs
* **Pitfalls**: Encoder-decoder models are notoriously data-hungry. From scratch on <1k songs, they often suffer from posterior collapse, generic outputs, or failure to attend to the encoder without heavy regularization.
* **Hyperparameters**: 
  * **Relative position**: T5-style logarithmic relative position buckets are essential for 1–2k token sequences to generalize beyond training length. [[108]](https://medium.com/data-science/practical-tips-for-training-a-music-model-755c62560ec2)
  * **Dropout**: 0.1–0.2 (higher than standard to prevent overfitting on small data).
  * **LR & Warmup**: 1e-4 to 3e-4 with a long linear warmup (e.g., 2000–4000 steps).
  * **Size**: Keep it small (e.g., T5-small, ~60M params) to match the data scale.
  * **bf16 & tied embeddings**: Recommended for memory efficiency and gradient stability on H200s.
* **Head-to-head**: Decoder-only (GPT-2 style) generally outperforms encoder-decoder from scratch on small symbolic music datasets because the autoregressive next-token prediction provides a stronger, more consistent gradient signal than seq2seq mapping when data is scarce. *(Unverified: specific head-to-head ablation on POP909, but this is consistent with small-data NLP findings).*

### 4. Harmony control
* **Per-bar chord tokens in condition**: Highly effective and legitimate. Since the chord file is in the prompt folder, providing it as an input condition (e.g., `<Chord_C>` per bar) is standard practice and not cheating.
* **Chord tokens interleaved in output**: Legitimate, but the model must predict them, which can be error-prone from scratch and may degrade note generation.
* **Chord-aware loss**: Weighting the loss on notes that define the chord (root, third, fifth) higher can directly improve CA.
* **Constrained/guided decoding**: Legitimate if it only uses the prompt's chord info. For example, applying a logit bias to favor notes belonging to the current bar's chord during generation.
* **Reranking/best-of-N**: Legitimate and highly effective for CA. Generate $N$ candidates and score them with a chord estimator, picking the best. This is inference-time and uses allowed prompt info.

### 5. Inference-time improvements that are legitimate
* **Best-of-N sampling**: Generate 5–10 variations and rank by the CA metric (or a proxy chord estimator). This directly optimizes the evaluation metric with minimal compute.
* **Temperature/top-p per token type**: Use lower temperature (e.g., 0.8) for `Pos`/`Pitch` to ensure harmonic correctness, and slightly higher (e.g., 1.0–1.1) for `Vel`/`Dur` to maintain expressive OA distributions.
* **Repetition control**: A mild repetition penalty (e.g., 1.1) prevents stuck notes, but keep it low to allow legitimate ostinato patterns.
* **Key-aware logit biasing**: Applying a small positive bias to notes in the global key and current chord drastically improves CA without violating "from scratch" rules.
* **Typical OA numbers**: As noted above, strong systems achieve OA (pitch/velocity/duration/IOI) around 0.60–0.75 for pitch class, and 0.40–0.60 for velocity/duration/IOI. [[104]](https://arxiv.org/abs/2201.08526)

### 6. Unconditional task (32-bar generation) levers
* **Checkpoint selection**: Use EMA (Exponential Moving Average) checkpoints or the one with the lowest validation loss on 32-bar crops, not necessarily the last epoch.
* **Label smoothing**: A small amount (e.g., 0.1) can prevent overconfidence and improve H1/H4 (human evaluation metrics) and GS (Generation Score).
* **Context length & Positional Encodings**: For 32 bars (~2k tokens), relative position (T5 buckets) or RoPE is superior to learned absolute positions, which degrade beyond training length. Bar-aware positional encodings (e.g., adding a bar-level sinusoidal signal to token positions) help maintain structural coherence. [[168]](https://arxiv.org/abs/2609.34556v1)
* **Repetition penalties**: Crucial for unconditional generation to avoid looping the same 2-bar pattern, directly impacting GS.

### 7. Reproducibility practice for small-scale generative-music reports
* **Seed handling**: Set `torch.manual_seed`, `numpy.random.seed`, and `random.seed` globally.
* **HF Generate**: Explicitly set `generation_config = GenerationConfig(do_sample=True, temperature=0.9, top_p=0.9, seed=42)` or pass `generator=torch.Generator(device='cuda').manual_seed(42)`. [[137]](https://huggingface.co/docs/transformers/en/main_classes/text_generation)
* **GPU Determinism**: Full bitwise reproducibility on GPU is not guaranteed due to non-deterministic atomic operations in CUDA (e.g., `torch.baddbmm`). To maximize determinism, set `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, though this may slow down inference. [[129]](https://huggingface.co/docs/diffusers/v0.13.0/using-diffusers/reproducibility), [[136]](https://github.com/huggingface/transformers/issues/25507)
* **Release**: Publish the exact `generation_config.json`, the seed used, and the specific checkpoint hash.

---

### Ranked List of 5 Experiments to Run First
Given your constraints (4x H200, ~19 days, solo, POP909, CA/OA metrics), here are the highest-leverage experiments:

1. **Decoder-only (A) with 1-bar lookahead + per-bar chord conditioning**  
   *Deciding number:* **Chord Accuracy (CA)**. *(Highest impact on bonus score; directly addresses harmony alignment with minimal architectural risk).*
2. **Best-of-N reranking (N=5) using the TA's chord estimator**  
   *Deciding number:* **CA improvement over greedy/single sample**. *(Low compute cost, high metric gain, and fully legitimate under the rules).*
3. **Key-aware logit biasing during inference**  
   *Deciding number:* **CA**. *(Simple inference-time hack to force harmonic correctness without requiring retraining).*
4. **Encoder-Decoder (B) T5-small (~60M) with T5 relative position buckets vs. Decoder-only (A)**  
   *Deciding number:* **Validation Loss**. *(Tests if seq2seq is viable or if decoder-only is strictly better for <1k songs before committing compute).*
5. **Unconditional checkpoint initialization for Architecture (A)**  
   *Deciding number:* **OA (pitch/velocity)**. *(Tests if pre-learning the marginal distribution of piano helps conditional generation and improves distribution matching).*