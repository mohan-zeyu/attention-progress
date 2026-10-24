# 01 · Positions

::: lede
RoPE decides how far apart two tokens look. Long context needs slow pairs that span the window, rescaled pairs, or no rotation at all.
:::

## RoPE: a spectrum of clocks, set by the base {#rope-base}

::: meta
**RoPE** 2021 · Su et al. · [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) — **ABF** 2023 · Xiong et al., Meta · [arXiv:2309.16039](https://arxiv.org/abs/2309.16039) · Used in: Llama 3 (b = 500K), gpt-oss (150K), Qwen3 (1M), Qwen3-Next (10M), Gemma 3 (10K local / 1M global)
:::

$$
\theta_i=b^{-2i/d_h},\qquad \lambda_i=\frac{2\pi}{\theta_i}=2\pi\,b^{2i/d_h},\qquad (R_t q)^{\top}(R_j k)=q^{\top}R_{j-t}\,k
$$

![Each RoPE pair is a clock with its own speed. For Llama 3 (b = 500K, d_h = 128), over 1,000 tokens pair 0 turns 159 times and pair 63 turns 0.14 degrees. The slowest wavelength is 54K tokens at b = 10K, 2.6M at 500K and 49M at 10M; at b = 500K, 29 of 64 pairs turn less than once in the 8K pre-training context. Gemma 3 and DeepSeek-V4 keep b = 10K in local layers and a large base in global layers.](../figures/cards/pos-rope-spectrum.svg)

::: tips
- **Why raise b:** a pair's angle repeats every $\lambda_i$ tokens. ABF (Llama 2 Long: 10K → 500K) slows every pair and weakens long-distance decay.
- **Numbers ($d_h=128$):** slowest wavelength 54K tokens at b = 10K, 2.6M at 500K, 5.1M at 1M. Fastest is always 2π ≈ 6.3.
- **Dual bases:** local layers never see distances beyond W, so they keep b = 10K and skip scaling (Gemma 3, DeepSeek-V4 SWA layers).
- **Not solved:** a larger base is not extrapolation. Positions past the trained length still need scaling (next card).
:::

## Context extension: PI → NTK-aware → YaRN / llama3 {#extension}

::: meta
**PI** 2023 · Chen et al., Meta · [arXiv:2306.15595](https://arxiv.org/abs/2306.15595) — **YaRN** 2023 · Peng et al., Nous Research / EleutherAI · ICLR 2024 · [arXiv:2309.00071](https://arxiv.org/abs/2309.00071) · Used in: gpt-oss, DeepSeek-V3, DeepSeek-V4, OLMo 3, Qwen3 (optional); llama3 type: Llama 3.1, Llama 4
:::

$$
\theta'_i=\big(1-\gamma(r_i)\big)\frac{\theta_i}{s}+\gamma(r_i)\,\theta_i,\qquad r_i=\frac{L}{\lambda_i},\qquad \gamma(r)=\operatorname{clip}\Big(\frac{r-\alpha}{\beta-\alpha},0,1\Big)
$$

![Per-pair divisor of the RoPE frequency for gpt-oss (d_h = 64, b = 150K, 4,096 to 131,072, s = 32). PI divides all 32 pairs by 32. NTK-aware rises smoothly from 1 to 32. YaRN keeps pairs 0 to 8 (wavelength under 128), ramps pairs 9 to 17 and divides pairs 18 to 31 by 32. Llama 3.1 llama3 scaling (s = 8) keeps 29 pairs, blends 6 and divides 29. YaRN's logit multiplier is 1.30 for s = 4, 1.46 for 8, 1.81 for 32 and 1.87 for 40; the llama3 type uses 1.](../figures/cards/pos-extension.svg)

::: tips
- **Compare:** PI divides every pair by $s$ and blurs local order. YaRN keeps pairs with $\lambda_i < L/\beta$ and interpolates only slow ones.
- **Numbers:** gpt-oss 4,096 → 131,072 (s = 32). DeepSeek-V3 4,096 → 163,840 (s = 40). Llama 3.1 8,192 → 131,072 (llama3, s = 8).
- **Temperature is free:** logits × $(0.1\ln s+1)^2$, folded into cos/sin or the softmax scale. The llama3 type has none.
- **Variants:** LongRoPE searches one factor per pair (Phi-4-mini: 1.0 → 47.8). Dual Chunk Attention remaps distances per chunk, training-free (Qwen3 inference).
:::

## Partial RoPE and p-RoPE: rotate only some dims {#partial}

::: meta
**Partial RoPE** 2022 · Black et al., EleutherAI (GPT-NeoX-20B) · [arXiv:2204.06745](https://arxiv.org/abs/2204.06745) — **p-RoPE** 2024 · Barbero et al., Oxford / Google DeepMind · ICLR 2025 · [arXiv:2410.06205](https://arxiv.org/abs/2410.06205) · Used in: GLM-4.5, Qwen3-Next, MiniMax-M2, MiMo-V2-Flash, Phi-4-mini, Gemma 4, DeepSeek-V4
:::

$$
q_t^{\top}k_j\;\to\;(R_t\,q^{\text{rot}}_t)^{\top}(R_j\,k^{\text{rot}}_j)+(q^{\text{nope}}_t)^{\top}k^{\text{nope}}_j,\qquad d_{\text{rot}}=p\,d_h
$$

![Rotated versus position-free dimensions per head, from configs: GPT-NeoX-20B 24 of 96, GLM-4.5 64 of 128, MiniMax-M2 64 of 128, Phi-4-mini 96 of 128, MiMo-V2-Flash 64 of 192, Qwen3-Next 64 of 256, Gemma 4 global 128 of 512, DeepSeek-V4 64 of 512, DeepSeek-V3 MLA lane 64 of 192. At p = 0.25, Qwen3-Next keeps wavelengths from 6 tokens to 38M; Gemma 4's p-RoPE keeps 6 to 188 tokens and freezes 192 pairs. Reported Gemma 2B perplexity at 8K: NoPE 4.86, RoPE 4.46, 0.25-RoPE 4.53, 0.75-RoPE 4.44.](../figures/cards/pos-partial.svg)

::: tips
- **Compare:** NeoX-style partial squeezes the whole spectrum into $p\,d_h$ dims. p-RoPE keeps the fastest $p$ of the full spectrum and freezes the rest (θ = 0).
- **Why it works:** slow pairs barely turn within the context and already act as content channels. p-RoPE makes them exactly position-free.
- **Cache link:** unrotated dims are position-free, so they can be absorbed or shared: [MLA's decoupled RoPE lane](04-latent-kv.md#rope), K=V in Gemma 4 and DeepSeek-V4.
:::

## NoPE layers: iRoPE and RNoPE {#nope}

::: meta
**NoPE** 2023 · Kazemnejad et al. · NeurIPS 2023 · [arXiv:2305.19466](https://arxiv.org/abs/2305.19466) — **iRoPE** 2025 · Meta (Llama 4) — **RNoPE** 2025 · Yang et al., Cohere · [arXiv:2501.18795](https://arxiv.org/abs/2501.18795) · Used in: Llama 4, Command A, SmolLM3
:::

$$
e^{\text{RoPE}}_{tj}=\frac{(R_t q_t)^{\top}(R_j k_j)}{\sqrt{d_h}}+m^{\text{local}}_{tj},\qquad e^{\text{NoPE}}_{tj}=\frac{q_t^{\top}k_j}{\sqrt{d_h}}+m^{\text{causal}}_{tj}
$$

![Llama 4 repeats three chunked RoPE layers (chunk 8,192) and one full-attention NoPE layer; toy masks use 16 tokens and chunk 4. Llama 4 Scout has 36 RoPE and 12 NoPE layers of 48; SmolLM3-3B has 27 RoPE and 9 NoPE of 36, all full attention; Command A repeats three sliding-window RoPE layers (W = 4,096) and one full NoPE layer. Llama 4 Scout's KV cache at 128K in BF16 is 7.1 GiB with iRoPE versus 24 GiB if all layers were global.](../figures/cards/pos-nope.svg)

::: tips
- **Why it works:** the causal mask alone leaks order. RoPE layers supply recency; NoPE layers retrieve distant tokens (RNoPE §2).
- **Cache:** Llama 4's RoPE layers are [chunked](05-fixed-patterns.md#chunked), so only 12 of 48 layers grow with N.
- **Gotcha:** raising θ in the hybrid's RoPE layers hurt retrieval. Needle attention in NoPE layers fell 0.077 → 0.037 (reported).
- **Not solved:** NoPE logits flatten as context grows. Llama 4 adds a temperature to exactly these layers ([next card](#temperature)).
:::

## Length-dependent logit temperature {#temperature}

::: meta
**SSMax** 2025 · Nakanishi · [arXiv:2501.19399](https://arxiv.org/abs/2501.19399) — **Attention temperature tuning** 2025 · Meta (Llama 4 config) · Used in: Llama 4 (NoPE layers); YaRN models use a constant factor
:::

$$
\operatorname{softmax}_j\big(\tau\,e_{tj}\big),\qquad \tau_{\text{Llama 4}}(t)=1+0.1\ln\Big(1+\Big\lfloor\frac{t+1}{8192}\Big\rfloor\Big),\qquad \tau_{\text{SSMax}}=s_h\ln n
$$

![Toy: one key 3 logits above n − 1 equal keys. Plain softmax's top weight falls from 0.020 at 1K keys to 2e-6 at 10M; Llama 4's temperature slows the fall; SSMax keeps it near 0.8. From configs, Llama 4 multiplies queries by 1.07 at 8K, 1.28 at 128K, 1.49 at 1M and 1.72 at 10M, while YaRN applies a constant 1.81 (gpt-oss) or 1.87 (DeepSeek-V3) to every position.](../figures/cards/pos-temperature.svg)

::: tips
- **Why:** with $n$ keys and a fixed gap Δ, top weight ≈ $e^{\Delta}/n$. Attention fades as context grows (SSMax §2).
- **Numbers:** Llama 4 τ = 1.07 at 8K, 1.28 at 128K, 1.72 at 10M (`attn_scale` 0.1, `floor_scale` 8,192).
- **Compare:** SSMax learns $s_h$ per layer and head (144 parameters at 162M). YaRN applies one constant to every position.
- **History · ALiBi (ICLR 2022):** no rotation; subtract $m_h(t-j)$, slopes $m_h=2^{-8h/H}$. Trained on 1,024 tokens, it matched sinusoidal trained on 2,048.
:::
