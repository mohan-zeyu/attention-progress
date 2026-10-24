# 02 · Stable softmax

::: lede
Softmax must sum to one, and its logits can grow without bound. Modern layers cap the logits, give each head an exit, and gate its output.
:::

## QK-norm, soft-cap, QK-Clip: bound the logits {#qk-norm}

::: meta
**QK-norm** 2023 · Dehghani et al. (ViT-22B); logit-growth study Wortsman et al., Google DeepMind · ICLR 2024 · [arXiv:2309.14322](https://arxiv.org/abs/2309.14322) — **Soft-cap** 2024 · Gemma 2 · [arXiv:2408.00118](https://arxiv.org/abs/2408.00118) — **QK-Clip** 2025 · Kimi K2 · [arXiv:2507.20534](https://arxiv.org/abs/2507.20534) · Used in: Qwen3, Qwen3-Next, OLMo 2/3, Gemma 3/4, GLM-4.5, MiniMax-M2, Llama 4 Scout, DeepSeek-V4
:::

$$
\hat q=g_q\odot\frac{q}{\operatorname{RMS}(q)},\quad \hat k=g_k\odot\frac{k}{\operatorname{RMS}(k)}\;\Rightarrow\;\Big|\frac{\hat q^{\top}\hat k}{\sqrt{d_h}}\Big|\le\sqrt{d_h}\,\|g_q\|_\infty\|g_k\|_\infty
$$

![QK-norm bounds the logit by the square root of d_h times the gains: 11.3 for d_h = 128 and unit gains. In a toy with q and k grown fourfold, logits spread to about ±50 and the top weight over 64 keys is 0.99; with QK-norm it is 0.07. Gemma 2 soft-caps attention logits with 50·tanh(e/50), mapping 50 to 38 and 100 to 48, and final logits with c = 30. Kimi K2's QK-Clip (τ = 100) scales W_q and W_k of a head whose max logit is 250 by √0.4 = 0.63.](../figures/cards/stable-qk-norm.svg)

::: tips
- **Why logits grow:** $1/\sqrt{d_h}$ fixes the scale only at initialisation. Trained ‖q‖‖k‖ grow until attention turns one-hot (Wortsman §3.1).
- **Variants:** soft-cap $e\leftarrow c\tanh(e/c)$ shapes logits in the forward pass. QK-Clip rescales $W_q,W_k$ by $\sqrt{\gamma_h}$, $\gamma_h=\min(1,\tau/S^h_{\max})$, after the optimizer step.
- **MLA gotcha:** absorbed MLA never materialises keys, so Kimi K2 uses QK-Clip instead of QK-norm.
- **Not free:** Cohere's RNoPE design (Command A) dropped QK-norm: it flattened attention and hurt long-context retrieval.
:::

## Attention sinks → a learned sink logit {#sinks}

::: meta
**Attention sinks** 2023 · Xiao et al., MIT / Meta · ICLR 2024 · [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) — **Learned sink logit** 2025 · OpenAI (gpt-oss) · [arXiv:2508.10925](https://arxiv.org/abs/2508.10925) · Used in: gpt-oss, MiMo-V2-Flash (SWA layers), DeepSeek-V4
:::

$$
\alpha_{tj}=\frac{\exp(e_{tj})}{\exp(z_h)+\sum_{r\le t}\exp(e_{tr})},\qquad \sum_{j\le t}\alpha_{tj}=1-\frac{\exp(z_h)}{\exp(z_h)+\sum_{r\le t}\exp(e_{tr})}<1
$$

![Toy 12-token causal attention. With vanilla softmax, rows with nothing to read put their slack on token 0 (0.34 in row 11). With a learned sink logit, an extra column outside the sequence absorbs the slack (0.24 in row 11) and token 0 gets 0.02. One query with six equal logits: vanilla gives each key 0.167; with z_h = 3 the sink takes 0.77 and each key 0.038, so the output shrinks to 0.23 of the mean value. Cost is one scalar per head, 2,304 parameters in gpt-oss-120b. MiMo-V2-Flash at window 128 reports MMLU 54.9 → 58.3 and BBH 52.4 → 56.1 with the sink.](../figures/cards/stable-sinks.svg)

::: tips
- **Why sinks appear:** weights must sum to 1, so a head with nothing to read parks mass on early tokens. StreamingLLM keeps 4 of them ([ch. 05](05-fixed-patterns.md#streaming)).
- **The fix:** one learned logit $z_h$ per head joins the denominator. It has no value vector, so the head can attend to nothing.
- **Cost:** one scalar per head. gpt-oss puts it in every layer; MiMo-V2-Flash only in SWA layers; DeepSeek-V4's kernel takes `attn_sink`.
- **Compare:** an output gate ([next card](#gated)) also removes the sink, without changing the softmax.
:::

## Gated attention: a sigmoid gate after SDPA {#gated}

::: meta
**Gated attention** 2025 · Qiu et al., Qwen (Alibaba) · NeurIPS 2025 Best Paper · [arXiv:2505.06708](https://arxiv.org/abs/2505.06708) · Used in: Qwen3-Next, Qwen3.5 (`attn_output_gate`)
:::

$$
Y'=Y\odot\sigma(XW_\theta),\qquad Y=\operatorname{softmax}\Big(\frac{QK^{\top}}{\sqrt{d_h}}\Big)V,\qquad \text{out}=W_O\,Y'
$$

![Qwen3-Next's q_proj outputs 16 heads × 256 of queries plus 4,096 gate logits from the same 2,048-dim input. The SDPA output Y is multiplied elementwise by the sigmoid gate before o_proj maps 4,096 to 2,048. In a toy head most gate values are near 0. Reported for a 15B MoE: first-token attention 46.7% → 4.8%, max activation 1,053 → 94, perplexity 6.026 → 5.761, RULER at 128K 31.65 → 58.82. An output gate changes one token's read; a state gate changes every later read.](../figures/cards/stable-gated.svg)

::: tips
- **Why it works:** a nonlinearity between $W_V$ and $W_O$, plus query-dependent sparsity (mean gate 0.116, reported).
- **Cost:** elementwise gating adds ≈201M parameters to the 15B MoE (headwise: 1.6M) and under 2% latency (reported).
- **Gate location:** this gate rescales one token's read. A state gate ($\alpha_t$ in [Gated DeltaNet](10-delta-rule.md)) changes what every later query reads.
- **Gotcha:** a gate near 0 does not skip work. SDPA has already run; only its output is suppressed.
:::

## Research knobs: forget gate and differential attention {#knobs}

::: meta
**FoX** 2025 · Lin et al., Mila · ICLR 2025 · [arXiv:2503.02130](https://arxiv.org/abs/2503.02130) — **Diff Transformer** 2024 · Ye et al., Microsoft Research / Tsinghua · ICLR 2025 (oral) · [arXiv:2410.05258](https://arxiv.org/abs/2410.05258) · Used in: Phi-4-mini-flash-reasoning (differential attention)
:::

$$
\begin{aligned}
\text{FoX: }& e_{tj}=\frac{q_t^{\top}k_j}{\sqrt{d_h}}+\sum_{l=j+1}^{t}\ln f_l,\qquad f_l=\sigma(w^{\top}x_l+b)\\
\text{Diff: }& o_t=\Big(\operatorname{softmax}\big(e^{(1)}_t\big)-\lambda\,\operatorname{softmax}\big(e^{(2)}_t\big)\Big)V
\end{aligned}
$$

![Toy decay bias over 10 tokens. A constant forget gate (f = 0.8, ALiBi-like) decays every row the same way; FoX's learned gates (0.97, with 0.05 at token 5) keep the past until token 5 and then wipe everything before it. Toy differential attention over 10 keys: softmax 1 gives 0.46 to key 6 and 0.06 elsewhere, λ·softmax 2 with λ = 0.6 gives 0.06 everywhere, so the difference keeps 0.40 on key 6 and cancels the rest. λ_init = 0.8 − 0.6·exp(−0.3(ℓ − 1)) is 0.20 at layer 1, 0.56 at layer 4 and 0.79 at layer 16.](../figures/cards/stable-knobs.svg)

::: tips
- **FoX bridges to recurrence:** $f\equiv1$ gives NoPE softmax, $f\equiv e^{-m}$ gives ALiBi, and a learned $f$ is a forget gate as in [GLA](09-linear-state.md#gla).
- **FoX kernel:** FlashAttention-compatible. Keep $c_t=\sum_{l\le t}\ln f_l$ and add $c_t-c_j$ per tile. No positional embedding needed.
- **Diff numbers:** per-head RMSNorm × $(1-\lambda_{\text{init}})$. Reported: a 6.8B Diff Transformer matches an 11B Transformer's loss.
- **Cost:** two QK maps per head (heads halved to keep parameters). No KV-cache saving.
:::

## 2026 anatomy: which trick sits where {#anatomy .insight}

$$
y=W_O\Big[\sigma(XW_\theta)\odot\sum_j\operatorname{softmax}_{j\,\cup\,\text{sink}}\Big(\frac{\operatorname{RoPE}_p(\hat q_t)^{\top}\operatorname{RoPE}_p(\hat k_j)}{\sqrt{d_h}}+m^{W}_{tj}\Big)v_j\Big]
$$

![Eight stages of a 2026 attention block and what four released models put at each, from configs and code. gpt-oss-120b: 8 KV heads × 64, full RoPE at 150K with YaRN ×32, 128-token window alternating 1:1 with full layers, a sink per head. Gemma 4 31B: K=V in global layers (4 × 512), q and k RMS-norm with scale 1.0, p-RoPE ¼ at 1M in global layers and 10K in local ones, window 1,024 at 5:1, V-norm. DeepSeek-V4-Pro: one 512-dim K=V entry (MQA), q RMS and kv-norm, 64 of 512 dims rotated, window 128 plus compressed entries, a sink per head, de-rotated output, 16-group low-rank output projection. Qwen3-Next-80B: 2 KV heads × 256, per-head q and k RMS-norm, ¼ of 256 dims rotated at 10M, full attention in 1 of 4 layers, elementwise output gate.](../figures/cards/stable-anatomy.svg)

::: tips
- **Read it as a checklist:** QK-norm ([card](#qk-norm)), partial RoPE ([01](01-positions.md#partial)), window ([05](05-fixed-patterns.md#swa-global)), sink ([card](#sinks)), K=V ([03](03-kv-heads.md#kv-tie)), gate ([card](#gated)).
- **Sink or gate:** both let a head output almost nothing. None of these four models uses both.
- **K=V needs partial RoPE:** the shared vector is also the value, so only a slice rotates (Gemma 4 ¼; DeepSeek-V4 64/512, output de-rotated).
:::
