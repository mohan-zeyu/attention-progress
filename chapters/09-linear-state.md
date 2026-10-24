# 09 · Linear attention & SSMs

::: lede
The KV cache grows with every token. Drop the softmax and the whole history folds into one fixed $d_v\times d_k$ matrix per head.
:::

## Linear attention: an RNN with a matrix state {#linear-attention}

::: meta
**Linear Transformer** 2020 · Katharopoulos, Vyas, Pappas, Fleuret (Idiap, EPFL) · ICML 2020 · [arXiv:2006.16236](https://arxiv.org/abs/2006.16236) · Used in: the base of every layer in chapters 09–12
:::

$$
o_t=\sum_{j\le t}v_j\,(k_j^{\top}q_t)=\Big(\underbrace{\sum_{j\le t}v_jk_j^{\top}}_{S_t\,\in\,\mathbb{R}^{d_v\times d_k}}\Big)q_t,
\qquad S_t=S_{t-1}+v_tk_t^{\top}
$$

![Softmax attention forms an N×N score matrix and adds one KV record per token. Linear attention reassociates (QKᵀ)V into Q(KᵀV) and keeps one 128×128 state per head: 16,384 numbers, against 2·N·128 for the KV cache, which is 1.05M at 4K, 33.6M at 128K and 268M at 1M tokens. Break-even is N = 64.](../figures/cards/linear-reassociate.svg)

::: tips
- **Why it works:** without the softmax, $(QK^\top)V=Q(K^\top V)$. Decode costs $O(d_vd_k)$ per token at any $N$.
- **Gotcha:** this is a different model, not a faster softmax. The 2020 version used a feature map $\phi$ and a normalizer $z_t^{\top}\phi(q_t)$.
- **Normalizer dropped:** it made gradients unstable. Modern layers use L2-normalized $q,k$ and output RMSNorm instead. mLSTM still keeps one.
- **Not solved:** pure addition never forgets or overwrites. Every later card changes how $S$ is updated.
:::

## RetNet → Lightning: one fixed decay per head {#retnet}

::: meta
**RetNet** 2023 · Sun et al., Microsoft Research & Tsinghua · [arXiv:2307.08621](https://arxiv.org/abs/2307.08621) — **Lightning attention** 2025 · MiniMax · [arXiv:2501.08313](https://arxiv.org/abs/2501.08313) · Used in: MiniMax-Text-01, MiniMax-M1
:::

$$
S_t=\gamma_hS_{t-1}+v_tk_t^{\top}
\quad\Leftrightarrow\quad
O=(QK^{\top}\odot D)V,\qquad D_{tj}=\gamma_h^{\,t-j}\ \ (j\le t)
$$

![Left: toy decay masks D for γ = 0.7 and γ = 0.95. RetNet's experiments space 1−γ from 1/32 to 1/512, horizons of about 32 to 512 tokens. Right: MiniMax-01 lightning decay λ = exp(−s_h(1−ℓ/79)) with s_h = 2^(−h/8); horizons 1/(1−λ) run from 1.7 tokens (head 1, layer 0) to about 20,000 tokens (head 64, layer 78). Bottom: 80 layers, one softmax layer after every seven lightning layers.](../figures/cards/linear-fixed-decay.svg)

::: tips
- **Why it works:** one mask $D$ gives three equal forms: parallel for training, recurrent for decode, chunkwise for both. Each head gets a horizon of about $1/(1-\gamma)$ tokens.
- **Numbers:** MiniMax-01 uses ALiBi slopes as decay rates and weakens them with depth (`modeling_minimax_text_01.py`). 70 lightning + 10 softmax layers.
- **Not solved:** the decay ignores content. An order number and a greeting fade at the same rate.
- **Gotcha:** MiniMax-M2 went back to full attention ([ch 12](12-hybrids.md)).
:::

## GLA: data-dependent decay per key channel {#gla}

::: meta
**Gated Linear Attention** 2023 · Yang, Wang, Shen, Panda, Kim (MIT, MIT-IBM Watson AI Lab) · ICML 2024 · [arXiv:2312.06635](https://arxiv.org/abs/2312.06635) · Used in: GLA research models; the same per-channel decay drives KDA in Kimi Linear
:::

$$
S_t=S_{t-1}\operatorname{Diag}(\alpha_t)+v_tk_t^{\top},\qquad
\alpha_t=\sigma(W_\alpha x_t+b)^{1/16}\in(0,1)^{d_k}
$$

![A toy 4×6 state times Diag(α_t) with α = 1, 0.95, 0.5, 0.9, 0.1, 0.99: each key-channel column fades at its own rate, and channel 5 is nearly wiped. A scalar decay (Mamba-2, mLSTM) scales every column by the same 0.8. The 1/16 temperature maps σ = 0.01, 0.1, 0.5, 0.9 to α = 0.75, 0.866, 0.958, 0.993.](../figures/cards/linear-gla.svg)

::: tips
- **Why it works:** the gate depends on the token, so a channel can be cleared at a topic switch and kept otherwise.
- **Gotcha:** GLA gates key channels (columns) only. Adding a value-side gate gave "only marginal improvements" (§4.4) and was not used.
- **Compare:** Mamba-2 and mLSTM use one scalar per head (coarser). HGRN2 also decays per key channel. Mamba-1 decays each entry, which has no matmul form.
- **Cost:** a diagonal decay still chunks well: cumulative products of $\alpha$ become cumulative sums in log space.
:::

## Mamba → Mamba-2: the state-space duality {#ssd}

::: meta
**Mamba-2 (SSD)** 2024 · Dao, Gu (Princeton, CMU) · ICML 2024 · [arXiv:2405.21060](https://arxiv.org/abs/2405.21060) · Used in: NVIDIA Nemotron-H, Nemotron 3 Nano, IBM Granite 4.0-H
:::

$$
S_t=a_tS_{t-1}+\Delta_t\,x_tB_t^{\top},\quad y_t=S_tC_t
\quad\Leftrightarrow\quad
Y=(L\circ CB^{\top})\,\Delta X,\quad L_{tj}=\prod_{s=j+1}^{t}a_s
$$

![Left: an 8×8 SSD mask L with L_tj = a_(j+1)…a_t for toy decays 0.9, 0.2, 0.95, 0.95, 0.9, 0.5, 0.95; the small a₃ = 0.2 cuts every path that crosses token 3. Middle: x↔v, B↔k, C↔q, N↔d_k, P↔d_v. Right: Nemotron 3 Nano has 64 heads of 64×128 state, 524,288 numbers per Mamba-2 layer, 8 B/C groups and chunk 128; its 52 layers are 23 Mamba-2, 23 MoE and 6 attention.](../figures/cards/linear-ssd.svg)

::: tips
- **Dictionary:** $x\to v$, $B\to k$, $C\to q$, $a_t=e^{\Delta_tA_h}$ with $A_h<0$. The mask $L$ replaces both softmax and positional encoding.
- **Gotcha:** the decay is one scalar per head, coarser than GLA or Mamba-1. That buys matmuls: reported 2–8× faster than Mamba's scan.
- **Gotcha:** $\Delta_t$ sets both decay and write strength, so a strong write also forgets more. Gated DeltaNet splits them ([ch 10](10-delta-rule.md#gdn)).
- **Used in:** Granite 4.0-H-Small has 36 Mamba-2 and 4 attention layers (config).
:::

## Mamba-3: trapezoid write, rotations, MIMO {#mamba3}

::: meta
**Mamba-3** 2026 · Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu · ICLR 2026 · [arXiv:2603.15569](https://arxiv.org/abs/2603.15569) · Used in: no released LLM found (Oct 2026)
:::

$$
S_t=\alpha_tS_{t-1}+\beta_t\,x_{t-1}B_{t-1}^{\top}+\gamma_t\,x_tB_t^{\top},\qquad
\alpha_t=e^{\Delta_tA_t},\ \ \beta_t=(1-\lambda_t)\Delta_t\alpha_t,\ \ \gamma_t=\lambda_t\Delta_t
$$

![Three changes to Mamba-2. (a) With toy Δ = 0.5 and α = 0.8, Mamba-2 (λ = 1) writes 0.5 of x_t; the trapezoid rule (λ = ½) writes 0.2 of x_(t−1) and 0.25 of x_t. (b) A complex state turns the real decay α into α·e^(±iθ), a rotation equivalent to data-dependent RoPE on B and C. (c) MIMO writes (P×R)(R×N) with R = 4: about 4× the decode FLOPs on the same state.](../figures/cards/linear-mamba3.svg)

::: tips
- **Trapezoid:** $\lambda=1$ is Mamba-2's Euler step; $\lambda=\tfrac12$ averages both ends. The short convolution is removed.
- **Rotations:** eigenvalues leave the real line, which parity needs ([ch 10](10-delta-rule.md#state-tracking)). Without the RoPE the reported parity gain disappears.
- **MIMO:** rank-$R$ writes add FLOPs per state byte, not state. Decode is memory-bound, so reported latency stays about flat.
:::

| 1.5B models, 100B FineWeb-Edu tokens (reported) | Perplexity | Avg. accuracy |
| --- | --- | --- |
| Transformer | 10.51 | 55.4 |
| Mamba-2 | 10.47 | 55.7 |
| Gated DeltaNet | 10.45 | 55.8 |
| Mamba-3 SISO | 10.35 | 56.4 |
| Mamba-3 MIMO | 10.24 | 57.6 |

## Chunkwise form: attention inside, state between {#chunkwise .insight}

$$
O_{[c]}=\underbrace{(Q_{[c]}K_{[c]}^{\top}\odot M)\,V_{[c]}}_{\text{inside chunk}}+\underbrace{Q_{[c]}S_{[c-1]}^{\top}}_{\text{earlier chunks}},
\qquad S_{[c]}=S_{[c-1]}+V_{[c]}^{\top}K_{[c]}
$$

![Sixteen toy tokens in chunks of 4: only the diagonal 4×4 blocks are computed as small masked attention, and everything below them is read from a state passed between chunks. Our count per head with d = 128, C = 64 and N = 32K: chunkwise 98,304 FLOPs per token in 512 sequential steps, recurrent 65,536 FLOPs per token in 32,768 matrix-vector steps, quadratic 8.4M FLOPs per token.](../figures/cards/linear-chunkwise.svg)

::: tips
- **Cost:** $O(NCd+Nd^2)$ instead of $O(N^2d)$, and all of it is matmul (GLA §2). In practice $C=64$–$256$.
- **Train vs. infer:** training and prefill use chunks; decode uses the one-token recurrence. Both give identical outputs.
- **Gotcha:** decays add cumulative-product factors inside each block. The delta rule needs a WY/UT transform ([ch 10](10-delta-rule.md#taxonomy)). Kernels are in [ch 13](13-kernels.md).
:::
