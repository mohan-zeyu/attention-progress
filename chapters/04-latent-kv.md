# 04 · Latent KV

::: lede
Head sharing throws away per-head views. MLA keeps all query heads yet caches one 576-element latent per token, trading cache bytes for decode FLOPs.
:::

## MLA: cache a latent, rebuild the heads {#mla}

::: meta
**MLA** 2024 · DeepSeek-AI · DeepSeek-V2 · [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) · Used in: DeepSeek-V2, V3, V3.2, Kimi K2, GLM-5, Mistral Large 3
:::

$$
c_t=W_{DKV}h_t\in\mathbb{R}^{d_c},\quad k^R_t=\operatorname{RoPE}_t(W_{KR}h_t)\in\mathbb{R}^{d_r},\quad k^C_{t,i}=W_{UK,i}\,c_t,\quad v_{t,i}=W_{UV,i}\,c_t
$$

![MLA record in DeepSeek-V3. h_t (7168) is projected to a latent c_t (512, by W_DKV 512 × 7168) and a shared RoPE key k^R (64); only these 576 elements are cached per token and layer. W_UK and W_UV (16384 × 512 each) rebuild per-head keys and values, 128 heads × 128, never cached; each head's key is 128 + 64 = 192. Elements per token per layer with V3 shapes: MHA 40,960 (80 KiB), GQA-8 2,048, MLA 576 (1,152 B), K=V MQA 512. All 61 layers: 70,272 B = 68.6 KiB per token.](../figures/cards/mla-record.svg)

::: tips
- **Numbers (V3 `config.json`):** $d_c$ 512, $d_r$ 64, 128 heads × 128, 61 layers → 61 × 576 × 2 B = 68.6 KiB per token.
- **Compare:** 576 = 4.5 $d_h$, the cache of GQA with 2.25 groups (V2 Table 1), yet every head keeps its own K and V.
- **Query is low-rank too:** $c^Q_t=W_{DQ}h_t\in\mathbb{R}^{1536}$, not cached.
- **Cost:** the FP8 layout (656 B per token per layer) is in [Serving the cache](14-serving.md).
:::

## Weight absorption: decode is MQA over 576 dims {#absorb}

::: meta
**Absorbed MLA** 2024 · DeepSeek-AI · DeepSeek-V2 §2.1 · [arXiv:2405.04434](https://arxiv.org/abs/2405.04434); MHA vs MQA mode: DeepSeek-V3.2 App. A · [arXiv:2512.02556](https://arxiv.org/abs/2512.02556) · Used in: DeepSeek-V3.x decode, FlashMLA
:::

$$
q_{t,i}^{C\top}\,W_{UK,i}\,c_j=\big(W_{UK,i}^{\top}q^C_{t,i}\big)^{\top}c_j,\qquad o_{t,i}=W_{UV,i}\sum_j p_{t,i,j}\,c_j
$$

![Weight absorption. In MHA mode (training and prefill) each of 128 heads expands the cache row to a 192-dim key and 128-dim value, 40,960 elements per token. In MQA mode (decode) W_UK folds into the query, giving 576-dim queries (512 + 64), and all 128 heads read one shared row: key 576, value 512; W_UV folds into W_O. Per cached token and layer in BF16: 1,152 B read for 278,528 FLOPs, about 242 FLOP/B versus 1 for an expanded cache; H100 ridge 295. Toy check: 7 × 23 = 161 = 14 × 4 + 21 × 5.](../figures/cards/mla-absorb.svg)

::: tips
- **Exact:** only the bracketing changes, so the absorbed score is the same number (toy check in the figure).
- **Train vs. infer:** MHA mode for training and prefill, MQA mode for decode (V3.2 App. A, stated for V3.1-Terminus).
- **Cost:** 1,152 B feed 278,528 FLOPs: $I\approx 2H_q\approx 242$, near the H100 ridge of 295.
- **Gotcha:** keys never exist at decode, so QK-norm cannot apply; Kimi K2 clips weights instead ([Stable softmax](02-stable-softmax.md)).
:::

## Decoupled RoPE: a separate 64-dim lane {#rope}

::: meta
**Decoupled RoPE** 2024 · DeepSeek-AI · DeepSeek-V2 §2.1.3 · [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) · Used in: DeepSeek-V2, V3, V3.2, Kimi K2, GLM-5, Mistral Large 3
:::

$$
e_{t,j,i}=\frac{\big(W_{UK,i}^{\top}q^C_{t,i}\big)^{\top}c_j+\big(q^R_{t,i}\big)^{\top}k^R_j}{\sqrt{d_h+d_r}},\qquad k^R_j=\operatorname{RoPE}_j(W_{KR}h_j)\in\mathbb{R}^{64}
$$

![Top: with RoPE on the content key the score is q^T R_(j−t) W_UK c_j; the rotation depends on the key position j and sits between the query and W_UK, so W_UK cannot be folded into the query once. Bottom: MLA's fix. Each head's query is an absorbed 512-dim content part plus a 64-dim rotated part; the cached row is c_j (512, no RoPE) plus one rotated k^R_j (64) shared by all 128 heads; the score sums both parts and divides by √192. A shared RoPE key costs 64 elements per token, per-head rotated keys would cost 8,192; the RoPE lane is 11% of the 576 cached.](../figures/cards/mla-rope.svg)

::: tips
- **Why it breaks:** $R_{j-t}$ sits between $q$ and $W_{UK}$ and changes with $j$; matrix products do not commute (V2 §2.1.3).
- **Why it is cheap:** one rotated key serves all 128 heads: 64 elements per token, not 8,192.
- **Not the only fix:** Kimi Linear and Kimi K3 run their MLA layers with NoPE (`mla_use_nope`); DeepSeek-V4 rotates 64 dims of a [K=V entry](03-kv-heads.md#kv-tie).
- **Positions:** RoPE base and scaling are in [Positions](01-positions.md).
:::

## Head count: heads cost FLOPs, not cache {#heads .insight}

$$
\text{cache/token/layer}=d_c+d_r=576\ \text{for any }H_q,\qquad \text{decode FLOPs/cached token}=2H_q\,(576+512)
$$

![DeepSeek-V3 (128 heads) and Kimi K2 (64 heads) cache the same 576 elements per token and layer. Decode FLOPs per cached token per layer: 278,528 versus 139,264; intensity in BF16 242 versus 121 FLOP/B, against the H100 ridge of 295. Kimi K2 reports +83% inference FLOPs at 128K for 128 instead of 64 heads, with only 0.5–1.2% lower validation loss.](../figures/cards/mla-heads.svg)

::: tips
- **Invisible in the cache:** Kimi K2 and DeepSeek-V3 store identical 576-element rows over 61 layers.
- **Reported (Kimi K2 §2.3):** 64 → 128 heads: +83% inference FLOPs at 128K, 0.5–1.2% lower validation loss. K2 kept 64.
- **Not solved:** one latent per token cannot be split across tensor-parallel ranks; each holds all 576 ([GLA](03-kv-heads.md#beyond-gqa)).
:::

| Model | $L$ | $H_q$ | $d_c+d_r$ | KV per token (BF16) |
| --- | --- | --- | --- | --- |
| DeepSeek-V2 | 60 | 128 | 512 + 64 | 67.5 KiB |
| DeepSeek-V3 / V3.2 | 61 | 128 | 512 + 64 | 68.6 KiB |
| Mistral Large 3 | 61 | 128 | 512 + 64 | 68.6 KiB |
| Kimi K2 | 61 | 64 | 512 + 64 | 68.6 KiB |
| GLM-5 | 78 | 64 | 512 + 64 | 87.8 KiB |

## Converting MHA/GQA → MLA {#convert}

::: meta
**MHA2MLA** 2025 · Ji et al. · ACL 2025 · [arXiv:2502.14837](https://arxiv.org/abs/2502.14837) — **TransMLA** 2025 · Meng et al. · [arXiv:2502.07864](https://arxiv.org/abs/2502.07864) · Used in: research checkpoints (SmolLM, Llama-2-7B)
:::

$$
\big[W_K^{\text{nope}};\,W_V\big]\approx U_r\Sigma_rV_r^{\top},\qquad W_{DKV}=\Sigma_r^{1/2}V_r^{\top},\quad \big[W_{UK};\,W_{UV}\big]=U_r\Sigma_r^{1/2}
$$

![Three steps to convert Llama-2-7B attention (W_K 4096 × 4096, 32 heads × 128) to MLA. 1: keep RoPE on a few dims (MHA2MLA: top-norm frequency pairs per head; TransMLA: fold RoPE into one head); the rest become NoPE and absorbable. 2: joint SVD or PCA of the stacked NoPE key and value projections gives W_DKV (r × 4096). 3: fine-tune with 0.3–0.6% of pre-training data (MHA2MLA) or 6B tokens (TransMLA). KV bytes per token per layer: MHA 16 KiB; MHA2MLA 5 KiB (−68.75%); with Int4 1.25 KiB (−92.19%); TransMLA 1,152 B, i.e. 576 elements (−92.97%).](../figures/cards/mla-convert.svg)

::: tips
- **Converts from:** MHA (MHA2MLA) or GQA (TransMLA). Rotated dims go first: they [cannot be absorbed](#rope).
- **Numbers (Llama-2-7B):** MHA2MLA −68.75% in BF16, −92.19% with Int4, LongBench 27.4 → 26.4. TransMLA −92.97% after 6B tokens; reported 10.6× at 8K.
- **Not lossless:** the factorisation changes the function; short fine-tuning recovers it.
- **Compare:** [GQA uptraining](03-kv-heads.md#gqa) mean-pools heads; conversion keeps every head and re-factors its projections.
:::
