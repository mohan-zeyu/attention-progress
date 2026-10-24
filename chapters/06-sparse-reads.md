# 06 · Sparse reads

::: lede
Keep every record, but let each query read only a few: a cheap scorer picks them, and exact attention runs on the picks.
:::

## What shrinks: reads or records? {#what-shrinks .insight}

$$
\text{HBM}=N_{\text{stored}}\cdot b_{\text{rec}},
\qquad
\text{bytes per step}=N_{\text{read}}\cdot b_{\text{rec}}+\text{scorer}
$$

![A 2×2 map: reads shrink and reversible holds sparse reads (06: Quest, MoBA, NSA, DSA, MSA); records leave HBM but can return holds offload (14); reads shrink but fixed holds static patterns (05); records shrink irreversibly holds eviction (07), compression (08) and sliding-window layers (05). Llama-3-8B at 128K in BF16: reading 2,048 records keeps 16 GiB in HBM and moves 256 MiB per step; keeping 2,048 records stores and moves 256 MiB.](../figures/cards/sparse-what-shrinks.svg)

::: tips
- **Reversible:** a later query can still reach every record. Only sparse reads and offload keep that property.
- **Cost:** sparse reads cut bandwidth, not memory. The scorer still touches all $N$ records per query (last card).
- **Compare:** fixed masks [05](05-fixed-patterns.md) · eviction [07](07-kv-eviction.md) · fewer records [08](08-fewer-records.md) · offload [14](14-serving.md).
:::

## Training-free prefill: MInference → XAttention {#prefill}

::: meta
**MInference** 2024 · Jiang et al., Microsoft · NeurIPS 2024 · [arXiv:2407.02490](https://arxiv.org/abs/2407.02490) — **FlexPrefill** 2025 · ICLR 2025 · [arXiv:2502.20766](https://arxiv.org/abs/2502.20766) — **XAttention** 2025 · MIT · ICML 2025 · [arXiv:2503.16428](https://arxiv.org/abs/2503.16428) · Used in: Qwen2.5-1M
:::

$$
M_{ij}=1 \iff j\in V\ \lor\ (i-j)\in S,
\qquad V,S=\text{top columns / diagonals of }\operatorname{softmax}(Q_{-64:}K^{\top})
$$

![Toy 32×32 masks for MInference's three head patterns: A-shape (1,024 first + 4,096 local tokens, static), Vertical-Slash (estimated from the last 64 queries), Block-Sparse (64×64 mean-pooled blocks, top 100). XAttention scores a block by its antidiagonal cells, 6.89% density at 128K. Reported 1M-token prefill on one A100: 30 minutes dense, 3 minutes MInference; Qwen2.5-1M 3–7× faster at 1M.](../figures/cards/sparse-prefill.svg)

::: tips
- **Prefill only:** the full KV cache is still written, and decode is untouched.
- **Per head:** an offline search fixes each head's pattern; indices are rebuilt for every input.
- **Threshold beats fixed k:** FlexPrefill (JS-divergence test) and XAttention (antidiagonal sums) keep blocks until their mass reaches $\tau$.
- **Used in:** Qwen2.5-1M serves Vertical-Slash prefill: reported 3–7× faster at 1M tokens.
:::

## Quest: page bounds choose what decode reads {#quest}

::: meta
**Quest** 2024 · Tang et al., MIT Han Lab · ICML 2024 · [arXiv:2406.10774](https://arxiv.org/abs/2406.10774) · Used in: training-free decode on any MHA/GQA model
:::

$$
U_p=\sum_{i=1}^{d}\max\big(q_i\,m_{p,i},\;q_i\,M_{p,i}\big)\;\ge\;\max_{j\in p}q^{\top}k_j,
\qquad \text{read }\operatorname{Top}_K(U)\text{ pages}
$$

![Toy query (2, −1, 1, 1) against four pages of four keys with per-channel min and max: bounds U = 6, 4, 1, 5 and true best scores 6, 3, −1, 3, so pages A and D are read. Real scale: 32K tokens in 16-token pages is 2,048 pages; a 2,048-token budget reads 128 (6.25%); metadata adds 12.5% to K. Reported 7.03× self-attention speedup at 32K.](../figures/cards/sparse-quest.svg)

::: tips
- **Why it works:** each channel's product peaks at an interval end, so $U_p$ never underestimates a page.
- **Selection, not eviction:** memory grows by 12.5% of K; bandwidth shrinks.
- **Gotcha:** wide pages loosen the bound (page D: bound 5, true 3). Hence 16-token pages.
- **Lineage:** cheap scorer → top-k → exact attention. [DSA](#dsa) trains the scorer instead.
:::

## MoBA: route each query to a few blocks {#moba}

::: meta
**MoBA** 2025 · Lu et al., Moonshot AI · [arXiv:2502.13189](https://arxiv.org/abs/2502.13189) · Used in: Kimi long-context service
:::

$$
s_{t,b}=q_t^{\top}\operatorname{mean}_{j\in b}k_j,
\qquad A_t=\operatorname{Top}_k(s_{t,:})\cup\text{current block (causal)}
$$

![Toy 32-token grid with 8 blocks of 4: each query block reads its own block causally plus the top-2 past blocks; the gate for block 7 picks blocks 2 and 4 by score. Mean pooling hides one sharp key: block X averages 0.55 despite a 3.6 peak, below uniform block Y at 0.95. Real settings read 18.75% at 8K (512 × top-3) and 4.69% at 1M (4,096 × top-12).](../figures/cards/sparse-moba.svg)

::: tips
- **No router weights:** the gate is $q$ against mean keys. Continued training adapts $q$ and $k$ to it.
- **Gotcha:** a mean dilutes one sharp key in a 4,096-token block. MSA max-pools instead ([last card](#indexers)).
- **Hybrid:** the last 3 layers stay full attention; SFT's masked prompt loss starved sparse layers of gradient.
- **Numbers:** reported 6.5× prefill at 1M; RULER 128K 0.7818 vs 0.7849 full.
:::

## NSA: compress, select, slide {#nsa}

::: meta
**NSA** 2025 · Yuan et al., DeepSeek-AI / PKU · ACL 2025 · [arXiv:2502.11089](https://arxiv.org/abs/2502.11089) · Used in: research 27B MoE; selection reused by InfLLM-v2 (MiniCPM4.1)
:::

$$
o_t=\sum_{c\in\{\text{cmp},\,\text{slc},\,\text{win}\}} g^c_t\,\operatorname{Attn}\big(q_t,\,K^c_t,\,V^c_t\big),
\qquad g^c_t=\sigma(\operatorname{MLP}(x_t))
$$

![One query at 64K reads 4,095 compressed records (32-token blocks, stride 16), 16 selected 64-token blocks (first block and 2 local blocks forced, 13 by score) and a 512-token window: 5,631 reads, 8.6% of 65,536. Block scores reuse the compression attention, summed over the 16 heads of a GQA group. Reported 9.0× forward and 6.0× backward at 64K versus FlashAttention-2.](../figures/cards/sparse-nsa.svg)

::: tips
- **Free scorer:** selection reuses the compression branch's attention scores, so no separate indexer.
- **Group-shared:** one top-16 per GQA group, so each KV block is loaded once for 16 heads.
- **Gotcha:** three branches with their own KV projections. InfLLM-v2 reuses the dense KV: 96 blocks of 64 (≈6K tokens), no new parameters.
- **Bridge:** the compression branch starts [08](08-fewer-records.md).
:::

## DSA: a lightning indexer picks 2,048 tokens {#dsa}

::: meta
**DSA** 2025 · DeepSeek-AI · DeepSeek-V3.2 report · [arXiv:2512.02556](https://arxiv.org/abs/2512.02556) · Used in: DeepSeek-V3.2(-Exp), GLM-5
:::

$$
I_{t,s}=\sum_{j=1}^{64} w_{t,j}\,\operatorname{ReLU}\big(q^{I}_{t,j}\cdot k^{I}_{s}\big),
\qquad A_t=\operatorname{Top}_{2048}(I_{t,:})
$$

![Indexer with 64 FP8 heads of 128 dims and one shared key head scores all 131,072 keys; the top 2,048 (1.56%) go to MQA-mode MLA, where 128 heads share each 576-dim latent (656 bytes in FP8). Multiply-accumulates per decode token and layer at 128K: dense MLA 18.3 G, indexer 1.07 G, sparse main 0.29 G; the indexer dominates past 34,816 tokens. Training: 1,000-step indexer-only KL warm-up (2.1B tokens), then 15,000 sparse steps (943.7B tokens).](../figures/cards/sparse-dsa.svg)

::: tips
- **ReLU per head, then weight.** ReLU was chosen for throughput; the indexer runs in FP8.
- **Cost:** main attention drops to $O(Nk)$, but the indexer stays $O(N^2)$ and outweighs it at 128K (derived 3.8×).
- **Token-level, shared:** one top-2,048 set per token for all 128 heads, so reads are gathers, not blocks.
- **Cache not smaller:** indexer keys are stored too. Latent bytes read per layer at 128K: 1.3 MB vs 86 MB dense.
:::

## 2026 indexers: same reads, cheaper scorer {#indexers .insight}

$$
\text{index dot products per query}\approx\frac{H_I\,N}{r\,s},
\qquad r=\text{tokens per scored key},\ s=\text{layers per indexer}
$$

![Index dot products per query and layer at N = 1M: DSA 64M (64 heads, every token), IndexShare 16M (one indexer per 4 layers), DeepSeek-V4-Flash CSA 16M (64 heads over 4-token entries), MSA 4M (one index head per GQA group), QSA 1M (4 heads over 4-token pooled keys). All read about 2,048 tokens per query.](../figures/cards/sparse-indexers.svg)

::: tips
- **MSA** (MiniMax-M3, [arXiv:2606.13392](https://arxiv.org/abs/2606.13392)): top-16 blocks of 128 per GQA group, scored by max-pooling. Reported 28.4× less attention compute at 1M.
- **QSA** (Qwen3.8-Flash-Next, [arXiv:2608.30320](https://arxiv.org/abs/2608.30320)): fills the 1-in-4 full-attention layers of a GDN hybrid ([12](12-hybrids.md)).
- **IndexShare** (GLM-5.2): IndexCache ([arXiv:2603.12201](https://arxiv.org/abs/2603.12201)) reuses top-k across layers: 75% fewer indexer computations.
- **Recipe:** MSA and QSA copy DSA's KL warm-up of the indexer alone, then joint sparse training.
:::
