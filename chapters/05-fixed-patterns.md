# 05 · Fixed patterns

::: lede
Most layers do not need the whole history. Fix in advance which keys each query may see, and those layers stop growing with $N$.
:::

## Sparse Transformer → Longformer / BigBird: masks fixed by position {#lineage}

::: meta
**Sparse Transformer** 2019 · Child et al., OpenAI · [arXiv:1904.10509](https://arxiv.org/abs/1904.10509) — **Longformer** 2020 · Beltagy et al., AI2 · [arXiv:2004.05150](https://arxiv.org/abs/2004.05150) — **BigBird** 2020 · Zaheer et al., Google · NeurIPS 2020 · [arXiv:2007.14062](https://arxiv.org/abs/2007.14062) · Used in: GPT-3, gpt-oss (dense/banded alternation)
:::

$$
o_t=\sum_{j\in A_t}\operatorname{softmax}_{j\in A_t}\!\Big(\frac{q_t^{\top}k_j}{\sqrt{d_h}}\Big)v_j,
\qquad A_t \text{ depends on positions only, never on content}
$$

![Three fixed masks on a toy 24×24 grid: Sparse Transformer band plus stride (two-hop path 23 → 3 → 1), Longformer window plus global tokens, BigBird window plus global plus random blocks. At n = 4,096: strided l = 64 reads about 128 keys per query, Longformer and BigBird-ITC 512. GPT-3 and gpt-oss alternate dense and banded layers.](../figures/cards/fixed-lineage.svg)

::: tips
- **Why it works:** depth routes information. Band then stride connects any pair in two layers, at $O(n\sqrt n)$ cost.
- **Gotcha:** strided routing failed on text (no periodic structure). The paper's "fixed" variant uses blocks plus summary columns instead.
- **Used in:** GPT-3 alternated dense and locally banded layers. gpt-oss does the same "following GPT-3", band 128.
- **Not solved:** global and random links are chosen before any query exists. Content-chosen reads: [06](06-sparse-reads.md).
:::

## Sliding window + global layers in production {#swa-global}

::: meta
**SWA : global** 2023–26 · Mistral 7B [arXiv:2310.06825](https://arxiv.org/abs/2310.06825) · Gemma 2 [arXiv:2408.00118](https://arxiv.org/abs/2408.00118) · Gemma 3 [arXiv:2503.19786](https://arxiv.org/abs/2503.19786) · gpt-oss [arXiv:2508.10925](https://arxiv.org/abs/2508.10925) · MiMo-V2-Flash [arXiv:2601.02780](https://arxiv.org/abs/2601.02780) · Used in: Gemma 2/3, gpt-oss, OLMo 3, MiMo-V2-Flash, DeepSeek-V4-Flash
:::

$$
\text{KV}(N)=N\sum_{\ell\in\text{global}}c_\ell\;+\;\min(N,W)\sum_{\ell\in\text{local}}c_\ell,
\qquad c_\ell=2H_{kv}d_h\,s
$$

![Layer layouts of eight models with local:global ratio and window: Mistral 7B all local W 4,096; Gemma 2 1:1 W 4,096; Gemma 3 5:1 W 1,024; gpt-oss 1:1 W 128; Llama 4 Scout 3:1 chunk 8,192; OLMo 3 3:1 W 4,096; MiMo-V2-Flash 39:9 W 128; DeepSeek-V4-Flash two local layers. KV at 128K in BF16: Gemma 3 27B 10.4 vs 62 GiB all-global, Llama 4 Scout 7.1 vs 24, gpt-oss-120b 4.5 vs 9.0, MiMo-V2-Flash 2.8 vs 15.](../figures/cards/fixed-swa-layouts.svg)

::: tips
- **Ring buffer:** a local layer writes token $t$ into slot $t \bmod W$. Mistral 7B: 8× less cache at 32K.
- **Global layers set the bill.** They grow with $N$, and only they retrieve beyond $W$.
- **Small windows need a sink.** MiMo 32B ablation: $W$ = 128 beat 512 on RULER-32K (89.4 vs 84.7). See [learned sinks](02-stable-softmax.md#sinks).
- **Numbers:** Gemma 3 report: global-only KV overhead about 60%, under 15% with 1,024-token windows.
:::

## Chunked local attention (Llama 4) vs sliding window {#chunked}

::: meta
**Chunked attention + iRoPE** 2025 · Meta · [Llama 4 blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) · config `attention_chunk_size` 8192, `no_rope_layers` 1,1,1,0 · Used in: Llama 4 Scout
:::

$$
A^{\text{chunk}}_t=\{\,j\le t:\lfloor j/C\rfloor=\lfloor t/C\rfloor\,\},
\qquad |A^{\text{chunk}}_t|=(t \bmod C)+1
$$

![Toy 24×24 masks with size 6: sliding window gives every query 6 keys; chunked gives 1 to 6 keys (mean 3.5) and query 12 at a chunk start sees only itself. Llama 4 Scout: 36 chunked RoPE layers (C 8,192) and 12 global NoPE layers; at 128K a chunked layer holds at most 32 MiB versus 512 MiB for a global layer; whole model 7.1 GiB versus 24 GiB.](../figures/cards/fixed-chunked.svg)

::: tips
- **Compare:** same cache cap as SWA with $W=C$, half the pairs, but each chunk start is nearly blind locally.
- **Why chunks:** the mask is block-diagonal, so each chunk is an ordinary causal attention and the cache empties at every boundary.
- **iRoPE:** chunked layers keep RoPE; every 4th layer is global with no RoPE ([positions](01-positions.md)).
:::

## StreamingLLM: window + 4 sink tokens {#streaming}

::: meta
**StreamingLLM** 2023 · Xiao et al., MIT · Meta AI · ICLR 2024 · [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) · Used in: training-free streaming; successor = learned sink logit (gpt-oss, MiMo-V2-Flash, DeepSeek-V4)
:::

$$
A_t=\{0,1,2,3\}\,\cup\,\{t-W+1,\dots,t\},
\qquad \text{pos}(j)=\text{rank of } j \text{ inside the cache}
$$

![Toy masks with a 6-token window. Window only evicts token 0: Llama-2-13B PG19 perplexity 5158.07. Four sinks plus 1,020 recent tokens: 5.40. Text positions 0,1,2,3,6,7,8,9 are re-numbered 0 to 7 inside the cache. The cache stays 4 + W records: 1 GiB for Llama-2-7B with 2,048 slots, streamed to 4M tokens.](../figures/cards/fixed-streaming.svg)

::: tips
- **Why it works:** softmax must sum to 1, so heads park spare mass on the first tokens. Evict them and that mass lands elsewhere.
- **Numbers:** 4 sinks suffice; 1–2 do not (§4.4). Up to 22.2× faster than a window with recomputation.
- **Not solved:** stability, not memory. Recall drops to zero once the answer is farther back than the cache (App. C).
- **Learned version:** one sink logit per head replaces the 4 tokens: [sinks](02-stable-softmax.md#sinks). Smarter eviction: [07](07-kv-eviction.md).
:::
