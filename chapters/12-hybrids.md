# 12 · Hybrid stacks

::: lede
A fixed state compresses but cannot look things up. Production models keep a few full-attention layers and make the rest cheap.
:::

## Why hybrid: state is cheap, retrieval is not {#why .insight}

$$
M(N)=\sum_{\ell\in\mathcal R}|S_\ell|+N\sum_{\ell\in\mathcal G}c_\ell,\qquad
N^{*}=\frac{\sum_{\ell\in\mathcal R}|S_\ell|}{\sum_{\ell\in\mathcal G}c_\ell},\qquad c_\ell=2H_{kv}d_h\ \ (\text{MLA: }d_c+d_r)
$$

![Elements per sequence against context length for the Qwen3-Next-80B stack: its 36 Gated DeltaNet layers hold a fixed 18.9M-element state and its 12 attention layers add 12,288 elements per token, so KV overtakes the state at N = 1,536. The same 48 layers as all attention would reach 51.5G elements at 1M tokens versus 12.9G for the hybrid. Right: whole recurrent state as the KV of N* tokens, computed from configs: Jamba 448, Qwen3-Next 1,536, Kimi Linear 2,601, Qwen3.5-397B 3,072, MiniMax-01 3,584, Nemotron 3 Nano 3,925, Granite 4.0-H-Small 4,608, Kimi K3 7,851, Nemotron-H-56B 11,059.](../figures/cards/hybrid-why.svg)

::: tips
- **Why it works:** recurrent layers carry a summary; a few attention layers keep every token for exact lookup ([11 · Limits](11-test-time-learning.md#limits)).
- **Numbers:** Qwen3-Next's whole state equals the KV of 1,536 tokens. Past that, its 12 attention layers dominate.
- **Gotcha:** Qwen3.5 and Nemotron 3 configs keep state in fp32 while KV is BF16, which doubles the byte crossover.
- **Not solved:** global-layer KV still grows with $N$, and their prefill stays quadratic.
:::

## Interleaved layers vs parallel heads {#seq-par}

::: meta
**Jamba** 2024 · AI21 Labs · [arXiv:2403.19887](https://arxiv.org/abs/2403.19887) — **Hymba** 2024 · Dong, Fu et al., NVIDIA · [arXiv:2411.13676](https://arxiv.org/abs/2411.13676) — **Falcon-H1** 2025 · TII · [arXiv:2507.22448](https://arxiv.org/abs/2507.22448) · Used in: Jamba, Samba, Nemotron-H (sequential); Hymba-1.5B, Falcon-H1 (parallel)
:::

$$
\text{sequential: } x_{\ell+1}=x_\ell+F_\ell(x_\ell),\ F_\ell\in\{\text{SSM},\text{attn}\}\qquad
\text{parallel (Hymba): } Y=W_{\text{out}}\big(\beta_1\odot\operatorname{norm}(M_{\text{attn}}\tilde X)+\beta_2\odot\operatorname{norm}(M_{\text{ssm}}\tilde X)\big)
$$

![Sequential hybrids alternate whole layers and only the attention layers keep a KV cache (Jamba 4 of 32, Nemotron-H 10 of 118). Parallel hybrids run attention heads and SSM heads side by side in every layer; Falcon-H1 splits channels SSM 2 : attention 1 : MLP 5. Computed KV per token in BF16: Falcon-H1-34B 144 KiB over 72 layers with 4 KV heads, Nemotron-H-56B 40 KiB over 10 layers with 8 KV heads. Reported for Hymba-1.5B at 8K: a 78.6 MB cache versus 918 MB for Llama-3.2-3B.](../figures/cards/hybrid-seq-par.svg)

::: tips
- **Why it works:** in Hymba, attention heads give high-resolution recall and SSM heads summarize. Norms rescale before fusion; SSM outputs run larger.
- **Gotcha:** parallel ≠ fewer KV layers. Falcon-H1-34B keeps KV in all 72 layers (144 KiB/token, computed).
- **Numbers:** Hymba-1.5B at 8K: 11.67× smaller cache and 3.49× faster than Llama-3.2-3B (reported).
- **Compare:** Samba (ICLR 2025) interleaves Mamba with 2048-token SWA and has no global layer.
:::

## NVIDIA: Mamba-2 stacks and PostNAS retrofits {#nvidia}

::: meta
**Nemotron-H** 2025 · NVIDIA · [arXiv:2504.03624](https://arxiv.org/abs/2504.03624) — **Jet-Nemotron** 2025 · Gu, Hu et al., NVIDIA · NeurIPS 2025 · [arXiv:2508.15884](https://arxiv.org/abs/2508.15884) · Used in: Nemotron-H 8B/56B, Nemotron Nano 2, Nemotron 3 Nano, Jet-Nemotron 2B/4B
:::

$$
\text{KV/token}=L_{\text{attn}}\cdot 2H_{kv}d_h\,s,\qquad |S|=L_{\text{ssm}}\cdot H\cdot P\cdot N_{\text{state}}
$$

![Token-mixer layers from config.json: Nemotron-H-56B has 10 attention among 64 mixers, Nemotron Nano 2 9B 4 among 31, Nemotron 3 Nano 6 among 29 with 2 KV heads. Reported throughput: 2.4 times Llama-3.1-70B, up to 6 times Qwen3-8B, 3.3 times Qwen3-30B-A3B. Jet-Nemotron-2B keeps full attention in layers 15 and 20 and SWA in 21 and 22 of Qwen2.5-1.5B and replaces the rest with JetBlocks: KV at 64K falls from 1,792 MB to 154 MB and throughput rises from 241 to 2,885 tokens per second.](../figures/cards/hybrid-nvidia.svg)

::: tips
- **Placement rule:** ≈8% attention, evenly spread; first layer Mamba-2; no position embeddings (Nemotron-H §2).
- **Numbers:** Nemotron-H-56B: 2.4× output tokens/s of Llama-3.1-70B at 64K in, 1K out (reported, H100).
- **Converts from:** PostNAS freezes the MLPs, trains a once-for-all supernet and beam-searches where full attention stays.
- **Gotcha:** important layers are task-specific: MMLU and retrieval each lean on different 2–3 layers (Jet-Nemotron).
:::

## Qwen3-Next, Qwen3.5, Kimi: delta rule 3:1 {#delta-3-1}

::: meta
**Qwen3-Next / Qwen3.5** 2025–26 · Qwen team, Alibaba · [model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) — **Kimi Linear** 2025 · Kimi Team, Moonshot AI · [arXiv:2510.26692](https://arxiv.org/abs/2510.26692) · Used in: Qwen3-Next-80B-A3B, Qwen3.5-397B-A17B, Qwen3.8-Flash-Next, Kimi Linear 48B-A3B, Kimi K3
:::

$$
\text{stack}=n\times\big[\,3\times(\text{GDN}\to\text{MoE})+1\times(\text{gated attention}\to\text{MoE})\,\big],\qquad n=12\ (\text{Qwen3-Next}),\ 15\ (\text{Qwen3.5})
$$

![Layer strips from config.json. Qwen3-Next-80B: 36 GDN and 12 GQA layers, 24 KiB per token. Qwen3.5-397B: 45 GDN and 15 GQA, 30 KiB. Kimi Linear 48B: 20 KDA and 7 NoPE MLA at layers 4, 8, …, 24 and 27, so the last group is 2:1, 7.9 KiB. Kimi K3: 69 KDA and 24 gated MLA over 93 layers, 27 KiB. Kimi Linear ratio ablation, validation perplexity: 0:1 5.77, 1:1 5.66, 3:1 5.65, 7:1 5.70, 15:1 5.82. Qwen3.8 ablation on a 25B-A3B model: full attention 49.87, 3:1 SWA hybrid 51.15, 3:1 GDN hybrid 53.81.](../figures/cards/hybrid-delta31.svg)

::: tips
- **Gotcha:** 3:1 is nominal. Kimi Linear has 27 layers (last group 2:1); Kimi K3 ends with two MLA layers.
- **Compare:** Kimi's MLA uses NoPE, letting KDA carry position. Qwen keeps partial RoPE: its NoPE variant generated endlessly more often ([01](01-positions.md)).
- **Numbers:** Kimi Linear at 1M: 1.84 vs 11.48 ms per output token for full MLA (reported).
- **Trend:** Qwen3.8 makes the global layer sparse (QSA, [06](06-sparse-reads.md)). Output gates: [02](02-stable-softmax.md#gated).
:::

## MiniMax: hybrid, then full, then sparse {#minimax}

::: meta
**MiniMax-01 / M1** 2025 · MiniMax · [arXiv:2501.08313](https://arxiv.org/abs/2501.08313), [arXiv:2506.13585](https://arxiv.org/abs/2506.13585) — **M2** blog 2025-10-30 · [Hugging Face](https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model) — **M3 / MSA** 2026 · [arXiv:2606.13392](https://arxiv.org/abs/2606.13392) · Used in: MiniMax-Text-01, M1, M2, M3
:::

$$
\text{KV/token (elements)}=L_{\text{full}}\cdot 2H_{kv}d_h:\quad 10\cdot2\cdot8\cdot128=20{,}480\ \to\ 62\cdot2\cdot8\cdot128=126{,}976\ \to\ 60\cdot2\cdot4\cdot128=61{,}440
$$

![MiniMax layouts from config.json. Text-01 and M1: 80 layers, 70 lightning-attention plus 10 softmax, 40 KiB per token. M2: 62 full GQA layers, 248 KiB. M3: 3 dense plus 57 sparse MSA layers, 120 KiB retained per token; each query reads 16 blocks of 128 tokens. The M2 blog lists four reasons for leaving the hybrid: reasoning deficits visible only at scale, expensive evidence and late precision problems, immature infrastructure, and a sliding-window retrofit that degraded with context length.](../figures/cards/hybrid-minimax.svg)

::: tips
- **Not solved:** MiniMax-01's own study found pure linear attention unable to retrieve, hence softmax every 8th layer.
- **Numbers:** M1 used ~25% of DeepSeek-R1's FLOPs at 100K generated tokens (reported).
- **Gotcha:** M2's reasons are evaluation and infrastructure, not a proof against hybrids; the blog expects them to be solvable.
- **Compare:** M3 keeps all KV and bounds reads; reported 9× prefill and 15× decode vs M2 at 1M ([06](06-sparse-reads.md)).
:::

## Frontier layouts 2024–2026 {#layouts .insight}

$$
\text{KV/token}=\sum_{\ell\in\mathcal G}c_\ell\quad(\text{global layers only; SWA bounded by }W,\ \text{recurrent by }|S_\ell|)
$$

![Layer strips for 13 production models, 2024 to 2026, read from config.json and scaled to equal width, with KV per token for the growing layers in BF16 computed from config on a log scale: Jamba 16 KiB, MiniMax-Text-01 40, Gemma 3 27B 80, Nemotron-H-56B 40, Falcon-H1-34B 144, gpt-oss-120b 36, Qwen3-Next 24, MiniMax-M2 248, Kimi Linear 7.9, Nemotron 3 Nano 6, Qwen3.5-397B 30, MiniMax-M3 120, Kimi K3 27; Llama-3.1-70B reference 320.](../figures/cards/hybrid-layouts.svg)

::: tips
- **Read it off config:** `layer_types`, `hybrid_override_pattern`, `full_attn_layers`, `attn_type_list`, `attn_layer_period`.
- **Numbers:** global share clusters at 1/2 (gpt-oss), 1/4 (Qwen, Kimi), 1/6 (Gemma 3), 1/8–1/10 (Jamba, MiniMax-01, Granite).
- **Gotcha:** "hybrid" can mean SWA, SSM, delta-rule or parallel heads beside global attention. KV savings differ by family.
- **Not solved:** MiniMax went the other way: M2 is all full attention, and M3 still stores every token.
:::

| Family | Examples | Global layers | Bounded part |
| --- | --- | --- | --- |
| Local / global | Gemma 3, gpt-oss | 1/6, 1/2 | SWA cache ($W$ = 1024, 128) |
| SSM + attention | Jamba, Nemotron-H, Nemotron 3 Nano, Granite 4.0-H | 4–10 layers | Mamba state |
| Delta rule + attention | Qwen3-Next, Qwen3.5, Kimi Linear, Kimi K3 | ~1/4 | GDN / KDA state |
| Lightning + softmax | MiniMax-Text-01, M1 | 1/8 | lightning state |
| Parallel heads | Hymba, Falcon-H1 | every layer | SSM state |
| Full / sparse | MiniMax-M2, M3 | all | M3: reads only |
