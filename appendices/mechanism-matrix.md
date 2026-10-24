# Mechanism matrix

Every card in the book, grouped by the lever it pulls. Generated from the chapter files by `scripts/build_matrix.py`. Insight cards (costs, comparisons) are in italics.


## Map

| Ch. | Card | Year | Used in |
| --- | --- | --- | --- |
| 00 | [*The bill: prefill pays $N^2$, decode pays $N$ bytes per token*](../chapters/00-map.md#bill) |  |  |
| 00 | [*Seven levers*](../chapters/00-map.md#levers) |  |  |
| 00 | [*Twelve years in five lanes*](../chapters/00-map.md#timeline) |  |  |
| 00 | [*How to read the cards*](../chapters/00-map.md#legend) |  |  |

## Positions & stability

| Ch. | Card | Year | Used in |
| --- | --- | --- | --- |
| 01 | [RoPE: a spectrum of clocks, set by the base](../chapters/01-positions.md#rope-base) | 2021–2023 | Llama 3 (b = 500K), gpt-oss (150K), Qwen3 (1M), Qwen3-Next (10M), Gemma 3 (10K local / 1M global) |
| 01 | [Context extension: PI → NTK-aware → YaRN / llama3](../chapters/01-positions.md#extension) | 2023–2024 | gpt-oss, DeepSeek-V3, DeepSeek-V4, OLMo 3, Qwen3 (optional); llama3 type: Llama 3.1, Llama 4 |
| 01 | [Partial RoPE and p-RoPE: rotate only some dims](../chapters/01-positions.md#partial) | 2022–2025 | GLM-4.5, Qwen3-Next, MiniMax-M2, MiMo-V2-Flash, Phi-4-mini, Gemma 4, DeepSeek-V4 |
| 01 | [NoPE layers: iRoPE and RNoPE](../chapters/01-positions.md#nope) | 2023–2025 | Llama 4, Command A, SmolLM3 |
| 01 | [Length-dependent logit temperature](../chapters/01-positions.md#temperature) | 2025 | Llama 4 (NoPE layers); YaRN models use a constant factor |
| 02 | [QK-norm, soft-cap, QK-Clip: bound the logits](../chapters/02-stable-softmax.md#qk-norm) | 2023–2025 | Qwen3, Qwen3-Next, OLMo 2/3, Gemma 3/4, GLM-4.5, MiniMax-M2, Llama 4 Scout, DeepSeek-V4 |
| 02 | [Attention sinks → a learned sink logit](../chapters/02-stable-softmax.md#sinks) | 2023–2025 | gpt-oss, MiMo-V2-Flash (SWA layers), DeepSeek-V4 |
| 02 | [Gated attention: a sigmoid gate after SDPA](../chapters/02-stable-softmax.md#gated) | 2025 | Qwen3-Next, Qwen3.5 (`attn_output_gate`) |
| 02 | [Research knobs: forget gate and differential attention](../chapters/02-stable-softmax.md#knobs) | 2024–2025 | Phi-4-mini-flash-reasoning (differential attention) |
| 02 | [*2026 anatomy: which trick sits where*](../chapters/02-stable-softmax.md#anatomy) |  |  |

## Thinner records

| Ch. | Card | Year | Used in |
| --- | --- | --- | --- |
| 03 | [MHA → GQA → MQA: one knob, $H_{kv}$](../chapters/03-kv-heads.md#gqa) | 2019–2023 | PaLM, Falcon-7B, Gemma-2B (MQA); Llama 2-70B, Llama 3, Mistral 7B, Qwen2/3 (GQA) |
| 03 | [*What it saves in real models*](../chapters/03-kv-heads.md#numbers) |  |  |
| 03 | [*Why decode benefits: intensity = group size*](../chapters/03-kv-heads.md#intensity) |  |  |
| 03 | [K=V: one vector is both key and value](../chapters/03-kv-heads.md#kv-tie) | 2026 | Gemma 4 31B / 26B-A4B (global layers), DeepSeek-V4-Pro / Flash |
| 03 | [Beyond GQA: MFA, GTA, GLA](../chapters/03-kv-heads.md#beyond-gqa) | 2024–2025 | Step-3 (MFA) |
| 03 | [Cross-layer sharing: fewer layers own a cache](../chapters/03-kv-heads.md#cross-layer) | 2024 | Hunyuan-Large (CLA2), Gemma 3n, Gemma 4 E2B / E4B |
| 04 | [MLA: cache a latent, rebuild the heads](../chapters/04-latent-kv.md#mla) | 2024 | DeepSeek-V2, V3, V3.2, Kimi K2, GLM-5, Mistral Large 3 |
| 04 | [Weight absorption: decode is MQA over 576 dims](../chapters/04-latent-kv.md#absorb) | 2024 | DeepSeek-V3.x decode, FlashMLA |
| 04 | [Decoupled RoPE: a separate 64-dim lane](../chapters/04-latent-kv.md#rope) | 2024 | DeepSeek-V2, V3, V3.2, Kimi K2, GLM-5, Mistral Large 3 |
| 04 | [*Head count: heads cost FLOPs, not cache*](../chapters/04-latent-kv.md#heads) |  |  |
| 04 | [Converting MHA/GQA → MLA](../chapters/04-latent-kv.md#convert) | 2025 | research checkpoints (SmolLM, Llama-2-7B) |

## Read fewer records

| Ch. | Card | Year | Used in |
| --- | --- | --- | --- |
| 05 | [Sparse Transformer → Longformer / BigBird: masks fixed by position](../chapters/05-fixed-patterns.md#lineage) | 2019–2020 | GPT-3, gpt-oss (dense/banded alternation) |
| 05 | [Sliding window + global layers in production](../chapters/05-fixed-patterns.md#swa-global) | 2023 | Gemma 2/3, gpt-oss, OLMo 3, MiMo-V2-Flash, DeepSeek-V4-Flash |
| 05 | [Chunked local attention (Llama 4) vs sliding window](../chapters/05-fixed-patterns.md#chunked) | 2025 | Llama 4 Scout |
| 05 | [StreamingLLM: window + 4 sink tokens](../chapters/05-fixed-patterns.md#streaming) | 2023–2024 | training-free streaming; successor = learned sink logit (gpt-oss, MiMo-V2-Flash, DeepSeek-V4) |
| 06 | [*What shrinks: reads or records?*](../chapters/06-sparse-reads.md#what-shrinks) |  |  |
| 06 | [Training-free prefill: MInference → XAttention](../chapters/06-sparse-reads.md#prefill) | 2024–2025 | Qwen2.5-1M |
| 06 | [Quest: page bounds choose what decode reads](../chapters/06-sparse-reads.md#quest) | 2024 | training-free decode on any MHA/GQA model |
| 06 | [MoBA: route each query to a few blocks](../chapters/06-sparse-reads.md#moba) | 2025 | Kimi long-context service |
| 06 | [NSA: compress, select, slide](../chapters/06-sparse-reads.md#nsa) | 2025 | research 27B MoE; selection reused by InfLLM-v2 (MiniCPM4.1) |
| 06 | [DSA: a lightning indexer picks 2,048 tokens](../chapters/06-sparse-reads.md#dsa) | 2025 | DeepSeek-V3.2(-Exp), GLM-5 |
| 06 | [*2026 indexers: same reads, cheaper scorer*](../chapters/06-sparse-reads.md#indexers) |  |  |

## Keep fewer records

| Ch. | Card | Year | Used in |
| --- | --- | --- | --- |
| 07 | [H2O: heavy hitters plus a recent window](../chapters/07-kv-eviction.md#h2o) | 2023 | NVIDIA kvpress (ObservedAttentionPress); the standard eviction baseline |
| 07 | [SnapKV: the prompt's tail votes once](../chapters/07-kv-eviction.md#snapkv) | 2024 | NVIDIA kvpress (SnapKVPress); the scorer behind PyramidKV and Ada-KV |
| 07 | [PyramidKV and Ada-KV: where the budget goes](../chapters/07-kv-eviction.md#budgets) | 2024–2025 | NVIDIA kvpress (PyramidKVPress, AdaKVPress) |
| 07 | [KVzip: keep what the model needs to repeat the context](../chapters/07-kv-eviction.md#kvzip) | 2025 | NVIDIA kvpress (KVzipPress) |
| 07 | [DuoAttention and RazorAttention: evict per head](../chapters/07-kv-eviction.md#duo) | 2024–2025 | NVIDIA kvpress (DuoAttentionPress) |
| 07 | [DMC → DMS: learn what to merge or evict](../chapters/07-kv-eviction.md#dms) | 2024–2025 | NVIDIA checkpoints Llama-2-7B/13B-DMC-4x/8x, Qwen3-8B-DMS-8x |
| 07 | [*Pitfalls: eviction cannot be undone*](../chapters/07-kv-eviction.md#pitfalls) |  |  |
| 08 | [Compressive Transformer: pool old memories, don't drop them](../chapters/08-fewer-records.md#compressive) | 2019–2020 | research only; the ancestor of the HCA idea, and the paper that introduced PG-19 |
| 08 | [Gist tokens → Activation Beacon: learned summary tokens](../chapters/08-fewer-records.md#beacon) | 2023–2025 | research (FlagEmbedding release) |
| 08 | [NSA's compression branch: the bridge](../chapters/08-fewer-records.md#nsa-cmp) | 2025 | research (27B MoE in the paper); full card in 06 |
| 08 | [CSA: compress by 4, then select top-k](../chapters/08-fewer-records.md#csa) | 2026 | DeepSeek-V4-Flash, DeepSeek-V4-Pro |
| 08 | [HCA: compress by 128, read everything](../chapters/08-fewer-records.md#hca) | 2026 | DeepSeek-V4-Flash, DeepSeek-V4-Pro |
| 08 | [*DeepSeek-V4: layer layout and bytes per token*](../chapters/08-fewer-records.md#v4-bytes) |  |  |

## Fixed state

| Ch. | Card | Year | Used in |
| --- | --- | --- | --- |
| 09 | [Linear attention: an RNN with a matrix state](../chapters/09-linear-state.md#linear-attention) | 2020 | the base of every layer in chapters 09–12 |
| 09 | [RetNet → Lightning: one fixed decay per head](../chapters/09-linear-state.md#retnet) | 2023–2025 | MiniMax-Text-01, MiniMax-M1 |
| 09 | [GLA: data-dependent decay per key channel](../chapters/09-linear-state.md#gla) | 2023–2024 | GLA research models; the same per-channel decay drives KDA in Kimi Linear |
| 09 | [Mamba → Mamba-2: the state-space duality](../chapters/09-linear-state.md#ssd) | 2024 | NVIDIA Nemotron-H, Nemotron 3 Nano, IBM Granite 4.0-H |
| 09 | [Mamba-3: trapezoid write, rotations, MIMO](../chapters/09-linear-state.md#mamba3) | 2026 | no released LLM found (Oct 2026) |
| 09 | [*Chunkwise form: attention inside, state between*](../chapters/09-linear-state.md#chunkwise) |  |  |
| 10 | [DeltaNet: read, compare, overwrite](../chapters/10-delta-rule.md#deltanet) | 2021–2024 | the base of every card below |
| 10 | [Gated DeltaNet: decay first, then correct](../chapters/10-delta-rule.md#gdn) | 2024–2025 | Qwen3-Next, Qwen3.5, Olmo Hybrid, Jet-Nemotron |
| 10 | [KDA: per-channel decay inside the delta rule](../chapters/10-delta-rule.md#kda) | 2025 | Kimi Linear 48B-A3B |
| 10 | [RWKV-7 and Comba: the sum form](../chapters/10-delta-rule.md#sum-form) | 2025 | RWKV-7 World 0.1B–2.9B |
| 10 | [Negative eigenvalues and Householder products](../chapters/10-delta-rule.md#state-tracking) | 2024–2025 | Olmo Hybrid 7B (β up to 2) |
| 10 | [*Compare: one axis, the transition $A_t$*](../chapters/10-delta-rule.md#taxonomy) |  |  |
| 10 | [2026: decouple strength, gain and address](../chapters/10-delta-rule.md#frontier) | 2026 | research models only (Oct 2026) |
| 11 | [*Every update is one optimizer step*](../chapters/11-test-time-learning.md#online-step) |  |  |
| 11 | [TTT layers: the state is a model, $b$ is the knob](../chapters/11-test-time-learning.md#ttt) | 2024–2025 | research models, 125M–1.3B |
| 11 | [Titans: surprise, momentum, forgetting](../chapters/11-test-time-learning.md#titans) | 2024–2025 | research models, 170M–760M |
| 11 | [ATLAS and MesaNet: fit the window, or solve it](../chapters/11-test-time-learning.md#atlas-mesa) | 2025–2026 | research models, ≤1.3B |
| 11 | [LaCT and TTT-E2E: huge chunks make it fast](../chapters/11-test-time-learning.md#lact) | 2025–2026 | research; LaCT also in a 14B autoregressive video model fine-tuned from Wan 2.1 |
| 11 | [Log-linear attention: $O(\log T)$ states](../chapters/11-test-time-learning.md#log-linear) | 2025–2026 | research models (log-linear Mamba-2 and Gated DeltaNet, ~800M) |
| 11 | [*Limits: learning is not recall*](../chapters/11-test-time-learning.md#limits) |  |  |

## Mix both

| Ch. | Card | Year | Used in |
| --- | --- | --- | --- |
| 12 | [*Why hybrid: state is cheap, retrieval is not*](../chapters/12-hybrids.md#why) |  |  |
| 12 | [Interleaved layers vs parallel heads](../chapters/12-hybrids.md#seq-par) | 2024–2025 | Jamba, Samba, Nemotron-H (sequential); Hymba-1.5B, Falcon-H1 (parallel) |
| 12 | [NVIDIA: Mamba-2 stacks and PostNAS retrofits](../chapters/12-hybrids.md#nvidia) | 2025 | Nemotron-H 8B/56B, Nemotron Nano 2, Nemotron 3 Nano, Jet-Nemotron 2B/4B |
| 12 | [Qwen3-Next, Qwen3.5, Kimi: delta rule 3:1](../chapters/12-hybrids.md#delta-3-1) | 2025 | Qwen3-Next-80B-A3B, Qwen3.5-397B-A17B, Qwen3.8-Flash-Next, Kimi Linear 48B-A3B, Kimi K3 |
| 12 | [MiniMax: hybrid, then full, then sparse](../chapters/12-hybrids.md#minimax) | 2025–2026 | MiniMax-Text-01, M1, M2, M3 |
| 12 | [*Frontier layouts 2024–2026*](../chapters/12-hybrids.md#layouts) |  |  |

## Move fewer bytes

| Ch. | Card | Year | Used in |
| --- | --- | --- | --- |
| 13 | [FlashAttention: tiles in SRAM, no $N\times N$ in HBM](../chapters/13-kernels.md#flash) | 2022 | PyTorch SDPA, vLLM, Megatron-LM |
| 13 | [FA-2 → FA-3 → FA-4: re-tiled for each GPU](../chapters/13-kernels.md#fa-versions) | 2023–2026 |  |
| 13 | [Split-KV decoding (Flash-Decoding)](../chapters/13-kernels.md#flash-decoding) | 2023 | FlashInfer (split-KV scheduler), FlashMLA (`num_splits`) |
| 13 | [FlashMLA and FlashInfer: decode kernels for real caches](../chapters/13-kernels.md#flashmla) | 2025 | DeepSeek-V3/V3.2/V4.1 serving — FlashInfer 2025 · Ye et al. · MLSys 2025 · arXiv:2501.01005 · Used in: SGLang, vLLM, MLC-Engine |
| 13 | [Chunkwise kernels for linear and delta layers (FLA)](../chapters/13-kernels.md#fla) | 2023–2024 | flash-linear-attention |
| 13 | [Low-bit attention: SageAttention 1 → 3](../chapters/13-kernels.md#sage) | 2024–2025 | video diffusion inference (HunyuanVideo, CogVideoX, reported) |
| 14 | [FP8 / FP4 KV cache: the production default](../chapters/14-serving.md#kv-quant) | 2025 | DeepSeek-V3.2, DeepSeek-V4.1, vLLM |
| 14 | [KIVI and KVQuant: quantize along the outliers](../chapters/14-serving.md#kivi) | 2024 |  |
| 14 | [PagedAttention: KV in fixed blocks](../chapters/14-serving.md#paged) | 2023 | vLLM, SGLang (page = 1 token), FlashMLA (block 64) |
| 14 | [Prefix caching: RadixAttention](../chapters/14-serving.md#radix) | 2023–2024 | SGLang |
| 14 | [Prefill–decode disaggregation](../chapters/14-serving.md#disagg) | 2024–2025 | Kimi, DeepSeek-V3/R1 serving |
| 14 | [Context parallelism: Ring vs Ulysses](../chapters/14-serving.md#cp) | 2023–2024 | DeepSpeed; Llama 3 (all-gather variant) |
| 14 | [Tiered storage: HBM → DRAM → SSD](../chapters/14-serving.md#tiers) | 2025 | Kimi, DeepSeek API, vLLM and SGLang (LMCache) |

## Compare

| Ch. | Card | Year | Used in |
| --- | --- | --- | --- |
| 15 | [*The bill across 17 real stacks*](../chapters/15-choose.md#bill) |  |  |
| 15 | [*What each lever guarantees*](../chapters/15-choose.md#properties) |  |  |
| 15 | [*Three 2026 recipes*](../chapters/15-choose.md#recipes) |  |  |
