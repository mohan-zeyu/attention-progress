# C-sparse — sparse attention, KV selection/eviction, sequence compression, KV quantization

Research notes for card writers. Checked 2026-10-21. "derived" = my arithmetic from verified configs. UNVERIFIED = not confirmed against a primary source.
Reference KV size used in derived numbers: Llama-3-8B = 32 layers × 8 KV heads × 128 dims × 2 (K,V) × 2 B = **128 KiB/token** (16 GiB at 128K tokens). This is the standard public config; I did not re-fetch it.

---

## 1. Coverage audit

**What the book has now**
- 06: Longformer and BigBird (static); MoBA (mean-key block routing); MInference (training-free prefill patterns); NSA (three gated branches); DSA (lightning indexer, two-stage training, O(Nk) main read vs O(N²) indexer). It states the storage-vs-access distinction well.
- 07: MLA → CSA/HCA (V4). Covers overlapping 2r-source windows, HCA reading all entries densely, ratios 4/128, window 128, and Pro's layer layout.
- 08: one paragraph on Qwen3.8-Next QSA.
- 09: KIVI granularity, CLA cross-layer reuse, DeepSeek-V4.1-Flash 890 B/token (CSA2 + FP4), PagedAttention.
- Missing entirely: the whole **eviction** family (cache shrinks); **attention sinks**; **production sliding-window/global hybrids**; decode-time page selection (Quest); head specialization (DuoAttention); offloaded retrieval; 2026 indexers (MSA, QSA details, IndexCache); learned eviction (DMC/DMS); classic compression lineage (Compressive Transformer, gist/beacon); FP8 KV as the production default.

**Errors and oversimplifications (with corrections)**
1. **DSA formula order (ch06).** The book says "Weighted indexer-head matches pass through ReLU". In fact ReLU is applied **per head first**, then the outputs are weighted and summed: `I_{t,s} = Σ_j w_{t,j} · ReLU(q^I_{t,j} · k^I_s)`. The `w_{t,j}` are query-derived per-head scalars. ReLU was chosen "for throughput". Source: https://arxiv.org/html/2512.02556 §2.1 Eq. 1.
2. **DSA has no numbers (ch06).** Add k = 2048 selected tokens per query, 64 indexer heads × 128 dims, FP8 indexer, MQA-mode MLA. Warm-up: 1,000 steps × 16 × 128K ≈ 2.1B tokens, LR 1e-3, indexer only, KL loss. Sparse stage: 15,000 steps × 480 × 128K ≈ 943.7B tokens, LR 7.3e-6. The indexer input is detached. DSA first shipped in **V3.2-Exp (Sept 2025)**, before the Dec 2025 V3.2 report. Sources: §2.1.1; config https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json (`index_n_heads 64`, `index_head_dim 128`, `index_topk 2048`).
3. **V4 layer layout (ch07)** "Core layers start with two HCA layers" holds for **V4-Pro only**. V4-Flash starts with **two pure sliding-window layers**, then interleaves CSA and HCA. Source: https://arxiv.org/html/2606.19348 §4.2.1.
4. **The CSA read budget is missing (ch07).** Attention top-k is **512 compressed entries (Flash)** and **1,024 (Pro)**. The paper says this is deliberately smaller than V3.2's 2048. The indexer has 64 heads × 128 dims, and its QK runs in **FP4** (§2.3.4). Without k, readers cannot see that CSA reads only about 2K–4K token-spans.
5. **V4 has learnable attention sinks (ch07 omits them).** Each head has a sink logit `z'_h`; `exp(z'_h)` is added to the softmax denominator (§2.3.3, Eq. 27). This is the same device as gpt-oss and MiMo-V2-Flash. It also links to ch08's sink-free gated attention.
6. **V4 storage precision (ch07/09).** KV is **FP8 except the 64 RoPE dims, which stay BF16**. The 1M-token KV is about 2% of a BF16 GQA8 (head dim 128) baseline. At 1M versus V3.2, Pro needs 27% of the FLOPs and 10% of the KV; Flash needs 10% and 7%. Sources: abstract, §1, §2.3.4.
7. **MoBA (ch06) lacks production and hybrid facts.** The abstract says it is "deployed to support Kimi's long-context requests". The last **3 layers stay full attention** in continued pretraining, because SFT with sparse layers on top suffered sparse gradients. Numbers: block 512/top-3 at 8K–32K; **block 4096/top-12 at 1M = 95.31% sparsity**. Source: https://arxiv.org/html/2502.13189 §3.2–3.3.
8. **MInference (ch06) is not only a research approximation.** Qwen2.5-1M's deployment implements MInference-based Vertical-Slash sparse prefill and reports **3–7× prefill speedup at 1M**. Source: https://arxiv.org/html/2501.15383 §5.2.
9. **NSA (ch06) has no numbers.** Compression block l = 32, stride d = 16; selection block l' = 64; n = 16 selected blocks, of which **1 initial + 2 local are always forced**; window w = 512. Selection is shared per GQA group (4 groups × 16 heads). Source: https://arxiv.org/html/2502.11089 §4.1, §3.3.2. NSA's ACL 2025 Best Paper status is widely reported but was not on the arXiv page (UNVERIFIED here).
10. **Biggest omission in "sparse attention":** the most widely deployed sparse pattern is **sliding window + periodic global layers**: Mistral, Gemma 2/3, gpt-oss, Llama 4 (chunked), MiMo-V2-Flash, V4-Flash's first layers. Ch06 presents static patterns only as 2020 document models.
11. **Ch06 comparison table: three budgets, not two.** Ch06 cleanly separates "selection keeps the cache". It never names the third option: **permanent eviction**, where the cache shrinks and the decision cannot be undone. Make the three budgets explicit: records stored (eviction/compression), records read (selection), bytes per record (quantization).
12. **Ch09 quantization treats KIVI as the exemplar.** In production, the default is **FP8 KV** with block scales (vLLM/SGLang/TRT-LLM). DeepSeek-V3.2's FP8 MLA cache is **656 B/token/layer**: 512 × e4m3 + 4 × fp32 scales + 64 × BF16 RoPE (https://vllm.ai/blog/2025-09-29-deepseek-v3-2). **NVFP4 KV** is about 50% of FP8 with <1% accuracy loss, but is **dequantized to FP8 before the matmuls**, so memory is saved but compute is not (https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/). FlashMLA now lists V4.1 KV at **528 B (FP8) / 288 B (FP4) per token** for head dim 512 (https://github.com/deepseek-ai/FlashMLA).
13. **QSA (ch08) is correct but thin.** Add micro-block r = 4, token budget K = 2048 (≤512 blocks), and a 4-query-head / 1-key-head MQA indexer. See its card.
14. **Mechanism matrix** lacks rows for: SWA/global hybrids, attention sinks, eviction (H2O/SnapKV), head split (DuoAttention), page selection (Quest), MSA/QSA/IndexCache, DMS, FP8/FP4 KV.

**2026 frontier scan** (what production models actually ship)
- **DeepSeek-V4 (Pro/Flash):** CSA (m = 4, top-k 512/1024, FP4 indexer) + HCA (m' = 128, dense) + SWA 128 + sinks. **V4.1-Flash** adds CSA2 cross-layer reuse and an FP4 cache (890 B/token global).
- **GLM-5 / 5.1 (Feb 2026):** "adopts DSA" (https://arxiv.org/abs/2602.15763). **GLM-5.2:** IndexShare reuses one indexer for every 4 sparse layers, "2.9× fewer per-token FLOPs at 1M" (model card https://huggingface.co/zai-org/GLM-5.2 → paper IndexCache 2603.12201).
- **MiniMax:** M1 used lightning (linear) attention. **M2 reverted to full attention** (blog: https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model). **M3 uses MiniMax Sparse Attention (MSA)**, block-sparse top-k per GQA group (https://arxiv.org/abs/2606.13392).
- **Qwen3.8-Flash-Next:** GDN 3:1 hybrid; its global layers are converted to **QSA** in continued pretraining (2608.30320).
- **Xiaomi MiMo-V2-Flash:** SWA 128 : global, 5:1, learned sink bias (2601.02780).
- **MiniCPM4/4.1:** InfLLM-v2 trainable block sparse (ICLR 2026).
- **Kimi K2.5:** full MLA. **K3:** KDA + gated MLA hybrid, no sparse attention reported. Both from secondary sources only (UNVERIFIED); this belongs to ch08, not here.
- **Gemini:** no published attention design.

---

## 2. Gap table

| Pri | Mechanism | Year | Venue / org | URL | What changes vs predecessor (≤15 words) | Adopted in | Suggested chapter |
|---|---|---|---|---|---|---|---|
| MUST | SWA + interleaved global layers | 2023–26 | Mistral, Google, OpenAI, Meta, Xiaomi | https://arxiv.org/abs/2508.10925 ; https://arxiv.org/abs/2503.19786 ; https://arxiv.org/abs/2601.02780 | Most layers see only W recent tokens; every R-th layer sees everything | Mistral 7B (W 4096), Gemma 2/3 (5:1, 1024), gpt-oss (1:1, 128), Llama 4 (chunk 8192, 3:1), MiMo-V2-Flash (5:1, 128), V4-Flash L1–2; all engines | 06 (new section: static patterns in production) |
| MUST | Attention sinks: StreamingLLM → learned sink logit | 2023 → 2025–26 | ICLR 2024 (MIT/Meta) → OpenAI, Xiaomi, DeepSeek | https://arxiv.org/abs/2309.17453 | Keep 4 first tokens + window; later a per-head learned denominator logit | TRT-LLM `sink_token_length`; HF; gpt-oss, MiMo-V2-Flash, DeepSeek-V4 | NEW: KV eviction (card 1); link 08 |
| MUST | H2O | 2023 | NeurIPS 2023 | https://arxiv.org/abs/2306.14048 | Evict by accumulated attention; budget split heavy-hitters/recent | Research baseline; kvpress (ObservedAttention); not in vLLM/SGLang | NEW: KV eviction |
| MUST | SnapKV | 2024 | NeurIPS 2024 | https://arxiv.org/abs/2404.14469 | One-shot prompt pruning by last-window votes, pooled; decode untouched | kvpress; de facto eviction baseline; not in vLLM (RFC #5751 stalled) | NEW: KV eviction |
| MUST | Quest | 2024 | ICML 2024 (MIT Han Lab) | https://arxiv.org/abs/2406.10774 | Cache kept; per-page min/max key bound picks pages per decode query | LServe; FlashInfer-based research kernels | 06 (training-free decode selection) |
| MUST | DuoAttention (+ RazorAttention) | 2024 | ICLR 2025 (MIT) / ICLR 2025 | https://arxiv.org/abs/2410.10819 ; https://arxiv.org/abs/2407.15891 | Head granularity: retrieval heads full KV, others sink+window only | LServe; kvpress DuoAttentionPress | NEW: KV eviction |
| MUST | DSA: numbers and adoption | 2025 | DeepSeek | https://arxiv.org/abs/2512.02556 | k = 2048, 64×128 FP8 indexer, KL warm-up, MQA-mode MLA | DeepSeek-V3.2(-Exp), GLM-5/5.1/5.2; vLLM, SGLang (nsa/dsa backend), FlashMLA | 06 (upgrade existing) |
| MUST | MSA (MiniMax Sparse Attention) | 2026 | MiniMax | https://arxiv.org/abs/2606.13392 | Block (128) top-16 per GQA group; max-pooled token index scores | MiniMax-M3 | 06 (2026 indexers) |
| MUST | CSA/HCA: numbers | 2026 | DeepSeek | https://arxiv.org/abs/2606.19348 | Adds top-k 512/1024, FP4 indexer, sinks, FP8+BF16 RoPE KV | DeepSeek-V4-Pro/Flash; vLLM (deepseek_v4 sparse_mla) | 07 (upgrade existing) |
| MUST | FP8 / FP4 KV cache in production | 2024–26 | vLLM, NVIDIA, DeepSeek | https://vllm.ai/blog/2025-09-29-deepseek-v3-2 ; NVIDIA NVFP4 blog (see audit 12) | Bytes per element 2 → 1 → about 0.5, block scales, dequant before math | vLLM, SGLang, TRT-LLM; V3.2 656 B, V4.1 288 B per token | 09 |
| MUST | MoBA / NSA: numbers and adoption | 2025 | Moonshot / DeepSeek | https://arxiv.org/abs/2502.13189 ; https://arxiv.org/abs/2502.11089 | Add block/top-k numbers, forced blocks, Kimi deployment, full-attention top layers | Kimi service (MoBA); NSA → lineage into DSA and InfLLM-v2 | 06 (upgrade existing) |
| SHOULD | Sparse Transformer → GPT-3 alternation | 2019 | OpenAI | https://arxiv.org/abs/1904.10509 | Factorized strided/fixed patterns, O(n√n); GPT-3 alternated dense/banded | GPT-3; gpt-oss ("Following GPT-3") | 06 (history line) |
| SHOULD | PyramidKV + Ada-KV (budget allocation) | 2024 | arXiv / NeurIPS 2025 | https://arxiv.org/abs/2406.02069 ; https://arxiv.org/abs/2407.11550 | Same scorer; budget varies by layer (pyramid) or by head (adaptive) | kvpress PyramidKVPress, AdaKVPress | NEW: KV eviction |
| SHOULD | KVzip | 2025 | NeurIPS 2025 oral | https://arxiv.org/abs/2505.23416 | Query-agnostic scores from a context-reconstruction pass; reusable cache | kvpress KVzipPress | NEW: KV eviction |
| SHOULD | DMC → DMS (learned merge/evict) | 2024–25 | ICML 2024 / NeurIPS 2025 (NVIDIA/Edinburgh) | https://arxiv.org/abs/2403.09636 ; https://arxiv.org/abs/2506.05345 | Model learns per-token append/merge or delayed evict; retrofit, about 1K steps | NVIDIA Qwen3-8B-DMS-8x checkpoint (secondary source); kvpress DMSPress | NEW: KV eviction (last card) |
| SHOULD | QSA (Qwen Sparse Attention) | 2026 | Qwen | https://arxiv.org/abs/2608.30320 | Indexer on 4-token pooled keys; O(n²/r) indexer; 2048-token budget | Qwen3.8-Flash-Next (report) | 06 (2026 indexers) |
| SHOULD | IndexCache / IndexShare | 2026 | Tsinghua + Z.ai | https://arxiv.org/abs/2603.12201 | Reuse top-k indices across ~4 layers; about 75% less indexer compute | GLM-5.2 | 06 (2026 indexers) |
| SHOULD | InfLLM-v2 | 2025 | ICLR 2026 (Tsinghua/OpenBMB) | https://arxiv.org/abs/2509.24663 | NSA-like selection with no extra params; dense below 8K, sparse above | MiniCPM4 / 4.1 | 06 |
| SHOULD | XAttention + FlexPrefill (prefill scorers) | 2025 | ICML 2025 (MIT) / ICLR 2025 oral | https://arxiv.org/abs/2503.16428 ; https://arxiv.org/abs/2502.20766 | Cheaper block-importance estimate (antidiagonals / JS test); threshold not fixed k | Research (Block-Sparse-Attention lib); vision/video too | 06 (next to MInference) |
| SHOULD | SeerAttention / SeerAttention-R | 2024–25 | Microsoft | https://arxiv.org/abs/2410.13276 ; https://arxiv.org/abs/2506.08889 | Learned block gate distilled from frozen model; R version for decode | Research (TileLang kernels) | 06 |
| SHOULD | Offloaded retrieval: RetrievalAttention / MagicPIG / ShadowKV | 2024–25 | MSR / ICLR 2025 spotlight (CMU) / ByteDance+CMU | https://arxiv.org/abs/2409.10516 ; https://arxiv.org/abs/2410.16179 ; https://arxiv.org/abs/2410.21465 | KV moved to CPU; ANN, LSH sampling, or landmarks pick what returns | Research systems | 06 (or 09 storage tier) |
| SHOULD | Compressive Transformer | 2019 | ICLR 2020 (DeepMind) | https://arxiv.org/abs/1911.05507 | Old memories compressed at rate c rather than discarded | Historical; conceptual ancestor of CSA/HCA | 07 (lineage) |
| SHOULD | Gist tokens + Activation Beacon | 2023 / 2024 | NeurIPS 2023 / ICLR 2025 | https://arxiv.org/abs/2304.08467 ; https://arxiv.org/abs/2401.03462 | Learned tokens absorb a span's KV; source KV dropped | Research (FlagEmbedding) | 07 |
| SHOULD | KVQuant | 2024 | NeurIPS 2024 (Berkeley) | https://arxiv.org/abs/2401.18079 | Pre-RoPE per-channel keys, non-uniform, dense-and-sparse outliers | Research; ideas in later 4-bit KV | 09 |
| SHOULD | MInference: production note | 2025 | Qwen | https://arxiv.org/abs/2501.15383 | Vertical-Slash sparse prefill in Qwen2.5-1M deployment, 3–7× at 1M | Qwen2.5-1M (vLLM/BladeLLM) | 06 (upgrade existing) |
| SKIP | Reformer (LSH) | 2020 | ICLR 2020 | https://arxiv.org/abs/2001.04451 | LSH buckets, shared QK | Not adopted; LSH idea lives on in MagicPIG | footnote 06 |
| SKIP | Routing Transformer | 2020 | TACL 2021 | https://arxiv.org/abs/2003.05997 | k-means routing | Not adopted; conceptual precursor of MoBA | footnote 06 |
| SKIP | Landmark Attention | 2023 | NeurIPS 2023 (EPFL) | https://arxiv.org/abs/2305.16300 | Landmark token per block retrieves blocks | Precursor of block retrieval | tip in MoBA card |
| SKIP | Scissorhands / FastGen | 2023 | NeurIPS 2023 / ICLR 2024 | https://arxiv.org/abs/2305.17118 ; https://arxiv.org/abs/2310.01801 | H2O sibling / head-profiled policies | Subsumed by H2O + DuoAttention cards | — |
| SKIP | Infini-attention | 2024 | Google | https://arxiv.org/abs/2404.07143 | Linear compressive memory beside local attention | No released model; belongs to ch04 | — |
| SKIP | AutoCompressor / ICAE | 2023 | EMNLP 2023 / ICLR 2024 | https://arxiv.org/abs/2305.14788 ; https://arxiv.org/abs/2307.06945 | Soft-prompt summary vectors | Niche | — |
| SKIP | SpargeAttn | 2025 | ICML 2025 (Tsinghua) | https://arxiv.org/abs/2502.18137 | Universal sparse+quantized kernel | Mostly diffusion/video | — |
| SKIP | Twilight (top-p budget) | 2025 | NeurIPS 2025 spotlight | https://arxiv.org/abs/2502.02770 | Adaptive budget via top-p over selected tokens | Research | tip in Quest card |
| SKIP | LServe | 2025 | MLSys 2025 (MIT) | https://arxiv.org/abs/2502.14866 | System combining streaming heads + page selection | Research system | tip in Duo/Quest cards |
| SKIP | InfLLM v1 | 2024 | NeurIPS 2024 | https://arxiv.org/abs/2402.04617 | Training-free block memory units | Superseded by v2 | — |
| SKIP | Token merging (ToMe; KV merging variants) | 2022+ | ICLR 2023 (vision) | https://arxiv.org/abs/2210.09461 | Merge similar tokens | Vision; LLM-KV variants niche (DMC covers learned merge) | — |
| SKIP | The Sparse Frontier (study) | 2025 | arXiv (Edinburgh/Cohere, UNVERIFIED affiliations) | https://arxiv.org/abs/2504.17768 | Evaluation, not a mechanism | — | tip source |

---

## 3. Taxonomy table

Four families. Column meanings: "decided when" = static (fixed before deployment), prefill (once per prompt), per query (every step). "Cache shrinks?" = are records deleted or never stored.

| Mechanism | Family | Granularity | Decided when | Scorer cost (per decode query, N = context) | Cache shrinks? | Trained? |
|---|---|---|---|---|---|---|
| Sparse Transformer / Longformer / BigBird | (i) static | token pattern | static | 0 | no (pattern only) | trained |
| SWA + global layers (gpt-oss, Gemma 3, MiMo) | (i) static | token window, per layer | static | 0 | **yes on local layers** (ring buffer W) | trained |
| Llama 4 chunked attention | (i) static | 8192-token chunk | static | 0 | yes on chunked layers | trained |
| StreamingLLM | (i)+(iii) | token: 4 sinks + window | static | 0 | **yes, permanent** | training-free |
| Learned sink logit | softmax fix | per head scalar | static | 0 | n/a | trained |
| H2O / Scissorhands | (iii) eviction | token, per head | per step (online) | needs attention probabilities (does not fit fused FlashAttention) | **yes, permanent** | training-free |
| SnapKV (+PyramidKV/Ada-KV) | (iii) eviction | token (pooled), per head | end of prefill, once | L_obs × N attention, about 32N | **yes** (prompt only) | training-free |
| KVzip | (iii) eviction | token, per head | after prefill (reconstruction pass) | one extra forward over context | **yes**, query-agnostic | training-free |
| DuoAttention / RazorAttention | (iii) eviction | **head** | static (offline calibration) | 0 at runtime | **yes for streaming heads** | gate-only optimization / training-free |
| DMC / DMS | (iii)/(iv) learned | token, per head | per step, learned | about 0 (reuses a q/k neuron, UNVERIFIED) | **yes** (4×/8×) | retrofit-trained |
| MInference / FlexPrefill / XAttention | (ii) selection | block / vertical-slash, per head | prefill, per input | estimator on last queries or antidiagonals | no (prefill only) | training-free |
| SeerAttention / -R | (ii) selection | block 64/128 | prefill / per query | gate on pooled K: N/B | no | gate-trained (0.4B tokens for -R) |
| Quest | (ii) selection | page 16, per head | per query | 2·d MACs per page: about N/8 · d | **no** (+12.5% K metadata) | training-free |
| RetrievalAttention / MagicPIG / ShadowKV | (ii) selection + offload | token / LSH sample / 8-token chunk | per query | ANN sublinear / hash lookups / N/8 landmarks | GPU yes, total no (moved to CPU) | training-free |
| MoBA | (ii) selection | block 512–4096, per head | per query | N/B mean-key dots | no | trained (CPT) |
| NSA | (ii)+(iv) | block 64, per GQA group | per query | compression branch (N/16) doubles as scorer | no (adds compressed KV) | trained (native) |
| InfLLM-v2 | (ii) | block 64, per GQA group | per query (sparse only if N > 8K) | kernels 32/16, no extra params | no | trained (switchable) |
| DSA | (ii) | **token**, shared by all heads (MQA) | per query | 64 heads × 128 FP8 over all N: O(N) per query, O(N²) prefill | no (adds indexer K about 132 B/token/layer, derived) | trained (2-stage KL) |
| QSA | (ii) | 4-token micro-block → token | per query | 4 heads × 128 over N/4 pooled keys | no | trained (2-stage KL) |
| MSA | (ii) | block 128, per GQA group | per query | 1 index head per group over N tokens, max-pooled | no | trained (scratch or CPT) |
| IndexCache / IndexShare | (ii) modifier | reuses indices across layers | per query, on Full layers only | indexer ÷ about 4 | no | training-free or -aware |
| CSA (V4) | (iv)+(ii) | compressed entry (m = 4, overlap 8) | per query | FP4 indexer over N/4 | **yes ×1/4** | trained |
| HCA (V4) | (iv) | compressed entry (m' = 128) | static (read all) | 0 | **yes ×1/128** | trained |
| Compressive Transformer | (iv) | c = 2–4 pooled memories | when a memory ages out | 0 | yes | trained |
| Gist / Activation Beacon | (iv) | learned tokens per span | at chunk completion | 0 | yes (Beacon 8× KV) | trained |
| KIVI / KVQuant / FP8 / NVFP4 | bytes | element | static | 0 | bytes shrink, count same | training-free (calibration) |

**Four-family comparison**

| Family | What shrinks | Reversible? | Failure mode | Canonical examples |
|---|---|---|---|---|
| (i) static mask | reads; cache too if window-only layers | n/a | fact outside window/pattern on all local layers; needs global layers | Sparse Transformer, Longformer, SWA:global, Llama 4 chunked |
| (ii) query-aware selection | **reads only**; cache kept or offloaded | yes: next query can pick other entries | scorer misses evidence; scorer cost O(N) per query remains | Quest, MoBA, NSA, DSA, MSA, QSA, MInference |
| (iii) permanent eviction | **records stored** | **no**: an evicted fact is gone for all later queries | the future question needs what the past attention ignored | StreamingLLM, H2O, SnapKV, DuoAttention, DMS |
| (iv) compression into fewer records | records (merged) | no for detail; association may survive | exact IDs blurred inside pooled records | Compressive Transformer, Beacon, DMC, CSA/HCA |

---

## 4. Card data

### Sliding window + interleaved global layers (2023–2026; Mistral, Google, OpenAI, Meta, Xiaomi)
- url: https://arxiv.org/abs/2508.10925 (gpt-oss §2.2); https://arxiv.org/abs/2503.19786 (Gemma 3); https://arxiv.org/abs/2601.02780 (MiMo-V2-Flash); Llama 4 HF docs https://huggingface.co/docs/transformers/model_doc/llama4
- one-line idea: most layers attend to a fixed recent window, so their KV is a ring buffer of W entries. A few global layers keep full history for long-range retrieval.
- key formula: $A^{\text{local}}_t=\{j:\,t-W<j\le t\}$, $A^{\text{global}}_t=\{j\le t\}$; KV bytes $\approx \big(L_g\,N + L_\ell\,\min(N,W)\big)\cdot b_{\text{tok}}$
- concrete numbers:
  - **gpt-oss:** 1:1 alternation, band 128, 64 Q / 8 KV heads × 64 dims, 131,072 ctx via YaRN on dense layers. It cites GPT-3's dense/banded alternation. Derived: 2 KiB/token/layer → a dense layer holds 256 MiB at 128K, a banded layer 256 KiB (1024× smaller).
  - **Gemma 3:** 5 local : 1 global, local span 1024, global RoPE base 1M. "Global only" has 60% memory overhead, <15% with 1:3 and window 1024 (§5.2). Derived: at 128K, 5:1 with W = 1024 keeps about 17% of all-global KV.
  - **MiMo-V2-Flash:** 48 layers = 39 SWA + 9 GA, window 128, 5:1. SWA layers 8 KV heads, GA layers 4. Learned sink bias. Abstract: "nearly 6×" KV/compute reduction.
  - **Llama 4:** `attention_chunk_size` 8192, every 4th layer global with no RoPE (`no_rope_layer_interval` 4).
  - **Mistral 7B:** W = 4096 (not re-fetched). **Gemma 2:** 1:1 with local 4096 (not re-fetched, UNVERIFIED).
- diagram idea: a strip of 12 layer bars coloured local/global for each model (gpt-oss 1:1, Gemma 3 5:1, Llama 4 3:1, MiMo 5:1). Beside it, a 32×32 query×key mask for one local and one global layer. Below, a stacked bar of KV bytes at 128K per model: local layers are thin slivers, global layers tall.
- tips:
  - Retrieval works only through global layers. Local layers move information within W per layer, so the effective receptive field is about W × depth (theory, not a guarantee).
  - Counter-intuitive (MiMo §2.2.1, 32B ablation): with a sink, **W = 128 beat W = 512** after long-context extension (RULER-32k 89.4 vs 84.7; MRCR 32.5 vs 19.6). Without the sink, SWA lost on MMLU (54.9 vs 57.3).
  - The serving engine must manage two cache kinds (bounded and growing). The global layers' cache still grows as O(N).
- verification: gpt-oss §2.2 quotes; Gemma 3 §2 and §5.2 quotes; MiMo §2.1, Table 1, §2.2.1 Tables 2–4; Llama 4 config defaults in the HF docs.

### Attention sinks: StreamingLLM → learned sink logit (2023 ICLR 2024 → 2025–26 gpt-oss / MiMo / DeepSeek-V4)
- url: https://arxiv.org/abs/2309.17453
- one-line idea: softmax must put its mass somewhere, so models dump it on the first tokens. Keep those 4 "sink" tokens and a window would otherwise collapse. Newer models learn a per-head logit in the denominator instead.
- key formula: StreamingLLM $A_t=\{0,1,2,3\}\cup\{t-W+1..t\}$, positions re-indexed **within the cache**. Learned sink: $o_t=\dfrac{\sum_{j\in A_t}e^{s_{tj}}v_j}{e^{z_h}+\sum_{j\in A_t}e^{s_{tj}}}$
- concrete numbers: 4 sinks (1–2 are insufficient, §4.4). Cache 2048 (Llama-2) / 1024 (Falcon, Pythia, MPT) on PG-19. Ran >4M tokens. 22.2× per-token speedup vs sliding window with recomputation. A 160M model pretrained with one learnable sink token streams stably. TRT-LLM's Mistral notebook uses window 2048 + 4 sinks.
- diagram idea: (a) a heatmap of attention averaged over heads for 64 keys, with column 0 bright. (b) Three caches at step t = 5000: window-only (sinks evicted, perplexity spikes), sinks + window (kept), learned sink (no token, just an extra "∅" column on the softmax bar). (c) Position IDs [0..3, 4996..4999] remapped to [0..7].
- tips:
  - Does **not** extend memory: accuracy drops to 0 once the answer is farther back than the cache (App. C). It is a stability fix, not retrieval.
  - Learned sinks let a head "attend to nothing", which is what makes 128-token windows viable (MiMo ablation).
  - Gated attention (ch08) removes sinks by another route, the output gate.
- verification: arXiv HTML §3–4, Appendix A/C; gpt-oss §2.2; V4 §2.3.3 Eq. 27; MiMo §2.2 Eqs. 1–4.

### H2O — Heavy-Hitter Oracle (2023, NeurIPS 2023)
- url: https://arxiv.org/abs/2306.14048
- one-line idea: a few tokens receive most attention, cumulatively. Keep those and the recent tokens, and evict the rest permanently.
- key formula: $S^{(h)}_j \mathrel{+}= a^{(h)}_{t,j}$ each step; $\mathcal{K}^{(h)}_t=\mathrm{Recent}_{B/2}\cup\mathrm{Top}_{B/2}(S^{(h)})$, evict $\arg\min S$ when full.
- concrete numbers: budget 20% (10% heavy hitters + 10% recent; the budget is split evenly, §5 Baselines). Up to 29× throughput vs DeepSpeed Zero-Inference / HF Accelerate, 3× vs FlexGen (OPT-6.7B/30B), latency −1.9×. Derived example: Llama-3-8B at 32K = 4 GiB KV → 0.8 GiB at 20%.
- diagram idea: decode timeline t = 1..40 (x) × cache slots (y). Kept cells are coloured by reason (recent band = diagonal stripe, heavy hitters = horizontal persistent lines). Evicted cells are greyed with an ✕ and never return. Side bar: accumulated score per token.
- tips:
  - Needs the attention probabilities each step, which fused kernels (FlashAttention) do not emit. This is one reason engines did not adopt it.
  - Accumulated score is biased toward early tokens (they have been summed longer).
  - Irreversible: our Friday correction, unread for 5K steps, is gone when the question arrives.
  - "Pitfalls of KV Cache Compression" (ACL 2026) finds certain instructions are effectively ignored under compression and identifies system-prompt-leakage factors: https://aclanthology.org/2026.acl-long.1926/
- verification: PDF text (extracted): NeurIPS 2023 header; "accumulated attention score"; "evenly assigns the caching budget"; abstract numbers.

### SnapKV (2024, NeurIPS 2024)
- url: https://arxiv.org/abs/2404.14469
- one-line idea: the prompt's last few queries already "know" what generation will need. Vote with their attention, pool, and keep the top prompt positions per head before decoding starts.
- key formula: $C_j=\sum_{i\in\text{obs}}\mathrm{softmax}(q_iK^\top/\sqrt d)_j,\ \tilde C=\mathrm{maxpool}_{k}(C),\ \mathcal K=\mathrm{Top}_B(\tilde C)\cup\text{obs}$
- concrete numbers: observation window L_obs = 32 (LongBench; 16 Needle; 64 Command-R). Pool kernel 7 (5 / 13). Budgets 1024 / 2048 / 4096. 3.6× decode speed and 8.2× memory efficiency at 16K. 380K ctx on 1×A100-80GB. Derived: a 16K prompt with budget 1024 keeps 6.25% (+32 obs).
- diagram idea: a prompt heatmap (rows = last 32 queries, cols = 16K prompt keys, binned). Below it the column-sum C, then the max-pooled C̃ (clusters widen), then a keep mask per head (4 heads, different masks). The decode region is shown uncompressed.
- tips:
  - Compresses the **prompt once**. Generated tokens are not evicted, so long reasoning outputs still grow the cache.
  - Pooling keeps neighbours of hits, which preserves multi-token entities like "order 7314".
  - Query-dependent: if the question arrives **after** compression (multi-turn, cached prefix), the votes were cast by the wrong queries. KVzip targets this.
  - PyramidKV/Ada-KV only change **how much** each layer/head keeps (next card).
- verification: arXiv HTML v2 §4 and experiment setup; NeurIPS 2024 proceedings page.

### PyramidKV + Ada-KV — budget allocation (SHOULD; 2024, arXiv / NeurIPS 2025)
- url: https://arxiv.org/abs/2406.02069 ; https://arxiv.org/abs/2407.11550
- one-line idea: use the same SnapKV scorer, but give lower layers more budget (pyramid) or let heads compete for one layer budget (Ada-KV).
- key formula: PyramidKV $B_\ell$ decreasing linearly in $\ell$ with $\sum_\ell B_\ell = L\bar B$; Ada-KV $\{B_h\}$ = counts of each head among the layer's $\mathrm{Top}_{H\bar B}$ of all heads' scores.
- concrete numbers: PyramidKV matches full KV on LongBench at **12%**. At 0.7% it gains up to +20.5 on TREC. 128 entries give 100 on NIAH for LLaMA-3-70B. Ada-KV is tested on 13 RULER + 16 LongBench sets.
- diagram idea: a layer × head grid of budget cells (32 × 8): uniform vs pyramid (row gradient) vs Ada (irregular, per-head).
- tips: an allocation policy, not a scorer; it composes with SnapKV. Ada-KV reasons from an upper bound on output change (paper's theory).
- verification: abstracts only (budget-curve shape UNVERIFIED beyond "more in lower layers").

### KVzip (SHOULD; 2025, NeurIPS 2025 oral)
- url: https://arxiv.org/abs/2505.23416
- one-line idea: score each KV pair by how much the model uses it when asked to **reconstruct** the context. The resulting compressed cache serves any future query.
- key formula: $\text{score}_j=\max_{i\in\text{reconstruction}}a_{i,j}$; keep $\mathrm{Top}_B$ (exact formula UNVERIFIED, abstract-level).
- concrete numbers: 3–4× KV reduction; about 2× FlashAttention decode latency; contexts up to 170K; LLaMA3.1, Qwen2.5, Gemma3.
- diagram idea: two queries hitting a SnapKV-pruned cache (query 2 misses its evidence) vs a KVzip-pruned cache (both hit).
- tips: query-agnostic suits prefix caching and multi-turn. It costs an extra pass over the context.
- verification: abstract.

### DuoAttention (+ RazorAttention) — head-level eviction (2024, ICLR 2025)
- url: https://arxiv.org/abs/2410.10819 ; https://arxiv.org/abs/2407.15891
- one-line idea: only some heads do long-range retrieval. Give them full KV; give the rest a constant sink + window cache.
- key formula: training $o_h=\alpha_h\,\mathrm{Attn}_{\text{full}}+(1-\alpha_h)\,\mathrm{Attn}_{\text{sink+recent}}$, $\mathcal L=\|h_{\text{full}}-h_\alpha\|_2^2+\lambda\|\alpha\|_1$ with $\lambda=0.05$; deploy $\alpha_h\to\{0,1\}$ by quantile threshold.
- concrete numbers: retrieval-head ratio 25% (Llama-2-7B MHA) / 50% (Llama-3-8B GQA). Streaming heads keep 64 sinks + 256 recent on long-context benchmarks; the ablation plateaus at 16 + 64. Gates trained 2,000 steps on synthetic passkey data (ten 32-word passkeys in BookSum), weights frozen. Memory up to 2.55× (MHA) / 1.67× (GQA); decode 2.18× / 1.50×; prefill 1.73× / 1.63×. With 8-bit weights + 4-bit KV: 3.3M tokens on 1×A100-80G. RazorAttention: >70% KV reduction, training-free, a "compensation token" summarizes dropped tokens.
- diagram idea: a heads × layers grid (32 × 8 for Llama-3-8B), half coloured "retrieval". For one retrieval and one streaming head, show the 128K-wide cache bar vs a 320-token bar. Derived total ≈ 8.02 GiB vs 16 GiB at 128K (8 GiB retrieval + 20 MiB streaming).
- tips:
  - Static and per head: zero runtime scorer cost and kernel-friendly (contrast H2O).
  - The same idea at layer granularity is the SWA:global hybrid, but trained in from scratch.
  - Under GQA a "head" is a KV head, so the savings are smaller (50% retrieval vs 25%).
- verification: arXiv HTML §3 (gate, loss, data), experiment settings, efficiency table; RazorAttention abstract; RazorAttention ICLR 2025 via DBLP (search result).

### DMC → DMS — learned merge/evict retrofit (SHOULD; 2024 ICML / 2025 NeurIPS)
- url: https://arxiv.org/abs/2403.09636 ; https://arxiv.org/abs/2506.05345
- one-line idea: make the model decide per token, per head whether to append a new KV slot or merge into the last one (DMC). DMS instead evicts with a delay. Both are retrofitted cheaply.
- key formula (DMC; exact weighting UNVERIFIED): $\alpha_t\in\{0,1\}$; if 1, $k_{\text{last}}\leftarrow\frac{z\,k_{\text{last}}+\omega_t k_t}{z+\omega_t}$, $z\mathrel{+}=\omega_t$; else append. DMS: an eviction decision made at t takes effect after a delay window (window size UNVERIFIED).
- concrete numbers: DMC up to 4× compression on Llama 2 7B/13B/70B; beats GQA up-training, H2O and TOVA; throughput "up to 7×" on H100 (abstract; metadata says 3.7×, setup-dependent). DMS: 8× compression with **1K training steps**. Qwen-R1 32B at equal memory reads: +12.0 AIME24, +8.6 GPQA, +9.7 LiveCodeBench.
- diagram idea: a token strip with per-token decisions (append = new slot, merge = arrow into previous slot, evict-after-delay = fading cell). The cache-length curve grows at about 1/8 the slope.
- tips: inference-time "hyper-scaling": fewer KV bytes means more reasoning tokens in the same budget. Learned, so it is not plug-and-play. NVIDIA released Qwen3-8B-DMS-8x (secondary source, UNVERIFIED details).
- verification: both abstracts.

### Quest (2024, ICML 2024, MIT Han Lab)
- url: https://arxiv.org/abs/2406.10774
- one-line idea: keep the whole cache, in 16-token pages, each with channel-wise min/max key metadata. Per query, an upper bound on any key's score in the page picks the top pages to read.
- key formula: $U_p=\sum_{i=1}^{d}\max(q_i m_{p,i},\,q_i M_{p,i})\ \ge\ \max_{j\in p}q^\top k_j$; read $\mathrm{Top}_K(U)$ pages.
- concrete numbers: page 16. Budgets 256–4096 tokens. The budget at which LongBench reaches full-cache quality is UNVERIFIED (the fetch was ambiguous between "1K" and "11K"). First 2 layers kept dense. Up to 7.03× self-attention speedup at 32K with budget 2048; 2.23× end-to-end with 4-bit weights (1.74× FP16). Derived: metadata = 2 vectors per 16 keys = +12.5% of K (6.25% of K+V); at 32K with budget 2048, read 128 of 2048 pages.
- diagram idea: one query vector (8 channels) × 6 pages with min/max bars per channel. Compute U per page, highlight top-2, then show the true max score inside each (always ≤ U). Then a 2048-page strip with 128 lit.
- tips:
  - Selection, not eviction: a different future query can read different pages. Memory is not saved; bandwidth is.
  - The bound is loose for wide pages and channels with large spread, hence small pages.
  - Training-free ancestor of DSA/MSA/QSA: same "cheap scorer → top-k → exact attention" loop with a heuristic scorer instead of a learned one.
  - Fixed budget is suboptimal per query (Twilight: top-p budget; Sparse Frontier: decode tolerates higher sparsity than prefill).
- verification: arXiv HTML §3–4, §5 (speedups); "ICML" keyword on the page (ICML 2024).

### Offloaded retrieval: RetrievalAttention / MagicPIG / ShadowKV (SHOULD; 2024–25)
- url: https://arxiv.org/abs/2409.10516 ; https://arxiv.org/abs/2410.16179 ; https://arxiv.org/abs/2410.21465
- one-line idea: move the cache off the GPU and fetch only what the query needs. RetrievalAttention uses an ANN index; MagicPIG uses LSH **sampling**, which is unbiased where top-k is biased; ShadowKV keeps low-rank pre-RoPE keys and landmarks on the GPU.
- key formula: MagicPIG estimator $o\approx\frac{\sum_{j\in S}e^{s_j}v_j/p_j}{\sum_{j\in S}e^{s_j}/p_j}$, where $p_j$ = probability key j collides with q in at least 2 of L SimHash tables (estimator form UNVERIFIED; collision rule verified). ShadowKV $K_{\text{pre}}\approx U_rV_r$, $r=160$; landmark $\ell_c=\mathrm{mean}_{j\in c}\mathrm{RoPE}(k_j)$, chunk = 8.
- concrete numbers: RetrievalAttention touches 1–3% of KV; 128K on an RTX 4090 at 0.188 s/token (8B). MagicPIG up to 5× throughput, 54 ms/token on a 4090 at 96K (latest abstract; v2 said 1.9–3.9× and 110 ms). ShadowKV: chunk 8, rank 160, 48 outlier chunks on GPU, budget about 1.56%, 6× batch, 3.04× throughput on A100.
- diagram idea: a GPU/CPU two-tier box. Per query, arrows show what crosses PCIe (MagicPIG: hashes and samples; ShadowKV: values of the selected 1.56%), with bytes per token per tier.
- tips: wins when HBM, not compute, is the limit. PCIe latency bounds decode. Query/key OOD makes stock ANN indexes weak (RetrievalAttention's core point).
- verification: three abstracts; ShadowKV HTML experiment settings; MagicPIG ICLR 2025 spotlight via repo tag (search).

### DSA — DeepSeek Sparse Attention (2025, DeepSeek; upgrade of existing card)
- url: https://arxiv.org/abs/2512.02556
- one-line idea: a tiny FP8 multi-head "lightning indexer" scores every past token. The main MLA attention reads only the top 2048, shared by all heads.
- key formula: $I_{t,s}=\sum_{j=1}^{64}w_{t,j}\,\mathrm{ReLU}(q^I_{t,j}\!\cdot k^I_s)$, $A_t=\mathrm{Top}_{2048}(I_{t,:})$; warm-up $\mathcal L=\sum_t\mathrm{KL}\big(p_{t,:}\,\|\,\mathrm{softmax}(I_{t,:})\big)$, $p$ = head-summed, L1-normalized dense attention.
- concrete numbers: indexer 64 heads × 128 dims, FP8. Top-k 2048 (padded with −1 if shorter). MQA-mode MLA: 128 Q heads share one latent KV, 512 + 64 dims. Warm-up 1K steps / 2.1B tokens at LR 1e-3 (indexer only). Sparse stage 15K steps / 943.7B tokens at LR 7.3e-6, indexer input detached. FP8 MLA cache 656 B/token/layer; indexer K about 132 B/token/layer (derived: 128 FP8 + one fp32 scale). Derived per decode token per layer at 128K: indexer ≈1.07 G MACs (FP8) vs sparse main ≈0.29 G MACs vs dense MLA ≈18.3 G MACs; bytes read ≈ 17.3 MB + 1.3 MB vs 86 MB.
- diagram idea: a 128K-key strip. Top: the indexer score curve (64 heads collapsed) with the top-2048 ticks lit (1.56%). Middle: the main attention reading only lit columns, with 128 query heads fanning out from one latent. Right: a MAC/byte bar triplet (dense / indexer / main) using the derived numbers.
- tips:
  - At long context **the indexer dominates the arithmetic** (≈3.8× the sparse main attention at 128K, derived). DSA pays off through FP8, small dims and bytes, not by removing O(N).
  - Token-level selection needs gather kernels (FlashMLA sparse); block methods (MSA/NSA) are friendlier to tensor cores.
  - Two-stage recipe (dense KL warm-up → sparse joint) is now the industry template (QSA, MSA copy it).
  - Cache is not smaller: indexer keys add about 20% bytes (derived 132/656).
- verification: HTML §2.1, §2.1.1, §2.3; HF config; vLLM blog (656 B layout, top-k 2048, indexer block size 64); FlashMLA README (FP8/FP4 sparse kernels); SGLang docs ("DeepSeek V3.2/GLM-5 Usage", nsa backend).

### MSA — MiniMax Sparse Attention (2026, MiniMax; production in MiniMax-M3)
- url: https://arxiv.org/abs/2606.13392
- one-line idea: DSA-style learned index, but scored and read at **128-token block** granularity, separately per GQA group. Kernels are co-designed so block reads hit tensor cores.
- key formula: $s^{(g)}_{t,b}=\max_{j\in b,\ j\le t} q^{\mathrm{idx},(g)}_t\cdot k^{\mathrm{idx}}_j$; $A^{(g)}_t=\mathrm{Top}_{16}\big(s^{(g)}_{t,:}\big)\cup\{b_{\text{local}}\}$; KL to group-averaged main attention (teacher detached).
- concrete numbers: block B_k = 128, k = 16 → **2,048 tokens per query per group** (fixed as N grows). Index branch = 1 query head per GQA group + 1 shared key head ("only two projection matrices"). Local block always selected; a sink emerges naturally in block 0. Training: from scratch with 40B-token indexer warm-up then sparse for the rest of 3T (MSA-PT); or from a GQA checkpoint with 40B warm-up + 400B sparse (MSA-CPT). Reported: 109B model "on par with GQA"; 28.4× lower per-token attention compute at 1M; kernel 14.2× prefill / 7.6× decode on H800. Derived: 2,048 / 1M = 0.2% read.
- diagram idea: a query×key grid (64 × 64 blocks) with 4 GQA groups as 4 colour layers, each lighting different 16 blocks plus the diagonal block. Inset: "KV-outer" kernel, one KV block gathering all queries that chose it.
- tips:
  - Max-pool of token scores (not mean-pooled keys like MoBA) keeps a single sharp hit visible inside a block.
  - Per-group selection costs more index work than DSA's single shared set, but keeps GQA head diversity.
  - "Exp-free top-k": softmax preserves order, so rank raw scores.
- verification: HTML §3.1–3.2, §4.1–4.2, §5.1; abstract numbers. Production M3 numbers (>9× prefill, >15× decode vs M2) are secondary (UNVERIFIED).

### QSA — Qwen Sparse Attention (SHOULD; 2026, Qwen3.8-Next report)
- url: https://arxiv.org/abs/2608.30320
- one-line idea: the indexer scores **4-token micro-blocks** of pooled keys (cutting indexer work by 4×), then expands the chosen blocks to a 2048-token read.
- key formula: $\bar k_b=\mathrm{RMSNorm}(\mathrm{mean}_{j\in b}k^I_j),\ r=4$; $I_{t,b}=\sum_{h=1}^{4}\mathrm{ReLU}(q^I_{t,h}\cdot\bar k_b)$ over complete blocks; read ≤512 blocks (≤2048 tokens) + the current incomplete block.
- concrete numbers: indexer 4 Q heads + 1 K head, 128 dims (64 with RoPE). Stage 1: 1K steps, LR 1e-3, 8 × 256K/step ≈ 2B tokens (indexer only, KL with max-pooled teacher). Stage 2: 8K steps, LR 2.5e-5, 96 × 256K ≈ 200B tokens. 1M context: attention module 7.6× prefill, 4.9× decode vs dense; speedup starts at 64K. RULER >512K 90.08 → 93.00; MRCR 8-needle at 1M 20.71 → 26.44.
- diagram idea: one row of 32 keys grouped in 4s → 8 pooled keys → scores → 2 lit blocks → 8 token reads. Annotate "indexer O(n²/4)".
- tips: sits in the 1-in-4 global layers of a GDN hybrid, so sparse retrieval complements recurrent state (link ch08). No separate sliding window.
- verification: HTML §2.1.2 (implementation, training, efficiency, evaluation).

### IndexCache / IndexShare (SHOULD; 2026, Tsinghua + Z.ai; GLM-5.2)
- url: https://arxiv.org/abs/2603.12201
- one-line idea: adjacent layers' top-k sets are highly similar, so compute indices only on "Full" layers and reuse them on the following "Shared" layers.
- key formula: $A^{(\ell)}_t=A^{(F(\ell))}_t$ for Shared $\ell$; training-aware variant: $\mathcal L=\mathrm{KL}\big(\tfrac1{|\mathcal S|}\sum_{\ell\in\mathcal S}p^{(\ell)}\,\|\,\mathrm{softmax}(I^{(F)})\big)$
- concrete numbers: about 75% of indexer compute removed; 1.82× prefill and 1.48× decode vs DSA (30B model). GLM-5.2 card: one indexer per 4 sparse layers, 2.9× lower per-token FLOPs at 1M.
- diagram idea: a layer stack of 8 with indexer icons on layers 1 and 5 only; arrows carry the 2048-index set down; an overlap heatmap of top-k sets between layer pairs (qualitative).
- tips: attacks the remaining O(N²) indexer term. Greedy layer choice (training-free) vs multi-layer distillation (training-aware).
- verification: abstract; GLM-5.2 model card quote. Index-overlap numbers not given (UNVERIFIED).

### NSA (2025, DeepSeek/PKU; upgrade existing)
- url: https://arxiv.org/abs/2502.11089
- one-line idea: three gated branches: compressed tokens (coarse, all history), selected 64-token blocks (fine, top-16), sliding window 512.
- key formula: $o_t=\sum_{c\in\{\text{cmp},\text{slc},\text{win}\}}g^c_t\,\mathrm{Attn}(q_t,\tilde K^c_t,\tilde V^c_t)$; $p^{\text{slc}}_t$ = sum of $p^{\text{cmp}}_t$ over each 64-block, summed over the 16 heads of the group; keep the top 16 blocks (1 initial + 2 local forced).
- concrete numbers: l = 32, d = 16, l' = 64, n = 16, w = 512. 27B MoE (3B active), 30 layers, GQA 4 groups × 16 heads. Up to 9.0× forward / 6.0× backward at 64K (8×A100 vs Triton FA2); decode up to 11.6× lower memory access. Derived reads per query at 64K ≈ 4,095 cmp + 1,024 slc + 512 win ≈ 5.6K (8.6%).
- diagram idea: three stacked 64K-key strips (cmp tokens as 16-stride ticks; 16 lit 64-blocks; final 512 window) with gate bars g_cmp/g_slc/g_win.
- tips: the compression branch doubles as the free scorer (no separate indexer). InfLLM-v2 notes NSA's three KV sets triple KV storage during pretraining.
- verification: HTML §3.2–3.4, §4.1, §5.1–5.2.

### InfLLM-v2 (SHOULD; 2025, ICLR 2026; MiniCPM4)
- url: https://arxiv.org/abs/2509.24663 ; config in https://huggingface.co/openbmb/MiniCPM4-8B
- one-line idea: NSA-style block selection that reuses the dense KV (no extra parameters or branches) and switches to sparse only above a length threshold, preserving pretrain-short/finetune-long.
- key formula: kernel scores $s_{t,\kappa}=q_t\cdot\bar k_\kappa$ over 32-token kernels (stride 16), pooled to 64-token blocks; $A_t=\mathrm{Top}_{64}\cup B_0\cup\mathrm{Win}_{2048}$ if $N>8192$, else dense.
- concrete numbers: kernel 32 / stride 16, block 64, top-k 64 (→4,096 tokens), init_blocks 1, window 2048, dense_len 8192. 81% sparsity at parity with full attention. Top-k selection "reduces 60%" of cost vs NSA. MiniCPM4: 7× on 128K documents on end-side devices vs Qwen3-8B. Derived ≤6.2K tokens read per query (4.7% at 128K).
- diagram idea: a length axis with a switch at 8K: dense mask left, sparse mask (init + 64 blocks + band) right.
- tips: shares top-k across the GQA group. The scorer is "semantic kernels", not mean pooling.
- verification: MiniCPM4 report HTML (InfLLM v2 section); model card `sparse_config` defaults; ICLR 2026 via proceedings search hit.

### MoBA (2025, Moonshot; upgrade existing)
- url: https://arxiv.org/abs/2502.13189
- concrete numbers to add: block 512 / top-3 (8K: 81.25% sparse; 32K: 95.31%); 2048 / top-3 (hybrid 32K); **4096 / top-12 at 1M → 95.31%**. RULER 128K 0.7818 vs 0.7849 full. 6.5× prefill at 1M; 16× attention time at 10M. Last 3 of 32 layers kept full attention. Deployed for Kimi long-context requests.
- tips to add: mean-pooling dilutes one sharp key in a 4096-token block (contrast MSA's max-pool). Landmark Attention (NeurIPS 2023) is the block-retrieval precursor; Routing Transformer the routing precursor.
- verification: HTML §2.2–2.3, §3.1–3.4, abstract.

### CSA/HCA (2026, DeepSeek-V4; upgrade existing ch07 card)
- url: https://arxiv.org/abs/2606.19348
- concrete numbers to add: CSA m = 4 (overlapping), indexer 64 heads × 128 dims with FP4 QK, **top-k 512 (Flash) / 1024 (Pro)** compressed entries. HCA m' = 128, dense. SWA n_win = 128. Learnable per-head sinks. Head dim c = 512 shared-KV MQA. Q heads 64 (Flash) / 128 (Pro); layers 43 / 61. KV FP8 + BF16 RoPE (64 dims). Flash L1–2 = pure SWA; Pro L1–2 = HCA. Vs V3.2 at 1M: Pro 27% FLOPs / 10% KV, Flash 10% / 7%; about 2% of a BF16 GQA8 cache. Derived at 1M tokens: CSA 262,144 entries (reads 1,024 + 128), HCA 8,192 entries (reads all + 128).
- diagram idea: a 1M-token bar → CSA row (262K ticks, 1,024 lit) and HCA row (8,192 ticks, all lit) + a 128 raw window, with a bytes bar vs V3.2.
- tips: CSA = compression (records ÷4) **and** selection (read 1–4% of entries at 1M). The top-k is smaller than V3.2's because each entry spans 4+ tokens.
- verification: HTML §2.3.1–2.3.4, §4.2.1 table; abstract and §1 ratios.

### XAttention + FlexPrefill — training-free prefill scorers (SHOULD; ICML 2025 / ICLR 2025 oral)
- url: https://arxiv.org/abs/2503.16428 ; https://arxiv.org/abs/2502.20766
- one-line idea: estimate block importance cheaply before computing it, then keep the smallest block set reaching a cumulative-mass threshold. XAttention sums antidiagonals inside each block; FlexPrefill picks per head between query-aware and vertical-slash patterns using a JS-divergence test.
- key formula: XAttention $\hat S_b=\sum_{(i,j)\in\mathrm{antidiag}_S(b)}A_{ij}$, choose the minimal $\mathcal B$ with $\sum_{b\in\mathcal B}\hat P_b\ge\tau$. FlexPrefill: $\mathrm{JS}(\bar a_{\text{est}}\|\bar a_{\text{true}})<\delta\Rightarrow$ query-aware, else vertical-slash; keep indices to cumulative mass $\gamma$.
- concrete numbers: XAttention stride S = 8 / 16, τ = 0.9 (video). Density 6.89% at 128K (S = 8), 7.32% at 256K (S = 16). Up to 13.5× attention speedup. Pattern selection 24.9× faster than MInference, 5.9× than FlexPrefill. Block size: 128 is common in such kernels, but the paper text did not state it (UNVERIFIED).
- diagram idea: one 8 × 8 block with its antidiagonal cells outlined (stride 8), sums → block heatmap → cumulative curve crossing τ = 0.9.
- tips: threshold beats fixed k (adapts per head and input). Prefill only; the decode cache is untouched. The same kernels serve video diffusion.
- verification: XAttention HTML §2.1–2.3, §3.2–3.4; PMLR v267 entry; FlexPrefill abstract (ICLR 2025 oral).

### SeerAttention / SeerAttention-R (SHOULD; 2024–25, Microsoft)
- url: https://arxiv.org/abs/2410.13276 ; https://arxiv.org/abs/2506.08889
- one-line idea: bolt a small learnable gate onto a frozen model. It predicts block importance from pooled Q/K and is self-distilled from the model's own attention map. The R version drops query pooling to work in decode.
- key formula: $g_{t,b}=\mathrm{softmax}_b\big(W_q\tilde q_t\cdot W_k\,\mathrm{pool}(K_b)\big)$, trained with $\mathrm{KL}(\mathrm{pool}(A)\,\|\,g)$ (pooling form UNVERIFIED).
- concrete numbers: R: blocks 64 / 128, 4K token budget near-lossless on AIME, 0.4B training tokens, up to 9× vs FA3 on H100 at 90% sparsity (TileLang kernel).
- diagram idea: frozen attention map (teacher) → pooled block map → gate prediction; side-by-side masks.
- tips: a middle ground between training-free (MInference) and native training (NSA/DSA): only gate parameters are trained. Reasoning decode (long outputs) is where decode sparsity pays off.
- verification: abstracts.

### Sparse Transformer → GPT-3 alternation (SHOULD; 2019, OpenAI)
- url: https://arxiv.org/abs/1904.10509
- one-line idea: factorize full attention into two sparse hops (local band + strided), so any pair connects in 2 layers; O(n√n).
- key formula: strided $A^{(1)}_t=\{t-\ell..t\}$, $A^{(2)}_t=\{j:(t-j)\bmod\ell=0\}$, $\ell\approx\sqrt n$
- concrete numbers: sequences of tens of thousands; hundreds of layers. GPT-3 alternated dense and locally banded layers; gpt-oss: "Following GPT-3 … alternate between banded window and fully dense", band 128.
- diagram idea: two 16 × 16 masks (band, stride 4) and their union, plus the two-hop path from query 15 to key 2.
- tips: a direct line from 2019 to today's SWA:global hybrids. Static patterns rely on depth to route information.
- verification: abstract; gpt-oss §2.2 quote.

### Compressive Transformer (SHOULD; 2019, ICLR 2020, DeepMind)
- url: https://arxiv.org/abs/1911.05507
- one-line idea: memories leaving the Transformer-XL FIFO are compressed (rate c) into a second memory instead of being discarded. This is the 2019 ancestor of CSA/HCA's "fewer records".
- key formula: $cm\leftarrow f_c(m_{\text{oldest}\,n_s})\in\mathbb R^{\lfloor n_s/c\rfloor\times d}$; range $=l\times(n_m+c\,n_{cm})$; cost $O(n_s^2+n_s(n_m+n_{cm}))$
- concrete numbers: f_c ∈ {max/mean pool, 1D conv, dilated conv, most-used}, plus an attention-reconstruction auxiliary loss. Enwik8: c = 3, memory 768, compressed 1152 (eval 3072) → 0.97 bpc. WikiText-103: conv, c = 4 → 17.1 ppl. PG-19: c = 2, 512/512, 36 layers, 33.6 vs TXL 36.3. Introduced PG-19.
- diagram idea: three bands (current window | FIFO memory | compressed memory) with arrows showing 3 old slots → 1 compressed slot, then the oldest compressed slot falling off.
- tips: like HCA, a dense read over pooled records; unlike CSA, no selection. The "most-used" variant is H2O-like eviction.
- verification: PDF text (pdftotext) §3, §5.1–5.3.

### Gist tokens + Activation Beacon (SHOULD; NeurIPS 2023 / ICLR 2025)
- url: https://arxiv.org/abs/2304.08467 ; https://arxiv.org/abs/2401.03462
- one-line idea: train special tokens whose KV absorbs a span (a prompt for gist; each chunk progressively for Beacon), then drop the span's own KV.
- key formula: gist mask: tokens after the gist $G$ cannot attend to the prompt $P$: $M_{ij}=0$ for $i>G,\ j\in P$. Beacon: per chunk of w tokens keep only $w/\alpha$ beacon KVs, with $\alpha$ sampled per step.
- concrete numbers: gist: up to 26× prompt compression, 40% FLOPs, 4.2% wall-clock (LLaMA-7B, FLAN-T5-XXL). Beacon: 2× inference speed, 8× KV memory reduction; trained ≤20K, evaluated to 128K; random compression ratio per training step.
- diagram idea: a mask matrix with the gist column block; for Beacon, a chunk strip where 1,024 tokens' KV collapse to 128 beacon slots (α = 8).
- tips: learned records, but intra-chunk only. Unlike CSA they are not trained into pretraining at scale. An exact ID like "7314" is the classic loss case.
- verification: both abstracts; Beacon ICLR 2025 via proceedings search hit.

### FP8 / FP4 KV cache in production (2024–26)
- url: https://vllm.ai/blog/2025-09-29-deepseek-v3-2 ; https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/ ; https://github.com/deepseek-ai/FlashMLA
- one-line idea: store KV elements in 8 or 4 bits with small block scales. Dequantize in-kernel before the matmul.
- key formula: $\hat x=s_b\cdot\mathrm{fp8}(x/s_b)$, $s_b=\max_{x\in b}|x|/448$ (e4m3 max; block = 128 values in V3.2's format). Bytes per token = elements × bits / 8 + scales + unquantized parts.
- concrete numbers: V3.2 FP8 MLA cache **656 B/token/layer** = 512 FP8 + 16 B (4 fp32 scales, one per 128) + 128 B (64 BF16 RoPE) vs 1,152 B in BF16 (derived: 576 × 2). V4: FP8 except BF16 RoPE. V4.1 (FlashMLA): 528 B FP8 / 288 B FP4 per token. NVFP4 KV ≈ 50% of FP8, <1% accuracy loss, dequantized to FP8 for the matmul (an independent SGLang test saw 1.68–1.69× capacity, not 2×).
- diagram idea: a byte-layout ruler for one token: BF16 1,152 B → FP8 656 B (colour segments: NoPE FP8 / scales / RoPE BF16) → V4.1 FP4 288 B.
- tips: RoPE dims stay high precision (position error hurts). Savings are memory and bandwidth, not FLOPs. Scales and alignment make real ratios <2×/4×.
- verification: vLLM blog layout; NVIDIA blog via search summary; FlashMLA README (528/288). The V4.1 byte breakdown is UNVERIFIED.

### KVQuant (SHOULD; 2024, NeurIPS 2024, Berkeley)
- url: https://arxiv.org/abs/2401.18079
- one-line idea: keys have per-channel outliers, and RoPE mixes channels. So quantize keys per channel **before RoPE**, use non-uniform levels, and keep outliers sparse in high precision.
- key formula: $\hat k=Q_{\text{nuq}}\big(k_{\text{pre-RoPE}}\big)$ per channel; $k=\hat k+k_{\text{outlier (sparse)}}$; RoPE applied after dequantization.
- concrete numbers: 3-bit with <0.1 perplexity loss (Wikitext-2, C4); LLaMA-7B at 1M ctx on 1×A100-80GB, 10M on 8 GPUs; about 1.7× matvec speedup.
- diagram idea: a channel × token heatmap of K pre/post RoPE (outlier channels as stripes that RoPE smears), then a 3-bit grid.
- tips: KIVI (per-channel K, per-token V) is the simplest form. KVQuant is the reason RoPE dims are treated separately.
- verification: abstract.

---

## 5. Suggested restructure

- **06 Sparse retrieval (reads shrink).** A: static patterns: Sparse Transformer → Longformer/BigBird → SWA:global in production (gpt-oss, Gemma 3, Llama 4, MiMo). B: training-free selection: MInference (+Qwen2.5-1M) → XAttention/FlexPrefill → Quest → offload trio. C: trained selection: MoBA → NSA → InfLLM-v2 → DSA → MSA/QSA → IndexCache (the indexer-cost problem).
- **NEW 06b "KV-cache eviction at inference (cache shrinks)":** 1) attention sinks/StreamingLLM + learned sink; 2) H2O; 3) SnapKV; 4) PyramidKV/Ada-KV budgets; 5) KVzip (query-agnostic); 6) DuoAttention/RazorAttention (head split); 7) DMC→DMS (learned); 8) evaluation-pitfalls card (irreversibility, instruction loss: ACL 2026).
- **07 Sequence compression (records merge):** Compressive Transformer → gist/Beacon → NSA cmp branch → CSA/HCA (with top-k, sinks, FP4 indexer) → V4.1 CSA2 pointer to 09.
- **09:** lead the bytes section with the FP8 default (V3.2 656 B layout) → KIVI/KVQuant → NVFP4/FP4 (V4.1 288 B).
- **Opening figure for 06/06b/07:** one 2×2 "what shrinks" matrix (reads vs records × reversible vs not) placing every card. Use the four-family table in §3.
