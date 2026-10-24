# D · Hybrid layouts and the systems layer (research notes)

Researcher D · checked 2026-10-21 · for card writers. Scope: (1) hybrid stacks and the attention layouts of production models; (2) kernels, KV-cache serving and distributed long-context attention.

**Method.** Every layout cell was read from the model's `config.json` (fetched raw from Hugging Face; gated repos via `unsloth/*` mirrors with identical weights) and cross-checked against the paper or model card. "KV B/tok" values are **computed** by me from config: growing-cache layers only, BF16, `L_full × 2 × H_kv × d_h × 2 B` (MLA: `L × (d_c+d_r) × 2 B`). Metadata, scales and local-window caches are excluded unless stated. Label these "computed from config" in figures. "Reported" means an author claim that we did not reproduce.

---

## 1. Coverage audit

**What the book has now** (legacy `planning/legacy/chapters/00, 08, 09`, `appendices/mechanism-matrix.md`, `bibliography/papers.json`; new `chapters/03-kv-heads.md`):

- 08: Qwen3-Next (GDN 3:1 gated attention) and Kimi Linear (KDA 3:1 MLA); output-gate vs state-gate distinction; hybrid storage formula; Qwen3.8-Next/QSA paragraph.
- 09: storage factor model $M=BNCH_{kv}(d_k+d_v)s$; FlashAttention 1–4 (mechanism only, no numbers); FLA library; Ring Attention; CLA; KIVI; DeepSeek-V4.1-Flash 890 B/token; PagedAttention (abstract-only); prefix caches (one paragraph).
- 00/matrix: one "Combined memory" row (Qwen3-Next, Kimi Linear); one "Better realization" row.
- New 03: decode intensity ≈ G, H100 ridge ≈ 295 FLOP/B. A cross-layer-sharing card is planned (CLA, YOCO, Gemma 3n).

**Errors and oversimplifications** (correction · source):

1. **Kimi Linear is not a strict 3:1 repeat in the released checkpoint.** The 48B-A3B config has 27 layers: 20 KDA and 7 MLA, with MLA at layers 4, 8, 12, 16, 20, 24 and 27 (1-indexed). The last group is therefore 2 KDA + 1 MLA. The MLA layers are **NoPE** (`mla_use_nope: true`). The paper says KDA carries the position signal and NoPE lets MLA convert to MQA at inference. · [config](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/config.json), [paper](https://arxiv.org/html/2510.26692v1)
2. **"Hybrid" means only GDN/KDA + softmax in the book.** The most widely deployed hybrid is **local/global attention**: Gemma 2/3/4, gpt-oss, Command A, MiMo-V2-Flash, Ministral-8B, and Llama 4's chunked layers. The second family is **Mamba-2 + attention**: Jamba, Nemotron-H, Nemotron Nano 2, Nemotron 3, Granite 4.0-H and Falcon-H1. Both families are missing. · table §2
3. **No counter-evidence.** MiniMax went from a 7:1 lightning hybrid (Text-01/M1) to all-full GQA (M2, Oct 2025), citing evaluation blind spots in multi-hop reasoning, immature infrastructure, low-precision state, and problems with prefix caching and speculative decoding. It then moved to block-sparse MSA (M3, 2026). This lineage belongs in the hybrid chapter. · [MiniMax blog](https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model), [M3 card](https://huggingface.co/MiniMaxAI/MiniMax-M3)
4. **Qwen3.8 naming.** The report title says "Qwen3.8-Next Architecture". The released checkpoint and the abstract say **Qwen3.8-Flash-Next** (125B total, 6B active). Config: 48 layers, 3 GDN : 1 full; full attention is 24 Q / 2 KV / d_h 256 with a sigmoid output gate. The QSA indexer is MQA with 4 heads and 1 K head of 128 dims, compression r=4 and a budget of 2048 tokens. The report also **rejected NoPE**, which caused "endless generation" after post-training. That is a useful contrast with Kimi's NoPE. · [arXiv:2608.30320](https://arxiv.org/abs/2608.30320), [config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json)
5. **Hybrid storage formula: state precision.** Recurrent states are usually kept in **fp32** (`mamba_ssm_dtype: float32` in Qwen3.5 and Qwen3.8; `mamba_ssm_cache_dtype: float32` in Nemotron 3), while KV is BF16. So $s_\ell$ differs by 2× between the two sums. Small conv states (kernel 4) are also omitted. The formula is right, but the example should use s=4 for state. · configs
6. **Hybrids have serving costs the book omits.** Prefix caching, speculative decoding and prefill/decode transfer need per-request **state checkpoints**, because states are updated in place and cannot be rolled back. · [PyTorch/SGLang blog, 2025-12](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)
7. **09 covers prefill/training kernels only.** Decode kernels (Flash-Decoding split-KV, FlashMLA, FlashInfer) are absent, yet decode is where KV bytes set latency. · [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
8. **The FlashAttention table has no numbers.** Verified figures: FA1 reaches 25–40% of peak on A100; FA2 50–73% on A100 but only 35% on H100; FA3 740 TFLOP/s FP16 (75%) and ≈1.2 PFLOP/s FP8; FA4 1613 TFLOP/s BF16 on B200 (71%), 1.3× cuDNN 9.13, 2.7× Triton. · [FA2](https://arxiv.org/abs/2307.08691), [FA3](https://arxiv.org/abs/2407.08608), [FA4](https://arxiv.org/abs/2603.05451)
9. **Ring Attention lacks its overlap condition and its alternatives.** Communication is hidden only if block size $c\ge F/B$: about 1.0K tokens on A100 NVLink and 24.5K on A100 InfiniBand. Missing alternatives: Ulysses all-to-all, Llama 3's all-gather CP with 2×CP load-balanced chunks, and Striped Attention. · [Ring Table 2](https://arxiv.org/html/2310.01889), [Ulysses](https://arxiv.org/html/2309.14509), [Llama 3 §3.3.2](https://arxiv.org/html/2407.21783)
10. **PagedAttention is abstract-only.** Add: prior systems use only 20.4–38.2% of KV memory for token states; default block size is 16; sharing saves 6.1–9.8% (parallel sampling) and 37.6–55.2% (beam search); 2–4× throughput (SOSP 2023). · [arXiv html](https://arxiv.org/html/2309.06180)
11. **The prefix-cache paragraph has no mechanism and no production number.** Add: radix tree, LRU leaf eviction, longest-shared-prefix-first scheduling (= DFS), and DeepSeek's production figure of 56.3% of 608B input tokens hitting its **on-disk** KV cache over 24 h. · [SGLang](https://arxiv.org/html/2312.07104v2), [DeepSeek day-6](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)
12. **V4.1-Flash: verified, but missing context.** 890 B/token is about ¼ of V4-Flash, and "approximate" replay is correct ("By design, the replayed prefix state is approximate"). Not mentioned: it is a **Causal Encoder-Decoder** (20+20 layers) whose decoder global KV is projected from the final encoder states. CSA2 modes: 4 Full, 4 Reindex, 30 Reuse (plus 2 SWA-only encoder layers). FP4 is E2M1 with one E4M3 scale per 16 channels; SWA KV stays FP8. Global KV persists ≥72 h; SWA KV lives in a host-DRAM pool with a TTL of minutes. · [arXiv:2609.19969](https://arxiv.org/html/2609.19969), [card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
13. **The KV-quant section stops at KIVI.** Production FP8 KV: DeepSeek-V3.2's FlashMLA format is **656 B per token per layer** (512 FP8 NoPE values + 4 fp32 scales + 64 BF16 RoPE values). · [FlashMLA README @2025-10](https://github.com/deepseek-ai/FlashMLA)
14. **Cross-layer reuse is CLA-only.** Production: Gemma 3n E4B shares KV in 15 of 35 layers (`num_kv_shared_layers: 15`); Hymba shares KV between consecutive layer pairs (`kv_reuse_group`); Gemma 4 global layers use K=V ("unified Keys and Values", `attention_k_eq_v`); V4.1 CSA2 Reuse mode. · configs, [Gemma 4 card](https://huggingface.co/google/gemma-4-31B-it)
15. **The glossary's "attention sink" covers only the emergent first-token phenomenon.** gpt-oss and MiMo-V2-Flash use a **learned per-head sink logit in the softmax denominator**, with no token involved. Disambiguate. · [gpt-oss card](https://arxiv.org/html/2508.10925), [MiMo §2.2](https://arxiv.org/html/2601.02780v1)
16. **MiniMax-01 RoPE base.** The paper says RoPE on half the head dim, base 10,000. The released Text-01/M1 config has `rotary_dim 64` and `rope_theta 1e7`. Do not quote the base without saying which. · [paper](https://arxiv.org/html/2501.08313), [config](https://huggingface.co/MiniMaxAI/MiniMax-Text-01/blob/main/config.json)
17. **DeepSeek-V4 arXiv ID.** The abs page for 2606.19348 shows v1 dated 26 Apr 2026, which is inconsistent with a 2606 prefix. Cite it, but do not derive a date from the ID. · [abs](https://arxiv.org/abs/2606.19348)

---

## 2. Frontier layouts table (2024–2026)

Notation: GA = global/full attention, SWA = sliding window, L = layers, idx are 0-indexed unless noted. KV B/tok is computed from config (growing layers only, BF16). Dates are public release months; † means a Hugging Face repo-creation month used as a proxy (UNVERIFIED release day).

| Model (date) | L | Layer pattern | Full-attn type (H_q/H_kv/d_h) | Local / linear type + params | Position | KV B/tok | Source |
|---|---|---|---|---|---|---|---|
| Llama 3.1-405B (2024-07), dense baseline | 126 | all GA | GQA 128/8/128 | — | RoPE | 516,096 | [config mirror](https://huggingface.co/unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit/blob/main/config.json) |
| Jamba v0.1 (2024-03) | 32 | 4 blocks × [1 attn + 7 Mamba]; attn idx 4,12,20,28; MoE every 2nd | GQA 32/8/128 | Mamba-1, d_state 16, expand 2, conv 4 | **none** (no explicit position) | 16,384 (paper: 4 GB @256K) | [paper](https://arxiv.org/html/2403.19887v2), [config](https://huggingface.co/ai21labs/Jamba-v0.1/blob/main/config.json) |
| Gemma 2 27B (2024-06) | 46 | 1 SWA : 1 GA alternating (23+23) | GQA 32/16/128, logit soft-cap 50 | SWA W=4096 | RoPE θ 1e4 | 188,416 | [config mirror](https://huggingface.co/unsloth/gemma-2-27b/blob/main/config.json) |
| Ministral-8B-2410 (2024-10) | 36 | [1 GA + 3 SWA] × 9 | GQA 32/8/128 | SWA W=32,768 | RoPE θ 1e8 | 36,864 | [config](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410/blob/main/config.json) |
| Zamba2-7B (2024-10) | 81 | 68 Mamba-2 + 13 invocations of 2 **shared** attn+MLP blocks (ABAB), "1:6" | MHA 32/32, `attention_head_dim` 224 (input = concat of hidden and embedding) | Mamba-2: 112 heads × 64, d_state 64 | RoPE | 372,736 (13 invocations; UNVERIFIED reading of d_h) | [paper](https://arxiv.org/html/2411.15242), [config](https://huggingface.co/Zyphra/Zamba2-7B-Instruct/blob/main/config.json) |
| Hymba-1.5B (2024-11) | 32 | **parallel** attn ∥ Mamba heads in every layer; GA at idx 0,15,31, SWA elsewhere; KV shared by consecutive pairs | GQA 25/5 (d_h UNVERIFIED) | SWA W=1024; Mamba d_state 16; 128 meta tokens | RoPE | paper: 79 MB @8K FP16 vs 918 MB for Llama-3.2-3B | [paper](https://arxiv.org/html/2411.13676v1), [config](https://huggingface.co/nvidia/Hymba-1.5B-Base/blob/main/config.json) |
| MiniMax-Text-01 (2025-01) / M1 (2025-06) | 80 | 10 × [7 lightning + 1 softmax]; softmax idx 7,15,…,79 | GQA 64/8/128 | lightning (linear, no decay): 64 heads, state 128×128 per head; block 256 | partial RoPE 64/128 (θ: paper 1e4, config 1e7) | 40,960 | [paper](https://arxiv.org/html/2501.08313), [config](https://huggingface.co/MiniMaxAI/MiniMax-Text-01/blob/main/config.json) |
| Gemma 3 27B (2025-03) | 62 | 5 SWA : 1 GA; GA idx 5,11,…,59 (52+10) | GQA 32/16/128, QK-norm | SWA W=1024 | RoPE θ 1e4 local / 1e6 global | 81,920 | [report](https://arxiv.org/html/2503.19786), [config mirror](https://huggingface.co/unsloth/gemma-3-27b-it/blob/main/config.json) |
| Command A (2025-03) | 64 | [3 SWA + 1 GA] × 16 | GQA 96/8/128, parallel block | SWA W=4096 | RoPE in SWA; **NoPE in GA** | 65,536 | [report PDF](https://arxiv.org/pdf/2504.00698), [config mirror](https://huggingface.co/unsloth/c4ai-command-a-03-2025/blob/main/config.json) |
| Llama 4 Scout/Maverick (2025-04) | 48 | [3 chunked + 1 GA] × 12; GA idx 3,7,…,47 | GQA 40/8/128 | chunked (block-diagonal) attention, chunk 8192 | iRoPE: RoPE in chunked, **NoPE in GA** + query scale $1+0.1\ln(1+\lfloor (t+1)/8192\rfloor)$ | 49,152 | [blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/), [config mirror](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct/blob/main/config.json), [HF code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/modeling_llama4.py) |
| Nemotron-H-56B (2025-04) | 118 | 54 Mamba-2 + 54 FFN + 10 attn (≈8%); attn idx 7,18,…,106; first layer Mamba | GQA 64/8/128 | Mamba-2: 256 heads × 64, d_state 256, 8 groups | **none** | 40,960 | [paper](https://arxiv.org/html/2504.03624), [config](https://huggingface.co/nvidia/Nemotron-H-56B-Base-8K/blob/main/config.json) |
| Falcon-H1-34B (2025-05†) | 72 | **parallel** attn ∥ Mamba-2 in every layer; channels SSM:attn:MLP ≈ 2:1:5 | GQA 20/4/128 | Mamba-2: 32 heads × 128, d_state 256 | RoPE θ **1e11** | 147,456 (all 72 layers keep KV) | [report](https://arxiv.org/html/2507.22448), [config](https://huggingface.co/tiiuae/Falcon-H1-34B-Instruct/blob/main/config.json) |
| Kimi K2 (2025-07) | 61 | all MLA | MLA 64 heads (V3 has 128), d_c 512, d_r 64 | — | decoupled RoPE, YaRN ×32 | 70,272 | [paper](https://arxiv.org/html/2507.20534), [config](https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/config.json) |
| GLM-4.5 / 4.6 / 4.7 (2025-07/09/12) | 92 | all GA | GQA **96**/8/128, QK-norm | — | partial RoPE (factor 0.5), θ 1e6 | 376,832 | [paper](https://arxiv.org/html/2508.06471), [config](https://huggingface.co/zai-org/GLM-4.6/blob/main/config.json) |
| gpt-oss-120b / 20b (2025-08) | 36 / 24 | SWA, GA alternating 1:1 (idx 0 = SWA) | GQA 64/8/**64**; learned per-head sink logit | banded W=**128** | RoPE + YaRN ×32 → 131,072 | 36,864 / 24,576 | [card](https://arxiv.org/html/2508.10925), [config](https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json) |
| Jet-Nemotron-2B (2025-08) | 28 | 24 JetBlock + GA at L15, L20 + SWA at L21, L22 (1-indexed) | GQA 12/2/128 | JetBlock = GDN + dynamic conv on V: 12 heads, d_k 96; SWA W=1152 | RoPE θ 1e6 | 2,048 (paper: 154 MB @64K vs 7,168 MB Qwen3-1.7B) | [paper](https://arxiv.org/html/2508.15884), [config](https://huggingface.co/jet-ai/Jet-Nemotron-2B/blob/main/config.json) |
| Nemotron Nano 9B v2 (2025-08) | 56 | 27 Mamba-2 + 25 FFN + 4 attn (idx 14,21,30,39) | GQA 40/8/128 | Mamba-2: 128 heads × 80, d_state 128 | **none** | 16,384 | [paper](https://arxiv.org/html/2508.14444), [config](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2/blob/main/config.json) |
| Qwen3-Next-80B-A3B (2025-09) | 48 | 12 × [3 GDN + 1 gated GA] | GQA 16/2/256 + sigmoid output gate | GDN: 16 QK heads, 32 V heads, d_k=d_v=128, conv 4 | partial RoPE 64/256, θ 1e7 | 24,576 | [card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct), [config](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json) |
| DeepSeek-V3.2 (Exp 2025-09; final 2025-12) | 61 | all MLA + DSA indexer | MLA 128 heads, d_c 512, d_r 64 | indexer 64 heads × 128, top-k 2048 | decoupled RoPE, YaRN ×40 | 70,272 BF16; FP8 format 656 B/layer → 40,016 | [config](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json), [FlashMLA](https://github.com/deepseek-ai/FlashMLA) |
| Granite 4.0-H-Small (2025-10) | 40 | 36 Mamba-2 + 4 GA (idx 5,15,25,35), "9:1" | GQA 32/8/128 | Mamba-2: 128 heads × 64, d_state 128 | **NoPE** | 16,384 | [IBM](https://www.ibm.com/new/announcements/ibm-granite-4-0-hyper-efficient-high-performance-hybrid-models), [config](https://huggingface.co/ibm-granite/granite-4.0-h-small/blob/main/config.json) |
| MiniMax-M2 (2025-10) | 62 | **all GA** (`attn_type_list` all 1) | GQA 48/8/128, per-layer QK-norm | — | partial RoPE 64/128, θ 5e6 | 253,952 | [blog](https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model), [config](https://huggingface.co/MiniMaxAI/MiniMax-M2/blob/main/config.json) |
| Kimi Linear 48B-A3B (2025-10) | 27 | 20 KDA + 7 MLA; MLA at L4,8,…,24,27 (1-indexed) | MLA 32 heads, d_c 512, qk_nope 128, v 128 | KDA: 32 heads × 128, conv 4 | **NoPE in MLA** | 8,064 | [paper](https://arxiv.org/html/2510.26692v1), [config](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/config.json) |
| Nemotron 3 Nano 30B-A3B (2025-12) | 52 | 23 Mamba-2 + 23 MoE + 6 GA (idx 5,12,19,26,33,42) | GQA 32/**2**/128 | Mamba-2: 64 heads × 64, d_state 128 | **none** | 6,144 | [paper](https://arxiv.org/html/2512.20848), [config](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/config.json) |
| MiMo-V2-Flash (2025-12) | 48 | GA (dense FFN), 4 SWA, GA, then 7 × [5 SWA + 1 GA] → 39 SWA + 9 GA ("5:1") | GA: GQA 64/4, d_qk 192, d_v 128 | SWA W=**128**: GQA 64/8, learned sink bias | partial RoPE 64/192; θ 5e6 GA / 1e4 SWA | 23,040 | [report](https://arxiv.org/html/2601.02780v1), [config](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash/blob/main/config.json) (V2.5, 2026-04: same pattern) |
| Ministral 3 14B (2025-12) | 40 | all GA (no SWA, unlike Ministral-8B) | GQA 32/8/128 | — | YaRN ×16, θ 1e9, Llama-4-style β 0.1 | 163,840 | [config](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512/blob/main/config.json) |
| Mistral Large 3 (2025-12) | 61 | all MLA | MLA 128 heads, d_c 512, d_r 64 | — | YaRN ×36 | 70,272 | [params.json](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512-BF16/blob/main/params.json) |
| GLM-5 / 5.1 (2026-02 / 2026-04) | 78 | all MLA + DSA | MLA 64 heads, d_c 512, qk 192+64, v 256 | indexer 32 heads × 128, top-k 2048 | RoPE 64 dims, θ 1e6 | 89,856 | [card](https://huggingface.co/zai-org/GLM-5), [config](https://huggingface.co/zai-org/GLM-5/blob/main/config.json) |
| Qwen3.5-397B-A17B (2026-02†) | 60 | 15 × [3 GDN + 1 gated GA] | GQA 32/2/256 + gate | GDN: 16 QK heads, 64 V heads × 128 | partial mRoPE 64/256, θ 1e7 | 30,720 | [card](https://huggingface.co/Qwen/Qwen3.5-397B-A17B), [config](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json) |
| Nemotron 3 Super 120B-A12B (2026-03†) | 88 | 40 Mamba-2 + 40 LatentMoE + 8 GA (idx 7,16,25,36,47,58,69,78) | GQA 32/2/128 | Mamba-2: 128 heads × 64, d_state 128 | UNVERIFIED (Nano report: none) | 8,192 | [config](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16/blob/main/config.json) |
| Gemma 4 31B (2026-04) | 60 | 5 SWA : 1 GA, last layer GA (50+10) | local GQA 32/16/256; global 32/**4**/**512**, **K=V** | SWA W=1024 | local RoPE θ 1e4; global p-RoPE on 25% of dims, θ 1e6 | 40,960 (K=V) | [card](https://huggingface.co/google/gemma-4-31B-it), [config mirror](https://huggingface.co/unsloth/gemma-4-31B-it/blob/main/config.json) |
| DeepSeek-V4-Pro / V4-Flash (2026-04) | 61 / 43 | Pro: 2 HCA, then CSA(r=4)/HCA(r=128) alternating (30 CSA + 31 HCA). Flash: 2 SWA-only, then alternating (21 CSA + 20 HCA). SWA W=128 in all layers | MQA: 1 KV head, d 512 (K=V entry), 128 / 64 Q heads | indexer 64 heads × 128, top-k 1024 / 512 | partial RoPE 64 dims, YaRN ×16 | n/a; reported: 10% of V3.2 KV at 1M | [abs](https://arxiv.org/abs/2606.19348), [config](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json) |
| MiniMax-M3 (2026-06†) | 60 | 3 dense GA + 57 **MSA** block-sparse | GQA 64/4/128, per-head QK-norm | MSA: block 128, top-16 blocks (2048 tokens), 4 index heads × 128, 1 local block | partial RoPE 64/128, θ 5e6 | 122,880 (all retained; reads bounded); index-key cache UNVERIFIED | [card](https://huggingface.co/MiniMaxAI/MiniMax-M3), [config](https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/config.json) |
| Kimi K3 (2026-07) | 93 | 69 KDA + 24 **gated** MLA; MLA at L4,8,…,92 and L93 (1-indexed) | MLA 96 heads, d_c 512, output gate | KDA: 96 heads × 128, full-rank gate | **NoPE in MLA** | 27,648 | [card](https://huggingface.co/moonshotai/Kimi-K3), [config](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json) |
| Qwen3.8-Flash-Next (2026-08) | 48 | 12 × [3 GDN + 1 GA→QSA after continued pretraining] | GQA 24/2/256 + sigmoid gate | GDN: 16 QK heads, 48 V heads × 128. QSA indexer: MQA 4 heads/1 K × 128, r=4, budget 2048 | partial mRoPE 64/256 (NoPE rejected) | 24,576 | [report](https://arxiv.org/html/2608.30320), [config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json) |
| DeepSeek-V4.1-Flash (2026-09) | 40 (20 enc + 20 dec) | enc: 2 SWA-only + 3×[1 Full + 5 Reuse]; dec: [1 Full + 3 Reuse] + 4×[1 Reindex + 3 Reuse] | MQA: 64 Q / 1 KV, d 512 | SWA W=128 (FP8); main KV FP4 (E2M1, E4M3 scale/16 ch) | partial RoPE 64, YaRN ×16 | **890 B/tok global (reported)** | [paper](https://arxiv.org/html/2609.19969), [card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |

**Patterns worth a figure** (all from the table):

- Global-attention fraction clusters at **1/2** (Gemma 2, gpt-oss), **1/4** (Qwen, Kimi, Command A, Llama 4, Ministral-8B), **1/6** (Gemma 3/4, MiMo) and **≈1/8–1/10** (Jamba, Nemotron, Granite, MiniMax-01).
- **NoPE in global layers**: Jamba, Nemotron-H/Nano 2/3, Granite 4.0-H, Command A, Llama 4, Kimi Linear, K3. Counterexample: Qwen3.8 rejected NoPE.
- **The KV head count keeps falling**: H_kv=2 in Qwen3.x and Nemotron 3; Gemma 4 global uses H_kv=4 with K=V; DeepSeek V4 uses 1.
- **MLA spreads**: Kimi K2/K3, GLM-5, Mistral Large 3. **DSA-style indexers spread**: V3.2, GLM-5, V4, QSA, MSA.
- **Full attention is still the choice of several 2025 flagships**: MiniMax-M2, GLM-4.x, Ministral 3.

Computed state sizes per sequence (fp32, recurrent layers only): Qwen3-Next 72 MiB · Qwen3.5-397B 180 MiB · Qwen3.8-Flash-Next 108 MiB · Kimi Linear 40 MiB · Kimi K3 414 MiB · MiniMax-01 280 MiB · Nemotron-H-56B 864 MiB · Nemotron 3 Nano 46 MiB · Granite-H-Small 144 MiB.

Computed bounded local caches per sequence (BF16): gpt-oss-120b 4.5 MiB · MiMo-V2-Flash 24.4 MiB · Gemma 3 27B 416 MiB · Command A 768 MiB · Llama 4 1.1 GiB · Ministral-8B 3.4 GiB.

---

## 3. Gap table

| Rank | Item | Year | Venue / org | URL | What changes (≤15 words) | Adopted in | Suggested chapter |
|---|---|---|---|---|---|---|---|
| MUST | Frontier layout strips | 2024–26 | configs | §2 | One figure: layer stacks coloured by type, with KV B/tok per model | — | NEW: Production layouts 2024–2026 |
| MUST | Local/global SWA interleave + learned sink | 2024–26 | Google, OpenAI, Xiaomi, Cohere, Meta | [Gemma 3](https://arxiv.org/html/2503.19786) | Most layers keep only W tokens; few global layers keep everything | Gemma 2/3/4, gpt-oss, MiMo-V2, Command A, Llama 4 | NEW: Production layouts (or 08) |
| MUST | Mamba-2 sequential hybrids (~8–10% attention) | 2024–26 | AI21, NVIDIA, IBM | [Nemotron-H](https://arxiv.org/abs/2504.03624) | Replace ~90% of attention layers with Mamba-2; KV only in a few | Jamba, Nemotron-H/Nano 2/3, Granite 4.0-H | 08 |
| MUST | GDN/KDA 3:1 hybrids, 2026 update | 2025–26 | Qwen, Moonshot | [Qwen3.8](https://arxiv.org/abs/2608.30320) | 3 linear : 1 global; global becomes sparse (QSA) in Qwen3.8 | Qwen3-Next/3.5/3.6/3.8, Kimi Linear, Kimi K3 | 08 (update existing) |
| MUST | NoPE global + positional local | 2024–26 | many | [Command A](https://arxiv.org/pdf/2504.00698) | Global layers drop RoPE; position comes from local or recurrent layers | Jamba, Nemotron, Granite, Command A, Llama 4, Kimi Linear/K3 | 08 |
| MUST | MiniMax: hybrid → full → sparse | 2025–26 | MiniMax | [blog](https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model) | Production reversal: 7:1 lightning abandoned, then block-sparse MSA | MiniMax-01/M1, M2, M3 | 08 |
| MUST | FlashAttention 1→4 numbers | 2022–26 | Dao et al. | [FA4](https://arxiv.org/abs/2603.05451) | Same exact attention, re-tiled per GPU generation | all frameworks | 09 |
| MUST | Flash-Decoding (split-KV) | 2023 | Stanford CRFM / Meta | [blog](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) | Parallelise decode over KV chunks, merge with log-sum-exp | FA2+ split-KV, FlashInfer, FlashMLA `num_splits` | 09 |
| MUST | FlashMLA (+ FP8/sparse KV) | 2025 | DeepSeek | [GitHub](https://github.com/deepseek-ai/FlashMLA) | MLA decode as 128-head MQA on one latent; FP8 656 B/token/layer | DeepSeek V3/V3.2/V4/V4.1 | 09 (link to MLA chapter) |
| MUST | PagedAttention + RadixAttention prefix cache | 2023–24 | vLLM (SOSP'23), SGLang | [SGLang](https://arxiv.org/abs/2312.07104) | Block tables kill fragmentation; radix tree reuses shared prefixes | vLLM, SGLang, TRT-LLM | NEW: Serving the KV cache |
| MUST | Prefill/decode disaggregation | 2024–25 | DistServe (OSDI'24), Splitwise (ISCA'24), Mooncake (FAST'25) | [DistServe](https://arxiv.org/abs/2401.09670) | Separate GPU pools for prefill and decode; ship KV between them | Kimi (Mooncake), DeepSeek prod | NEW: Serving the KV cache |
| MUST | Context parallelism: ring vs all-to-all vs all-gather | 2023–24 | Berkeley, Microsoft, Meta | [Ulysses](https://arxiv.org/abs/2309.14509) | Split the sequence over GPUs; choose which tensor moves | Llama 3 training (all-gather CP), DeepSpeed, Megatron | 09 |
| SHOULD | Parallel (intra-layer) hybrids | 2024–25 | NVIDIA, TII, Meta study | [Hymba](https://arxiv.org/abs/2411.13676) | Attention and SSM heads side by side in each layer | Hymba, Falcon-H1 | 08 |
| SHOULD | Choosing the ratio (evidence card) | 2025 | Moonshot, academic, Meta | [2507.06457](https://arxiv.org/abs/2507.06457) | Quality/recall vs linear:full ratio, window size and placement | — | 08 |
| SHOULD | Jet-Nemotron PostNAS | 2025 | NVIDIA | [arXiv](https://arxiv.org/abs/2508.15884) | Retrofit a pretrained model: search where to keep full attention | Jet-Nemotron 2B/4B | 08 |
| SHOULD | Hybrid serving: state caches | 2025 | SGLang, vLLM | [blog](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/) | Per-request state slots and checkpoints for prefix cache and speculative decoding | SGLang, vLLM | NEW: Serving the KV cache |
| SHOULD | Chunked prefill (Sarathi-Serve) | 2024 | OSDI'24 | [arXiv](https://arxiv.org/abs/2403.02310) | Split prompts into chunks co-batched with decodes; no stalls | vLLM, SGLang default | NEW: Serving the KV cache |
| SHOULD | Tiered KV offload (LMCache, Mooncake store, disk cache) | 2024–25 | LMCache, Moonshot, DeepSeek | [LMCache](https://arxiv.org/abs/2510.09665) | KV moved to CPU/SSD/network and reused across engines and requests | vLLM, SGLang, DeepSeek API | NEW: Serving the KV cache |
| SHOULD | FlashInfer | 2025 | MLSys'25 | [arXiv](https://arxiv.org/abs/2501.01005) | One block-sparse KV format and JIT templates for all serving attention | SGLang, vLLM, MLC | 09 |
| SHOULD | Low-precision attention (FA3-FP8, SageAttention 1/2/3) | 2024–25 | Tri Dao; Tsinghua (ICLR'25, ICML'25) | [Sage3](https://arxiv.org/abs/2505.11594) | INT8/INT4/FP8/FP4 inside QKᵀ and PV with smoothing and scales | video/diffusion inference; FA3 FP8 | 09 |
| SHOULD | Cross-layer KV reuse in production | 2024–26 | Google, NVIDIA, DeepSeek | [Gemma 4 card](https://huggingface.co/google/gemma-4-31B-it) | Later layers read earlier layers' KV; or K=V | Gemma 3n, Hymba, Gemma 4 (K=V), V4.1 CSA2 | 03 (planned card) / 09 |
| SHOULD | FlexAttention | 2024–25 | PyTorch (MLSys'25) | [blog](https://pytorch.org/blog/flexattention/) | `score_mod`/`mask_mod` compiled to a fused, block-sparse kernel | Hymba config (`attn_implementation: flex`), torchtune | 09 |
| SKIP | Flash-Decoding++ | 2023 | MLSys'24 | [arXiv](https://arxiv.org/abs/2311.01282) | Unified max for softmax; flat GEMM. Niche | — | — |
| SKIP | Striped Attention (standalone) | 2023 | MIT | [arXiv](https://arxiv.org/abs/2311.09431) | Fold into CP card as the causal-load-balance tip | — | (CP card) |
| SKIP | Inference context parallelism (pass-KV/pass-Q) | 2024 | Meta | [arXiv](https://arxiv.org/abs/2411.01783) | Fold into CP card as a number | — | (CP card) |
| SKIP | Zamba2 shared blocks, Samba/SambaY, YOCO | 2024–25 | Zyphra, Microsoft | [YOCO](https://arxiv.org/abs/2405.05254) | Niche; one table row or a cross-layer tip | Phi-4-mini-flash (SambaY) | (table / 03) |
| SKIP | USP, LoongTrain, MagiAttention, CacheGen/CacheBlend, Lean/POD-Attention, Dynamo | 2024–25 | various | — | Not mainstream enough for a card | — | — |
| SKIP | Character.AI hybrid inference recipe | 2024 | Character.AI | — | Primary page unreachable; details UNVERIFIED | — | — |

---

## 4. Card data

### Production layouts 2024–2026 (2024–26, configs)
- url: see §2 table (one config per row)
- one-line idea: Every frontier model now mixes cheap layers with a few expensive "full-memory" layers. The mix is readable from `config.json`.
- key formula:
  $$M_{\text{seq}}(N)=N\!\!\sum_{\ell\in\mathcal G}\!c_\ell+\sum_{\ell\in\mathcal W}\!\min(N,W)\,c_\ell+\sum_{\ell\in\mathcal R}\!|S_\ell|\,s_\ell,\qquad c_\ell=2H_{kv}d_h s\ \ (\text{MLA: }(d_c+d_r)s)$$
- concrete numbers (computed from config, BF16, growing layers): Llama-3.1-405B 504 KiB/tok · GLM-4.6 368 · MiniMax-M2 248 · Gemma 2 27B 184 · Falcon-H1-34B 144 · Gemma 3 27B 80 · DeepSeek-V3 68.6 · Llama 4 48 · Nemotron-H-56B 40 · Gemma 4 31B 40 · gpt-oss-120b 36 · Kimi K3 27 · Qwen3-Next 24 · Jamba 16 · Kimi Linear 7.9 · Nemotron 3 Nano 6 · V4.1-Flash 0.87 (reported 890 B).
- diagram idea: one horizontal strip per model (x = layer index, scaled to equal width). Colours: blue = global attention, light blue = SWA/chunked (with W printed), purple = recurrent (GDN/KDA/Mamba/lightning), teal dot = NoPE, grey hatch = sparse indexer. Right margin: bar of KV KiB/token on a log scale. Sort by date and group by family: local/global, SSM, linear, sparse, full.
- tips:
  - **Read it off config:** `layer_types`, `hybrid_override_pattern`, `full_attn_layers`, `attn_type_list`, `no_rope_layers`.
  - **Gotcha:** "hybrid" can mean SWA+GA, SSM+GA, linear+GA or parallel heads. The KV saving differs by family.
  - **Not solved:** global layers still grow with N, and their prefill is still quadratic.
  - **Numbers:** global fraction clusters at 1/2, 1/4, 1/6 and 1/8–1/10.
- verification: all 35 configs fetched 2026-10-21. Patterns re-derived by script, and layer counts matched against paper/card text where stated (Nemotron-H, Nano 2, Nemotron 3 Nano, MiMo, Kimi K3, Qwen3-Next/3.5).

### Local/global sliding-window interleave + learned sinks (2024–26, Google / OpenAI / Xiaomi / Cohere / Meta)
- url: https://arxiv.org/html/2503.19786 · https://arxiv.org/html/2508.10925 · https://arxiv.org/html/2601.02780v1
- one-line idea: Most layers attend to the last W tokens. One in r layers is global. A per-head sink logit lets small windows "attend to nothing".
- key formula:
  $$p_{tj}=\frac{e^{a_{tj}}}{e^{\sigma_h}+\sum_{t-W<j'\le t}e^{a_{tj'}}},\qquad M(N)=L_g N c+L_l\min(N,W)\,c$$
- concrete numbers:
  - Gemma 3: 5:1, W=1024. KV memory overhead at 32K (2B, text): about 60% with global-only, below 15% with 1:3 and W=1024. Perplexity impact is "minimal, even with 7-to-1" (report §5.2).
  - gpt-oss: 1:1, W=128.
  - MiMo-V2-Flash: 5:1, W=128 plus sink, "nearly 6×" KV and compute reduction. Ablation on a 32B dense model: W=128 without sink gives MMLU 54.9; with sink 58.3; all-GA 57.3. After long-context extension, W=512 degrades (MRCR 19.6 vs 34.4 for W=128).
  - Command A: 3:1, W=4096. Ministral-8B: 3:1, W=32K. Llama 4: 3:1 chunked, 8192.
  - Computed for Gemma 3 27B at 128K: 10.4 GiB vs 62 GiB if all 62 layers were global (5.96×).
- diagram idea: left, a 16×16 causal mask grid for SWA (band, W=4 toy), chunked (block-diagonal) and global, plus one extra "sink" column coloured teal. Right, bars of per-sequence KV at 128K for Gemma 2 (1:1, 4096), Gemma 3 (5:1, 1024), Gemma 4 (5:1, K=V) and gpt-oss (1:1, 128).
- tips:
  - **Why it works:** most heads are local; a few global layers carry retrieval.
  - **Gotcha:** chunked ≠ sliding. Llama 4 resets context at each 8192-token chunk boundary.
  - **Sink ≠ sink token:** gpt-oss and MiMo add a learned logit $\sigma_h$ to the denominator; no position is stored.
  - **Not solved:** global-layer KV still grows. MiniMax dropped an SWA hybrid because quality fell "as context length grew" (M2 blog).
- verification: Gemma 3 §2/§5.2 via arXiv HTML; gpt-oss card §2; MiMo §2.1–2.2 and Tables 2–4; all configs; Llama 4 chunk mask in HF `modeling_llama4.py`.

### Mamba-2 sequential hybrids, ~8–10% attention (2024–26, AI21 / NVIDIA / IBM)
- url: https://arxiv.org/html/2403.19887v2 · https://arxiv.org/html/2504.03624 · https://arxiv.org/html/2512.20848
- one-line idea: Mamba-(2) layers do most token mixing. A handful of GQA layers, evenly spread and never first, keep exact recall.
- key formula:
  $$\text{KV/tok}=L_{\text{attn}}\cdot 2H_{kv}d_h s,\qquad |S|=L_{\text{ssm}}\cdot H\cdot P\cdot N_{\text{state}}\cdot s_{\text{state}}$$
- concrete numbers:
  - Jamba 1:7, 4 of 32 layers: KV at 256K is 4 GB vs Mixtral 32 GB and Llama-2 128 GB (Table 1). Computed 4.0 GiB ✓.
  - Nemotron-H-56B: 10/118 attention layers. Reported 2.4× throughput vs Qwen-2.5-72B/Llama-3.1-70B (65,536 in / 1,024 out, H100).
  - Nemotron Nano 2 9B: 4/56. Reported up to 6× vs Qwen3-8B (8K in / 16K out, A10G).
  - Nemotron 3 Nano: 6/52, H_kv=2. Reported 3.3× vs Qwen3-30B-A3B (8K/16K, H200).
  - Granite 4.0-H: 9:1, NoPE. Reported ">70% reduction in RAM" for long or concurrent inputs.
- diagram idea: three stacked strips (Jamba 32, Nemotron 3 Nano 52, Nemotron-H-56B 118) with attention cells in blue at their exact indices. Below, a KV-vs-N line plot: Llama-3.1-70B-style full GQA against Nemotron-H-56B, with a flat state offset (864 MiB fp32, computed).
- tips:
  - **Placement rule (Nemotron-H):** first layer Mamba-2; attention evenly dispersed; attention precedes FFN.
  - **No positions:** Jamba, Nemotron-H and Granite use no position embedding; Mamba supplies order.
  - **Gotcha:** the SSM state is per request and fp32, so short prompts can cost more memory than KV.
  - **Not solved:** throughput claims are measured at long input/output; short-context gains are smaller.
- verification: Jamba §3 Table 1 (HTML v2); Nemotron-H §2 (layer counts, "do not use any position embeddings", "≈8%"); Nano 2 and Nemotron 3 Nano intro and abstract; IBM announcement 2025-10-02; configs.

### GDN/KDA 3:1 hybrids, 2026 update (2025–26, Qwen / Moonshot)
- url: https://huggingface.co/Qwen/Qwen3.5-397B-A17B · https://arxiv.org/html/2608.30320 · https://huggingface.co/moonshotai/Kimi-K3
- one-line idea: Delta-rule recurrent layers handle 3 of 4 layers. The 4th layer is gated GQA (Qwen) or NoPE MLA (Kimi). In Qwen3.8 that layer is itself made sparse (QSA).
- key formula:
  $$\text{stack}=n\times\big[\,3\times(\text{GDN}\!\to\!\text{MoE})\;+\;1\times(\text{GatedAttn}\!\to\!\text{MoE})\,\big],\quad n=12\ (\text{3-Next, 3.8}),\ 15\ (\text{3.5-397B})$$
- concrete numbers:
  - Qwen3-Next: 24 KiB/tok + 72 MiB fp32 state, so the state equals the KV of ≈3,072 tokens (computed). Reported "10 times inference throughput for context over 32K" vs Qwen3-32B (card).
  - Kimi Linear: TPOT 1.84 ms vs 11.48 ms for MLA at 1M (6.3×); "up to 75%" KV reduction (paper Fig. 1, abstract).
  - Kimi K3: 69 KDA + 24 gated MLA, 96 heads, 2.8T total / 104B active, 1M context.
  - Qwen3.8 QSA: kernel-level 7.6× prefill and 4.9× decode at 1M. On a 25B-A3B ablation, GDN hybrid scores 53.81 vs full 49.87 vs SWA hybrid 51.15 (Table 1).
- diagram idea: four aligned strips (Qwen3-Next 48, Qwen3.5 60, Qwen3.8 48 with hatched QSA cells, Kimi K3 93) plus a crossover plot: memory vs N for state + KV against pure GQA. Mark the computed crossover at ≈3K tokens.
- tips:
  - **Gotcha:** the last group can be irregular (Kimi Linear ends 2:1; K3 ends with two MLA layers).
  - **Compare:** Kimi uses NoPE in MLA; Qwen keeps partial RoPE and rejected NoPE (endless generation).
  - **Trend:** the global layer itself goes sparse (QSA: r=4 micro-blocks, budget 2048).
  - **Not solved:** global layers still store every token; QSA bounds reads, not storage.
- verification: Qwen3-Next and 3.5 model cards ("Hidden Layout" lines) and configs; Kimi Linear §5 and Table 1; K3 card architecture table and config; Qwen3.8 §2.1–2.1.2 (via WebFetch of HTML; the forward FlashQLA speedup was garbled, so it is omitted).

### NoPE global layers (2024–26, many)
- url: https://arxiv.org/pdf/2504.00698 · https://ai.meta.com/blog/llama-4-multimodal-intelligence/ · https://arxiv.org/html/2510.26692v1
- one-line idea: Let the local, chunked or recurrent layers encode order. Global layers attend without rotation, so they need no RoPE extension at long context.
- key formula (Llama 4 NoPE layers):
  $$s_{tj}=\frac{(\tau_t q_t)^\top k_j}{\sqrt{d_h}},\qquad \tau_t=1+0.1\,\ln\!\big(1+\lfloor (t+1)/8192\rfloor\big)$$
- concrete numbers:
  - Command A: 3 SWA (RoPE) + 1 GA (NoPE).
  - Llama 4: 36 chunked RoPE + 12 GA NoPE; Scout context 10M (blog).
  - Jamba, Nemotron-H/Nano 2/3 Nano and Granite 4.0-H: no position embeddings anywhere.
  - Kimi Linear and K3: NoPE MLA. Reasons: KDA is position-aware; enables MQA conversion; no YaRN tuning.
  - Counter: Qwen3.8 NoPE variant shows "substantially higher rate of endless generation".
- diagram idea: a strip per model with teal "RoPE" dots on local layers and empty dots on global. Inset: the τ_t curve from 0 to 1M tokens (steps at multiples of 8192; τ≈1.48 at 1M, computed).
- tips:
  - **Why it works:** the local or recurrent layers already inject order; global layers act as content retrieval.
  - **Gotcha:** NoPE scores do not decay with distance, so Llama 4 adds query temperature scaling.
  - **Not solved:** evidence is mixed (Qwen3.8 negative).
- verification: Command A PDF §2 text; Llama 4 HF code lines 358–366 (`attn_scales`); Kimi Linear §4; Qwen3.8 §2.1.1.

### MiniMax: lightning hybrid → full → sparse (2025–26, MiniMax)
- url: https://arxiv.org/html/2501.08313 · https://www.minimax.io/news/why-did-m2-end-up-as-a-full-attention-model · https://huggingface.co/MiniMaxAI/MiniMax-M3
- one-line idea: A production lab abandoned a 7:1 linear hybrid for full GQA, then moved to block-sparse attention. "Linear wins" is not settled.
- key formula (MSA read budget):
  $$|\text{read}_t| = k_{\text{blk}}\cdot B_s = 16\times128=2048\ \text{tokens per query (+1 local block)},\quad \text{retained}=N$$
- concrete numbers:
  - Text-01: 80 layers, softmax at every 8th, 64 heads × 128, lightning block 256; 1M training, 4M inference extrapolation.
  - M2: 62 all-GA layers, 48/8/128: computed 248 KiB/tok.
  - M3: 3 dense + 57 MSA, 64/4/128: computed 120 KiB/tok. Reported "9× prefill and 15× decode" vs M2 at 1M, "per-token compute to 1/20" (card).
  - M2 blog reasons: evaluation blind spots in multi-hop reasoning; experiment cost; issues appear only at scale; memory-bound linear kernels; low-precision state sensitivity; prefix caching and speculative decoding; an SWA hybrid "degraded noticeably as context length grew".
- diagram idea: three strips (80 / 62 / 60 layers) on a timeline (2025-01, 2025-10, 2026-06) with KV KiB/tok bars and the read budget per query. An annotated column lists the blog's reasons as icons-free short labels.
- tips:
  - **Not a mechanism card:** it is evidence about what fails in production.
  - **Gotcha:** "full attention" in M2 is GQA (H_kv=8), not MHA.
  - **Compare:** M3 keeps all KV (like DSA); only reads are sparse.
- verification: MiniMax-01 §2 via HTML; blog by Haohai Sun, 2025-10-29; M2/M3 configs and M3 card. The M3 tech report was not read; its speedups are card-level claims.

### Parallel (intra-layer) hybrids (2024–25, NVIDIA / TII / Meta study)
- url: https://arxiv.org/html/2411.13676v1 · https://arxiv.org/html/2507.22448 · https://arxiv.org/html/2510.04800v3
- one-line idea: Run attention heads and SSM heads on the same input in every layer, then normalise and fuse.
- key formula (Hymba):
  $$Y=W_{\text{out}}\big(\beta_1\odot\operatorname{norm}(M_{\text{attn}}\tilde X)+\beta_2\odot\operatorname{norm}(M_{\text{ssm}}\tilde X)\big)$$
- concrete numbers:
  - Hymba-1.5B: KV 79 MB vs 918 MB for Llama-3.2-3B (8K, FP16); 664 vs 191 tok/s (A100, batch 128); 128 meta tokens; GA only at layers 0, 15, 31.
  - Falcon-H1-34B: channels SSM:attn:MLP ≈ 2:1:5; attention in all 72 layers (computed 144 KiB/tok).
  - Meta study (1B, 60B tokens): intra-hybrid NLL 2.728 vs inter (1:5) 2.735 vs Transformer 2.750; KV 38 vs 43 vs 256 (reported units).
- diagram idea: side-by-side block diagrams with tensor widths. Sequential (layer strip) vs parallel (split channels in each layer). Bar of KV/tok for Falcon-H1 vs Nemotron-H at similar size.
- tips:
  - **Gotcha:** parallel ≠ fewer KV layers. Falcon-H1 keeps KV in every layer; savings come from few KV heads.
  - **Why it works:** attention heads give "high-resolution recall"; SSM heads summarise (Hymba abstract).
  - **Numbers:** normalisation before fusion is "crucial due to the scale differences" (Meta study).
- verification: Hymba §2, Table 2 and config; Falcon-H1 §2 (parallel, 2:1:5) and Table 1; Meta study Tables 6–7 via HTML (first 100K characters).

### Choosing the ratio (2025, evidence)
- url: https://arxiv.org/abs/2507.06457 · https://arxiv.org/html/2510.26692v1 · https://arxiv.org/html/2510.04800v3
- one-line idea: The linear:full ratio trades recall for memory. The evidence points to 3:1–6:1 for linear and SSM layers and 5:1 for SWA.
- key formula: $r=\dfrac{L_{\text{cheap}}}{L_{\text{global}}}$; KV/tok ∝ $\dfrac{L}{r+1}$.
- concrete numbers:
  - Kimi Linear (validation PPL): 3:1 5.65 · 1:1 5.66 · 7:1 5.70 · 15:1 5.82 · 0:1 (full) 5.77.
  - Systematic analysis (72 models, 340M/1.3B): "between 3:1 and 6:1" for Transformer-level recall; GatedDeltaNet and HGRN-2 recommended; "superior standalone linear models do not necessarily excel in hybrids".
  - Meta: quality best at 1:1, efficiency at ~1:5; "never place Transformer blocks at the front".
  - Gemma 3: 7:1 gives minimal perplexity change.
- diagram idea: x = ratio (0:1 … 15:1), y = validation PPL (Kimi) with a secondary axis of KV/tok fraction (1/(r+1)). Mark production choices on the x-axis: gpt-oss 1, Qwen and Kimi 3, Gemma 3 and MiMo 5, MiniMax-01 7, Granite 9, Nemotron-H ≈11.
- tips:
  - **Not solved:** small-scale ablations; MiniMax saw deficits appear only at scale.
  - **Gotcha:** ratio alone hides placement (front vs middle) and window size.
- verification: abstracts and tables via WebFetch.

### Jet-Nemotron / PostNAS (2025, NVIDIA)
- url: https://arxiv.org/html/2508.15884
- one-line idea: Start from a pretrained full-attention model with frozen MLPs. Search which few layers stay full or SWA, and replace the rest with JetBlock (GDN plus dynamic conv).
- key formula: $\text{KV/tok}=|\mathcal G|\cdot 2H_{kv}d_h s$, with $|\mathcal G|$ chosen by beam search over a once-for-all supernet.
- concrete numbers (2B on Qwen2.5-1.5B): full attention at L15/L20, SWA at L21/L22 (W=1152). KV 154 MB vs 7,168 MB for Qwen3-1.7B at 64K. Reported 47× generation throughput at 64K (2,885 vs 61 tok/s), up to 53.6× decode and 6.14× prefill at 256K (H100). Prefill only 1.14× at 4K.
- diagram idea: 28-layer strip for Qwen2.5 (all blue) → Jet-Nemotron (2 blue, 2 light-blue, 24 purple), with a throughput-vs-context curve.
- tips:
  - **Key finding 4:** "KV cache size is the most critical factor" for long-context throughput.
  - **Gotcha:** the 4B text and Table 9 disagree on layer indices. Do not quote the 4B layout.
  - **Not solved:** gains are small at short context.
- verification: §2–3 and the efficiency tables via HTML; 2B config matches the paper's layer indices.

### FlashAttention 1→4 (2022–26, Dao et al.)
- url: https://arxiv.org/abs/2205.14135 · https://arxiv.org/abs/2307.08691 · https://arxiv.org/abs/2407.08608 · https://arxiv.org/abs/2603.05451
- one-line idea: Exact attention, tiled into SRAM with online softmax. Each version re-schedules for one GPU generation's bottleneck.
- key formula (HBM accesses, FA1 Thm 2; $M$ = SRAM size):
  $$\text{standard: }\Theta(Nd+N^2)\qquad\text{FlashAttention: }\Theta\!\Big(\frac{N^2d^2}{M}\Big)$$
- concrete numbers:
  - FA1 (A100): 25–40% of peak; 3× GPT-2 at 1K; 15% BERT-large end-to-end.
  - FA2 (A100): 50–73% of peak; 225 TFLOP/s per GPU in training (72% MFU); only 35% on H100.
  - FA3 (H100): 1.5–2.0×, 740 TFLOP/s FP16 (75%), FP8 ≈1.2 PFLOP/s, 2.6× lower FP8 error.
  - FA4 (B200 BF16): 1613 TFLOP/s (71%), 1.3× cuDNN 9.13, 2.7× Triton. CuTe-DSL cuts compile time 20–30×.
- diagram idea: GPU timeline A100 → H100 → B200. Each column shows the bottleneck (non-matmul FLOPs; async TMA and tensor cores; exp units and SMEM bandwidth) and the utilisation bar (25–40 → 50–73 → 75 → 71%). Inset: tile grid of Q-blocks × K-blocks with the running (m, ℓ).
- tips:
  - **Not solved:** FLOPs stay $O(N^2)$; only bytes move less.
  - **FA4 tricks:** software-emulated exp, conditional softmax rescaling, tensor memory and 2-CTA MMA.
  - **Gotcha:** speed ranks depend on GPU, dtype, head dim and mask; versions are not monotone across hardware.
- verification: four abstracts read via WebFetch; FA1 Theorem 2 from memory of §3.2 (asymptotic statement; check before printing).

### Flash-Decoding / split-KV (2023, Stanford CRFM / Meta)
- url: https://crfm.stanford.edu/2023/10/12/flashdecoding.html
- one-line idea: At decode the query length is 1, so parallelise over KV chunks too, then merge partial softmaxes.
- key formula:
  $$o=\sum_{s=1}^{S}\frac{\ell_s e^{m_s-m^\*}}{\sum_r \ell_r e^{m_r-m^\*}}\,o_s,\qquad m^\*=\max_s m_s$$
- concrete numbers: at batch 1, FlashAttention uses "<1%" of the GPU. Reported up to 8× faster decode for very long sequences (CodeLlama-34B, 512–64K) and up to 50× faster attention vs FlashAttention. Runtime stays almost flat to 64K (A100).
- diagram idea: one query row vs a long KV bar split into S coloured chunks on different SMs. Each chunk emits (m_s, ℓ_s, o_s) and a reduction tree merges them. Bar: SM occupancy before and after.
- tips:
  - **Why it works:** decode is HBM-bound; more SMs means more aggregate bandwidth.
  - **Used in:** split-KV paths in FA2+, FlashInfer and FlashMLA (`num_splits` in its API).
  - **Not solved:** total bytes read are unchanged; only latency at small batch improves.
- verification: blog read (authors Dao, Haziza, Massa, Sizov; published around 2023-10-13).

### FlashMLA (2025, DeepSeek)
- url: https://github.com/deepseek-ai/FlashMLA
- one-line idea: With absorbed weights, MLA decode is MQA: 128 query heads read one 576-d latent per token. Arithmetic intensity is high enough to be near compute-bound.
- key formula (computed intensity per cached token per layer):
  $$I=\frac{2H_q\,(2d_c+d_r)}{s\,(d_c+d_r)}=\frac{2\cdot128\cdot1088}{2\cdot576}\approx242\ \text{FLOP/B (BF16)};\ \ \approx425\ \text{(FP8, 656 B)}$$
- concrete numbers:
  - Feb 2025 release: paged KV block 64, BF16/FP16; "3000 GB/s memory-bound, 580 TFLOPS compute-bound" (H800 SXM5).
  - Apr 2025: up to 660 TFLOPS.
  - V3.2 DSA kernels (2025-09-29): 640 TFLOPS sparse prefill, 410 TFLOPS sparse decode with FP8 KV.
  - FP8 KV = 656 B/token/layer (512 B FP8 + 16 B scales + 128 B BF16 RoPE).
  - V4.1 (2026-09): decode KV 528 B (FP8) and 288 B (FP4) per token (per-layer scope UNVERIFIED). Hopper support removed 2026-09-30.
- diagram idea: 576-wide latent row (512 blue + 64 teal) read once, fanned out to 128 gold query heads. Byte bar BF16 1152 B → FP8 656 B. Intensity marker vs H800 ridge ≈295 (03 card's H100 figure; the H800 has the same compute; UNVERIFIED as a spec here).
- tips:
  - **Why it works:** sharing one latent across 128 heads multiplies FLOPs per byte by about 128 vs MHA.
  - **Gotcha:** RoPE part kept in BF16 for accuracy; only the NoPE latent is FP8.
  - **Link:** connect to the MLA card and 03's intensity = G card.
- verification: README at commits 480405a (2025-02-26) and 1408756 (≤2025-11-01), plus the current README.

### PagedAttention + RadixAttention prefix caching (2023–24, vLLM SOSP'23 / SGLang)
- url: https://arxiv.org/html/2309.06180 · https://arxiv.org/html/2312.07104v2
- one-line idea: Store KV in fixed blocks addressed by a block table (no fragmentation), and index finished requests' blocks in a radix tree so shared prefixes skip prefill.
- key formula:
  $$\text{waste}\le (b-1)\ \text{tokens/seq},\qquad \text{prefill FLOPs}\propto N-N_{\text{hit}},\qquad N_{\text{hit}}=\text{longest cached prefix}$$
- concrete numbers:
  - vLLM: prior systems use only 20.4–38.2% of KV memory for token states; block b=16; sharing saves 6.1–9.8% (parallel sampling) and 37.6–55.2% (beam); 2–4× throughput vs FasterTransformer/Orca.
  - SGLang: page = 1 token, LRU leaf eviction, longest-shared-prefix-first = DFS order; hit rate 50–99% across benchmarks; production 52.4% (LLaVA-NeXT-34B) and 74.1% (Vicuna-33B); up to 6.4× throughput.
  - DeepSeek production: 56.3% of input tokens hit the on-disk cache.
- diagram idea: left, a logical sequence → block table → scattered physical blocks (blue), with the partially filled last block in red. Right, a radix tree of 4 requests sharing a system prompt; green nodes are hits, grey nodes evicted LRU leaves; hit-rate bars.
- tips:
  - **Gotcha:** a prefix hit requires a byte-identical token prefix; context truncation can halve hit rate (LMCache).
  - **Hybrids:** recurrent state has no per-token blocks; see the state-cache card.
  - **Not solved:** cache capacity; this leads to tiered offload.
- verification: vLLM §1, §6.3, §7.2 via HTML; SGLang §3 and §6 via HTML; DeepSeek day-6 doc.

### Prefill/decode disaggregation (2024–25, DistServe OSDI'24 / Splitwise ISCA'24 / Mooncake FAST'25)
- url: https://arxiv.org/abs/2401.09670 · https://arxiv.org/abs/2311.18677 · https://www.usenix.org/conference/fast25/presentation/qin
- one-line idea: Prefill is compute-bound (TTFT) and decode is bandwidth-bound (TPOT). Run them on separate GPU pools and ship the KV cache between them.
- key formula (bytes moved per request, computed):
  $$\text{KV}_{\text{xfer}}=N_{\text{prompt}}\sum_{\ell\in\mathcal G}c_\ell\;(+\,\text{states}),\quad\text{e.g. V3 FP8: }61\cdot656\,\text{B}\times32\text{K}\approx1.22\ \text{GiB}$$
- concrete numbers:
  - DistServe: 7.4× more requests or 12.6× tighter SLO.
  - Splitwise: 1.4× throughput at 20% lower cost, or 2.35× at the same cost and power.
  - Mooncake (Kimi): +59–498% effective request capacity on real traces; Kimi handles +115% (A800) / +107% (H800) more requests; FAST'25 best paper.
  - DeepSeek production (2025-02-27/28): prefill unit 4 nodes (EP32/DP32), decode 18 nodes (EP144/DP144); 73.7k input tok/s/node (incl. cache hits) and 14.8k output tok/s/node.
- diagram idea: two GPU pools with an RDMA arrow carrying KV blocks (size label in GiB); TTFT and TPOT timelines, co-located (interference spikes) vs disaggregated. Side bar: DeepSeek node counts 4 vs 18.
- tips:
  - **Why it works:** removes prefill-decode interference; each pool picks its own parallelism.
  - **Gotcha:** transfer volume ∝ KV/tok. MLA, FP8 and hybrids make disaggregation cheaper.
  - **Hybrids:** recurrent state is sent atomically, not page by page (SGLang).
  - **Compare:** chunked prefill solves the same interference on one pool.
- verification: abstracts (DistServe venue in comments); USENIX page and Tsinghua note for Mooncake award and numbers; DeepSeek day-6 GitHub doc read directly; Splitwise venue from PDF filename and search (UNVERIFIED wording).

### Context parallelism: ring vs all-to-all vs all-gather (2023–24, Berkeley / Microsoft / Meta)
- url: https://arxiv.org/html/2310.01889 · https://arxiv.org/html/2309.14509 · https://arxiv.org/html/2407.21783 · https://arxiv.org/abs/2311.09431
- one-line idea: Split a long sequence over P GPUs. Either rotate KV blocks (ring), all-to-all from sequence-split to head-split (Ulysses), or all-gather the small GQA K/V (Llama 3).
- key formula:
  $$\text{Ring overlap: } c\ge F/B;\qquad \text{Ulysses per-link volume: }\tfrac{4Nh}{P}\ \text{vs Megatron-SP } 4Nh$$
- concrete numbers:
  - Ring minimum block c: A100 NVLink 1.0K, A100 InfiniBand 24.5K, TPU v4 1.0K tokens (Table 2).
  - Ulysses: 2.5× faster at 4× longer sequences; trains on >1M tokens.
  - Llama 3: all-gather K/V, input in 2×CP chunks (rank i gets chunks i and 2CP−1−i); CP=16 at 131,072 tokens (Table 4).
  - Striped: 1.45× over Ring at 256K (A100/TPUv4), 1.65× at 786K (16 TPUv4).
  - Inference CP: Llama-3-405B 1M prefill in 77 s on 128 H100 (93% parallel efficiency).
- diagram idea: three panels, each with 4 GPUs. (a) ring arrows passing KV blocks plus a causal-mask workload heatmap (contiguous vs striped); (b) all-to-all reshaping [N/P, H] → [N, H/P]; (c) all-gather of small K/V. Bottom: bytes moved per GPU for N=128K, Llama-3-70B dims (computed by the writer).
- tips:
  - **Gotcha (Ulysses):** P is bounded by head count, because heads are split by construction. GQA's small H_kv makes this tighter.
  - **Gotcha (ring):** the causal mask unbalances work; fix with striping or 2×CP chunking.
  - **Why all-gather (Llama 3):** GQA K/V are small and arbitrary document masks are easy.
  - **Not solved:** total FLOPs stay $O(N^2)$; CP spreads them across GPUs.
- verification: Ring §3 and Table 2; Ulysses §3.1–3.2; Llama 3 §3.3.2 and Table 4; Striped abstract; Meta inference CP abstract.

### Hybrid-model serving: state caches (2025, SGLang / vLLM)
- url: https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/ · https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/
- one-line idea: Recurrent state is per request, updated in place and large. Prefix caching, speculative decoding and PD transfer need explicit state checkpoints and slots.
- key formula (computed crossover):
  $$N^\*=\frac{|S|\,s_{\text{state}}}{\text{KV/tok}}\quad(\text{Qwen3-Next: }72\ \text{MiB}/24\ \text{KiB}\approx3072\ \text{tokens})$$
- concrete numbers:
  - SGLang: Mamba pool (per request) separate from KV pool (per token); MambaRadixCache copies state on match, forks a checkpoint on insert, and keeps two LRU lists.
  - Speculative decoding: one state slot per draft token. Qwen3-Next-FP8 at batch 1: 257 → 307 → 325 tok/s with MTP 2/3/4.
  - LMCache docs: Qwen3-Next is not supported (state cannot be paged).
  - vLLM sliding-window layers: prefix lookup scans right to left; only the last W−1 tokens must be cached.
- diagram idea: memory panel split into a "KV pool (blocks)" and a "state pool (slots)"; radix tree whose nodes carry KV blocks and only some carry state snapshots. Crossover chart.
- tips:
  - **Why it matters:** this is MiniMax's reason #8 for leaving linear attention.
  - **Gotcha:** a state hit is valid only at exact checkpoint positions ("all or nothing").
  - **Not solved:** hierarchical offload of states (SGLang future work).
- verification: PyTorch blog (2025-12-03) read in full; vLLM design-doc summary via search snippet only (UNVERIFIED detail beyond the above).

### Chunked prefill / Sarathi-Serve (2024, OSDI'24)
- url: https://arxiv.org/abs/2403.02310
- one-line idea: Cut prompts into chunks and co-schedule them with ongoing decodes under a per-iteration token budget. Decodes never stall.
- key formula: $n_{\text{decode}}+|\text{chunk}|\le\tau$ tokens per iteration.
- concrete numbers: reported serving capacity vs vLLM: 2.6× Mistral-7B (1×A100), up to 3.7× Yi-34B (2×A100); up to 5.6× Falcon-180B with pipeline parallelism.
- diagram idea: iteration timeline, before (a long prefill blocks decode tokens: a red stall) vs after (uniform mixed batches).
- tips:
  - **Compare:** same interference problem as disaggregation; one pool, no KV transfer.
  - **Gotcha:** small chunks re-read the prefix KV every chunk (more attention bytes).
- verification: abstract; OSDI'24 confirmed via the USENIX listing.

### Tiered KV offload (2024–25, LMCache / Mooncake / DeepSeek)
- url: https://arxiv.org/abs/2510.09665 · https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md
- one-line idea: Keep computed KV beyond GPU memory (CPU DRAM, SSD, remote) and reload instead of recomputing.
- key formula: reload wins if $\dfrac{N\cdot\text{KV/tok}}{BW_{\text{tier}}}<T_{\text{prefill}}(N)$.
- concrete numbers: LMCache up to 15× throughput with vLLM (multi-round QA, document analysis); truncation can halve hit rate. DeepSeek: 56.3% of 608B input tokens hit the on-disk KV cache over 24 h. V4.1: global KV kept ≥72 h; SWA KV in 10% of host DRAM with a TTL of minutes.
- diagram idea: memory-hierarchy pyramid (HBM / DRAM / SSD) labelled with capacity and bandwidth; bars of KV/tok for V3 (BF16), V3.2 (FP8) and V4.1 (890 B) showing how much fits per TB.
- tips:
  - **Gotcha:** small KV/tok (MLA, FP4) is what makes SSD caching viable.
  - **Not semantic memory:** cached KV is tied to model and exact prefix.
- verification: LMCache abstract; DeepSeek doc; V4.1 paper (persistent cache section).

### FlashInfer (2025, MLSys'25)
- url: https://arxiv.org/abs/2501.01005
- one-line idea: One block-sparse KV representation covers paged, radix and sparse layouts. JIT templates and a CUDAGraph-compatible load balancer serve all attention variants.
- key formula: KV layout = BSR matrix with block size $(b_r,b_c)$; a page table is the special case $b_c=$ page size.
- concrete numbers: 29–69% inter-token-latency reduction vs compiler backends; 28–30% for long context; 13–17% for parallel generation. Used in SGLang, vLLM and MLC-Engine.
- diagram idea: the same KV pool viewed as a page table, a radix tree and a top-k index, all mapping to one BSR grid.
- tips: **Why:** one kernel family for heterogeneous serving. **Gotcha:** the block format must match the sparsity granularity.
- verification: abstract only. The BSR-generalisation wording comes from the abstract's "block-sparse format"; check §3 before printing the formula.

### Low-precision attention: FA3-FP8 and SageAttention 1/2/3 (2024–25)
- url: https://arxiv.org/abs/2407.08608 · https://arxiv.org/abs/2410.02367 · https://arxiv.org/abs/2411.10958 · https://arxiv.org/abs/2505.11594
- one-line idea: Quantise Q, K, P and V inside the attention kernel with smoothing and fine-grained scales to use INT8/INT4/FP8/FP4 tensor cores.
- key formula: $QK^\top\approx \delta_Q\delta_K\,\hat Q\hat K^\top$ with per-block scales $\delta$; smooth $K\leftarrow K-\bar K$ (softmax-invariant).
- concrete numbers:
  - FA3 FP8 ≈1.2 PFLOP/s (H100) with 2.6× lower error than baseline FP8.
  - Sage1 (ICLR'25): 2.1× / 2.7× over FA2 / xformers.
  - Sage2 (ICML'25): INT4 Q,K plus FP8 P̃V, ~3× / 4.5× on RTX4090; matches FA3-FP8 speed on Hopper.
  - Sage3: FP4 microscaling, 1038 TOPS on RTX5090, 5× the fastest FA on that GPU. 8-bit training is lossless for fine-tuning but converges slower in pretraining.
- diagram idea: bit-width ladder (BF16 → FP8 → INT4/FP4) per operand (Q, K, P, V) with TOPS bars per GPU.
- tips:
  - **Gotcha:** K smoothing is exact because softmax is shift-invariant per row. The Sage1 K-mean detail is UNVERIFIED (abstract does not state it).
  - **Not solved:** pretraining with low-bit attention.
- verification: abstracts. Sage3 venue UNVERIFIED.

### Cross-layer KV reuse in production (2024–26, feeds the planned 03 card)
- url: https://huggingface.co/unsloth/gemma-3n-E4B-it/blob/main/config.json · https://huggingface.co/google/gemma-4-31B-it · https://arxiv.org/html/2609.19969
- one-line idea: Later layers read KV produced by an earlier layer, or K and V are the same tensor. Fewer distinct caches, same number of reads.
- key formula: $C_{\text{distinct}}=L-L_{\text{shared}}$; K=V halves $c_\ell$: $c_\ell=H_{kv}d_h s$.
- concrete numbers: Gemma 3n E4B: 15 of 35 layers share (`num_kv_shared_layers 15`). Hymba: pairs share (14 groups). Gemma 4 31B global: 4 KV heads × 512, K=V. V4.1-Flash: 30 of 38 CSA2 layers in Reuse mode (reuse KV and top-k indices).
- diagram idea: layer strip with arrows from producer layers to consumer layers; byte bar per model.
- tips:
  - **Gotcha:** reads per token do not fall; only storage does.
  - **YOCO:** caches KV once for a whole cross-decoder (venue UNVERIFIED).
- verification: configs; Gemma 4 card ("unified Keys and Values"); V4.1 §4.2.1.

### FlexAttention (2024–25, PyTorch, MLSys'25)
- url: https://pytorch.org/blog/flexattention/ · https://arxiv.org/abs/2412.05496
- one-line idea: Write `score_mod(score,b,h,q_idx,kv_idx)` and `mask_mod(b,h,q_idx,kv_idx)` in Python; torch.compile emits a fused FlashAttention-style kernel that skips empty blocks (BlockMask, default 128).
- key formula: $o=\operatorname{softmax}(\operatorname{score\_mod}(QK^\top/\sqrt d))\,V$ over blocks where `mask_mod` is nonzero.
- concrete numbers: 90% of FA2 forward and 85% backward (A100, causal); BlockMask for 1M tokens ≈60 MB; torchtune packing +71%.
- diagram idea: four mask_mods (causal, SWA, document, prefix-LM) as block grids with skipped blocks grey.
- tips: **Used in:** Hymba (`attn_implementation: flex`). **Gotcha:** rebuilding the BlockMask each batch costs time; reuse it across layers.
- verification: PyTorch blog read in full (last updated 2025-05-30); MLSys'25 via mlsys.org listing.

---

## 5. Suggested restructure

1. **NEW chapter "Production layouts 2024–2026"** (between the linear/delta and sparse chapters, or replacing legacy 08's intro): ① layout strips (§2), ② local/global + sinks, ③ Mamba-2 hybrids, ④ GDN/KDA 3:1 (with the 2026 update), ⑤ NoPE globals, ⑥ MiniMax reversal, ⑦ Compare: ratio evidence. Parallel hybrids and Jet-Nemotron fit as tips or an optional 8th card.
2. Keep legacy 08's gate-location card (output gate vs state gate) as card ④'s `more` block or as a short separate card.
3. **Split legacy 09 into two chapters.** "Kernels and parallelism": FA1→4 · Flash-Decoding · FlashMLA · (FlashInfer/FlexAttention as tips) · low-precision attention · context parallelism. **NEW "Serving the KV cache"**: Paged + Radix · chunked prefill vs disaggregation · tiered offload · hybrid state caches · Compare (bytes/tok across tiers).
4. Cross-layer reuse goes to the planned 03 card; feed it the §4 data. FP8/FP4 KV formats (656 B, 890 B) go to "Serving" or the MLA chapter.
5. Every KV-bytes figure should reuse the §2 computed column so numbers stay consistent book-wide.
