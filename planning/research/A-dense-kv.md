# A · Dense softmax + KV-efficient attention — research notes (checked 2026-10-21)

Scope: head structure (MHA/MQA/GQA/MLA → K=V, MFA, GTA/GLA), positional encoding (RoPE base/scaling, partial RoPE, NoPE, ALiBi), logit/softmax stability (QK-norm, soft-cap, QK-Clip, sinks, temperature, Diff, FoX), local:global interleaving, cross-layer KV sharing.
Conventions: `d_h` head dim; `H_q/H_kv`; `N` tokens; `L` layers; **e/t/l = KV elements per token per layer**; BF16 (2 B) unless noted. "cfg" = Hugging Face `config.json` fetched 2026-10-21 (repo given). Numbers marked *(computed)* are my arithmetic from cfg values. Anything not checked in a primary source is marked **UNVERIFIED**.

---

## 1. Coverage audit

### What the book has (my area)
- ch01: Bahdanau/Luong → QKV with mask + 1/√d_k → MHA; self vs cross, causal vs bidirectional; relative PE (Shaw 2018) + RoPE in one paragraph; prefill vs decode; toy KV byte count.
- ch02: capacity vs bandwidth; `M_KV = B N L H_kv (d_k+d_v) s`; MQA, GQA (uptraining from MHA); worked 4 GiB / 1 GiB / 128 MiB example (arithmetic checked: correct).
- ch03: MLA latent `c_j`, key/value absorption, decoupled RoPE lane, cache `d_c+d_r`=576, 93.3% vs DeepSeek-67B (all correct vs DeepSeek-V2 §2.1).
- ch09: storage factor `C` (cache sets) < `L`; CLA (Brandon et al.); KV quantisation; V4.1-Flash 890 B/token (verified in abstract of arXiv 2609.19969).
- matrix/glossary: rows for Relative/RoPE, MQA, GQA, MLA, cross-layer reuse; glossary has "attention sink" (descriptive only).
- papers.json (my area): transformer2017, relative2018, rope2021, mqa2019, gqa2023, mla2024, cla2024, v41flash2026, deepseek-v32-dsa-2025, deepseek-v4-*. **Nothing** on RoPE scaling, partial RoPE/NoPE, SWA interleaving, sinks, QK-norm/soft-cap, K=V, arithmetic intensity.

### Errors / oversimplifications → corrections
- **E1 (ch01, "the scale prevents dot-product magnitudes from growing unchecked")** — oversimplified. 1/√d_k only normalises logit variance at init; trained logits still grow ("attention logit growth" → one-hot attention / entropy collapse), which is why QK-norm, soft-capping and QK-Clip exist. Source: Wortsman et al., ICLR 2024, §3.1.1 https://arxiv.org/abs/2309.14322 ; Kimi K2 §2.1 https://arxiv.org/abs/2507.20534
- **E2 (ch01 positions paragraph)** — presents RoPE as the endpoint. 2025–26 frontier models mix RoPE with NoPE layers (Llama 4, Command A, SmolLM3, Kimi Linear MLA, Granite 4.0-H), rotate only part of each head (Qwen3-Next/3.5, GLM-4.5, MiniMax-M2, MiMo, Gemma 4, DeepSeek-V4), and their long-context behaviour is set by RoPE base + scaling (YaRN etc.). Not wrong, but now a major gap. Sources: cfgs listed in cards M3/M4.
- **E3 (ch03 absorption)** — implies MLA always runs absorbed. DeepSeek runs **MHA mode for training/prefill and MQA mode (absorbed) for decode**; DSA is built on the MQA mode where "each latent vector … will be shared across all query heads". DeepSeek-V3.2 paper §2.1 + App. A Fig. 7 caption (states it for V3.1-Terminus) https://arxiv.org/abs/2512.02556
- **E4 (ch03)** — omits that absorbed MLA *is* MQA with `d_k = d_c+d_r = 576`, `d_v = d_c = 512`; that is why MLA decode is compute-heavy (arithmetic intensity ≈ 2H_q). Zadouri, Strauss, Dao 2025 Table 1 https://arxiv.org/abs/2505.21487
- **E5 (ch03)** — omits that MLA's **head count does not change the cache** (cache is per-token latent); it changes FLOPs. Kimi K2 keeps DeepSeek-V3's 576 e/t/l but uses 64 heads instead of 128 because 128 heads = +83% inference FLOPs at 128K for only 0.5–1.2% lower val loss (K2 §2.3). Cfgs: `moonshotai/Kimi-K2-Instruct` `num_attention_heads: 64`, `kv_lora_rank: 512`, `qk_rope_head_dim: 64`.
- **E6 (ch03 minor)** — query is also low-rank (`q_lora_rank`/d_c′ = 1536, V2 §3.1.2) and the latent is RMSNorm-ed before caching (`self.kv_cache[...] = self.kv_norm(kv)`, DeepSeek-V3 `inference/model.py` L484). Book writes `q^C = P_i h_t`. Low priority.
- **E7 (ch03 "positional lane must remain separate")** — true for RoPE-MLA, but Kimi Linear runs its MLA layers with **NoPE** ("we apply NoPE to all full attention (MLA) layers", Kimi Linear §4) https://arxiv.org/abs/2510.26692 ; and DeepSeek-V4 dropped the latent up-projection for a **shared K=V MQA entry (512 dims, last 64 rotated) with output de-rotation** (V4 §2.3.1/2.3.3) https://arxiv.org/abs/2606.19348 . Add as "what came after MLA".
- **E8 (ch02 "cannot … turn arbitrary MHA weights into … GQA … sharing must be designed or learned")** — correct, but now incomplete: MHA→MLA conversion with 0.3–0.6% of pretraining data (MHA2MLA, ACL 2025) https://arxiv.org/abs/2502.14837 and GQA→MLA with 6B tokens (TransMLA) https://arxiv.org/abs/2502.07864 .
- **E9 (ch09 cross-layer)** — only CLA (a research paper); misses production adopters: Hunyuan-Large (GQA-8 + CLA every 2 layers, ≈95% vs MHA), Gemma 3n and Gemma 4 E2B/E4B (top 15–20 layers reuse KV), YOCO/SambaY (Phi-4-mini-flash-reasoning), DeepSeek-V4.1-Flash `kv_source_layer_ids`. See card M11.
- **E10 (glossary "attention sink")** — defines only the emergent phenomenon; 2025–26 models add an explicit **learned per-head sink logit** in the softmax denominator (gpt-oss, MiMo-V2-Flash, DeepSeek-V4; FlashMLA `attn_sink` arg). See card M6.
- **E11 (matrix)** — no rows for SWA interleave, sinks, QK-norm, partial RoPE/NoPE, YaRN, K=V, temperature scaling.
- FYI outside my area (do not fix here): (a) ch08 "Kimi Linear repeats three KDA layers followed by one full MLA layer" — cfg `moonshotai/Kimi-Linear-48B-A3B-Instruct`: 27 layers, MLA at 4,8,12,16,20,24,27 → 20 KDA : 7 MLA, last group 2:1. (b) ch07 V4-Pro ratio list is right, but V4-**Flash** cfg `compress_ratios` starts `[0, 0, 4, 128, …]` (two pure-SWA layers), last entry 0 in both.

---

## 2. Gap table (ranked)

| # | mechanism | year | venue / org | arXiv / official URL | what changes vs predecessor (≤15 words) | adopted in (named models) | suggested chapter |
|---|---|---|---|---|---|---|---|
| **MUST** |
| M5 | Local:global sliding-window interleave | 2023–26 | Mistral AI; Google; OpenAI | https://arxiv.org/abs/2310.06825 | Most layers keep only W recent tokens; few global layers keep full cache | Mistral 7B, Gemma 2/3/3n/4, gpt-oss, Command A, OLMo 3, MiMo-V2-Flash, Step-3.5-Flash, Ministral 8B, Llama 4 (chunked) | NEW: Local + global layers |
| M6 | Attention sinks → learned sink logit | 2023 / 2025 | ICLR 2024 (StreamingLLM); OpenAI | https://arxiv.org/abs/2309.17453 · https://arxiv.org/abs/2508.10925 | Per-head extra logit in softmax denominator lets a head attend to nothing | gpt-oss, MiMo-V2-Flash, DeepSeek-V4/V4.1 | NEW: Local + global layers |
| M1 | RoPE spectrum, base (ABF), dual local/global bases | 2021–26 | RoFormer; Qwen3; Gemma 3 | https://arxiv.org/abs/2104.09864 · https://arxiv.org/abs/2505.09388 | Raise base so slowest pair spans the context; local layers keep small base | Llama 3/4 (5e5), Qwen3 (1e6), Qwen3-Next/3.5 (1e7), Gemma 3/4, MiMo, DeepSeek-V4 | NEW: Position |
| M2 | RoPE context extension: PI → NTK → YaRN / llama3 | 2023 | Meta (PI); ICLR 2024 (YaRN) | https://arxiv.org/abs/2306.15595 · https://arxiv.org/abs/2309.00071 | Interpolate only low-frequency pairs, keep high ones, add logit temperature | Qwen3, DeepSeek-V2/V3/V3.2/V4, Kimi K2/K2.5, gpt-oss, OLMo 3, Mistral Large 3; llama3-type: Llama 3.1/4, Step-3.5 | NEW: Position |
| M3 | Partial RoPE / p-RoPE | 2021 / 2024 | EleutherAI; ICLR 2025 (GDM) | https://arxiv.org/abs/2410.06205 | Rotate only a fraction of dims; p-RoPE drops the lowest frequencies | GPT-J/NeoX, GLM-4.5, Qwen3-Next/3.5, MiniMax-M2, MiMo-V2-Flash, Phi-4-mini, Gemma 4 (p=0.25), MLA (64/192), DeepSeek-V4 (64/512) | NEW: Position |
| M4 | NoPE layers (iRoPE, RNoPE) | 2023 / 2025 | NeurIPS 2023; Meta; Cohere | https://arxiv.org/abs/2305.19466 · https://arxiv.org/abs/2501.18795 | Global layers drop RoPE; local/chunked layers keep it | Llama 4, Command A, SmolLM3, Kimi Linear (MLA), Granite 4.0-H | NEW: Position |
| M7 | QK-norm (and logit growth) | 2020 / 2023 | ICLR 2024 (analysis); Qwen3, OLMo 2 | https://arxiv.org/abs/2309.14322 | RMSNorm q and k before dot product; bounds logits | Qwen3, OLMo 2/3, Gemma 3/4, GLM-4.5, MiniMax-M2, Llama 4 Scout, Step-3.5, Hunyuan-A13B, DeepSeek-V4 | NEW: Stability |
| M8 | K=V tying (shared key/value projection) | 2025–26 | ICML 2026; Google Gemma 4; DeepSeek-V4 | https://arxiv.org/abs/2606.04032 · https://arxiv.org/abs/2607.02770 | One projection is both key and value; halves stored KV | Gemma 4 26B-A4B/31B (global layers), DeepSeek-V4-Pro/Flash, V4.1-Flash | 02 (new card) |
| M9 | MLA in practice: MQA-mode decode, head-count choice, adopters | 2024–26 | DeepSeek; Moonshot | https://arxiv.org/abs/2512.02556 · https://arxiv.org/abs/2507.20534 | Absorbed MLA = MQA (576/512); heads cost FLOPs, not cache | DeepSeek-V2/V3/V3.2, Kimi K2/K2.5, Kimi Linear, GLM-5, Mistral Large 3 | 03 |
| M10 | Decode arithmetic-intensity lens | 2025 | Princeton (ICML'25 ES-FoMo workshop oral); StepFun | https://arxiv.org/abs/2505.21487 · https://arxiv.org/abs/2507.19427 | FLOPs per KV byte: MHA≈1, GQA≈g, MQA≈H_q, MLA≈2H_q | Step-3 design rationale | 02 |
| M11 | Cross-layer KV sharing in production (YOCO, KV-shared top layers, CLA2) | 2024–26 | NeurIPS 2024 (YOCO, CLA); Google; Tencent | https://arxiv.org/abs/2405.05254 · https://arxiv.org/abs/2411.02265 | Upper layers reuse one lower layer's KV; prefill can skip them | Hunyuan-Large, Gemma 3n, Gemma 4 E2B/E4B, Phi-4-mini-flash-reasoning, DeepSeek-V4.1-Flash | 09 |
| **SHOULD** |
| S1 | Length-dependent logit temperature | 2023–25 | YaRN; Meta; arXiv (SSMax) | https://arxiv.org/abs/2501.19399 | Multiply q by f(position or N) against attention fading | Llama 4 (NoPE layers), Mistral Large 3, DeepSeek-V3 (mscale²), YaRN users | NEW: Position |
| S2 | Logit soft-capping; QK-Clip | 2024 / 2025 | Google Gemma 2; Moonshot Kimi K2 | https://arxiv.org/abs/2408.00118 · https://arxiv.org/abs/2507.20534 | tanh-cap logits, or shrink W_q,W_k of heads whose max logit > τ | Gemma 2 (attn 50 / final 30), Gemma 3n/4 (final 30 only), Kimi K2 (τ=100) | NEW: Stability |
| S3 | ALiBi | 2021 | ICLR 2022 | https://arxiv.org/abs/2108.12409 | Per-head linear distance penalty replaces position embeddings | BLOOM, MPT (**UNVERIFIED** cfg) | NEW: Position (history) |
| S4 | Forgetting Transformer (FoX) | 2025 | ICLR 2025 | https://arxiv.org/abs/2503.02130 | Add cumulative log forget-gate bias to softmax logits | research only | NEW: Stability (bridge to 04/05) |
| S5 | Differential Transformer | 2024 | ICLR 2025 oral (MSR/Tsinghua) | https://arxiv.org/abs/2410.05258 | Output = difference of two softmax maps; cancels common-mode noise | Phi-4-mini-flash-reasoning | NEW: Stability |
| S6 | MFA (Multi-matrix Factorization Attention) | 2024/25 | StepFun | https://arxiv.org/abs/2412.19255 | MQA with many wide heads + low-rank query; tuned to hardware AI | Step-3 | 02 |
| S7 | GTA / GLA | 2025 | Princeton (workshop) | https://arxiv.org/abs/2505.21487 | Tie K/V per group; split MLA latent into TP-shardable heads | none major found | 02/03 |
| S8 | MHA2MLA / TransMLA (conversion) | 2025 | ACL 2025; PKU | https://arxiv.org/abs/2502.14837 · https://arxiv.org/abs/2502.07864 | Convert trained MHA/GQA to MLA: partial-RoPE + SVD + short fine-tune | research checkpoints | 03 |
| S9 | LongRoPE | 2024 | ICML 2024 (Microsoft) | https://arxiv.org/abs/2402.13753 | Searched per-pair rescale factors; progressive 2M extension | Phi-3-mini-128k, Phi-4-mini (cfg `longrope`) | NEW: Position |
| S10 | 2026 attention-layer anatomy (V4 / Gemma 4 / gpt-oss) | 2025–26 | DeepSeek; Google; OpenAI | https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py | Composite: K=V, sinks, QK-norm, partial RoPE, SWA, grouped low-rank out-proj | DeepSeek-V4, Gemma 4, gpt-oss | NEW chapter closer |
| **SKIP** |
| K1 | Tensor Product Attention (T6) | 2025 | NeurIPS 2025 spotlight | https://arxiv.org/abs/2501.06425 | Factorise Q/K/V into contextual tensor products | none | — |
| K2 | Mixture-of-Head attention | 2024 | ICML 2025 | https://arxiv.org/abs/2410.11842 | Route tokens to 50–90% of heads | none major | — |
| K3 | Multi-Token Attention | 2025 | Meta (arXiv) | https://arxiv.org/abs/2504.00927 | Convolve over queries/keys/heads before softmax | none | — |
| K4 | Stick-breaking attention | 2024 | IBM/Mila (ICLR 2025 **UNVERIFIED**) | https://arxiv.org/abs/2410.17980 | Stick-breaking weights replace softmax; built-in recency | none | — |
| K5 | Selective Attention | 2024 | ICLR 2025 (Google) | https://arxiv.org/abs/2410.02703 | Tokens mask later attention to themselves; smaller buffer | none | — |
| K6 | Dual Chunk Attention | 2024 | arXiv (ICML 2024 **UNVERIFIED**) | https://arxiv.org/abs/2402.17463 | Intra/inter-chunk relative positions, training-free extension | Qwen3 inference (report §3.2) | tip in M2 |
| K7 | Diff Transformer V2 | 2026 | MSR blog | **UNVERIFIED** (secondary only) | Extra Q params, no per-head norm | — | tip in S5 |
| K8 | Character.AI MQA + 1-in-6 global + CLA | 2024 | blog | **UNVERIFIED** (primary post unreachable; MixAttention https://arxiv.org/abs/2409.15012 studies it) | — | Character.AI | omit or footnote |
| — | Gated (output-gated) attention | 2025 | NeurIPS 2025 Best Paper | https://arxiv.org/abs/2505.06708 | already in ch08 | Qwen3-Next, Qwen3.5 (`attn_output_gate: true`) | cross-ref from M6 |

---

## 3. Card data

### M1 · RoPE spectrum, base and dual bases (2021 RoFormer → 2026)
- url: https://arxiv.org/abs/2104.09864 ; base-raising ("ABF") as used in Qwen3 §3.2 https://arxiv.org/abs/2505.09388
- one-line idea: each of the d_h/2 coordinate pairs rotates at its own frequency; the base b sets the slowest wavelength, so b must grow with context.
- key formula:
  `θ_i = b^{-2i/d_h},  i=0..d_h/2-1;   (R_t q)^⊤(R_j k) = q^⊤ R_{j-t} k;   λ_i = 2π/θ_i = 2π b^{2i/d_h}`
- concrete numbers (cfg `rope_theta`): DeepSeek-V3/V3.2, Kimi Linear 10,000; Kimi K2/K2.5 50,000; gpt-oss 150,000; Llama 3.1 & Llama 4 500,000; Qwen3 1,000,000 (report: "increase … from 10,000 to 1,000,000"); MiniMax-M2 & MiMo-V2-Flash (global) 5,000,000; Qwen3-Next, Qwen3.5, Seed-OSS 10,000,000; Mistral Small 3.1 1e9. Dual bases: Gemma 3/4 local 10k / global 1M (Gemma 3 report §2; Gemma 4 cfg `rope_parameters`); MiMo `swa_rope_theta` 10k vs 5M; Step-3.5-Flash per-layer list 5e6 (full) / 1e4 (SWA); DeepSeek-V4 pure-SWA layers base 10k without YaRN vs compressed layers `compress_rope_theta` 160,000 + YaRN.
  Longest wavelength at d_h=128 *(computed)*: b=10k → 54,410 tokens; 5e5 → 2.56M; 1e6 → 5.06M. Shortest always 2π ≈ 6.3 tokens.
- diagram idea: log-y bar strip of 64 pairs (d_h=128) showing λ_i for b=1e4, 5e5, 1e6; horizontal lines at training lengths 4K/8K/32K; shade pairs that complete <1 rotation within L ("NoPE-like / semantic", per Barbero) vs many rotations ("positional"). Side panel: local layer (b=10k, W=1024) vs global layer (b=1M) of Gemma 3, arrow "local never sees distance > W → no scaling needed".
- tips:
  - Pairs with λ_i ≫ L barely rotate during training — the model uses them as semantic channels (Gemma 7B analysis, Barbero et al. ICLR 2025); that motivates p-RoPE (M3).
  - Local/SWA layers only see distances ≤ W, so they keep a small base and skip scaling (DeepSeek-V4 `model.py` L478: "disable YaRN and use base rope_theta in pure sliding-window attention").
  - Raising b is not extrapolation: positions beyond trained L still need scaling (M2) or long-context training.
- verification: cfg fields above (repos: deepseek-ai/DeepSeek-V3, moonshotai/Kimi-K2-Instruct, openai/gpt-oss-120b, unsloth/Llama-3.1-8B-Instruct, Qwen/Qwen3-8B, MiniMaxAI/MiniMax-M2, XiaomiMiMo/MiMo-V2-Flash, Qwen/Qwen3-Next-80B-A3B-Instruct, google/gemma-4-31b-it, stepfun-ai/Step-3.5-Flash); Qwen3 report §3.2; DeepSeek-V4-Pro `inference/model.py` Attention.__init__.

### M2 · RoPE context extension: PI → NTK-aware → YaRN; Llama-3 "llama3" type (2023)
- url: PI https://arxiv.org/abs/2306.15595 (Meta, 2023) · YaRN https://arxiv.org/abs/2309.00071 (ICLR 2024 poster)
- one-line idea: to serve L′ = sL tokens, slow (low-frequency) pairs are interpolated by s, fast pairs are left alone, and logits get a mild temperature.
- key formula (YaRN §3.2–3.3, Eqs. 10–15):
  `r_i = L/λ_i,  γ(r)=clip((r-α)/(β-α),0,1),  θ'_i = (1-γ(r_i)) θ_i/s + γ(r_i) θ_i;   logits ×(0.1 ln s + 1)^2`
  (PI: θ'_i = θ_i/s for all i. NTK-aware: b′ = b·s^{d_h/(d_h-2)}, YaRN App. A.2 Eq. 22.)
- concrete numbers (cfg `rope_scaling`):
  | model | type | factor | from → to | extra |
  |---|---|---|---|---|
  | Qwen3 | yarn (model card/report App. A.1.1) | 4 | 32,768 → 131,072 | cfg ships `rope_scaling: null`; user enables |
  | DeepSeek-V3/V3.2 | yarn | 40 | 4,096 → 163,840 | β_fast 32, β_slow 1, mscale 1.0 |
  | Kimi K2 / K2.5 | yarn | 32 / 64 | 4,096 → 131,072 / 262,144 | θ 50,000 |
  | gpt-oss-20b/120b | yarn | 32 | 4,096 → 131,072 | θ 150,000, `truncate: false` |
  | OLMo 3 7B | yarn | 8 | 8,192 → 65,536 | `attention_factor` 1.2079 = 0.1 ln 8 + 1 |
  | DeepSeek-V4-Pro/Flash | yarn | 16 | 65,536 → 1,048,576 | compressed layers only |
  | Mistral Large 3 | yarn (`params.json`) | 36 | 8,192 → 294,912 | `apply_scale: false` (temperature via S1 instead) |
  | Llama 3.1 (8B/405B) | llama3 | 8 | 8,192 → 131,072 | low_freq_factor 1, high_freq_factor 4 |
  | Llama 4 Scout | llama3 | 16 | 8,192 → (10M max_pos) | low = high = 1 |
  | Gemma 3 | linear (= PI) | 8 | 32K → 128K | global layers only (report §2) |
  YaRN ramp for gpt-oss (d_h 64, b 1.5e5, L 4096, α 1, β 32) *(computed)*: pairs 0–8 unchanged, ≈18–31 divided by 32, ramp in 9–17. llama3 on Llama 3.1 (d_h 128, b 5e5, L 8192) *(computed)*: 29 pairs unchanged (λ<2048), 29 pairs ÷8 (λ>8192), 6 blended.
  Temperature √(1/t) = 0.1 ln s + 1 *(computed)*: s=4 → 1.139, 8 → 1.208, 32 → 1.347, 40 → 1.369, 64 → 1.416.
- diagram idea: x = pair index i (0→d_h/2), y = log(θ′_i/θ_i); four curves on one plot — PI flat at −log s; NTK-aware smooth decline; YaRN piecewise ramp between r=β and r=α; llama3 ramp between r=4 and r=1. Annotate the gpt-oss pair boundaries (8 / 18). Inset: logit multiplier vs s.
- tips:
  - "llama3" rope = NTK-by-parts with α=1, β=4 and **no** temperature (HF `_compute_llama3_parameters`).
  - YaRN's temperature costs nothing: it is folded into cos/sin (scales both q and k), YaRN §3.3.
  - PI alone degraded beyond s≈8 even after fine-tuning (YaRN §3.1); YaRN claims 10× fewer tokens / 2.5× fewer steps than prior methods (abstract).
  - Gotcha: dynamic (inference-time) scaling with a KV cache must cache pre-RoPE keys (YaRN §3.4). Does nothing for KV memory.
- verification: YaRN PDF §2.2–3.4 + App. A.2/A.3 (Eqs. 8–15, 21–23) read in full text; PI abstract; HF `modeling_rope_utils.py` (`_compute_llama3_parameters`, `get_mscale`); all cfgs above; Mistral Large 3 `params.json`.

### M3 · Partial RoPE and p-RoPE (GPT-J 2021 → p-RoPE ICLR 2025 → Gemma 4 2026)
- url: https://arxiv.org/abs/2410.06205 (Barbero et al., Google DeepMind, ICLR 2025)
- one-line idea: rotate only d_r < d_h dims; the rest match on content alone (a per-head NoPE channel).
- key formula:
  `e_{tj} = [(R_t q^{rot}_t)^⊤ R_j k^{rot}_j + (q^{nope}_t)^⊤ k^{nope}_j]/√d_h,   d_r = p·d_h`
  Partial (NeoX/HF default): θ_i = b^{-2i/d_r} over d_r dims. p-RoPE / Gemma 4 "proportional": θ_i = b^{-2i/d_h} for i < p·d_h/2, θ_i = 0 otherwise (keeps the fastest p of the full spectrum).
- concrete numbers (cfg): GPT-J-6B `rotary_dim` 64 of d_h 256; GPT-NeoX-20B `rotary_pct` 0.25 (24 of 96); GLM-4.5 `partial_rotary_factor` 0.5 (64/128); Qwen3-Next & Qwen3.5 0.25 (64/256); MiniMax-M2 `rotary_dim` 64/128; MiMo-V2-Flash 0.334 (64 of qk dim 192); Phi-4-mini 0.75 (96/128); Gemma 4 global layers p=0.25 of 512 (report §2: "p-RoPE at p = 0.25"); MLA decoupled lane 64 of 192 qk dims; DeepSeek-V4 64 of 512.
  Barbero Table 2 (Gemma 2B, 8K ctx, Wiki ppl): NoPE 4.859 · RoPE θ=10k 4.463 · RoPE θ=500k 4.449 · 0.25-RoPE 4.530 · **0.75-RoPE 4.441**.
- diagram idea: one head vector as a 128-cell bar: left block coloured "rotated" (with tiny clock icons spinning fast→slow), right block grey "NoPE"; three rows: NeoX partial (full spectrum squeezed into d_r), p-RoPE (only the fast end kept), MLA/V4 (separate 64-d rope lane). Under each: which frequencies survive on a shared log-frequency axis.
- tips:
  - Partial RoPE is what makes MLA/K=V caching cheap: un-rotated dims are position-independent, so they can be absorbed or shared (MLA: content via latent; Gemma 4: V reuses K's un-rotated 75%).
  - NeoX-style partial and p-RoPE are **different**: the former keeps the full frequency range on fewer dims, the latter deletes the slowest frequencies (HF `_compute_proportional_rope_parameters` vs default).
  - Barbero's "reversed" control (drop high freqs) is worse — fast pairs carry positional heads.
  - Does not remove the need for scaling on rotated dims.
- verification: cfgs (EleutherAI/gpt-j-6b, EleutherAI/gpt-neox-20b, zai-org/GLM-4.5, Qwen/Qwen3-Next-80B-A3B-Instruct, Qwen/Qwen3.5-397B-A17B, MiniMaxAI/MiniMax-M2, XiaomiMiMo/MiMo-V2-Flash, microsoft/Phi-4-mini-instruct, google/gemma-4-31b-it); Barbero HTML §6.1 Table 2; Gemma 4 report §2; HF `modeling_rope_utils.py` L193–261; GLM-4.5 report §2.1.

### M4 · NoPE layers: iRoPE (Llama 4) and RNoPE-SWA (Cohere) (2023 → 2025)
- url: NoPE https://arxiv.org/abs/2305.19466 (NeurIPS 2023) · RNoPE https://arxiv.org/abs/2501.18795 (Cohere) · Llama 4 blog https://ai.meta.com/blog/llama-4-multimodal-intelligence/
- one-line idea: causal masking alone leaks position, so some global layers drop RoPE entirely; RoPE stays in local layers where its recency bias helps.
- key formula:
  `ℓ∈NoPE: e_{tj}=q_t^⊤k_j/√d_h  (global, causal);   ℓ∈RoPE: e_{tj}=(R_tq_t)^⊤R_jk_j/√d_h + m^{local}_{tj}`
- concrete numbers:
  - Llama 4 Scout/Maverick cfg: 48 layers, `no_rope_layers` = [1,1,1,0]×12 → 36 RoPE + 12 NoPE; RoPE layers are `chunked_attention` with `attention_chunk_size` 8192 (mask: ⌊j/8192⌋ = ⌊t/8192⌋), NoPE layers full attention; 40 Q / 8 KV, d_h 128. Scout `use_qk_norm` true (RoPE layers only), Maverick false.
  - Command A design (RNoPE §3): 3 SWA layers (window 4,096, RoPE) : 1 full layer (NoPE), QK-norm removed; ratios 1:1, 1:3, 1:7 ablated, 1:3 chosen. HF `cohere2` code applies RoPE only when `sliding_window` is set; default pattern 4.
  - SmolLM3-3B cfg: `no_rope_layer_interval` 4 → 27 RoPE + 9 NoPE, all global, 16 Q / 4 KV.
  - Kimi Linear: all MLA layers NoPE (§4); Granite-4.0-H-Small cfg `position_embedding_type: nope` (4 attention + 36 Mamba2 layers).
  - KV at 128K, Llama 4 Scout *(computed)*: 12 global × 131,072 × 2,048 e + 36 chunked × 8,192 × 2,048 e = 7.1 GiB vs 24 GiB if all 48 were global.
- diagram idea: 4-layer repeating block as a column; three layers drawn with a narrow band mask (chunk boxes for Llama 4 / diagonal band for Command A) + rotating-clock icon; fourth layer a full lower triangle with "no clock". Right side: RNoPE attention-mass bars (Begin/Needle/End) for NoPE vs RoPE layers showing "NoPE retrieves, RoPE stays local".
- tips:
  - Division of labour emerges by itself: NoPE layers carry long-range retrieval, RoPE layers recency (RNoPE §2.2.3).
  - Raising θ in RoPE layers of an RNoPE model *hurt* long-context retrieval (needle attention in NoPE layers 0.0765 → lower; RNoPE §2.2.3) — opposite of pure-RoPE practice.
  - NoPE global layers can flatten with length → Llama 4 adds a temperature on exactly those layers (S1).
  - Pure NoPE lags RoPE in-distribution (RNoPE Table 2; Barbero Table 2) — use it as a minority layer type.
- verification: Llama 4 cfgs (unsloth mirrors of meta-llama/Llama-4-Scout/Maverick), HF `configuration_llama4.py` (no_rope_layers semantics, layer_types), `masking_utils.chunked_overlay`; RNoPE PDF §2–3 (pdftotext); Llama 4 blog ("i" = interleaved layers without positional embeddings); SmolLM3, Granite, Kimi Linear cfgs; Kimi Linear §4.

### M5 · Local:global sliding-window interleave (Mistral 2023 → Gemma 3 → gpt-oss → 2026)
- url: Mistral 7B https://arxiv.org/abs/2310.06825 · Gemma 2 https://arxiv.org/abs/2408.00118 · Gemma 3 https://arxiv.org/abs/2503.19786 · gpt-oss https://arxiv.org/abs/2508.10925 · MiMo-V2-Flash https://arxiv.org/abs/2601.02780
- one-line idea: local layers attend to the last W tokens (bounded rolling cache); a minority of global layers keep the growing cache, so cache ≈ (global fraction) × dense.
- key formula:
  `m^{loc}_{tj} = 0 if t-W < j ≤ t else -∞;   M ≈ B·s·[ N Σ_{ℓ∈G} c_ℓ + min(N,W) Σ_{ℓ∈Loc} c_ℓ ],  c_ℓ = 2H_kv d_h`
- concrete numbers (cfg unless noted):
  | model | layers | local:global | W | KV heads × d_h | note |
  |---|---|---|---|---|---|
  | Mistral 7B | 32 | all SWA | 4,096 | 8 × 128 | receptive field ≈ 4096×32 ≈ 131K; rolling slot i mod W; 8× cache cut at 32K (paper §2) |
  | Gemma 2 27B | 46 | 1:1 | 4,096 (global 8,192) | 16 × 128 | report §2 |
  | Gemma 3 27B | 62 | 5:1 (52/10) | 1,024 | 16 × 128 | `sliding_window_pattern` 6; ratio ablation flat "even with 7-to-1" (§5.2) |
  | Gemma 3n E4B | 35 | 4:1 (28/7) | 512 | 2 × 256 | + KV sharing (M11) |
  | Gemma 4 31B / 26B-A4B | 60 / 30 | 5:1 | 1,024 | local 16/8 × 256; global 4/2 × 512 (K=V) | E2B 4:1, W 512 (report §2) |
  | gpt-oss-120b / 20b | 36 / 24 | 1:1 | 128 | 8 × 64 | + sinks; "banded window … 128 tokens" (§2.2) |
  | Command A | — | 3:1 | 4,096 | — | global layers NoPE (M4) |
  | OLMo 3 7B | 32 | 3:1 (24/8) | 4,096 | 32 × 128 (MHA) | last layer always full (report) |
  | MiMo-V2-Flash | 48 | 39 SWA / 9 GA | 128 | SWA 8, GA 4 KV heads; qk 192 / v 128 | + sinks on SWA layers |
  | Step-3.5-Flash | 45 (48 `layer_types`, **UNVERIFIED** why) | 3:1 | 512 | 8 × 128; SWA layers 96 Q heads vs 64 | head-wise gate |
  | Ministral 8B | 36 | 3:1 (27/9) | **UNVERIFIED** (cfg says 32,768) | 8 × 128 | — |
  | Llama 4 | 48 | 3:1 chunked | chunk 8,192 | 8 × 128 | non-overlapping chunks, not a sliding band |
  KV at 128K *(computed, BF16)*: Gemma 3 27B 10.4 GiB vs 62 GiB all-global; gpt-oss-120b 4.5 vs 9.0 GiB; MiMo-V2-Flash 2.8 vs 15 GiB (all-GA with 4 KV heads); Mistral 7B at 32K 0.5 vs 4 GiB (= the paper's 8×).
  Gemma 3 report §5.2: KV overhead ≈60% (global-only) → <15% with 1024 window at 32K (ratio wording in text inconsistent: says 1:3).
- diagram idea: N×N causal grid with stacked layer strips: thin diagonal band (local) ×5, full triangle (global) ×1; beside it a stacked bar of KV bytes vs N (local layers flatten at W, global grow linearly); small table of the 6 production ratios/windows as coloured pills. Optional: ring buffer with slot = t mod W.
- tips:
  - Effective receptive field grows with depth (≈W per local layer), but exact long-range retrieval only happens in global layers.
  - Trend is to *smaller* windows with sinks: 4096 → 1024 → 128 (gpt-oss, MiMo). MiMo: W=128+sink beats W=512+sink on RULER-32K (89.4 vs 84.7) and MRCR (34.4 vs 19.6) (§2.2.1 Tables 2–3).
  - Prefill FLOPs also drop (local layers are O(NW)), but global layers stay O(N²).
  - Not everyone uses it: Qwen3, Qwen3-Next/3.5, Mistral Small 3.1/NeMo, MiniMax-M2, DeepSeek-V3 all-global (cfg `use_sliding_window: false` / `sliding_window: null`).
- verification: cfgs (mistralai/Mistral-7B-v0.1, unsloth/gemma-2-27b-it, unsloth/gemma-3-27b-it, unsloth/gemma-3n-E4B-it, google/gemma-4-31b-it, google/gemma-4-26b-a4b-it, google/gemma-4-E2B-it, openai/gpt-oss-*, allenai/Olmo-3-1025-7B, XiaomiMiMo/MiMo-V2-Flash, stepfun-ai/Step-3.5-Flash, mistralai/Ministral-8B-Instruct-2410); Mistral §2; Gemma 2 §2; Gemma 3 §2, §5.2; gpt-oss §2.2; OLMo 3 report (arXiv 2512.13961) via search excerpt; MiMo §2.1–2.2.

### M6 · Attention sinks → learned sink logit (2023 → 2025–26)
- url: StreamingLLM https://arxiv.org/abs/2309.17453 (ICLR 2024) · softmax₁ https://www.evanmiller.org/attention-is-off-by-one.html (2023-07-24) · gpt-oss https://arxiv.org/abs/2508.10925 · When sinks emerge https://arxiv.org/abs/2410.10781 (ICLR 2025 spotlight)
- one-line idea: softmax must sum to 1, so heads dump unneeded mass on early tokens; give each head an explicit "attend to nothing" logit instead.
- key formula:
  `α_{tj} = exp(e_{tj}) / ( exp(z_h) + Σ_{r≤t} exp(e_{tr}) ),   Σ_j α_{tj} = 1 - exp(z_h)/(…) < 1`
  (softmax₁: z_h ≡ 0. FlashMLA: `out ×= exp(lse)/(exp(lse)+exp(sink))`.)
- concrete numbers: StreamingLLM keeps **4** initial tokens + rolling window; positions re-indexed within cache; stable to 4M tokens; up to 22.2× faster than sliding-window recompute (abstract, §3.2). gpt-oss: one learned scalar per head (64 per layer, `self.sinks = nn.Parameter(num_attention_heads)`), every layer. MiMo-V2-Flash cfg: `add_swa_attention_sink_bias: true`, `add_full_attention_sink_bias: false`. MiMo ablation (Table 2, W=128): MMLU 54.9 without sink → 58.3 with sink (all-GA 57.3); BBH 52.4 → 56.1. DeepSeek-V4 §2.3.3 Eq. 27: learnable z′_h per head; `attn_sink` in model.py.
- diagram idea: two attention rows over the same 8 tokens: (a) vanilla — big bar on token 0, rest tiny; (b) with sink — an extra grey "∅" column outside the sequence absorbing mass, token 0 no longer special; under (a) a SWA ring buffer showing why evicting token 0 breaks streaming (StreamingLLM), under (b) the buffer is free to evict.
- tips:
  - Sinks act like a key bias that stores excess attention without contributing value (Gu et al. ICLR 2025); unnormalised (sigmoid) attention removes them up to 1B params.
  - Learned sinks are what make tiny windows (128) viable in gpt-oss and MiMo.
  - Alternative fix: output gating after SDPA ("attention-sink-free", ch08 gated attention).
  - Cost: one scalar per head; kernels need a sink argument (FlashMLA `attn_sink`, FlashAttention-3 path) — **UNVERIFIED** for FA3 specifics.
- verification: StreamingLLM HTML §1, §3.2, §4; gpt-oss §2.2 ("learned bias in the denominator of the softmax"); HF `modeling_gpt_oss.py` L243–251 (concat sink logit, softmax, drop column); MiMo HTML §2.2 Eqs. 1–4, Tables 2–4; MiMo cfg; DeepSeek-V4 HTML §2.3.3; FlashMLA README (`attn_sink`).

### M7 · QK-norm (2020 → standard by 2025)
- url: Wortsman et al. ICLR 2024 https://arxiv.org/abs/2309.14322 (§3.1.1 qk-layernorm vs attention logit growth); origin Henry et al. 2020 / Dehghani et al. 2023 (ViT-22B) — venues **UNVERIFIED**
- one-line idea: normalise each head's q and k (RMSNorm with gain) before the dot product so logits cannot blow up during training.
- key formula:
  `q̂ = g_q ⊙ q/RMS(q),  k̂ = g_k ⊙ k/RMS(k),  e = q̂^⊤k̂·σ  ⇒  |e| ≤ σ d_h ‖g_q‖_∞‖g_k‖_∞`  (σ = 1/√d_h usually)
- concrete numbers / variants (code + cfg):
  - Per-head norm (dim d_h): Qwen3 (`q_norm = RMSNorm(head_dim)`; report: "remove QKV-bias … introduce QK-Norm"), Gemma 3, Gemma 4, Hunyuan-A13B (`use_qk_norm`).
  - Whole-projection norm (dim H·d_h): OLMo 2, OLMo 3, MiniMax-M2 (`qk_norm_type: per_layer`).
  - Gain-free: Llama 4 Scout L2/RMS norm on RoPE layers only (Maverick none); DeepSeek-V4 q per-head RMS (no gain) + `kv_norm` RMSNorm.
  - Scale after QK-norm: Gemma 3 `query_pre_attn_scalar` 168 (=5376/32) → 1/√168; **Gemma 4 `scaling = 1.0`** (no 1/√d_h) plus V-norm (gain-free).
  - GLM-4.5 uses QK-norm; GLM-4.5-Air does not (report Table 1).
  - Counter-evidence: Cohere RNoPE found QK-norm flattens attention and hurts needle retrieval → removed for Command A (RNoPE §2.2, §3).
- diagram idea: histogram of max attention logit over training steps: no-norm curve drifting to ~10³ with "one-hot collapse" annotation vs QK-norm flat; inset showing q and k vectors projected onto a sphere of radius √d_h·g. Second inset: three norm placements (per head / whole projection / gain-free) as small block diagrams.
- tips:
  - Fixes logit growth, not output-logit divergence (that is z-loss / final soft-cap).
  - Incompatible with absorbed MLA: keys are never materialised at decode → Kimi K2 uses QK-Clip instead (S2).
  - Bound makes the gain g (and σ) the effective temperature; Gemma 4 drops σ entirely.
  - Possible long-context cost (Cohere). Not a positional method.
- verification: Wortsman §3.1.1 (ar5iv); Qwen3 report §2; HF `modeling_qwen3.py` L229, `modeling_olmo2.py` L213, `modeling_olmo3.py` L161, `modeling_gemma3.py` L307–328, `modeling_gemma4.py` L913–949, L1166–1238, `modeling_minimax_m2.py` L292, `modeling_llama4.py` L331–356; cfgs; GLM-4.5 report §2.1/Table 1; DeepSeek-V4 §2.3.3 + model.py L497–503.

### M8 · K=V tying (2025 GTA → ICML 2026 study → Gemma 4 & DeepSeek-V4)
- url: Kayyam et al., ICML 2026 https://arxiv.org/abs/2606.04032 · Gemma 4 report https://arxiv.org/abs/2607.02770 · DeepSeek-V4 https://arxiv.org/abs/2606.19348
- one-line idea: values reuse the key projection, so each position stores one vector per KV head instead of two.
- key formula:
  `u_j = W_{KV} h_j;  k_j = RoPE_p(norm_g(u_j)),  v_j = norm(u_j);   cache: 2H_kv d_h → H_kv d_h (+ rotated part of v)`
  DeepSeek-V4 (one 512-d entry c_j used as both): `o_t = Σ_j α_{tj} c̃_j,  o_t^{[rope]} ← R_{-t} o_t^{[rope]}` (de-rotation makes the value's 64 rotated dims relative).
- concrete numbers:
  - Gemma 4 cfg `attention_k_eq_v`: true for 31B (global: 4 KV × 512) and 26B-A4B (2 × 512); false for E2B/E4B. Report §2: values = keys in global layers except E2B/E4B; with p-RoPE "up to 37.5%" smaller global KV. *(My derivation, UNVERIFIED as theirs: store K (512) + only the 25% rotated part of V (128) = 640 vs 1024 → −37.5%.)*
  - Gemma 4 31B KV at 256K *(computed)*: 10 global layers × 4 × 512 × (2 or 1.25) + 50 local × 16 × 256 × 2 × 1024 → 20.8 GiB (K and V stored) vs 13.3 GiB (K + rotated V only).
  - DeepSeek-V4-Pro cfg: 128 Q heads, `num_key_value_heads` 1, `head_dim` 512, `qk_rope_head_dim` 64; Flash 64 Q heads; V4.1-Flash 64 Q heads, same 512/64.
  - Kayyam (300M/1.2B, 10B tokens): Q-K=V −50% KV at +3.1% ppl; + GQA-4 → −87.5%; + MQA → −96.9%.
  - GTA (Zadouri §3.3.1): tied state = value; key = [first d_h/2 of tied state ; separate single-head RoPE half-key]; GTA-8 1.5 d_h vs GQA-8 2 d_h per token at TP=8 (§6).
  - MFA-KR (StepFun): "repurposing the key cache as value" (abstract).
- diagram idea: three per-token cache cells side by side: GQA (K block + V block), GTA (one tied block + thin shared rope strip), V4 (one 512 block, last 64 hatched "rotated", with an arrow on the output "rotate back by −t"). Annotate bytes per token per layer under each.
- tips:
  - Works because attention is low-rank and keys/values can share a subspace (Kayyam abstract); quality cost is small but nonzero.
  - RoPE is the obstacle: a rotated vector is a poor value → keep values un-rotated (Gemma 4: separate norms, no RoPE on V) or de-rotate the output (V4).
  - Doubles decode arithmetic intensity at fixed H_q/H_kv (m_kv = 1 in M10).
  - Halves storage, not the number of positions read.
- verification: Gemma 4 cfgs (google/gemma-4-31b-it, -26b-a4b-it, -E4B-it, -E2B-it); HF `modeling_gemma4.py` L1164–1238 (`use_alternative_attention`, `v_proj None`, `value_states = key_states`), `configuration_gemma4.py` L210–223 (per-layer head_dim/kv-head override); Gemma 4 report §1–2 (via HTML fetch); V4 cfgs + `inference/model.py` L436–543; V4 HTML §2.3.1/2.3.3; Kayyam abstract; Zadouri HTML §3.3.1, §6.

### M9 · MLA in practice: MQA-mode decode, head count, adopters (2024 → 2026)
- url: DeepSeek-V2 https://arxiv.org/abs/2405.04434 · V3.2 https://arxiv.org/abs/2512.02556 · Kimi K2 https://arxiv.org/abs/2507.20534
- one-line idea: after absorption, decode is plain MQA over a 576-d key / 512-d value shared by all H_q heads; H_q is a FLOPs knob, not a cache knob.
- key formula:
  `k̃_j = [c_j; k^R_j] ∈ R^{576},  ṽ_j = c_j ∈ R^{512},  q̃_{t,i} = [W_{UK,i}^⊤ q^C_{t,i}; q^R_{t,i}]   ⇒  MQA(H_q, d_k=576, d_v=512)`
  `cache/layer = (d_c+d_r) = 4.5 d_h  ≡  GQA with 2.25 groups   (V2 Table 1)`
- concrete numbers (cfg):
  | model | layers | H_q | d_c | d_r | qk_nope / v | q_lora | e/t/l |
  |---|---|---|---|---|---|---|---|
  | DeepSeek-V2 | 60 | 128 | 512 | 64 | 128 / 128 | 1536 | 576 |
  | DeepSeek-V3 / V3.2 | 61 | 128 | 512 | 64 | 128 / 128 | 1536 | 576 (+ V3.2 indexer 64 heads × 128, top-2048) |
  | Kimi K2 / K2.5 | 61 | **64** | 512 | 64 | 128 / 128 | 1536 | 576 |
  | GLM-5 | 78 | 64 | 512 | 64 | 192 / 256 | 2048 | 576 (+ DSA indexer 32 heads, top-2048) |
  | Mistral Large 3 (`params.json`) | 61 | 128 | 512 | 64 | 128 / 128 | 1536 | 576 |
  | Kimi Linear (MLA layers, NoPE) | 7 of 27 | 32 | 512 | (64 in cfg; NoPE per §4) | 128 / 128 | none | — |
  Per-token KV across layers *(computed, BF16)*: DeepSeek-V3 61×576×2 B = 68.6 KiB vs Qwen3-235B (94 layers, 4 KV × 128 × 2) 188 KiB vs Llama-3.1-405B (126 layers, 8 KV) 504 KiB.
  Kimi K2 §2.3: 64 → 128 heads = +83% inference FLOPs at 128K; only 0.5–1.2% lower val loss.
- diagram idea: left: training/prefill path (latent → expand per-head K,V → MHA, 128 heads × 192/128 dims); right: decode path (cache only 576-d row; queries absorbed to 576-d; one shared K/V "MQA" box feeding 128 heads). Bottom ruler: per-token per-layer cache for MHA-128 (32,768 e) / GQA-8 (2,048) / MLA (576) / V4 K=V (512).
- tips:
  - Absorption trades bytes for FLOPs: decode intensity ≈ 2H_q (≈242 FLOP/B for V3 in BF16 *(computed)*), which is why DeepSeek runs MHA mode for prefill (V3.2 App. A) and why Kimi halved heads.
  - Head count is invisible in the cache: K2 and V3 store identical 576-element rows.
  - Tensor parallelism: one latent must be replicated on every rank (motivates GLA, S7).
  - MLA spread beyond DeepSeek: Moonshot, Zhipu (GLM-5), Mistral (Large 3) — all keep d_c=512, d_r=64.
- verification: cfgs (deepseek-ai/DeepSeek-V2, -V3, -V3.2, moonshotai/Kimi-K2-Instruct, Kimi-K2.5, zai-org/GLM-5, mistralai/Mistral-Large-3-675B-Instruct-2512 `params.json`, moonshotai/Kimi-Linear-48B-A3B-Instruct); V2 HTML §2.1.3 Table 1 + §3.1.2; V3.2 HTML §2.1 + App. A Fig. 7; K2 HTML §2.3; DeepSeek-V3 `inference/model.py` L428–495.

### M10 · Decode arithmetic intensity (Zadouri–Strauss–Dao 2025; Step-3 2025)
- url: https://arxiv.org/abs/2505.21487 (Princeton; ICML 2025 ES-FoMo workshop oral) · Step-3 https://arxiv.org/abs/2507.19427
- one-line idea: decode attention loads each cached byte once; FLOPs per byte = how many query heads reuse it — head sharing raises it toward the hardware roofline.
- key formula:
  `AI_decode ≈ 2 g_q / m_kv   (g_q = H_q/H_kv;  m_kv = 2 separate K,V · 1 tied)   →  MHA≈1, GQA≈g, MQA≈H_q, MLA≈2H_q, GTA≈2g`
- concrete numbers: Zadouri Table 1 as above. Step-3 §5.1 (8-bit KV): MFA 128, DeepSeek-V3 MLA 512, Qwen3-MoE GQA 32; rooflines (Table 4) H800 591, A800 156, Ascend 910B 175, H20 74 FLOP/B. *(computed, BF16)*: Qwen3-235B 16, gpt-oss 8, Gemma 4 31B global (g=8, tied) 16, Step-3 MFA 64, Kimi K2 ≈121, DeepSeek-V3 ≈242, DeepSeek-V4-Pro (128 heads, tied 512) 256.
- diagram idea: log-x roofline plot (AI vs attainable FLOP/s) with vertical hardware ridge points (H20 74, A800 156, 910B 175, H800 591) and dots for each attention design; arrow "MQA/MLA/K=V move right"; annotate "left of ridge = bandwidth-bound decode".
- tips:
  - This explains choices the cache-size view cannot: Step-3 deliberately sits *below* H800's ridge to run well on H20/910B (§5.1–5.2).
  - Speculative decoding / MTP multiplies query rows → raises AI further (Step-3 §5.2; GLA kernel 2× FlashMLA when query length > 1).
  - AI says nothing about quality or cache capacity; quantised KV doubles AI per byte.
- verification: Zadouri HTML §3.1–3.3 Table 1; Step-3 HTML §4.1–5.2 Tables 2–4; my BF16 arithmetic from cfgs.

### M11 · Cross-layer KV sharing in production (CLA/YOCO 2024 → Gemma 3n 2025 → Gemma 4, V4.1 2026)
- url: YOCO https://arxiv.org/abs/2405.05254 (NeurIPS 2024) · CLA https://arxiv.org/abs/2405.12981 (NeurIPS 2024) · Hunyuan-Large https://arxiv.org/abs/2411.02265 · Gemma 3n guide https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/ · SambaY https://arxiv.org/abs/2507.06607 (NeurIPS 2025)
- one-line idea: layer ℓ reads K,V produced by an earlier source layer src(ℓ); only C < L cache sets are stored.
- key formula:
  `K^{(ℓ)},V^{(ℓ)} := K^{(src(ℓ))},V^{(src(ℓ))};   M_KV = B N s Σ_{ℓ∈Src} 2H_kv d_h  (|Src| = C)`
- concrete numbers:
  - Hunyuan-Large: 64 layers, 80 Q heads, 8 KV heads, CLA sharing every 2 layers (Table 2 caption) → ≈95% saving vs MHA (§2.2.2); check: 2·8·32 / (2·80·64) = 5% *(computed)*. (Text says "every 22 layers" — extraction/typo; Table 2 caption says 2.)
  - Gemma 3n E4B cfg: 35 layers, `num_kv_shared_layers` 15 → layers 20–34 reuse layer 18 (local) / 19 (global) KV (HF `kv_shared_layer_index`); Google: "2x improvement on prefill … compared to Gemma 3 4B".
  - Gemma 4 report §2: KV shared "with ratios of 20/35 and 18/42" for E2B/E4B (cfg `num_kv_shared_layers` 20 / 18); 31B and 26B use 0.
  - YOCO: self-decoder (efficient attention, e.g. SWA) produces one global KV; cross-decoder layers cross-attend to it; prefill can exit early.
  - Phi-4-mini-flash-reasoning (SambaY) cfg: 32 layers, `sliding_window` 512, 40 Q / 20 KV, `mb_per_layer` 2; up to 10× decode throughput at 2K prompt + 32K generation (abstract).
  - DeepSeek-V4.1-Flash cfg: 40 layers, `kv_source_layer_ids` [2, 8, 14, 20], `index_source_layer_ids` [2,8,14,20,24,28,32,36]; 890 B/token global HBM cache (abstract). Exact sharing map **UNVERIFIED**.
- diagram idea: layer stack (35 boxes); bottom 20 coloured "produce KV" with cache icons; top 15 hatched with arrows down to layers 18/19; prompt-token flow arrow stopping at layer 19 ("prefill exit"), only the last token continuing up. Side bar: cache sets C vs L for Hunyuan (32/64), Gemma 3n (20/35), Gemma 4 E2B (15/35), YOCO (1 global + local).
- tips:
  - Big win is prefill: if top layers never produce KV, prompt tokens need not traverse them — only the last token does (YOCO's "early exit"; consistent with Gemma 3n's 2× — the mechanism link is my inference).
  - Reads are not saved: every consumer layer still reads the shared cache.
  - Must be trained in (book ch09 is right); shared layers have no k_proj/v_proj at all (Gemma 4 code).
  - Combines multiplicatively with GQA/MLA/SWA/quantisation (V4.1: cross-layer + FP4).
- verification: YOCO + CLA NeurIPS listings (search); Hunyuan-Large HTML §2.2.2 Tables 1–2; Gemma 3n dev guide; HF `modeling_gemma3n.py` L1177–1233; Gemma 4 report §2 + cfgs + `modeling_gemma4.py` L1171–1186; SambaY abstract + cfg; V4.1-Flash cfg + abstract.

### S1 · Length-dependent logit temperature (YaRN 2023 → Llama 4 2025 → Mistral Large 3)
- url: SSMax https://arxiv.org/abs/2501.19399 · Llama 4 cfg/code · YaRN §3.3
- one-line idea: softmax over more keys flattens ("attention fading"); sharpen by multiplying q with a slowly growing function of length.
- key formula:
  `Llama 4 (NoPE layers): q_t ← q_t·(1 + 0.1·ln(1+⌊(t+1)/8192⌋));   SSMax: q_t ← s_h·ln(n)·q_t;   YaRN: e ← e·(0.1 ln s+1)^2`
- concrete numbers: Llama 4 cfg `attn_temperature_tuning` true, `floor_scale` 8192, `attn_scale` 0.1 → ×1.069 at 8K, ×1.283 at 128K, ×1.486 at 1M, ×1.716 at 10M *(computed)*. Mistral Large 3 `params.json` `llama_4_scaling` {beta 0.1, original 8192} (vLLM `_get_llama_4_scaling`: 1+β·ln(1+⌊t/8192⌋), applied to q in MLA). DeepSeek-V3 inference: `softmax_scale *= mscale²` = (0.1 ln 40 + 1)² = 1.874 when serving > 4096. SSMax: s learned per layer & head (Eq. 2, Eq. 11).
- diagram idea: curve of max softmax weight vs N for fixed logit gap (decays ∝ 1/N) with and without log-N scaling; overlay Llama 4 step function (steps every 8192 tokens).
- tips: applies to positions/lengths not seen in training; Llama 4 applies it only to NoPE (global) layers; HF code cites SSMax as source. Not a substitute for long-context data.
- verification: HF `modeling_llama4.py` L359–366, `configuration_llama4.py` L98–104; Mistral Large 3 `params.json`; vLLM `deepseek_v2.py` L445–451, L623; DeepSeek-V3 `inference/model.py` L434–437; SSMax HTML Eqs. 2, 11.

### S2 · Logit soft-capping (Gemma 2, 2024) and QK-Clip (Kimi K2, 2025)
- url: https://arxiv.org/abs/2408.00118 · https://arxiv.org/abs/2507.20534
- one-line idea: bound logits either at inference-time shape (tanh cap) or by shrinking the weights of offending heads during training.
- key formula:
  `soft-cap: e ← c·tanh(e/c),  c=50 (attn), 30 (final);   QK-Clip: γ_h = min(1, τ/S^h_max),  W^h_q ← √γ_h W^h_q,  W^h_k ← √γ_h W^h_k`
- concrete numbers: Gemma 2 §2 caps 50/30; Gemma 3 "replace the soft-capping of Gemma 2 with QK-norm" (§2); Gemma 3n & Gemma 4 cfg keep `final_logit_softcapping` 30 only. Kimi K2 τ=100; MLA variant: q^C,k^C × √γ, q^R × γ, shared k^R untouched (§2.1, Alg. 1); applied only to heads above τ.
- diagram idea: tanh curve vs identity with cap lines ±50; second panel: per-head max-logit traces with τ=100 ceiling and a clipped head.
- tips: soft-cap needs a kernel hook (blocks plain fused kernels — **UNVERIFIED** claim, Gemma 2 report does not say); QK-Clip leaves the forward pass unchanged and is the MLA-compatible substitute for QK-norm.
- verification: Gemma 2 HTML §2; Gemma 3 HTML §2; cfgs; K2 HTML §2.1.

### S3 · ALiBi (2021, ICLR 2022)
- url: https://arxiv.org/abs/2108.12409
- one-line idea: no position embeddings; subtract a head-specific slope × distance from each logit.
- key formula: `e_{tj} = q_t^⊤k_j/√d_h − m_h (t−j),   m_h = 2^{−8h/H}, h=1..H`
- concrete numbers: H=8 → slopes 1/2 … 1/256 (§3); 1.3B trained on 1024 matches sinusoidal trained on 2048 when tested at 2048; 11% faster, 11% less memory (abstract). Adopters: BLOOM, MPT (**UNVERIFIED** here; MPT cfg fetch failed).
- diagram idea: N×N heatmaps of the bias for 3 heads (steep → shallow slopes).
- tips: ALiBi = FoX with a constant, data-independent forget gate (ln f ≡ −m_h); strong recency bias limits long-range retrieval; superseded by RoPE+scaling in 2024+ models.
- verification: ALiBi HTML §3; ICLR 2022 poster page (search).

### S4 · Forgetting Transformer (FoX, ICLR 2025)
- url: https://arxiv.org/abs/2503.02130
- one-line idea: softmax attention with a recurrent-style scalar forget gate per head, added as a log-decay bias.
- key formula: `f_t = σ(w_h^⊤x_t + b_h),  e_{tj} = q_t^⊤k_j/√d_h + Σ_{l=j+1}^{t} ln f_l = q_t^⊤k_j/√d_h + c_t − c_j`
- concrete numbers: no positional embedding by default; FlashAttention-compatible (store cumsum c in HBM, add c_i − c_j per tile in SRAM); "Pro" block adds output gate, QK-norm, KV-shift (§3).
- diagram idea: causal grid with decay shading that varies by row (data-dependent) next to ALiBi's uniform decay.
- tips: bridges ch04/05 (decay gates) and softmax attention: f≡1 → NoPE softmax; f≡e^{−m} → ALiBi. No production adoption found.
- verification: FoX HTML §3, App. A/E.

### S5 · Differential Transformer (ICLR 2025 oral)
- url: https://arxiv.org/abs/2410.05258
- one-line idea: subtract a second softmax map to cancel attention spread on irrelevant context.
- key formula: `o = (softmax(Q_1K_1^⊤/√d) − λ softmax(Q_2K_2^⊤/√d)) V,  λ = e^{λ_{q1}·λ_{k1}} − e^{λ_{q2}·λ_{k2}} + λ_init,  λ_init = 0.8 − 0.6 e^{−0.3(ℓ−1)}`
- concrete numbers: heads halved (h = d_model/2d) to match params; per-head RMSNorm × (1−λ_init); 6.8B Diff ≈ 11B Transformer (62.2%), 160B tokens ≈ 251B (63.7%) (§3.2). Adopted: Phi-4-mini-flash-reasoning (SambaY abstract). V2 (Jan 2026) **UNVERIFIED**.
- diagram idea: two heatmaps and their difference; bar of mass on "answer" span vs distractors.
- tips: needs two QK products per head (custom kernels in V1); does not reduce KV (V is shared but K doubles in dim split).
- verification: Diff HTML §2.1, §3.2, §3.8.

### S6 · MFA in Step-3 (2025)
- url: https://arxiv.org/abs/2412.19255 · https://arxiv.org/abs/2507.19427
- one-line idea: MQA with many wide heads (64 × 256) and a low-rank query, sized so decode AI matches mid-range accelerators.
- key formula: `q_{t,i} = W_{UQ,i} RMSNorm(W_{DQ} h_t) (rank 2048);  k_j = W_K h_j,  v_j = W_V h_j ∈ R^{256} (1 head each)`
- concrete numbers: Step-3 cfg: 61 layers, 64 heads, `num_attention_groups` 1, `head_dim` 256, `share_q_dim` 2048 → 512 e/t/l (512 B at 8-bit, §6). Decode memory access per token at 8K: Step-3 2.56e8 B vs DSv3 2.88e8 vs Qwen3-MoE 7.89e8 (§4.1). MFA paper: up to 56% / 93.7% KV reduction (baselines ambiguous in abstract).
- diagram idea: head-config strip: Qwen3 (64 Q / 4 KV × 128), DSv3 MLA (128 Q / 1 latent 576), Step-3 MFA (64 Q / 1K+1V × 256), with AI dots from M10.
- tips: sits between GQA and MLA in AI (128 vs 32/512 at 8-bit) — a hardware choice, not a quality claim.
- verification: Step-3 cfg; Step-3 HTML §2, §4.1, §5.1, §6, Table 4; MFA abstract.

### S7 · GTA / GLA (2025)
- url: https://arxiv.org/abs/2505.21487
- one-line idea: GTA ties K and V per group; GLA splits MLA's single latent into h_c latent heads so tensor-parallel ranks each hold one.
- key formula: `GLA: c^{(m)}_j ∈ R^{d_c}, m=1..h_c;  head i reads c^{(⌈i h_c/H_q⌉)};  cache h_c d_c (+rope)`
- concrete numbers: GLA-2 uses d_c = 2d_h per latent → total 4d_h = MLA's; no duplication at TP = h_c; GLA kernel up to 2× FlashMLA in speculative decoding; up to 2× end-to-end latency/throughput (abstract, §3.3.2).
- diagram idea: TP=2 ranks: MLA replicates the 576-d latent on both; GLA puts one latent per rank.
- tips: addresses MLA's TP replication cost (M9 tip); no major model adoption found.
- verification: Zadouri HTML §3.3, §6; abstract.

### S8 · MHA2MLA / TransMLA (2025)
- url: https://arxiv.org/abs/2502.14837 (ACL 2025) · https://arxiv.org/abs/2502.07864
- one-line idea: convert an existing MHA/GQA checkpoint into MLA: drop RoPE from low-contribution dims, SVD the K/V projections into a latent, fine-tune briefly.
- key formula: `[W_K^{nope}; W_V] ≈ U Σ V^⊤  →  W_D = Σ_r V_r^⊤ (d_c rows),  W_{UK},W_{UV} from U_r`
- concrete numbers: MHA2MLA: 0.3–0.6% of data; Llama2-7B KV −92.19%, LongBench −0.5%. TransMLA: LLaMA-2-7B −93% KV, 10.6× at 8K, 6B tokens.
- diagram idea: before/after cache row per token (Llama2-7B 32 heads × 256 → latent + rope strip).
- tips: corrects the "cannot convert" impression in ch02, but conversion = re-training, not lossless.
- verification: both abstracts.

### S9 · LongRoPE (ICML 2024)
- url: https://arxiv.org/abs/2402.13753
- one-line idea: search a separate rescale factor per RoPE pair (and keep early tokens unscaled), then extend progressively.
- key formula: `θ'_i = θ_i / λ_i,  λ_i ∈ searched vector (short vs long sets)`
- concrete numbers: to 2048K tokens; ≤1k steps at 256K; 8× without fine-tune (abstract). Phi-4-mini cfg: `rope_scaling.type` longrope, 48 long factors 1.0 → 47.77, 4,096 → 131,072, partial_rotary 0.75; Phi-3-mini-128k: 48 factors 1.07 → 64.76.
- diagram idea: plot the 48 Phi-4-mini long factors vs pair index next to YaRN's ramp for s=32.
- tips: a data-driven YaRN; two factor sets switch at the original length (short/long).
- verification: abstract; ICML proceedings (PMLR 235); Phi cfgs.

### S10 · 2026 attention-layer anatomy (composite card)
- url: DeepSeek-V4 model.py https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/model.py · Gemma 4 cfg https://huggingface.co/google/gemma-4-31b-it · gpt-oss https://arxiv.org/abs/2508.10925
- one-line idea: modern dense layers stack 5–7 of the cards above; show them in one exploded view.
- key formula (DeepSeek-V4 layer): `q = RMS_head(W_{QB} RMSNorm(W_{QA} x)),  c = RMSNorm(W_{KV} x) ∈ R^{512},  rotate last 64;  o = SinkSoftmax(q c^⊤/√512) c;  derotate;  x' = W_{OB}[W_{OA}^{(g)} o^{(g)}]_{g=1..16}`
- concrete numbers: V4-Pro: 128 Q heads × 512, 1 KV (K=V), window 128 + compressed entries, per-head sink, `o_groups` 16 × `o_lora_rank` 1024 → 7168 (Flash: 8 groups); V4 §2.3.1 rationale: projecting c·n_h directly "will impose a substantial computational burden". Gemma 4 31B: 5:1, local 32 Q/16 KV × 256 (θ 10k), global 32 Q/4 KV × 512, K=V, p-RoPE 0.25 (θ 1M), QK-norm + V-norm, scale 1.0. gpt-oss-120b: 1:1, W 128, 64 Q/8 KV × 64, sink per head, YaRN ×32 (θ 150k), attention bias on.
- diagram idea: three exploded layer diagrams in a row with feature chips (K=V, sink, QK-norm, partial RoPE %, window, KV heads) aligned in a matrix underneath.
- tips: use as the chapter's closing "spot the card" exercise; every chip should link back to its card.
- verification: as in M5–M8.

---

## 4. Suggested restructure
New chapter **"02b · Position, stability and layer mixing"** (between 02 and 03), cards in order:
1. M1 RoPE spectrum & base → 2. M2 PI/NTK/YaRN/llama3 → 3. M3 partial RoPE / p-RoPE → 4. M4 NoPE layers (iRoPE, RNoPE) → 5. S1 length temperature (+S3 ALiBi as a history footnote, S9 LongRoPE as a tip)
6. M7 QK-norm (+S2 soft-cap / QK-Clip) → 7. M6 sinks → 8. M5 local:global interleave → 9. S4 FoX / S5 Diff as "research knobs" (one card) → 10. S10 anatomy closer.
Existing chapters: add M10 + M8 + S6/S7 to **ch02** (after GQA: arithmetic intensity → K=V → MFA/GTA); add M9 + S8 to **ch03** (MQA-mode, head count, adopters, conversion; note V4's move to K=V); add M11 to **ch09** (replace the CLA-only section). Update matrix rows + glossary "attention sink" accordingly.
