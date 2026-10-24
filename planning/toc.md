# Table of contents and ownership (new edition)

16 chapters of mechanism cards. File names are final, so cross-link with them. Research notes are in `planning/research/` (A = dense/KV, B = recurrent, C = sparse, D = hybrid/systems). The card format is defined in `planning/card-style.md`.

| # | File | Title | Owner | Cards (planned, writer may refine within the chapter's scope) |
| --- | --- | --- | --- | --- |
| 00 | `00-map.md` | The map | coordinator | cost ledger (prefill N² vs decode bytes) · the levers map · timeline 2014–2026 · colour + notation legend |
| 01 | `01-positions.md` | Positions | W1 | RoPE spectrum & base (ABF, dual local/global bases) · context extension (PI → NTK → YaRN / llama3; LongRoPE tip) · partial RoPE / p-RoPE · NoPE layers (iRoPE, RNoPE) · length-dependent logit temperature (ALiBi as history tip) |
| 02 | `02-stable-softmax.md` | Stable softmax | W1 | QK-norm (+ soft-capping, QK-Clip) · attention sinks → learned sink logit (gpt-oss, MiMo, V4) · gated attention (output gate, Qwen3-Next) · research knobs: FoX, Differential Transformer (one card) · 2026 attention-layer anatomy closer (V4 / Gemma 4 / gpt-oss) |
| 03 | `03-kv-heads.md` | Share KV heads | W2 (cards 1–3 done by coordinator; do not rewrite them) | MQA/GQA · real-model bytes · decode intensity · **add:** K=V tying (Gemma 4, V4) · MFA / GTA (beyond GQA) · cross-layer sharing (CLA, YOCO, KV-shared top layers) |
| 04 | `04-latent-kv.md` | Latent KV | W2 | MLA records · weight absorption = MQA-mode decode (576/512) · decoupled RoPE lane · head-count choice (V3 128 vs Kimi K2 64: heads cost FLOPs, not cache) · converting MHA/GQA → MLA (TransMLA / MHA2MLA) |
| 05 | `05-fixed-patterns.md` | Fixed patterns | W3 | Sparse Transformer → Longformer/BigBird (one history card) · sliding window + interleaved global layers in production (Mistral, Gemma 2/3, gpt-oss, MiMo, V4-Flash) · chunked local attention (Llama 4) · StreamingLLM: window + sink tokens |
| 06 | `06-sparse-reads.md` | Sparse reads | W3 | "what shrinks" 2×2 opener · training-free prefill (MInference → XAttention/FlexPrefill) · Quest (page bounds) · MoBA · NSA · InfLLM-v2 · DSA (indexer numbers) · 2026 indexers MSA/QSA + IndexCache (indexer cost) |
| 07 | `07-kv-eviction.md` | Evict from the cache | W4 | H2O · SnapKV · PyramidKV/Ada-KV budgets · KVzip · DuoAttention/RazorAttention · DMC → DMS learned eviction · pitfalls (irreversibility) |
| 08 | `08-fewer-records.md` | Compress the sequence | W4 | Compressive Transformer → gist tokens / Activation Beacon · NSA compression branch as bridge · CSA (overlap, top-k) · HCA · V4 layer layout & numbers |
| 09 | `09-linear-state.md` | Linear attention & SSMs | W5 | linear attention as RNN · RetNet (fixed decay) · GLA / per-channel decay · Mamba → Mamba-2 (SSD duality) · Lightning attention / RWKV / xLSTM / HGRN2 / Mamba-3 per research B · chunkwise parallel form |
| 10 | `10-delta-rule.md` | The delta rule | W5 | DeltaNet · Gated DeltaNet · KDA · RWKV-7 generalized delta · DeltaProduct / PaTH (Householder products) · frontier: GDN-2, FG²-GDN, KDN · transition-matrix taxonomy |
| 11 | `11-test-time-learning.md` | Memory as test-time learning | W6 | "update = one SGD step" unifying view · TTT layers · Titans · ATLAS / MesaNet / Miras · LaCT · log-linear attention · unified update-rule table |
| 12 | `12-hybrids.md` | Hybrid stacks | W6 (after 11) | why hybrid (state vs. retrieval) · interleaved vs. parallel heads (Jamba/Samba vs Hymba/Falcon-H1) · Qwen3-Next & Kimi Linear · MiniMax-01 → M1 → M2 (the return to full attention) · NVIDIA Nemotron-H / Jet-Nemotron · frontier layout table 2024–26 |
| 13 | `13-kernels.md` | Kernels | W7 | FlashAttention (tiling + online softmax) · FA-2/3/4 · split-KV decoding (FlashDecoding) · FlashMLA / FlashInfer · chunkwise linear-attention kernels (FLA) · low-bit attention (SageAttention) |
| 14 | `14-serving.md` | Serving the cache | W7 | FP8/FP4 KV cache (V3.2 656 B, V4.1 288 B) + KIVI/KVQuant · PagedAttention · prefix caching / RadixAttention · prefill–decode disaggregation (DistServe, Mooncake) · context parallelism (Ring, Ulysses) · tiered offload |
| 15 | `15-choose.md` | Compare and choose | coordinator | what shrinks table · evaluation (NIAH/RULER-style) · design checklist · open questions |

## Boundaries (avoid duplicate cards)

* **Sinks:** the learned per-head sink logit (gpt-oss, MiMo, V4) is a card in **02**. The StreamingLLM window+sink eviction pattern is a card in **05**. Chapter 07 links to both and does not repeat them.
* **SWA:** sliding windows and local:global interleaving are in **05**. Chapter 12 shows only layer patterns of hybrids with *recurrent* layers, plus the layout table, which may mention SWA ratios.
* **Output gate:** in **02**. Chapter 12 links to it.
* **NSA:** the full card is in **06**. Chapter 08 may show only its compression branch as a bridge.
* **KV quantization:** in **14**. Chapter 07 is only about eviction and merging.
* **Cross-layer KV sharing:** in **03**.
* **Chunkwise parallel algorithm for linear/delta layers:** the formula view is in **09/10**, the kernel view in **13**.
* **Positional handling inside MLA:** the decoupled RoPE lane is in **04**. Chapter 01 links to it.
