# Reference article and coverage

The organizing reference is **《一文看懂 Attention 演进史：从 MHA、GQA、MLA 到 DeltaNet、KDA 与 Hybrid Attention》** by **寒江雪**, published by **对话系统漫谈** on **9 October 2026** ([WeChat article](https://mp.weixin.qq.com/s/RBsvDYoGdgE1abo78llvmw)). Its 45 screenshot pages are kept as [screencapture.pdf](../screencapture.pdf). They have no text layer, so formulas clipped in the screenshots were re-derived from primary sources.

This edition keeps the article's memory-system lens. It turns every mechanism into a card: one formula, one figure with real numbers, and a few tips. A literature survey (October 2026) added the mainstream work the article leaves out.

## Where each part of the article went

| Article sections · pages | Topic | Chapters |
| --- | --- | --- |
| Intro, §§1–4 · pp. 1–3 | Retrieval view, fixed-vector bottleneck, Transformer, MHA | [00](../chapters/00-map.md) |
| §§5–7 · pp. 3–6 | Incremental decoding, KV cache, MQA, GQA | [00](../chapters/00-map.md), [03](../chapters/03-kv-heads.md) |
| §§8–14 · pp. 6–10 | Linear state, retention, GLA, Mamba, state-space duality | [09](../chapters/09-linear-state.md) |
| §§15–18 · pp. 10–13 | Delta correction, adaptive forgetting, finite capacity | [10](../chapters/10-delta-rule.md) |
| §§19–22 · pp. 13–16 | MLA, weight absorption, record growth, cost factors | [04](../chapters/04-latent-kv.md), [00](../chapters/00-map.md) |
| §23 · pp. 16–17 | FlashAttention, GPU memory traffic | [13](../chapters/13-kernels.md) |
| §§24–30 · pp. 17–20 | Fixed sparsity, MoBA, MInference, NSA, DSA | [05](../chapters/05-fixed-patterns.md), [06](../chapters/06-sparse-reads.md) |
| §§31–32 · pp. 20–22 | DeepSeek-V4 CSA/HCA | [08](../chapters/08-fewer-records.md) |
| §§33–40 · pp. 22–28 | Output gates, Qwen3-Next, KDA, Kimi Linear | [02](../chapters/02-stable-softmax.md), [10](../chapters/10-delta-rule.md), [12](../chapters/12-hybrids.md) |
| §§41–45 · pp. 28–31 | Erase/write control, optimizer view, KDN, FG²-GDN | [10](../chapters/10-delta-rule.md), [11](../chapters/11-test-time-learning.md) |
| §§46–49 · pp. 31–33 | Explicit vs recurrent routes, hybrid convergence | [00](../chapters/00-map.md), [12](../chapters/12-hybrids.md) |
| §§50–51 · pp. 33–34 | V4.1-Flash, cross-layer reuse, FP4, storage dimensions | [03](../chapters/03-kv-heads.md), [14](../chapters/14-serving.md) |
| §§52–57 · pp. 34–39 | Update progression, MLA vs KDA, kernels, Ring Attention | [10](../chapters/10-delta-rule.md), [13](../chapters/13-kernels.md), [14](../chapters/14-serving.md) |
| §§58–64 · pp. 39–45 | Addressability, learned memory allocation, consolidated table | [12](../chapters/12-hybrids.md), [15](../chapters/15-choose.md), [matrix](../appendices/mechanism-matrix.md) |

## What this edition adds beyond the article

* **[01 Positions](../chapters/01-positions.md):** RoPE base and spectrum, PI → NTK → YaRN, partial RoPE, NoPE layers.
* **[02 Stable softmax](../chapters/02-stable-softmax.md):** QK-norm, soft-capping, learned sink logits.
* **[03](../chapters/03-kv-heads.md):** K=V tying, MFA/GTA, and cross-layer sharing in production.
* **[04](../chapters/04-latent-kv.md):** MLA head-count choice and MHA → MLA conversion.
* **[05 Fixed patterns](../chapters/05-fixed-patterns.md):** sliding-window : global interleaving as shipped in Gemma, gpt-oss, Llama 4 and MiMo.
* **[06](../chapters/06-sparse-reads.md):** Quest, the 2026 learned indexers (MSA, QSA, IndexCache) and the indexer-cost problem.
* **[07 Evict from the cache](../chapters/07-kv-eviction.md):** H2O, SnapKV, PyramidKV/Ada-KV, KVzip, DuoAttention, DMS.
* **[08](../chapters/08-fewer-records.md):** compression lineage (Compressive Transformer, gist tokens, Activation Beacon).
* **[09](../chapters/09-linear-state.md)–[10](../chapters/10-delta-rule.md):** RWKV-7's sum form, negative eigenvalues and Householder products (DeltaProduct, PaTH, Mamba-3), and the transition-matrix taxonomy.
* **[11 Memory as test-time learning](../chapters/11-test-time-learning.md):** TTT, Titans, ATLAS/Mesa, LaCT and log-linear attention.
* **[12](../chapters/12-hybrids.md):** a verified table of frontier attention layouts, and MiniMax-M2's return to full attention.
* **[13](../chapters/13-kernels.md)–[14](../chapters/14-serving.md):** split-KV decoding, FlashMLA, low-bit attention, FP8/FP4 KV caches, prefix caching, disaggregation and context parallelism.

## Corrections relative to the previous edition

* **[Kimi Linear](../chapters/12-hybrids.md):** the released model has 27 layers, 20 KDA and 7 MLA, so its 3:1 ratio is nominal.
* **[GLA](../chapters/09-linear-state.md):** the deployed GLA gates key channels only.
* **[Linear attention](../chapters/09-linear-state.md):** modern variants drop the normalizer; only mLSTM keeps one.
* **[DSA](../chapters/06-sparse-reads.md):** the indexer applies ReLU to each head's score before the weighted sum over heads.
* **[HCA](../chapters/08-fewer-records.md):** "two initial HCA layers" holds for DeepSeek-V4-Pro only.
* **[Per-channel write strength](../chapters/10-delta-rule.md):** it predates 2026; RWKV-7 already used it in 2025.

All reported results stay attributed to their authors. No model was trained or benchmarked for this book. The [primary sources](sources.md) page lists every paper, report and config that was checked.
