# Attention, one change at a time

[**Open the book →**](reader/index.html) · [PDF version](attention-visual-guide.pdf)

A visual field guide to how attention in LLMs developed from 2014 to 2026: MHA → GQA → MLA, sliding windows and sparse reads, cache eviction and compression, linear/delta/test-time-learning memories, hybrid stacks, kernels and serving.

It is written for readers who already know softmax attention and the KV cache. Each of the **91 mechanism cards** has the same parts:

- **meta line:** year, authors or lab, venue, link, and the production models that use it
- **one formula** that defines the mechanism
- **one dense figure** with a colour-coded formula strip, real tensor shapes and real numbers taken from papers or `config.json`
- **2–4 short tips:** why it works, the gotcha, what it does not solve
- **an optional collapsed derivation**

The book has 92 figures and 189 primary sources. Figure colours have the same meaning in every chapter: gold = query, blue = key / KV cache, purple = value / state, green = kept / output, red = evicted / cost, teal = gate / position.

![Seven levers organise the book.](figures/cards/map-routes.svg)

| # | Chapter | The question | Cards |
| --- | --- | --- | --- |
| 00 | [The map](chapters/00-map.md) | Every mechanism in this book attacks one factor of the attention bill. Here are the bill, the levers, and when each lever appeared. | The bill · Seven levers · Twelve years in five lanes · How to read the cards |
| 01 | [Positions](chapters/01-positions.md) | RoPE decides how far apart two tokens look. Long context needs slow pairs that span the window, rescaled pairs, or no rotation at all. | RoPE · Context extension · Partial RoPE and p-RoPE · NoPE layers · Length-dependent logit temperature |
| 02 | [Stable softmax](chapters/02-stable-softmax.md) | Softmax must sum to one, and its logits can grow without bound. Modern layers cap the logits, give each head an exit, and gate its output. | QK-norm, soft-cap, QK-Clip · Attention sinks → a learned sink logit · Gated attention · Research knobs · 2026 anatomy |
| 03 | [Share KV heads](chapters/03-kv-heads.md) | Decode re-reads the whole cache for every new token. First lever: store fewer K/V heads per token and keep every query head. | MHA → GQA → MQA · What it saves in real models · Why decode benefits · K=V · Beyond GQA · Cross-layer sharing |
| 04 | [Latent KV](chapters/04-latent-kv.md) | Head sharing throws away per-head views. MLA keeps all query heads yet caches one 576-element latent per token, trading cache bytes for decode FLOPs. | MLA · Weight absorption · Decoupled RoPE · Head count · Converting MHA/GQA → MLA |
| 05 | [Fixed patterns](chapters/05-fixed-patterns.md) | Most layers do not need the whole history. Fix in advance which keys each query may see, and those layers stop growing with $N$. | Sparse Transformer → Longformer / BigBird · Sliding window + global layers in production · Chunked local attention (Llama 4) vs sliding window · StreamingLLM |
| 06 | [Sparse reads](chapters/06-sparse-reads.md) | Keep every record, but let each query read only a few: a cheap scorer picks them, and exact attention runs on the picks. | What shrinks · Training-free prefill · Quest · MoBA · NSA · DSA · 2026 indexers |
| 07 | [Evict from the cache](chapters/07-kv-eviction.md) | Selection reads fewer records but keeps them all. Eviction deletes records for good, so it must guess what future queries will need. | H2O · SnapKV · PyramidKV and Ada-KV · KVzip · DuoAttention and RazorAttention · DMC → DMS · Pitfalls |
| 08 | [Compress the sequence](chapters/08-fewer-records.md) | Eviction deletes positions. Compression merges each run of m tokens into one learned record, so the cache grows as N/m instead of N. | Compressive Transformer · Gist tokens → Activation Beacon · NSA's compression branch · CSA · HCA · DeepSeek-V4 |
| 09 | [Linear attention & SSMs](chapters/09-linear-state.md) | The KV cache grows with every token. Drop the softmax and the whole history folds into one fixed $d_v\times d_k$ matrix per head. | Linear attention · RetNet → Lightning · GLA · Mamba → Mamba-2 · Mamba-3 · Chunkwise form |
| 10 | [The delta rule](chapters/10-delta-rule.md) | Additive states can only pile up evidence. The delta rule reads what the state predicts for a key and overwrites only the error. | DeltaNet · Gated DeltaNet · KDA · RWKV-7 and Comba · Negative eigenvalues and Householder products · Compare · 2026 |
| 11 | [Memory as test-time learning](chapters/11-test-time-learning.md) | A recurrent state is a small model trained during inference. Changing its loss, optimizer or batch size gives the newest memory layers. | Every update is one optimizer step · TTT layers · Titans · ATLAS and MesaNet · LaCT and TTT-E2E · Log-linear attention · Limits |
| 12 | [Hybrid stacks](chapters/12-hybrids.md) | A fixed state compresses but cannot look things up. Production models keep a few full-attention layers and make the rest cheap. | Why hybrid · Interleaved layers vs parallel heads · NVIDIA · Qwen3-Next, Qwen3.5, Kimi · MiniMax · Frontier layouts 2024–2026 |
| 13 | [Kernels](chapters/13-kernels.md) | Same exact attention, different schedule: kernels decide how many bytes cross HBM and how many SMs stay busy. | FlashAttention · FA-2 → FA-3 → FA-4 · Split-KV decoding (Flash-Decoding) · FlashMLA and FlashInfer · Chunkwise kernels for linear and delta layers (FLA) · Low-bit attention |
| 14 | [Serving the cache](chapters/14-serving.md) | A server holds thousands of caches. Shrink each record, page them, reuse shared prefixes, ship them between GPUs, spill them to cheaper tiers. | FP8 / FP4 KV cache · KIVI and KVQuant · PagedAttention · Prefix caching · Prefill–decode disaggregation · Context parallelism · Tiered storage |
| 15 | [Compare and choose](chapters/15-choose.md) | Every lever trades recall for bytes in a different place. Compare the real stacks, check what each guarantees by construction, then measure. | The bill across 17 real stacks · What each lever guarantees · Three 2026 recipes |

[Mechanism matrix](appendices/mechanism-matrix.md) · [Notation and glossary](appendices/glossary.md) · [Reference article and coverage](bibliography/reference-note.md) · [Primary sources](bibliography/sources.md)

## Rebuild

```sh
python3 scripts/build_cards.py      # draw every figure (scripts/cards/*.py, toolkit scripts/figkit.py)
python3 build_reader.py             # HTML reader, matrix, source register, link check
python3 scripts/build_pdf.py        # PDF via headless Chrome
python3 scripts/check_chapter.py chapters/*.md   # card-format lint
```

Requires `pandoc`, and Google Chrome for the PDF. The card format is specified in [planning/card-style.md](planning/card-style.md); the October 2026 literature survey behind the new chapters is in [planning/research/](planning/research/). The previous long-form edition is kept in [planning/legacy/](planning/legacy/). Literature checked through 21 October 2026. No experiments were reproduced; reported numbers are attributed to their authors.
