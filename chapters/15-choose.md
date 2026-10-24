# 15 · Compare and choose

::: lede
Every lever trades recall for bytes in a different place. Compare the real stacks, check what each guarantees by construction, then measure.
:::

## The bill across 17 real stacks {#bill .insight}

$$
\text{growing KV per token}=\sum_{\ell\in\text{full layers}}\text{record}_\ell\cdot s
\qquad\text{at }10^6\text{ tokens: }\times 2^{20}
$$

![Growing KV bytes per token on a log scale: Llama 3.1-405B 516,096; GLM-4.6 376,832; MiniMax-M2 253,952; Gemma 2 27B 188,416; Gemma 3 27B 81,920; Kimi K2 70,272; Llama 4 Scout 49,152; MiniMax-01 and Nemotron-H-56B 40,960; DeepSeek-V3.2 in FP8 40,016; gpt-oss-120b 36,864; Qwen3-Next 24,576; MiMo-V2-Flash 23,040; Jamba 16,384; Kimi Linear 8,064; Nemotron 3 Nano 6,144; DeepSeek-V4.1-Flash reported 890 including indexer keys.](../figures/cards/choose-kv-bytes.svg)

::: tips
- **A 580× spread.** Dense 405B → Nemotron 3 Nano is 84×. V4.1-Flash's reported 890 B (main KV + indexer keys, FP4) adds the rest.
- **Excluded on purpose:** bounded parts. Qwen3-Next's recurrent state is 72 MiB per sequence and gpt-oss's windows 4.5 MiB, at any length.
- **Full attention is not dead.** MiniMax-M2, GLM-4.x and Ministral 3 shipped all-global in 2025 ([why](12-hybrids.md)).
- **Source:** each `config.json`; arithmetic in the research notes.
:::

## What each lever guarantees {#properties .insight}

$$
\text{pick the failure you can afford:}\quad \text{recall lost}\;\leftrightarrow\;\text{bytes per token}
$$

![Seven rows by construction. Full attention: recall yes, flat cost no, nothing discarded yes. Windows alone: recall no beyond W. Sparse reads: recall if found, cache kept. Eviction: recall if kept, flat cost, irreversible. Compression: blurred, grows as N over r. Recurrent state: capacity-limited recall, flat cost. Hybrid: recall through few full layers. Last column: the test that exposes each.](../figures/cards/choose-properties.svg)

::: tips
- **Yellow cells need measuring.** "If found", "if kept" and "capacity" depend on training and budget, not on the formula.
- **Test the failure, not the average.** Needle-in-a-haystack is easy for most levers; multi-key RULER tasks, aggregation and questions asked after the context are not.
- **Multi-turn matters.** A prompt-only eviction can lose what turn 3 asks for ([07](07-kv-eviction.md#pitfalls)).
:::

## Three 2026 recipes {#recipes .insight}

$$
\text{stack}=\text{positions}+\text{stability}+\text{thin records}+\text{fewer reads or a state}+\text{fewer bytes}
$$

![Three 2026 stacks with chapter tags. DeepSeek-V4-Flash: partial RoPE, sinks, one K=V head, a 128-token window branch, a top-512 indexer, CSA/HCA compression, FP8 storage. Qwen3.8-Flash-Next: partial mRoPE, output gate, GQA with 2 KV heads, QSA indexer, 36 Gated DeltaNet layers in a 3:1 pattern. Gemma 4 31B: dual RoPE bases, QK-norm, K=V global layers, 5:1 windows.](../figures/cards/choose-recipes.svg)

::: tips
- **Nobody pulls one lever.** Each stack combines 4–7 chapters of this book.
- **Converging defaults:** partial RoPE, a sink or output gate, very few KV heads, a small local window, a learned top-$k$ reader.
- **Open questions:** where the indexer cost goes at 10M tokens, how to evaluate learned forgetting, and whether state tracking ([10](10-delta-rule.md)) matters at scale.
:::
