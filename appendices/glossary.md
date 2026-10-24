# Notation and glossary

## Notation

| Symbol | Meaning |
| --- | --- |
| $N$, $t$, $j$ | context length, current position, history index |
| $L$ | layers |
| $H_q$, $H_{kv}$, $G=H_q/H_{kv}$ | query heads, cached KV heads, group size |
| $d$, $d_h$ | model width, head dimension |
| $d_c$, $d_r$ | MLA latent width, decoupled RoPE width |
| $s$ | bytes per stored element (2 for BF16, 1 for FP8) |
| $W$, $C$ | sliding window, chunk length |
| $B_s$, $n_{sel}$ | block size, number of selected blocks or tokens (top-$k$) |
| $q$, $k$, $v$ | query, key, value: column vectors |
| $S_t\in\mathbb R^{d_v\times d_k}$ | recurrent state; read $o_t=S_tq_t$; write $v_tk_t^\top$ |
| $\alpha_t$, $\operatorname{diag}(a_t)$ | scalar decay, per-channel decay |
| $\beta_t$ | write (learning) strength of a delta update |

**Figure colours:** gold = query · blue = key / KV cache · purple = value / state · green = output / kept · red = erased / evicted / cost · teal = gate / position · grey = inactive. See the [legend card](../chapters/00-map.md#legend).

## Terms

| Term | Meaning | Card |
| --- | --- | --- |
| Prefill / decode | Process the prompt in parallel (compute-bound, $N^2$) / generate one token at a time (bandwidth-bound, reads the whole cache) | [00](../chapters/00-map.md#bill) |
| KV cache | Stored keys and values of every past token, per layer | [00](../chapters/00-map.md#bill) |
| Arithmetic intensity | FLOPs per byte moved; decode attention ≈ $G$, far below a GPU's ridge point | [03](../chapters/03-kv-heads.md#intensity) |
| MHA / MQA / GQA | One KV head per query head / one KV head for all / one per group of $G$ | [03](../chapters/03-kv-heads.md#gqa) |
| K=V tying | One projection serves as both key and value, halving the stored vectors | [03](../chapters/03-kv-heads.md#kv-tie) |
| Cross-layer sharing | Later layers reuse an earlier layer's KV (CLA, YOCO, Gemma 3n) | [03](../chapters/03-kv-heads.md#cross-layer) |
| MLA | Cache one latent $c_t$ ($d_c=512$) plus a shared RoPE key ($d_r=64$) per token | [04](../chapters/04-latent-kv.md#mla) |
| Weight absorption | Fold the key/value up-projections into the query and output so decode attends over latents directly | [04](../chapters/04-latent-kv.md#absorb) |
| Decoupled RoPE | A separate small rotated key lane, because RoPE cannot pass through absorbed weights | [04](../chapters/04-latent-kv.md#rope) |
| RoPE base $b$ | Sets the slowest rotation's wavelength; raised to cover long contexts | [01](../chapters/01-positions.md#rope-base) |
| PI / NTK / YaRN | Context extension by rescaling RoPE frequencies, low frequencies most | [01](../chapters/01-positions.md#extension) |
| Partial RoPE / NoPE | Rotate only some dimensions / layers with no positional encoding at all | [01](../chapters/01-positions.md#partial), [01](../chapters/01-positions.md#nope) |
| QK-norm / soft-cap | Normalise $q,k$ / squash logits with tanh to keep softmax stable | [02](../chapters/02-stable-softmax.md#qk-norm) |
| Attention sink | Heads park spare attention mass on early tokens; a learned sink logit does this explicitly | [02](../chapters/02-stable-softmax.md#sinks) |
| Output gate | Sigmoid gate on the attention output (Qwen3-Next); acts on this read, not on stored memory | [02](../chapters/02-stable-softmax.md#gated) |
| Sliding window (SWA) | A layer that only sees the last $W$ tokens; its cache is a ring buffer | [05](../chapters/05-fixed-patterns.md#swa-global) |
| Chunked attention | Block-diagonal causal attention over fixed chunks (Llama 4) | [05](../chapters/05-fixed-patterns.md#chunked) |
| Sparse read | Attend to a selected subset; the cache is kept, so the next query can choose again | [06](../chapters/06-sparse-reads.md#what-shrinks) |
| Indexer | A cheap scorer that picks the top-$k$ tokens or blocks for the main attention (DSA, MSA, QSA) | [06](../chapters/06-sparse-reads.md#dsa) |
| NSA / MoBA / Quest | Learned three-branch sparse attention / learned block routing / training-free page bounds | [06](../chapters/06-sparse-reads.md#nsa) |
| Eviction | Permanently delete cache entries (H2O, SnapKV); cannot be undone | [07](../chapters/07-kv-eviction.md#pitfalls) |
| Retrieval vs streaming heads | Heads that need the full cache vs heads that work with sinks + a window (DuoAttention) | [07](../chapters/07-kv-eviction.md#duo) |
| Sequence compression | Merge many positions into fewer learned records (CSA, HCA, Activation Beacon) | [08](../chapters/08-fewer-records.md#csa) |
| Linear attention | Drop the softmax so history folds into a fixed matrix state | [09](../chapters/09-linear-state.md#linear-attention) |
| Decay (retention / gate) | Multiply the state by $\alpha_t$ or $\operatorname{diag}(a_t)$ each step to forget | [09](../chapters/09-linear-state.md#gla) |
| State-space duality (SSD) | Mamba-2 equals masked linear attention with a decay mask | [09](../chapters/09-linear-state.md#ssd) |
| Chunkwise form | Attention inside a chunk, state passed between chunks: how these layers train fast | [09](../chapters/09-linear-state.md#chunkwise) |
| Delta rule | Read the old prediction, write only the residual: $S\leftarrow S+\beta(v-Sk)k^\top$ | [10](../chapters/10-delta-rule.md#deltanet) |
| GDN / KDA | Gated DeltaNet (scalar decay + delta) / Kimi Delta Attention (per-channel decay + delta) | [10](../chapters/10-delta-rule.md#gdn), [10](../chapters/10-delta-rule.md#kda) |
| Negative eigenvalues | $\beta\in(0,2)$ lets a state flip sign, enabling parity / state tracking | [10](../chapters/10-delta-rule.md#state-tracking) |
| Test-time training (TTT) | The state is a small model updated by gradient steps while reading | [11](../chapters/11-test-time-learning.md#ttt) |
| Hybrid stack | Mostly fixed-state layers plus a few full-attention layers | [12](../chapters/12-hybrids.md#why) |
| FlashAttention | Tiled attention with an online softmax; never writes the $N\times N$ matrix to HBM | [13](../chapters/13-kernels.md#flash) |
| Split-KV decoding | Split one query's history across many thread blocks, then merge with log-sum-exp | [13](../chapters/13-kernels.md#flash-decoding) |
| Paged / prefix cache | Cache stored in fixed pages; shared prompt prefixes reused across requests | [14](../chapters/14-serving.md) |
| Disaggregation | Prefill and decode on different GPUs, with the KV cache transferred between them | [14](../chapters/14-serving.md) |
