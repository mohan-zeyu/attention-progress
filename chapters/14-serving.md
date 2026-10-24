# 14 · Serving the cache

::: lede
A server holds thousands of caches. Shrink each record, page them, reuse shared prefixes, ship them between GPUs, spill them to cheaper tiers.
:::

## FP8 / FP4 KV cache: the production default {#kv-quant}

::: meta
**FP8 / FP4 KV formats** 2025–26 · DeepSeek · [FlashMLA](https://github.com/deepseek-ai/FlashMLA) (V3.2 656 B, V4.1 528 / 288 B) · [vLLM V3.2 blog](https://vllm.ai/blog/2025-09-29-deepseek-v3-2) · [NVIDIA NVFP4 KV](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/) · Used in: DeepSeek-V3.2, DeepSeek-V4.1, vLLM
:::

$$
\text{bytes per token per layer}=\underbrace{n\,b/8}_{\text{quantized}}+\underbrace{(n/g)\,s_{\text{scale}}}_{\text{scales}}+\underbrace{2\,n_{\text{bf16}}}_{\text{kept}},\qquad \text{V3.2: }512+4\cdot 4+64\cdot 2=656
$$

![Per token and layer: V3.2 BF16 1,152 B; V3.2 FP8 656 B = 512 e4m3 + four fp32 scales + 64 BF16 RoPE values; V4.1 FP8 528 B with RoPE quantized and e8m0 scales per 32; V4.1 FP4 288 B = 256 B of e2m1 + 32 e4m3 scales per 16. Over 61 layers a 128K V3.2 sequence needs 4.88 GiB instead of 8.58 GiB.](../figures/cards/serve-kv-bytes.svg)

::: tips
- **Gotcha:** scales and the BF16 RoPE lane make V3.2's FP8 cache 57% of BF16, not 50%. V4.1 quantizes RoPE too.
- **Cost:** saves memory and bandwidth, not FLOPs. NVFP4 KV is dequantized to FP8 before the matmuls (NVIDIA).
- **Numbers:** NVIDIA reports NVFP4 KV at ≈50% of FP8 memory with <1% accuracy loss on Blackwell.
- **Compare:** low-bit *arithmetic* inside the kernel is [SageAttention](13-kernels.md#sage).
:::

## KIVI and KVQuant: quantize along the outliers {#kivi}

::: meta
**KIVI** 2024 · Liu et al. · ICML 2024 · [arXiv:2402.02750](https://arxiv.org/abs/2402.02750) — **KVQuant** 2024 · Hooper et al., UC Berkeley · NeurIPS 2024 · [arXiv:2401.18079](https://arxiv.org/abs/2401.18079)
:::

$$
\hat x=z+\Delta\operatorname{round}\!\Big(\frac{x-z}{\Delta}\Big),\quad \Delta=\frac{\max-\min}{2^{b}-1},\ z=\min;\qquad K:\ \text{group per channel},\ \ V:\ \text{group per token}
$$

![Toy caches: keys have two outlier channels, so KIVI groups keys per channel and values per token, keeping the last R tokens in FP16 (paper: G = 32, R = 128, 2 bits). RoPE mixes channel pairs by position, so KVQuant quantizes keys before RoPE. Reported LLaMA-7B Wikitext-2 perplexity: FP16 5.68, KVQuant 3-bit with 1% outliers 5.75, uniform INT3 10.87.](../figures/cards/serve-kivi.svg)

::: tips
- **Why it works:** key outliers sit in a few fixed channels; values show no channel pattern (KIVI).
- **Gotcha:** RoPE smears those channels across positions. KVQuant dequantizes pre-RoPE keys, then applies RoPE.
- **Numbers:** KIVI 2-bit: 2.6× lower peak memory, up to 4× batch, 2.35–3.47× throughput. KVQuant: 1M tokens on one A100-80GB.
- **Converts from:** both are training-free; KVQuant calibrates its non-uniform levels offline.
:::

## PagedAttention: KV in fixed blocks {#paged}

::: meta
**PagedAttention / vLLM** 2023 · Kwon et al., UC Berkeley · SOSP 2023 · [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) · Used in: vLLM, SGLang (page = 1 token), FlashMLA (block 64)
:::

$$
\text{slot}(t)=\text{table}\big[\lfloor t/b\rfloor\big]\cdot b+(t\bmod b),\qquad \text{waste}\le b-1\ \text{tokens per sequence}
$$

![Toy with block size 4: request A's logical blocks map through its block table to physical blocks 7, 1 and 3, request B's to 5 and 2; only last blocks have empty slots. Reported KV memory holding token states: Orca 20.4–26.8%, vLLM 96.3%. OPT-13B at 800 KB per token: reserving 2,048 tokens costs 1.6 GB, paging wastes at most 12 MB.](../figures/cards/serve-paged.svg)

::: tips
- **Why it works:** blocks are allocated on demand and all have one size: no reservation, no external fragmentation.
- **Numbers:** reported 2–4× throughput vs FasterTransformer and Orca; default $b=16$.
- **Sharing:** reference-counted blocks with copy-on-write save 6.1–9.8% (parallel sampling) and 37.6–55.2% (beam search).
- **Gotcha:** kernels must gather scattered blocks; FlashMLA and FlashInfer read the block table directly ([13](13-kernels.md#flashmla)).
:::

## Prefix caching: RadixAttention {#radix}

::: meta
**SGLang / RadixAttention** 2023 · Zheng et al., Stanford / UC Berkeley · NeurIPS 2024 · [arXiv:2312.07104](https://arxiv.org/abs/2312.07104) · Used in: SGLang
:::

$$
\text{prefill tokens}=N-N_{\text{hit}},\qquad N_{\text{hit}}=\max\{\,n:\ x_{1:n}\ \text{is a path in the radix tree}\,\}
$$

![Toy radix tree: a 1,000-token system prompt shared by two chats and a few-shot branch. A new turn of chat 1 matches 1,350 of 1,410 tokens and prefills 60. LRU leaves are evicted first. Reported hit rates: 50–99% on SGLang benchmarks, 52.4% and 74.1% in Chatbot Arena, 56.3% of DeepSeek's daily input tokens.](../figures/cards/serve-radix.svg)

::: tips
- **Why it works:** tree nodes own KV pages; LRU evicts leaves first, so shared prefixes survive longest.
- **Gotcha:** a hit needs the exact token prefix. Context truncation can halve the hit ratio (LMCache).
- **Numbers:** SGLang reports up to 6.4× throughput. DeepSeek: 342B of 608B input tokens hit its disk cache in 24 h.
:::

## Prefill–decode disaggregation {#disagg}

::: meta
**DistServe** 2024 · Zhong et al. · OSDI 2024 · [arXiv:2401.09670](https://arxiv.org/abs/2401.09670) — **Splitwise** 2024 · Patel et al., UW / Microsoft · ISCA 2024 · [arXiv:2311.18677](https://arxiv.org/abs/2311.18677) — **Mooncake** 2025 · Qin et al., Moonshot AI / Tsinghua · FAST 2025 · [arXiv:2407.00079](https://arxiv.org/abs/2407.00079) · Used in: Kimi, DeepSeek-V3/R1 serving
:::

$$
T_{\text{xfer}}=\frac{N_{\text{prompt}}\sum_{\ell}c_\ell}{\text{BW}},\qquad \text{V3.2 FP8, 32K: }\frac{32{,}768\cdot 61\cdot 656\ \text{B}}{50\ \text{GB/s}}\approx 26\ \text{ms}
$$

![A long prefill on a shared GPU stalls every running decode; split pools ship the KV over RDMA instead. Computed for a 32K prompt: V3.2 FP8 1.22 GiB, 26 ms over one 400 Gb/s NIC; Llama-3-70B BF16 10 GiB, 215 ms. Reported: DistServe 7.4× more requests, Splitwise 1.4× throughput at 20% lower cost, Mooncake 59–498% more capacity.](../figures/cards/serve-disagg.svg)

::: tips
- **Why it works:** prefill is compute-bound (TTFT), decode bandwidth-bound (TPOT). Separate pools stop interference and size each phase.
- **Cost:** transfer scales with KV per token: V3.2 FP8 ships 8× fewer bytes than Llama-3-70B BF16.
- **Numbers:** DeepSeek runs prefill on 4-node units and decode on 18-node units: ≈73.7k input and 14.8k output tok/s per H800 node.
:::

## Context parallelism: Ring vs Ulysses {#cp}

::: meta
**Ring Attention** 2023 · Liu, Zaharia, Abbeel, UC Berkeley · ICLR 2024 · [arXiv:2310.01889](https://arxiv.org/abs/2310.01889) — **DeepSpeed-Ulysses** 2023 · Jacobs et al., Microsoft · [arXiv:2309.14509](https://arxiv.org/abs/2309.14509) · Used in: DeepSpeed; Llama 3 (all-gather variant)
:::

$$
\text{Ring: comm hidden if } c\ge \frac{F}{B};\qquad \text{Ulysses: }\frac{4Nh}{P}\ \text{per link, vs } 4Nh\ \text{(Megatron-SP)},\quad P\le H
$$

![Ring Attention keeps query blocks in place and passes K/V blocks around; minimum block F/B is 1.0K tokens on A100 NVLink and 24.5K on A100 InfiniBand (reported). Ulysses all-to-alls from a token split to a head split, so P cannot exceed the head count. Computed per GPU and layer at 128K, P = 8: MHA ring 4 GiB vs Ulysses 1 GiB; GQA 64/8 ring 512 MiB vs Ulysses 576 MiB.](../figures/cards/serve-cp.svg)

::: tips
- **Compare:** ring traffic ≈ $2NH_{kv}d_hs$ per GPU whatever $P$; Ulysses falls as $1/P$ but caps at $P=H$ (computed).
- **Gotcha:** under GQA, Ulysses splits the $H_{kv}$ heads too; beyond $P=H_{kv}$ K/V must be replicated.
- **Used in:** Llama 3 all-gathers K/V (small under GQA) over 2×CP load-balanced chunks; CP = 16 at 131K.
- **Not solved:** FLOPs stay $O(N^2)$; context parallelism only spreads them.
:::

## Tiered storage: HBM → DRAM → SSD {#tiers}

::: meta
**Mooncake Store** 2025 · Moonshot AI · [GitHub](https://github.com/kvcache-ai/Mooncake) — **LMCache** 2025 · Liu et al. · [arXiv:2510.09665](https://arxiv.org/abs/2510.09665) — **3FS** 2025 · DeepSeek · [GitHub](https://github.com/deepseek-ai/3FS) · Used in: Kimi, DeepSeek API, vLLM and SGLang (LMCache)
:::

$$
\text{reload if}\quad \frac{N\,b_{\text{tok}}}{\text{BW}_{\text{tier}}}<T_{\text{prefill}}(N)\approx\frac{2\,n_{\text{active params}}\,N}{F_{\text{GPU}}}
$$

![Typical bandwidths of an H100-class node: HBM3 3,350 GB/s, NVLink 900, PCIe 5 64 per direction, 400 Gb/s NIC 50, one NVMe SSD about 7 GB/s. Reloading a 32K-token V3.2 FP8 prefix (1.22 GiB) takes 0.39 ms to 187 ms, versus about 2.5 GPU-seconds to recompute it (computed). Reported: 56.3% of DeepSeek input tokens hit its on-disk cache; LMCache up to 15× throughput.](../figures/cards/serve-tiers.svg)

::: tips
- **Why it works:** thin records make even SSD reloads beat prefill: one TB holds 25M tokens of V3.2 FP8 KV (computed).
- **Numbers:** 3FS reports ≈6.6 TiB/s aggregate reads on 180 nodes; LMCache up to 15× throughput with vLLM.
- **Gotcha:** cached KV is tied to one model, one precision and an exact prefix. It is not semantic memory.
:::
