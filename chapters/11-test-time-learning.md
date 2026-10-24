# 11 · Memory as test-time learning

::: lede
A recurrent state is a small model trained during inference. Changing its loss, optimizer or batch size gives the newest memory layers.
:::

## Every update is one optimizer step {#online-step .insight}

$$
S_t=\alpha_tS_{t-1}-\beta_t\,\nabla_S\ell_t\big(\alpha_tS_{t-1}\big),\qquad
\ell_t^{\text{LA}}(S)=-\langle Sk_t,v_t\rangle,\quad \ell_t^{\Delta}(S)=\tfrac12\lVert Sk_t-v_t\rVert^2
$$

![Toy with d = 3 and beta = 1. S_{t-1} stores e1 to e1 and e2 to e2; the new pair is k = e1, v = e3. The linear-attention gradient is minus v k-transpose, so the new state reads back (1, 0, 1) and keeps the old value (error 1). The DeltaNet gradient is (Sk − v) k-transpose, so the new state reads back exactly (0, 0, 1) (error 0). Decay alpha before the step acts as weight decay.](../figures/cards/ttt-one-step.svg)

::: tips
- **Why it works:** $S$ is a regressor's weights and the read $S_tq_t$ is its prediction. The outer loop trains the projections and gates.
- **Compare:** linear attention and DeltaNet differ only in the loss. Details: [10 · The delta rule](10-delta-rule.md).
- **Gotcha:** this is a reading of fixed algebra. At inference, the inner learner has no free choices.
- **Framework:** GDN Table 1 lists these objectives. Miras splits any layer into memory, loss, retention, optimizer.
:::

| Rule | Loss $\ell_t(S)$ | Gradient | Decay $\alpha_t$ |
| --- | --- | --- | --- |
| Linear attention | $-\langle Sk_t,v_t\rangle$ | $-v_tk_t^\top$ | 1 |
| Mamba-2, RetNet | $-\langle Sk_t,v_t\rangle$ | $-v_tk_t^\top$ | scalar < 1 |
| DeltaNet | $\tfrac12\lVert Sk_t-v_t\rVert^2$ | $(Sk_t-v_t)k_t^\top$ | 1 |
| Gated DeltaNet | $\tfrac12\lVert Sk_t-v_t\rVert^2$ | $(Sk_t-v_t)k_t^\top$ | scalar < 1 |

## TTT layers: the state is a model, $b$ is the knob {#ttt}

::: meta
**TTT-Linear / TTT-MLP** 2024 · Sun, Li, Dalal et al., Stanford · UC San Diego · UC Berkeley · ICML 2025 · [arXiv:2407.04620](https://arxiv.org/abs/2407.04620) · Used in: research models, 125M–1.3B
:::

$$
W_t=W_{t-1}-\eta_t\nabla_W\big\lVert f(\theta_Kx_t;W_{t'})-\theta_Vx_t\big\rVert^2,\quad t'=t-(t\bmod b);\qquad z_t=f(\theta_Qx_t;W_t)
$$

![Mini-batch TTT on a toy 32-token sequence. With b = 1 every gradient uses the latest W: 32 sequential steps. With b = 8 the gradients in a batch share the batch-start W: 4 steps. With b = T there is one step, which equals linear attention. Reported 125M ablation: perplexity falls from 15.91 for linear attention to 11.09; mini-batch TTT with b = 16 is the largest single gain, −1.70.](../figures/cards/ttt-minibatch.svg)

::: tips
- **Why it works:** a batch shares $W_{t'}$, so $b$ gradients run as one matmul. The dual form trains >5× faster than the primal.
- **Compare:** $b=1$ is online GD (delta-like). $b=T$ with linear $f$, $\eta=\tfrac12$, $W_0=0$ is linear attention (Thm 1).
- **Gotcha:** $f(x)=x+\operatorname{LN}(f_{\text{res}}(x))$, so TTT-Linear is not DeltaNet. TTT-MLP: 4× hidden, GELU.
- **Numbers:** $b=16$ throughout; $\eta=\eta_{\text{base}}\,\sigma(\theta_{lr}x)$, $\eta_{\text{base}}$ = 1 (Linear) or 0.1 (MLP).
:::

## Titans: surprise, momentum, forgetting {#titans}

::: meta
**Titans** 2024 · Behrouz, Zhong, Mirrokni, Google Research · NeurIPS 2025 · [arXiv:2501.00663](https://arxiv.org/abs/2501.00663) · Used in: research models, 170M–760M
:::

$$
\mathcal M_t=(1-\alpha_t)\,\mathcal M_{t-1}+m_t,\qquad
m_t=\eta_t\,m_{t-1}-\theta_t\,\nabla\big\lVert\mathcal M_{t-1}(k_t)-v_t\big\rVert^2,\qquad y_t=\mathcal M_t(q_t)
$$

![Toy trace over 16 tokens: the gradient norm (momentary surprise) spikes at token 6, and with momentum 0.7 the write continues for about five more tokens. Three placements: MAC feeds memory read-outs as tokens to per-segment attention, MAG gates sliding-window attention with the memory branch, MAL stacks the memory layer before SWA. Reported ablation, long-context accuracy: MAC 97.95, MAL 96.91, MAG 96.70, memory alone 92.68, without momentum 87.12, without weight decay 85.60, linear memory 85.34.](../figures/cards/ttt-titans.svg)

::: tips
- **Why it works:** gradient = surprise. Momentum $m_t$ (the paper's $S_t$) keeps writing after a surprising token; $\alpha_t$ is weight decay.
- **Compare:** the gradient uses the undecayed $\mathcal M_{t-1}$, so linear Titans is the sum form, like RWKV-7, not GDN's product.
- **Cost:** $\mathcal M$ is an MLP. Deep memory loses closed-form chunk algebra and trains slower than Mamba2/GDN (§5.8).
- **Numbers:** removing weight decay costs the most: perplexity 27.01 → 29.04 (Table 5).
:::

## ATLAS and MesaNet: fit the window, or solve it {#atlas-mesa}

::: meta
**ATLAS** 2025 · Behrouz et al., Google Research · [arXiv:2505.23735](https://arxiv.org/abs/2505.23735) — **MesaNet** 2025 · von Oswald, Scherrer et al., Google Paradigms of Intelligence · ICLR 2026 · [arXiv:2506.05233](https://arxiv.org/abs/2506.05233) · Used in: research models, ≤1.3B
:::

$$
\begin{aligned}
\text{ATLAS (Omega rule): }&\min_{\mathcal M}\textstyle\sum_{i=t-c+1}^{t}\gamma_i^{(t)}\lVert\mathcal M(\phi(k_i))-v_i\rVert^2,\ \text{Muon steps}\\
\text{Mesa: }&o_t=G_t(H_t+\Lambda)^{-1}q_t,\quad G_t=\gamma_tG_{t-1}+\beta_tv_tk_t^\top,\quad H_t=\gamma_tH_{t-1}+\beta_tk_tk_t^\top
\end{aligned}
$$

![Which past pairs the loss sees, over 12 toy tokens: DeltaNet only token t, ATLAS the last c = 4 with gates, Mesa every token with a fading discount. Toy with correlated keys k1 = (1, 0) → +1 and k2 = (0.6, 0.8) → −1: reading at k1 gives 0.40 for linear attention, 0.04 for DeltaNet and 1.00 for Mesa. Reported costs: Mesa needs about k times GLA FLOPs for k conjugate-gradient steps; stopping at 1e-4 averages about 9 steps instead of 30.](../figures/cards/ttt-window-exact.svg)

::: tips
- **Compare:** $c=1$ is the delta rule. $c=t$ with linear $\mathcal M$ is least squares: DeltaNet takes one step, Mesa solves it.
- **Cost:** Mesa ≈ $k\times$ GLA FLOPs for $k$ CG steps. Stopping at $\epsilon=10^{-4}$ matches 30 steps with ≈9 (reported).
- **Gotcha:** Mesa's writes stay additive and chunk-friendly. The editing happens at read time, inside $(H_t+\Lambda)^{-1}$.
- **Numbers:** ATLAS capacity bound $O(d_k)\to O(d_k^{\,p})$ with degree-$p$ features; BABILong above 80% at 10M (reported).
:::

## LaCT and TTT-E2E: huge chunks make it fast {#lact}

::: meta
**LaCT** 2025 · Zhang, Bi, Hong et al., MIT · Adobe Research · ICLR 2026 · [arXiv:2505.23884](https://arxiv.org/abs/2505.23884) — **TTT-E2E** 2025 · Tandon, Dalal, Li et al., Astera · Stanford · NVIDIA et al. · [arXiv:2512.23675](https://arxiv.org/abs/2512.23675) · Used in: research; LaCT also in a 14B autoregressive video model fine-tuned from Wan 2.1
:::

$$
W\leftarrow\operatorname{L2Norm}\Big(W-\operatorname{Muon}\big(\nabla_W\textstyle\sum_{i=1}^{b}-\eta_i\,f_W(k_i)^\top v_i\big)\Big),\qquad
f_W(x)=W_2\big[\operatorname{SiLU}(W_1x)\odot W_3x\big]
$$

![Arithmetic intensity r = b / (1 + 2b/h) of a fast-weight matmul against chunk size b for widths h = 64, 256 and 1024. TTT with b = 16 and h = 64 reaches about 11 FLOP per byte, far below the H100 ridge of about 290. LaCT mask on 12 toy tokens with chunk 4: tokens read the fast weights updated on earlier chunks and use sliding-window attention locally. Reported: under 5% FLOPs utilization for earlier TTT versus up to 70% on A100 for LaCT; fast-weight state per block 2304d versus 384d for GLA and 128d for DeltaNet.](../figures/cards/ttt-large-chunk.svg)

::: tips
- **Why it works:** one update per 2K–1M tokens makes TTT matmul-bound and lets the state grow to ~40% of weights.
- **Gotcha:** a chunk never sees its own update, so every LaCT layer pairs with window attention.
- **TTT-E2E:** SWA-8K Transformer; trains last-¼ MLPs by next-token loss, 1K mini-batches. Prefill 2.7× faster than full attention at 128K (3B).
- **Cost:** gradients of gradients in training: 3.4× slower than full attention at 8K, 1.2× faster at 128K.
:::

## Log-linear attention: $O(\log T)$ states {#log-linear}

::: meta
**Log-Linear Attention** 2025 · Guo, Yang, Goel, Xing, Dao, Kim, MIT · CMU · Princeton · ICLR 2026 · [arXiv:2506.04761](https://arxiv.org/abs/2506.04761) · Used in: research models (log-linear Mamba-2 and Gated DeltaNet, ~800M)
:::

$$
o_t=\sum_{\ell=0}^{L-1}\lambda_t^{(\ell)}S_t^{(\ell)}q_t,\qquad S_t^{(\ell)}=\sum_{s\in\mathcal B_t^{(\ell)}}v_sk_s^\top,\qquad L\le\lceil\log_2t+1\rceil+1
$$

![A 16 by 16 causal mask coloured by Fenwick level: each query row splits its prefix into power-of-two buckets, one state each. For t = 13 the buckets are tokens 0–7, 8–11, 12 and token 13 itself: 4 states. At t = 2^20 a KV cache holds 1,048,576 records, log-linear attention at most 22 states, linear attention 1. Reported Wiki perplexity: Mamba-2 22.44 to 22.11, Gated DeltaNet 21.73 to 21.45.](../figures/cards/ttt-fenwick.svg)

::: tips
- **Compare:** one bucket per token is the KV cache; equal $\lambda$ across levels collapses to linear attention (§3.1).
- **Cost:** decode keeps $O(\log T)$ states; training is a chunkwise scan, $O(T\log T)$.
- **Used in:** wraps any linear cell. $\lambda$ adds <3% parameters to Mamba-2 and <0.4% to GDN.
- **Not solved:** the authors still report a significant gap to Transformers on every benchmark.
:::

## Limits: learning is not recall {#limits .insight}

$$
\#\{\text{pairs stored exactly}\}\le O(d_k)\ \text{per } d_v\times d_k\ \text{state (ATLAS Prop. 1)}\qquad\text{vs.}\qquad N\ \text{for a KV cache}
$$

![Left: reported in-context recall averages from MesaNet Table 3 (1B models, 50B tokens): Mamba2 37.67, GLA 39.64, Gated DeltaNet 39.54, DeltaNet 40.46, Mesa 41.79, SWA-1024 40.84, Transformer 52.27. Right: reported pass-key retrieval from TTT-E2E Table 2 (3B models): full attention stays at 1.00 to 0.99 from 8K to 128K, while TTT-E2E falls to 0.46, 0.24, 0.13 and 0.06, tracking 8K/N, the share of needles inside its 8K window.](../figures/cards/ttt-recall-gap.svg)

::: tips
- **Not solved:** Hebbian → exact solve adds ~4 recall points; the full cache adds 10.5 more (MesaNet Table 3).
- **Numbers:** TTT-E2E pass-key at 128K: 0.06 vs 0.99 for full attention. Its authors: compression "leaves out seemingly irrelevant details".
- **Gotcha:** ATLAS and Titans long-context wins are fine-tuned BABILong or ≤16K needles; check the setup.
- **Fix in practice:** keep a few full-attention layers: [12 · Hybrid stacks](12-hybrids.md).
:::

| Layer | Loss | Optimizer | State |
| --- | --- | --- | --- |
| TTT-Linear / MLP | $\lVert f(\theta_Kx)-\theta_Vx\rVert^2$ | GD, mini-batch $b=16$ | $W$ or 2-layer MLP per head |
| Titans | $\lVert\mathcal M(k)-v\rVert^2$ | GD + momentum + decay | MLP, depth $L_{\mathcal M}\ge 1$ |
| ATLAS | window of $c$ pairs, $\phi(k)$ | Muon | memory $\mathcal M$ on $\phi(k)$ |
| Mesa | all pairs, discounted, ridge | exact (CG) | $G$ and $H$ |
| LaCT | $-f_W(k)^\top v$ | Muon per 2K–1M chunk | SwiGLU, up to $12d^2$ |
| TTT-E2E | next-token loss | GD, 1K mini-batch | last ¼ of the MLPs |
