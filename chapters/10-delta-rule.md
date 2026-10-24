# 10 · The delta rule

::: lede
Additive states can only pile up evidence. The delta rule reads what the state predicts for a key and overwrites only the error.
:::

## DeltaNet: read, compare, overwrite {#deltanet}

::: meta
**DeltaNet** 2021 · Schlag, Irie, Schmidhuber (IDSIA) · ICML 2021 · [arXiv:2102.11174](https://arxiv.org/abs/2102.11174) — **Parallel DeltaNet** 2024 · Yang, Wang, Zhang, Shen, Kim · NeurIPS 2024 · [arXiv:2406.06484](https://arxiv.org/abs/2406.06484) · Used in: the base of every card below
:::

$$
S_t=S_{t-1}+\beta_t\,\underbrace{(v_t-S_{t-1}k_t)}_{\text{residual}}\,k_t^{\top}
=S_{t-1}(I-\beta_tk_tk_t^{\top})+\beta_tv_tk_t^{\top}
$$

![Toy delta update for order 7314: state (2, 4) over two key directions, k = e₁, new value 5 (Friday). The read gives 2 (Tuesday), the residual is 3, and with β = 1 the write (3, 0) gives (5, 4). Additive linear attention would give (7, 4). β = 0, 0.5 and 1 give 2, 3.5 and 5. A query halfway between both keys moves from 4.24 to 6.36.](../figures/cards/delta-deltanet.svg)

<!-- delta-demo -->

::: tips
- **Why it works:** it is one gradient step on $\tfrac12\lVert Sk_t-v_t\rVert^2$ (the LMS rule). Linear attention only adds. More in [ch 11](11-test-time-learning.md).
- **Gotcha:** the edit is exact only for orthogonal keys. A query that overlaps $k_t$ shifts too: 4.24 → 6.36 in the toy.
- **Train vs. infer:** $q,k$ are L2-normalized and $\beta\in(0,1)$. Chunked training turns products of $(I-\beta kk^\top)$ into a WY/UT form.
:::

## Gated DeltaNet: decay first, then correct {#gdn}

::: meta
**Gated DeltaNet** 2024 · Yang, Kautz, Hatamizadeh (MIT, NVIDIA) · ICLR 2025 · [arXiv:2412.06464](https://arxiv.org/abs/2412.06464) · Used in: Qwen3-Next, Qwen3.5, Olmo Hybrid, Jet-Nemotron
:::

$$
S_t=\alpha_t\,S_{t-1}(I-\beta_tk_tk_t^{\top})+\beta_tv_tk_t^{\top},\qquad \alpha_t,\beta_t\in(0,1)
$$

![Gated DeltaNet on the toy with α = 0.8 and β = 0.5: (2, 4) fades to (1.6, 3.2), the read is 1.6, the residual 3.4, the write 1.7, and the result (3.3, 3.2). A 3×3 grid gives the result for α = 1, 0.8, 0 and β = 0, 0.5, 1; α = 1 is DeltaNet. Qwen3-Next-80B-A3B has 36 GDN and 12 gated-attention layers with 32 value heads × 128 × 128 = 524,288 state numbers per GDN layer.](../figures/cards/delta-gated.svg)

::: tips
- **Why it works:** $\alpha$ clears a whole head at once; $\beta$ edits one association. Mamba-2 had only the first, DeltaNet only the second.
- **Order:** decay first, so the residual is read from the faded state: 1.6, not 2. The other order is a different rule ([sum form](#sum-form)).
- **Numbers:** count state per value head. Qwen3-Next pairs 32 value heads with 16 query/key heads.
- **Gotcha:** a scalar $\alpha$ also fades unrelated associations: 4 → 3.2 in the toy.
:::

## KDA: per-channel decay inside the delta rule {#kda}

::: meta
**Kimi Delta Attention** 2025 · Kimi Team, Moonshot AI · [arXiv:2510.26692](https://arxiv.org/abs/2510.26692) · Used in: Kimi Linear 48B-A3B
:::

$$
S_t=S_{t-1}\operatorname{Diag}(\alpha_t)\,(I-\beta_tk_tk_t^{\top})+\beta_tv_tk_t^{\top},\qquad \alpha_t\in(0,1)^{d_k}
$$

![KDA on the toy with α = (0.8, 1) and β = 0.5 gives (3.3, 4); scalar-decay Gated DeltaNet gives (3.3, 3.2). The transition is diagonal plus a rank-1 term tied to k, which cuts general DPLR's four second-level chunk matrix computations to two. Kimi-Linear-48B-A3B has 27 layers, 20 KDA and 7 MLA, with 32 heads × 128 × 128 = 524,288 state numbers per KDA layer.](../figures/cards/delta-kda.svg)

::: tips
- **Why it works:** it puts [GLA's](09-linear-state.md#gla) per-channel decay into the gated delta rule. Order 2208 keeps 4 while 7314 is overwritten.
- **Cost:** tying both low-rank vectors to $k$ is the speed trick: reported ≈2× faster kernels than general DPLR up to 64K.
- **Gotcha:** $\operatorname{Diag}(\alpha)$ and $I-\beta kk^\top$ do not commute. The correction reads the decayed state.
- **Used in:** 3 KDA : 1 MLA, nominally; the released 27 layers end with a 2:1 group. Layouts: [ch 12](12-hybrids.md).
:::

## RWKV-7 and Comba: the sum form {#sum-form}

::: meta
**RWKV-7 "Goose"** 2025 · Peng et al., RWKV Project · [arXiv:2503.14456](https://arxiv.org/abs/2503.14456) — **Comba** 2025 · Hu et al. · [arXiv:2506.02475](https://arxiv.org/abs/2506.02475) · Used in: RWKV-7 World 0.1B–2.9B
:::

$$
\begin{aligned}
\text{product (GDN, KDA):}\quad & S_{t-1}\,\alpha_t(I-\beta_tk_tk_t^{\top})+\beta_tv_tk_t^{\top}\\
\text{sum (RWKV-7, Comba):}\quad & S_{t-1}\big(\operatorname{diag}(w_t)-\hat\kappa_t(a_t\odot\hat\kappa_t)^{\top}\big)+v_t\tilde k_t^{\top}
\end{aligned}
$$

![The same toy, α = 0.8 and β = 0.5, in both forms. Product: decay to (1.6, 3.2), the erase reads the faded 1.6 and removes 0.8, result (3.3, 3.2). Sum: the erase reads the unfaded 2 and removes 1.0, result (3.1, 3.2). RWKV-7 makes the decay w and the learning rate a vectors over key channels; with head size 64 each head holds 4,096 state numbers.](../figures/cards/delta-sum-form.svg)

::: tips
- **Compare:** the product form erases what the decay left; the sum form erases from the old state while it decays: 3.3 vs 3.1.
- **Not new in 2026:** RWKV-7 already had per-channel decay and per-channel write strength (vector $a_t$) in March 2025.
- **Gotcha:** the removal key $\hat\kappa$ differs from the replacement key $\tilde k$, so this is general DPLR and costs about 2× KDA's chunk work.
- **Comba** uses $\alpha_tI-\tilde\beta_tk_tk_t^\top$ and reads with $q_t-d\,k_t$, an output correction.
:::

## Negative eigenvalues and Householder products {#state-tracking}

::: meta
**Negative eigenvalues** 2024 · Grazzi, Siems, Zela, Franke, Hutter, Pontil · ICLR 2025 · [arXiv:2411.12537](https://arxiv.org/abs/2411.12537) — **DeltaProduct** 2025 · Siems et al. · NeurIPS 2025 · [arXiv:2502.10297](https://arxiv.org/abs/2502.10297) · Used in: Olmo Hybrid 7B (β up to 2)
:::

$$
\operatorname{eig}(I-\beta kk^{\top})=\{1,\dots,1,\;1-\beta\},\qquad
A_t=\prod_{j=1}^{n_h}\big(I-\beta_{t,j}k_{t,j}k_{t,j}^{\top}\big),\quad \beta\in(0,2)
$$

![The eigenvalue 1 − β lies in (0, 1) for β ≤ 1 and reaches −1 at β = 2, a reflection. Parity toy: with β = 2 for every 1-bit the 1-D state flips sign and ends at +1 after bits 1 0 1 1 0 1; with β ≤ 1 it can never flip. Two reflections with mirrors 30° apart make a 60° rotation. Olmo Hybrid 7B sets linear_allow_neg_eigval to true: 24 GDN + 8 attention layers.](../figures/cards/delta-eigen.svg)

::: tips
- **Why it works:** finite-precision linear RNNs with only positive eigenvalues provably cannot solve parity. $\beta=2\sigma(\cdot)$ adds the sign flip.
- **DeltaProduct:** $n_h$ delta steps per token make the transition identity plus rank $n_h$. Two reflections give rotations; the state does not grow.
- **Compare:** PaTH uses the same Householder products as a position encoding inside softmax. Mamba-3 gets rotations from a complex state ([ch 09](09-linear-state.md#mamba3)).
- **Not solved:** this adds expressivity, not capacity.
:::

## Compare: one axis, the transition $A_t$ {#taxonomy .insight}

$$
S_t=S_{t-1}A_t+W_t,\qquad A_t\in\big\{I,\ \alpha_tI,\ \operatorname{Diag}(\alpha_t),\ \alpha_t(I-\beta_tk_tk_t^{\top}),\ \dots,\ \alpha_t\mathcal{R}_t\big\}
$$

![Ten transition matrices as toy 6×6 heat maps, teal for positive and red for negative entries: identity (linear attention), fixed γI (RetNet, Lightning), α_t I (Mamba-2, mLSTM), Diag(α_t) (GLA, HGRN2), I − βkkᵀ (DeltaNet), α(I − βkkᵀ) (Gated DeltaNet), Diag(α)(I − βkkᵀ) (KDA), Diag(w) − κ̂(a⊙κ̂)ᵀ (RWKV-7, Comba), Householder products (DeltaProduct) and α times a rotation (Mamba-3), each with its eigenvalues and chunk algorithm.](../figures/cards/delta-taxonomy.svg)

| $A_t$ | Models | Chunk trick |
| --- | --- | --- |
| $I$, $\gamma I$, $\alpha_tI$ | linear attention, RetNet, Lightning, Mamba-2, mLSTM | cumulative product |
| $\operatorname{Diag}(\alpha_t)$ | GLA, HGRN2 | cumulative sum in log space |
| $\alpha(I-\beta kk^\top)$, $\operatorname{Diag}(\alpha)(I-\beta kk^\top)$ | DeltaNet, GDN, KDA, 2026 variants | WY / UT transform |
| $\operatorname{diag}(w)-\hat\kappa(a\odot\hat\kappa)^\top$ | RWKV-7, Comba | general DPLR |
| $\prod_j(I-\beta_jk_jk_j^\top)$, $\alpha\mathcal{R}_t$ | DeltaProduct, Mamba-3 | UT per sub-step; rotations |

::: tips
- **Cost:** diagonal classes commute, so a chunk needs only cumulative decays. Householder classes need WY/UT.
- **Product vs. sum:** decides whether the erase reads the faded or the unfaded state.
- **Not solved:** every class keeps the same $d_v\times d_k$ state. A richer $A_t$ buys editing and state tracking, not recall.
:::

## 2026: decouple strength, gain and address {#frontier}

::: meta
**FG²-GDN** 2026 · Sun et al. · [arXiv:2604.19021](https://arxiv.org/abs/2604.19021) — **Gated DeltaNet-2** 2026 · Hatamizadeh, Choi, Kautz · [arXiv:2605.22791](https://arxiv.org/abs/2605.22791) — **Erase-then-Delta** 2026 · Li et al. · [arXiv:2606.26560](https://arxiv.org/abs/2606.26560) — **Kalman Delta Networks** 2026 · Bui, Huang, Ying · [arXiv:2609.07816](https://arxiv.org/abs/2609.07816) · Used in: research models only (Oct 2026)
:::

$$
S_t=S_{t-1}\operatorname{Diag}(\alpha_t)\big(I-(b_t\odot k_t)\,k_t^{\top}\big)+(w_t\odot v_t)\,k_t^{\top}
\qquad\text{(GDN-2; } b=w=\beta\mathbf{1}\text{ gives KDA)}
$$

![Four 2026 refinements on a toy 5×5 state. FG²-GDN: a per-channel strength β on key and value (tied; separate in FG²-GDN+). Gated DeltaNet-2: an erase gate b per key channel and a write gate w per value channel, both at address k. Kalman Delta Networks: the step size comes from an uncertainty state p per key channel. Erase-then-Delta: a separate erase address e before the write at k. Tested scales range from 340M to a 25B-A2.8B MoE. Timeline from RWKV-7 (March 2025) to KDN (September 2026).](../figures/cards/delta-frontier.svg)

::: tips
- **Correction:** per-channel write strength is not new. RWKV-7's vector $a_t$ had it; FG²-GDN brings it to GDN/KDA kernels.
- **Strength vs. address:** FG²-GDN+ and GDN-2 gate how much is erased and written at $k$. EDA erases along a separate direction $e_t$.
- **Gain:** KDN's isotropic variant is KDA with $\beta_t$ set by a scalar covariance; the diagonal variant writes along a gain vector.
- **Not solved:** only EDA was trained beyond 1.3B. No released model uses any of them.
:::
