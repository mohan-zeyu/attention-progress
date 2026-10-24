# B · Recurrent, fast-weight, SSM, delta-rule and TTT memory: research notes

Checked 2026-10-21. Working notes for chapter writers. Only this file was written; no book files were changed.

**Convention (everywhere below):** the state is \(S\in\mathbb R^{d_v\times d_k}\), vectors are columns, reads are \(o_t=S_tq_t\), writes are \(v_tk_t^\top\), and the update is \(S_t=S_{t-1}A_t+W_t\).
Many papers (GLA, KDA, FG²-GDN, GDN-2, KDN, EDA, DeltaProduct) instead use \(S\in\mathbb R^{d_k\times d_v}\) with \(S_t=A^{\text{paper}}_tS_{t-1}+k_tv_t^\top\) and \(o=S^\top q\). Convert with \(A_t=(A^{\text{paper}}_t)^\top\). For example, KDA's \((I-\beta kk^\top)\mathrm{Diag}(\alpha)\) becomes \(\mathrm{Diag}(\alpha)(I-\beta kk^\top)\).
The Gated DeltaNet paper, RWKV-7 (row vectors, \(wkv\in\mathbb R^{d_v\times d_k}\)), Comba and HGRN2 already use the book's orientation.

---

## 1. Coverage audit

**What the book already has (chapters 04, 05, 08 and 10, plus the matrix):**
- Kernel linear attention (Katharopoulos 2020) with a normalizer, and the \(d_vd_k+d_k\) state count.
- RetNet's fixed per-head \(\gamma\), with parallel, recurrent and chunkwise forms.
- GLA's key-channel diagonal decay.
- Mamba and Mamba-2/SSD in prose only, with no formula.
- DeltaNet derived as an SGD step, with an orthogonal-key toy example.
- Gated DeltaNet (GDN), using the faded-state ordering. The numbers are correct: [3.3, 3.2] vs 3.1.
- KDA as \(D_t(I-\beta kk^\top)\), with a note that the two factors do not commute.
- Chunkwise composition, explained conceptually.
- Ch08: the Qwen3-Next 36/12 and Kimi Linear 3:1 hybrids, gate locations, a state+KV memory formula, and the Qwen3.8-Next GDN+QSA design.
- Ch10: FG²-GDN and its "+" variant, GDN-2 and KDN, each with research-scope boxes.

**Formulas checked against primary sources:**
- **Ch04/05 core rules: correct.** RetNet Eq. 6 and 8 ([html](https://arxiv.org/html/2307.08621v4)), GLA Eq. 3 ([html](https://arxiv.org/html/2312.06635v6)), GDN Eq. 10 ([html](https://arxiv.org/html/2412.06464v3); it uses the same \(d_v\times d_k\) orientation as the book) and KDA Eq. 1 ([html](https://arxiv.org/html/2510.26692v1)) all match after transposition.
- **Ch10: all three formulas correct.**
  - FG²-GDN Eq. 10–11 and the "+" variant's Eq. 13 match the book's transposed form. They need \(d_k=d_v\); the paper uses 128/128.
  - GDN-2 Eq. 8–12 match, and setting \(b=w=\beta\mathbf 1\) recovers KDA (paper §3.1, App. A.5).
  - KDN Eq. 10–22 match: diagonal \(p_t\in\mathbb R^{d_k}\), isotropic per-head scalar, and a write along the gain \(\kappa_t\). The 9+9 hybrid uses a 2K window; the HTML shows "22K" in places, which is a rendering error.

**Errors and oversimplifications, with corrections:**
1. **Ch08: Kimi Linear "3 KDA + 1 MLA, repeated" is nominal.** The released 48B-A3B config has 27 layers: 20 KDA and 7 MLA (layers 4, 8, …, 24 and 27). The last group is therefore 2:1. KDA uses 32 heads × 128 dims and short conv 4; MLA uses NoPE with kv_lora 512 + rope 64. [config.json](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/config.json)
2. **Ch04: GLA's "broader formulation allows gates on key and value dimensions."** The deployed GLA uses \(G_t=\alpha_t^\top\mathbf 1\), a key-side gate only. Footnote 5 says the \(\alpha^\top\beta\) form gave "only marginal improvements." The gate also has a temperature: \(\alpha_t=\sigma(\cdot)^{1/16}\), which keeps decay near 1. [GLA §4.1, §4.4](https://arxiv.org/html/2312.06635v6)
3. **Ch04: normalizer.** The book introduces \(z_t\) and the \(d_vd_k+d_k\) state, then silently drops \(z_t\) for RetNet and later rules. State explicitly that production linear attention removed the denominator because it causes unbounded gradients.
   - TransNormer, which became Lightning, replaced it with output normalization ([The Devil in Linear Transformer, EMNLP 2022](https://arxiv.org/abs/2210.10340)).
   - GLA, GDN and KDA use L2-normalized q/k, per-head RMSNorm and an output gate instead.
   - mLSTM is the exception: it keeps \(n_t\) and divides by \(\max(|n_t^\top q|,e^{-m_t})\) ([xLSTM 7B](https://arxiv.org/html/2503.13427)).
4. **Ch04: Mamba-2 in prose only.** Missing facts:
   - Its transition is a scalar per head, which is *coarser* than GLA's or Mamba-1's.
   - \(\Delta_t\) couples decay \(a_t=e^{\Delta_tA}\) with write strength.
   - The exact dictionary to attention is v=x, k=B, q=C.
   - See the card below and [SSD §2.1, §2.4](https://arxiv.org/html/2405.21060).
5. **Ch05: "Computing the residual from the unfaded value … is a different recurrence."** That recurrence is a published family, so it should be named.
   - The sum form \(S_{t-1}(\alpha I-\beta kk^\top)+\beta vk^\top\) is used by Comba ([Eq. 5](https://arxiv.org/html/2506.02475)), RWKV-7 (\(\mathrm{diag}(w)-\hat\kappa(a\odot\hat\kappa)^\top\), [Eq. 16–17](https://arxiv.org/html/2503.14456)) and linear-memory Titans, whose gradient is taken at the undecayed \(M_{t-1}\) ([Eq. 13–14](https://arxiv.org/html/2501.00663)).
   - The book's 3.1 vs 3.3 example is exactly SPLR (scalar plus low-rank) vs the product form (identity plus low-rank, scaled).
6. **Ch05: "bounded update parameters matter for stability."** This is true but incomplete. With \(\|k\|=1\) and \(\beta\in[0,1]\), the eigenvalues of \(I-\beta kk^\top\) lie in [0,1].
   - Finite-precision LRNNs with only positive eigenvalues cannot solve parity.
   - Allowing \(\beta\in[0,2]\) (eigenvalue \(1-\beta\in[-1,1]\)) fixes this. [Grazzi et al., ICLR 2025](https://arxiv.org/abs/2411.12537)
   - This is shipped: Olmo Hybrid 7B sets `linear_allow_neg_eigval: true` ([config](https://huggingface.co/allenai/Olmo-Hybrid-7B/blob/main/config.json)).
7. **Ch10 opening: "KDA … one scalar delta strength" frames per-channel write strength as 2026 news.** RWKV-7 (March 2025) already used a *vector* in-context learning rate \(a_t\in(0,1)^{d}\) and a vector decay. FG²-GDN's contribution is bringing per-channel β into the GDN/KDA kernel family. [RWKV-7 Eq. 4, Table 1](https://arxiv.org/html/2503.14456)
8. **Ch10 erase/write section should separate two kinds of decoupling.**
   - *Strength* decoupling: FG²-GDN+ and GDN-2 (same address \(k_t\), separate gates).
   - *Address* decoupling: Erase-then-Delta Attention from the Qwen team, June 2026, adds a separate erase direction \(e_t\). [EDA Eq. 8](https://arxiv.org/html/2606.26560)
9. **Ch04/08: per-head state count.** Qwen-style GDN has more V heads than QK heads: Qwen3-Next has 32 V / 16 QK and Qwen3.5-397B has 64 V / 16 QK. Count \(d_vd_k\) per **V** head. The decode cache also holds a short-conv state (kernel 4). Prefix caching needs a state snapshot at the exact prefix boundary. [Qwen3-Next card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct), [Qwen3.5 config](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json), [vLLM K3 blog](https://vllm.ai/blog/2026-07-22-kimi-k3-preview)
10. **Attribution, minor (ch05).** The "one SGD step on ½‖Sk−v‖²" reading is the Widrow–Hoff/LMS rule. The modern side-by-side table of objectives is GDN Table 1 ([html](https://arxiv.org/html/2412.06464v3)). Schlag et al. frame it as retrieve–remove–write. Keep the citation but add GDN Table 1.
11. **Missing from the whole book:** state tracking (negative eigenvalues, Householder products, complex/rotational transitions); the online-learning/TTT branch (TTT, Titans, Mesa, ATLAS, LaCT, TTT-E2E); production counter-evidence (MiniMax M2 returning to full attention); and the 2026 adoption wave (Qwen3.5, Kimi K3, Olmo Hybrid, Nemotron 3).

---

## 2. Gap table (ranked)

| Pri | Mechanism | Year | Venue / org | URL | What changes vs. predecessor (≤15 words) | Adopted in | Chapter |
|---|---|---|---|---|---|---|---|
| MUST | Transition-matrix taxonomy (synthesis card) | — | GDN Tbl 1; DeltaProduct; Grazzi | [GDN](https://arxiv.org/abs/2412.06464) | One axis: \(A_t\) goes from identity to scalar, diagonal, Householder, product, rotation | — | 05 (closing card) |
| MUST | Online-regression view (test-time regression + Miras) | 2025 | arXiv (Wang/Shi/Fox); Google | [TTR](https://arxiv.org/abs/2501.12352), [Miras](https://arxiv.org/abs/2504.13173) | Recurrence = optimizer step on a key→value loss; choose loss, retention, optimizer | framing | NEW: Memory as online learning |
| MUST | Mamba-2 formula + mapping (upgrade) | 2024 | ICML 2024 | [2405.21060](https://arxiv.org/abs/2405.21060) | Scalar-per-head decay; v=x, k=B, q=C; Δ couples decay and write | Nemotron 3 Nano, Nemotron-H, Granite 4.0 | 04 |
| MUST | Mamba-3 | 2026 | ICLR 2026 (oral) | [2603.15569](https://arxiv.org/abs/2603.15569) | Trapezoid two-tap write; complex state = data-dependent RoPE; MIMO rank-R | none verified | 04 |
| MUST | Lightning attention + MiniMax hybrid; M2 reversal | 2024–25 | MiniMax | [01](https://arxiv.org/abs/2501.08313), [M1](https://arxiv.org/abs/2506.13585), [M2 blog](https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model) | Fixed per-head slope decay, tiled; 7:1 hybrid; M2 abandoned it for quality and infra reasons | MiniMax-Text-01, M1; dropped in M2 | 04 card + 08 |
| MUST | RWKV-7 "Goose" (generalized delta rule) | 2025 | arXiv; RWKV/EleutherAI | [2503.14456](https://arxiv.org/abs/2503.14456) | Vector decay and vector ICL rate; removal key ≠ replacement key; sum form | RWKV-7 World 0.1–2.9B | 05 |
| MUST | Negative eigenvalues (β∈[0,2]) | 2024/25 | ICLR 2025 | [2411.12537](https://arxiv.org/abs/2411.12537) | Allows eigenvalue 1−β<0; enables parity and state tracking | Olmo Hybrid 7B (`allow_neg_eigval`) | 05 |
| MUST | DeltaProduct | 2025 | NeurIPS 2025 | [2502.10297](https://arxiv.org/abs/2502.10297) | n_h Householder steps per token: identity + rank-n_h transition | research | 05 |
| MUST | KDA chunkwise / constrained DPLR (upgrade) | 2025 | Moonshot | [2510.26692](https://arxiv.org/abs/2510.26692) | DPLR with a=βk, b=α⊙k; ≈2× faster than general DPLR | Kimi Linear, Kimi K3 | 05 + 09 |
| MUST | Production state budgets 2026 (adoption update) | 2025–26 | Qwen, Moonshot, Ai2, NVIDIA | see card | 3:1 GDN/KDA hybrids; ~0.5–1M state elements per recurrent layer | Qwen3.5, Qwen3.8-Next, Kimi K3, Olmo Hybrid, Nemotron 3 | 08 |
| MUST | TTT layers (TTT-Linear/MLP) | 2024 | ICML 2025 | [2407.04620](https://arxiv.org/abs/2407.04620) | State = model weights; update = gradient step on learned-view reconstruction | research | NEW |
| MUST | Titans | 2024/25 | NeurIPS 2025 (Google) | [2501.00663](https://arxiv.org/abs/2501.00663) | Deep MLP memory; GD + momentum ("surprise") + weight decay; attention combos | research | NEW |
| MUST | MesaNet / Mesa layer | 2025 | Google PoI + MIT, arXiv | [2506.05233](https://arxiv.org/abs/2506.05233) | Exact discounted ridge-regression solution at read (CG), not one step | research | NEW |
| SHOULD | xLSTM / mLSTM | 2024 | NeurIPS 2024; xLSTM 7B (NXAI) | [2405.04517](https://arxiv.org/abs/2405.04517), [7B](https://arxiv.org/abs/2503.13427) | Exponential input gate + stabilizer; keeps normalizer; scalar gates | xLSTM 7B | 04 |
| SHOULD | Comba / SPLR ordering | 2025 | arXiv (HKUST-GZ, Shanghai AI Lab) | [2506.02475](https://arxiv.org/abs/2506.02475) | αI−β̃kkᵀ (sum, not product) + output correction q−dk | research | 05 tip |
| SHOULD | Longhorn | 2024 | arXiv (UT Austin) | [2407.14207](https://arxiv.org/abs/2407.14207) | Implicit (proximal) online step; forget gate derived from key | research | NEW (ladder) |
| SHOULD | Log-Linear Attention | 2025 | ICLR 2026 (MIT, Princeton, CMU) | [2506.04761](https://arxiv.org/abs/2506.04761) | O(log T) Fenwick-bucket states with data-dependent level weights | research | 10 or 04 |
| SHOULD | PaTH attention | 2025 | NeurIPS 2025 (MIT, IBM et al.) | [2505.16381](https://arxiv.org/abs/2505.16381) | Softmax with accumulated data-dependent Householder position transforms | research (SmolLM2 conversion) | 10 (or 01 bridge) |
| SHOULD | Forgetting Transformer (FoX) | 2025 | ICLR 2025 | [2503.02130](https://arxiv.org/abs/2503.02130) | Softmax logits + Σ log f: scalar data-dependent decay inside softmax | research | 08 or 10 |
| SHOULD | ATLAS (Omega rule) | 2025 | Google, arXiv | [2505.23735](https://arxiv.org/abs/2505.23735) | Sliding-window loss over c tokens; Muon optimizer; polynomial features | research | NEW |
| SHOULD | LaCT (large-chunk TTT) | 2025 | ICLR 2026 (MIT/Adobe) | [2505.23884](https://arxiv.org/abs/2505.23884) | One update per 2K–1M-token chunk; SwiGLU fast weights; window attention | 14B AR video diffusion (research) | NEW |
| SHOULD | TTT-E2E | 2025 | arXiv (Astera, Stanford, Berkeley, UCSD, NVIDIA) | [2512.23675](https://arxiv.org/abs/2512.23675) | Next-token TTT on last-¼ MLPs of an SWA Transformer; meta-learned init | research 3B | NEW |
| SHOULD | Jet-Nemotron / JetBlock (PostNAS) | 2025 | NVIDIA, arXiv | [2508.15884](https://arxiv.org/abs/2508.15884) | Retrofit: search full-attention placement; GDN + dynamic conv on V | Jet-Nemotron 2B/4B | 08 |
| SHOULD | Erase-then-Delta Attention (EDA) | 2026 | Qwen team + NJU, arXiv | [2606.26560](https://arxiv.org/abs/2606.26560) | Separate erase address e_t: extra Householder before delta write | research (25B-A2.8B MoE) | 10 |
| SHOULD | Recall–state tradeoff (Based) | 2024 | ICML 2024 (Stanford) | [2402.18668](https://arxiv.org/abs/2402.18668) | Recall rises with state size; LA + small SWA spans the Pareto curve | evidence | 04 |
| SKIP | S4 / H3 / Hyena | 2021–23 | ICLR'22/'23, ICML'23 | lineage | Lineage line in 04: LTI SSM → gated conv → selective SSM | — | 04 footnote |
| SKIP | HGRN2 | 2024 | COLM 2024 | [2404.07904](https://arxiv.org/abs/2404.07904) | GLA-like with key tied to 1−f; layerwise forget lower bound | — | 04 footnote |
| SKIP | Gated Slot Attention | 2024 | NeurIPS 2024 | [2409.07146](https://arxiv.org/abs/2409.07146) | Two-pass GLA with softmax over m=64 slots | — | — |
| SKIP | RWKV-4/5/6 | 2023–24 | EMNLP-F'23, COLM'24 | [Eagle/Finch](https://arxiv.org/abs/2404.05892) | Lineage to RWKV-7 | — | — |
| SKIP | Complex KDA | 2026-09 | arXiv | [2609.24797](https://arxiv.org/abs/2609.24797) | Gates ∈[−1,1], β∈[0,2]; rotations without n_h=2 | — | tip in DeltaProduct card |
| SKIP | Preconditioned DeltaNet | 2026-04 | arXiv (MIT) | [2604.21100](https://arxiv.org/abs/2604.21100) | Diagonal curvature preconditioning of delta/GDN/KDA | — | ladder footnote |
| SKIP | Nested Learning / HOPE | 2025 | NeurIPS 2025 (Google) | [2512.24695](https://arxiv.org/abs/2512.24695) | Multi-frequency memory levels; self-modifying Titans | — | NEW chapter footnote |
| SKIP | Miras instances (Moneta, Yaad, Memora) | 2025 | Google | [2504.13173](https://arxiv.org/abs/2504.13173) | ℓp / Huber / KL-retention variants | — | merged into framework card |

---

## 3. Unified update-rule table (book convention: \(S_t=S_{t-1}A_t+W_t\), read \(S_tq_t\))

Class key:
- **0** = no transition.
- **σ** = scalar (×I).
- **D** = diagonal on key channels.
- **⊙** = elementwise (Hadamard).
- **P** = product (decay ∘ Householder); the erase reads the **decayed** state.
- **Σ** = sum (decay − low rank); the erase reads the **undecayed** state.
- **H** = multiple Householders.
- **R** = rotation.

| Model | Class | Transition \(A_t\) | Write term \(W_t\) | Gate granularity | Year / source (checked) |
|---|---|---|---|---|---|
| Linear attention | 0 | \(I\) | \(v_t\phi(k_t)^\top\); plus \(z_t=z_{t-1}+\phi(k_t)\) | none | 2020 ICML, [§3.2–3.4](https://proceedings.mlr.press/v119/katharopoulos20a.html) |
| RetNet | σ | \(\gamma_hI\), \(\gamma_h=1-2^{-5-h}\) (Eq. 8; experiments use a different fixed schedule, §3.1) | \(v_tk_t^\top\) (xPos rotation on q, k) | fixed scalar per head | 2023, [Eq. 6, 8](https://arxiv.org/html/2307.08621v4) |
| Lightning (MiniMax-01) | σ | \(\lambda_{h,\ell}I\), \(\lambda=\exp(-s_h(1-\tfrac{\ell}{L-1}))\), \(s_h\) = ALiBi slopes | \(v_tk_t^\top\) | fixed scalar per head per layer | 2025, [HF modeling code](https://huggingface.co/MiniMaxAI/MiniMax-Text-01/blob/main/modeling_minimax_text_01.py) |
| mLSTM (xLSTM) | σ | \(f_tI\) (σ or exp, stabilized by \(m_t\)) | \(i_tv_tk_t^\top\) (exp input gate); plus \(n_t\) | data-dependent scalar per head | NeurIPS 2024; [7B eqs](https://arxiv.org/html/2503.13427) |
| Mamba-2 (SSD) | σ | \(a_tI\), \(a_t=e^{\Delta_tA_h}\) | \(\Delta_tx_tB_t^\top\) (v=x, k=B, q=C) | data-dependent scalar per head (Δ shared by decay and write) | ICML 2024, [Eq. 2a–b, §2.4](https://arxiv.org/html/2405.21060) |
| Mamba-1 | ⊙ | \(S_{t-1}\mapsto e^{\Delta_t\otimes A}\odot S_{t-1}\) (not a right-multiplication) | \((\Delta_t\odot x_t)B_t^\top\) | per value channel × per state | 2023 (per paper §3; not re-fetched) |
| GLA | D | \(\mathrm{Diag}(\alpha_t)\), \(\alpha_t\in(0,1)^{d_k}\), τ=16 | \(v_tk_t^\top\) | per key channel | ICML 2024, [Eq. 3, §4.4](https://arxiv.org/html/2312.06635v6) |
| HGRN2 | D | \(\mathrm{Diag}(f_t)\) | \(i_t(1-f_t)^\top\) (key tied to 1−f) | per key channel + layer lower bound | COLM 2024, [Eq. 2–3](https://arxiv.org/html/2404.07904) |
| RWKV-6 | D | \(\mathrm{diag}(w_t)\), data-dependent | \(v_tk_t^\top\) | per channel | **UNVERIFIED** (abstract only) |
| Longhorn (deployed) | ⊙ | \((\mathbf 1-\varepsilon_t(k_t^{\odot2})^\top)\odot S_{t-1}\) | \((\varepsilon_t\odot x_t)k_t^\top\) | per value channel ε, key-derived decay | 2024, [Eq. 7](https://arxiv.org/html/2407.14207) |
| Longhorn (exact) | — | \(S_t=S_{t-1}+\mathrm{diag}(\varepsilon_t)(x_t-S_{t-1}k_t)k_t^\top\), \(\varepsilon=\beta/(1+\beta\|k\|^2)\) | (inside) | per value channel | [Thm 3.1, Eq. 6](https://arxiv.org/html/2407.14207) |
| DeltaNet | P | \(I-\beta_tk_tk_t^\top\) | \(\beta_tv_tk_t^\top\) | scalar β per head | 2021/2024, [Eq. 4–11](https://arxiv.org/html/2406.06484) |
| DeltaNet −eig | P | same, \(\beta_t=2\sigma(\cdot)\in(0,2)\) | \(\beta_tv_tk_t^\top\) | scalar | ICLR 2025, [abs](https://arxiv.org/abs/2411.12537) |
| Gated DeltaNet | P | \(\alpha_t(I-\beta_tk_tk_t^\top)\) | \(\beta_tv_tk_t^\top\) | scalar α, scalar β | ICLR 2025, [Eq. 10](https://arxiv.org/html/2412.06464v3) |
| KDA | P | \(\mathrm{Diag}(\alpha_t)(I-\beta_tk_tk_t^\top)\) | \(\beta_tv_tk_t^\top\) | per key-channel α, scalar β | 2025, [Eq. 1](https://arxiv.org/html/2510.26692v1) |
| FG²-GDN | P | \(\mathrm{Diag}(\alpha_t)(I-\tilde k_t\tilde k_t^\top)\), \(\tilde k=\sqrt{\boldsymbol\beta}\odot k\) | \(\tilde v_t\tilde k_t^\top\), \(\tilde v=\sqrt{\boldsymbol\beta}\odot v\) (\(d_k=d_v\)) | per-channel α, per-channel β | 2026, [Eq. 10–11](https://arxiv.org/html/2604.19021v3) |
| FG²-GDN+ | P | \(\mathrm{Diag}(\alpha_t)(I-\hat k_t\hat k_t^\top)\), \(\hat k=\sqrt{\beta^k}\odot k\) | \(\tilde v_t\hat k_t^\top\), \(\tilde v=\sqrt{\beta^v}\odot v\) | per key channel and per value channel | 2026, [Eq. 13](https://arxiv.org/html/2604.19021v3) |
| GDN-2 | P | \(\mathrm{Diag}(\alpha_t)(I-(b_t\odot k_t)k_t^\top)\) | \((w_t\odot v_t)k_t^\top\) | per-key-channel erase b, per-value-channel write w | 2026, [Eq. 8–12](https://arxiv.org/html/2605.22791v1) |
| KDN (diagonal) | P | \(\mathrm{Diag}(\alpha_t)(I-k_t\kappa_t^\top)\), \(\kappa=\hat p\odot k/(r+\sum_i\hat p_ik_i^2)\) | \(v_t\kappa_t^\top\) | per key channel, from uncertainty state \(p_t\) | 2026, [Eq. 10–21](https://arxiv.org/html/2609.07816v2) |
| KDN (isotropic) | P | \(\mathrm{Diag}(\alpha_t)(I-\beta_tk_tk_t^\top)\), β from a scalar covariance | \(\beta_tv_tk_t^\top\) | scalar per head | 2026, [Eq. 22](https://arxiv.org/html/2609.07816v2) |
| EDA | P+H | \(\mathrm{Diag}(\alpha_t)(I-\gamma_te_te_t^\top)(I-\beta_tk_tk_t^\top)\) | \(\beta_tv_tk_t^\top\) | per-channel α, scalar γ and β; separate erase address e | 2026, [Eq. 8–13](https://arxiv.org/html/2606.26560) |
| Comba | Σ | \(\alpha_tI-\tilde\beta_tk_tk_t^\top\), \(\tilde\beta=b\odot\beta\), \(b\in(0,1)\) | \(\beta_tv_tk_t^\top\); read \(S_t(q_t-dk_t)\) | scalar α and β (shape of b **UNVERIFIED**) | 2025, [Eq. 5, Tbl 1](https://arxiv.org/html/2506.02475) |
| RWKV-7 | Σ | \(\mathrm{diag}(w_t)-\hat\kappa_t(a_t\odot\hat\kappa_t)^\top\) | \(v_t\tilde k_t^\top\), \(\tilde k=k\odot(1+(a-1)\odot\alpha)\) | per-channel w, per-channel ICL rate a | 2025, [Eq. 4, 6–7, 16–17](https://arxiv.org/html/2503.14456) |
| Titans (linear M, η=0) | Σ | \((1-\alpha_t)I-\theta'_tk_tk_t^\top\) (θ′=2θ) | \(\theta'_tv_tk_t^\top\); with momentum add \(\eta_t\,\mathcal S_{t-1}\) | scalar per token (chunk-constant option) | NeurIPS 2025, [Eq. 12–14](https://arxiv.org/html/2501.00663) (reduction is my algebra) |
| TTT-Linear (no LN, b=1) | P | \(I-2\eta_tk_tk_t^\top\) | \(2\eta_tv_tk_t^\top\) | scalar η per token | ICML 2025, [Eq. 2, 4](https://arxiv.org/html/2407.04620); real f has LN + residual, b=16 |
| DeltaProduct\(_{n_h}\) | H | \(\prod_{j=1}^{n_h}(I-\beta_{t,j}k_{t,j}k_{t,j}^\top)\); gated variant ×\(g_t\) | \(\sum_j\beta_{t,j}v_{t,j}k_{t,j}^\top\prod_{l>j}(I-\beta_{t,l}k_{t,l}k_{t,l}^\top)\) | scalar per sub-step, β∈[0,2] | NeurIPS 2025, [Eq. 3, Prop. 1](https://arxiv.org/html/2502.10297) |
| Mamba-3 (SISO) | σR | \(\alpha_t\mathcal R_t\) (block-diagonal 2×2 rotations by \(\Delta_t\theta_t\)) | \(\gamma_tx_tB_t^\top+\beta_tx_{t-1}B_{t-1}^\top\) (rotation placement on the β term follows Prop. 4; not checked term-by-term) | scalar decay per head + per-pair angle | ICLR 2026, [Eq. 5–11](https://arxiv.org/html/2603.15569) |
| Mesa layer | σ (2 states) | \(G\): \(\gamma_tI\); \(H\): \(\gamma_tI\) | \(G\mathrel{+}=\beta_tv_tk_t^\top\), \(H\mathrel{+}=\beta_tk_tk_t^\top\); read \(G_t(H_t+\Lambda)^{-1}q_t\) | scalar γ and β | 2025, [Eq. 7–8](https://arxiv.org/html/2506.05233) |
| GSA | D (slots) | \(K,V\in\mathbb R^{d\times m}\): \(\cdot\,\mathrm{Diag}(\alpha_t)\) | \(k_t(1-\alpha_t)^\top\), \(v_t(1-\alpha_t)^\top\); read \(V\,\mathrm{softmax}(K^\top q)\) | per slot, m=64 | NeurIPS 2024, [Eq. 6](https://arxiv.org/html/2409.07146) |
| Log-linear X | any | each level \(S^{(\ell)}\) uses X's transition | read \(\sum_\ell\lambda_t^{(\ell)}S_t^{(\ell)}q_t\) | per-level λ | ICLR 2026, [Eq. 3–5](https://arxiv.org/html/2506.04761) |
| FoX (softmax) | — | no state; logit \(q_i^\top k_j+\sum_{l=j+1}^i\log f_l\) | — | scalar per head | ICLR 2025 (formula via PaTH-FoX restatement) |
| PaTH (softmax) | — | logit \(k_j^\top(\prod_{s=j+1}^iH_s)q_i\), \(H=I-\beta ww^\top\), β∈(0,2) | — | — | NeurIPS 2025, [§3, Eq. 1](https://arxiv.org/html/2505.16381) |

**Online-regression ladder (the online-learning reading of the table):**

| Model | Per-token loss \(\ell_t(S)\) | Retention | Optimizer | Source |
|---|---|---|---|---|
| Linear attention | \(-\langle Sk_t,v_t\rangle\) | none | GD, step 1 | GDN Tbl 1; Miras Tbl 1 |
| Mamba-2 / RetNet | \(-\langle Sk_t,v_t\rangle\) | \(\|S-\alpha S_{t-1}\|^2\) | GD | GDN Tbl 1 |
| DeltaNet | \(\tfrac12\|Sk_t-v_t\|^2\) | none | GD, step β | GDN Tbl 1 |
| GDN / KDA | same | scalar / diagonal decay | GD | GDN Tbl 1; Kimi Linear |
| Longhorn | \(\beta\|Sk-x\|^2\) | proximal \(\|S-S_{t-1}\|^2\) | implicit GD (closed form) | Longhorn Eq. 5 |
| TTT-Linear/MLP | \(\|f(\theta_Kx;W)-\theta_Vx\|^2\) | none | mini-batch GD, b=16 | TTT Eq. 4 |
| Titans | \(\|M(k)-v\|^2\), MLP M | weight decay \(\alpha_t\) | GD + momentum | Titans Eq. 12–14 |
| ATLAS | \(\sum_{i=t-c+1}^t\gamma_i\|M(\phi(k_i))-v_i\|^2\) | gate | Muon (Newton–Schulz) | ATLAS Eq. 9, 32–33 |
| Mesa | \(\sum_{i\le t}(\prod\gamma)\beta_i\|Sk_i-v_i\|^2+\|S\|_\Lambda^2\) | γ | exact minimizer (CG) | Mesa Eq. 7–8 |
| KDN | linear-Gaussian observation \(v=Sk+\text{noise}\) | process model D, ω | Kalman gain | KDN Eq. 10–21 |
| LaCT | \(-\langle f_W(k),v\rangle\), SwiGLU W | L2 weight norm | GD or Muon per chunk | LaCT §3 |
| TTT-E2E | next-token CE on context | — | GD on last-¼ MLPs, 1K mini-batch | TTT-E2E |
| Softmax attention | Nadaraya–Watson (non-parametric) | — | — | TTT Thm 2 |

---

## 4. Card data

### Transition-matrix taxonomy (synthesis card)
- **url:** [GDN Table 1](https://arxiv.org/html/2412.06464v3), [Grazzi](https://arxiv.org/abs/2411.12537), [DeltaProduct](https://arxiv.org/abs/2502.10297), [Mamba-3](https://arxiv.org/abs/2603.15569)
- **one-line idea:** All of chapters 04–05 are \(S_t=S_{t-1}A_t+W_t\); the design axis is the structure of \(A_t\).
- **key formula:** \(A_t\in\{I,\ \gamma I,\ \alpha_tI,\ \mathrm{Diag}(\alpha_t),\ I-\beta kk^\top,\ \alpha(I-\beta kk^\top),\ \mathrm{Diag}(\alpha)(I-\beta kk^\top),\ \mathrm{Diag}(w)-\hat\kappa(a\odot\hat\kappa)^\top,\ \prod_j(I-\beta_jk_jk_j^\top),\ \alpha\mathcal R_t\}\)
- **concrete numbers:**
  - Eigenvalues of \(I-\beta kk^\top\) with \(\|k\|=1\): \(d_k-1\) copies of 1, plus \(1-\beta\).
  - β∈[0,1] gives [0,1]; β∈[0,2] gives [−1,1].
  - GDN multiplies all eigenvalues by α.
  - Two reflections (β=2) compose to a rotation by 2θ, where θ = ∠(k₁, k₂).
- **diagram idea:** A row of 8 small 8×8 heatmaps of \(A_t\), one per class.
  - Under each, a unit disc with its eigenvalues: a point at 1; a point at γ; dots on [0,1]; one dot at 1−β; one dot below 0; pairs on a circle (rotation).
  - A bottom strip of model names and badges: "parity ✓/✗", "chunk algorithm: cumsum / WY-UT / UT×n_h".
- **tips:**
  - Diagonal classes commute, so a chunk only needs cumulative products (log-space cumsum). Householder classes need WY/UT transforms (Parallel DeltaNet Eq. 10–11).
  - Product (P) vs sum (Σ) decides whether the erase reads the faded or the unfaded state. GDN/KDA are P; RWKV-7/Comba/Titans are Σ.
  - All classes keep the same \(d_v\times d_k\) state. A richer \(A_t\) buys expressivity (tracking, editing), not capacity.
- **verification:** Each row was checked against the equations listed in §3. The eigenvalue statements follow directly from the algebra. The parity claim comes from the Grazzi ICLR 2025 abstract.

### Online-regression view (test-time regression + Miras)
- **url:** [Wang, Shi, Fox 2025](https://arxiv.org/abs/2501.12352); [Miras (Behrouz et al., Google)](https://arxiv.org/abs/2504.13173)
- **one-line idea:** The state is a regressor's parameters; each token takes one optimizer step on a key→value loss; the read is a prediction at q.
- **key formula:** \(S_t=\arg\min_S\ \underbrace{\mathrm{Ret}(S,S_{t-1})}_{\text{retention}}+\eta_t\,\underbrace{\ell(S;k_t,v_t)}_{\text{attentional bias}}\). Choosing ℓ, Ret and the optimizer gives each row of the §3 ladder.
- **concrete numbers:**
  - Miras Table 1 classifies linear attention, RetNet, Mamba-2, DeltaNet, GDN, TTT, Titans, Longhorn and RWKV-7 by memory, bias, retention and algorithm.
  - Miras' 2025 variants: Moneta (ℓ_p, p=3, q=4), Yaad (Huber), Memora (KL retention, softmax-normalized W).
  - TTR's three choices: regression weights, function class, optimizer.
- **diagram idea:** A 2-D map.
  - x-axis: how much history the loss sees (current token → window c → discounted full history).
  - y-axis: optimizer strength (GD → implicit GD → momentum → Muon → exact or Kalman).
  - Place LA, DeltaNet, GDN, Longhorn, TTT, Titans, ATLAS, Mesa and KDN as dots. Put softmax attention in a "non-parametric" corner (Nadaraya–Watson, TTT Thm 2).
- **tips:**
  - This is a *reading* of fixed algebra. The outer loop trains the projections, and the inner "learning" has no free choices at inference.
  - Chunk/mini-batch gradients taken at chunk-start weights (TTT b=16, Titans, LaCT) are a different model from per-token steps.
  - The view explains the update rules; it does not lift the capacity limit.
- **verification:** TTR abstract (three choices); Miras §4 and Table 1, Eq. 24, 26, 27; GDN Table 1 objectives.

### Mamba-2 (upgrade card for ch04)
- **url:** https://arxiv.org/abs/2405.21060 (ICML 2024)
- **one-line idea:** SSD restricts the SSM transition to scalar×identity per head. That makes it exactly a decayed linear attention with v=x, k=B, q=C.
- **key formula:** \(S_t=a_tS_{t-1}+\Delta_t\,x_tB_t^\top,\ a_t=e^{\Delta_tA_h}\ (A_h<0);\quad y_t=S_tC_t+Dx_t,\ S\in\mathbb R^{P\times N}\)
- **concrete numbers:**
  - Nemotron 3 Nano 30B-A3B has 52 layers: 23 Mamba-2, 23 MoE, 6 GQA.
  - Mamba-2 uses 64 heads × P=64 × N=128, which is 524,288 state elements per layer.
  - n_groups=8 means B and C are shared across heads, like GQA for keys and queries.
  - chunk_size=128, conv kernel 4. ([config](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/config.json))
  - The SSD paper reports 2–8× faster than Mamba's selective scan and P typically 64 or 128.
- **diagram idea:** A dual-label figure: a 64×128 state heatmap with "v/x" on rows and "k/B" on columns. Show q/C reading it, and a scalar decay dial \(a_t\) beside it. A side strip shows 8 groups of heads sharing one B/C pair.
- **tips:**
  - Mamba-2's decay is *coarser* than Mamba-1 (per channel × state) or GLA (per key channel); SSD bought matmul speed and large N.
  - Δ appears in both the decay and the write, so a strong write also forgets more. GDN separates α and β.
  - It has no delta correction; Gated DeltaNet is literally "Mamba-2 + delta rule" (its paper's title).
- **verification:** Eq. 2a–2b, §2.1 (scalar×identity), §2.4 (P values); Nemotron config.json fields.

### Mamba-3
- **url:** https://arxiv.org/abs/2603.15569 (ICLR 2026; Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu)
- **one-line idea:** Mamba-2 plus three changes derived from SSM principles:
  - exponential-trapezoidal discretization, which gives a two-tap write;
  - a complex-valued state, which is equivalent to data-dependent RoPE on B and C;
  - a MIMO rank-R write.
- **key formula:** \(S_t=\alpha_tS_{t-1}\mathcal R_t+\gamma_tx_tB_t^\top+\beta_tx_{t-1}B_{t-1}^\top\), with \(\alpha=e^{\Delta A},\ \beta=(1-\lambda)\Delta\alpha,\ \gamma=\lambda\Delta\). For MIMO, \(x_t\in\mathbb R^{P\times R}\) and \(B_t\in\mathbb R^{N\times R}\).
- **concrete numbers:**
  - λ=1 recovers Mamba-2; λ=½ is the classical trapezoid rule.
  - 1.5B models on 100B FineWeb-Edu tokens:

    | Model | Perplexity | Avg. accuracy |
    |---|---|---|
    | GDN | 10.45 | 55.8 |
    | Mamba-2 | 10.47 | 55.7 |
    | Mamba-3 SISO | 10.35 | 56.4 |
    | Mamba-3 MIMO (R=4) | 10.24 | 57.6 |

  - It matches Mamba-2's perplexity at half the state size.
  - Decode FLOPs: ≈4NPR (MIMO) vs ≈5NP (SISO). Latency stays about flat because decode is memory-bound.
  - The short conv is removed. BCNorm on B and C is the analogue of QK-norm.
  - GDN-2's baseline uses Mamba-3 MIMO R=4 with d_state 64.
- **diagram idea:** Three panels.
  - (a) Write taps on a token timeline: Euler (t only) vs trapezoid (t−1, t, with weights β and γ).
  - (b) The N-dimensional state drawn as N/2 rotating 2-D planes; the cumulative angle gives "RoPE on B and C".
  - (c) A roofline: SISO outer product (low intensity) → MIMO (P×R)(R×N) matmul moving right; the state byte count is unchanged.
- **tips:**
  - Rotations put eigenvalues on a circle of radius α. That gives parity and modular counting without β>1.
  - MIMO adds FLOPs, not state bytes, so it helps exactly where decode is bandwidth-bound.
  - No released production LLM using Mamba-3 was found as of 2026-10.
- **verification:** Prop. 1 and Eq. 5–6; Prop. 2–4 and Eq. 9–11; Table 2(b); Table 3; Table 5a (conv ablation). The rotation placement on the β term was not checked term-by-term.

### Lightning attention, the MiniMax hybrid, and the M2 reversal
- **url:** [MiniMax-01](https://arxiv.org/abs/2501.08313); [Lightning-2](https://arxiv.org/abs/2401.04658); [M1](https://arxiv.org/abs/2506.13585); [M2 blog, 2025-10-30](https://huggingface.co/blog/MiniMax-AI/why-did-m2-end-up-as-a-full-attention-model); [modeling code](https://huggingface.co/MiniMaxAI/MiniMax-Text-01/blob/main/modeling_minimax_text_01.py)
- **one-line idea:** TransNormer-style linear attention with no denominator, RMSNorm and a sigmoid output gate, and a *fixed* per-head decay. It is tiled into intra-block left products and inter-block state products, and used in a 7:1 hybrid.
- **key formula:** \(S_t=\lambda_{h,\ell}S_{t-1}+v_tk_t^\top,\ \lambda_{h,\ell}=e^{-s_h(1-\ell/(L-1))};\quad o_t=\sigma(W_gx_t)\odot\mathrm{RMSNorm}(S_tq_t)\)
- **concrete numbers:**
  - MiniMax-01 has 80 layers, with a softmax block after every 7 lightning blocks, giving 70 lightning + 10 softmax.
  - 64 heads × 128 dims; 456B total / 45.9B active parameters.
  - Block size 256 in training; 32, 64 or 128 at inference.
  - 1M-token training context, extrapolating to 4M.
  - Lightning state is 64×128² = 1.05M elements per layer. The softmax GQA (8 KV heads) KV is 2,048 elements per token per layer, so one lightning layer's state ≈ 512 tokens of KV.
  - M1 keeps the same 7:1 design and reports <50% of DeepSeek-R1's FLOPs at 64K generation and ≈25% at 100K.
  - M1's training/inference mismatch: they moved the LM head to FP32, raising the probability correlation from about 0.9x to 0.99x.
- **diagram idea:** Three parts.
  - A T×T causal grid cut into 256×256 tiles: diagonal tiles labelled "(QKᵀ⊙M)V", off-diagonal tiles labelled "Q·(KV) via state".
  - A strip of per-layer λ values rising to ≈1 at the last layer.
  - A side panel titled "M2 went back to full attention" listing the four reasons.
- **tips:**
  - The decay is RetNet-class: fixed, not data-dependent, scaled by depth, and essentially none in the last layer (code: `slope_rate * (1 - idx/(L-1) + 1e-5)`).
  - MiniMax's own report says pure lightning attention has limited retrieval, which is why the softmax layers exist.
  - M2's stated reasons for dropping it:
    - deficits in multi-hop reasoning appeared only at scale;
    - linear attention is "far more sensitive to numerical precision";
    - prefix caching and speculative decoding were immature;
    - conversion to a sliding-window hybrid "degraded noticeably as context length grew."
  - The blog puts the theoretical compute/memory crossover at "a few thousand tokens," which matches the state-equivalence numbers in the production card.
- **verification:** MiniMax-01 §2 (Eq. 5, tiling, 7:1, 80 layers, block sizes); modeling code (slopes, `BLOCK=256`, decode `kv = ratio*kv + k^T v`, sigmoid gate); M1 introduction and §3.2; M2 blog.

### RWKV-7 "Goose"
- **url:** https://arxiv.org/abs/2503.14456 (arXiv 2025; RWKV Project/EleutherAI; venue not confirmed)
- **one-line idea:** A "generalized delta rule" with vector decay, a vector in-context learning rate, and a removal key \(\hat\kappa\) that differs from the replacement key \(\tilde k\). It uses the sum form, not the product.
- **key formula:**
  - \(S_t=S_{t-1}\big(\mathrm{diag}(w_t)-\hat\kappa_t(a_t\odot\hat\kappa_t)^\top\big)+v_t\tilde k_t^\top\)
  - \(w_t=e^{-e^{-0.5}\sigma(d_t)},\ \hat\kappa=\tfrac{k\odot\xi}{\|k\odot\xi\|},\ \tilde k=k\odot(1+(a-1)\odot\alpha)\)
- **concrete numbers:**
  - Head size is 64, so each head has a 64×64 = 4,096-element state.
  - Per-step decay lies in (0.545, 1).
  - World v3 corpus is 3.1T tokens; models range 0.1B–2.9B and were trained on 1.6–5.6T tokens. Some started from RWKV-5/6 checkpoints.
  - The paper claims S₅ state tracking in one layer and recognition of all regular languages with a constant number of layers.
- **diagram idea:** A 64×64 state heatmap, then columns scaled by \(w_t\) (a vertical stripe pattern), minus a rank-1 "removal" heatmap (\(\hat\kappa\) column × \(a\odot\hat\kappa\) row), plus a rank-1 write \(v\tilde k^\top\). Beside it, a bar chart of the vector \(a_t\) labelled "per-channel learning rate."
- **tips:**
  - The decay is *added* to the rank-1 term rather than multiplied by it, so the removal reads the undecayed state. Compare GDN's product form.
  - The vector \(a_t\) gave per-channel write strength a year before FG²-GDN.
  - The expressivity claims assume \(\mathsf{TC}^0\neq\mathsf{NC}^1\) and depend on depth.
  - Jet-Nemotron's search dropped RWKV-7 for low training throughput; kernel maturity matters.
- **verification:** Eq. 4, 6–7, 11–12, 16–17; Table 1 ("generalized Δ rule"); §5–6 sizes. The paper is inconsistent between 0.19B and 0.1B.

### Negative eigenvalues (β ∈ [0, 2])
- **url:** https://arxiv.org/abs/2411.12537 (ICLR 2025; Grazzi, Siems, Zela, Franke, Hutter, Pontil)
- **one-line idea:** Positive-spectrum linear RNNs cannot do parity. Letting the delta transition reflect (eigenvalue 1−β<0) unlocks state tracking.
- **key formula:** \(S_t=S_{t-1}(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top,\ \beta_t=2\sigma(\cdot);\ \ \mathrm{eig}=\{1,\dots,1,\,1-\beta_t\}\subset[-1,1]\)
- **concrete numbers:**
  - Products of (I − outer product) matrices with eigenvalues in [−1,1] can recognize any regular language.
  - Experiments were pretrained at 1.3B.
  - Production: Olmo Hybrid 7B has 32 layers (24 GDN + 8 full MHA), 30 heads, d_k=96, d_v=192, `linear_allow_neg_eigval: true` ([config](https://huggingface.co/allenai/Olmo-Hybrid-7B/blob/main/config.json), [paper](https://arxiv.org/abs/2604.03444)).
  - PaTH also uses β∈(0,2) and notes clipping below 2 in BF16.
- **diagram idea:** A real axis [−1,1] with two shaded ranges (β≤1 vs β≤2). Inset: a 2-state parity automaton as the reflection \(I-2kk^\top\) flipping a vector across a line.
- **tips:**
  - It is a one-line change (2×sigmoid) and keeps the spectral norm ≤1, so it stays stable.
  - It adds expressivity, not memory capacity.
  - At β=2 there is no forgetting along k, only reflection.
- **verification:** ICLR proceedings abstract (via search); Olmo config.json; PaTH §3 (β range, BF16 note).

### DeltaProduct
- **url:** https://arxiv.org/abs/2502.10297 (NeurIPS 2025; Siems, Carstensen, Zela, Hutter, Pontil, Grazzi)
- **one-line idea:** Take n_h delta steps per token, each with its own key, value and β. The transition becomes a product of n_h generalized Householders, i.e. identity plus rank n_h.
- **key formula:** \(A_t=\prod_{j=1}^{n_h}(I-\beta_{t,j}k_{t,j}k_{t,j}^\top),\quad W_t=\sum_j\beta_{t,j}v_{t,j}k_{t,j}^\top\prod_{l>j}(I-\beta_{t,l}k_{t,l}k_{t,l}^\top)\). Gated variant: \(g_tA_t\).
- **concrete numbers:**
  - n_h ∈ {1, 2, 3} (4 for S₅), β∈[0,2].
  - A 392M DeltaProduct₂ (8 heads) is parameter-matched to DeltaNet (12 heads).
  - State tracking on S₃, S₄, A₅, S₅: trained on 128-step sequences, tested to 512.
  - Throughput measured at 1.3B.
  - Kernel cost ≈ running DeltaNet over n_h·T steps (**UNVERIFIED**).
- **diagram idea:** In a 2-D plane, one reflection (mirror line) vs two reflections, which compose to a rotation by 2θ. Next to it, \(A_t\) as an identity heatmap plus a rank-2 patch, and a bar chart of "groups solved vs n_h."
- **tips:**
  - n_h is a compute/expressivity knob that does not grow the state.
  - Complex KDA (arXiv 2026-09) reaches DeltaProduct₂-level tracking with n_h=1 by allowing gates in [−1,1]; the gate supplies the second reflection.
  - EDA (Qwen, 2026) is effectively a 2-step product whose first step writes nothing.
  - No released LLM adoption was found.
- **verification:** §4 Eq. 3; Prop. 1 (β range, norm ≤1); experiment sections.

### KDA chunkwise / constrained DPLR (upgrade card for 05/09)
- **url:** https://arxiv.org/abs/2510.26692 (Moonshot, 2025)
- **one-line idea:** KDA is a DPLR transition \(D-ab^\top\) with both low-rank vectors tied to k. That lets the decay factor out and halves the chunk matmuls compared with general DPLR (the RWKV-7 class).
- **key formula:** \(A_t=\mathrm{Diag}(\alpha_t)-\beta_t(\alpha_t\odot k_t)k_t^\top=\mathrm{Diag}(\alpha_t)(I-\beta_tk_tk_t^\top)\). Chunk UT transform: \(T=(I+\mathrm{tril}(\mathrm{diag}(\beta)KK^\top,-1))^{-1}\mathrm{diag}(\beta)\), \(W=TK\), \(U=TV\).
- **concrete numbers:**
  - Kimi-Linear-48B-A3B: 27 layers (20 KDA / 7 MLA), 32 KDA heads × 128 × 128, short conv 4, MLA with NoPE.
  - C=64 in the FLOP analysis.
  - Chunk matrix computations drop from four to two (plus three fewer matmuls); the KDA kernel is ≈2× general DPLR up to 64K.
  - At 1M tokens: TPOT 1.84 vs 11.48 ms (6.3×, Fig. 1b; §6.3 says 2.3×, unreconciled); prefill 2.9× faster; KV −75%.
  - 5.7T training tokens.
  - α comes from a low-rank projection (rank = head dim); β is a sigmoid. Output = σ(low-rank gate) ⊙ RMSNorm(·).
- **diagram idea:** One 64-token chunk.
  - Left: a 64×64 lower-triangular T heatmap (UT transform).
  - Middle: W=TK and U=TV as tall strips.
  - Right: the incoming state (128×128) → the outgoing state, with per-key-channel cumulative decay shown as a column-shaded mask.
  - Annotate "general DPLR: 4 chunk matrices; KDA: 2."
- **tips:**
  - Tying a and b to k is the efficiency trick; RWKV-7's untied form pays for generality.
  - Quote speedups with batch and context: Fig. 7 uses batch 1 at 1M.
  - Kimi K3 (2026) reuses KDA with MLA every 4th layer (vLLM blog).
- **verification:** Eq. 1; DPLR paragraph and Fig. 2; config.json; Parallel DeltaNet Eq. 10–11 (UT/WY).

### Production state budgets, 2025–26 (adoption card for ch08)
- **url:** [Qwen3-Next](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct), [Qwen3.5-397B config](https://huggingface.co/Qwen/Qwen3.5-397B-A17B/blob/main/config.json), [Qwen3.8-Next](https://arxiv.org/abs/2608.30320), [Kimi Linear config](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/blob/main/config.json), [Kimi K3 (vLLM)](https://vllm.ai/blog/2026-07-22-kimi-k3-preview), [Olmo Hybrid](https://arxiv.org/abs/2604.03444), [Nemotron 3 Nano config](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/config.json)
- **one-line idea:** Shipped hybrids converge on about 3:1 recurrent:attention, with about 0.5–1M state elements per recurrent layer. The whole fixed state equals the KV of only a few thousand tokens.
- **key formula:** \(N^\*=\frac{\sum_{\ell\in\mathcal R}H_\ell d_vd_k}{\sum_{\ell\in\mathcal E}c_\ell}\) is the crossover where the growing KV overtakes the fixed state (\(c_\ell\) = KV elements per token per attention layer).
- **concrete numbers** (elements; my arithmetic from configs):

| Model | Rec : Attn layers | Recurrent head layout | State / layer | Total state (bf16) | Attn KV / token | \(N^\*\) |
|---|---|---|---|---|---|---|
| Qwen3-Next-80B-A3B | 36 GDN : 12 GA | 32 V × 16 QK × 128 | 524,288 | 18.9M (37.7 MB) | 12×(2×2×256) = 12,288 | 1,536 |
| Qwen3.5-397B-A17B | 45 : 15 | 64 V × 16 QK × 128 | 1,048,576 | 47.2M (94 MB) | 15×1,024 = 15,360 | 3,072 |
| Kimi-Linear-48B-A3B | 20 KDA : 7 MLA | 32 × 128² | 524,288 | 10.5M (21 MB) | 7×576 = 4,032 | 2,601 |
| Nemotron 3 Nano 30B-A3B | 23 Mamba-2 : 6 GQA | 64 × 64 × 128 | 524,288 | 12.1M (24 MB) | 6×512 = 3,072 | 3,925 |
| MiniMax-01 | 70 : 10 | 64 × 128² | 1,048,576 | 73.4M (147 MB) | 10×2,048 = 20,480 | 3,584 |
| Olmo Hybrid 7B | 24 GDN : 8 MHA | 30 × 96 × 192 (−eig) | 552,960 | 13.3M | head dim not checked | — |
| Kimi K3 (2.8T) | MLA every 4th of 93 layers | KDA (dims not given) | — | — | — | — |
| Qwen3.8-Next | 3:1 GDN : global → QSA | not given | — | — | — | — |

- **diagram idea:** A log-log plot of bytes vs context N. Each model's flat recurrent-state line meets its rising KV line at \(N^\*\) (1.5K–3.9K). Above it, stacked layer-pattern strips per model (colored cells: GDN, KDA, Mamba, attention).
- **tips:**
  - Beyond about 4K tokens, the 1/4 of layers that are attention dominate memory. Hybrids shrink the KV coefficient; they don't remove the KV term.
  - Count state per V head: Qwen uses 2× (Next) or 4× (3.5) more V heads than QK heads.
  - Prefix caching needs exact-boundary state snapshots: vLLM uses copy-on-write state blocks spanning thousands of tokens (K3 blog).
  - Qwen3.8-Next findings:
    - the sigmoid output gate in GDN beat SiLU;
    - NoPE caused "substantially higher" endless generation after post-training;
    - the GDN hybrid averaged 53.81 vs 51.15 for an SWA hybrid and 49.87 for full attention (25B-A3B ablation).
- **verification:** Each config.json was fetched (fields as listed). Qwen3.8-Next §2.1. The vLLM K3 blog is a vLLM-team source; Moonshot's K3 report was not read. Qwen3.5's release date was not checked.

### TTT layers (TTT-Linear / TTT-MLP)
- **url:** https://arxiv.org/abs/2407.04620 (ICML 2025; Sun, Li, Dalal, …, Hashimoto, Guestrin)
- **one-line idea:** The hidden state is the weights W of a small model. Each update is a gradient step on a self-supervised reconstruction loss with learned views.
- **key formula:** \(W_t=W_{t-1}-\eta_t\nabla\|f(\theta_Kx_t;W_{t-1})-\theta_Vx_t\|^2,\quad z_t=f(\theta_Qx_t;W_t)\). For linear f without LN: \(W_t=W_{t-1}(I-2\eta kk^\top)+2\eta vk^\top\).
- **concrete numbers:**
  - Mini-batch b=16, with gradients at the batch-start W. The dual form trains more than 5× faster than the primal form.
  - \(\eta=\eta_{base}\sigma(\theta_{lr}x)\), with η_base = 1 (Linear) or 0.1 (MLP).
  - \(f(x)=x+\mathrm{LN}(f_{res}(x))\). TTT-MLP uses 4× hidden width and GELU.
  - 125M–1.3B models; contexts 2k/8k (Pile) and 1k–32k (Books3).
  - Thm 1: linear f + batch GD + η=½ + W₀=0 ≡ linear attention. Thm 2: Nadaraya–Watson ≡ softmax attention.
  - Per-head W size **UNVERIFIED**.
- **diagram idea:** A b-slider strip from b=1 (delta rule, sequential) through b=16 (TTT) to b=T (linear attention, no correction), with a W heatmap and gradient arrows at each setting. A two-loop inset shows the inner loop updating W and the outer loop training θ_K, θ_V, θ_Q.
- **tips:**
  - The mini-batch size b interpolates between DeltaNet and linear attention. Within a batch, there is no correction between tokens.
  - LN and the residual inside f make "TTT-Linear" nonlinear, so it is not exactly DeltaNet.
  - TTT-MLP is I/O-bound; LaCT reports earlier TTT at under 5% FLOP utilization.
- **verification:** Eq. 2, 4, 5; Theorems 1–2; mini-batch, dual form and η sections; ICML 2025 per PMLR v267.

### Titans
- **url:** https://arxiv.org/abs/2501.00663 (NeurIPS 2025; Behrouz, Zhong, Mirrokni, Google)
- **one-line idea:** A deep (MLP) memory updated at test time by gradient descent with momentum ("surprise") and weight decay (forgetting). It pairs with attention as context (MAC), gate (MAG) or layer (MAL).
- **key formula:** \(\mathcal M_t=(1-\alpha_t)\mathcal M_{t-1}+\mathcal S_t,\ \ \mathcal S_t=\eta_t\mathcal S_{t-1}-\theta_t\nabla\|\mathcal M_{t-1}(k_t)-v_t\|^2\); read \(y_t=\mathcal M^\*(q_t)\).
- **concrete numbers:**
  - Memory depth L_M ≥ 1; deeper is strictly more expressive; ablation over L_M 1–4.
  - Linear vs deep memory perplexity: 28.49 vs 27.01.
  - Training uses chunked mini-batch GD (gradients at chunk start) with momentum computed by a parallel scan.
  - Claims >2M context on BABILong; S-NIAH Table 2 stops at 16K in v1.
- **diagram idea:** A token stream with a "surprise" trace (‖∇‖) that spikes at "Friday" and decays via η. The memory MLP's weight heatmaps change at the spike. Small MAC/MAG/MAL wiring icons sit underneath.
- **tips:**
  - The gradient is evaluated at the undecayed \(\mathcal M_{t-1}\), so linear Titans is a Σ-class rule like RWKV-7 or Comba, not GDN's product.
  - Momentum makes the memory a two-state linear system: one surprising token keeps writing for several steps.
  - A deep memory loses the closed-form chunk algebra; throughput falls with L_M.
- **verification:** Eq. 12–15; §3.2 (Eq. 16–18); §4 variants; §5.5. Venue from the NeurIPS 2025 poster page and Google Research listing.

### MesaNet / Mesa layer
- **url:** https://arxiv.org/abs/2506.05233 (Google Paradigms of Intelligence, Google DeepMind and MIT; arXiv 2025, venue **UNVERIFIED**)
- **one-line idea:** Rather than taking one gradient step, the layer reads out the exact minimizer of a discounted ridge regression over all past (k, v).
- **key formula:** \(G_t=\gamma_tG_{t-1}+\beta_tv_tk_t^\top,\ \ H_t=\gamma_tH_{t-1}+\beta_tk_tk_t^\top;\quad o_t=G_t(H_t+\Lambda)^{-1}q_t\) (conjugate-gradient solve)
- **concrete numbers:**
  - The extra \(d_k\times d_k\) state H is <1% of total memory (paper).
  - 30 CG steps in the main runs; a stopping rule ε=1e−4 averages about 9 steps.
  - k CG steps ≈ k× GLA FLOPs; 0 steps recovers GLA.
  - 140M, 440M and 1B models on SlimPajama.
- **diagram idea:** Two additive heatmaps, G (d_v×d_k) and H (d_k×d_k). The read path is q → k CG arrows → \((H+\Lambda)^{-1}q\) → G.
  - A contrast panel with two correlated keys (Tuesday/Friday) shows DeltaNet's one-step residual vs Mesa's least-squares answer.
- **tips:**
  - Writes are purely additive and chunk-friendly; the editing happens at read time through the inverse covariance.
  - The number of CG steps is a test-time compute knob.
  - Mesa is RLS/Kalman with no process noise; KDN (2026) is the diagonal or isotropic covariance cousin, and Preconditioned DeltaNet (2026) is the diagonal-curvature cousin.
- **verification:** Eq. 7–8; CG-step ablations; footprint statement. Venue: only an #ICLR2026 social post was found.

### xLSTM / mLSTM
- **url:** https://arxiv.org/abs/2405.04517 (NeurIPS 2024; venue not re-checked); xLSTM 7B https://arxiv.org/abs/2503.13427 (NXAI/JKU)
- **one-line idea:** LSTM gating on a matrix memory: an exponential input gate with a log-space max-stabilizer, a normalizer state, and scalar gates per head.
- **key formula:** \(S_t=f_tS_{t-1}+i_tv_tk_t^\top,\ n_t=f_tn_{t-1}+i_tk_t;\quad \tilde h_t=\frac{S_t\hat q_t}{\max(|n_t^\top\hat q_t|,\,e^{-m_t})},\ \hat q=q/\sqrt{d_{qk}}\)
- **concrete numbers:**
  - xLSTM 7B: 32 blocks, d=4096, 8 heads, d_v=512, d_qk=256, so 131,072 elements per head and 1.05M per layer.
  - Gate pre-activations soft-capped at 15; input-gate bias −10.
  - 2.3T tokens at 8K context.
  - The paper reports ~50% faster generation than Mamba at batch 1.
- **diagram idea:** A normalizer comparison strip: Katharopoulos \(z^\top q\) vs mLSTM \(\max(|n^\top q|,e^{-m})\) vs modern RMSNorm-only, with the exponential-gate stabilizer drawn as a running max.
- **tips:**
  - The exponential input gate lets one token overwrite strongly, and the stabilizer keeps it finite.
  - Its transition class is the same as Mamba-2 (scalar), with no delta correction.
  - It is the main surviving design with an explicit normalizer.
- **verification:** xLSTM 7B mLSTM equations and §architecture/§training.

### Comba and the product-vs-sum ordering (tip card for ch05)
- **url:** https://arxiv.org/abs/2506.02475 (arXiv 2025; HKUST-GZ, Shanghai AI Lab et al.)
- **one-line idea:** Comba uses a scalar-plus-low-rank (SPLR) transition, in which decay and correction act in parallel on the undecayed state, and adds an output correction.
- **key formula:** \(S_t=S_{t-1}(\alpha_tI-\tilde\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top,\quad o_t=S_t(q_t-d\,k_t)\)
- **concrete numbers:**
  - The book's ch05 toy (α=0.8, β=0.5, S=[2,4], k=e₁, v=5): the product form (GDN) gives [3.3, 3.2]; the SPLR form with β̃=β gives [3.1, 3.2].
  - d is initialized to 0.02 (340M) or 1 (1.3B); keys are L2-normalized.
- **diagram idea:** Two-lane flow. Product: decay → read decayed → correct. Sum: read undecayed ‖ decay → add. Annotate both with the toy numbers.
- **tips:**
  - RWKV-7, Comba and linear Titans are all Σ-class; GDN, KDA and the 2026 GDN variants are P-class.
  - The output correction \(q-dk\) subtracts the current key's direction at read time.
  - Evidence so far is limited to ≤1.3B arXiv experiments.
- **verification:** Eq. 5; Table 1 (IPLR vs SPLR); toy arithmetic is mine.

### Longhorn
- **url:** https://arxiv.org/abs/2407.14207 (arXiv 2024; UT Austin)
- **one-line idea:** The SSM recurrence is the closed-form *implicit* (proximal) online-regression step, so the forget gate falls out of the key with no separate parameter.
- **key formula:** \(S_t=S_{t-1}+\mathrm{diag}(\varepsilon_t)(x_t-S_{t-1}k_t)k_t^\top,\ \varepsilon_{t,i}=\frac{\beta_{t,i}}{1+\beta_{t,i}\|k_t\|^2}\). The deployed version uses the diagonal approximation \((\mathbf 1-\varepsilon_t(k_t^{\odot2})^\top)\odot S_{t-1}\).
- **concrete numbers:**
  - 1.3B on SlimPajama reaches Mamba's average perplexity with about half the data (1.8× sample efficiency).
  - Trained at 2K context, it extrapolates to 32K (16×).
  - Saves O(d·m) forget-gate parameters per module.
- **diagram idea:** Effective step size vs β. The explicit step grows linearly: it overshoots once \(\beta>1/\|k\|^2\) and diverges past \(2/\|k\|^2\). The implicit ε saturates at \(1/\|k\|^2\).
- **tips:**
  - The implicit step is a self-normalized learning rate that never overshoots.
  - The deployed model is diagonal, i.e. Mamba-like, not a true delta rule.
  - Its step size is per value channel (left diagonal); compare GDN-2's per-value write gate.
- **verification:** Eq. 5–7; Theorem 3.1; results summary.

### Log-Linear Attention
- **url:** https://arxiv.org/abs/2506.04761 (ICLR 2026; Guo, Yang, Goel, Xing, Dao, Kim)
- **one-line idea:** Split the prefix into Fenwick buckets of exponentially growing size, keep one state per bucket (O(log T) states), and mix them with data-dependent level weights.
- **key formula:** \(o_t=\sum_{\ell=0}^{L-1}\lambda_t^{(\ell)}S_t^{(\ell)}q_t,\ L=O(\log T);\quad O=(QK^\top\odot M^{H})V\)
- **concrete numbers:**
  - Decode is O(log T) time and memory; training is O(T log T).
  - At T=1M, about 20–21 states per head (my arithmetic).
  - Setup: 21 layers, d=1536, 50B tokens, 16K context.
  - Mamba-2 802M → 825M; GDN 793M → 796M.
- **diagram idea:** 16 positions with the Fenwick buckets for t=13 (sizes 8 + 4 + 1) as colored bars, next to the 16×16 mask \(M^H\) with block levels colored by λ.
- **tips:**
  - Equal λ collapses to linear attention.
  - It wraps any linear-transition cell (Mamba-2, GDN).
  - Old history is still pooled coarsely; it sits between fixed-state and full-KV memory.
- **verification:** §3.1 Eq. 3; Eq. 4–5; Table 1; §3.4 variants; §4.2 setup. Venue from mlanthology ICLR 2026.

### PaTH attention
- **url:** https://arxiv.org/abs/2505.16381 (NeurIPS 2025; Yang, Shen, Wen, Tan, Mishra, Ren, Panda, Kim)
- **one-line idea:** Softmax attention whose position encoding is the accumulated product of data-dependent Householder-like transforms between key and query. It reuses DeltaNet's transition as a positional encoding.
- **key formula:** \(A_{ij}\propto\exp\big(k_j^\top\textstyle\prod_{s=j+1}^{i}H_s\,q_i\big),\ H_s=I-\beta_sw_sw_s^\top,\ \beta_s=2\sigma(\cdot)\in(0,2)\)
- **concrete numbers:**
  - 760M models on 50B FineWeb-Edu tokens. Wikitext perplexity: RoPE 19.01, PaTH 18.03, PaTH-FoX 17.35. Zero-shot average 52.7 / 54.2 / 54.2.
  - w comes from a low-rank projection, then conv(3), then L2 norm.
  - Decode updates cached keys in place (\(k_i\leftarrow H_tk_i\)), so FlashDecoding and PagedAttention work unchanged.
  - SmolLM2-1.7B was converted from RoPE to PaTH with 21B tokens of continued pretraining.
- **diagram idea:** RoPE vs PaTH as fixed rotations by offset vs a product of token-dependent reflections between j and i, with a matrix of transforms shaded by data.
- **tips:**
  - A bridge card between ch01 (RoPE) and ch05 (Householder transitions).
  - The in-place key update touches the whole cache every step (extra O(N·d)).
  - KV still grows; this is about expressivity, not compression.
- **verification:** §3 core formula and Eq. 1; UT/blockwise section; Table 2; Table 4; BF16 limitation.

### Forgetting Transformer (FoX)
- **url:** https://arxiv.org/abs/2503.02130 (ICLR 2025; Lin, Nikishin, He, Courville)
- **one-line idea:** A scalar, data-dependent forget gate inside softmax, implemented as a cumulative log-decay bias on the logits.
- **key formula:** \(o_i=\frac{\sum_{j\le i}e^{q_i^\top k_j+D_{ij}}v_j}{\sum_{j\le i}e^{q_i^\top k_j+D_{ij}}},\ D_{ij}=\sum_{l=j+1}^{i}\log f_l,\ f_l=\sigma(w_f^\top x_l+b_f)\)
- **concrete numbers:** No positional embedding is needed and it is FlashAttention-compatible. PaTH-FoX (760M) reaches the best Wikitext perplexity, 17.35.
- **diagram idea:** A causal attention heatmap multiplied by a decay mask \(\prod f\). Label the mask "same mask as Mamba-2/SSD, but inside softmax."
- **tips:**
  - It is the softmax counterpart of ch04's decay, but the KV still grows.
  - The "Pro" block adds recurrent-model components to both FoX and the Transformer.
- **verification:** Abstract (ICLR 2025); formula via the PaTH-FoX restatement in PaTH §3 (FoX equations not read directly).

### ATLAS (Omega rule)
- **url:** https://arxiv.org/abs/2505.23735 (Google; arXiv 2025)
- **one-line idea:** Fit the memory to the last c tokens (a sliding-window loss) rather than the current token, using Muon updates and polynomial or exponential feature maps.
- **key formula:** \(\min_{\mathcal M}\sum_{i=t-c+1}^{t}\gamma_i^{(t)}\|\mathcal M(\phi(k_i))-v_i\|^2\) (c=1 gives the delta rule), with update \(\mathcal M_t=\alpha_t\mathcal M_{t-1}-\eta_t\,\mathrm{NS}_k(\text{momentum})\)
- **concrete numbers:**
  - Capacity: a matrix memory with a degree-p feature map stores O(d_k^p) independent pairs, vs O(d_k) without (Prop. 1–2).
  - Models: 340M, 400M, 790M, 1.3B (15–100B tokens) at 4K context.
  - Reported "+80%" on BABILong at 10M; on par with Titans up to 1M.
- **diagram idea:** A sequence with a highlighted window of c tokens feeding one loss. Inset: capacity curves d_k vs d_k² vs d_k³.
- **tips:**
  - "Memorize the context, not the token."
  - The number of Newton–Schulz steps k is a test-time compute knob.
  - The capacity gain comes from the feature-map width, so the state grows to \(d_v\times\dim\phi\).
- **verification:** Eq. 9; Eq. 22; Eq. 32–33; Prop. 1–2; §6.3. The update form is summarized; the exact Eq. 32 notation was not copied.

### LaCT (large-chunk TTT)
- **url:** https://arxiv.org/abs/2505.23884 (ICLR 2026; MIT and Adobe)
- **one-line idea:** Update nonlinear fast weights once per very large chunk (2K–1M tokens), and let window attention handle structure inside the chunk.
- **key formula:** \(W\leftarrow\mathrm{L2Norm}\big(W-\eta\,\mathrm{Muon}(\nabla_W\textstyle\sum_{i\in\text{chunk}}-\langle f_W(k_i),v_i\rangle)\big)\), with \(f_W\) a bias-free SwiGLU. "Apply-then-update" gives a shifted block-causal mask.
- **concrete numbers:**
  - State up to 12d² per block, about 40% of model parameters.
  - Earlier TTT ran below 5% FLOP utilization.
  - 14B autoregressive video diffusion (from Wan 2.1) with 56K visual tokens; novel-view synthesis with about 1M tokens.
- **diagram idea:** A sequence cut into huge chunks: window attention inside each chunk, a weight-update arrow between chunks, and a GPU-utilization bar for small vs large chunks.
- **tips:**
  - Large chunks make TTT matmul-bound, so it runs as plain PyTorch.
  - The current chunk is invisible to the memory, so local attention is mandatory.
  - It suits set-like data (views, frames) best; for language it is a hyperparameter study.
- **verification:** Abstract; §3 (loss, Muon, normalization, ordering, state size); tasks section.

### TTT-E2E
- **url:** https://arxiv.org/abs/2512.23675 (Astera, Stanford, UC Berkeley, UCSD, NVIDIA; arXiv 2025-12)
- **one-line idea:** A sliding-window Transformer that keeps training its last-quarter MLP weights on the context by next-token prediction, with a meta-learned initialization.
- **key formula:** \(\theta^{\text{MLP}}_{\text{last }1/4}\leftarrow\theta-\eta\nabla\mathcal L_{\text{NTP}}(\text{1K-token mini-batch})\); the outer loop differentiates through these updates.
- **concrete numbers:**
  - 3B models trained on 164B tokens; window 8K.
  - Constant latency, 2.7× faster than full attention at 128K (H100).
  - Training is 3.4× slower at 8K and 1.2× faster at 128K.
  - Loss scales with context like full attention, while Mamba-2 and GDN do not.
  - NIAH is clearly worse than full attention.
- **diagram idea:** A block stack with the last quarter's MLPs highlighted, the context stream feeding gradient arrows, and a latency-vs-context chart (full attention rising, TTT-E2E flat).
- **tips:**
  - The "memory" is ordinary MLP weights; a second static MLP preserves pretraining knowledge.
  - The loss is next-token prediction, not key→value regression.
  - Compressing into weights still fails exact recall.
- **verification:** Abstract; method and ablation sections; limitations.

### Jet-Nemotron / JetBlock (PostNAS)
- **url:** https://arxiv.org/abs/2508.15884 (NVIDIA; arXiv 2025)
- **one-line idea:** Start from a pretrained Transformer with frozen MLPs, search where to keep full attention, and replace the other layers with JetBlock (GDN plus input-generated dynamic convolutions on V).
- **key formula:** No equations in the paper (prose only). JetBlock = GDN time mixing + kernel generator applied to V; static Q/K convolutions removed.
- **concrete numbers:**
  - 2B (from Qwen2.5-1.5B): 28 blocks, full attention at layers 15 and 20, SWA at 21 and 22.
  - 2B: KV cache 154 MB at 64K; 2,885 tok/s on H100; about 47× Qwen3-1.7B-Base at 64K; 53.6× decode and 6.14× prefill at 256K.
  - 4B: 258 MB; 1,271 tok/s. Layer indices for the 4B model are inconsistent between the text and the tables.
- **diagram idea:** A 28-cell layer strip (2 full, 2 SWA, 24 JetBlock) next to a scatter of KV cache vs throughput.
- **tips:**
  - The paper finds KV size, not parameter count, predicts decode throughput.
  - The placement of the few full-attention layers is task-dependent (retrieval vs math).
  - GDN won the block search; RWKV-7 was dropped for training throughput.
- **verification:** PostNAS steps; Tables 9 and 11; §3.1 inconsistency noted.

### Erase-then-Delta Attention (EDA)
- **url:** https://arxiv.org/abs/2606.26560 (Qwen team + Nanjing University; arXiv 2026-06)
- **one-line idea:** A delta rule can only clear content at the address it writes to. EDA adds a separate low-rank erase address \(e_t\) and applies one Householder erase before the KDA-style write.
- **key formula:** \(S_t=S_{t-1}\mathrm{Diag}(\alpha_t)(I-\gamma_te_te_t^\top)(I-\beta_tk_tk_t^\top)+\beta_tv_tk_t^\top\)
- **concrete numbers:**
  - Dense 2.5B: 24 layers (18 linear / 6 full).
  - MoE 25B-A2.8B: 28 layers (21 / 7).
  - Linear K/V heads 8/16 (dense) or 16/32 (MoE), dim 128.
  - 400B tokens at 4K, plus 80B at 32K midtraining.
  - The erase address comes from a rank-16 projection.
- **diagram idea:** A state heatmap with a rank-1 erase along column direction e (stale fact), then a rank-1 write along k (new fact), the two shown as different stripes.
- **tips:**
  - Equivalent to DeltaProduct₂ with a zero value on the first sub-step.
  - It decouples the *address*, whereas GDN-2 and FG²+ decouple the *strength*.
  - It comes from the same team as Qwen3.8-Next, so it may be a candidate for future Qwen cells (speculative).
- **verification:** Eq. 8–13; Appendix A; config table.

### Recall–state tradeoff (Based)
- **url:** https://arxiv.org/abs/2402.18668 (ICML 2024 from memory, not confirmed on the abstract page; Arora, Eyuboglu, …, Ré). Copying limit: Jelassi et al. "Repeat After Me" (ICML 2024; arXiv id not re-checked).
- **one-line idea:** Recall accuracy tracks recurrent state size; varying the linear-attention feature dimension and the sliding-window size moves a model along the Pareto curve.
- **key formula:** none (empirical). For the figure, plot recall vs state bytes.
- **concrete numbers:**
  - Based: +6.22 points on recall-intensive tasks vs Mamba; 24× FlashAttention-2 throughput generating 1,024 tokens at 1.3B.
  - DeltaNet 1.3B on FDA: 17.2 pure vs 29.8 with 2 global-attention layers ([Parallel DeltaNet Tbl](https://arxiv.org/html/2406.06484)).
- **diagram idea:** A scatter of recall accuracy vs state size (log) for model families, with a "hybrid" arrow jumping up. Redraw from the paper's figure data.
- **tips:**
  - This is the quantitative core of ch04's "what aggregation gives up."
  - A few attention layers fix recall more cheaply than enlarging the state.
- **verification:** Abstract; Parallel DeltaNet Table 1. The Taylor feature-map detail is not on the abstract page (**UNVERIFIED** here).

---

## 5. Suggested restructure

A new chapter is warranted: **"05b · Memory as online learning"** (between 05 and 06). Card order:
1. The regression lens: LA = dot loss, Delta = L2 loss, decay = retention (GDN Table 1, Miras Table 1).
2. Explicit vs implicit vs exact steps: DeltaNet → Longhorn (implicit) → Mesa (exact, CG).
3. Mini-batch semantics: the TTT b-slider (b=1 delta, b=16 TTT, b=T LA).
4. Deep memory: TTT-MLP → Titans (momentum, weight decay; Σ-ordering).
5. Wider losses and optimizers: ATLAS (window c, Muon, polynomial features).
6. Uncertainty-weighted steps: KDN (move from ch10, or cross-reference it).
7. Scaling the chunk: LaCT (2K–1M chunks), then TTT-E2E (whole-model MLPs).
8. Limits card: none of these remove the capacity limit or fix NIAH; it needs hybrids (→ 08).

Changes to existing chapters:
- **Ch04:** add cards for the Mamba-2 mapping, Mamba-3 and Lightning, plus the normalizer note.
- **Ch05:** end with the transition-taxonomy card and the P vs Σ ordering card. Add the negative-eigenvalue and DeltaProduct cards, and the KDA-DPLR chunk card.
- **Ch08:** add the production state-budget table and the MiniMax M2 counter-evidence.
- **Ch10:** add EDA (address decoupling), Log-linear, and PaTH/FoX ("softmax borrows recurrent transitions").
