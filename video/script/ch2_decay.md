# Chapter 2 · Forgetting: decay the state

*Narration for `video/scenes/ch2_decay.py`, from section 2 of `Fixed-Size States.md`. Lines in quote blocks are on-screen formulas and are not spoken.*

## The problem

<!-- intro -->
Chapter two: forgetting. At the end of chapter one, the state could only add. After many tokens it is crowded with overlapping entries, and every read is mostly crosstalk. The simplest fix is to let the state fade a little before each write.

## One rule, one mask

<!-- rule -->
Multiply the old state by a decay, alpha t, somewhere between zero and one. Then add the new outer product, as before.

> $S_t=\alpha_t\,S_{t-1}+v_tk_t^{\top},\qquad 0<\alpha_t\le1$

<!-- unroll -->
Now unroll it. A token j that was written earlier has been multiplied by every decay since then. So its weight in the output carries the product of the alphas, from j plus 1 up to t. The output is attention without the softmax, with a weight mask, D.

> $o_t=\sum_{j\le t}\Big(\prod_{s=j+1}^{t}\alpha_s\Big)(k_j^{\top}q_t)\,v_j \iff O=(QK^{\top}\odot D)V$

<!-- mask -->
Picture D as a lower triangular grid. The diagonal is one, and each step away from it multiplies in one more decay. Plain linear attention is the case where every alpha is one, so its mask is all ones below the diagonal. With a decay, the mask fades away from the diagonal. It also tells the model how far away each token is, so these layers usually need no other positional encoding.

## A fixed decay per head

<!-- retnet -->
RetNet, from 2023, uses one fixed decay per head, called gamma. Then D t j is gamma to the power t minus j, and old tokens fade at a set rate. A head remembers roughly the last one over one minus gamma tokens. With gamma equal to 0.99, the horizon is about 100 tokens, and a token 100 steps back keeps 0.99 to the 100, about 0.37 of its weight.

> $D_{tj}=\gamma^{\,t-j},\qquad 0.99^{100}\approx0.37$

<!-- heads -->
Different heads get different decays, so together they cover several scales. RetNet's experiments space one minus gamma from one over 32 to one over 512. So the horizons run from about 32 to 512 tokens. Lightning attention, from 2025, keeps this fixed decay and adds a fast kernel.

<!-- blind -->
The price is that the decay ignores content. An important number and a greeting fade at exactly the same rate.

## A decay chosen by the input

<!-- selective -->
The next step is to let each token decide how much to forget. Mamba-2, from 2024, uses one decay per head that depends on the token. a t equals e to the delta t times A h. A h is negative, and delta t is computed from the input. So the mask is no longer smooth. A token can choose to keep almost everything, or to forget a lot.

> $a_t=e^{\Delta_tA_h},\qquad A_h<0$

<!-- ssd -->
Its paper shows that this state space model is the same computation as the masked linear attention above, under a simple dictionary. The SSM input, x, plays the value. B plays the key. And C plays the query. This is called the state space duality.

<!-- ssdcost -->
Because the decay is one scalar per head, the whole layer runs as matrix products, reported two to eight times faster than the scan of the first Mamba. One catch: delta t sets both the decay and the write strength. So a strong write also forgets more.

## Per-channel decay: GLA

<!-- gla -->
GLA, gated linear attention, from 2023, goes finer. It decays each key channel separately. The old state is multiplied on the right by a diagonal matrix, with one decay per key channel.

> $S_t=S_{t-1}\operatorname{Diag}(\alpha_t)+v_tk_t^{\top}$

<!-- glatopic -->
Each column of the state belongs to one key channel. So one channel can be cleared at a topic switch, while the others are kept.

<!-- glapower -->
The decays come from a sigmoid, raised to the power one sixteenth. That power keeps them close to one, so by default the state forgets slowly. Sigmoid outputs of 0.01, 0.1, 0.5 and 0.9 become decays of 0.75, 0.866, 0.958 and 0.993.

> $\alpha_t=\sigma(W_\alpha x_t+b)^{1/16}\in(0,1)^{d_k}$

<!-- gllog -->
A diagonal decay still trains well in chunks, because the products of decays become sums in log space.

## Mamba-3

<!-- m3 -->
Mamba-3, from 2026, changes the write in three ways. First, a trapezoid write. Each write averages the inputs at both ends of the step. With a toy step of 0.5 and a decay of 0.8, Mamba-2 writes 0.5 of x t. The trapezoid rule writes 0.2 of x t minus 1, and 0.25 of x t.

<!-- m3rot -->
Second, rotations. A complex valued state turns the real decay into a decay plus a rotation. That works like a position encoding that depends on the data.

<!-- m3mimo -->
Third, MIMO. Writes of rank 4 instead of rank 1 do about four times the work on the same state. Decode is limited by bytes, not compute, so the reported latency stays about flat.

## Recap

<!-- recap -->
To recap. Decay multiplies the old state by alpha before each write. It is fixed per head in RetNet, chosen by the input in Mamba-2, and set per key channel in GLA.

<!-- leftover -->
Left over: decay fades everything a little. It cannot remove one wrong association and keep the rest. And a scalar decay also fades memories that were still correct. Next: correcting the state, with the delta rule.
