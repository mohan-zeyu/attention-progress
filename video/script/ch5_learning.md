# Chapter 5 · The state as a small model, trained while reading

*Narration for `video/scenes/ch5_learning.py`, from section 5 of `Fixed-Size States.md`. Lines in quote blocks are on-screen formulas and are not spoken.*

## Every write is one learning step

<!-- intro -->
Chapter five: the state as a small model, trained while reading. So far we have treated the state as a table. There is a second way to read all of these rules. Think of S as the weights of a small linear model, one that should map keys to values. Then each write is one gradient step, on a loss for the newest pair.

> $S_t=\alpha_tS_{t-1}-\beta_t\,\nabla_S\,\ell_t\big(\alpha_tS_{t-1}\big)$

<!-- model -->
Picture it. A key goes in, the model S makes a prediction, and the loss compares that prediction with the value. The gradient tells S how to change, and one step is taken. Then the next token arrives.

<!-- losses -->
Linear attention uses the loss minus the inner product of S k and v. Its gradient is minus v k transpose, so one step adds v k transpose: exactly the linear attention write. DeltaNet uses the squared error, one half the norm of S k minus v, squared. Its gradient is the error times k transpose, which gives the delta rule.

<!-- decay -->
RetNet and Mamba-2 add a scalar decay to the first loss, and Gated DeltaNet adds it to the second. So linear attention and DeltaNet differ only in the loss, and the decay acts as weight decay.

<!-- toy -->
A toy, with d equal to 3 and beta equal to 1. The state maps e 1 to e 1, and e 2 to e 2. A new pair arrives: e 1 should now map to e 3. The linear attention step reads back 1, 0, 1. It keeps the old value, with an error of 1. The delta step reads back exactly 0, 0, 1, with an error of zero.

<!-- next -->
This view suggests the next moves: a richer model, a better optimizer, or a better loss.

## TTT

<!-- ttt -->
TTT layers, from 2024, make the state a model, f, either linear or a small MLP, and train it on each token with a gradient step. The key, the value and the query are learned projections of the input: theta K x, theta V x, and theta Q x.

> $W_t=W_{t-1}-\eta_t\nabla_W\big\lVert f(\theta_Kx_t;W)-\theta_Vx_t\big\rVert^2,\qquad z_t=f(\theta_Qx_t;W_t)$

<!-- batch -->
Updating after every token is slow, so TTT updates in mini batches of b tokens. All the gradients in a batch use the weights from the start of the batch, so b gradients run as one matrix product.

<!-- batchends -->
With b equal to 1, this is online learning, close to the delta rule. With b equal to the whole sequence, a linear f, a step size of one half, and zero start weights, it is exactly linear attention. TTT uses b equal to 16.

<!-- tttnum -->
In the reported 125 million parameter ablation, perplexity falls from 15.91 for linear attention to 11.09. The mini batch step alone is the largest single gain, minus 1.70.

## Titans

<!-- titans -->
Titans, from 2024, adds two optimizer ideas to the memory update: momentum, and forgetting.

> $\mathcal M_t=(1-\alpha_t)\,\mathcal M_{t-1}+m_t,\qquad m_t=\eta_t\,m_{t-1}-\theta_t\,\nabla\big\lVert\mathcal M_{t-1}(k_t)-v_t\big\rVert^2$

<!-- surprise -->
First, surprise. The gradient is large when the memory's prediction is wrong, so surprising tokens get written strongly. Second, momentum. m t keeps writing for a few tokens after a surprise, so the context around a surprising token is stored too.

<!-- forget -->
Third, forgetting. Alpha t acts as weight decay. In the reported ablation, removing it costs the most, raising perplexity from 27.01 to 29.04.

<!-- deep -->
The memory is a deep MLP. The price is that a deep memory loses the neat chunk algebra, so it trains slower than Mamba-2 or Gated DeltaNet.

## ATLAS and Mesa

<!-- atlas -->
The delta rule fits only the newest pair. Two 2025 methods fit more. ATLAS optimizes the memory on a window of the last c pairs, with the Muon optimizer, and with polynomial features that raise the capacity bound from order d k, to order d k to the power p.

<!-- mesa -->
Mesa solves the least squares problem over all pairs, exactly, at read time. The output is G t, times the inverse of H t plus lambda, times q t. Here G sums v k transpose, and H sums k k transpose.

> $o_t=G_t(H_t+\Lambda)^{-1}q_t$

<!-- mesatoy -->
Take two correlated keys. k 1 equals 1, 0, and maps to plus 1. k 2 equals 0.6, 0.8, and maps to minus 1. Reading at k 1 gives 0.40 for linear attention, 0.04 for DeltaNet, and 1.00 for Mesa. The price is compute. With k conjugate gradient steps, Mesa costs about k times GLA.

## LaCT

<!-- lact -->
Small mini batches leave the GPU idle. With b equal to 16 and a width of 64, a TTT update reaches only about 11 FLOPs per byte, far below the roughly 290 that an H100 needs.

<!-- lact2 -->
LaCT, from 2025, updates only once per 2 thousand to 1 million tokens, so each update is a big matrix product. It reports up to 70 percent GPU use, against under 5 percent for earlier TTT, and its state can grow far larger.

<!-- lact3 -->
Because a chunk never sees its own update, every LaCT layer is paired with sliding window attention, for the local part.

## Recap

<!-- table -->
Here is the family at a glance. Each layer picks a loss, an optimizer, and a kind of state.

<!-- recap -->
To recap. Every write is one gradient step on the newest pair. TTT, Titans, ATLAS, Mesa and LaCT change the model, the optimizer, or the loss.

<!-- leftover -->
Left over: learning is not recall. A better learner still writes into a fixed state. Next: the limits.
