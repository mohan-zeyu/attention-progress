# Chapter 4 · One axis: the transition matrix

*Narration for `video/scenes/ch4_transition.py`, from section 4 of `Fixed-Size States.md`. Lines in quote blocks are on-screen formulas and are not spoken.*

## The shared shape

<!-- intro -->
Chapter four: one axis. We now have three tools: add, fade, and correct. They look like different ideas, but every rule so far has the same shape. S t equals S t minus 1 times A t, plus W t.

> $S_t=S_{t-1}A_t+W_t$

<!-- parts -->
W t is the write. It is almost always a scaled v k transpose. A t is the transition: what happens to the old state before the write. The methods differ almost only in A t. So let's line them up, and watch what each transition does to the space of keys.

<!-- keyside -->
One detail helps. A t multiplies the state from the right, so it acts on the key side. Each column of S belongs to one key direction. That is why a diagonal A t fades key channels, and why I minus beta k k transpose edits along one key.

## The table, row by row

<!-- rowI -->
A t equal to the identity is linear attention. It does nothing to the old state. It only adds. A chunk is trained with plain sums.

<!-- rowScalar -->
A scalar times the identity, gamma or alpha t, covers RetNet, Lightning and Mamba-2. It fades everything, equally, in every direction. A chunk needs a cumulative product of the decays.

<!-- rowDiag -->
A diagonal matrix of decays is GLA. It fades each key channel on its own, so one direction can shrink fast while another barely moves. A chunk uses a cumulative sum, in log space.

<!-- rowDelta -->
I minus beta k k transpose is DeltaNet. It overwrites along one direction, k, and leaves the rest alone. Gated DeltaNet puts a scalar decay in front: fade, then overwrite. KDA puts a diagonal decay in front: fade per channel, then overwrite. All three train chunks with the WY or UT form.

<!-- rowRWKV -->
RWKV-7 and Comba use the sum form: a diagonal, minus a low rank term. Their chunks are trained as a general diagonal plus low rank matrix.

<!-- rowProd -->
And products of reflections, or rotations, as in DeltaProduct and Mamba-3, can track structure such as parity. These are trained per sub-step, or as rotations.

<!-- toyrecall -->
Remember the order toy from chapter three. The same new fact, order 7314 now ships on day 5, gives a different answer under each transition. Adding gives 7. The delta rule gives exactly 5, and keeps 4. Gated DeltaNet gives 3.3, but fades order 2208 to 3.2. KDA gives 3.3, and keeps 4. The sum form gives 3.1 and 3.2. Same write, different transitions.

## Why the training column splits

<!-- commute -->
Here is why the training column splits in two. Diagonal transitions commute with each other. Apply two of them in either order, and you land in the same place. So a chunk needs only cumulative decays. Householder-like transitions do not commute. Swap the order, and the result changes. So they need the WY or UT rewrite.

## The 2026 refinements

<!-- research -->
The 2026 research models refine the delta rule along this same axis. They separate how strongly to erase, how strongly to write, and where to erase. That means per channel erase and write gates, an erase direction separate from the write key, or a step size set by an uncertainty estimate. None of them changes the size of the state.

<!-- dial -->
So think of A t as a dial. At one end, it does nothing. Turn it, and it fades. Turn it further, and it fades per channel. Then it overwrites along a key. At the far end, it reflects and rotates, and can track structure.

## Recap

<!-- recap -->
To recap. Every rule is S t equals S t minus 1 times A t, plus W t. The transition decides what the state can do: nothing, fade, fade per channel, overwrite, or track structure.

<!-- leftover -->
Left over: the transition changes how the table is edited, never how big it is. Next, a second way to read all of these rules: the state as a small model, trained while it reads.
