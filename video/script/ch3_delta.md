# Chapter 3 · Correcting: the delta rule

*Narration for `video/scenes/ch3_delta.py`, from section 3 of `Fixed-Size States.md`. Lines in quote blocks are on-screen formulas and are not spoken.*

## The problem

<!-- intro -->
Chapter three: correcting, with the delta rule. Decay can only fade. If the state holds one wrong association, decay cannot remove just that one. It fades everything, including memories that were still correct. The delta rule fixes this by checking the table before it writes.

## Read, compare, overwrite

<!-- steps -->
DeltaNet, from 2021, with a parallel training method in 2024, works in three steps. First, read what the state currently returns for the new key: v hat equals S t minus 1, times k t. Second, compare it with the new value. The error is v t minus v hat. Third, write only the error, scaled by a strength, beta, between zero and one.

<!-- formula -->
Rearranged, the same rule reads: the old state times I minus beta k k transpose, plus beta v k transpose. If the state already holds the right value, the error is zero, and nothing changes. If it holds a stale value, that value is replaced.

> $S_t=S_{t-1}+\beta_t\,(v_t-S_{t-1}k_t)\,k_t^{\top}=S_{t-1}\big(I-\beta_tk_tk_t^{\top}\big)+\beta_tv_tk_t^{\top}$

<!-- toy -->
Take a toy example. The state maps two order numbers to a shipping day, with one key direction per order. Order 7314 ships on day 2, a Tuesday. Order 2208 ships on day 4. Now a new fact arrives: order 7314 now ships on day 5, a Friday.

<!-- toyla -->
Linear attention just adds 5 on top of 2, and returns 7. That is a mix of the old answer and the new one.

<!-- toydelta -->
DeltaNet reads 2, and finds an error of 3. With beta equal to 1, it writes 3, and now the state returns exactly 5. With beta equal to one half, it moves halfway, to 3.5. Either way, order 2208 stays at 4.

<!-- leak -->
But the edit is clean only along the key that was written. A query halfway between the two order keys also sees the change. It reads 4.24 before the update, and 6.36 after.

## The geometry of I minus beta k k transpose

<!-- geom -->
Now look at the transition itself, I minus beta k k transpose. It leaves every direction except k untouched. Along k, it multiplies by one minus beta. So its eigenvalues are all ones, plus a single one minus beta.

> $\operatorname{eig}\big(I-\beta kk^{\top}\big)=\{1,\dots,1,\;1-\beta\}$

<!-- geom2 -->
Watch a vector as beta grows. With beta between zero and one, the old entry along k is shrunk. With beta equal to one, it is erased completely, and the write puts in the new value. With beta equal to two, it is flipped. That is a reflection, and it will matter for state tracking.

## Gated DeltaNet and KDA

<!-- gated -->
Gated DeltaNet, from 2024, combines both tools: decay first, then correct. The decay, alpha, can clear a whole head at once, and the delta step edits a single association. Mamba-2 had only the first tool, and DeltaNet only the second.

> $S_t=\alpha_t\,S_{t-1}\big(I-\beta_tk_tk_t^{\top}\big)+\beta_tv_tk_t^{\top}$

<!-- gatedtoy -->
The order matters. The state fades first, so the error is read from the faded state. In the toy, with alpha 0.8 and beta 0.5, the state fades from 2 and 4, to 1.6 and 3.2. The read is 1.6, the error is 3.4, and the write adds 1.7. The result is 3.3 and 3.2. The price: a scalar decay also fades the unrelated order 2208, from 4 down to 3.2.

<!-- kda -->
KDA, Kimi Delta Attention, from 2025, puts GLA's per-channel decay inside the gated delta rule. In the toy, with alpha equal to 0.8 for the first channel and 1 for the second, only order 7314's channel fades. The result is 3.3 and 4. Order 2208 is kept.

> $S_t=S_{t-1}\operatorname{Diag}(\alpha_t)\big(I-\beta_tk_tk_t^{\top}\big)+\beta_tv_tk_t^{\top}$

<!-- kdafast -->
KDA's speed trick is that both low rank vectors in its transition are tied to k t. That is reported to make its kernels about two times faster than a general diagonal plus low rank transition.

## The sum form

<!-- sum -->
RWKV-7 and Comba, from 2025, write the same idea as a sum instead of a product. The decay and the erase are both applied to the old state, so the erase reads the unfaded value. In the toy, the erase reads 2 instead of 1.6, and removes 1 instead of 0.8. The result is 3.1 and 3.2, instead of 3.3 and 3.2.

<!-- rwkv -->
RWKV-7 also makes the decay and the write strength vectors over key channels. Its removal key differs from its write key, so its transition is a general diagonal plus low rank matrix. That costs about twice KDA's work per chunk.

## Negative eigenvalues and state tracking

<!-- parity -->
Some tasks need the state to track something, not just remember it. The simplest is parity: is the number of 1 bits so far even, or odd? A linear RNN whose transition has only positive eigenvalues provably cannot solve parity in finite precision. The state can only shrink or stay the same. It can never flip.

<!-- flip -->
The fix is to let beta go up to 2, for example two times a sigmoid. Then the eigenvalue, one minus beta, reaches minus 1. That is a reflection. On the bits 1 0 1 1 0 1, a one number state that flips its sign at every 1 bit ends at plus 1, correctly showing an even count.

<!-- product -->
DeltaProduct, from 2025, goes further and takes several delta steps per token. Two reflections make a rotation. For example, two mirrors 30 degrees apart give a 60 degree rotation, and the state does not grow.

<!-- capnote -->
Note that this adds expressivity, not capacity. The state can track more kinds of structure, but it still holds about d k exact pairs.

## Training the delta rule

<!-- train -->
One more detail: training. For a decay alone, a chunk only needs cumulative products of decays. But the delta rule's transitions do not commute. So the product of a chunk's transitions is rewritten in a compact form, called the WY or UT form, before the matrix products.

## Recap

<!-- recap -->
To recap. The delta rule reads what the state returns for the new key, and writes only the error. Gated DeltaNet puts a decay in front, and KDA makes that decay per channel. Letting beta reach 2 adds reflections, for state tracking.

<!-- leftover -->
Left over: the delta rule keeps the table clean, but the table is still d v by d k. A better write cannot add room. Next, all of these rules on one axis.
