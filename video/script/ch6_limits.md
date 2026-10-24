# Chapter 6 · Limits: learning is not recall

*Narration for `video/scenes/ch6_limits.py`, from section 6 of `Fixed-Size States.md`. Lines in quote blocks are on-screen formulas and are not spoken.*

## The gap

<!-- intro -->
Chapter six: limits. Learning is not recall. Every mechanism in this series improves how the table is written. But a fixed state still holds only about d k exact pairs, per d v by d k state. A KV cache keeps all N.

<!-- why -->
Remember chapter one. A space with d k dimensions has room for only d k orthogonal keys. Every later mechanism kept the table cleaner, but none of them made it larger.

<!-- sim -->
Here is a small simulation of that limit. The state is 8 by 8. We store random key value pairs, then read every stored key back. With plain additive writes, the reads blur as soon as keys overlap. With the best possible write, an exact least squares solve, every read is exact up to 8 pairs. Past 8 pairs, both errors climb, and they keep climbing.

<!-- sim2 -->
This is the limit no write rule escapes. A better write keeps the reads clean up to d k pairs, but past d k, every fixed state starts to blur.

<!-- numbers -->
The reported numbers show the same gap. In one comparison of 1 billion parameter models on in context recall tasks, the best fixed state layer, Mesa, scores 41.79. A Transformer scores 52.27.

<!-- numbers2 -->
Going from plain additive writes to an exact solve gains about 4 points. Keeping the full cache gains 10.5 more.

## Log-linear attention

<!-- loglin -->
Log linear attention, from 2025, is a middle ground. Instead of one state, it keeps one state per power of two bucket of the past, like a Fenwick tree. The output adds up the read of each bucket's state, with a weight lambda for each.

> $o_t=\sum_{\ell}\lambda_t^{(\ell)}S_t^{(\ell)}q_t,\qquad S_t^{(\ell)}=\sum_{s\in\mathcal B_t^{(\ell)}}v_sk_s^{\top}$

<!-- buckets -->
Recent tokens get small buckets, and far tokens are merged into big ones. Watch the buckets as tokens arrive. At token 13, the buckets are tokens 0 to 7, 8 to 11, and 12, plus the token itself. That is 4 states.

<!-- merge -->
Each bucket is itself a linear attention state, the sum of v k transpose over its tokens. When a new token arrives, two buckets of the same size merge into one twice as big, like carrying in binary.

<!-- scale -->
After about a million tokens, two to the 20, a KV cache holds 1,048,576 records. Log linear attention holds at most 22 states. Linear attention holds 1.

<!-- growth -->
So the number of states grows with the logarithm of the length. Double the context, and you add just one more state.

<!-- scale2 -->
The reported gains in perplexity are small, and the gap to Transformers remains.

## Hybrids

<!-- hybrid -->
So in practice, models keep a few full attention layers for exact recall, and make the rest fixed state layers. That is the hybrid design, in Part 5 of the essay.

<!-- hybrid2 -->
Tasks that need one exact detail from far back, like a number to repeat or a passage to copy, still need attention over real records. The many cheap state layers carry the rest.

## The whole series

<!-- journey -->
Let's put the whole series together. Chapter one: linear attention turned the cache into one matrix, written with outer products and read with the query.

<!-- j2 -->
Chapter two: decay let the state forget, at a fixed rate, at a rate chosen by the input, or at one rate per key channel.

<!-- j3 -->
Chapter three: the delta rule read the old value and wrote only the error, so a single entry could be corrected.

<!-- j4 -->
Chapter four: all of these became one rule, S t equals S t minus 1 times A t, plus W t, differing only in the transition.

<!-- j5 -->
Chapter five: each write became one gradient step, which opened the door to richer models, better optimizers, and better losses.

<!-- recap -->
To recap. A fixed state can be written better and better, but it holds only about d k exact pairs, while a cache holds all N.

<!-- leftover -->
Left over: exact recall still needs real records. That is why hybrids mix a few full attention layers with many fixed state layers. Thanks for watching.
