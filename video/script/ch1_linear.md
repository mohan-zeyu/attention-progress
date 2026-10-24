# Chapter 1 · Linear attention: a lookup table stored as a matrix

*Narration for `video/scenes/ch1_linear.py`, from section 1 of `Fixed-Size States.md`. Lines in quote blocks are on-screen formulas and are not spoken.*

## The problem

<!-- intro -->
Chapter one: linear attention. Here is the problem it solves. Softmax attention keeps a record for every past token, a key and a value. To answer a new query, it scores the query against every key. The weights have to be normalized over all past tokens, so nothing can be thrown away. The cache grows with every token you read.

## From the cache to the state

<!-- drop -->
Linear attention makes one change. It drops the softmax. Then the weight on value j is just the score, k j transpose q t.

<!-- regroup -->
Now the products can be regrouped. Pull the query out of the sum. What is left in the bracket, the sum of v j times k j transpose, does not depend on the query at all. Call it the state, S t. The output is just S t times q t.

> $o_t=\sum_{j\le t}v_j\,(k_j^{\top}q_t)=\Big(\sum_{j\le t}v_jk_j^{\top}\Big)q_t=S_t\,q_t$

<!-- write -->
So the state can be kept up to date, one token at a time. Step one is the write. When token t arrives, take the outer product of its value and its key, v t times k t transpose. That is a whole matrix, and it is added onto the state.

> $S_t=S_{t-1}+v_tk_t^{\top}$

<!-- read -->
Step two is the read. Multiply the state by the query, and out comes the output. o t equals S t times q t.

<!-- discard -->
Step three: discard. The key and the value are not needed any more. They live inside the state now. The next token does the same thing. Write, read, discard.

<!-- size -->
How big is this state? With a head width of 128, it is 128 by 128, which is 16,384 numbers, whatever the length. A KV cache holds 2 times N times 128 numbers per head. So the cache passes the state after only 64 tokens.

<!-- size2 -->
At 128 thousand tokens, the cache is 2,048 times larger. And every decode step reads the same fixed matrix.

## What the state remembers

<!-- lookup -->
So what does the state remember? It is a lookup table from keys to values, stored as one matrix. To test the lookup, read it with one of the stored keys, k i. The result is v i, the value we want, plus a sum over every other value, weighted by how much its key overlaps with k i. That second part is crosstalk.

> $S\,k_i=v_i+\sum_{j\ne i}v_j\,(k_j^{\top}k_i)$

<!-- orth -->
If all the keys are orthogonal, every overlap is zero, and the read returns exactly v i. Picture two keys at right angles. Read with the first one, and the second contributes nothing.

<!-- overlap -->
But a space with d k dimensions has room for only d k orthogonal directions. Add a third key in two dimensions, and it has to overlap the others. Now a read with the first key also picks up part of the third value.

<!-- gram -->
Here is the same thing as a grid of overlaps. With four orthogonal keys in four dimensions, only the diagonal is lit, so every read is clean. With eight keys in four dimensions, the off-diagonal cells light up, and every read mixes in a little of every overlapping value.

<!-- capacity -->
So a d v by d k state can hold only about d k key value pairs exactly. A KV cache holds all N of them. This one fact drives everything that follows. The later mechanisms try to keep the table clean, but none of them makes it larger.

## What the softmax was doing

<!-- focus -->
Dropping the softmax looks like a small change, but the softmax did two jobs. The first is sharp focus. Take two keys with scores 5 and 3. Softmax gives the first key e to the 5 over e to the 3, about 7.4 times the weight of the second. A plain linear score gives only 5 over 3, about 1.7 times.

<!-- drown -->
So with linear scores, many slightly matching keys add up, and they can drown out the one right key.

<!-- sum1 -->
The second job: weights that sum to one. Without the normalization, the size of the output drifts as more tokens are added to the state, and that makes training unstable.

<!-- history -->
The first version, the Linear Transformer from 2020, replaced the exponential of q dot k with a dot product of features, phi of q and phi of k. It used a positive feature map, elu plus one, and kept a normalizer to divide by. Later versions dropped the normalizer, because it made gradients unstable. Instead, they normalize q and k to length one, and add a normalization layer on the output.

<!-- different -->
Note that this is a different model, not a faster way to compute softmax attention. A model trained with softmax cannot switch. It has to be trained with linear attention from the start.

## Three ways to compute the same thing

<!-- parallel -->
The same layer can be computed in three ways, and all three give exactly the same outputs. The parallel form builds the full score matrix, Q K transpose, applies a causal mask, and multiplies by V. It is fully parallel, but its cost grows as N squared times d, which is what we wanted to avoid.

> $O=(QK^{\top}\odot M)V$

<!-- recurrent -->
The recurrent form is the token by token loop: write, then read. Its cost grows only as N times d squared, but every step waits for the one before. That wastes a GPU during training and prefill.

<!-- chunk -->
The chunkwise form mixes the two, and it is the one used in practice for training and prefill. Cut the sequence into chunks of C tokens, usually 64 to 256. Inside a chunk, compute the parallel form on the small C by C block. For all earlier chunks, read the state passed in from the previous chunk, and add it. Then update the state, once per chunk.

<!-- same -->
All three steps are matrix products. Here are eight outputs from a small random example, computed all three ways. They match exactly.

<!-- cost -->
For one head with d equal to 128, chunks of 64, and 32 thousand tokens, the chunkwise form costs about 98 thousand FLOPs per token, in 512 sequential steps. The recurrent loop costs 65 thousand FLOPs per token, but in 32,768 sequential steps. The quadratic form costs 8.4 million FLOPs per token. Decode then uses the recurrent loop, one token at a time.

## Recap

<!-- recap -->
To recap. Linear attention folds the whole history into one fixed matrix per head. Write with an outer product, read with the query.

<!-- leftover -->
Left over: the state only adds. Nothing is ever forgotten or overwritten. So after many tokens, the table is full of overlapping entries, and every read is mostly crosstalk. Next: forgetting.
