# 01 · Find the right context

Different outputs need different parts of the input.

## Why keep several records?

A single summary must serve every output. Attention lets each output consult different positions.

![One summary is a bottleneck. Keeping separate contextual records allows different output steps to consult different positions.](../figures/visual/01-bottleneck.svg)

> Keep context accessible, then choose the read.

## What do Q, K and V do?

Query Q asks. Keys K provide matching features. Values V supply the content being combined.

![A query matches keys. Softmax supplies weights, shown illustratively as 0.1, 0.2, 0.6 and 0.1. The output combines the corresponding values.](../figures/visual/01-qkv.svg)

> Match keys → normalize scores → combine values. The weights shown are illustrative.

## Why several heads?

Heads learn different matching spaces over the same context. Their readouts are combined.

![Multiple heads make different learned reads over the same context. The pictured patterns are examples, not assigned head roles.](../figures/visual/01-heads.svg)

> Multiple learned views; no permanently assigned head roles.

## Why does long context cost more?

Prefill handles the prompt. Cached decode appends one position and reads the retained history.

![Dense causal prefill has many query-key pairs. Cached decoding introduces one query that reads historical records and appends the new key/value.](../figures/visual/01-prefill.svg)

> Causal visibility excludes future positions; positional information helps represent order.

The records remain available, but correct retrieval is learned. Next: [share their KV](02-kv-sharing.md).
