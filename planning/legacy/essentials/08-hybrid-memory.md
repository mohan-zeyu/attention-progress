# 08 · Use both kinds of memory

A compact state can maintain context. Historical records can supply detail that the state lost.

## Why combine layers?

Qwen3-Next and Kimi Linear both repeat three recurrent layers followed by one explicit-attention layer. Their component mechanisms differ.

![One repeating group in Qwen3-Next has three Gated DeltaNet layers plus gated global GQA; Kimi Linear has three KDA layers plus full MLA. Residual and feed-forward operations are omitted.](../figures/visual/08-stacks.svg)

> This is a model design: state layers plus record-based layers.

## Where does a gate act?

An output gate scales a computed readout. A state gate changes memory carried to later tokens.

![An output gate modulates this readout after attention. A state gate alters carried memory and therefore future reads. Both can coexist in a recurrent layer.](../figures/visual/08-gates.svg)

> Output suppression does not imply the attention read was skipped.

## Is the whole hybrid fixed-memory?

The state component stays bounded; explicit layers retain growing historical caches.

![A hybrid retains a fixed-size recurrent component and a growing explicit-cache component. Fewer explicit layers reduce the growing term, but do not remove it.](../figures/visual/08-cost.svg)

> Fewer explicit layers reduce the growing term. They do not remove it.

Layer composition, record representation and sparse selection are separate choices. Next: [execute and store them efficiently](09-execution-and-storage.md).
