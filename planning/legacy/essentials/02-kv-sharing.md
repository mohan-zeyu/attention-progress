# 02 · Share the KV cache

Many query heads do not always need separate stored keys and values.

## Can heads share stored history?

MHA keeps separate KV heads. GQA shares within groups. MQA shares one KV head across all queries.

![Eight query heads use eight KV heads in MHA, two in illustrated GQA, or one in MQA. Each query still forms its own read.](../figures/visual/02-sharing.svg)

> Queries still differ, so their attention weights can differ.

## How much storage changes?

Hold context, head dimensions and precision fixed. KV elements scale with KV head count.

![For the same eight query heads, holding context length, dimensions and precision fixed, KV element counts are proportional to eight, two or one KV heads.](../figures/visual/02-cache-size.svg)

> Fewer KV views save cache space, with a representation-sharing tradeoff.

## What still grows?

Every retained position still needs its record.

![Sharing reduces width, but the number of retained position records still grows. Twice the retained history doubles this cache component.](../figures/visual/02-growth.svg)

> A smaller cache is not automatically an equal reduction in total latency.

Sharing must be learned or adapted into the model. Next: [make each record compact](03-latent-kv.md).
