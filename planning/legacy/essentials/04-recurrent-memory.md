# 04 · Keep a working state

Can useful history survive without a record for every old position?

## Can many writes become one state?

Kernel linear attention changes the matching rule so key–value contributions can accumulate in a matrix. Normalized forms also keep a normalizer.

![Kernel linear attention accumulates feature-mapped key/value contributions into fixed-size state. A normalized variant also keeps a normalizer.](../figures/visual/04-aggregate.svg)

> The carried state stays fixed in size at fixed model dimensions.

## What is the price?

Several associations share state coordinates. Similar keys can interfere.

![Different associations can overlap in the same state coordinates. Bounded state has finite capacity; similar keys can interfere.](../figures/visual/04-collision.svg)

> Fixed dimensions mean finite capacity, not unlimited memory.

## Can old information fade?

RetNet uses fixed retention time scales. GLA learns input-dependent forgetting.

![RetNet uses fixed retention factors per head. GLA makes state forgetting input dependent; the illustrated variant decays key channels separately.](../figures/visual/04-decay.svg)

> Forgetting frees capacity; an additive write still does not explicitly replace an association.

Mamba and Mamba-2 reach recurrent state through state-space models. Next: [correct an association](05-delta-memory.md).
