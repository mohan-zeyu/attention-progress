# 09 · Move and store fewer bytes

Cache capacity, data traffic and arithmetic are different costs.

## Must attention write full score matrices?

FlashAttention works in tiles and maintains running softmax statistics and output.

![Naive attention stores full score and probability intermediates in HBM. FlashAttention maintains tiled scores and running softmax/output on chip; dense pairwise arithmetic stays quadratic.](../figures/visual/09-tiles.svg)

> The dense read is preserved, subject to numerical differences. Pairwise arithmetic remains quadratic.

## Can execution suit the hardware?

FLA implements multiple recurrent rules efficiently. Ring Attention circulates KV blocks across devices.

![FLA implements multiple recurrent rules with efficient chunkwise kernels. Ring Attention distributes dense context across devices while circulating KV blocks; scheduling is distinct from the attention rule.](../figures/visual/09-execution.svg)

> An implementation library and a distributed schedule solve different execution problems.

## Can the same information occupy less space?

Cross-layer reuse shares cache sets. Quantization lowers stored precision, with numerical and metadata costs.

![Sharing cache sets across layers reduces repeated storage. Quantizing sixteen-bit elements to four bits reduces raw element bits fourfold, before metadata and numerical costs.](../figures/visual/09-storage.svg)

> Storage savings alone do not determine read traffic or latency.

## Can computed prefixes be reused?

Paged storage reduces fragmentation. Compatible prefix computations can persist on GPU, in host RAM or on SSD.

![Computed compatible prefixes can be retained in GPU, host RAM or SSD. Reuse may save recomputation, while loading has transfer costs. Persistence alone is not semantic memory.](../figures/visual/09-tiers.svg)

> Reuse can save recomputation; moving cached prefixes still costs time.

Persistent computed cache is not automatically lifelong semantic memory. Next: [finer state control](10-frontier-and-design.md).
