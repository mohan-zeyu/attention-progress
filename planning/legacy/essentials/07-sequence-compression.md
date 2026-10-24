# 07 · Keep fewer records

Small records can still become a very long list.

## Can several positions share one record?

Sequence compression pools positions. MLA instead keeps a compact record per position.

![MLA retains a compact record at each position. A simple pooling example maps twelve positions to three group records. Pooling diagrams are schematic, not complete V4 cache layouts.](../figures/visual/07-pool.svg)

> A shorter record sequence still grows; it is not fixed recurrent state.

## How should compressed history be read?

CSA selects moderately compressed entries. HCA reads all eligible heavily compressed entries. Both also read recent uncompressed positions in the same core read.

![CSA selects moderately compressed history. HCA reads every eligible heavily compressed historical entry. Both also read recent uncompressed positions. Counts are illustrative.](../figures/visual/07-csa-hca.svg)

> V4 pairs pooling with shared-KV MQA, where entries serve as both key and value.

## Does overlap halve record count again?

CSA windows overlap, but a new entry is emitted at each stride. HCA uses stronger, non-overlapping pooling.

![CSA ratio four emits every four positions. Interior source windows overlap eight positions, but output count remains about N divided by four, not N divided by eight.](../figures/visual/07-overlap.svg)

> Pooling can lose detail. A recent correction depends on compressed history after leaving the local window.

Dense reads of the compressed sequence remain quadratic at fixed ratios, with a smaller coefficient. Next: [combine memory organizations](08-hybrid-memory.md).
