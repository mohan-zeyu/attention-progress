# 06 · Choose what to read

Keep history available. Spend the expensive read on selected entries.

## Can we prescribe the connections?

Longformer combines local and global links. BigBird also adds random links.

![Illustrative causal visibility graphs compare dense connections, local plus global links, and the added random links used by BigBird. Original document models also use bidirectional forms.](../figures/visual/06-patterns.svg)

> Connectivity sketch with causal visibility shown; the original document models also use bidirectional forms.

## Can the query choose blocks?

MoBA scores mean-key summaries, then reads original entries inside selected blocks. The current block stays causally available.

![MoBA scores mean-key summaries, selects historical blocks, and reads original entries inside them. The current block is included causally; unselected history remains stored.](../figures/visual/06-blocks.svg)

> A routing summary chooses access; it is not the detailed value read.

## Can an existing model use sparse prefill?

MInference approximates its reads with suitable sparse head patterns, without model retraining.

![MInference approximates attention during inference using sparse head patterns, primarily accelerating prefill without retraining the model.](../figures/visual/06-minference.svg)

> An inference approximation changes access; quality needs checking.

## Can summaries and detail cooperate?

NSA trains compressed, selected-detail and local branches together. Gates mix their outputs.

![NSA trains three gated read branches: compressed summaries, selected original blocks and a local window. Detailed selectable history remains stored.](../figures/visual/06-nsa.svg)

> Selectable original history remains stored alongside the summaries.

## Can selection be cheaper than the main read?

DSA uses a lightweight indexer to choose entries. DeepSeek-V3.2 combines it with MLA.

![DSA uses a lightweight indexer to select entries for the main attention. All pictured historical records remain retained; selector work grows with history.](../figures/visual/06-indexer.svg)

> The indexer still scores growing history; its work is quadratic over a complete sequence.

**Sparse reads do not imply cache eviction.** Next: [reduce how many records exist](07-sequence-compression.md).
