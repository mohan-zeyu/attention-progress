# Book plan and editorial contract

The supplied 45-page PDF is the organizing reference. Its 64 short sections connect attention mechanisms through context-memory representation, update, retrieval and routing. The new book expands those connections into problem-centered chapters rather than reproducing the section list or stretching the earlier overview.

| Chapter | Main question | Owner |
| --- | --- | --- |
| 00. The memory-system map | Which problem does each mechanism actually solve? | Coordinator |
| 01. Addressable attention | How does a token find information in context? | Foundations researcher |
| 02. Sharing the KV cache | Can query heads share stored history without losing too much capacity? | Foundations researcher |
| 03. Latent KV memory | Can each historical position be represented more compactly? | Foundations researcher |
| 04. Recurrent memory | Can history be summarized without retaining every position? | Recurrent-memory researcher |
| 05. Editing memory | How can a bounded state correct associations and forget selectively? | Recurrent-memory researcher |
| 06. Sparse retrieval | Which historical positions should a query read? | Retrieval researcher |
| 07. Sequence compression | Can the model retain fewer historical records? | Retrieval researcher |
| 08. Hybrid memory | How do recurrent state and addressable history complement each other? | Recurrent-memory researcher |
| 09. Execution and physical storage | How can the same memory system move and store fewer bytes? | Coordinator |
| 10. Frontier and design | How can finite-memory updates become more selective, and how should mechanisms be compared? | Recurrent-memory researcher |

Each substantive chapter develops a concrete problem, the constraint that motivates it, the mechanism that changes that constraint, and the remaining tradeoff. Original diagrams explain operations and distinctions. Equations support the reasoning and use one notation across chapters. The running teaching example uses order 7314 and a delivery day corrected from Tuesday to Friday.

The coordinator and three researchers visually inspected all PDF pages in three contiguous packages. Primary papers and official model reports verify mechanisms and refine the article's simplifications. Future proposals are identified as proposals; maximum speedups are omitted unless their experimental conditions can be stated accurately.

Notation: N = context positions; L = layers; B = batch or concurrent sequences; Hq/Hkv = query/KV head counts; s = bytes per stored element. Column vectors q and k read a recurrent matrix S in R^(dv × dk), giving S q.

The result is a navigable offline reader with separate chapter pages, an index, a glossary, a mechanism comparison matrix, and a concise source register. No independent model experiments are claimed.
