#!/usr/bin/env python3
"""Write README.md: what the book is, the chapter table (from the chapter files), and how to rebuild."""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    rows, total = [], 0
    for path in sorted((ROOT/'chapters').glob('[0-9][0-9]-*.md')):
        text = path.read_text()
        title = re.search(r'^# (.+)$', text, re.M).group(1)
        lede = re.search(r'^::: lede\n(.+?)\n:::', text, re.M | re.S).group(1).strip()
        cards = re.findall(r'^## (.+?)\s*\{#', text, re.M)
        total += len(cards)
        number, name = title.split(' · ', 1)
        names = ' · '.join(re.sub(r'\$([^$]+)\$', r'\1', c).split(':')[0] for c in cards)
        rows.append(f'| {number} | [{name}](chapters/{path.name}) | {lede} | {names} |')
    manifest = json.loads((ROOT/'reader'/'manifest.json').read_text()) if (ROOT/'reader'/'manifest.json').exists() else {}
    sources = manifest.get('primary_sources', '')
    figures = len(list((ROOT/'figures'/'cards').glob('*.svg')))
    readme = f"""# Attention, one change at a time

[**Open the book →**](reader/index.html) · [PDF version](attention-visual-guide.pdf)

A visual field guide to how attention in LLMs developed from 2014 to 2026: MHA → GQA → MLA, sliding windows and sparse reads, cache eviction and compression, linear/delta/test-time-learning memories, hybrid stacks, kernels and serving.

It is written for readers who already know softmax attention and the KV cache. Each of the **{total} mechanism cards** has the same parts:

- **meta line:** year, authors or lab, venue, link, and the production models that use it
- **one formula** that defines the mechanism
- **one dense figure** with a colour-coded formula strip, real tensor shapes and real numbers taken from papers or `config.json`
- **2–4 short tips:** why it works, the gotcha, what it does not solve
- **an optional collapsed derivation**

The book has {figures} figures and {sources} primary sources. Figure colours have the same meaning in every chapter: gold = query, blue = key / KV cache, purple = value / state, green = kept / output, red = evicted / cost, teal = gate / position.

![Seven levers organise the book.](figures/cards/map-routes.svg)

| # | Chapter | The question | Cards |
| --- | --- | --- | --- |
{chr(10).join(rows)}

[Mechanism matrix](appendices/mechanism-matrix.md) · [Notation and glossary](appendices/glossary.md) · [Reference article and coverage](bibliography/reference-note.md) · [Primary sources](bibliography/sources.md)

## Rebuild

```sh
python3 scripts/build_cards.py      # draw every figure (scripts/cards/*.py, toolkit scripts/figkit.py)
python3 build_reader.py             # HTML reader, matrix, source register, link check
python3 scripts/build_pdf.py        # PDF via headless Chrome
python3 scripts/check_chapter.py chapters/*.md   # card-format lint
```

Requires `pandoc`, and Google Chrome for the PDF. The card format is specified in [planning/card-style.md](planning/card-style.md); the October 2026 literature survey behind the new chapters is in [planning/research/](planning/research/). The previous long-form edition is kept in [planning/legacy/](planning/legacy/). Literature checked through 21 October 2026. No experiments were reproduced; reported numbers are attributed to their authors.
"""
    (ROOT/'README.md').write_text(readme)
    print(f'README: {len(rows)} chapters, {total} cards')


if __name__ == '__main__':
    main()
