# Card style guide (contract for chapter writers)

**Reader:** already fluent in softmax attention and the KV cache. They want **small tips, easy formulas and dense visuals**. They skip paragraphs. Every card should be understandable in about 30 seconds from the formula plus the figure.

**Exemplar to imitate:** `chapters/03-kv-heads.md` with `scripts/cards/kv_heads.py`. The rendered page is `reader/chapters/03-kv-heads.html`.

## Chapter file

```
# NN · Short title            (2–4 words after the number)

::: lede
One sentence, ≤ 25 words: the constraint this chapter attacks.
:::

## Mechanism · what it changes {#slug}
...card...
```

* Use 3–7 cards per chapter. Do not add text between cards and do not add an intro beyond the lede.
* An optional last card can be **“Compare”**: one small table of all mechanisms in the chapter plus one summary figure.

## Card anatomy, in this order

1. `::: meta` holds one line. Omit it only on **insight cards** that explain a cost or comparison rather than one paper, and mark their heading `{#slug .insight}`. The line is: `**Name** year · authors/org · venue · [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX) · Used in: Model A, Model B`.
2. **One display formula** `$$ … $$` that defines the mechanism, at most 2 lines. Pandoc's MathML supports `\operatorname`, `\text`, `\tfrac`/`\frac`, `\underbrace`, `\odot`, `\mathbb`, `\Big`, `\top` and `\le`. It does **not** support `\color`, `\boxed` or `\textcolor`. Colour belongs in the figure.
3. **One figure (mandatory):** `![Alt text that states what the figure shows, including its numbers.](../figures/cards/<slug>.svg)`. Add a second figure only when it carries different information.
4. `::: tips` takes 2–4 bullets in the form `- **Label:** ≤ 20 words.` Labels: *Why it works · Gotcha · Not solved · Used in · Numbers · Compare · Train vs. infer · Converts from · Cost*.
5. Optional `::: {.more title="Derivation"}` of ≤ 6 lines (mostly formula steps), for readers who want to check a claim.
6. Optional small table (≤ 6 rows) for cross-model numbers.

**Word budget:** ≤ 90 words of prose per card, excluding meta, formulas and alt text. Never write paragraphs of explanation. Move the information into the figure or a tip.

## Figures carry the information

Every figure must contain **at least three** of these:

* (a) **A colour-coded formula strip** at the top: `f.math(x, y, r'\c{purple}{S_t} = ...')`. It can be the card's formula or its key part.
* (b) **Tensor shapes with real dimension labels** (`f.brace`, `f.matrix`, `f.vec`).
* (c) **A concrete numeric instance**: hand-checkable toy numbers labelled "toy", or real config numbers.
* (d) **A before/after or A-vs-B contrast.**
* (e) **A cost readout**, such as bytes/token, FLOPs, state size, top-k or window, shown as bars or labels.

Avoid flowcharts made only of boxes with words, and decorative icons.

**Fixed colour meaning** (figkit `COL`):

| Colour | Meaning |
| --- | --- |
| gold | query |
| blue | key / KV cache record |
| purple | value / recurrent state S |
| green | output / selected / kept / benefit |
| red | erased / evicted / cost |
| teal | gates and positional signals (RoPE) |
| line (grey) | inactive / skipped / not computed |

**Canvas:** width 900; height 260–460. Text sizes: ≥ 13 for labels, 15–17 for body, 20–22 for the formula strip. Keep ≥ 20 px margins. **No overlapping text.** Check it visually.

## Toolkit (`scripts/figkit.py`, do not edit; write local helpers in your module)

`f = Fig(name, desc, height)`, then:

* Text: `text(x,y,s,size,color,bold,anchor)`, `lines(x,y,[rows])`, `math(x,y,expr,size,color,anchor)`, `label`, `pill`.
  * `math` takes a TeX subset: `_x _{..} ^{..} \c{colour}{..} \text{..} \b{..}` plus greek and symbols (`\alpha \beta \top \odot \times \to \sum \in \R`); also `\frac{a}{b}` (renders a/b), `\mathcal{K}`, `\subseteq \notin \cup \cap \lfloor \rfloor \langle \rangle \iff \lor \tilde{x} \sqrt{x} \%`, and spacing `\, \; \quad \qquad` that survives SVG.
* Shapes: `box(x,y,w,h,label,color,fill,size)` (labels starting with `$` render as math), `rect`, `panel(x,y,w,h,title)`, `circle`, `line`, `arrow`, `path(points, curve=True)`, `brace(x1,y1,x2,y2,label,side)`, `callout`, `divider`, `hdivider`, `legend(x,y,[(label,colour)])`.
* Data shapes:
  * `matrix(x,y,rows,cols,cell,color,values,label,rlabel,clabel)`: values in [0,1], `None` for empty, `(v,'colour')` per cell.
  * `mask(x,y,n,cell,fn)`: query×key grid; `fn(i,j)` returns a colour, `(colour, opacity)` or `None`.
  * `vec`, `bars(x,y,[(label,value,colour)],width,…,value_fmt)`, `strip(x,y,[(label,colour)])` for layer stacks, `records`.
* `f.save()` writes `figures/cards/<name>.svg`.

Use **one module per chapter**: `scripts/cards/<chapter-slug>.py` with `def build():` calling your figure functions. Prefix every figure name with the chapter's short slug, for example `mla-absorb` or `sparse-nsa`.

## Verification loop (required for every figure)

```
python3 scripts/build_cards.py <module-name-substring>
rsvg-convert -w 900 figures/cards/<name>.svg -o <scratchpad>/<name>.png
# Read the PNG and fix overlaps, clipping and illegible text. Repeat.
python3 scripts/check_chapter.py chapters/NN-*.md   # must print OK for every card: math, meta, formula, figure, tips, word budget
# Do NOT run build_reader.py or build_cards.py without a module filter. Other writers run in parallel; the coordinator builds the reader.
```

## Accuracy

* Every real-world number (head counts, windows, ratios, top-k, bytes) must come from a primary source: a paper section, an official tech report, or a `config.json`/model card. Toy numbers must be labelled "toy" in the figure.
* If you could not verify something, **leave it out**. Do not hedge in prose.
* Attribute reported results to their authors, e.g. "reported: …". Nothing was reproduced.

## Sources

Write `bibliography/<writer-id>.json`, a list of records:

```json
{"title": "...", "year": 2025, "url": "https://arxiv.org/abs/....", "venue": "ICLR 2025 · Tsinghua", "chapters": ["06"], "reading_depth": "sections", "evidence_note": "≤ 20 words: what was checked (e.g. §3.2 Eq. 7; Table 1 block size 64)."}
```

`reading_depth` is one of `sections`, `abstract` or `official report`. The older files `foundations.json`, `recurrent.json`, `sparse.json` and `systems.json` hold verified records from the previous edition. Their chapter codes are remapped by the coordinator. Do not edit them.

## Notation (book-wide)

| Symbol | Meaning |
| --- | --- |
| N, t, j | context length, current position, history index |
| L | layers |
| H_q, H_kv, G = H_q/H_kv | query heads, KV heads, group size |
| d, d_h | model width, head dim |
| d_c, d_r | MLA latent dim, decoupled RoPE dim |
| B, s | batch, bytes per element |
| W | sliding window |
| B_s, n_sel | block size, selected blocks/tokens (top-k) |
| q, k, v | column vectors |
| S_t ∈ ℝ^{d_v×d_k} | recurrent state; read o_t = S_t q_t; write v_t k_tᵀ |
| α_t | scalar decay |
| diag(a_t) | per-channel decay |
| β_t | write strength |

Cross-link with relative chapter filenames, e.g. `[MLA](04-latent-kv.md#mla)`.

## Ownership

Edit **only**:

* your assigned `chapters/NN-*.md`
* your `scripts/cards/<slug>.py` modules
* your `bibliography/<writer-id>.json`

Never edit `figkit.py`, `style.css`, `build_reader.py`, other writers' files, or anything in `planning/legacy/`. `planning/legacy/` is the previous edition and is a useful source of already-verified facts. If you need a toolkit change, say so in your final message.
