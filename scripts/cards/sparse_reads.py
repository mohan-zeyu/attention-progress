"""06 · Sparse reads: keep every record, read only what the query needs."""
from figkit import Fig, COL

KIB = 1024


def legend(f, x, y, items, size=13, gap=18):
    for label, color, op in items:
        f.rect(x, y-10, 12, 12, color, op, 2, 'line' if color == 'paper' else None)
        f.text(x+17, y, label, size, 'ink', anchor='start')
        x += 17 + len(label)*size*.52 + gap
    return x


def outline(f, x, y, w, h, color, width=2, dash=False):
    d = ' stroke-dasharray="4 3"' if dash else ''
    f.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{COL.get(color, color)}" stroke-width="{width}"{d}/>')


def cache_row(f, x, y, n, state, w=12, h=16, gap=2):
    """state(i) -> 'read' | 'kept' | 'gone' | 'away' | 'back'."""
    for i in range(n):
        s = state(i)
        xx = x + i*(w+gap)
        if s == 'read':
            f.rect(xx, y, w, h, 'green', .9, 2)
        elif s == 'kept':
            f.rect(xx, y, w, h, 'blue', .28, 2)
        elif s == 'gone':
            f.rect(xx, y, w, h, 'red', .14, 2)
            f.line(xx+3, y+3, xx+w-3, y+h-3, 'red', 1.2)
            f.line(xx+w-3, y+3, xx+3, y+h-3, 'red', 1.2)
        elif s == 'away':
            f.parts.append(f'<rect x="{xx}" y="{y}" width="{w}" height="{h}" rx="2" fill="white" stroke="{COL["muted"]}" stroke-dasharray="3 2"/>')
        elif s == 'back':
            f.rect(xx, y, w, h, 'green', .9, 2)
            f.parts.append(f'<rect x="{xx-1}" y="{y-1}" width="{w+2}" height="{h+2}" rx="2" fill="none" stroke="{COL["muted"]}" stroke-dasharray="3 2"/>')
    return n*(w+gap)


# ---------------------------------------------------------------- 2x2 opener
def what_shrinks():
    f = Fig('sparse-what-shrinks', 'Two questions sort every route of chapters 05 to 08: does it shrink the records read or the records stored, and can the next query choose differently? Reads shrink and reversible: sparse reads (06: Quest, MoBA, NSA, DSA). Reads shrink and fixed: static patterns (05). Records shrink and irreversible: eviction (07), compression (08) and sliding-window layers (05). Records leave HBM but can return: offload (14). Llama-3-8B at 128K in BF16: reading 2,048 records keeps 16 GiB in HBM and moves 256 MiB per step; keeping 2,048 records stores and moves 256 MiB.', 486)
    f.math(30, 34, r'\text{HBM} = \c{blue}{N_{\text{stored}}} \cdot b_{\text{rec}}', 21)
    f.math(330, 34, r'\text{bytes / step} = \c{green}{N_{\text{read}}} \cdot b_{\text{rec}} + \text{scorer}', 21)
    cx, cw, ch = [200, 545], 335, 148
    ry = [86, 244]
    f.text(cx[0]+cw/2, 70, 'reads shrink · every record kept', 15, 'green', True)
    f.text(cx[1]+cw/2, 70, 'records shrink · cache smaller', 15, 'blue', True)
    f.text(30, ry[0]+60, 'reversible', 16, 'ink', True, 'start')
    f.lines(30, ry[0]+82, ['next query can', 'choose again'], 13, 'muted')
    f.text(30, ry[1]+60, 'irreversible', 16, 'ink', True, 'start')
    f.lines(30, ry[1]+82, ['decided before', 'the query arrives'], 13, 'muted')
    q1 = {2, 3, 9, 10, 17, 18, 19}
    q2 = {5, 6, 13, 14, 17, 18, 19}
    fixed = {0, 4, 8, 12, 16, 17, 18, 19}
    gone = set(range(4, 14))
    cells = [
        (0, 0, '06 · Sparse reads', 'Quest · MoBA · NSA · DSA · MSA', lambda i: 'read' if i in q1 else 'kept', lambda i: 'read' if i in q2 else 'kept', 'q2 reads other records', 'green'),
        (0, 1, '14 · Offload', 'records leave HBM, fetched on demand', lambda i: 'read' if i > 15 else ('away' if 4 <= i < 14 else 'kept'), lambda i: 'read' if i > 15 else ('back' if i in (8, 9) else ('away' if 4 <= i < 14 else 'kept')), 'q2 pulls 8, 9 back', 'muted'),
        (1, 0, '05 · Fixed patterns', 'Sparse Transformer · Longformer · BigBird', lambda i: 'read' if i in fixed else 'kept', lambda i: 'read' if i in fixed else 'kept', 'pattern ignores content', 'muted'),
        (1, 1, '07 Evict · 08 Compress · 05 SWA', 'StreamingLLM · H2O · SnapKV · CSA/HCA', lambda i: 'gone' if i in gone else 'read', lambda i: 'gone' if i in gone else 'read', 'q2 needs record 9: gone', 'red'),
    ]
    for r, c, tag, names, s1, s2, note, ncol in cells:
        x, y = cx[c], ry[r]
        f.panel(x, y, cw, ch)
        f.pill(x+12, y+24, tag, 'green' if (r, c) == (0, 0) else ('red' if (r, c) == (1, 1) else 'muted'), 13)
        f.text(x+12, y+50, names, 13, 'ink', anchor='start')
        f.text(x+12, y+82, 'q1', 13, 'gold', True, 'start')
        cache_row(f, x+36, y+69, 20, s1)
        f.text(x+12, y+108, 'q2', 13, 'gold', True, 'start')
        cache_row(f, x+36, y+95, 20, s2)
        f.text(x+12, y+136, note, 13, ncol, True, 'start')
    legend(f, 200, 412, [('read', 'green', .9), ('stored, not read', 'blue', .28), ('evicted / merged away', 'red', .14), ('off-GPU', 'paper', 1)])
    f.hdivider(426)
    f.text(30, 452, 'Llama-3-8B · 128K · BF16', 14, 'ink', True, 'start')
    f.text(30, 470, '128 KiB per token', 13, 'muted', anchor='start')
    f.text(250, 452, 'read 2,048 of 131,072 (06)', 14, 'green', True, 'start')
    f.text(250, 470, 'HBM 16 GiB · 256 MiB moved per step', 13, 'ink', anchor='start')
    f.text(580, 452, 'keep 2,048 (07)', 14, 'blue', True, 'start')
    f.text(580, 470, 'HBM 256 MiB · 256 MiB moved per step', 13, 'ink', anchor='start')
    f.save()


# ---------------------------------------------------------------- training-free prefill
def prefill():
    n, cell = 32, 6
    f = Fig('sparse-prefill', 'Three per-head prefill patterns from MInference on toy 32×32 masks: A-shape (static first columns plus local band; real 1,024 + 4,096), Vertical-Slash (columns and diagonals estimated from the last 64 queries), Block-Sparse (64×64 blocks scored by mean-pooled q and k, top 100 kept). XAttention scores each block by the sum of its antidiagonal cells and keeps the fewest blocks whose softmax mass passes a threshold: 6.89% density at 128K with stride 8. Reported: 1M-token prefill on one A100 from 30 to 3 minutes; Qwen2.5-1M 3–7× prefill speedup at 1M.', 452)
    f.math(30, 34, r'M_{ij} = 1 \; ⟺ \; j \in \c{blue}{V} \; ∨ \; (i - j) \in \c{teal}{S}', 21)
    f.math(870, 34, r'\c{blue}{V}, \c{teal}{S} = \text{top columns / diagonals of } \text{softmax}(\c{gold}{Q_{-64:}} K^{\top})', 16, 'muted', 'end')
    xs = [30, 250, 470]
    y0 = 126

    def ashape(i, j):
        if j > i:
            return None
        if j < 2:
            return 'green'
        return 'blue' if i - j < 5 else None
    V, S = {0, 6, 13, 22}, {0, 1, 8, 17}

    def vslash(i, j):
        if j > i:
            return None
        if j in V:
            return 'blue'
        if (i - j) in S:
            return 'teal'
        return None
    chosen = {(0, 0), (1, 0), (1, 1), (2, 0), (2, 2), (3, 1), (3, 3), (4, 0), (4, 2), (4, 4), (5, 3), (5, 5), (6, 1), (6, 4), (6, 6), (7, 0), (7, 5), (7, 7)}

    def block(i, j):
        if j > i:
            return None
        return 'purple' if (i // 4, j // 4) in chosen else None
    for x, fn, t, sub, real in ((xs[0], ashape, 'A-shape', 'static', '1,024 first + 4,096 local'),
                                (xs[1], vslash, 'Vertical-Slash', 'dynamic', 'indices from last 64 queries'),
                                (xs[2], block, 'Block-Sparse', 'dynamic', '64×64 blocks, mean-pooled, top-100')):
        f.text(x+n*cell/2, y0-28, t, 17, bold=True)
        f.text(x+n*cell/2, y0-10, sub, 13, 'muted')
        f.mask(x, y0, n, cell, fn)
        f.text(x+n*cell/2, y0+n*cell+20, real, 13, 'ink')
    outline(f, xs[1]-2, y0+28*cell-2, n*cell+2, 4*cell+2, 'gold', 2)
    f.text(xs[1]+n*cell/2+20, y0+60, 'last queries', 13, 'gold', True, 'start')
    f.text(xs[1]+n*cell/2+20, y0+76, 'estimate V, S', 13, 'gold', anchor='start')
    # XAttention inset
    px = 690
    f.text(px+90, y0-28, 'XAttention scorer', 17, bold=True)
    f.text(px+90, y0-10, 'one 8×8 block, stride 4', 13, 'muted')
    vals = [[.15 + .7*((r*5 + c*3) % 7)/7 for c in range(8)] for r in range(8)]
    for r in range(8):
        for c in range(8):
            anti = (r + c) % 4 == 3
            f.rect(px+30+c*15, y0+r*15, 13, 13, 'purple', vals[r][c]*.6, 2)
            if anti:
                f.parts.append(f'<rect x="{px+30+c*15}" y="{y0+r*15}" width="13" height="13" rx="2" fill="none" stroke="{COL["gold"]}" stroke-width="2"/>')
    f.lines(px, y0+140, ['score = Σ gold cells', 'keep fewest blocks with', 'Σ softmax(score) ≥ τ'], 13, 'ink')
    f.text(px, y0+n*cell+20, '128K: 6.89% density (S = 8)', 13, 'ink', anchor='start')
    legend(f, 30, 64, [('first tokens', 'green', .85), ('local / vertical', 'blue', .85), ('slash', 'teal', .85), ('block', 'purple', .85)])
    f.hdivider(356)
    f.text(30, 380, '1M-token prefill, 8B model, one A100 (reported)', 14, 'ink', True, 'start')
    f.bars(30, 392, [('dense', 30, 'red'), ('MInference', 3, 'green')], width=260, bar=16, gap=8, label_w=100, size=13, value_fmt=lambda v: f'{v} min')
    f.divider(470, 366, 444)
    f.text(490, 380, 'In production', 14, 'ink', True, 'start')
    f.text(490, 402, 'Qwen2.5-1M: Vertical-Slash prefill,', 13, 'ink', anchor='start')
    f.text(490, 420, '3–7× faster at 1M (H20 / A100)', 13, 'green', True, 'start')
    f.text(490, 438, 'decode untouched: full KV kept', 13, 'muted', anchor='start')
    f.save()


# ---------------------------------------------------------------- Quest
QUEST_Q = [2, -1, 1, 1]
QUEST_PAGES = {
    'A': [[2, 0, 1, 1], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
    'B': [[1, 1, 0, 0], [0, 0, 1, 0], [1, 0, 1, 0], [0, 1, 0, 1]],
    'C': [[-1, 1, 0, 0], [0, 1, -1, 0], [0, 2, 0, 1], [-1, 0, 0, 0]],
    'D': [[1, -1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1], [0, -1, 0, 0]],
}


def num(v):
    return str(v).replace('-', '−')


def quest():
    q = QUEST_Q
    stats = {}
    for p, keys in QUEST_PAGES.items():
        mn = [min(k[i] for k in keys) for i in range(4)]
        mx = [max(k[i] for k in keys) for i in range(4)]
        U = sum(max(q[i]*mn[i], q[i]*mx[i]) for i in range(4))
        true = max(sum(q[i]*k[i] for i in range(4)) for k in keys)
        assert true <= U
        stats[p] = (mn, mx, U, true)
    top = sorted(stats, key=lambda p: -stats[p][2])[:2]
    f = Fig('sparse-quest', 'Toy Quest example with a 4-channel query q = (2, −1, 1, 1) and four pages of four keys. Each page stores per-channel min and max of its keys. The bound U_p is ' + ', '.join(f'{p} {stats[p][2]}' for p in stats) + '; true best scores ' + ', '.join(f'{p} {stats[p][3]}' for p in stats) + f'. Top-2 by bound: {top[0]} and {top[1]}. Real scale: 32K context in 16-token pages is 2,048 pages; a 2,048-token budget reads 128 pages (6.25%); metadata adds 12.5% to K. Reported 7.03× self-attention speedup at 32K.', 380)
    f.math(30, 34, r'\c{green}{U_p} = \sum_i \max(\c{gold}{q_i} \c{blue}{m_{p,i}}, \c{gold}{q_i} \c{blue}{M_{p,i}}) \;\ge\; \max_{j \in p} \c{gold}{q}^{\top} \c{blue}{k_j}', 21)
    x0, y0, cw, rh = 150, 96, 104, 34
    f.text(60, y0-14, 'q (toy)', 13, 'gold', True)
    for i, v in enumerate(q):
        f.box(42, y0+i*rh, 36, 26, num(v), 'gold', True, 15, .85, 4)
        f.text(108, y0+i*rh+18, f'ch {i+1}', 13, 'muted')
    for c, p in enumerate(stats):
        mn, mx, U, true = stats[p]
        x = x0 + c*cw
        f.text(x+cw/2-6, y0-14, f'page {p}', 14, 'ink', True)
        for i in range(4):
            y = y0 + i*rh
            f.rect(x, y, cw-12, 26, 'blue', .08, 4)
            # range bar on a −2..2 axis
            ax0, ax1 = x+8, x+52
            sc = (ax1-ax0)/4
            f.line(ax0, y+13, ax1, y+13, 'line', 1)
            f.rect(ax0 + (mn[i]+2)*sc - 1, y+7, max(3, (mx[i]-mn[i])*sc + 2), 12, 'blue', .75, 2)
            f.text(x+cw-16, y+18, f'[{num(mn[i])}, {num(mx[i])}]', 13, 'blue', anchor='end')
        sel = p in top
        f.text(x+cw/2-6, y0+4*rh+18, f'U = {U}', 16, 'green' if sel else 'muted', True)
        f.text(x+cw/2-6, y0+4*rh+40, f'true max {num(true)}', 13, 'ink')
        if sel:
            outline(f, x-4, y0-30, cw-4, 4*rh+82, 'green', 2.2)
    mn, mx, U, _ = stats['A']
    terms = ' + '.join(f'max({num(q[i]*mn[i])}, {num(q[i]*mx[i])})' for i in range(4))
    f.text(30, y0+4*rh+78, f'page A: U = {terms} = {U}', 14, 'ink', anchor='start')
    f.text(30, y0+4*rh+100, 'bars: [min, max] per channel (axis −2…2) · green: top-2 by U, read exactly', 13, 'muted', anchor='start')
    px = 600
    f.panel(px, 64, 280, 300, 'Real scale · decode')
    f.lines(px+16, 112, ['32K tokens / page 16 = 2,048 pages', 'budget 2,048 tokens = 128 pages'], 13, 'ink')
    import random
    rng = random.Random(7)
    lit = rng.sample(range(2048), 128)
    sx, sw = px+16, 248
    f.rect(sx, 146, sw, 18, 'blue', .18, 2)
    for p in lit:
        f.rect(sx + p*sw/2048, 146, 1.2, 18, 'green', .95)
    f.text(sx, 182, 'read 6.25% of pages', 13, 'green', True, 'start')
    f.lines(px+16, 210, ['metadata: min + max per 16 keys', '= +12.5% of K stored'], 13, 'ink')
    f.text(px+16, 262, 'reported (§4.3):', 13, 'muted', anchor='start')
    f.text(px+16, 282, '7.03× self-attention at 32K', 14, 'green', True, 'start')
    f.text(px+16, 302, 'passkey 100K: 100% at 2,048', 13, 'ink', anchor='start')
    f.text(px+16, 322, 'layers 1–2 stay dense', 13, 'ink', anchor='start')
    f.text(px+16, 350, 'cache size unchanged', 13, 'red', True, 'start')
    f.save()


# ---------------------------------------------------------------- MoBA
def moba():
    n, cell, B = 32, 6, 4
    picks = {7: (2, 4), 6: (0, 3), 5: (1, 2), 4: (0, 2), 3: (0, 1), 2: (0, 1), 1: (0,)}
    f = Fig('sparse-moba', 'MoBA on a toy 32-token grid with 8 blocks of 4: each query block reads its own block causally plus the top-2 past blocks by q times the mean key. Gate for the last block: scores per past block with blocks 2 and 4 selected. Mean pooling can hide a single sharp key: block X with one score of 3.6 averages 0.55, below block Y at 0.95; max pooling (MSA) would pick X. Real settings: block 512 top-3 at 8K reads 18.75%; block 4,096 top-12 at 1M reads 4.69%. Reported 6.5× prefill at 1M; RULER 128K 0.7818 versus 0.7849 for full attention.', 430)
    f.math(30, 34, r'\c{green}{s_{t,b}} = \c{gold}{q_t}^{\top} \text{mean}_{j \in b}\, \c{blue}{k_j}', 21)
    f.math(380, 34, r'A_t = \text{Top}_k(s_{t,:}) ∪ \text{current block (causal)}', 21)

    def fn(i, j):
        if j > i:
            return None
        bi, bj = i // B, j // B
        if bi == bj:
            return 'blue'
        return 'green' if bj in picks.get(bi, ()) else None
    x0, y0 = 40, 100
    f.text(x0+n*cell/2, y0-12, 'toy: 8 blocks of 4, top-2', 14, 'muted')
    f.mask(x0, y0, n, cell, fn)
    legend(f, 30, 66, [('current block', 'blue', .85), ('routed block', 'green', .85)])
    # gate for last query block
    gx, gy = 290, 100
    f.text(gx, gy-12, 'gate for query block 7', 14, 'muted', anchor='start')
    scores = [.4, .9, 2.1, .6, 1.7, .3, 1.1]
    for b, s in enumerate(scores):
        y = gy + b*24
        sel = b in picks[7]
        f.text(gx+44, y+14, f'block {b}', 13, 'ink', anchor='end')
        f.rect(gx+52, y+2, s*60, 15, 'green' if sel else 'line', .9, 3)
        f.text(gx+58+s*60, y+14, f'{s}', 13, 'green' if sel else 'muted', sel, 'start')
    f.text(gx, gy+7*24+16, 'block 7: always (causal)', 13, 'blue', True, 'start')
    # mean vs max
    mx = 580
    f.text(mx, gy-12, 'mean dilutes one sharp key (toy)', 14, 'muted', anchor='start')
    X = [.1, 0, .2, 3.6, .1, 0, .2, .2]
    Y = [.9, 1.0, .8, 1.1, .9, 1.0, 1.0, .9]
    for r, (nm, vals) in enumerate((('X', X), ('Y', Y))):
        y = gy + 10 + r*92
        f.text(mx, y+50, nm, 18, 'ink', True, 'start')
        for k, v in enumerate(vals):
            h = v*14
            f.rect(mx+24+k*14, y+56-h, 11, max(1, h), 'blue', .75, 1)
        f.line(mx+22, y+56, mx+24+8*14, y+56, 'line', 1)
        mean = sum(vals)/len(vals)
        f.text(mx+150, y+30, f'mean {mean:.2f}', 14, 'green' if nm == 'Y' else 'muted', True, 'start')
        f.text(mx+150, y+50, f'max {max(vals):.1f}', 14, 'teal' if nm == 'X' else 'muted', True, 'start')
    f.text(mx, gy+200, 'MoBA (mean) picks Y', 13, 'green', True, 'start')
    f.text(mx, gy+218, 'MSA (max-pool) picks X', 13, 'teal', True, 'start')
    f.hdivider(338)
    f.text(30, 362, 'Share of keys read (paper settings)', 14, 'ink', True, 'start')
    f.bars(30, 374, [('8K · 512 × top-3', 18.75, 'green'), ('1M · 4,096 × top-12', 4.69, 'green')], width=180, bar=15, gap=8, label_w=150, size=13, value_fmt=lambda v: f'{v}%')
    f.divider(470, 348, 420)
    f.lines(490, 362, ['reported: 6.5× prefill at 1M', 'RULER 128K: 0.7818 vs 0.7849 full', 'last 3 layers kept full in SFT'], 13, 'ink')
    f.save()


# ---------------------------------------------------------------- NSA
def nsa():
    N = 65536
    cmp_n = (N - 32)//16 + 1
    total = cmp_n + 16*64 + 512
    f = Fig('sparse-nsa', f'NSA reads three key sets for one query at 64K tokens: about {cmp_n:,} compressed records (32-token blocks, stride 16), 16 selected 64-token blocks (1,024 tokens: the first block, 2 local blocks and 13 by score) and a 512-token sliding window, about {total:,} reads or {total/N*100:.1f}% of 65,536. Three sigmoid gates mix the branches. Block scores come from the compression attention, summed over the 16 heads of a GQA group, so the group shares one selection. Reported: 9.0× forward and 6.0× backward speedup at 64K versus FlashAttention-2.', 440)
    f.math(30, 34, r'o_t = \sum_{c \in \{\text{cmp}, \text{slc}, \text{win}\}} \c{teal}{g_t^c} \cdot \text{Attn}(\c{gold}{q_t}, \c{blue}{K_t^c}, \c{purple}{V_t^c})', 21)
    f.math(870, 34, r'\c{teal}{g_t^c} = \sigma(\text{MLP}(x_t))', 17, 'muted', 'end')
    ax0, ax1 = 150, 690
    sc = (ax1 - ax0)/N
    rows = [('compress', 'l 32 · stride 16', 92), ('select', "l′ 64 · top-16", 146), ('window', 'w 512', 200)]
    for name, sub, y in rows:
        f.text(30, y+12, name, 15, 'ink', True, 'start')
        f.text(30, y+29, sub, 13, 'muted', anchor='start')
        f.rect(ax0, y, ax1-ax0, 22, 'blue', .12, 2)
    # compress: every 16 tokens one record covering 32 (drawn as dense ticks)
    for k in range(0, 270):
        f.rect(ax0 + k*2, 92+3, 1.2, 16, 'purple', .7)
    # select: 16 blocks (drawn 6 px wide, not to scale)
    sel = [0, 3100, 9800, 14000, 21000, 23500, 30100, 36000, 41000, 44200, 49000, 52500, 57000, 60500, N-128, N-64]
    for k, s in enumerate(sel):
        forced = k == 0 or k >= 14
        f.rect(ax0 + s*sc - (6 if s > N-200 else 0) - (6 if s == N-128 else 0), 146+2, 6, 18, 'green', 1 if forced else .75, 1)
    win_w = max(5, 512*sc)
    f.rect(ax1 - win_w, 200+2, win_w, 18, 'green', .95, 1)
    f.text(ax0, 82, 'key position 0', 12, 'muted', anchor='start')
    f.text(ax1, 82, '65,536', 12, 'muted', anchor='end')
    for (label, val, y) in ((f'{cmp_n:,}', 'compressed', 92), ('1,024', 'tokens', 146), ('512', 'tokens', 200)):
        f.text(ax1+14, y+16, label, 15, 'green', True, 'start')
        f.text(ax1+70, y+16, val, 13, 'muted', anchor='start')
    f.text(ax0, 240, 'select: block 0 + 2 local blocks forced (dark), 13 by score · block widths not to scale', 12, 'muted', anchor='start')
    # selection score derivation
    f.hdivider(256)
    f.text(30, 280, 'Free scorer: the compress branch already has scores', 15, 'ink', True, 'start')
    vals = [.1, .3, .2, .05, .6, .8, .4, .1, .05, .1, .2, .15]
    for k, v in enumerate(vals):
        f.rect(30 + k*22, 292, 20, 20, 'purple', .15 + v, 2)
    f.text(30 + 6*22, 330, 'p_cmp (per compressed record)', 12, 'muted')
    f.arrow(300, 302, 340, 302, 'muted', width=1.4)
    blocks = [.6, 1.8, .35, .45]
    for k, v in enumerate(blocks):
        f.box(350 + k*46, 290, 40, 24, f'{v}', 'green' if v > 1 or v == .6 else 'line', v > 1 or v == .6, 13)
    f.text(350 + 92, 330, '64-block score, Σ over 16 heads', 12, 'muted')
    f.arrow(540, 302, 580, 302, 'muted', width=1.4)
    f.text(592, 300, 'top-16 per GQA group', 14, 'green', True, 'start')
    f.text(592, 318, '→ every head loads the same blocks', 13, 'ink', anchor='start')
    f.hdivider(346)
    f.text(30, 370, f'Reads per query at 64K (27B MoE, GQA 4 × 16 heads)', 14, 'ink', True, 'start')
    f.bars(30, 382, [('dense', N, 'red'), ('NSA', total, 'green')], width=270, bar=15, gap=8, label_w=60, size=13, value_fmt=lambda v: f'{v:,}' + ('' if v == N else f'  ({v/N*100:.1f}%)'))
    f.divider(520, 356, 428)
    f.lines(540, 370, ['reported vs FlashAttention-2 at 64K:', '9.0× forward · 6.0× backward', 'decode memory access: 11.6× less'], 13, 'ink')
    f.save()


# ---------------------------------------------------------------- DSA
def dsa():
    N, K = 131072, 2048
    idx_per_key = 64*128
    main_per_key = 128*(576 + 512)
    dense = main_per_key*N/1e9
    idx = idx_per_key*N/1e9
    main = main_per_key*K/1e9
    cross = K*main_per_key//idx_per_key
    f = Fig('sparse-dsa', f'DeepSeek Sparse Attention. A lightning indexer with 64 FP8 heads of 128 dims scores every past token: ReLU per head, then a weighted sum. The top 2,048 of 131,072 tokens (1.56%) are read by MQA-mode MLA, where 128 query heads share each 576-dim latent record (656 bytes in FP8). Multiply-accumulates per decode token per layer at 128K: dense MLA {dense:.1f} G, indexer {idx:.2f} G, sparse main attention {main:.2f} G; the indexer costs more than the main read once N exceeds {cross:,}. Training: 1,000-step indexer-only KL warm-up (2.1B tokens), then 15,000 sparse steps (943.7B tokens).', 460)
    f.math(30, 34, r'I_{t,s} = \sum_{j=1}^{64} \c{teal}{w_{t,j}} \cdot \text{ReLU}(\c{gold}{q^I_{t,j}} \cdot \c{blue}{k^I_s})', 21)
    f.math(500, 34, r'A_t = \text{Top}_{2048}(I_{t,:})', 21)
    # indexer shapes
    f.text(30, 66, 'Indexer (FP8)', 14, 'ink', True, 'start')
    f.matrix(96, 86, 8, 8, 12, 'gold', None, 2)
    f.brace(96, 86, 96, 86+8*12-2, '64 heads', 'muted', 'left', 13, 4)
    f.brace(96, 86, 96+8*12-2, 86, '128', 'muted', 'above', 12, 4)
    f.math(96+4*12, 200, r'\c{gold}{q^I_{t,j}}', 16, anchor='middle')
    f.vec(212, 86, 8, 12, 'blue', True)
    f.math(218, 200, r'\c{blue}{k^I_s}', 16, anchor='middle')
    f.text(218, 216, '128', 12, 'muted')
    f.vec(258, 86, 8, 12, 'teal', True)
    f.math(264, 200, r'\c{teal}{w_t}', 16, anchor='middle')
    f.text(264, 216, '64', 12, 'muted')
    f.text(30, 252, 'one key head shared by 64 query heads; ReLU each', 12, 'muted', anchor='start')
    # score strip
    sx, sw, sy = 320, 400, 92
    f.text(sx, 72, f'scores over N = 131,072 keys → top-2,048 (1.56%)', 14, 'ink', True, 'start')
    import math
    import random
    rng = random.Random(3)
    pts = []
    bins = 140
    vals = []
    for b in range(bins):
        v = .25 + .15*math.sin(b/7) + .1*rng.random() + (.55 if b in (12, 13, 47, 88, 89, 90, 121) else 0) + (.4 if b > bins-4 else 0)
        vals.append(v)
        pts.append((sx + b*sw/bins, sy + 70 - v*55))
    f.path(pts, 'gold', width=1.4)
    f.line(sx, sy+70, sx+sw, sy+70, 'line', 1)
    thr = sorted(vals)[-12]
    f.line(sx, sy+70-thr*55, sx+sw, sy+70-thr*55, 'green', 1, True)
    f.text(sx+sw+6, sy+70-thr*55+4, 'top-k', 12, 'green', True, 'start')
    f.rect(sx, sy+80, sw, 16, 'blue', .15, 2)
    for b, v in enumerate(vals):
        if v >= thr:
            f.rect(sx + b*sw/bins, sy+80, sw/bins, 16, 'green', .95)
    f.text(sx, sy+112, 'every key scored (O(N) per query); only lit keys go to the main attention', 12, 'muted', anchor='start')
    # main attention
    f.text(sx, sy+140, 'MQA-mode MLA: 128 query heads share each latent', 14, 'ink', True, 'start')
    f.box(sx, sy+150, 128, 26, '512 + 64 dims', 'blue', True, 13, .85, 4)
    f.text(sx+140, sy+168, '= 656 B / token in FP8 (vLLM layout)', 13, 'ink', anchor='start')
    f.hdivider(292)
    f.text(30, 316, 'Multiply-accumulates per decode token, one layer, N = 128K', 14, 'ink', True, 'start')
    f.bars(30, 328, [('dense MLA', round(dense, 1), 'red'), ('indexer (FP8)', round(idx, 2), 'gold'), ('main, top-2,048', round(main, 2), 'green')],
           width=300, bar=16, gap=8, label_w=130, size=13, value_fmt=lambda v: f'{v} G')
    f.text(30, 420, f'indexer > main once N > 2,048 × {main_per_key//idx_per_key} = {cross:,} tokens (derived)', 13, 'gold', True, 'start')
    f.divider(560, 302, 446)
    f.text(580, 316, 'Two-stage training (V3.2)', 14, 'ink', True, 'start')
    f.box(580, 330, 290, 46, '', 'gold', False, 13)
    f.lines(592, 349, ['dense warm-up · 1,000 steps · 2.1B tok', 'indexer only, KL to summed attention'], 13, 'ink')
    f.box(580, 386, 290, 46, '', 'green', False, 13)
    f.lines(592, 405, ['sparse · 15,000 steps · 943.7B tok', 'all weights; indexer input detached'], 13, 'ink')
    f.save()


# ---------------------------------------------------------------- 2026 indexers
def indexers():
    M = 1  # in millions of tokens
    rows = [
        ('DSA', 'V3.2 · GLM-5', 64, 1, 1, '2,048 tokens', 'gold'),
        ('IndexShare', 'GLM-5.2', 64, 1, 4, '2,048 tokens', 'gold'),
        ('CSA (→ 08)', 'V4-Flash', 64, 4, 1, '512 entries × 4', 'purple'),
        ('MSA', 'MiniMax-M3', 4, 1, 1, '16 × 128 / group', 'blue'),
        ('QSA', 'Qwen3.8-Flash-Next', 4, 4, 1, '2,048 tokens', 'teal'),
    ]
    f = Fig('sparse-indexers', 'Index dot products per query per layer at N = 1M, counted as heads times keys scored divided by layers sharing one indexer: DSA 64 heads over every token, 64M; IndexShare in GLM-5.2 reuses one indexer for 4 sparse layers, 16M; DeepSeek-V4-Flash CSA scores 1M/4 compressed entries with 64 heads, 16M; MSA uses one index head per GQA group (4) over every token, 4M; QSA scores 4-token pooled keys with 4 heads, 1M. Every design reads about 2,048 tokens per query in the main attention.', 400)
    f.math(30, 34, r'\text{index dots / query} \approx \c{gold}{H_I} \cdot N \;/\; (\c{blue}{r} \cdot \c{green}{s})', 21)
    f.text(870, 30, 'r = tokens per scored key · s = layers per indexer', 13, 'muted', anchor='end')
    hx = [30, 140, 300, 345, 385, 430, 740]
    for x, h in zip(hx, ['method', 'model', '', '', '', 'index dots per query at N = 1M', 'main read']):
        f.text(x, 76, h, 13, 'muted', True, 'start')
    f.math(hx[2], 76, r'\c{gold}{H_I}', 15)
    f.math(hx[3], 76, r'\c{blue}{r}', 15)
    f.math(hx[4], 76, r'\c{green}{s}', 15)
    for k, (name, model, H, r, s, read, col) in enumerate(rows):
        y = 104 + k*40
        dots = H*M/(r*s)
        f.text(hx[0], y, name, 15, 'ink', True, 'start')
        f.text(hx[1], y, model, 13, 'ink', anchor='start')
        f.text(hx[2], y, str(H), 15, 'gold', True, 'start')
        f.text(hx[3], y, str(r), 15, 'blue', True, 'start')
        f.text(hx[4], y, str(s), 15, 'green', True, 'start')
        w = 230*dots/64
        f.rect(hx[5], y-14, max(2, w), 18, col, .8, 3)
        f.text(hx[5]+max(2, w)+8, y, f'{dots:g}M', 14, col, True, 'start')
        f.text(hx[6], y, read, 13, 'green', True, 'start')
    f.hdivider(306)
    f.lines(30, 330, ['Same budget, cheaper scorer: every design reads ≈ 2,048 tokens per query.',
                      'The O(N) indexer pass is what is left to cut: fewer heads (MSA, QSA), pooled keys (QSA, CSA), shared layers (IndexShare).'], 14, 'ink')
    f.text(30, 384, 'index head dim 128 for DSA, CSA, QSA (CSA in FP4); MSA index dim not stated · MSA reads 16 blocks per GQA group', 12, 'muted', anchor='start')
    f.save()


def build():
    what_shrinks()
    prefill()
    quest()
    moba()
    nsa()
    dsa()
    indexers()
