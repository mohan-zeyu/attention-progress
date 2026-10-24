"""05 · Fixed patterns: masks decided before any query sees the data."""
from figkit import Fig

GIB = 2**30


def legend(f, x, y, items, size=13, gap=18):
    """Small colour squares with labels, laid out left to right."""
    for label, color, op in items:
        f.rect(x, y-10, 12, 12, color, op, 2, 'line' if color == 'paper' else None)
        f.text(x+17, y, label, size, 'ink', anchor='start')
        x += 17 + len(label)*size*.52 + gap
    return x


def mask_panel(f, x, y, n, cell, fn, title, sub):
    f.text(x + n*cell/2, y-30, title, 17, bold=True)
    f.text(x + n*cell/2, y-12, sub, 13, 'muted')
    f.mask(x, y, n, cell, fn)
    count = sum(1 for i in range(n) for j in range(n) if fn(i, j) is not None)
    return count


# ---------------------------------------------------------------- history
def lineage():
    n, cell = 24, 8
    f = Fig('fixed-lineage', 'Three fixed sparse patterns on a toy 24×24 query×key grid. Sparse Transformer (2019): local band of l = 4 plus every 4th key, causal, so any pair connects in two hops (example 23 → 3 → 1). Longformer (2020): window of ±3 plus global tokens 0 and 1. BigBird (2020): 3-block window, one global block and one random block per query block. Real configs at n = 4,096: Sparse Transformer with l = 64 reads at most 128 keys per query; Longformer window 512; BigBird-ITC 128 global + 192 window + 192 random = 512 keys. GPT-3 alternated dense and banded layers; gpt-oss does the same with a 128-token band.', 500)
    f.math(30, 34, r'\c{blue}{A_t^{(1)}} = \{t-l, \dots, t\}', 20)
    f.math(250, 34, r'\c{teal}{A_t^{(2)}} = \{j : (t-j) \text{ mod } l = 0\}', 20)
    f.math(560, 34, r'l \approx √n \;\to\; O(n√n) \text{ pairs}', 20)

    def sparse_tf(i, j):
        if j > i:
            return None
        if i - j <= 4 - 1:
            return 'blue'
        if (i - j) % 4 == 0:
            return 'teal'
        return None

    def longformer(i, j):
        if i in (0, 1) or j in (0, 1):
            return 'green'
        if abs(i - j) <= 3:
            return 'blue'
        return None

    rnd = {0: 5, 1: 6, 2: 7, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4}

    def bigbird(i, j):
        bi, bj = i // 3, j // 3
        if bi == 0 or bj == 0:
            return 'green'
        if abs(bi - bj) <= 1:
            return 'blue'
        if rnd[bi] == bj:
            return 'purple'
        return None

    legend(f, 30, 66, [('local band', 'blue', .85), ('strided', 'teal', .85), ('global', 'green', .85), ('random', 'purple', .85), ('not computed', 'paper', 1)])
    xs = [60, 350, 640]
    y0 = 136
    c1 = mask_panel(f, xs[0], y0, n, cell, sparse_tf, 'Sparse Transformer · 2019', 'causal · band l + stride l')
    c2 = mask_panel(f, xs[1], y0, n, cell, longformer, 'Longformer · 2020', 'bidirectional · window + global')
    c3 = mask_panel(f, xs[2], y0, n, cell, bigbird, 'BigBird · 2020', 'blocks of 3 · window + global + random')
    # two-hop path in the Sparse Transformer mask (annotation sits in the empty upper triangle)
    for (i, j) in ((23, 3), (3, 1)):
        f.parts.append(f'<rect x="{xs[0]+j*cell-1.5}" y="{y0+i*cell-1.5}" width="{cell+1.5}" height="{cell+1.5}" fill="none" stroke="#d09124" stroke-width="2.2"/>')
    f.path([(xs[0]+3*cell+4, y0+23*cell-2), (xs[0]+3*cell+4, y0+3*cell+10)], 'gold', True, 1.4, True)
    f.text(xs[0]+96, y0+40, 'q 23 → k 3 → k 1', 13, 'gold', True, 'start')
    f.text(xs[0]+96, y0+56, 'any pair: two hops', 13, 'gold', anchor='start')
    for x, c, tot in ((xs[0], c1, n*(n+1)//2), (xs[1], c2, n*n), (xs[2], c3, n*n)):
        f.text(x + n*cell/2, y0+n*cell+20, f'toy n = 24: {c} of {tot} pairs', 13, 'muted')
    f.text(xs[0]+n*cell/2, y0+n*cell+40, 'real: l ≈ √n, two heads', 14, 'ink')
    f.text(xs[1]+n*cell/2, y0+n*cell+40, 'real: w = 512 at n = 4,096', 14, 'ink')
    f.text(xs[2]+n*cell/2, y0+n*cell+40, 'real: b = 64 · g 2b · w 3b · r 3b', 14, 'ink')
    f.hdivider(388)
    f.text(30, 412, 'Keys per query at n = 4,096', 15, 'ink', True, 'start')
    f.bars(30, 424, [('full', 4096, 'red'), ('strided, l = 64', 128, 'teal'), ('Longformer', 512, 'blue'), ('BigBird-ITC', 512, 'green')],
           width=200, bar=13, gap=5, label_w=125, size=13, value_fmt=lambda v: ('≈' if v == 128 else '') + f'{v:,}')
    f.divider(440, 398, 494)
    f.text(460, 412, 'Same idea, one layer at a time', 15, 'ink', True, 'start')
    f.strip(460, 428, [('D', 'blue'), ('B', 'line'), ('D', 'blue'), ('B', 'line'), ('D', 'blue'), ('B', 'line')], w=20, h=19, size=11)
    f.text(600, 443, 'GPT-3 · 2020 · dense / banded, n_ctx 2,048', 13, 'ink', anchor='start')
    f.strip(460, 458, [('B', 'line'), ('D', 'blue'), ('B', 'line'), ('D', 'blue'), ('B', 'line'), ('D', 'blue')], w=20, h=19, size=11)
    f.text(600, 473, 'gpt-oss · 2025 · band 128 / dense', 13, 'ink', anchor='start')
    f.save()


# ---------------------------------------------------------------- production SWA
MODELS = [
    # name, year, layers, ratio text, window text, first-12 pattern
    ('Mistral 7B', '2023', 32, 'all local', '4,096', 'L'*12),
    ('Gemma 2 27B', '2024', 46, '1 : 1', '4,096', 'LG'*6),
    ('Gemma 3 27B', '2025', 62, '5 : 1', '1,024', 'LLLLLG'*2),
    ('gpt-oss-120b', '2025', 36, '1 : 1', '128', 'LG'*6),
    ('Llama 4 Scout', '2025', 48, '3 : 1', 'chunk 8,192', 'CCCG'*3),
    ('OLMo 3 7B', '2025', 32, '3 : 1', '4,096', 'LLLG'*3),
    ('MiMo-V2-Flash', '2026', 48, '39 : 9', '128', 'GLLLLGLLLLLG'),
    ('DeepSeek-V4-Flash', '2026', 43, 'L1–2 local', '128', 'LL'),
]


def kv_128k(layers_g, layers_l, per_tok_g, per_tok_l, w, n=131072):
    return (layers_g*per_tok_g*n + layers_l*per_tok_l*min(n, w)) / GIB


def swa_layouts():
    rows = [
        # name, actual GiB, all-global GiB
        ('Gemma 3 27B', kv_128k(10, 52, 8192, 8192, 1024), kv_128k(62, 0, 8192, 0, 0)),
        ('Llama 4 Scout', kv_128k(12, 36, 4096, 4096, 8192), kv_128k(48, 0, 4096, 0, 0)),
        ('gpt-oss-120b', kv_128k(18, 18, 2048, 2048, 128), kv_128k(36, 0, 2048, 0, 0)),
        ('MiMo-V2-Flash', kv_128k(9, 39, 2*4*160*2, 2*8*160*2, 128), kv_128k(48, 0, 2*4*160*2, 0, 0)),
    ]
    desc = ('First twelve layers of eight production models, local (L), global (G) or chunked (C): Mistral 7B all local W 4,096; Gemma 2 1:1 W 4,096; '
            'Gemma 3 5:1 W 1,024; gpt-oss 1:1 W 128; Llama 4 Scout 3:1 chunk 8,192; OLMo 3 3:1 W 4,096; MiMo-V2-Flash 39 local and 9 global, W 128; '
            'DeepSeek-V4-Flash two local layers then CSA/HCA. KV cache at 128K tokens in BF16 versus an all-global stack: '
            + '; '.join(f'{n} {a:.1f} vs {g:.1f} GiB' for n, a, g in rows) + '.')
    f = Fig('fixed-swa-layouts', desc, 460)
    f.math(30, 34, r'\text{KV}(N) = \c{red}{N} \cdot \sum_{\text{global}} c_{\ell} + \c{green}{\min(N, W)} \cdot \sum_{\text{local}} c_{\ell}', 21)
    f.math(870, 34, r'c_{\ell} = 2 H_{kv} d_h s \text{ per layer}', 17, 'muted', 'end')
    hx = [30, 190, 240, 300, 400, 520]
    for x, h in zip(hx, ['model', 'year', 'layers', 'local : global', 'W', 'first 12 layers']):
        f.text(x, 66, h, 13, 'muted', True, 'start')
    tone = {'L': ('blue', .3), 'G': ('blue', .95), 'C': ('teal', .45)}
    for r, (name, year, layers, ratio, win, pat) in enumerate(MODELS):
        y = 88 + r*25
        f.text(hx[0], y, name, 14, 'ink', True, 'start')
        f.text(hx[1], y, year, 14, 'muted', anchor='start')
        f.text(hx[2], y, str(layers), 14, 'ink', anchor='start')
        f.text(hx[3], y, ratio, 14, 'ink', anchor='start')
        f.text(hx[4], y, win, 14, 'teal' if win.startswith('chunk') else 'ink', True, 'start')
        for k, ch in enumerate(pat):
            c, op = tone[ch]
            f.rect(hx[5] + k*21, y-13, 18, 17, c, op, 3)
            f.text(hx[5] + k*21 + 9, y, ch, 11, 'white' if op > .5 else 'ink', True)
        if name.startswith('DeepSeek'):
            f.text(hx[5] + 2*21 + 6, y, 'then CSA / HCA + 128 SWA branch (ch. 08)', 13, 'muted', anchor='start')
    legend(f, 30, 300, [('L local window', 'blue', .3), ('G global (full cache)', 'blue', .95), ('C chunked', 'teal', .45)])
    f.text(870, 300, 'trend: W 4,096 → 1,024 → 128', 14, 'green', True, 'end')
    f.hdivider(314)
    f.text(30, 338, 'KV cache at N = 128K, BF16, whole model', 15, 'ink', True, 'start')
    f.text(870, 338, 'computed from each config.json', 13, 'muted', anchor='end')
    for r, (name, act, glob) in enumerate(rows):
        y = 352 + r*26
        f.text(170, y+13, name, 14, 'ink', anchor='end')
        scale = 440 / 62
        f.rect(180, y, glob*scale, 9, 'red', .35, 2)
        f.rect(180, y+11, max(2, act*scale), 9, 'blue', .9, 2)
        f.text(186 + glob*scale, y+9, f'all global {glob:.1f} GiB', 12, 'red', anchor='start')
        f.text(186 + act*scale, y+20, f'{act:.1f} GiB ({act/glob*100:.0f}%)', 12, 'blue', True, 'start')
    f.text(870, 432, 'MiMo: K 192 + V 128 dims; global layers 4 KV heads,', 12, 'muted', anchor='end')
    f.text(870, 448, 'local layers 8; "all global" = 48 × its global layer', 12, 'muted', anchor='end')
    f.save()


# ---------------------------------------------------------------- chunked vs sliding
def chunked():
    n, cell, W = 24, 8, 6
    f = Fig('fixed-chunked', 'Toy 24×24 masks with equal size 6: a sliding window gives every query the last 6 keys; chunked attention gives a query only the keys of its own 6-token chunk, so keys per query saw-tooth from 1 to 6 (mean 3.5) and the cache empties at each boundary. Llama 4 Scout: 36 chunked RoPE layers with C = 8,192 and 12 global NoPE layers; at 128K in BF16 a chunked layer holds at most 32 MiB of KV versus 512 MiB for a global layer.', 420)
    f.math(30, 34, r'\c{teal}{A_t^{\text{chunk}}} = \{ j \le t : ⌊j / C⌋ = ⌊t / C⌋ \}', 20)
    f.math(400, 34, r'\text{vs.}', 18, 'muted')
    f.math(460, 34, r'\c{blue}{A_t^{\text{SWA}}} = \{ j : t - W < j \le t \}', 20)
    swa = lambda i, j: 'blue' if 0 <= i - j < W else None
    chk = lambda i, j: 'teal' if j <= i and j // W == i // W else None
    xs = [50, 320]
    y0 = 104
    mask_panel(f, xs[0], y0, n, cell, swa, 'Sliding window', 'toy W = 6')
    mask_panel(f, xs[1], y0, n, cell, chk, 'Chunked (Llama 4)', 'toy C = 6')
    for k in range(1, 4):
        x = xs[1] + k*W*cell - 1
        f.line(x, y0+k*W*cell-2, x, y0+n*cell+2, 'red', 1, True)
    f.parts.append(f'<rect x="{xs[1]+12*cell-1.5}" y="{y0+12*cell-1.5}" width="{cell+1.5}" height="{cell+1.5}" fill="none" stroke="#cf5f59" stroke-width="2.2"/>')
    f.callout(xs[1]+12*cell+4, y0+12*cell, xs[1]+108, y0+40, 'query 12: 1 key', 'red', 13)
    f.text(xs[1]+112, y0+58, '(chunk start)', 13, 'red', anchor='start')
    # keys-per-query profiles
    base = y0 + n*cell + 64
    for x, fn, color, lab in ((xs[0], lambda t: min(t+1, W), 'blue', 'keys / query: 6 after warm-up'),
                              (xs[1], lambda t: t % W + 1, 'teal', 'keys / query: 1…6, mean 3.5')):
        pts = []
        for t in range(n):
            v = fn(t)
            pts += [(x + t*cell, base - v*6), (x + (t+1)*cell, base - v*6)]
        f.path(pts, color, width=2)
        f.line(x, base, x + n*cell, base, 'line', 1)
        f.text(x + n*cell/2, base + 18, lab, 13, color, True)
    # real Llama 4 numbers
    px = 590
    f.panel(px, 64, 290, 340, 'Llama 4 Scout · 48 layers')
    f.strip(px+16, 100, [('C', 'teal'), ('C', 'teal'), ('C', 'teal'), ('G', 'blue')]*3, w=19, h=26, gap=2, size=11)
    f.text(px+16, 146, 'C: chunk 8,192 + RoPE (36 layers)', 13, 'teal', True, 'start')
    f.text(px+16, 166, 'G: full causal + NoPE (12 layers)', 13, 'blue', True, 'start')
    f.text(px+16, 198, 'KV per layer at 128K, BF16', 14, 'ink', True, 'start')
    f.text(px+16, 216, '8 KV heads × 128 = 4 KiB / token', 13, 'muted', anchor='start')
    f.bars(px+16, 228, [('global', 512, 'blue'), ('chunked', 32, 'teal')], width=120, bar=16, gap=8, label_w=70, size=13, value_fmt=lambda v: f'{v} MiB')
    f.text(px+16, 302, 'whole model at 128K: 7.1 GiB', 14, 'green', True, 'start')
    f.text(px+16, 320, 'vs 24 GiB if all global (30%)', 13, 'muted', anchor='start')
    f.text(px+16, 352, 'keys per query, C = W = 8,192:', 13, 'ink', anchor='start')
    f.text(px+16, 370, 'SWA 8,192 · chunked mean 4,096.5', 13, 'ink', True, 'start')
    f.text(px+16, 388, 'chunked: half the pairs, blind spots', 13, 'red', anchor='start')
    f.save()


# ---------------------------------------------------------------- StreamingLLM
def streaming():
    n, cell, W = 24, 8, 6
    f = Fig('fixed-streaming', 'Toy 24×24 masks with a 6-token window. Window only: once token 0 leaves the cache, Llama-2-13B perplexity on PG19 jumps to 5158.07 (cache 1024). StreamingLLM keeps the first 4 tokens as sinks plus 1020 recent tokens: perplexity 5.40. Positions are assigned inside the cache: text positions 0,1,2,3,6,7,8 and new token 9 become 0 to 7. The cache stays 4 + W records: 1 GiB for Llama-2-7B with 2,048 slots, at any length.', 440)
    f.math(30, 34, r'A_t = \c{green}{\{0,1,2,3\}} ∪ \c{blue}{\{t-W+1, \dots, t\}}', 21)
    f.math(470, 34, r'\text{pos}(j) = \text{rank of } j \text{ inside the cache}', 19)
    win = lambda i, j: 'blue' if 0 <= i - j < W else (('red', .18) if j < i - W + 1 else None)
    sink = lambda i, j: 'green' if j < 4 and j <= i else ('blue' if 0 <= i - j < W else (('red', .18) if j <= i else None))
    xs = [50, 320]
    y0 = 122
    mask_panel(f, xs[0], y0, n, cell, win, 'Window only', 'toy W = 6, sinks evicted')
    mask_panel(f, xs[1], y0, n, cell, sink, 'StreamingLLM', '4 sinks + W = 6')
    yb = y0 + n*cell + 26
    f.text(xs[0]+96, yb, 'PPL 5158.07', 18, 'red', True)
    f.text(xs[1]+96, yb, 'PPL 5.40', 18, 'green', True)
    f.text(xs[0]+96+135, yb+24, 'Llama-2-13B, PG19 · cache 0 + 1024 vs 4 + 1020 (reported)', 13, 'muted')
    legend(f, 30, 66, [('sink', 'green', .85), ('window', 'blue', .85), ('evicted', 'red', .18)])
    # position remap
    px = 590
    f.panel(px, 64, 290, 210, 'Positions inside the cache')
    toks = [0, 1, 2, 3, 6, 7, 8, 9]
    for k, t in enumerate(toks):
        c = 'green' if t < 4 else ('gold' if t == 9 else 'blue')
        f.box(px+18+k*32, 120, 28, 28, str(t), c, True, 14, .85, 4)
        f.box(px+18+k*32, 176, 28, 28, str(k), c, False, 14, 1, 4)
        f.arrow(px+32+k*32, 150, px+32+k*32, 172, 'muted', width=1)
    f.text(px+18, 112, 'text position', 13, 'muted', anchor='start')
    f.text(px+18, 218, 'RoPE position used', 13, 'muted', anchor='start')
    f.text(px+18, 240, 'tokens 4, 5 evicted; 9 is new', 13, 'red', anchor='start')
    f.text(px+18, 258, 'example from the paper, §3.2', 12, 'muted', anchor='start')
    f.panel(px, 290, 290, 126, 'Cost: constant')
    f.text(px+16, 336, 'cache = 4 + W records, any length', 13, 'ink', anchor='start')
    f.text(px+16, 356, 'Llama-2-7B, 2,048 slots × 512 KiB', 13, 'ink', anchor='start')
    f.text(px+16, 378, '= 1 GiB, streamed to 4M tokens', 14, 'green', True, 'start')
    f.text(px+16, 400, 'up to 22.2× vs window + recompute', 13, 'muted', anchor='start')
    f.save()


def build():
    lineage()
    swa_layouts()
    chunked()
    streaming()
