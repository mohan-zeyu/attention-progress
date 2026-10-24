#!/usr/bin/env python3
"""Lighter figures for the essay "Cheaper Attention.md": the book's cards with most labels removed.

Run: python3 scripts/essay_figures.py  →  figures/essay/*.svg
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE/'cards')]
from figkit import Fig, ROOT
from kv_heads import _layer_cell

OUT = ROOT / 'figures' / 'essay'


def kv_cross_layer():
    f = Fig('kv-cross-layer', 'Which layers own a KV cache. CLA: every second layer reuses its neighbour. YOCO: one global KV read by every upper layer. Gemma 3n: layers 20 to 34 reuse layers 18 and 19. Green dashes mark where prefill can stop.', 240, 640)
    # legend
    lx, ly = 20, 24
    for kind, lab in (('G', 'global cache'), ('L', 'window cache'), ('SG', 'reuses global'), ('S', 'reuses window')):
        _layer_cell(f, lx, ly-11, 12, 14, kind)
        f.text(lx+18, ly, lab, 13, 'ink', anchor='start')
        lx += 18 + len(lab)*6.6 + 22
    f.line(lx+4, ly-12, lx+4, ly+3, 'green', 1.6, dash=True)
    f.text(lx+12, ly, 'prefill exit', 13, 'ink', anchor='start')
    x0 = 110   # strip area
    w, gap = 12, 2
    cx = lambda i, w=w, gap=gap: x0 + i*(w+gap) + w/2
    # --- CLA: layer 2i+1 reads layer 2i ---
    y = 56
    f.text(20, y+15, 'CLA', 15, 'ink', True, 'start')
    for i in range(64):
        _layer_cell(f, x0+i*7.6, y, 6, 22, 'G' if i % 2 == 0 else 'SG')
    for i in range(1, 64, 2):
        f.path([(cx(i, 6, 1.6), y+24), (cx(i, 6, 1.6)-3.8, y+30), (cx(i-1, 6, 1.6), y+24)], 'blue', width=1)
    # --- YOCO: window layers, one global KV, every upper layer reads it ---
    y = 116
    f.text(20, y+15, 'YOCO', 15, 'ink', True, 'start')
    for i in range(16):
        _layer_cell(f, x0+i*(w+gap), y, w, 22, 'L')
    gx = x0 + 16*(w+gap) + 4
    f.box(gx, y-4, 44, 30, 'KV', 'blue', True, 14)
    for i in range(16):
        xx = gx + 52 + i*(w+gap)
        _layer_cell(f, xx, y, w, 22, 'SG')
        if i % 3 == 0:
            f.path([(xx+w/2, y+24), ((xx+gx+22)/2, y+44), (gx+22, y+28)], 'blue', width=1, curve=True)
    f.line(gx+48, y-6, gx+48, y+28, 'green', 1.6, dash=True)
    # --- Gemma 3n: layers 20–34 reuse L18 (local) / L19 (global) ---
    y = 180
    f.text(20, y+15, 'Gemma 3n', 15, 'ink', True, 'start')
    first, full = 20, {4, 9, 14, 19, 24, 29, 34}
    for i in range(35):
        g = i in full
        _layer_cell(f, x0+i*(w+gap), y, w, 22, ('G' if g else 'L') if i < first else ('SG' if g else 'S'))
    xa, xb = cx(first)-7, cx(34)+7
    f.brace(xa, y+24, xb, y+24, '', 'muted', 'below', 12, 5)
    f.path([(xa+3, y+31), (xa-6, y+44), (cx(19), y+26)], 'blue', arrow=True, width=1.4, curve=True)
    f.line(cx(first)-8, y-6, cx(first)-8, y+28, 'green', 1.6, dash=True)
    f.save(OUT)


def _cell(f, x, y, w, h, kind):
    """kind: read = stored and read, kept = stored but not read, dropped = gone from the cache,
    evicted = deleted for good, new = current query."""
    style = {'read': 'fill="#3f83b6" fill-opacity=".85"',
             'kept': 'fill="white" stroke="#3f83b6" stroke-width="1.4"',
             'dropped': 'fill="none" stroke="#cad5db" stroke-width="1.2" stroke-dasharray="2 2"',
             'evicted': 'fill="none" stroke="#cf5f59" stroke-width="1.2" stroke-dasharray="2 2"',
             'new': 'fill="none" stroke="#d09124" stroke-width="1.8"'}[kind]
    f.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="1.5" {style}/>')


def read_fewer():
    f = Fig('read-fewer', 'Which past keys each query reads (rows are queries, columns are keys) and what the cache still holds after 16 tokens. Full attention reads and keeps everything. A sliding window of 4 reads the last 4 keys and can drop the rest. A sparse read picks its own block plus one earlier block chosen by content, but every record stays in the cache.', 300, 700)
    # legend
    lx, ly = 20, 24
    for kind, lab in (('read', 'read'), ('kept', 'stored, not read'), ('dropped', 'dropped'), ('new', 'new token')):
        _cell(f, lx, ly-11, 12, 12, kind)
        f.text(lx+18, ly, lab, 13, 'ink', anchor='start')
        lx += 18 + len(lab)*6.8 + 26
    n, c, W, B = 16, 11, 4, 4
    pick = {1: 0, 2: 0, 3: 1}   # earlier block each query block chooses by content
    reads = {
        'Full': lambda i, j: True,
        'Sliding window': lambda i, j: i - j < W,
        'Sparse read': lambda i, j: j//B == i//B or pick.get(i//B) == j//B,
    }
    gy, sy = 76, 76 + n*c + 14
    f.text(70, gy + n*c/2 + 4, 'reads', 13, 'muted', anchor='end')
    f.text(70, sy + 11, 'cache', 13, 'muted', anchor='end')
    for k, (title, fn) in enumerate(reads.items()):
        x = 82 + k*210
        f.text(x + n*c/2, gy - 12, title, 15, 'ink', True)
        f.mask(x, gy, n, c, lambda i, j: (('blue', .85) if fn(i, j) else ('line', .55)) if j <= i else None)
        _cell(f, x - 1.5, gy + (n-1)*c - 1.5, n*c, c + 1.5, 'new')
        for j in range(n):
            kind = 'read' if fn(n-1, j) else ('dropped' if title == 'Sliding window' else 'kept')
            _cell(f, x + j*c, sy, c - 2, 14, kind)
    f.save(OUT)


def moba():
    f = Fig('moba', 'MoBA on a toy cache of 32 tokens in 8 blocks of 4. Each block is summarised by its mean key. The query (last token) scores every earlier block with one dot product against the mean key, keeps the top 2 (blocks 2 and 4) plus its own block, and reads only the tokens of those 3 blocks; the rest stay in the cache unread.', 226, 680)
    n_blk, B, c, gap, pitch = 8, 4, 13, 2, 70
    score = [.4, .7, 1.5, .3, 1.2, .5, .6]   # q · mean key for the 7 earlier blocks
    picked = {2, 4, 7}                        # top 2 by score + the query's own block
    x0, y1, y2, base, y4 = 110, 30, 76, 168, 186
    bx = lambda b: x0 + b*pitch
    bc = lambda b: bx(b) + (B*c - gap)/2
    f.text(96, y1+15, 'keys', 13, 'muted', anchor='end')
    f.text(96, y2+13, 'mean key', 13, 'muted', anchor='end')
    f.text(96, base-14, 'score', 13, 'muted', anchor='end')
    f.text(96, y4+15, 'read', 13, 'muted', anchor='end')
    f.line(x0-4, base, bx(n_blk-1)+B*c, base, 'line', 1.2)
    for b in range(n_blk):
        x = bx(b)
        for i in range(B):
            f.rect(x + i*c, y1, c-gap, 22, 'blue', .35, 2)
            _cell(f, x + i*c, y4, c-gap, 22, 'read' if b in picked else 'kept')
        # funnel from the block to its mean key
        f.path([(x, y1+26), (bc(b), y2-5), (x + B*c - gap, y1+26)], 'line', width=1.2)
        f.rect(bc(b)-9, y2, 18, 18, 'blue', .95 if b < n_blk-1 else .3, 2)   # own block is never scored
        if b < n_blk-1:
            h = score[b]*44
            f.rect(bc(b)-9, base-h, 18, h, 'green' if b in picked else 'line', .9 if b in picked else .8, 2)
        else:
            f.text(bc(b), base-8, 'own', 13, 'green', True)
        if b in picked:
            f.parts.append(f'<rect x="{x-4}" y="{y4-4}" width="{B*c-gap+8}" height="30" rx="4" fill="none" stroke="#1f9a74" stroke-width="1.6"/>')
    # the query is the last token
    q = bx(n_blk-1) + (B-1)*c
    for y in (y1, y4):
        _cell(f, q-1.5, y-1.5, c-gap+3, 25, 'new')
    f.text(q + (c-gap)/2, y1-6, 'q', 14, 'gold', True)
    f.save(OUT)


def nsa():
    f = Fig('nsa', 'NSA on a toy cache of 40 tokens. Compressed branch: every 4 tokens become one record, all read, shaded by the query\'s attention. Selected branch: blocks of 8; the first and local blocks are always read, and block 2 is picked because its compressed records scored highest. Window branch: the last 6 tokens. A gate adds the three results.', 236, 720)
    n, c, x0 = 40, 12, 120
    tx = lambda j: x0 + j*c
    yk, yc, ys, yw = 34, 80, 126, 186
    for lab, y in (('keys', yk), ('compressed', yc), ('selected', ys), ('window', yw)):
        f.text(106, y+15, lab, 13, 'muted', anchor='end')
    for j in range(n):
        f.rect(tx(j), yk, c-2, 22, 'blue', .35, 2)
        _cell(f, tx(j), ys, c-2, 22, 'read' if j//8 in (0, 2, 4) else 'kept')
        _cell(f, tx(j), yw, c-2, 22, 'read' if j >= n-6 else 'dropped')
    # compressed branch: one record per 4 tokens, shaded by the query's attention weight
    score = [.5, .35, .2, .3, .95, .8, .25, .2, .45, .55]
    for b, s in enumerate(score):
        f.rect(tx(4*b), yc, 4*c-2, 22, 'blue', s, 3)
    # selected branch: first and local blocks always, block 2 by the compressed scores
    for b, tag in ((0, 'first'), (2, 'top score'), (4, 'local')):
        f.parts.append(f'<rect x="{tx(8*b)-3}" y="{ys-3}" width="{8*c+4}" height="28" rx="4" fill="none" stroke="#1f9a74" stroke-width="1.6"/>')
        f.text(tx(8*b) + (8*c-2)/2, ys+42, tag, 12, 'green' if b == 2 else 'muted', b == 2)
    for b in (4, 5):
        xm = tx(4*b) + (4*c-2)/2
        f.line(xm, yc+24, xm, ys-5, 'green', 1.4, dash=True, arrow=True)
    # gate: the three branch outputs are weighted and added
    gx, gy = 668, ys+11
    for y in (yc, ys, yw):
        f.line(tx(n)+4, y+11, gx-15, gy + (y-ys)*.25, 'muted', 1.2, arrow=True)
    f.circle(gx, gy, 13, '+', 'muted', False, 16)
    f.line(gx+13, gy, gx+32, gy, 'muted', 1.2, arrow=True)
    f.text(gx+42, gy+5, 'o', 16, 'green', True)
    # the query is the last token
    for y in (yk, ys, yw):
        _cell(f, tx(n-1)-1.5, y-1.5, c+1, 25, 'new')
    f.text(tx(n-1) + (c-2)/2, yk-6, 'q', 14, 'gold', True)
    f.save(OUT)


def dsa():
    f = Fig('dsa', 'DSA on a toy cache of 40 tokens. Every token keeps a small indexer key; the lightning indexer scores all of them for the query, and only the top 8 tokens, scattered anywhere in the cache, are read by the main attention.', 176, 620)
    n, c, x0 = 40, 12, 120
    tx = lambda j: x0 + j*c
    score = [.25, .15, .3, .85, .2, .35, .1, .25, .3, .9, .75, .2, .15, .3, .25, .1, .35, .8, .2, .3,
             .15, .25, .2, .3, .7, .2, .1, .35, .25, .15, .3, .78, .2, .25, .3, .15, .35, .72, .4, .95]
    top = set(sorted(range(n), key=lambda j: -score[j])[:8])
    yk, base, yr = 40, 116, 130
    f.text(106, yk+8, 'index keys', 13, 'muted', anchor='end')
    f.text(106, base-14, 'score', 13, 'muted', anchor='end')
    f.text(106, yr+15, 'read', 13, 'muted', anchor='end')
    f.line(x0-4, base, tx(n), base, 'line', 1.2)
    for j in range(n):
        f.rect(tx(j), yk, c-2, 8, 'blue', .55, 1.5)
        h = score[j]*52
        f.rect(tx(j)+1, base-h, c-4, h, 'green' if j in top else 'line', .9 if j in top else .8, 1.5)
        _cell(f, tx(j), yr, c-2, 22, 'read' if j in top else 'kept')
    for y, h in ((yk, 8), (yr, 22)):
        _cell(f, tx(n-1)-1.5, y-1.5, c+1, h+3, 'new')
    f.text(tx(n-1) + (c-2)/2, yk-8, 'q', 14, 'gold', True)
    f.save(OUT)


def _box(f, x, y, w, h, color='#1f9a74'):
    f.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="none" stroke="{color}" stroke-width="1.6"/>')


def keep_fewer():
    f = Fig('keep-fewer', 'A toy cache of 32 tokens, where token 9 will be needed by a later question. Eviction keeps 10 exact records (a few important ones plus the recent ones) and deletes token 9 for good. Compression keeps 8 records, each merging 4 tokens, so token 9 survives only blurred inside a merged record.', 166, 640)
    n, c, x0 = 32, 13, 110
    tx = lambda j: x0 + j*c
    need, kept = 9, {1, 4, 14, 20} | set(range(26, 32))
    yt, ye, yc = 40, 82, 124
    for lab, y in (('tokens', yt), ('evict', ye), ('compress', yc)):
        f.text(96, y+15, lab, 13, 'muted', anchor='end')
    for j in range(n):
        f.rect(tx(j), yt, c-2, 22, 'blue', .35, 2)
        _cell(f, tx(j), ye, c-2, 22, 'read' if j in kept else 'evicted')
    for b in range(n//4):
        f.rect(tx(4*b), yc, 4*c-2, 22, 'blue', .55, 3)
    # the token a later question needs
    xn = tx(need) + (c-2)/2
    f.parts.append(f'<path d="M{xn-5} {yt-12}L{xn+5} {yt-12}L{xn} {yt-4}Z" fill="#253340"/>')
    f.text(xn+10, yt-6, 'needed later', 12, 'ink', anchor='start')
    f.text(xn, ye+16, '×', 15, 'red', True)
    f.circle(xn, yc+11, 3.5, '', 'ink', True)
    f.text(tx(n)+8, ye+15, '10 / 32', 13, 'blue', True, 'start')
    f.text(tx(n)+8, yc+15, '8 / 32', 13, 'blue', True, 'start')
    f.save(OUT)


def h2o():
    f = Fig('h2o', 'H2O on a toy decode of 16 tokens with a budget of 6 records: the 3 tokens with the highest accumulated attention score (1, 4 and 9) plus the 3 most recent tokens stay; every other token is evicted for good.', 170, 460)
    n, c, x0 = 16, 18, 100
    tx = lambda j: x0 + j*c
    S = [.6, 2.4, .5, .7, 2.0, .4, .6, .3, .5, 1.7, .4, .3, .35, .2, .1, .05]
    heavy, recent = {1, 4, 9}, {13, 14, 15}
    base, yc = 84, 98
    f.text(86, base-14, 'score', 13, 'muted', anchor='end')
    f.text(86, yc+15, 'cache', 13, 'muted', anchor='end')
    f.line(x0-4, base, tx(n), base, 'line', 1.2)
    for j in range(n):
        h = S[j]*24
        f.rect(tx(j)+2, base-h, c-6, h, 'green' if j in heavy else 'line', .9 if j in heavy else .8, 2)
        _cell(f, tx(j), yc, c-3, 22, 'read' if j in heavy | recent else 'evicted')
    for j in heavy:
        _box(f, tx(j)-3, yc-3, c+3, 28)
    _box(f, tx(13)-3, yc-3, 3*c+3, 28)
    f.text(tx(4) + (c-3)/2, yc+44, 'heavy hitters', 12, 'green', True)
    f.text(tx(14) + (c-3)/2, yc+44, 'recent', 12, 'green', True)
    _cell(f, tx(15)-1.5, yc-1.5, c, 25, 'new')
    f.save(OUT)


def snapkv():
    f = Fig('snapkv', 'SnapKV on a toy 40-token prompt. The last 8 tokens (the observation window) vote with their attention over the earlier 32. Votes are max-pooled over neighbours, so the tokens next to a peak are kept too. The top 9 positions plus the window stay; the rest of the prompt is evicted.', 178, 640)
    n, w, c, x0 = 40, 8, 13, 110
    tx = lambda j: x0 + j*c
    v = [.15, .1, .2, .15, .1, .3, 1.0, .25, .1, .15, .2, .1, .5, .15, .1, .2, .1, .2, .9, .3, .1, .15, .1, .2, .25, .8, .2, .1, .15, .2, .1, .15]
    pooled = [max(v[max(0, j-1):j+2]) for j in range(n-w)]
    keep = set(sorted(range(n-w), key=lambda j: (-pooled[j], j))[:9])
    base, yc = 98, 112
    f.text(96, base-14, 'votes', 13, 'muted', anchor='end')
    f.text(96, yc+15, 'cache', 13, 'muted', anchor='end')
    f.line(x0-4, base, tx(n-w), base, 'line', 1.2)
    for j in range(n):
        if j < n-w:
            h = v[j]*56
            f.rect(tx(j)+1, base-h, c-4, h, 'green' if j in keep else 'line', .9 if j in keep else .8, 1.5)
        _cell(f, tx(j), yc, c-2, 22, 'read' if j in keep or j >= n-w else 'evicted')
    # the observation window votes
    _box(f, tx(n-w)-3, yc-3, w*c+4, 28, '#d09124')
    f.text(tx(n-w) + (w*c-2)/2, yc+44, 'observation window', 12, 'gold', True)
    xw = tx(n-w) + (w*c-2)/2
    f.path([(xw, yc-6), (xw-14, base-44), (tx(n-w)-12, base-50)], 'gold', arrow=True, width=1.4, curve=True)
    f.save(OUT)


def compress_v4():
    f = Fig('compress-v4', 'DeepSeek-V4 compression on a toy cache of 52 tokens. CSA emits one entry every 4 tokens, each pooling 8 tokens so neighbours overlap; an indexer picks the top 3 entries. HCA emits one entry every 16 tokens (128 in the real model) and reads them all. Both also read the last 4 raw tokens.', 196, 700)
    n, c, x0, rw = 52, 11, 110, 4
    tx = lambda j: x0 + j*c
    yt, ycs, yh = 40, 100, 152
    for lab, y in (('tokens', yt), ('CSA', ycs), ('HCA', yh)):
        f.text(96, y+15, lab, 13, 'muted', anchor='end')
    for j in range(n):
        f.rect(tx(j), yt, c-2, 22, 'blue', .35, 2)
    for y in (ycs, yh):
        for j in range(n-rw, n):
            _cell(f, tx(j), y, c-2, 22, 'read')
        _cell(f, tx(n-1)-1.5, y-1.5, c+1, 25, 'new')
    # CSA: entry i pools blocks i-1 and i (8 tokens), one entry per 4 tokens; top 3 are read
    picked = {2, 7, 9}
    for b in range(12):
        x = tx(4*b)
        if b in picked:
            f.rect(x, ycs, 4*c-2, 22, 'blue', .85, 3)
            _box(f, x-3, ycs-3, 4*c+4, 28)
        else:
            f.parts.append(f'<rect x="{x}" y="{ycs}" width="{4*c-2}" height="22" rx="3" fill="white" stroke="#3f83b6" stroke-width="1.4"/>')
    f.path([(tx(24), yt+26), (tx(28), ycs-6), (tx(32)-2, ycs-6), (tx(32)-2, yt+26)], 'muted', width=1)
    # HCA: one entry per 16 tokens, all read
    for b in range(3):
        f.rect(tx(16*b), yh, 16*c-2, 22, 'blue', .85, 3)
    f.text(tx(n-1) + (c-2)/2, yt-6, 'q', 14, 'gold', True)
    f.save(OUT)


def _quant(xs, b=2):
    """Min-max quantize a group to b bits and decode it back."""
    lo, hi = min(xs), max(xs)
    d = (hi - lo) / (2**b - 1) or 1
    return [lo + d*round((x - lo)/d) for x in xs]


def low_bits():
    f = Fig('low-bits', 'Left: bits per cached number in BF16, FP8 and FP4, plus the small share taken by the group scales. Right: a toy key cache of 8 tokens by 8 channels with two outlier channels, quantized to 2 bits. Grouping per token lets each outlier stretch the scale, so the ordinary numbers collapse to two levels; grouping per channel keeps them close to the original.', 236, 760)
    import math
    # bits per number: payload + scale overhead (FP8: 4-byte scale per 128, FP4: 1-byte scale per 16)
    f.text(20, 40, 'bits per number', 13, 'muted', anchor='start')
    for k, (lab, bits, scale) in enumerate((('BF16', 16, 0), ('FP8', 8, .25), ('FP4', 4, .5))):
        y = 62 + k*44
        f.text(60, y+15, lab, 13, 'ink', True, 'end')
        f.rect(70, y, bits*8, 22, 'blue', .8, 2)
        if scale:
            f.rect(70 + bits*8 + 1, y, max(2, scale*8), 22, 'gold', .9, 1)
        f.text(70 + (bits + scale)*8 + 8, y+15, str(bits), 13, 'ink', anchor='start')
    f.rect(20, 204, 10, 10, 'gold', .9, 1)
    f.text(36, 213, 'scale', 12, 'muted', anchor='start')
    # toy keys: rows = tokens, columns = channels; channels 2 and 5 are outliers
    n, c = 8, 16
    K = [[(6 + .8*math.sin(i)) if ch == 2 else (-5.5 + .6*math.cos(i)) if ch == 5 else .9*math.sin(1.3*i + 2.1*ch)
          for ch in range(n)] for i in range(n)]
    per_token = [_quant(row) for row in K]
    cols = [_quant([K[i][ch] for i in range(n)]) for ch in range(n)]
    per_channel = [[cols[ch][i] for ch in range(n)] for i in range(n)]
    tone = lambda v: ('ink', .85) if abs(v) > 3 else ('blue' if v >= 0 else 'purple', .08 + .82*min(1, abs(v)/.9))
    for k, (title, M) in enumerate((('keys', K), ('per token', per_token), ('per channel', per_channel))):
        x, y = 268 + k*168, 66
        f.text(x + n*c/2, y-14, title, 14, 'ink', True)
        f.mask(x, y, n, c, lambda i, j: tone(M[i][j]))
        if k == 1:
            _box(f, x-3, y + 3*c - 3, n*c + 4, c + 4, '#253340')
        if k == 2:
            _box(f, x + 3*c - 3, y-3, c + 4, n*c + 4, '#253340')
    f.save(OUT)


def linear_state():
    f = Fig('linear-state', 'Left: a KV cache adds one record per token (4, 8, 16 records), while a linear-attention state stays one fixed matrix. Right: the write step adds the outer product of the new value and key to the state.', 196, 720)
    # left: cache grows, state does not
    x0, yc, ys, p = 84, 58, 106, 8
    f.text(70, yc+15, 'cache', 13, 'muted', anchor='end')
    f.text(70, ys+25, 'state', 13, 'muted', anchor='end')
    vals = lambda r, c, s: .25 + .6*((r*5 + c*3 + s) % 7)/6
    x = x0
    for t in (4, 8, 16):
        f.text(x + t*p/2, yc-12, f't = {t}', 13, 'ink', anchor='middle')
        for j in range(t):
            f.rect(x + j*p, yc, p-2, 22, 'blue', .7, 1.5)
        sx = x + t*p/2 - 24
        f.matrix(sx, ys, 6, 6, 8, 'purple', [[vals(r, c, t) for c in range(6)] for r in range(6)], 1.5)
        x += t*p + 40
    f.divider(436, 30, 176)
    # right: S_t = S_{t-1} + v_t k_t^T
    v, k = [.9, .3, .6, .2, .8, .4], [.5, .9, .2, .7, .3, .6]
    S0 = [[vals(r, c, 1)*.7 for c in range(6)] for r in range(6)]
    outer = [[v[r]*k[c] for c in range(6)] for r in range(6)]
    S1 = [[min(1, S0[r][c] + outer[r][c]) for c in range(6)] for r in range(6)]
    y, cell = 86, 8
    f.matrix(460, y, 6, 6, cell, 'purple', S0, 1.5, '$S_{t-1}', size=12)
    f.text(522, y+29, '+', 18, 'ink')
    f.matrix(540, y, 6, 1, cell, 'purple', [[a] for a in v], 1.5)
    f.matrix(552, y-12, 1, 6, cell, 'blue', [k], 1.5)
    f.matrix(552, y, 6, 6, cell, 'purple', outer, 1.5)
    f.math(546, y+66, r'v_t k_t^{\top}', 13, 'purple')
    f.text(614, y+29, '=', 18, 'ink')
    f.matrix(634, y, 6, 6, cell, 'purple', S1, 1.5, '$S_t', size=12)
    f.save(OUT)


if __name__ == '__main__':
    kv_cross_layer()
    read_fewer()
    moba()
    nsa()
    dsa()
    keep_fewer()
    h2o()
    snapkv()
    compress_v4()
    low_bits()
    linear_state()
