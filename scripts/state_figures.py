#!/usr/bin/env python3
"""Figures for "Fixed-Size States.md", the detailed companion to Part 4 of the essay.

Run: python3 scripts/state_figures.py  →  figures/states/*.svg
"""
import math
from essay_figures import Fig, ROOT, _cell, _box

OUT = ROOT / 'figures' / 'states'


def _signed(v, scale=1.0):
    """Blue for positive, red for negative, opacity by size."""
    return ('blue' if v >= 0 else 'red', .06 + .84*min(1, abs(v)/scale))


def readout():
    f = Fig('readout', 'Key similarity k_i·k_j for unit keys. With 4 orthogonal keys in 4 dimensions only the diagonal is non-zero, so reading with key i returns exactly v_i. With 8 keys in 4 dimensions the off-diagonal entries are non-zero, so every read mixes in other values (crosstalk).', 262, 560)
    n8 = []
    for j in range(8):
        k = [math.cos(1.1*j), math.sin(.7*j + .3), math.cos(2.3*j + 1), math.sin(1.9*j + .5)]
        r = math.sqrt(sum(x*x for x in k))
        n8.append([x/r for x in k])
    G8 = [[sum(a*b for a, b in zip(ki, kj)) for kj in n8] for ki in n8]
    G4 = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    c = 18
    for x, G, title, tag in ((60, G4, '4 keys, d = 4', 'exact'), (300, G8, '8 keys, d = 4', 'crosstalk')):
        n = len(G)
        f.text(x + n*c/2, 36, title, 14, 'ink', True)
        f.mask(x, 50, n, c, lambda i, j: _signed(G[i][j]))
        f.text(x + n*c/2, 50 + n*c + 22, tag, 13, 'green' if tag == 'exact' else 'red', True)
    f.text(60, 248, 'row = read with key i, column = stored key j', 12, 'muted', anchor='start')
    f.save(OUT)


def chunkwise():
    f = Fig('chunkwise', 'Chunkwise form on 16 toy tokens in chunks of 4. Inside each chunk, the diagonal block is a small masked attention. Everything below the diagonal blocks comes from the state carried from earlier chunks; the state is updated once per chunk.', 300, 520)
    n, C, c, x0, y0 = 16, 4, 14, 60, 50
    def fn(i, j):
        if j > i:
            return None
        return ('blue', .85) if j//C == i//C else ('purple', .35)
    f.mask(x0, y0, n, c, fn)
    for b in range(1, n//C):
        f.line(x0, y0 + b*C*c, x0 + n*c, y0 + b*C*c, 'line', 1)
        f.line(x0 + b*C*c, y0, x0 + b*C*c, y0 + n*c, 'line', 1)
    # the state passed from chunk to chunk
    sx = x0 + n*c + 40
    for b in range(n//C):
        y = y0 + b*C*c + 8
        f.rect(sx, y, 40, 40, 'purple', .55, 4)
        f.math(sx + 20, y + 25, rf'S_{{[{b}]}}', 13, 'white', anchor='middle')
        if b < n//C - 1:
            f.line(sx + 20, y + 42, sx + 20, y + C*c - 2, 'purple', 1.4, arrow=True)
    f.text(x0, 30, 'queries ↓  keys →', 12, 'muted', anchor='start')
    lx = sx + 70
    f.rect(lx, 70, 12, 12, 'blue', .85, 2); f.text(lx + 18, 81, 'inside the chunk', 13, 'ink', anchor='start')
    f.rect(lx, 96, 12, 12, 'purple', .35, 2); f.text(lx + 18, 107, 'from the state', 13, 'ink', anchor='start')
    f.save(OUT)


def decay_masks():
    f = Fig('decay-masks', 'Effective weight on past token j for query t, D_tj = product of the decays between them, on 12 toy tokens. No decay: every past token keeps full weight. Fixed decay 0.8: weights fade with distance. Input-dependent decay 0.95 except 0.1 at token 5 and 0.5 at token 9: the small decay at token 5 cuts almost every path that crosses it.', 240, 600)
    n, c = 12, 12
    alpha = [.95]*n
    alpha[5], alpha[9] = .1, .5
    def D(i, j, a):
        p = 1.0
        for s in range(j + 1, i + 1):
            p *= a[s]
        return p
    panels = (('no decay', lambda i, j: 1.0), ('fixed decay', lambda i, j: .8**(i - j)), ('input-dependent', lambda i, j: D(i, j, alpha)))
    for k, (title, w) in enumerate(panels):
        x = 40 + k*190
        f.text(x + n*c/2, 36, title, 14, 'ink', True)
        f.mask(x, 50, n, c, lambda i, j: ('blue', .05 + .85*w(i, j)) if j <= i else None)
    # mark token 5 in the third panel
    xm = 40 + 2*190 + 5*c + c/2
    f.parts.append(f'<path d="M{xm-5} {50 + n*c + 14}L{xm+5} {50 + n*c + 14}L{xm} {50 + n*c + 6}Z" fill="#cf5f59"/>')
    f.text(xm, 50 + n*c + 30, 'α = 0.1', 12, 'red', True)
    f.save(OUT)


def gla_channels():
    f = Fig('gla-channels', 'A toy 4×6 state multiplied by Diag(alpha) with alpha = 1, 0.95, 0.5, 0.9, 0.1, 0.99: each key-channel column fades at its own rate and channel 5 is nearly wiped. A scalar decay multiplies every column by the same 0.8.', 230, 600)
    rows, cols, c = 4, 6, 20
    S = [[.35 + .55*((r*5 + col*3) % 7)/6 for col in range(cols)] for r in range(rows)]
    for k, (lab, a) in enumerate((('per channel', [1, .95, .5, .9, .1, .99]), ('scalar', [.8]*6))):
        y = 40 + k*96
        f.text(20, y + 44, lab, 13, 'muted', anchor='start')
        f.matrix(120, y, rows, cols, c, 'purple', S, 2)
        f.text(258, y + 46, '×', 18, 'ink')
        for j, v in enumerate(a):
            f.rect(280 + j*c, y + 30, c - 2, c - 2, 'green', .1 + .8*v, 2)
            f.text(280 + j*c + (c - 2)/2, y + 26, f'{v:g}'.lstrip('0') if v < 1 else '1', 10, 'muted')
        f.text(420, y + 46, '=', 18, 'ink')
        f.matrix(444, y, rows, cols, c, 'purple', [[S[r][j]*a[j] for j in range(cols)] for r in range(rows)], 2)
    f.save(OUT)


def delta_toy():
    f = Fig('delta-toy', 'The order-status toy. The state maps two order keys to a shipping day: order 7314 → 2 (Tuesday) and order 2208 → 4. The new fact is order 7314 → 5 (Friday). Adding gives 7, a mix. The delta rule with beta 1 gives exactly 5 and keeps 4. Gated DeltaNet (alpha 0.8, beta 0.5) gives 3.3 and fades 2208 to 3.2. KDA with alpha (0.8, 1) keeps 2208 at 4. The sum form gives 3.1 and 3.2.', 300, 600)
    rows = (('before', 2, 4), ('add', 7, 4), ('delta, β = 1', 5, 4), ('gated delta, α = 0.8, β = 0.5', 3.3, 3.2),
            ('KDA, α = (0.8, 1), β = 0.5', 3.3, 4), ('sum form, α = 0.8, β = 0.5', 3.1, 3.2))
    x1, x2, w, h = 330, 440, 90, 26
    f.text(x1 + w/2, 30, 'order 7314', 13, 'ink', True)
    f.text(x2 + w/2, 30, 'order 2208', 13, 'ink', True)
    f.text(x1 + w/2, 46, 'should become 5', 11, 'muted')
    f.text(x2 + w/2, 46, 'should stay 4', 11, 'muted')
    tone = lambda v, goal, start: 'green' if v == goal else ('red' if (v - goal)*(start - goal) < 0 or abs(v - goal) > abs(start - goal) else 'gold')
    for k, (lab, a, b) in enumerate(rows):
        y = 60 + k*36
        f.text(310, y + 18, lab, 13, 'ink', k == 0, 'end')
        for x, v, goal in ((x1, a, 5), (x2, b, 4)):
            col = 'line' if k == 0 else tone(v, goal, 2 if goal == 5 else 4)
            f.rect(x, y, w, h, col, .25 if k == 0 else .3, 4, stroke=col)
            f.text(x + w/2, y + 18, f'{v:g}', 14, 'ink', True)
    for k, (col, lab) in enumerate((('green', 'right'), ('gold', 'partly fixed'), ('red', 'wrong or damaged'))):
        x = 40 + k*110
        f.rect(x, 274, 12, 12, col, .3, 2, stroke=col)
        f.text(x + 18, 285, lab, 12, 'ink', anchor='start')
    f.save(OUT)


def householder():
    f = Fig('householder', 'The transition I − beta k kᵀ acts only along the key direction k. A vector x = (0.8, 0.6) with k along the horizontal axis: beta 0.5 halves its k-component, beta 1 erases it, and beta 2 flips it, a reflection with eigenvalue −1.', 196, 600)
    u = 70
    for p, (beta, title) in enumerate(((.5, 'β = 0.5: shrink'), (1, 'β = 1: erase'), (2, 'β = 2: reflect'))):
        ox, oy = 100 + p*200, 170
        f.text(ox, 34, title, 14, 'ink', True)
        f.line(ox - 85, oy, ox + 85, oy, 'blue', 1.4, dash=True)            # key direction
        f.text(ox + 88, oy + 4, 'k', 13, 'blue', True, 'start')
        f.line(ox, oy + 12, ox, oy - 110, 'line', 1.2)                       # what is left untouched
        x, y = .8, .6
        nx = x*(1 - beta)
        f.line(ox, oy, ox + x*u, oy - y*u, 'muted', 2, arrow=True)
        f.text(ox + x*u + 6, oy - y*u - 4, 'x', 13, 'muted', True, 'start')
        f.line(ox, oy, ox + nx*u, oy - y*u, 'purple', 2.2, arrow=True)
        if beta != 1:
            f.line(ox + x*u, oy - y*u, ox + nx*u + (4 if nx > 0 else -4)*(1 if x > nx else -1), oy - y*u, 'line', 1, dash=True)
    f.save(OUT)


def transitions():
    f = Fig('transitions', 'The transition matrix A_t of each rule as a toy 6×6 heat map, blue positive and red negative: identity (linear attention), a scalar times identity (RetNet, Mamba-2), a diagonal (GLA), I − beta k kᵀ (DeltaNet), alpha (I − beta k kᵀ) (Gated DeltaNet), Diag(alpha)(I − beta k kᵀ) (KDA).', 170, 700)
    n, c = 6, 11
    k = [1, 1, 0, 0, 1, 0]
    r = math.sqrt(sum(x*x for x in k))
    k = [x/r for x in k]
    H = [[(1 if i == j else 0) - k[i]*k[j] for j in range(n)] for i in range(n)]   # beta = 1
    a = [1, .95, .5, .9, .1, .99]
    mats = (('linear attn', 'I', [[1.0 if i == j else 0 for j in range(n)] for i in range(n)]),
            ('RetNet · Mamba-2', 'αI', [[.7 if i == j else 0 for j in range(n)] for i in range(n)]),
            ('GLA', 'Diag(α)', [[a[i] if i == j else 0 for j in range(n)] for i in range(n)]),
            ('DeltaNet', 'I − βkkᵀ', H),
            ('Gated DeltaNet', 'α(I − βkkᵀ)', [[.7*v for v in row] for row in H]),
            ('KDA', 'Diag(α)(I − βkkᵀ)', [[a[i]*H[i][j] for j in range(n)] for i in range(n)]))
    for p, (name, formula, M) in enumerate(mats):
        x = 26 + p*112
        f.text(x + n*c/2, 34, name, 12, 'ink', True)
        f.mask(x, 46, n, c, lambda i, j: _signed(M[i][j]) if abs(M[i][j]) > 1e-9 else None)
        f.text(x + n*c/2, 46 + n*c + 20, formula, 12, 'purple')
    f.text(26, 150, 'diagonal: decays only', 12, 'muted', anchor='start')
    f.text(26 + 3*112, 150, 'off-diagonal: edits along k', 12, 'muted', anchor='start')
    f.save(OUT)


def ttt_batch():
    f = Fig('ttt-batch', 'Mini-batch TTT on 32 toy tokens. With b = 1 every gradient uses the latest weights: 32 sequential steps. With b = 8 the gradients in a batch share the batch-start weights: 4 steps, each one matrix product. With b = T there is one step, which equals linear attention.', 170, 640)
    n, c, x0 = 32, 13, 70
    for k, (b, lab, steps) in enumerate(((1, 'b = 1', '32 steps'), (8, 'b = 8', '4 steps'), (32, 'b = T', '1 step'))):
        y = 34 + k*42
        f.text(x0 - 12, y + 15, lab, 13, 'ink', True, 'end')
        for j in range(n):
            g = j // b
            f.rect(x0 + j*c, y, c - 2, 20, 'purple', .75 if g % 2 == 0 else .35, 2)
        f.text(x0 + n*c + 14, y + 15, steps, 13, 'ink', anchor='start')
    f.text(x0 + n*c + 14, 34 + 2*42 + 32, '= linear attention', 12, 'muted', anchor='start')
    f.save(OUT)


def log_linear():
    f = Fig('log-linear', 'Log-linear attention on 16 toy tokens: each query splits its past into power-of-two buckets and keeps one state per bucket. Row 13 uses buckets 0–7, 8–11 and 12, plus its own token: 4 states instead of 14 records.', 290, 560)
    n, c, x0, y0 = 16, 14, 60, 46
    colors = {0: 'gold', 1: 'blue', 2: 'teal', 3: 'green', 4: 'purple'}   # bucket size 2^(level-1); 0 = own token
    def level(t, s):
        if s == t:
            return 0
        start = 0
        for b in range(4, -1, -1):
            if t >> b & 1:
                if start <= s < start + 2**b:
                    return b + 1
                start += 2**b
    f.mask(x0, y0, n, c, lambda i, j: (colors[level(i, j)], .8) if j <= i else None)
    _box(f, x0 - 2, y0 + 13*c - 2, 14*c + 2, c + 2, '#253340')
    f.text(x0 - 8, y0 + 13*c + 11, '13', 12, 'ink', True, 'end')
    f.text(x0, 30, 'queries ↓  keys →', 12, 'muted', anchor='start')
    lx = x0 + n*c + 40
    f.text(lx, 70, 'bucket', 13, 'muted', anchor='start')
    for k, (lev, lab) in enumerate(((0, 'own token'), (1, '1 token'), (2, '2 tokens'), (3, '4 tokens'), (4, '8 tokens'))):
        y = 84 + k*24
        f.rect(lx, y, 12, 12, colors[lev], .8, 2)
        f.text(lx + 18, y + 11, lab, 13, 'ink', anchor='start')
    f.save(OUT)


if __name__ == '__main__':
    for fig in (readout, chunkwise, decay_masks, gla_channels, delta_toy, householder, transitions, ttt_batch, log_linear):
        fig()
