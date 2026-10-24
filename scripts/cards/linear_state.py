"""09 · Linear attention & SSMs: one matrix state, different decays."""
import math
from figkit import Fig, COL


# ---------------------------------------------------------------- helpers
def numgrid(f, x, y, vals, cw=44, ch=30, color='purple', vmax=None, size=15, fmt=None, gap=3, colors=None):
    """Matrix of numbers: cell opacity follows |value|/vmax, the number is printed on top."""
    vmax = vmax or max(abs(v) for row in vals for v in row) or 1
    for r, row in enumerate(vals):
        for c, v in enumerate(row):
            tone = colors[r][c] if colors else color
            op = .08 + .5*min(1, abs(v)/vmax)
            f.rect(x+c*cw, y+r*ch, cw-gap, ch-gap, tone, op, 4)
            f.text(x+c*cw+(cw-gap)/2, y+r*ch+(ch-gap)/2+size*.36, (fmt or '{:g}').format(v), size, 'ink')
    return len(vals[0])*cw-gap, len(vals)*ch-gap


def logbars(f, x, y, items, floor, width, bar=16, gap=6, label_w=170, size=14, fmt=None):
    """Horizontal bars on a log10 axis starting at 10**floor."""
    logs = [(name, math.log10(v)-floor, c) for name, v, c in items]
    return f.bars(x, y, logs, width=width, bar=bar, gap=gap, label_w=label_w, size=size,
                  value_fmt=fmt or (lambda v: f'{10**(v+floor):,.0f}'))


def stateicon(f, x, y, rows, cols, cell, color='purple', seed=1, fade=None):
    vals = [[.25+.6*(((r*5+c*3+seed) % 7)/6) * (fade[c] if fade else 1) for c in range(cols)] for r in range(rows)]
    return f.matrix(x, y, rows, cols, cell, color, vals)


# ---------------------------------------------------------------- card 1
def reassociate():
    f = Fig('linear-reassociate', 'Softmax attention forms an N×N score matrix and keeps N key/value records. Linear attention drops the softmax, reassociates (QKᵀ)V = Q(KᵀV) and keeps one d_v×d_k state per head: 16,384 numbers at d_h = 128, against 2·N·128 for the KV cache (1.05M at 4K, 33.6M at 128K, 268M at 1M tokens). They break even at N = 64.', 462)
    f.math(30, 36, r'\c{green}{o_t} = \c{purple}{S_t}\c{gold}{q_t}, \quad \c{purple}{S_t} = \c{purple}{S_{t-1}} + \c{purple}{v_t}\c{blue}{k_t}^\top', 22)
    f.math(870, 36, r'(\c{gold}{Q}\c{blue}{K}^\top)\c{purple}{V} \to \c{gold}{Q}(\c{blue}{K}^\top\c{purple}{V})', 20, anchor='end')
    # left: softmax
    f.panel(20, 56, 420, 276, 'Softmax · (QKᵀ)V')
    n, cell, mx, my = 10, 17, 52, 124
    f.mask(mx, my, n, cell, lambda i, j: ('blue', .75) if j <= i else None)
    f.brace(mx, my, mx+n*cell-2, my, 'N × N scores', side='above', size=13)
    f.brace(mx, my, mx, my+n*cell-2, 'N', side='left', size=13)
    f.text(292, 110, 'KV cache', 14, 'muted', True)
    for i in range(n):
        f.rect(250, my+i*cell, 40, cell-4, 'blue', .8, 2)
        f.rect(294, my+i*cell, 40, cell-4, 'purple', .8, 2)
    f.text(340, my+(n-1)*cell+11, '← +1 per token', 13, 'red', anchor='start')
    f.text(230, 318, 'grows with N: N² d FLOPs, 2·N·d_h numbers', 14, 'red')
    # right: linear
    f.panel(460, 56, 420, 276, 'Linear · Q(KᵀV): one fixed state')
    sx, sy, sc = 604, 134, 18
    f.vec(574, sy, 8, sc, 'purple', True, [.35+.6*((i*3) % 5)/4 for i in range(8)])
    f.vec(sx, sy-24, 8, sc, 'blue', False, [.3+.6*((i*2+1) % 5)/4 for i in range(8)])
    stateicon(f, sx, sy, 8, 8, sc)
    f.brace(sx, sy-24, sx+8*sc-2, sy-24, '$d_k = 128', side='above', size=14)
    f.brace(574, sy, 574, sy+8*sc-2, '$d_v = 128', side='left', size=14)
    f.math(581, sy+8*sc+18, r'\c{purple}{v_t}', 16, anchor='middle')
    f.math(sx+8*sc+6, sy-12, r'\c{blue}{k_t}^\top', 16)
    f.vec(786, sy, 8, sc, 'gold', True, [.8]*8)
    f.vec(846, sy+36, 4, sc, 'green', True, [.85]*4)
    f.arrow(806, sy+72, 842, sy+72, 'green', width=1.6)
    f.math(794, sy+8*sc+18, r'\c{gold}{q_t}', 16, anchor='middle')
    f.math(824, sy+62, r'\c{purple}{S_t}\c{gold}{q_t}', 13, anchor='middle')
    f.math(853, sy+36+4*sc+20, r'\c{green}{o_t}', 16, anchor='middle')
    f.text(sx+72, 318, '128 × 128 = 16,384 numbers, for any N', 14, 'green', True)
    # bottom: log bars
    f.text(30, 362, 'Numbers stored per head, d_h = 128 (log scale)', 15, 'ink', True, 'start')
    rows = [('state S, any N', 16384, 'purple'), ('KV cache @ 4K', 2*4096*128, 'blue'),
            ('KV cache @ 128K', 2*131072*128, 'blue'), ('KV cache @ 1M', 2*1048576*128, 'red')]
    logbars(f, 30, 374, rows, 3, 300)
    f.lines(870, 394, ['break-even: 2·N·128 = 128²', '→ N = 64 tokens', 'at 1M: 16,384× smaller'], 14, 'muted', 'end')
    f.save()


# ---------------------------------------------------------------- card 2
def fixed_decay():
    f = Fig('linear-fixed-decay', 'Fixed per-head decay. Left: toy decay masks D_tj = γ^(t−j) for γ = 0.7 and γ = 0.95. RetNet experiments space 1−γ from 1/32 to 1/512, horizons of about 32 to 512 tokens. Right: MiniMax-01 lightning decay λ = exp(−s_h(1−ℓ/79)) with ALiBi slopes s_h = 2^(−h/8); the grid gives the horizon 1/(1−λ) in tokens per head and layer, from about 1.7 tokens (head 1, layer 0) to tens of thousands of tokens in the last lightning layer. Bottom: the 80-layer MiniMax-01 stack with one softmax layer after every seven lightning layers.', 452)
    f.math(30, 36, r'\c{purple}{S_t} = \c{teal}{γ_h}\c{purple}{S_{t-1}} + \c{purple}{v_t}\c{blue}{k_t}^\top \quad ⇔ \quad O = (\c{gold}{Q}\c{blue}{K}^\top \odot \c{teal}{D})\c{purple}{V}, \; \c{teal}{D_{tj}} = \c{teal}{γ}^{t-j}', 21)
    f.panel(20, 56, 370, 290, 'RetNet · D for two heads (toy γ)')
    for p, g in enumerate((0.7, 0.95)):
        x0 = 44 + p*176
        f.mask(x0, 110, 12, 12.5, lambda i, j, g=g: ('teal', max(.06, g**(i-j))) if j <= i else None)
        f.math(x0+75, 280, f'γ = {g}', 16, 'teal', 'middle')
        f.text(x0+75, 300, 'short head' if p == 0 else 'long head', 13, 'muted')
    f.text(205, 96, 'row t: weight γ^(t−j) fades with distance', 13, 'muted')
    f.text(205, 326, 'real: 1−γ log-spaced 1/32 … 1/512 (§3.1)', 13, 'ink')
    f.text(205, 342, '→ horizons ≈ 32 … 512 tokens, fixed per head', 13, 'ink')
    # lightning grid
    f.panel(404, 56, 476, 290, 'MiniMax-01 lightning · horizon 1/(1−λ) in tokens')
    heads = (1, 8, 16, 32, 48, 64)
    layers = (0, 20, 40, 60, 78)
    gx, gy, cw, chh = 520, 118, 70, 30
    for c, l in enumerate(layers):
        f.text(gx+c*cw+33, gy-8, f'ℓ = {l}', 13, 'muted')
    f.text(gx-62, gy-8, 'head', 13, 'muted', anchor='start')  # noqa
    for r, h in enumerate(heads):
        s = 2**(-h/8)
        f.text(gx-12, gy+r*chh+19, f'h = {h}', 13, 'ink', anchor='end')
        for c, l in enumerate(layers):
            lam = math.exp(-s*(1-l/79+1e-5))
            hor = 1/(1-lam)
            op = .08 + .7*min(1, math.log10(hor)/4.5)
            f.rect(gx+c*cw, gy+r*chh, cw-3, chh-3, 'teal', op, 3)
            txt = f'{hor:.1f}' if hor < 10 else (f'{hor:,.0f}' if hor < 1e5 else '>10⁵')
            f.text(gx+c*cw+33, gy+r*chh+19, txt, 13, 'white' if op > .55 else 'ink')
    f.math(642, 318, r'\c{teal}{λ_{h,ℓ}} = \exp(-s_h(1 - ℓ/79)), \; s_h = 2^{-h/8}', 16, anchor='middle')
    f.text(642, 337, 'fixed per head and layer, not per token; weaker with depth', 13, 'muted')
    # layer strip
    f.text(30, 378, 'MiniMax-01, 80 layers', 14, 'ink', True, 'start')
    cells = [('', 'blue' if (i % 8) == 7 else 'purple') for i in range(80)]
    f.strip(30, 388, cells, w=8.6, h=22, gap=1.4)
    f.rect(30, 424, 12, 12, 'purple', .88, 2)
    f.text(48, 435, 'lightning (state 64 heads × 128 × 128 = 1.05M numbers)', 13, 'ink', anchor='start')
    f.rect(470, 424, 12, 12, 'blue', .88, 2)
    f.text(488, 435, 'softmax GQA, 8 KV heads (KV grows with N): 10 of 80', 13, 'ink', anchor='start')
    f.save()


# ---------------------------------------------------------------- card 3
def gla():
    f = Fig('linear-gla', 'GLA multiplies the state by Diag(α_t): each key channel (column) fades at its own data-dependent rate, toy α = 1.0, 0.95, 0.5, 0.9, 0.1, 0.99. A scalar decay (Mamba-2, mLSTM) fades every column by the same 0.8. The 1/16 temperature maps sigmoid outputs 0.01, 0.1, 0.5 and 0.9 to α = 0.75, 0.866, 0.958 and 0.993. Bottom right: what one decay value covers in each design.', 440)
    f.math(30, 36, r'\c{purple}{S_t} = \c{purple}{S_{t-1}}\,\text{Diag}(\c{teal}{α_t}) + \c{purple}{v_t}\c{blue}{k_t}^\top, \quad \c{teal}{α_t} = σ(W_α x_t + b)^{1/16} \in (0,1)^{d_k}', 21)
    alpha = [1.0, .95, .5, .9, .1, .99]
    base = [[.3+.6*(((r*5+c*3+2) % 7)/6) for c in range(6)] for r in range(4)]
    cell = 29
    xs = (40, 330, 640)
    titles = ('S_{t-1} (toy 4 × 6)', 'GLA: × Diag(α_t)', 'scalar: × 0.8')
    f.math(xs[0]+77, 80, r'\c{purple}{S_{t-1}} \text{ (toy, 4 × 6)}', 16, anchor='middle')
    f.math(xs[1]+77, 80, r'\text{GLA: } \c{purple}{S_{t-1}}\text{Diag}(\c{teal}{α_t})', 16, anchor='middle')
    f.math(xs[2]+77, 80, r'\text{Mamba-2, mLSTM: } \c{teal}{0.8}\,\c{purple}{S_{t-1}}', 16, anchor='middle')
    # alpha bars above GLA columns
    for c, a in enumerate(alpha):
        f.rect(xs[1]+c*cell, 132-36*a, cell-4, 36*a, 'teal', .8, 2)
        f.text(xs[1]+c*cell+12, 146, f'{a:g}', 12.5, 'teal', True)
    for c in range(6):
        f.rect(xs[2]+c*cell, 132-36*.8, cell-4, 36*.8, 'teal', .8, 2)
    f.text(xs[2]+86, 146, '0.8 for every column', 13, 'teal', True)
    my = 156
    f.matrix(xs[0], my, 4, 6, cell, 'purple', base)
    f.matrix(xs[1], my, 4, 6, cell, 'purple', [[base[r][c]*alpha[c] for c in range(6)] for r in range(4)])
    f.matrix(xs[2], my, 4, 6, cell, 'purple', [[base[r][c]*.8 for c in range(6)] for r in range(4)])
    f.brace(xs[0], my, xs[0]+6*cell-2, my, 'key channels', side='above', size=12)
    f.brace(xs[0], my, xs[0], my+4*cell-2, '$d_v', side='left', size=13)
    f.arrow(xs[0]+190, my+50, xs[0]+278, my+50, 'teal', width=2)
    f.text(xs[0]+234, my+40, 'per-column decay', 13, 'teal')
    f.line(xs[2]-34, 96, xs[2]-34, my+30, 'line', 1.2, dash=True)
    f.line(xs[2]-34, my+64, xs[2]-34, 290, 'line', 1.2, dash=True)
    f.text(xs[2]-34, my+52, 'vs', 15, 'muted', True)
    f.callout(xs[1]+4*cell+13, my+4*cell+4, xs[1]+150, my+4*cell+26, 'channel 5 nearly wiped (α = 0.1)', 'red', 13, 'start')
    f.callout(xs[1]+13, my+4*cell+4, xs[1]-6, my+4*cell+26, 'channel 1 kept (α = 1)', 'green', 13, 'end')
    f.hdivider(306)
    # temperature table
    f.text(30, 334, 'Temperature 1/16 keeps decay near 1', 15, 'ink', True, 'start')
    for i, s in enumerate((0.01, 0.1, 0.5, 0.9)):
        y = 356 + i*20
        f.text(60, y, f'σ = {s:g}', 13, 'muted', anchor='start')
        a = s**(1/16)
        f.rect(130, y-11, 200*a, 13, 'teal', .75, 2)
        f.text(136+200*a, y, f'α = {a:.3g}', 13, 'teal', True, 'start')
    # granularity icons
    f.text(470, 334, 'What one decay value covers', 15, 'ink', True, 'start')
    icons = [('scalar per head', 'RetNet, Mamba-2, mLSTM', lambda r, c: .55),
             ('per key channel', 'GLA, HGRN2, KDA', lambda r, c: .15+.8*alpha[c % 6]*(.6 if c % 2 else 1)),
             ('per entry', 'Mamba-1 (no matmul form)', lambda r, c: .2+.7*(((r*7+c*5) % 9)/8))]
    for i, (name, who, fn) in enumerate(icons):
        x0 = 470 + i*140
        f.matrix(x0, 346, 4, 6, 13, 'teal', [[fn(r, c) for c in range(6)] for r in range(4)])
        f.text(x0+38, 414, name, 13, 'teal', True)
        f.text(x0+38, 431, who, 12, 'muted')
    f.save()


# ---------------------------------------------------------------- card 4
def ssd():
    a = [None, .9, .2, .95, .95, .9, .5, .95]
    n = len(a)

    def L(t, j):
        if j > t:
            return None
        p = 1.0
        for s in range(j+1, t+1):
            p *= a[s]
        return p
    f = Fig('linear-ssd', 'Mamba-2 written as masked attention. Left: the 8×8 mask L with L_tj = a_(j+1)…a_t for toy decays a_t = 0.9, 0.2, 0.95, 0.95, 0.9, 0.5, 0.95; the small a_3 = 0.2 cuts every path that crosses token 3. Middle: dictionary x↔v, B↔k, C↔q, N↔d_k, P↔d_v. Right: Nemotron 3 Nano numbers: 64 heads of 64×128 state = 524,288 numbers per Mamba-2 layer, 8 B/C groups, chunk 128, and 23 Mamba-2 + 23 MoE + 6 attention layers.', 470)
    f.math(30, 36, r'\c{purple}{S_t} = \c{teal}{a_t}\c{purple}{S_{t-1}} + \c{teal}{Δ_t}\c{purple}{x_t}\c{blue}{B_t}^\top, \; \c{green}{y_t} = \c{purple}{S_t}\c{gold}{C_t} \quad ⇔ \quad \c{green}{Y} = (\c{teal}{L} \circ \c{gold}{C}\c{blue}{B}^\top)(\c{teal}{Δ}\c{purple}{X})', 21)
    mx, my, cell = 100, 108, 26
    for j in range(1, n):
        f.text(mx+j*cell+12, my-8, f'{a[j]:g}'.lstrip('0'), 12, 'teal', True)
    f.math(mx-10, my-8, r'\c{teal}{a_t}', 14, anchor='end')
    f.mask(mx, my, n, cell, lambda i, j: ('teal', max(.07, L(i, j))) if L(i, j) is not None else None)
    for i in range(n):
        for j in range(i+1):
            v = L(i, j)
            f.text(mx+j*cell+12, my+i*cell+17, '1' if v == 1 else f'{v:.2f}'[1:], 11.5, 'white' if v > .55 else 'ink')
    f.brace(mx, my+n*cell+2, mx+n*cell-2, my+n*cell+2, 'key position j (B)', side='below', size=13)
    f.brace(mx, my, mx, my+n*cell-2, 'query t (C)', side='left', size=13)
    f.text(mx+n*cell/2, my+n*cell+44, 'a₃ = 0.2 cuts every path across token 3', 13, 'red')
    # dictionary
    f.panel(330, 64, 200, 256, 'SSM ↔ attention')
    pairs = [(r'\c{purple}{x}', r'\c{purple}{v}', 'value'), (r'\c{blue}{B}', r'\c{blue}{k}', 'key'),
             (r'\c{gold}{C}', r'\c{gold}{q}', 'query'), (r'\c{teal}{L}', r'\c{teal}{\text{mask}}', 'causal × decay'),
             (r'N', r'd_k', 'state size'), (r'P', r'd_v', 'head dim')]
    for i, (src, tgt, note) in enumerate(pairs):
        y = 116 + i*34
        f.math(356, y, src, 19, anchor='middle')
        f.text(382, y-4, '↔', 16, 'muted')
        f.math(418, y, tgt, 17 if 'mask' in tgt else 19, anchor='middle')
        f.text(454, y-1, note, 12, 'muted', anchor='start')
    # nemotron numbers
    f.panel(546, 64, 334, 256, 'Nemotron 3 Nano 30B-A3B (config)')
    stateicon(f, 600, 122, 4, 8, 16)
    f.brace(600, 122, 600+8*16-2, 122, '$N = 128', side='above', size=13)
    f.brace(600, 122, 600, 122+4*16-2, '$P = 64', side='left', size=13)
    f.lines(744, 136, ['× 64 heads', '= 524,288', 'per layer'], 14, 'purple', 'start', bold=True)
    f.lines(566, 230, ['n_groups = 8: B, C shared by 8 heads (GQA-like)',
                       'chunk_size = 128 · conv kernel 4',
                       'one decay scalar per head per token'], 13, 'ink', 'start')
    # layer strip
    pat = 'MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME'
    f.text(30, 392, '52 layers', 14, 'ink', True, 'start')
    cells = [('', {'M': 'purple', 'E': 'line', '*': 'blue'}[ch]) for ch in pat]
    f.strip(110, 377, cells, w=13.6, h=22, gap=1.6)
    for x, c, t in ((110, 'purple', 'Mamba-2 × 23'), (250, 'line', 'MoE × 23'), (360, 'blue', 'GQA attention × 6')):
        f.rect(x, 414, 12, 12, c, .88, 2)
        f.text(x+18, 425, t, 13, 'ink', anchor='start')
    f.text(870, 425, 'decode cache = fixed SSM state + KV of 6 layers', 13, 'muted', anchor='end')
    f.text(450, 456, 'ΔX: row t of X scaled by its step size Δ. Mamba-1 decays each (channel, state) entry separately: no matmul form.', 13, 'muted')
    f.save()


# ---------------------------------------------------------------- card 5
def mamba3():
    f = Fig('linear-mamba3', 'Mamba-3 changes three things. (a) Trapezoid write: with toy Δ = 0.5, α = 0.8, Mamba-2 (λ = 1) writes 0.5 of x_t only, the trapezoid rule (λ = ½) writes 0.2 of x_(t−1) and 0.25 of x_t. (b) A complex state turns the decay into a rotation: eigenvalues move from the real point α to α·e^(±iθ), equivalent to data-dependent RoPE on B and C. (c) MIMO: rank-R writes (P×R)(R×N) with R = 4 do about 4× the decode FLOPs on the same 64×128 state.', 420)
    f.math(30, 36, r'\c{purple}{S_t} = \c{teal}{α_t}\c{purple}{S_{t-1}} + \c{teal}{β_t}\c{purple}{x_{t-1}}\c{blue}{B_{t-1}}^\top + \c{teal}{γ_t}\c{purple}{x_t}\c{blue}{B_t}^\top', 21)
    f.math(870, 36, r'\c{teal}{β_t} = (1-λ_t)Δ_t\c{teal}{α_t}, \; \c{teal}{γ_t} = λ_tΔ_t', 18, anchor='end')
    # (a) taps
    f.panel(20, 58, 280, 300, '(a) write taps (toy Δ = .5, α = .8)')
    base = 300
    for i, (name, taps, col) in enumerate((('Mamba-2 · λ = 1', (0, .5), 'muted'), ('trapezoid · λ = ½', (.2, .25), 'teal'))):
        x0 = 56 + i*126
        f.line(x0-10, base, x0+90, base, 'line', 1.4)
        for j, (lab, v) in enumerate(zip(('t−1', 't'), taps)):
            h = v*300
            if v:
                f.rect(x0+j*44, base-h, 34, h, col, .8, 3)
            f.text(x0+j*44+17, base-h-6, f'{v:g}', 13, col, True)
            f.text(x0+j*44+17, base+16, lab, 13, 'muted')
        f.text(x0+38, 112 if i == 0 else 112, name, 13, col, True)
    f.text(160, 340, 'λ = 1 recovers Mamba-2', 13, 'muted')
    # (b) rotation
    f.panel(316, 58, 266, 300, '(b) complex state = rotation')
    cx, cy, r = 448, 208, 84
    f.parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{COL["line"]}" stroke-width="1.2"/>')
    f.parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r*.8}" fill="none" stroke="{COL["teal"]}" stroke-width="1" stroke-dasharray="4 3"/>')
    f.line(cx-r-14, cy, cx+r+14, cy, 'line', 1)
    f.line(cx, cy-r-10, cx, cy+r+10, 'line', 1)
    f.circle(cx+r*.8, cy, 5, color='muted', fill=True)
    th = math.radians(40)
    for sgn in (1, -1):
        f.circle(cx+r*.8*math.cos(th), cy-sgn*r*.8*math.sin(th), 5, color='teal', fill=True)
    f.text(cx+r*.8+6, cy+20, 'α (Mamba-2)', 12, 'muted', anchor='start')
    f.math(cx+r*.8*math.cos(th)+8, cy-r*.8*math.sin(th)-8, r'\c{teal}{α e^{iθ}}', 15)
    f.math(cx+r*.8*math.cos(th)+8, cy+r*.8*math.sin(th)+18, r'\c{teal}{α e^{-iθ}}', 15)
    f.text(cx-r-4, cy-r+4, '|z| = 1', 12, 'muted', anchor='start')
    f.lines(449, 316, ['≡ data-dependent RoPE on B, C', 'cumulative angle Σ Δθ; enables parity'], 13, 'teal', 'middle')
    # (c) MIMO
    f.panel(598, 58, 282, 300, '(c) MIMO: rank-R write')
    f.text(620, 106, 'SISO', 14, 'muted', True, 'start')
    f.rect(672, 92, 8, 40, 'purple', .8, 1)
    f.text(690, 117, '×', 15, 'muted')
    f.rect(700, 106, 64, 8, 'blue', .8, 1)
    f.text(772, 117, 'P×1 · 1×N', 12, 'muted', anchor='start')
    f.text(620, 172, 'MIMO', 14, 'teal', True, 'start')
    f.rect(672, 150, 26, 40, 'purple', .8, 1)
    f.text(706, 175, '×', 15, 'muted')
    f.rect(716, 158, 64, 26, 'blue', .8, 1)
    f.text(786, 175, 'P×R · R×N', 12, 'teal', True, 'start')
    f.text(686, 204, 'R = 4', 12, 'teal', True)
    f.text(739, 228, 'state stays P × N (e.g. 64 × 128)', 13, 'purple', True)
    f.text(612, 252, 'decode FLOPs per state byte', 12, 'muted', anchor='start')
    f.bars(612, 260, [('SISO', 1, 'muted'), ('MIMO R=4', 4, 'teal')], width=100, bar=14, gap=6, label_w=74, size=13,
           value_fmt=lambda v: f'≈{v:g}×')
    f.lines(739, 318, ['decode is memory-bound:', 'reported latency ≈ unchanged'], 13, 'muted', 'middle')
    f.text(450, 392, 'Same state bytes as Mamba-2; reported: MIMO with N = 64 matches Mamba-2 perplexity at N = 128.', 14, 'ink')
    f.save()


# ---------------------------------------------------------------- card 6
def chunkwise():
    f = Fig('linear-chunkwise', 'Chunkwise form for N = 16 toy tokens in chunks of C = 4: only the four diagonal 4×4 blocks are computed as small masked attention; everything below them is carried by a state S passed between chunks. Our count per head with d = 128, C = 64, N = 32K: chunkwise about 98K FLOPs per token in 512 sequential steps, recurrent 66K FLOPs per token in 32,768 matrix-vector steps, quadratic form 8.4M FLOPs per token.', 452)
    f.math(30, 36, r'\c{green}{O_{[c]}} = (\c{gold}{Q_{[c]}}\c{blue}{K_{[c]}}^\top \odot M)\c{purple}{V_{[c]}} + \c{gold}{Q_{[c]}}\c{purple}{S_{[c-1]}}^\top, \quad \c{purple}{S_{[c]}} = \c{purple}{S_{[c-1]}} + \c{purple}{V_{[c]}}^\top\c{blue}{K_{[c]}}', 20)
    N, C, cell, mx, my = 16, 4, 15, 64, 92

    def fn(i, j):
        if j > i:
            return None
        if i // C == j // C:
            return ('blue', .8)
        return ('purple', .16)
    f.mask(mx, my, N, cell, fn)
    for b in range(1, N//C):
        f.line(mx, my+b*C*cell-1, mx+N*cell, my+b*C*cell-1, 'ink', 1)
        f.line(mx+b*C*cell-1, my, mx+b*C*cell-1, my+N*cell, 'ink', 1)
    f.brace(mx, my, mx, my+C*cell-2, '$C', side='left', size=14)
    f.brace(mx, my+N*cell+2, mx+N*cell-2, my+N*cell+2, 'N = 16 (toy), C = 4', side='below', size=13)
    # state pass
    sx = mx + N*cell + 26
    for b in range(N//C):
        y = my + b*C*cell + 14
        f.matrix(sx, y, 2, 2, 14, 'purple', [[.7, .5], [.4, .8]])
        f.math(sx+34, y+20, f'\\c{{purple}}{{S_{{[{b}]}}}}', 15)
        if b < N//C - 1:
            f.arrow(sx+13, y+30, sx+13, y+C*cell-2, 'purple', width=1.6)
    f.callout(mx+C*cell*1.5, my+C*cell*1.5, 36, 384, 'intra-chunk: C × C masked attention (matmul)', 'blue', 13)
    f.callout(mx+C*cell*.5, my+C*cell*3.5, 36, 404, 'never formed: read through the passed state', 'purple', 13)
    # costs
    x0 = 420
    f.text(x0, 82, 'Per head, d = 128, C = 64, N = 32K (our count)', 15, 'ink', True, 'start')
    f.text(x0, 106, 'FLOPs per token (log scale)', 13, 'muted', anchor='start')
    rows = [('quadratic (QKᵀ ⊙ M)V', 2*32768*128, 'red'), ('chunkwise', 4*64*128 + 4*128*128, 'green'), ('recurrent', 4*128*128, 'purple')]
    logbars(f, x0, 116, rows, 4, 170, bar=18, gap=8, label_w=170, size=13)
    f.lines(x0+170, 210, ['chunkwise = 4Cd + 2d² + 2d²', '= 32,768 intra + 32,768 read + 32,768 write'], 13, 'muted', 'start')
    f.text(x0, 268, 'Sequential steps (log scale)', 13, 'muted', anchor='start')
    rows = [('quadratic', 1, 'red'), ('chunkwise: N / C', 512, 'green'), ('recurrent: N', 32768, 'purple')]
    f.bars(x0, 278, [(n, math.log10(v)+.3, c) for n, v, c in rows], width=170, bar=18, gap=8, label_w=170, size=13,
           value_fmt=lambda v: f'{10**(v-.3):,.0f}')
    f.text(x0, 376, 'recurrent steps are matrix-vector products: no tensor cores', 13, 'muted', anchor='start')
    f.text(x0, 396, 'real C: 64 (FLA kernels), 128 (Nemotron Mamba-2), 256 (MiniMax)', 13, 'muted', anchor='start')
    f.text(450, 438, 'Decay versions add cumulative-decay factors inside each block; the block structure is unchanged.', 14, 'ink')
    f.save()


def build():
    reassociate()
    fixed_decay()
    gla()
    ssd()
    mamba3()
    chunkwise()
