"""11 · Memory as test-time learning: figures (prefix ttt-)."""
import math
from figkit import Fig

EM = '\u2003'  # SVG collapses ordinary spaces; em spaces survive


# ---------------------------------------------------------------- helpers
def fmt(v):
    if v == 0:
        return '0'
    s = f'{v:g}'
    return s.replace('-', '−')


def num_mat(f, x, y, M, cell=30, color='purple', neg='red', size=13, vmax=None, zero_op=.07):
    """Small heat-map matrix with the value printed in every cell."""
    vmax = vmax or max(abs(v) for row in M for v in row) or 1
    for r, row in enumerate(M):
        for c, v in enumerate(row):
            tone = color if v >= 0 else neg
            op = zero_op if v == 0 else .3 + .6*abs(v)/vmax
            f.rect(x + c*cell, y + r*cell, cell - 3, cell - 3, tone, op, 3)
            f.text(x + c*cell + (cell - 3)/2, y + r*cell + (cell - 3)/2 + size*.36, fmt(v), size,
                   'white' if op > .55 else 'ink')
    return len(M[0])*cell - 3, len(M)*cell - 3


def num_col(f, x, y, vals, cell=30, color='blue', size=13, vmax=1):
    return num_mat(f, x, y, [[v] for v in vals], cell, color, 'red', size, vmax)


def hbars(f, x, y, rows, width, lo, hi, bar=18, gap=8, label_w=200, size=14, value_fmt=None, ticks=()):
    """Horizontal bars drawn from a baseline `lo` (axis does not start at 0)."""
    for i, (name, v, c) in enumerate(rows):
        yy = y + i*(bar + gap)
        f.text(x + label_w - 10, yy + bar*.72, name, size, 'ink', anchor='end')
        w = max(2, width*(v - lo)/(hi - lo))
        f.rect(x + label_w, yy, w, bar, c, .85, 3)
        f.text(x + label_w + w + 7, yy + bar*.72, value_fmt(v) if value_fmt else f'{v:g}', size,
               'muted' if c == 'line' else c, True, 'start')
    H = len(rows)*(bar + gap) - gap
    for t in ticks:
        tx = x + label_w + width*(t - lo)/(hi - lo)
        f.line(tx, y - 4, tx, y + H + 4, 'line', 1, True)
        f.text(tx, y + H + 18, f'{t:g}', 12, 'muted')
    return H


# ---------------------------------------------------------------- card 1
def one_step():
    f = Fig('ttt-one-step', 'Toy, d = 3, beta = 1: S_{t-1} stores e1 to e1 and e2 to e2; the new pair is k = e1, v = e3. '
            'Linear attention takes the gradient of minus <Sk, v>, adds v k^T and reads back [1, 0, 1]: the old value '
            'survives, error 1. DeltaNet takes the gradient of 1/2 ||Sk - v||^2, subtracts (Sk - v) k^T and reads back '
            'exactly [0, 0, 1], error 0. Decay alpha_t before the step acts as weight decay.', 440)
    f.math(450, 36, r'\c{purple}{S_t} = \c{teal}{α_t}\c{purple}{S_{t-1}} − \c{teal}{β_t} \c{red}{∇ℓ_t}(\c{teal}{α_t}\c{purple}{S_{t-1}})'
           + EM*2 + r'\text{read}' + EM + r'\c{green}{o_t} = \c{purple}{S_t}\c{gold}{q_t}', 22, anchor='middle')
    # shared inputs
    f.panel(24, 62, 170, 340, 'inputs (toy)')
    S0 = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]
    f.math(109, 108, r'\c{purple}{S_{t-1}} \text{ (3×3)}', 16, anchor='middle')
    num_mat(f, 65, 118, S0)
    f.text(109, 228, 'stores e₁→e₁, e₂→e₂', 13, 'muted')
    f.math(68, 268, r'\c{blue}{k_t}', 16, anchor='middle')
    f.math(150, 268, r'\c{purple}{v_t}', 16, anchor='middle')
    num_col(f, 54, 278, [1, 0, 0], color='blue')
    num_col(f, 136, 278, [0, 0, 1], color='purple')
    f.text(109, 392, 'rewrite key e₁ → e₃', 13, 'ink')
    rows = [
        ('Linear attention', r'ℓ = −⟨\c{purple}{S}\c{blue}{k}, \c{purple}{v}⟩', r'\c{red}{∇ℓ} = −\c{purple}{v}\c{blue}{k}^{\top}',
         [[0, 0, 0], [0, 0, 0], [-1, 0, 0]], [[1, 0, 0], [0, 1, 0], [1, 0, 0]], [1, 0, 1], 'red', 'error 1.0', 'old value survives'),
        ('DeltaNet', r'ℓ = ½‖\c{purple}{S}\c{blue}{k} − \c{purple}{v}‖^{2}', r'\c{red}{∇ℓ} = (\c{purple}{S}\c{blue}{k} − \c{purple}{v})\c{blue}{k}^{\top}',
         [[1, 0, 0], [0, 0, 0], [-1, 0, 0]], [[0, 0, 0], [0, 1, 0], [1, 0, 0]], [0, 0, 1], 'green', 'error 0', 'old value replaced'),
    ]
    for i, (name, loss, grad, G, S1, read, tone, err, note) in enumerate(rows):
        y0 = 70 + i*170
        if i:
            f.hdivider(y0 - 10, 210, 880)
        f.text(214, y0 + 16, name, 17, 'ink', True, 'start')
        f.math(214 + (150 if i == 0 else 92), y0 + 16, loss, 17)
        # gradient
        f.math(275, y0 + 46, grad, 15, anchor='middle')
        num_mat(f, 232, y0 + 56, G, color='red', neg='red')
        f.arrow(330, y0 + 100, 372, y0 + 100, 'muted')
        f.text(351, y0 + 90, '−β·', 13, 'teal', True)
        # new state
        f.math(424, y0 + 46, r'\c{purple}{S_t}', 16, anchor='middle')
        num_mat(f, 382, y0 + 56, S1)
        f.arrow(480, y0 + 100, 522, y0 + 100, 'muted')
        f.math(500, y0 + 90, r'\c{blue}{k_t}', 13, anchor='middle')
        # read-back vs target
        f.math(560, y0 + 46, r'\c{purple}{S_t}\c{blue}{k_t}', 15, anchor='middle')
        num_col(f, 545, y0 + 56, read, color='green')
        f.text(610, y0 + 104, 'vs', 14, 'muted')
        f.math(660, y0 + 46, r'\c{purple}{v_t}', 15, anchor='middle')
        num_col(f, 645, y0 + 56, [0, 0, 1], color='purple')
        f.text(700, y0 + 92, err, 20, tone, True, 'start')
        f.text(700, y0 + 116, note, 14, tone, anchor='start')
    f.text(24, 426, 'Decay α < 1 before the step = weight decay (Mamba-2, GDN).', 14, 'teal', True, 'start')
    f.text(880, 424, 'Other optimizers → TTT, Titans, ATLAS, Mesa', 14, 'muted', anchor='end')
    f.save()


# ---------------------------------------------------------------- card 2
def minibatch():
    f = Fig('ttt-minibatch', 'Mini-batch TTT over a toy 32-token sequence. With b = 1 every gradient uses the latest W: 32 '
            'sequential steps (online GD). With b = 8 gradients inside a batch share the batch-start W: 4 sequential steps. '
            'With b = T one step, which with linear f, eta = 1/2 and W0 = 0 equals linear attention. The paper uses b = 16. '
            'Reported 125M ablation (Table 1): perplexity 15.91 for linear attention falls to 11.09; mini-batch TTT is the '
            'largest single gain, -1.70.', 450)
    f.math(450, 34, r'\c{purple}{W_t} = \c{purple}{W_{t-1}} − \c{teal}{η_t}\, \c{red}{∇ℓ}(\c{purple}{W_{t′}};\, x_t),'
           + EM*2 + r't′ = t − (t \text{ mod } \c{teal}{b})', 21, anchor='middle')
    T, cw, x0 = 32, 14, 168
    cfg = [(1, 'b = 1', '32 sequential steps', 'online GD ≈ delta rule', 'red'),
           (8, 'b = 8', '4 sequential steps', 'TTT (paper: b = 16)', 'gold'),
           (32, 'b = T', '1 step', '= linear attention (Thm 1)', 'green')]
    for r, (b, lab, steps, note, tone) in enumerate(cfg):
        y = 70 + r*58
        f.text(30, y + 24, lab, 17, 'ink', True, 'start')
        for t in range(T):
            batch = t // b
            f.rect(x0 + t*cw, y + 14, cw - 2, 16, 'blue', .35 if batch % 2 else .75, 2)
        # W states at batch starts and the sequential chain
        nb = T // b
        pts = [x0 + i*b*cw + 2 for i in range(nb)]
        for i, px in enumerate(pts):
            f.rect(px - 2, y, 9, 9, 'purple', .95, 1)
            if i:
                f.line(pts[i-1] + 7, y + 4.5, px - 2, y + 4.5, 'purple', 1.4)
        if b > 1:
            for i in range(nb):
                f.brace(x0 + i*b*cw, y + 36, x0 + (i+1)*b*cw - 3, y + 36, '', 'muted', 'below', 12, 2)
        f.text(x0 + T*cw + 22, y + 16, steps, 15, tone, True, 'start')
        f.text(x0 + T*cw + 22, y + 34, note, 14, 'ink', anchor='start')
    f.text(x0, 248, '■ W at batch start (purple chain = sequential dependency) · ▬ tokens whose gradients run as one matmul', 13, 'muted', anchor='start')
    f.hdivider(262)
    f.text(30, 288, 'Reported ablation, 125M on the Pile (T = 2048), perplexity ↓', 15, 'ink', True, 'start')
    f.text(870, 288, 'TTT-Linear, 1.3B: 32 heads × 64 dims', 13, 'muted', anchor='end')
    rows = [('Linear attention', 15.91, 'line'), ('+ improvements', 15.23, 'line'),
            ('+ LN and residual in f', 14.05, 'purple'), ('+ mini-batch TTT, b = 16', 12.35, 'green'),
            ('+ learnable η', 11.99, 'purple'), ('+ Mamba backbone', 11.09, 'purple')]
    hbars(f, 30, 302, rows, 430, 10, 16.2, bar=17, gap=7, label_w=220, size=14,
          value_fmt=lambda v: f'{v:.2f}' + ('   −1.70, largest gain' if v == 12.35 else ''))
    f.text(870, 446, 'axis starts at 10', 12, 'muted', anchor='end')
    f.save()


# ---------------------------------------------------------------- card 3
def titans():
    f = Fig('ttt-titans', 'Titans memory update. Toy trace over 16 tokens: momentary surprise (gradient norm) spikes at '
            'token 6; with momentum eta = 0.7 the write keeps going for several tokens. Three placements: MAC prepends '
            'memory read-outs as tokens to segment attention, MAG gates sliding-window attention with the memory branch, '
            'MAL stacks the memory layer before SWA. Reported ablation (Table 5): long-context accuracy MAC 97.95, MAG '
            '96.70, MAL 96.91, memory alone 92.68, linear memory 85.34, without momentum 87.12, without weight decay 85.60.', 460)
    f.math(450, 34, r'\c{purple}{M_t} = (1 − \c{teal}{α_t})\c{purple}{M_{t-1}} + m_t,' + EM*2 + r'm_t = \c{teal}{η_t} m_{t-1} − \c{teal}{θ_t}\,'
           r'\c{red}{∇}‖\c{purple}{M_{t-1}}(\c{blue}{k_t}) − \c{purple}{v_t}‖^{2}', 21, anchor='middle')
    # surprise trace
    f.panel(24, 56, 470, 196, 'surprise → write  (toy, η = 0.7, α = 0)')
    g = [.12, .1, .14, .1, .12, 1.0, .3, .12, .1, .14, .1, .12, .1, .14, .1, .12]
    m, ms = 0, []
    for v in g:
        m = .7*m + v
        ms.append(m)
    bx, by, bw, bh, vmax = 52, 228, 26, 140, 1.25
    for i, v in enumerate(g):
        f.rect(bx + i*bw + 4, by - bh*v/vmax, 12, bh*v/vmax, 'red', .8, 2)
    pts = [(bx + i*bw + 10, by - bh*v/vmax) for i, v in enumerate(ms)]
    f.path(pts, 'purple', width=2.2)
    for x, y in pts:
        f.circle(x, y, 2.8, color='purple', fill=True)
    f.line(bx, by, bx + 16*bw, by, 'line', 1.2)
    f.text(bx + 5*bw + 10, by + 16, 'token 6', 12, 'red')
    f.text(bx + 16*bw + 2, pts[-1][1] + 4, '|m|', 13, 'purple', True, 'start')
    f.text(bx + 5*bw + 2, by - bh*g[5]/vmax + 14, 'gradient ‖∇ℓ‖', 13, 'red', True, 'end')
    f.text(bx + 5*bw + 2, by - bh*g[5]/vmax + 30, '= momentary surprise', 13, 'red', anchor='end')
    f.text(bx + 9*bw, pts[7][1] - 10, 'momentum m keeps writing', 13, 'purple', True, 'start')
    f.text(bx + 9*bw, pts[7][1] + 6, 'for ~5 more tokens', 13, 'purple', anchor='start')
    # placements
    f.panel(508, 56, 368, 196, 'where the memory sits')
    cell = 9
    # MAC: persistent | memory read | causal within segment
    x, y = 524, 104
    f.text(x + 45, y - 10, 'MAC', 14, 'ink', True)
    for i in range(7):
        for j in range(10):
            if j < 2:
                c = ('teal', .8)
            elif j < 4:
                c = ('purple', .8)
            elif j - 4 <= i:
                c = ('blue', .8)
            else:
                c = None
            if c:
                f.rect(x + j*cell, y + i*cell, cell - 1.5, cell - 1.5, c[0], c[1])
            else:
                f.rect(x + j*cell, y + i*cell, cell - 1.5, cell - 1.5, 'paper', 1)
    f.text(x + 45, y + 96, 'memory read = tokens', 12, 'muted')
    f.text(x + 45, y + 111, 'full attn per segment', 12, 'muted')
    # MAG: SWA band + gate with memory branch
    x = 648
    f.text(x + 45, y - 10, 'MAG', 14, 'ink', True)
    for i in range(8):
        for j in range(8):
            ok = 0 <= i - j < 3
            f.rect(x + j*cell, y + i*cell, cell - 1.5, cell - 1.5, 'blue' if ok else 'paper', .8 if ok else 1)
    f.text(x + 80, y + 40, '⊗', 18, 'teal', True)
    f.rect(x + 92, y, 9, 70, 'purple', .8, 2)
    f.text(x + 45, y + 96, 'SWA ⊗ memory', 12, 'muted')
    f.text(x + 45, y + 111, 'parallel branches', 12, 'muted')
    # MAL: stacked layers
    x = 780
    f.text(x + 40, y - 10, 'MAL', 14, 'ink', True)
    f.box(x + 6, y + 2, 70, 26, 'SWA', 'blue', True, 13, .85, 4)
    f.box(x + 6, y + 44, 70, 26, 'memory', 'purple', True, 13, .85, 4)
    f.arrow(x + 41, y + 44, x + 41, y + 30, 'muted')
    f.text(x + 41, y + 96, 'memory layer', 12, 'muted')
    f.text(x + 41, y + 111, 'then SWA', 12, 'muted')
    for k, (lab, tone) in enumerate((('persistent', 'teal'), ('memory', 'purple'), ('attention', 'blue'))):
        f.rect(560 + k*105, 233, 10, 10, tone, .8, 2)
        f.text(575 + k*105, 242, lab, 12, 'muted', anchor='start')
    # ablation
    f.text(30, 282, 'Reported ablation (Table 5): long-context accuracy ↑ · perplexity ↓', 15, 'ink', True, 'start')
    rows = [('+ attention · MAC', 97.95, 'green'), ('+ attention · MAL', 96.91, 'green'), ('+ attention · MAG', 96.70, 'green'),
            ('deep memory alone (LMM)', 92.68, 'purple'), ('w/o momentum', 87.12, 'red'), ('w/o weight decay', 85.60, 'red'),
            ('linear memory (1 layer)', 85.34, 'red')]
    ppl = {97.95: 26.67, 96.91: 25.91, 96.70: 25.70, 92.68: 27.01, 87.12: 28.98, 85.60: 29.04, 85.34: 28.49}
    hbars(f, 30, 296, rows, 360, 80, 100, bar=16, gap=6, label_w=230, size=14,
          value_fmt=lambda v: f'{v:.2f}  · ppl {ppl[v]:.2f}')
    f.text(870, 452, 'accuracy axis starts at 80', 12, 'muted', anchor='end')
    f.save()


# ---------------------------------------------------------------- card 4
def window_exact():
    f = Fig('ttt-window-exact', 'Which past pairs the memory is fit to. Delta rule: only token t, one gradient step. ATLAS: '
            'the last c tokens with gates gamma, Muon (Newton-Schulz) steps. Mesa: every past token with discount zeta, '
            'solved exactly by conjugate gradient. Toy with correlated keys k1 = [1, 0], k2 = [0.6, 0.8], values +1 and -1: '
            'reading at k1 gives 0.40 (linear attention), 0.04 (DeltaNet), 1.00 (Mesa). Mesa costs about k times GLA FLOPs '
            'for k CG steps; a stopping rule eps = 1e-4 matches 30 fixed steps with about 9 on average.', 452)
    f.math(450, 34, r'ℓ_t(M) = \sum_{i=t-c+1}^{t} \c{teal}{γ_i}\, ‖M(φ(\c{blue}{k_i})) − \c{purple}{v_i}‖^{2}'
           + EM*2 + r'\c{teal}{c} = 1: \text{delta} · \c{teal}{c} = t: \text{least squares}', 21, anchor='middle')
    f.panel(24, 56, 452, 230, 'weight of each past pair in ℓ (12 toy tokens)')
    n, cw, x0 = 12, 26, 150
    rows = [('DeltaNet', lambda i: .9 if i == n-1 else 0, 'red', 'c = 1 · one GD step'),
            ('ATLAS', lambda i: [0, .45, .7, .95][i-(n-4)] if i >= n-4 else 0, 'red', 'c = 4 · gates γ · Muon (Newton–Schulz)'),
            ('Mesa', lambda i: .9*0.86**(n-1-i), 'red', 'c = t · discount ζ · exact solve (CG)')]
    for r, (name, w, tone, note) in enumerate(rows):
        y = 96 + r*62
        f.text(40, y + 17, name, 16, 'ink', True, 'start')
        for i in range(n):
            v = w(i)
            if v > 0:
                f.rect(x0 + i*cw, y, cw - 3, 24, tone, max(.12, v), 3)
            else:
                f.rect(x0 + i*cw, y, cw - 3, 24, 'paper', 1, 3, stroke='line')
        f.text(x0, y + 44, note, 13, 'muted', anchor='start')
    f.text(x0 + (n-1)*cw + 11, 88, 't', 13, 'ink', True)
    # toy contrast
    f.panel(490, 56, 386, 230, 'correlated keys (toy, Λ = 0)')
    # key geometry
    ox, oy, L = 512, 262, 78
    f.line(ox - 4, oy, ox + L + 10, oy, 'line', 1)
    f.line(ox, oy + 4, ox, oy - L - 10, 'line', 1)
    f.arrow(ox, oy, ox + L, oy, 'blue', width=2.2)
    f.arrow(ox, oy, ox + .6*L, oy - .8*L, 'blue', width=2.2)
    f.text(ox + L - 2, oy - 8, 'k₁ → +1', 13, 'blue', True, 'end')
    f.text(ox + .6*L + 4, oy - .8*L - 4, 'k₂ → −1', 13, 'blue', True, 'start')
    f.text(ox + 4, oy - 96 - 14, 'k₁ᵀk₂ = 0.6', 13, 'muted', anchor='start')
    f.text(ox + 4, 104, 'k₁ = [1, 0], k₂ = [0.6, 0.8]', 12, 'muted', anchor='start')
    # read-back table
    cols = [(752, 'read at k₁', 1), (830, 'read at k₂', -1)]
    for cx, lab, tgt in cols:
        f.text(cx, 122, lab, 13, 'ink', True)
        f.text(cx, 138, f'target {fmt(tgt)}', 12, 'green')
    reads = [('linear attn', .40, -.40), ('DeltaNet', .04, -1.0), ('Mesa (exact)', 1.0, -1.0)]
    for r, (name, a, b) in enumerate(reads):
        y = 146 + r*36
        f.text(712, y + 21, name, 14, 'ink', anchor='end')
        for (cx, _, tgt), val in zip(cols, (a, b)):
            ok = abs(val - tgt) < .05
            f.box(cx - 32, y + 2, 64, 28, f'{val:+.2f}'.replace('-', '−'), 'green' if ok else 'red', True, 14, .85, 5)
    f.text(790, 274, 'DeltaNet: 2nd write clobbers k₁', 12, 'red')
    # cost readouts
    f.hdivider(300)
    f.text(30, 326, 'Cost readouts (reported)', 15, 'ink', True, 'start')
    hbars(f, 30, 340, [('Mesa · fixed CG steps', 30, 'red'), ('Mesa · stop at ε = 10⁻⁴', 9, 'green')], 200, 0, 30,
          bar=18, gap=8, label_w=190, size=14, value_fmt=lambda v: f'{v} steps' + (' avg, same ppl' if v == 9 else ''))
    f.text(30, 412, 'k CG steps ≈ k × GLA FLOPs; k = 0 is GLA.', 13, 'muted', anchor='start')
    f.math(30, 434, r'\text{extra state } H \in \R^{\c{blue}{d_k} × \c{blue}{d_k}}\text{: < 1 % of footprint}', 14, 'muted')
    f.text(500, 344, 'ATLAS capacity bound (Prop. 1–2):', 14, 'ink', True, 'start')
    f.math(500, 370, r'O(\c{blue}{d_k}) \text{ pairs} \to O(\c{blue}{d_k}^{p}) \text{ with degree-}p\; φ', 16)
    f.lines(500, 396, ['Newton–Schulz steps: test-time compute knob.',
                       'BABILong at 10M tokens: > 80 % acc; Titans drops.'], 14, 'ink', gap=1.45)
    f.save()


# ---------------------------------------------------------------- card 5
def large_chunk():
    f = Fig('ttt-large-chunk', 'Arithmetic intensity of a fast-weight matmul r = b / (1 + 2b/h) FLOP per byte versus chunk '
            'size b, for fast-weight width h = 64, 256 and 1024. TTT with b = 16, h = 64 reaches about 11, far below the '
            'H100 ridge of about 290; only large chunks with wide fast weights cross it. LaCT mask: tokens read W updated on '
            'earlier chunks and use sliding-window attention locally. Reported: GPU utilization below 5 % for small-batch '
            'TTT versus up to 70 % on A100 for LaCT; state per block 2304d versus 384d (GLA) and 128d (DeltaNet).', 452)
    f.math(450, 34, r'r = 2h^{2}\c{teal}{b} / (2h^{2} + 4h\c{teal}{b}) \le \min(h/2,\; \c{teal}{b}) \text{ FLOP/B}'
           + EM*2 + r'\text{one update per } \c{teal}{b} = 2\text{K–1M tokens}', 21, anchor='middle')
    # intensity plot
    f.panel(24, 56, 452, 246, 'intensity of one fast-weight matmul (Eq. 3)')
    px, py, pw, ph = 70, 96, 340, 168
    lx = lambda b: px + pw*math.log10(b)/6
    ly = lambda r: py + ph - ph*math.log10(r)/3
    for e in range(7):
        x = lx(10**e)
        f.line(x, py, x, py + ph, 'line', .8)
        f.text(x, py + ph + 16, ['1', '10', '100', '1K', '10K', '100K', '1M'][e], 12, 'muted')
    for e in range(4):
        y = ly(10**e)
        f.line(px, y, px + pw, y, 'line', .8)
        f.text(px - 6, y + 4, ['1', '10', '100', '1K'][e], 12, 'muted', anchor='end')
    f.text(px + pw/2, py + ph + 32, 'chunk size b (tokens, log)', 13, 'muted')
    ridge = ly(290)
    f.line(px, ridge, px + pw, ridge, 'red', 1.6, True)
    f.text(px + 4, ridge - 6, 'H100 ridge ≈ 290', 12, 'red', True, 'start')
    for h, tone in ((64, 'muted'), (256, 'purple'), (1024, 'green')):
        pts = []
        for k in range(61):
            b = 10**(k/10)
            pts.append((round(lx(b), 1), round(ly(b/(1 + 2*b/h)), 1)))
        f.path(pts, tone, width=2)
        f.text(px + pw + 4, pts[-1][1] + 4, f'h={h}', 12, tone, True, 'start')
    r16 = 16/(1 + 32/64)
    f.circle(lx(16), ly(r16), 4.5, color='red', fill=True)
    f.text(lx(16) + 8, ly(r16) + 18, 'TTT b=16: r≈11', 12, 'red', True, 'start')
    f.line(lx(2048), py, lx(2048), py + ph, 'green', 1.2, True)
    f.text(lx(2048) + 4, py + 12, 'LaCT LM b=2K', 12, 'green', True, 'start')
    # mask
    f.panel(490, 56, 386, 246, 'LaCT language model: who reads what')
    n, cell, mx, my, C = 12, 13, 520, 96, 4
    for i in range(n):
        for j in range(n):
            if j > i:
                tone = None
            elif i - j < C:
                tone = ('blue', .85)
            elif j // C < i // C:
                tone = ('purple', .55)
            else:
                tone = None
            f.rect(mx + j*cell, my + i*cell, cell - 1.5, cell - 1.5, tone[0] if tone else 'paper', tone[1] if tone else 1)
    for c in range(1, n // C):
        f.line(mx + c*C*cell - 1, my, mx + c*C*cell - 1, my + n*cell, 'ink', .8)
        f.line(mx, my + c*C*cell - 1, mx + n*cell, my + c*C*cell - 1, 'ink', .8)
    f.text(mx + n*cell/2, my + n*cell + 18, 'toy: 12 tokens, chunk 4', 12, 'muted')
    f.rect(700, 103, 10, 10, 'purple', .55, 2)
    f.lines(716, 112, ['via W: earlier chunks', '(apply, then update)'], 13, 'ink', gap=1.4)
    f.rect(700, 141, 10, 10, 'blue', .85, 2)
    f.text(716, 150, 'SWA, window = chunk', 13, 'ink', anchor='start')
    f.lines(700, 190, ['current chunk is invisible', 'to W → local attention', 'is mandatory'], 13, 'red', gap=1.4)
    # reported numbers
    f.hdivider(316)
    f.text(30, 342, 'FLOPs utilization (reported)', 14, 'ink', True, 'start')
    f.bars(30, 354, [('prior TTT', 5, 'red'), ('LaCT · A100', 70, 'green')], width=180, bar=18, gap=8, label_w=110, size=13,
           value_fmt=lambda v: ('< ' if v == 5 else 'up to ') + f'{v} %')
    f.text(30, 430, '3B, 32K, A100: train tok/s 4.3K (LaCT-Muon) vs 4.1K full attention', 13, 'muted', anchor='start')
    f.text(470, 342, 'Fast-weight state per block (3B)', 14, 'ink', True, 'start')
    f.bars(470, 354, [('LaCT', 2304, 'green'), ('GLA + SWA', 384, 'purple'), ('DeltaNet + SWA', 128, 'purple')],
           width=230, bar=16, gap=7, label_w=130, size=13, value_fmt=lambda v: f'{v}·d')
    f.text(470, 440, 'largest LaCT state: 12d² per block ≈ 40 % of weights', 13, 'muted', anchor='start')
    f.save()


# ---------------------------------------------------------------- card 6
def fenwick_level(t, s):
    """Level of history token s (< t) in the Fenwick partition of prefix [0, t); 0 = the token t itself."""
    if s == t:
        return 0
    b = t
    while b > 0:
        low = b & -b
        if b - low <= s < b:
            return low.bit_length()
        b -= low
    return None


def fenwick():
    f = Fig('ttt-fenwick', 'Log-linear attention. A 16 by 16 causal mask coloured by Fenwick level: each query row t splits '
            'its prefix into power-of-two buckets, one state per bucket. For t = 13 the buckets are tokens 0 to 7 (8 tokens, '
            'level 4), 8 to 11 (4, level 3), 12 (1, level 1) and token 13 itself (level 0): 4 states. At t = 2^20 a KV cache '
            'holds 1,048,576 records, log-linear at most 22 states, linear attention 1 state.', 420)
    f.math(450, 34, r'\c{green}{o_t} = \sum_{ℓ} \c{teal}{λ_t^{(ℓ)}}\, \c{purple}{S_t^{(ℓ)}} \c{gold}{q_t},' + EM*3 +
           r'\c{purple}{S_t^{(ℓ)}} = \sum_{s \in B_t^{(ℓ)}} \c{purple}{v_s} \c{blue}{k_s}^{\top}', 21, anchor='middle')
    n, cell, mx, my = 16, 15, 64, 80
    op = {0: .95, 1: .75, 2: .58, 3: .42, 4: .26}
    for t in range(n):
        for s in range(n):
            lv = fenwick_level(t, s) if s <= t else None
            if lv is None:
                f.rect(mx + s*cell, my + t*cell, cell - 1.5, cell - 1.5, 'paper', 1)
            else:
                f.rect(mx + s*cell, my + t*cell, cell - 1.5, cell - 1.5, 'purple', op[lv])
    f.parts.append(f'<rect x="{mx-2}" y="{my+13*cell-2}" width="{14*cell+2}" height="{cell+2}" fill="none" stroke="#d09124" stroke-width="2"/>')
    f.text(mx - 8, my + 13*cell + 11, 't=13', 12, 'gold', True, 'end')
    f.text(mx + n*cell/2, my - 10, 'history s →', 12, 'muted')
    f.text(mx + n*cell/2, my + n*cell + 20, 'shade = level ℓ (each weighted by λ)', 12, 'muted')
    f.text(mx + n*cell/2, my + n*cell + 36, 'rows = query t (16 toy tokens)', 12, 'muted')
    # the partition for t = 13
    x0, y0, cw = 340, 92, 30
    f.text(x0, 76, 'query t = 13 reads 4 states', 15, 'ink', True, 'start')
    groups = [(0, 8, 4), (8, 12, 3), (12, 13, 1), (13, 14, 0)]
    for a, b, lv in groups:
        for s in range(a, b):
            f.rect(x0 + s*cw, y0, cw - 3, 26, 'purple', op[lv], 3)
            f.text(x0 + s*cw + (cw - 3)/2, y0 + 18, str(s), 12, 'white' if op[lv] > .55 else 'ink')
        lab = {4: 'S⁽⁴⁾ · 8 tokens', 3: 'S⁽³⁾ · 4', 1: 'S⁽¹⁾', 0: 'S⁽⁰⁾'}[lv]
        f.brace(x0 + a*cw, y0 + 26, x0 + b*cw - 3, y0 + 26, lab, 'purple', 'below', 13, 6)
    f.text(x0, y0 + 76, '13 = 1101₂: one bucket per set bit, plus the token itself.', 13, 'muted', anchor='start')
    f.text(x0, y0 + 96, 'Each new token: merge buckets up to lssb(t), promote one level.', 13, 'muted', anchor='start')
    # memory readout
    f.hdivider(212, 330, 880)
    f.text(x0, 238, 'States held at t = 2²⁰ ≈ 1M (per head, per layer)', 15, 'ink', True, 'start')
    rows = [('KV cache', 1048576, 'blue'), ('log-linear', 22, 'purple'), ('linear attention', 1, 'purple')]
    for i, (name, v, tone) in enumerate(rows):
        y = 254 + i*30
        f.text(x0 + 120, y + 15, name, 14, 'ink', anchor='end')
        w = max(3, 250*math.log10(v + 1)/math.log10(1048577))
        f.rect(x0 + 130, y, w, 20, tone, .85, 3)
        f.text(x0 + 138 + w, y + 15, f'{v:,}' + (' records' if v > 100 else ' states max (⌈log₂t+1⌉+1)' if v == 22 else ' state'), 13, tone, True, 'start')
    f.text(x0 + 130, 352, 'log scale', 12, 'muted', anchor='start')
    f.hdivider(366, 330, 880)
    f.lines(x0, 388, ['Reported, 21 layers, 50B tokens, 16K: Wiki ppl Mamba-2 22.44 → 22.11,',
                      'GDN 21.73 → 21.45; λ adds < 3 % / < 0.4 % parameters.'], 13, 'ink')
    f.save()


# ---------------------------------------------------------------- card 7
def recall_gap():
    f = Fig('ttt-recall-gap', 'Left: reported in-context recall average (MesaNet Table 3, 1B models, 50B tokens, 2K context): '
            'Mamba2 37.67, GLA 39.64, Gated DeltaNet 39.54, DeltaNet 40.46, Mesa 41.79, SWA-1024 40.84, Transformer 52.27. '
            'Right: reported pass-key retrieval (TTT-E2E Table 2, 3B models): full attention stays at 1.00 to 0.99 from 8K to '
            '128K, while TTT-E2E falls 1.00, 0.46, 0.24, 0.13, 0.06, tracking 8K/N, the share of needles inside its 8K window.', 420)
    f.math(450, 34, r'\text{exactly recallable pairs} \le O(\c{blue}{d_k}) \text{ per } \c{purple}{d_v × d_k} \text{ state}'
           + EM*2 + r'\text{vs.}' + EM*2 + r'N \text{ for a KV cache}', 21, anchor='middle')
    f.panel(24, 56, 420, 340, 'in-context recall, 1B / 50B tokens (reported)')
    rows = [('Mamba2 · Hebbian+decay', 37.67, 'purple'), ('GLA', 39.64, 'purple'), ('Gated DeltaNet', 39.54, 'purple'),
            ('DeltaNet · 1 GD step', 40.46, 'purple'), ('Mesa · exact solve', 41.79, 'purple'),
            ('SWA-1024', 40.84, 'blue'), ('Transformer · full KV', 52.27, 'green')]
    hbars(f, 30, 98, rows, 170, 30, 55, bar=22, gap=12, label_w=200, size=14, ticks=(30, 40, 50))
    f.text(234, 360, 'Hebbian → exact solve: +4.1 · full KV: +10.5 more', 13, 'ink', True)
    f.text(234, 380, 'axis starts at 30', 12, 'muted')
    # NIAH lines
    f.panel(458, 56, 418, 340, 'pass-key retrieval vs context, 3B (reported)')
    px, py, pw, ph = 512, 112, 320, 200
    ctx = [8, 16, 32, 64, 128]
    lx = lambda i: px + pw*i/4
    ly = lambda v: py + ph - ph*v
    for i, c in enumerate(ctx):
        f.line(lx(i), py, lx(i), py + ph, 'line', .8)
        f.text(lx(i), py + ph + 18, f'{c}K', 13, 'muted')
    for v in (0, .5, 1):
        f.line(px, ly(v), px + pw, ly(v), 'line', .8)
        f.text(px - 8, ly(v) + 4, f'{v:g}', 12, 'muted', anchor='end')
    series = [('full attention', [1.00, 1.00, 1.00, 1.00, .99], 'green'),
              ('TTT-E2E', [1.00, .46, .24, .13, .06], 'red'),
              ('GDN + SWA', [1.00, .50, .26, .13, .07], 'purple')]
    guide = [(lx(i), round(ly(min(1, 8/c)), 1)) for i, c in enumerate(ctx)]
    f.path(guide, 'muted', width=1.2, dash=True)
    for name, vals, tone in series:
        pts = [(lx(i), ly(v)) for i, v in enumerate(vals)]
        f.path(pts, tone, width=2.4)
        for x, y in pts:
            f.circle(x, y, 3.5, color=tone, fill=True)
    f.text(lx(4) - 4, ly(.99) + 20, 'full attention: 0.99 at 128K', 13, 'green', True, 'end')
    f.text(lx(1) + 10, ly(.46) - 8, 'TTT-E2E', 13, 'red', True, 'start')
    f.text(lx(1) + 10, ly(.5) - 26, 'GDN + SWA', 13, 'purple', True, 'start')
    f.text(lx(4) - 4, ly(.06) - 62, 'dashed: 8K / N', 12, 'muted', anchor='end')
    f.text(lx(4) - 4, ly(.06) - 46, '(needle inside the window)', 12, 'muted', anchor='end')
    f.text(px + pw/2, py + ph + 40, 'context length · every model except full attention has an 8K window', 12, 'muted')
    f.save()


def build():
    one_step()
    minibatch()
    titans()
    window_exact()
    large_chunk()
    fenwick()
    recall_gap()
