"""10 · The delta rule: read, compare, overwrite. Toy: order 7314, Tuesday = 2 -> Friday = 5."""
import math
from figkit import Fig, COL


# ---------------------------------------------------------------- helpers
def cells(f, x, y, vals, cw=50, ch=34, color='purple', vmax=5, size=16, gap=3, tones=None, outline=None, fmt=None):
    """A 1-row state (or vector) with printed numbers; opacity follows |v|/vmax."""
    for c, v in enumerate(vals):
        tone = tones[c] if tones else color
        op = .08 + .5*min(1, abs(v)/vmax)
        f.rect(x+c*cw, y, cw-gap, ch, tone, op, 4)
        if outline and c in outline:
            f.parts.append(f'<rect x="{x+c*cw-1.5}" y="{y-1.5}" width="{cw-gap+3}" height="{ch+3}" rx="5" fill="none" stroke="{COL[outline[c]]}" stroke-width="2"/>')
        f.text(x+c*cw+(cw-gap)/2, y+ch/2+size*.36, (fmt or '{:g}').format(v), size, 'ink', True)
    return len(vals)*cw - gap


def pill_num(f, x, y, label, value, color, w=96, h=34):
    f.box(x, y, w, h, '', color, False, rx=17)
    f.text(x+w/2, y+h/2+5, f'{label}{value}', 15, color, True)


def keyheads(f, x, y, cw=50, size=12):
    f.text(x+(cw-3)/2, y, '7314', size, 'blue', True)
    f.text(x+cw+(cw-3)/2, y, '2208', size, 'muted', True)


def heat(f, x, y, M, cell=14, size=None):
    """Signed heat map: teal for positive entries, red for negative ones."""
    vals = [[(min(1, abs(v)**.6) if abs(v) > .02 else .03, 'teal' if v >= 0 else 'red') for v in row] for row in M]
    return f.matrix(x, y, len(M), len(M[0]), cell, 'teal', vals, gap=1.5)


def matmul(A, B):
    return [[sum(A[i][k]*B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def eye(d, s=1.0):
    return [[s if i == j else 0 for j in range(d)] for i in range(d)]


def house(k, beta):
    n = math.sqrt(sum(x*x for x in k))
    k = [x/n for x in k]
    return [[(1 if i == j else 0) - beta*k[i]*k[j] for j in range(len(k))] for i in range(len(k))]


# ---------------------------------------------------------------- card 1
def deltanet():
    f = Fig('delta-deltanet', 'Toy delta update for order 7314 with d_v = 1, d_k = 2, k_t = e₁ and v_t = 5 (Friday). The state [2, 4] predicts 2 (Tuesday) at k_t; the residual is 5 − 2 = 3; with β = 1 the write [3, 0] gives [5, 4], so 7314 now reads Friday and order 2208 keeps 4. Additive linear attention would give [7, 4]. β = 0, 0.5, 1 give 2, 3.5, 5. A query halfway between both keys moves from 4.24 to 6.36.', 468)
    f.math(30, 36, r'\c{purple}{S_t} = \c{purple}{S_{t-1}} + \c{teal}{β_t}(\c{purple}{v_t} - \c{purple}{S_{t-1}}\c{blue}{k_t})\c{blue}{k_t}^\top = \c{purple}{S_{t-1}}(I - \c{teal}{β_t}\c{blue}{k_t}\c{blue}{k_t}^\top) + \c{teal}{β_t}\c{purple}{v_t}\c{blue}{k_t}^\top', 20)
    f.text(870, 64, 'toy: 1 value channel, 2 key channels, k = e₁, v = 5 (Friday)', 13, 'muted', anchor='end')
    y = 104
    f.text(30, y-20, 'key direction →', 12, 'muted', anchor='start')
    keyheads(f, 130, y-8)
    f.text(30, y+22, 'β = 1', 15, 'teal', True, 'start')
    cells(f, 130, y, [2, 4])
    f.math(178, y+56, r'\c{purple}{S_{t-1}}', 15, anchor='middle')
    f.text(154, y+74, 'Tue', 12, 'muted')
    f.arrow(234, y+17, 280, y+17, 'blue', width=1.8)
    f.text(257, y+8, 'read', 12, 'blue')
    pill_num(f, 284, y, 'Sk = ', 2, 'purple', 84)
    f.text(326, y+56, 'predicts Tue', 12, 'muted')
    f.arrow(372, y+17, 412, y+17, 'red', width=1.8)
    pill_num(f, 416, y, '5 − 2 = ', 3, 'red', 110)
    f.text(471, y+56, 'residual', 12, 'red')
    f.arrow(530, y+17, 570, y+17, 'green', width=1.8)
    f.text(550, y+8, '× βkᵀ', 12, 'green')
    cells(f, 574, y, [3, 0], tones=['green', 'line'])
    f.text(622, y+56, 'write along k', 12, 'green')
    f.text(690, y+23, '=', 22, 'ink')
    cells(f, 712, y, [5, 4], outline={0: 'green'})
    f.math(760, y+56, r'\c{purple}{S_t}', 15, anchor='middle')
    f.text(736, y+74, 'Fri', 12, 'green', True)
    f.text(786, y+74, 'untouched', 12, 'muted')
    # contrast: additive
    y2 = 206
    f.text(30, y2+22, 'add', 15, 'red', True, 'start')
    f.text(30, y2+40, '(linear attn)', 12, 'muted', anchor='start')
    cells(f, 130, y2, [2, 4])
    f.text(266, y2+23, '+ 5 kᵀ =', 15, 'ink')
    cells(f, 310, y2, [7, 4], outline={0: 'red'})
    f.text(420, y2+23, 'reads 7: neither Tue nor Fri, nothing was replaced', 14, 'red', anchor='start')
    f.hdivider(268)
    # beta sweep
    f.text(30, 296, 'Partial edit: Sk after the update = 2 + β · 3', 15, 'ink', True, 'start')
    x0, x1, ly = 60, 400, 344
    f.line(x0, ly, x1, ly, 'line', 3)
    for b in (0, .25, .5, .75, 1):
        v = 2 + 3*b
        px = x0 + (x1-x0)*b
        f.circle(px, ly, 6, color='green' if b == 1 else 'teal', fill=True)
        f.text(px, ly-14, f'{v:g}', 14, 'ink', True)
        f.text(px, ly+24, f'β = {b:g}', 12, 'teal')
    f.text(x0, ly+44, 'Tue', 12, 'muted')
    f.text(x1, ly+44, 'Fri', 12, 'green', True)
    # interference
    f.text(470, 296, 'Who else moves? (β = 1)', 15, 'ink', True, 'start')
    rows = [('q = e₂ (order 2208)', 4.0, 4.0), ('q = e₁ (order 7314)', 2.0, 5.0), ('q = (e₁ + e₂)/√2', 6/math.sqrt(2), 9/math.sqrt(2))]
    for i, (lab, a, b) in enumerate(rows):
        yy = 324 + i*30
        f.text(470, yy, lab, 13, 'gold', anchor='start')
        f.text(690, yy, f'{a:.2f}', 14, 'ink', anchor='end')
        f.text(716, yy, '→', 14, 'muted')
        f.text(782, yy, f'{b:.2f}', 14, 'red' if i == 2 else ('green' if i == 1 else 'ink'), True, 'end')
        f.text(800, yy, ['kept', 'edited', 'shifted'][i], 12, ['muted', 'green', 'red'][i], anchor='start')
    f.text(450, 446, 'Cost over linear attention: one extra matrix-vector read of S per token. State size unchanged.', 14, 'muted')
    f.save()


# ---------------------------------------------------------------- card 2
def gated():
    f = Fig('delta-gated', 'Gated DeltaNet on the toy with α = 0.8 and β = 0.5: decay [2, 4] to [1.6, 3.2], read 1.6, residual 3.4, write 1.7, result [3.3, 3.2]. The grid gives S_t for α = 1, 0.8, 0 and β = 0, 0.5, 1: α = 1 is DeltaNet, α = 0 wipes the state before writing. Qwen3-Next-80B-A3B: 36 Gated DeltaNet layers and 12 gated-attention layers, 32 value heads × 128 × 128 = 524,288 state numbers per layer.', 470)
    f.math(30, 36, r'\c{purple}{S_t} = \c{teal}{α_t}\c{purple}{S_{t-1}}(I - \c{teal}{β_t}\c{blue}{k_t}\c{blue}{k_t}^\top) + \c{teal}{β_t}\c{purple}{v_t}\c{blue}{k_t}^\top', 21)
    f.text(870, 36, 'toy: α = 0.8, β = 0.5', 14, 'muted', anchor='end')
    y = 84
    keyheads(f, 30, y-8)
    cells(f, 30, y, [2, 4])
    f.arrow(134, y+17, 184, y+17, 'teal', width=1.8)
    f.text(159, y+8, '× α', 13, 'teal', True)
    cells(f, 188, y, [1.6, 3.2])
    f.text(237, y+54, 'faded state', 12, 'teal')
    f.arrow(292, y+17, 330, y+17, 'blue', width=1.8)
    f.text(311, y+8, 'read', 12, 'blue')
    pill_num(f, 334, y, 'Sk = ', 1.6, 'purple', 92)
    f.arrow(430, y+17, 466, y+17, 'red', width=1.8)
    pill_num(f, 470, y, '5 − 1.6 = ', 3.4, 'red', 124)
    f.arrow(598, y+17, 634, y+17, 'green', width=1.8)
    f.text(616, y+8, '× β', 12, 'green')
    cells(f, 638, y, [1.7, 0], tones=['green', 'line'])
    f.text(746, y+23, '=', 22, 'ink')
    cells(f, 766, y, [3.3, 3.2], outline={0: 'green'})
    f.math(816, y+56, r'\c{purple}{S_t}', 15, anchor='middle')
    f.hdivider(158)
    # alpha x beta grid
    f.text(30, 186, 'One step on the toy, for 3 × 3 gate settings', 15, 'ink', True, 'start')
    gx, gy = 120, 222
    betas, alphas = (0, .5, 1), (1, .8, 0)
    for c, b in enumerate(betas):
        f.text(gx+c*110+48, gy-12, f'β = {b:g}', 14, 'teal', True)
    for r, a in enumerate(alphas):
        f.text(gx-14, gy+r*44+23, f'α = {a:g}', 14, 'teal', True, 'end')
        for c, b in enumerate(betas):
            v = [a*2*(1-b)+5*b, a*4]
            hi = (a == .8 and b == .5)
            cells(f, gx+c*110, gy+r*44, v, cw=50, ch=32, size=14, outline={0: 'green', 1: 'green'} if hi else None)
    f.text(gx+330, gy+21, '← DeltaNet', 13, 'muted', anchor='start')
    f.text(gx+330, gy+109, '← reset', 13, 'muted', anchor='start')
    f.text(gx+48, gy+150, 'no write', 12, 'muted')
    f.text(gx+268, gy+150, 'Sk = 5: full overwrite', 12, 'green')
    # production
    f.panel(560, 176, 320, 200, 'Qwen3-Next-80B-A3B (config)')
    f.lines(576, 226, ['48 layers: 36 GDN + 12 gated attention',
                       'GDN heads: 16 for q, k · 32 for v, d = 128',
                       'state / layer: 32 × 128 × 128 = 524,288',
                       'short conv kernel 4 on q, k, v'], 13, 'ink', 'start')
    cellz = [('', 'blue' if i % 4 == 3 else 'purple') for i in range(48)]
    f.strip(576, 316, cellz, w=5.4, h=20, gap=0.9)
    f.rect(576, 348, 11, 11, 'purple', .88, 2)
    f.text(592, 358, 'GDN', 12, 'ink', anchor='start')
    f.rect(636, 348, 11, 11, 'blue', .88, 2)
    f.text(652, 358, 'attention (KV grows)', 12, 'ink', anchor='start')
    f.text(450, 410, 'α clears the whole head (cheap reset at a topic switch); β edits one association.', 14, 'ink')
    f.text(450, 432, 'Scalar α also shrinks order 2208: 4 → 3.2. The next card fixes that per channel.', 14, 'red')
    f.text(450, 456, 'Olmo Hybrid 7B uses the same rule: 24 GDN + 8 attention layers, 30 heads, key dim 96, value dim 192.', 13, 'muted')
    f.save()


# ---------------------------------------------------------------- card 3
def kda():
    f = Fig('delta-kda', 'KDA on the toy with per-channel decay α = (0.8, 1) and β = 0.5: [2, 4] fades to [1.6, 4], the residual 3.4 is written at half strength, giving [3.3, 4]; scalar-decay Gated DeltaNet gives [3.3, 3.2]. The transition is diagonal plus a rank-1 term tied to k, which halves the chunk matrices of general DPLR. Kimi-Linear-48B-A3B: 27 layers, 20 KDA and 7 MLA, 32 heads × 128 × 128 = 524,288 state numbers per KDA layer.', 460)
    f.math(30, 36, r'\c{purple}{S_t} = \c{purple}{S_{t-1}}\,\text{Diag}(\c{teal}{α_t})(I - \c{teal}{β_t}\c{blue}{k_t}\c{blue}{k_t}^\top) + \c{teal}{β_t}\c{purple}{v_t}\c{blue}{k_t}^\top', 21)
    f.text(870, 36, 'toy: α = (0.8, 1), β = 0.5', 14, 'muted', anchor='end')
    y = 92
    keyheads(f, 30, y-10)
    cells(f, 30, y, [2, 4])
    f.arrow(134, y+17, 196, y+17, 'teal', width=1.8)
    f.text(165, y+8, '× Diag(α)', 12, 'teal', True)
    for c, a in enumerate((0.8, 1)):
        f.rect(200+c*50, y-30, 47, 18*a, 'teal', .75, 2)
        f.text(223+c*50, y-36, f'α = {a:g}', 11.5, 'teal', True)
    cells(f, 200, y, [1.6, 4], tones=['purple', 'purple'])
    f.arrow(304, y+17, 340, y+17, 'red', width=1.8)
    pill_num(f, 344, y, '5 − 1.6 = ', 3.4, 'red', 124)
    f.arrow(472, y+17, 508, y+17, 'green', width=1.8)
    f.text(490, y+8, '× β', 12, 'green')
    f.text(530, y+23, '→', 18, 'ink')
    cells(f, 552, y, [3.3, 4], outline={0: 'green', 1: 'green'})
    f.text(602, y+54, 'KDA', 13, 'green', True)
    cells(f, 714, y, [3.3, 3.2], outline={1: 'red'})
    f.text(764, y+54, 'GDN (scalar 0.8)', 13, 'red', True)
    f.text(681, y+23, 'vs', 14, 'muted', True)
    f.hdivider(170)
    # DPLR panel
    f.panel(20, 184, 400, 196, 'Transition = diagonal + rank 1, tied to k')
    f.math(220, 230, r'\c{teal}{A_t} = \text{Diag}(\c{teal}{α_t}) - \c{teal}{β_t}(\c{teal}{α_t}\odot\c{blue}{k_t})\c{blue}{k_t}^\top', 17, anchor='middle')
    d = 6
    k = [0.7, 0.5, 0.35, 0.25, 0.2, 0.15]
    n = math.sqrt(sum(x*x for x in k))
    k = [x/n for x in k]
    al = [0.95, 0.6, 0.9, 0.4, 0.8, 0.7]
    A = matmul([[al[i] if i == j else 0 for j in range(d)] for i in range(d)], house(k, .8))
    heat(f, 40, 248, A, 18)
    f.text(93, 370, 'toy A, 6 key channels', 12, 'muted')
    f.text(170, 266, 'second-level chunk matrix computations', 12.5, 'muted', anchor='start')
    f.bars(160, 276, [('general DPLR', 4, 'red'), ('KDA', 2, 'green')], width=110, bar=18, gap=10, label_w=100, size=13)
    f.lines(170, 344, ['reported: ≈2× faster kernel than general', 'DPLR (RWKV-7 class), up to 64K tokens'], 12, 'muted', 'start')
    # Kimi Linear
    f.panel(436, 184, 444, 196, 'Kimi-Linear-48B-A3B (config)')
    full = {4, 8, 12, 16, 20, 24, 27}
    cellz = [('', 'blue' if i+1 in full else 'purple') for i in range(27)]
    f.strip(452, 222, cellz, w=13.5, h=24, gap=1.6)
    f.text(452, 262, 'MLA at layers 4, 8, …, 24 and 27: 20 KDA + 7 MLA', 13, 'ink', anchor='start')
    f.lines(452, 290, ['KDA: 32 heads × 128 × 128 = 524,288 state numbers / layer',
                       'MLA (NoPE): 512 + 64 = 576 cached numbers / token / layer',
                       'reported at 1M context: up to 75% less KV cache,',
                       'up to 6× decoding throughput'], 13, 'ink', 'start')
    f.text(450, 418, 'Per-channel α keeps order 2208 at 4 while 7314 is overwritten. Same state size as GDN.', 14, 'ink')
    f.text(450, 442, 'Order matters: Diag(α)(I − βkkᵀ) ≠ (I − βkkᵀ)Diag(α) when k spans channels with different α.', 13, 'muted')
    f.save()


# ---------------------------------------------------------------- card 4
def sum_form():
    f = Fig('delta-sum-form', 'Product form versus sum form on the toy with α = 0.8 and β = 0.5. Product (Gated DeltaNet, KDA): decay first, the erase reads the faded 1.6 and removes 0.8, result [3.3, 3.2]. Sum (RWKV-7, Comba): decay and erase act in parallel, the erase reads the unfaded 2 and removes 1.0, result [3.1, 3.2]. RWKV-7 uses vector decay w_t in (0.545, 1) and a vector in-context learning rate a_t in (0, 1) per channel, with head size 64 (4,096 state numbers per head).', 452)
    f.math(30, 36, r'\text{product: } \c{purple}{S_{t-1}}\,\c{teal}{α}(I - \c{teal}{β}\c{blue}{kk}^\top) \quad \text{vs} \quad \text{sum: } \c{purple}{S_{t-1}}(\c{teal}{α}I - \c{teal}{β}\c{blue}{kk}^\top), \quad \text{both } + \c{teal}{β}\c{purple}{v}\c{blue}{k}^\top', 20)
    lanes = [('Product · GDN, KDA', 76, [1.6, 3.2], 'erase reads faded 1.6', [0.8, 3.2], [3.3, 3.2], 'green'),
             ('Sum · RWKV-7, Comba', 176, [1.6, 3.2], 'erase reads unfaded 2', [0.6, 3.2], [3.1, 3.2], 'teal')]
    for name, y, faded, note, erased, out, col in lanes:
        f.text(30, y+6, name, 15, 'ink', True, 'start')
        y += 20
        cells(f, 30, y, [2, 4], cw=46, ch=32, size=14)
        f.arrow(126, y+16, 160, y+16, 'teal', width=1.6)
        f.text(143, y+8, '× α', 12, 'teal', True)
        cells(f, 164, y, faded, cw=46, ch=32, size=14)
        f.arrow(260, y+16, 294, y+16, 'red', width=1.6)
        rm = round(faded[0]-erased[0] if col == 'green' else 2*.5, 2)
        f.text(277, y+8, f'−{rm:g}', 12, 'red', True)
        cells(f, 298, y, erased, cw=46, ch=32, size=14)
        f.arrow(394, y+16, 428, y+16, 'green', width=1.6)
        f.text(411, y+8, '+2.5', 12, 'green', True)
        cells(f, 432, y, out, cw=46, ch=32, size=14, outline={0: col})
        f.text(30, y+52, note, 13, 'red', anchor='start')
    f.text(30, 278, 'Same toy, different first entry: 3.3 vs 3.1. Order 2208 is identical: 3.2.', 13, 'muted', anchor='start')
    # RWKV-7 panel
    f.panel(560, 64, 320, 236, 'RWKV-7: everything per channel')
    f.math(720, 108, r'\c{teal}{A_t} = \text{diag}(\c{teal}{w_t}) - \c{blue}{κ_t}(\c{teal}{a_t}\odot\c{blue}{κ_t})^\top', 17, anchor='middle')
    w = [.98, .7, .9, .6, .95, .8, .56, .99]
    a = [.2, .9, .5, .7, .1, .6, .95, .3]
    bx = 640
    f.math(bx-12, 152, r'\c{teal}{w_t}', 16, anchor='end')
    f.math(bx-12, 202, r'\c{green}{a_t}', 16, anchor='end')
    for i in range(8):
        f.rect(bx+i*26, 156-36*w[i], 20, 36*w[i], 'teal', .75, 2)
        f.rect(bx+i*26, 206-36*a[i], 20, 36*a[i], 'green', .75, 2)
    f.text(bx+104, 222, 'toy, 8 of 64 key channels', 12, 'muted')
    f.lines(576, 248, ['w ∈ (0.545, 1): decay per channel', 'a ∈ (0, 1): learning rate per channel',
                       'removal key κ̂ ≠ replacement key k̃'], 12.5, 'ink', 'start')
    f.hdivider(310)
    f.text(30, 340, 'Sizes', 15, 'ink', True, 'start')
    f.lines(30, 364, ['RWKV-7 World: head size 64 → 64 × 64 = 4,096 state numbers per head',
                      '0.1B–2.9B models trained on 1.6T–5.6T tokens'], 13, 'ink', 'start')
    f.text(500, 340, 'With w = 0.8, a = 0.5 on every channel', 13, 'muted', anchor='start')
    f.text(500, 360, '(and k̃ = a ⊙ k), RWKV-7 gives [3.1, 3.2] too.', 13, 'muted', anchor='start')
    f.text(450, 426, 'Product: the erase sees what decay left. Sum: decay and erase both act on the old state, then add.', 14, 'ink')
    f.save()


# ---------------------------------------------------------------- card 5
def eigen():
    f = Fig('delta-eigen', 'The delta transition I − βkkᵀ has eigenvalues 1 (d_k − 1 times) and 1 − β. With β in [0, 1] they stay in [0, 1]; allowing β up to 2 reaches −1, a reflection. Parity toy: with β = 2 for every 1-bit, the 1-D state flips sign and after bits 1 0 1 1 0 1 ends at +1 (even); with β ≤ 1 the sign can never flip. Two reflections with mirrors 30° apart make a 60° rotation, as in DeltaProduct with n_h = 2. Olmo Hybrid 7B ships linear_allow_neg_eigval = true.', 470)
    f.math(30, 36, r'\text{eig}(I - \c{teal}{β}\c{blue}{kk}^\top) = \{1, \dots, 1, \; 1 - \c{teal}{β}\}, \quad \c{teal}{β} \in (0, 2) ⇒ 1 - \c{teal}{β} \in (-1, 1)', 21)
    # number line
    f.text(30, 82, 'Where the eigenvalue 1 − β can sit', 15, 'ink', True, 'start')
    x0, x1, ly = 64, 420, 150
    def X(v):
        return x0 + (v+1)/2*(x1-x0)
    f.rect(X(0), ly-14, X(1)-X(0), 28, 'green', .18, 4)
    f.rect(X(-1), ly-14, X(0)-X(-1), 28, 'teal', .18, 4)
    f.line(x0-10, ly, x1+10, ly, 'muted', 1.4)
    for v, lab, col in ((1, 'β = 0: keep', 'green'), (0, 'β = 1: erase', 'green'), (-1, 'β = 2: flip', 'teal')):
        f.circle(X(v), ly, 6, color=col, fill=True)
        f.text(X(v), ly+32, f'{v:g}', 14, 'ink', True)
        f.text(X(v), ly-22, lab, 12.5, col, True)
    f.text(X(.5), ly+52, 'β ∈ [0, 1]: DeltaNet, GDN, KDA', 13, 'green')
    f.text(X(-.5), ly+52, 'β ∈ (1, 2): reflections', 13, 'teal')
    f.text(X(-.5), ly+68, '(Grazzi 2025; Olmo Hybrid 7B)', 12, 'muted')
    # parity toy
    f.panel(500, 64, 380, 182, 'Parity toy: β = 2·bit, k = e₁, d = 1')
    bits = [1, 0, 1, 1, 0, 1]
    s = 1
    f.text(530, 120, 'bit', 13, 'muted', anchor='start')
    f.text(530, 162, 'β ≤ 2', 13, 'teal', True, 'start')
    f.text(530, 204, 'β ≤ 1', 13, 'red', True, 'start')
    f.box(588, 142, 34, 30, '+1', 'teal', size=14)
    f.box(588, 184, 34, 30, '+1', 'red', size=14)
    for i, b in enumerate(bits):
        x = 632 + i*40
        f.text(x+17, 120, str(b), 15, 'ink', True)
        s = s*(1-2*b)
        f.box(x, 142, 34, 30, f'{s:+d}', 'teal' if s > 0 else 'red', fill=s < 0, size=14)
        f.box(x, 184, 34, 30, '+', 'line', size=14)
    f.text(690, 236, 'ends at +1: four 1-bits, even', 12.5, 'teal')
    # reflections -> rotation
    f.panel(20, 262, 470, 192, 'Two reflections = one rotation (DeltaProduct, 2 steps/token)')
    cx, cy, R = 150, 380, 70
    for ang, col in ((0, 'blue'), (30, 'blue')):
        t = math.radians(ang)
        f.line(cx-R*1.15*math.cos(t), cy+R*1.15*math.sin(t), cx+R*1.15*math.cos(t), cy-R*1.15*math.sin(t), col, 1.4, dash=True)
    for ang, col, lab in ((10, 'purple', 'x'), (-10, 'muted', ''), (70, 'green', '')):
        t = math.radians(ang)
        f.arrow(cx, cy, cx+R*math.cos(t), cy-R*math.sin(t), col, width=2.2)
        if lab:
            f.text(cx+R*math.cos(t)+12, cy-R*math.sin(t)+4, lab, 14, col, True)
    f.text(cx+R*.5+30, cy+44, 'after mirror 1', 12, 'muted')
    f.text(cx+R*math.cos(math.radians(70))+8, cy-R*math.sin(math.radians(70))-6, 'after mirror 2', 12, 'green', anchor='start')
    f.math(360, 318, r'(I - 2\c{blue}{k_2k_2}^\top)(I - 2\c{blue}{k_1k_1}^\top)', 16, anchor='middle')
    f.lines(360, 346, ['mirrors 30° apart', '→ rotation by 60° (2θ)', 'x at 10° → 70°'], 13, 'ink', 'middle')
    f.text(360, 420, 'Mamba-3 gets rotations directly', 12.5, 'teal')
    f.text(360, 438, 'from a complex state', 12.5, 'teal')
    # production
    f.panel(500, 262, 380, 192, 'Shipped: Olmo Hybrid 7B (config)')
    f.parts.append(f'<rect x="516" y="298" width="348" height="30" rx="5" fill="{COL["white"]}" stroke="{COL["line"]}"/>')
    f.text(530, 318, 'linear_allow_neg_eigval: true', 14, 'teal', True, 'start', family='Menlo,Consolas,monospace')
    f.lines(516, 352, ['32 layers: 24 GDN + 8 full attention (3:1)',
                       'GDN: 30 heads, key dim 96, value dim 192',
                       'state / layer: 30 × 192 × 96 = 552,960',
                       'β = 2·sigmoid(·): one-line change, ‖A‖ ≤ 1'], 13, 'ink', 'start')
    f.save()


# ---------------------------------------------------------------- card 6
def taxonomy():
    d = 6
    k = [0.7, 0.5, 0.35, 0.25, 0.2, 0.15]
    k2 = [0.1, -0.4, 0.6, 0.2, -0.5, 0.4]
    al = [0.95, 0.6, 0.9, 0.4, 0.8, 0.7]
    D = [[al[i] if i == j else 0 for j in range(d)] for i in range(d)]
    rkda = matmul(D, house(k, .8))
    n = math.sqrt(sum(x*x for x in k))
    kh = [x/n for x in k]
    aa = [.3, .8, .5, .6, .2, .9]
    rwkv = [[(al[i] if i == j else 0) - kh[i]*aa[j]*kh[j] for j in range(d)] for i in range(d)]
    prod = matmul(house(k, 1.6), house(k2, 1.6))
    rot = [[0]*d for _ in range(d)]
    for b, th in enumerate((0.5, 1.1, 0.25)):
        c, s = .9*math.cos(th), .9*math.sin(th)
        i = 2*b
        rot[i][i], rot[i][i+1], rot[i+1][i], rot[i+1][i+1] = c, -s, s, c
    classes = [
        (r'I', 'linear attention', 'eig 1', 'prefix sum', eye(d)),
        (r'\c{teal}{γ}I', 'RetNet, Lightning', 'eig γ, fixed', 'cumprod', eye(d, .85)),
        (r'\c{teal}{α_t}I', 'Mamba-2, mLSTM', 'eig α, per token', 'cumprod', eye(d, .6)),
        (r'\text{Diag}(\c{teal}{α_t})', 'GLA, HGRN2', 'eig α per channel', 'log-space cumsum', D),
        (r'I - \c{teal}{β}\c{blue}{kk}^\top', 'DeltaNet', 'eig 1, 1−β', 'WY / UT', house(k, .8)),
        (r'\c{teal}{α}(I - \c{teal}{β}\c{blue}{kk}^\top)', 'Gated DeltaNet', 'eig α, α(1−β)', 'WY / UT', [[.8*x for x in r] for r in house(k, .8)]),
        (r'\text{Diag}(\c{teal}{α})(I - \c{teal}{β}\c{blue}{kk}^\top)', 'KDA, 2026 GDN variants', 'product form', 'UT, tied DPLR', rkda),
        (r'\text{Diag}(\c{teal}{w}) - \c{blue}{κ}(\c{teal}{a}\odot\c{blue}{κ})^\top', 'RWKV-7, Comba', 'sum form', 'general DPLR', rwkv),
        (r'\prod_j(I - \c{teal}{β_j}\c{blue}{k_jk_j}^\top)', 'DeltaProduct', 'β ∈ (0,2): eig < 0', 'UT × steps/token', prod),
        (r'\c{teal}{α}R_t', 'Mamba-3 (complex)', 'eig α·e^(±iθ)', 'RoPE-style', rot),
    ]
    f = Fig('delta-taxonomy', 'Ten transition matrices A_t drawn as toy 6×6 heat maps (teal positive, red negative), with the models that use them, their eigenvalues and their chunk algorithm: identity (linear attention), fixed γI (RetNet, Lightning), α_t I (Mamba-2, mLSTM), Diag(α_t) (GLA, HGRN2), I − βkkᵀ (DeltaNet), α(I − βkkᵀ) (Gated DeltaNet), Diag(α)(I − βkkᵀ) (KDA), Diag(w) − κ̂(a⊙κ̂)ᵀ (RWKV-7, Comba), a product of Householders (DeltaProduct), and α times a rotation (Mamba-3). All keep the same d_v×d_k state.', 470)
    f.math(30, 36, r'\c{purple}{S_t} = \c{purple}{S_{t-1}}\c{teal}{A_t} + \c{purple}{W_t}', 22)
    f.math(870, 36, r'\text{state } \c{purple}{S} \in \mathbb{R}^{d_v \times d_k} \text{ in every class}', 17, 'muted', anchor='end')
    for idx, (expr, who, eig, chunk, M) in enumerate(classes):
        r, c = divmod(idx, 5)
        x0, y0 = 20 + c*176, 62 + r*202
        tint = 'green' if idx < 4 else ('blue' if idx < 8 else 'purple')
        f.parts.append(f'<rect x="{x0+4}" y="{y0}" width="168" height="194" rx="8" fill="{COL["soft"]}" stroke="{COL["line"]}"/>')
        f.text(x0+16, y0+20, f'{idx+1}', 13, tint, True, 'start')
        heat(f, x0+46, y0+12, M, 14)
        f.math(x0+88, y0+120, expr, 14 if len(expr) < 60 else 12.5, anchor='middle')
        f.text(x0+88, y0+142, who, 13, 'ink', True)
        f.text(x0+88, y0+160, eig, 12, 'muted')
        f.text(x0+88, y0+178, chunk, 12, tint, True)
    f.save()


# ---------------------------------------------------------------- card 7
def frontier():
    f = Fig('delta-frontier', 'Four 2026 refinements of the delta rule, each shown as a toy 5×5 state with the part it gates. FG²-GDN: per-channel write strength β on key and value (tied; separate in FG²-GDN+). Gated DeltaNet-2: an erase gate b per key channel and a write gate w per value channel, both at address k. Kalman DeltaNet: step size from an uncertainty state p per key channel. Erase-then-Delta: a separate erase address e before the write at k. Timeline: RWKV-7 already had per-channel write strength in March 2025.', 478)
    f.math(30, 36, r'\c{purple}{S_t} = \c{purple}{S_{t-1}}\,\text{Diag}(\c{teal}{α_t})(I - (\c{teal}{b_t}\odot\c{blue}{k_t})\c{blue}{k_t}^\top) + (\c{teal}{w_t}\odot\c{purple}{v_t})\c{blue}{k_t}^\top', 21)
    f.text(870, 36, 'GDN-2 form', 14, 'muted', anchor='end')
    base = [[.25+.55*(((r*5+c*3+1) % 7)/6) for c in range(5)] for r in range(5)]
    panels = [
        ('FG²-GDN (+)', 'Apr 2026', r'\c{blue}{k} \to √\c{teal}{β}\odot\c{blue}{k}, \; \c{purple}{v} \to √\c{teal}{β}\odot\c{purple}{v}', 'strength: vector β', '340M/15B, 1.3B/100B', ([.9, .3, .7, .2, .5], [.9, .3, .7, .2, .5]), None),
        ('Gated DeltaNet-2', 'May 2026', r'\c{teal}{b} \in \mathbb{R}^{d_k}, \; \c{teal}{w} \in \mathbb{R}^{d_v}', 'erase b ≠ write w', '1.3B/100B', ([.8, .2, .6, .9, .3], [.4, .9, .2, .7, .5]), None),
        ('Kalman Delta Networks', 'Sep 2026', r'\c{teal}{κ} = \c{teal}{p}\odot\c{blue}{k} / (r + \sum_i \c{teal}{p_i}\c{blue}{k_i}^2)', 'step from uncertainty p', '750M/50B, 1.3B/100B', ([.95, .15, .5, .3, .8], None), None),
        ('Erase-then-Delta', 'Jun 2026', r'(I - \c{teal}{γ}\c{red}{e}\c{red}{e}^\top)(I - \c{teal}{β}\c{blue}{kk}^\top)', 'address: e ≠ k', '2.5B dense, 25B-A2.8B MoE', None, (1, 3)),
    ]
    for p, (name, date, expr, what, scale, gates, ek) in enumerate(panels):
        x0 = 20 + p*218
        f.parts.append(f'<rect x="{x0}" y="60" width="210" height="312" rx="10" fill="{COL["soft"]}" stroke="{COL["line"]}"/>')
        f.text(x0+105, 84, name, 15, 'ink', True)
        f.text(x0+105, 102, date, 12, 'muted')
        mx, my, cell = x0+62, 172, 20
        f.matrix(mx, my, 5, 5, cell, 'purple', base)
        if gates:
            top, left = gates
            for j, g in enumerate(top):
                f.rect(mx+j*cell, my-6-30*g, cell-4, 30*g, 'teal', .8, 2)
            if left:
                for i, g in enumerate(left):
                    f.rect(mx-6-30*g, my+i*cell, 30*g, cell-4, 'teal', .8, 2)
        if p == 0:
            f.text(mx-42, my+118, 'β per channel (key = value)', 12, 'teal', anchor='start')
        if p == 1:
            f.text(mx+48, my-42, 'b: erase, per key', 12, 'teal')
            f.text(mx-42, my+118, 'w: write, per value', 12, 'teal', anchor='start')
        if p == 2:
            f.text(mx+48, my-42, 'p: uncertainty per key', 12, 'teal')
        if ek:
            e, kk = ek
            f.rect(mx+e*cell-2, my-4, cell, 5*cell+4, 'red', .0, 3, stroke='red')
            f.rect(mx+kk*cell-2, my-4, cell, 5*cell+4, 'green', .0, 3, stroke='green')
            f.text(mx+e*cell+8, my-10, 'e', 14, 'red', True)
            f.text(mx+kk*cell+8, my-10, 'k', 14, 'blue', True)
            f.text(mx+48, my-30, 'erase stale at e, write at k', 12, 'muted')
        f.math(x0+105, 312, expr, 13 if p != 2 else 12.5, anchor='middle')
        f.text(x0+105, 340, what, 13, 'teal', True)
        f.text(x0+105, 360, 'tested: ' + scale, 12, 'muted')
    # timeline
    tl = [('RWKV-7', 'Mar 2025', 'per-channel a_t'), ('FG²-GDN', 'Apr 2026', ''), ('GDN-2', 'May 2026', ''), ('EDA', 'Jun 2026', ''), ('KDN', 'Sep 2026', '')]
    ty = 412
    f.line(60, ty, 840, ty, 'line', 2)
    for i, (n, dte, note) in enumerate(tl):
        x = 60 + i*195
        f.circle(x, ty, 6, color='green' if i == 0 else 'teal', fill=True)
        f.text(x, ty-14, n, 13, 'green' if i == 0 else 'ink', True)
        f.text(x, ty+22, dte, 12, 'muted')
    f.text(60, ty+42, 'per-channel write strength first shipped in RWKV-7 (vector a), a year before FG²-GDN', 12.5, 'green', anchor='start')
    f.save()


def build():
    deltanet()
    gated()
    kda()
    sum_form()
    eigen()
    taxonomy()
    frontier()
