"""02 · Stable softmax: QK-norm / soft-cap / QK-Clip, learned sinks, gated attention, FoX + Diff, 2026 anatomy."""
import math
import random
from figkit import Fig


def strip(f, x, y, parts, size=21, gap=36):
    """Formula strip of several expressions separated by real gaps (SVG collapses spaces)."""
    for part in parts:
        x += f.math(x, y, part, size) + gap


def softmax(xs):
    m = max(xs)
    e = [math.exp(x-m) for x in xs]
    s = sum(e)
    return [v/s for v in e]


# --------------------------------------------------------------------------
def qk_norm():
    f = Fig('stable-qk-norm', 'Three ways to bound attention logits. QK-norm: RMS-normalised q and k with gain g give |logit| at most sqrt(d_h) g_q g_k, 11.3 for d_h = 128 and g = 1; toy histograms of 4,096 logits: without norm and with q, k grown 4x the logits spread to about plus or minus 50 and the top weight over 64 keys is near 1, with QK-norm they stay within the bound. Soft-cap (Gemma 2): c tanh(e / c) with c = 50 maps 50 to 38 and 100 to 48; final logits use c = 30. QK-Clip (Kimi K2, tau = 100): a head with max logit 250 gets gamma = 0.4 and W_q, W_k scaled by 0.63.', 458)
    strip(f, 30, 36, [r'\c{gold}{q} \leftarrow g_q \odot \c{gold}{q} / \text{RMS}(\c{gold}{q})', r'\c{blue}{k} \leftarrow g_k \odot \c{blue}{k} / \text{RMS}(\c{blue}{k})',
                      r'⇒ |\c{gold}{q}^{\top}\c{blue}{k}| / \sqrt d_h \le \sqrt d_h\, g_q g_k'], 21, 34)
    # panel 1: histogram
    f.panel(20, 56, 280, 310, 'QK-norm · toy, d_h = 128')
    rng = random.Random(7)
    d = 128

    def sample(scale, norm):
        q = [rng.gauss(0, 1) for _ in range(d)]
        k = [rng.gauss(0, 1) for _ in range(d)]
        # shared direction: trained q and k align a little
        for i in range(8):
            k[i] += .9*q[i]
        if norm:
            rq = math.sqrt(sum(x*x for x in q)/d)
            rk = math.sqrt(sum(x*x for x in k)/d)
            q = [x/rq for x in q]
            k = [x/rk for x in k]
        else:
            q = [x*scale for x in q]
            k = [x*scale for x in k]
        return sum(a*b for a, b in zip(q, k))/math.sqrt(d)
    raw = [sample(4, False) for _ in range(4096)]
    nrm = [sample(1, True) for _ in range(4096)]
    lo, hi, nb = -60, 60, 40
    x0, x1, base = 40, 280, 300

    def hist(vals, color, op):
        bins = [0]*nb
        for v in vals:
            b = int((v-lo)/(hi-lo)*nb)
            if 0 <= b < nb:
                bins[b] += 1
        mx = 1100
        w = (x1-x0)/nb
        for b, c in enumerate(bins):
            h = min(150, 150*c/mx)
            if c:
                f.rect(round(x0+b*w, 1), round(base-h, 1), round(w-1, 1), round(h, 1), color, op, 1)
    hist(raw, 'red', .75)
    hist(nrm, 'green', .8)
    f.line(x0, base, x1, base, 'muted', 1.2)
    for v in (-50, -25, 0, 25, 50):
        x = x0 + (v-lo)/(hi-lo)*(x1-x0)
        f.text(x, base+18, str(v), 13, 'muted')
    bnd = 11.3
    for v in (-bnd, bnd):
        x = x0 + (v-lo)/(hi-lo)*(x1-x0)
        f.line(x, 130, x, base, 'green', 1.4, dash=True)
    f.text(x0 + (bnd-lo)/(hi-lo)*(x1-x0) + 6, 140, 'bound ±11.3', 13, 'green', True, 'start')
    # readouts: top weight over 64 toy keys
    pr = max(softmax(raw[:64]))
    pn = max(softmax(nrm[:64]))
    f.text(36, 100, f'no norm, q,k × 4: top weight {pr:.2f}', 13, 'red', True, 'start')
    f.text(36, 344, f'QK-norm: top weight {pn:.2f} (64 keys)', 13, 'green', True, 'start')
    # panel 2: soft-cap
    f.panel(310, 56, 280, 310, 'Soft-cap · Gemma 2')
    X0, X1, Y0, Y1 = 352, 576, 300, 110

    def px(v):
        return X0 + v/150*(X1-X0)

    def py(v):
        return Y0 - v/80*(Y0-Y1)
    f.line(X0, Y0, X1, Y0, 'muted', 1.2)
    f.line(X0, Y0, X0, Y1, 'muted', 1.2)
    for v in (0, 50, 100, 150):
        f.text(px(v), Y0+18, str(v), 13, 'muted')
    for v in (0, 30, 50):
        f.text(X0-6, py(v)+4, str(v), 13, 'muted', anchor='end')
    f.text(X1, Y0-8, 'raw logit e', 13, 'muted', anchor='end')
    f.path([(px(0), py(0)), (px(80), py(80))], 'red', width=2, dash=True)
    f.text(px(62)-4, py(76), 'raw', 13, 'red', True, 'end')
    for c, col, lab in ((50, 'green', 'attention c = 50'), (30, 'teal', 'final logits c = 30')):
        pts = [(round(px(v), 1), round(py(c*math.tanh(v/c)), 1)) for v in range(0, 151, 3)]
        f.path(pts, col, width=2.6)
        f.line(X0, py(c), X1, py(c), col, 1, dash=True, opacity=.6)
        f.text(X1, py(c)-7 if c == 50 else py(c)+18, lab, 13, col, True, 'end')
    for v in (50, 100):
        y = 50*math.tanh(v/50)
        f.circle(px(v), py(y), 3.5, color='green', fill=True)
    f.text(450, 340, 'c = 50:  50 → 38 · 100 → 48 · 150 → 50', 13, 'green', True)
    f.text(450, 358, 'Gemma 3 replaced it with QK-norm', 13, 'muted')
    # panel 3: QK-Clip
    f.panel(600, 56, 280, 310, 'QK-Clip · Kimi K2, τ = 100')
    heads = [42, 76, 250, 61, 130, 88]
    B0, scale = 300, .66
    tau_y = B0 - 100*scale
    for h, smax in enumerate(heads):
        x = 628 + h*40
        top = B0 - smax*scale
        if smax > 100:
            f.rect(x, top, 26, smax*scale - 100*scale, 'red', .35, 2)
            f.rect(x, tau_y, 26, 100*scale, 'green', .85, 2)
            g = 100/smax
            f.text(x+13, top-6, f'γ={g:.2f}', 12, 'red', True)
        else:
            f.rect(x, top, 26, smax*scale, 'blue', .7, 2)
        f.text(x+13, B0+16, f'h{h+1}', 13, 'muted')
    f.line(618, tau_y, 870, tau_y, 'ink', 1.4, dash=True)
    f.text(866, tau_y-6, 'τ = 100', 13, 'ink', True, 'end')
    f.math(614, 100, r'S_{max} \text{ per head (toy)}', 14, 'muted')
    f.math(614, 340, r'h_3: W_q, W_k \times \sqrt 0.4 = \times 0.63', 14, 'red')
    f.text(614, 358, 'K2 without clip: > 1,000 (reported)', 13, 'red', anchor='start')
    f.hdivider(382, 20, 880)
    f.text(30, 408, 'where the norm sits', 14, 'ink', True, 'start')
    f.pill(190, 408, 'per head: Qwen3, Gemma 3/4, Qwen3-Next', 'green', 13)
    f.pill(530, 408, 'whole projection: OLMo 2/3, MiniMax-M2', 'green', 13)
    f.pill(190, 440, 'gain-free: Llama 4 Scout (RoPE layers), DeepSeek-V4 queries', 'green', 13)
    f.save()


# --------------------------------------------------------------------------
def sinks():
    f = Fig('stable-sinks', 'Toy 12-token causal attention. Vanilla softmax: rows with nothing to read dump weight on token 0, which becomes a sink. With a learned per-head sink logit z_h the extra column outside the sequence absorbs that mass and token 0 is ordinary; rows sum to less than 1. One query with six equal logits: vanilla must give each key 0.167; with z_h = 3 the sink takes 0.77 and each key 0.038, so the head outputs almost nothing. Cost: one scalar per head, 2,304 parameters in gpt-oss-120b. MiMo-V2-Flash at window 128 reports MMLU 54.9 to 58.3 with the sink.', 440)
    strip(f, 30, 36, [r'α_{tj} = exp(e_{tj}) / (\c{teal}{exp(z_h)} + Σ_{r \le t} exp(e_{tr}))', r'Σ_j α_{tj} = 1 - \c{teal}{α_{\text{sink}}} < 1'], 21, 50)
    n, cell = 12, 16
    rng = random.Random(3)
    noise = [[rng.uniform(-.4, .4) for _ in range(n)] for _ in range(n)]

    def logits(i, sink0):
        row = []
        for j in range(i+1):
            v = noise[i][j]
            if j == i:
                v += 1.2
            if j == 0 and sink0:
                v += 3.0
            if (i, j) in ((7, 3), (10, 5), (11, 8)):
                v += 3.2
            row.append(v)
        return row
    # panel A
    f.text(30, 74, 'Vanilla softmax (toy)', 15, 'ink', True, 'start')
    A = [softmax(logits(i, True)) for i in range(n)]
    ax, ay = 48, 96
    f.mask(ax, ay, n, cell, lambda i, j: ('blue', round(max(.06, A[i][j]), 2)) if j <= i else None)
    f.rect(ax-2, ay-2, cell+1, n*cell+2, 'red', 0, 2, 'red')
    f.text(ax, ay+n*cell+20, 'token 0 = sink', 13, 'red', True, 'start')
    f.text(ax, ay+n*cell+38, f'row 11: {A[11][0]:.2f} on token 0', 13, 'red', anchor='start')
    # panel B
    f.math(310, 74, r'\text{+ learned sink logit } \c{teal}{z_h}', 16, 'ink', bold=True)
    zh = 2.6
    Bw, Bs = [], []
    for i in range(n):
        p = softmax(logits(i, False) + [zh])
        Bw.append(p[:-1])
        Bs.append(p[-1])
    bx, by = 356, 96
    for i in range(n):
        f.rect(bx-26, by+i*cell, cell-1.5, cell-1.5, 'teal', round(max(.08, Bs[i]), 2))
    f.text(bx-18, by-6, '∅', 15, 'teal', True)
    f.mask(bx, by, n, cell, lambda i, j: ('blue', round(max(.06, Bw[i][j]), 2)) if j <= i else None)
    f.text(bx-26, by+n*cell+20, 'extra column: no value', 13, 'teal', True, 'start')
    f.text(bx-26, by+n*cell+38, f'row 11: sink {Bs[11]:.2f}, token 0 {Bw[11][0]:.2f}', 13, 'teal', anchor='start')
    # panel C: one query, nothing relevant
    f.divider(588, 60, 350)
    f.text(606, 74, 'One query, nothing relevant (toy)', 15, 'ink', True, 'start')
    f.text(606, 96, 'six keys, all logits 0', 13, 'muted', anchor='start')
    van = [1/6]*6
    zs = 3.0
    den = math.exp(zs) + 6
    ps, pk = math.exp(zs)/den, 1/den
    base = 250
    for k in range(6):
        h = van[k]*400
        f.rect(612+k*20, base-h, 15, h, 'blue', .8, 2)
    f.text(662, base+20, 'vanilla', 13, 'blue', True)
    f.text(662, base-van[0]*400-10, '0.167 each', 13, 'blue', True)
    for k in range(6):
        h = pk*400
        f.rect(752+k*16, base-h, 12, h, 'blue', .8, 2)
    f.rect(752+6*16+6, base-ps*160, 18, ps*160, 'teal', .85, 2)
    f.text(752+6*16+15, base-ps*160-8, f'{ps:.2f}', 13, 'teal', True)
    f.math(800, base+20, r'z_h = 3', 14, 'teal', 'middle', True)
    f.text(800, base-pk*400-8, f'{pk:.3f}', 13, 'blue', True)
    f.text(606, 296, 'vanilla: o = mean(v), forced read', 13, 'blue', anchor='start')
    f.text(606, 316, f'sink: o = {6*pk:.2f} × mean(v)', 13, 'teal', True, 'start')
    f.text(606, 336, '→ “attend to nothing”', 13, 'teal', anchor='start')
    f.hdivider(368, 20, 880)
    f.text(30, 396, 'Cost: one scalar per head', 14, 'ink', True, 'start')
    f.text(30, 418, 'gpt-oss-120b: 64 heads × 36 layers = 2,304 params', 13, 'muted', anchor='start')
    f.text(460, 396, 'MiMo-V2-Flash, W = 128 (reported)', 14, 'ink', True, 'start')
    f.text(460, 418, 'MMLU 54.9 → 58.3 · BBH 52.4 → 56.1 with sink', 13, 'green', True, 'start')
    f.save()


# --------------------------------------------------------------------------
def gated():
    f = Fig('stable-gated', 'Qwen3-Next attention layer shapes: x has 2,048 dims; q_proj outputs 16 heads x 256 for queries plus another 4,096 for the gate; SDPA output Y (4,096) is multiplied elementwise by sigmoid of the gate, then o_proj maps 4,096 to 2,048. Toy head with 12 channels: most gate values are near 0. Reported for a 15B MoE: first-token attention 46.7 percent to 4.8, max activation 1,053 to 94, perplexity 6.026 to 5.761, RULER at 128K 31.65 to 58.82. An output gate rescales one token read; a state gate changes every later read.', 460)
    strip(f, 30, 36, [r"\c{green}{Y'} = \c{green}{Y} \odot σ(\c{teal}{X W_θ})", r'\c{green}{Y} = \text{softmax}(\c{gold}{Q}\c{blue}{K}^{\top}/\sqrt d_h)\,\c{purple}{V}', r"\text{out} = W_O\,\c{green}{Y'}"], 21, 40)
    # shape pipeline (Qwen3-Next)
    f.text(30, 72, 'Qwen3-Next attention layer (config dims)', 14, 'ink', True, 'start')
    f.box(30, 86, 70, 34, '$x_t', 'ink', False, 16)
    f.text(65, 136, '2,048', 13, 'muted')
    f.arrow(102, 103, 128, 103, 'muted')
    f.box(130, 86, 92, 34, 'q_proj ×2', 'gold', False, 14)
    f.arrow(224, 96, 250, 92, 'gold')
    f.arrow(224, 110, 250, 128, 'teal')
    f.box(252, 78, 86, 26, '$Q · 16×256', 'gold', True, 13)
    f.box(252, 116, 86, 26, 'gate 4,096', 'teal', True, 13)
    f.arrow(340, 91, 366, 91, 'gold')
    f.box(368, 76, 66, 30, 'SDPA', 'green', False, 14)
    f.text(401, 70, 'K,V: 2 × 256', 12, 'blue')
    f.arrow(436, 91, 462, 104, 'green')
    f.arrow(340, 129, 462, 112, 'teal')
    f.circle(476, 108, 13, '⊙', 'green', False, 18)
    f.arrow(490, 108, 514, 108, 'green')
    f.box(516, 92, 86, 32, 'o_proj', 'green', False, 14)
    f.text(559, 140, '→ 2,048', 13, 'muted')
    f.text(476, 86, '4,096', 12, 'muted')
    # zoom: one head, 12 channels (toy)
    f.text(30, 178, 'One head, 12 of 256 channels (toy)', 14, 'ink', True, 'start')
    Y = [.8, -.6, .5, .9, -.3, .7, -.8, .4, .6, -.5, .3, .9]
    G = [.03, .92, .05, .01, .08, .71, .02, .04, .88, .02, .06, .03]
    rows = [('Y', Y, 'green'), ('σ(XW_θ)', G, 'teal'), ("Y' = Y \\odot σ", [a*b for a, b in zip(Y, G)], 'green')]
    for r, (lab, vals, col) in enumerate(rows):
        y = 192 + r*38
        f.math(130, y+20, lab, 15, col, 'end', True)
        for c, v in enumerate(vals):
            x = 142 + c*36
            op = min(1, abs(v)) if col == 'green' else v
            f.rect(x, y, 32, 28, col, round(max(.06, op*.9), 2), 3)
            f.text(x+16, y+19, f'{v:+.2f}' if col == 'green' else f'{v:.2f}', 11, 'white' if op > .55 else 'ink')
    f.text(142, 326, 'mean gate 0.116 (reported): most channels shut per query', 13, 'teal', anchor='start')
    # right: reported numbers
    f.divider(588, 60, 336)
    f.text(606, 72, 'Reported · 15B MoE, 400B tokens', 14, 'ink', True, 'start')
    stats = [('first-token attention', 46.7, 4.8, '%'), ('max activation', 1053, 94, ''), ('perplexity', 6.026, 5.761, ''), ('RULER 128K (YaRN)', 31.65, 58.82, '')]
    for k, (lab, a, b, u) in enumerate(stats):
        y = 92 + k*60
        f.text(606, y+12, lab, 13, 'muted', anchor='start')
        m = max(a, b)
        f.rect(606, y+18, 170*a/m, 12, 'red', .7, 2)
        f.text(606+170*a/m+6, y+29, f'{a:g}{u}', 12, 'red', True, 'start')
        f.rect(606, y+33, 170*b/m, 12, 'green', .85, 2)
        f.text(606+170*b/m+6, y+44, f'{b:g}{u} gated', 12, 'green', True, 'start')
    # bottom: gate location
    f.hdivider(346, 20, 880)
    f.text(30, 372, 'Gate location', 15, 'ink', True, 'start')
    f.math(30, 400, r'\text{output gate: } \c{green}{y_t} = \c{teal}{g_t} \odot \c{green}{o_t}', 16)
    f.math(30, 432, r'\text{state gate: } \c{purple}{S_t} = \c{teal}{α_t} \c{purple}{S_{t-1}} + \c{purple}{v_t}\c{blue}{k_t}^{\top}', 16)
    for k in range(8):
        x = 360 + k*34
        f.rect(x, 384, 28, 22, 'green' if k == 4 else 'line', .85 if k == 4 else .5, 3)
        f.rect(x, 416, 28, 22, 'purple' if k >= 4 else 'line', .85 if k >= 4 else .5, 3)
    f.text(360+4*34+14, 380, 't', 13, 'ink', True)
    f.text(640, 400, 'changes this token’s read only', 13, 'green', True, 'start')
    f.text(640, 432, 'changes every later read', 13, 'purple', True, 'start')
    f.save()


# --------------------------------------------------------------------------
def knobs():
    f = Fig('stable-knobs', 'Left: decay bias multiplying attention weights, toy 10 tokens. ALiBi-like constant forget gate f = 0.8 gives the same smooth decay in every row; FoX with data-dependent gates (f = 0.97 except f = 0.05 at token 5) keeps the past until token 5 and then wipes everything before it. Right: differential attention, toy 10 keys: softmax 1 puts 0.46 on key 6 and 0.06 on each other key; lambda times softmax 2 with lambda = 0.6 is 0.06 everywhere; the difference keeps 0.40 on key 6 and zero elsewhere. lambda_init = 0.8 - 0.6 exp(-0.3(l - 1)) is 0.20 at layer 1, 0.56 at layer 4 and 0.79 at layer 16.', 452)
    strip(f, 30, 36, [r'\text{FoX: } e_{tj} = \c{gold}{q_t}^{\top}\c{blue}{k_j}/\sqrt d_h + \c{teal}{c_t - c_j}', r'\c{teal}{c_t} = Σ_{l \le t} \ln \c{teal}{f_l}'], 20, 30)
    strip(f, 30, 66, [r'\text{Diff: } \c{green}{o_t} = (\text{softmax}(e^{(1)}) - λ\,\text{softmax}(e^{(2)}))\,\c{purple}{V}'], 20, 30)
    n, cell = 10, 18
    alibi_f = [.8]*n
    fox_f = [.97]*n
    fox_f[5] = .05

    def decay(fs):
        c = [0.0]
        for t in range(1, n):
            c.append(c[-1] + math.log(fs[t]))
        return lambda i, j: ('teal', round(max(.05, math.exp(c[i]-c[j])*.9), 2)) if j <= i else None
    f.text(30, 102, 'ALiBi-like: constant f = 0.8', 14, 'ink', True, 'start')
    f.mask(40, 116, n, cell, decay(alibi_f))
    f.text(250, 102, 'FoX: learned f per token', 14, 'ink', True, 'start')
    f.mask(256, 116, n, cell, decay(fox_f))
    for t in range(n):
        f.text(256+t*cell+8, 116+n*cell+16, f'{fox_f[t]:.2f}'[1:] if t else '', 10, 'red' if t == 5 else 'muted')
    f.math(250, 116+n*cell+34, r'f_t \text{ (toy): } f_5 = 0.05 \text{ wipes tokens } < 5', 14, 'red')
    f.text(40, 116+n*cell+16, 'same decay in every row', 13, 'teal', anchor='start')
    f.math(40, 116+n*cell+34, r'\text{shade} = exp(c_t - c_j)', 14, 'muted')
    # right: Diff
    f.divider(470, 84, 420)
    f.text(488, 102, 'Differential attention (toy, 10 keys)', 14, 'ink', True, 'start')
    s1 = [.06]*n
    s1[6] = .46
    s2 = [.1]*n
    lam = .6
    rows = [('softmax₁', s1, 'blue'), ('λ·softmax₂', [lam*v for v in s2], 'red'), ('difference', [a-lam*b for a, b in zip(s1, s2)], 'green')]
    for r, (lab, vals, col) in enumerate(rows):
        base = 170 + r*70
        f.text(572, base-2, lab, 13, col, True, 'end')
        for k, v in enumerate(vals):
            h = v*100
            f.rect(580+k*20, base-h, 15, max(.8, h), col, .8, 2)
        f.line(576, base, 580+n*20, base, 'muted', 1)
        f.text(580+6*20+7, base-vals[6]*100-6, f'{vals[6]:.2f}', 12, col, True)
    f.text(580+n*20+10, 236, 'λ = 0.6 (toy)', 13, 'red', anchor='start')
    f.text(580+n*20+10, 306, 'noise cancels', 13, 'green', True, 'start')
    f.hdivider(338, 480, 880)
    f.math(488, 362, r'λ_{\text{init}} = 0.8 - 0.6\,e^{-0.3(\ell-1)} \text{ by layer}', 15)
    for k, l in enumerate((1, 2, 4, 8, 16)):
        v = .8 - .6*math.exp(-.3*(l-1))
        x = 500 + k*76
        f.rect(x, 426-v*56, 30, v*56, 'red', .7, 2)
        f.text(x+15, 444, f'ℓ = {l}', 12, 'muted')
        f.text(x+15, 426-v*56-6, f'{v:.2f}', 12, 'red', True)
    f.save()


# --------------------------------------------------------------------------
STAGES = [('$W_Q · W_{KV}', 'gold'), ('QK-norm', 'gold'), ('RoPE (part)', 'teal'), ('mask', 'blue'), ('softmax', 'blue'),
          ('· V', 'purple'), ('⊙ σ gate', 'teal'), ('$W_O', 'green')]
ROWS = [
    ('gpt-oss-120b', [('8 KV × 64', 'GQA, bias'), None, ('full · 150K', 'YaRN ×32'), ('128 ⇄ full', '1 : 1'), ('+ sink', 'per head'), None, None, None]),
    ('Gemma 4 31B', [('global: K=V', '4 × 512'), ('q, k RMS', 'scale 1.0'), ('global ¼ · 1M', 'local 10K'), ('1,024 ⇄ full', '5 : 1'), None, ('V-norm', 'no gain'), None, None]),
    ('DeepSeek-V4-Pro', [('K=V · MQA', '1 × 512'), ('q RMS', '+ kv-norm'), ('64 of 512', '10K | 160K'), ('128 + comp.', 'entries'), ('+ sink', 'per head'), ('de-rotate', 'output'), None, ('16 groups', 'low-rank')]),
    ('Qwen3-Next-80B', [('2 KV × 256', 'GQA'), ('q, k RMS', 'per head'), ('¼ of 256', '10M'), ('full', '1 in 4 layers'), None, None, ('σ gate', 'elementwise'), None]),
]


def anatomy():
    f = Fig('stable-anatomy', 'A 2026 attention block as eight stages, with what four released models put at each stage, from configs and code. gpt-oss-120b: 8 KV heads of 64, full RoPE base 150K with YaRN x32, 128-token window alternating 1:1 with full layers, learned sink per head. Gemma 4 31B: K=V on global layers (4 x 512), q and k RMS-norm with scale 1.0, p-RoPE 1/4 at base 1M on global layers and 10K on local, window 1,024 at 5:1, V-norm. DeepSeek-V4-Pro: K=V single 512-d MQA entry, q RMS and kv-norm, last 64 of 512 dims rotated, window 128 plus compressed entries, sink per head, output de-rotation, 16-group low-rank output projection. Qwen3-Next-80B: 2 KV heads of 256, per-head q and k RMS-norm, 1/4 of 256 dims rotated at base 10M, full attention in 1 of 4 layers, elementwise sigmoid output gate.', 440)
    f.math(30, 34, r'y = W_O[\c{teal}{σ(XW_θ)} \odot \text{softmax}_{j ∪ \c{teal}{\text{sink}}}(\text{RoPE}_p(\c{gold}{q})^{\top}\text{RoPE}_p(\c{blue}{k}) / \sqrt d_h + m^{W})\,\c{purple}{v}]', 20)
    x0, w, gap = 168, 82, 8
    centers = []
    for k, (lab, col) in enumerate(STAGES):
        x = x0 + k*(w+gap)
        centers.append(x + w/2)
        f.box(x, 58, w, 34, lab, col, True, 13, .9, 5)
        if k:
            f.arrow(x-gap+1, 75, x-1, 75, 'muted', width=1.2)
    f.text(30, 80, 'stage →', 14, 'muted', True, 'start')
    for r, (name, cells) in enumerate(ROWS):
        y = 112 + r*78
        f.rect(20, y, 860, 68, 'soft' if r % 2 == 0 else 'white', 1, 6)
        f.text(30, y+40, name, 15, 'ink', True, 'start')
        for k, cell in enumerate(cells):
            cx = centers[k]
            if cell is None:
                f.text(cx, y+40, '—', 15, 'line', True)
                continue
            a, b = cell
            col = 'green' if k in (4, 6) or 'K=V' in a else 'teal' if k == 2 else 'ink'
            f.text(cx, y+30, a, 13, col, True)
            f.text(cx, y+48, b, 12, 'muted')
    f.text(30, 432, 'Sink or gate, never both here · K=V comes with partial RoPE · every row but gpt-oss normalises q (and k)', 13, 'muted', anchor='start')
    f.save()


def build():
    qk_norm()
    sinks()
    gated()
    knobs()
    anatomy()
