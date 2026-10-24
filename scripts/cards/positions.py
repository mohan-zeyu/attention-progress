"""01 · Positions: RoPE spectrum, context extension, partial RoPE, NoPE layers, length temperature."""
import math
from figkit import Fig

TAU = 2*math.pi


def lam(base, d, i):
    """Wavelength (tokens) of RoPE pair i for head dim d."""
    return TAU*base**(2*i/d)


def ymap(v, lo, hi, top, bottom):
    """log10 value v in [lo, hi] -> y (top = hi)."""
    return bottom - (math.log10(v)-lo)/(hi-lo)*(bottom-top)


def xmap(v, lo, hi, left, right):
    return left + (math.log10(v)-lo)/(hi-lo)*(right-left)


def fmt_tok(v):
    if v >= 1e6:
        return f'{v/1e6:.1f}M'.replace('.0M', 'M')
    if v >= 1e5:
        return f'{v/1e3:.0f}K'
    if v >= 1e3:
        return f'{v/1e3:.1f}K'.replace('.0K', 'K')
    return f'{v:.1f}' if v < 10 else f'{v:.0f}'


def strip(f, x, y, parts, size=21, gap=36):
    """Formula strip made of several expressions separated by real gaps (SVG collapses spaces)."""
    for part in parts:
        x += f.math(x, y, part, size) + gap


def dial(f, cx, cy, r, deg, color='teal'):
    """A clock dial: hand at `deg` degrees clockwise from 12 o'clock."""
    f.parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="white" stroke="#2a9aa6" stroke-width="2"/>')
    for k in range(12):
        a = math.radians(k*30)
        f.line(cx+(r-5)*math.sin(a), cy-(r-5)*math.cos(a), cx+r*math.sin(a), cy-r*math.cos(a), 'line', 1.2)
    a = math.radians(deg)
    if deg > 1:
        large = 1 if deg > 180 else 0
        ex, ey = cx+(r-8)*math.sin(a), cy-(r-8)*math.cos(a)
        f.parts.append(f'<path d="M{cx} {cy}L{cx} {cy-(r-8)}A{r-8} {r-8} 0 {large} 1 {ex:.1f} {ey:.1f}Z" fill="#2a9aa6" fill-opacity=".18"/>')
    f.line(cx, cy, cx+(r-6)*math.sin(a), cy-(r-6)*math.cos(a), color, 2.6)
    f.line(cx, cy, cx, cy-(r-6), 'line', 1, dash=True)
    f.circle(cx, cy, 3, color=color, fill=True)


# --------------------------------------------------------------------------
def rope_spectrum():
    f = Fig('pos-rope-spectrum', 'RoPE pair i rotates by theta_i = b^(-2i/d_h) per token. Left: four pairs of Llama 3 (b = 500K, d_h = 128) over 1,000 tokens: pair 0 turns 159 times, pair 21 turns 2.1 times, pair 42 turns 10 degrees, pair 63 turns 0.14 degrees. Right: wavelength of all 64 pairs for b = 10K, 500K and 10M on a log axis with 8K and 128K context lines; the slowest pair spans 54K, 2.6M and 49M tokens. Bottom: Gemma 3 and DeepSeek-V4 use a small base in local layers and a large base in global layers.', 450)
    strip(f, 30, 36, [r'\c{teal}{θ_i} = b^{-2i/d_h}', r'\c{teal}{λ_i} = 2π / \c{teal}{θ_i}', r'(R_t\,\c{gold}{q})^{\top}(R_j\,\c{blue}{k}) = \c{gold}{q}^{\top} R_{j-t}\,\c{blue}{k}'], 21, 44)
    # left: dials
    f.panel(20, 56, 300, 300, 'Each pair is a clock')
    f.text(34, 98, 'Llama 3 · b = 500K · d_h = 128', 13, 'muted', anchor='start')
    f.text(34, 116, 'angle swept over Δ = 1,000 tokens', 13, 'muted', anchor='start')
    base, d = 5e5, 128
    for k, i in enumerate((0, 21, 42, 63)):
        cx, cy = 92 + (k % 2)*156, 172 + (k//2)*112
        th = base**(-2*i/d)
        ang = 1000*th
        dial(f, cx, cy, 30, math.degrees(ang) % 360)
        turns = ang/TAU
        shown = f'{turns:.0f} turns' if turns >= 10 else (f'{turns:.1f} turns' if turns >= 1 else f'{math.degrees(ang):.2g}°')
        f.text(cx+38, cy-6, f'pair {i}', 13, 'ink', True, 'start')
        f.text(cx+38, cy+11, f'λ = {fmt_tok(lam(base, d, i))}', 13, 'muted', anchor='start')
        f.text(cx+38, cy+28, shown, 13, 'teal', True, 'start')
    # right: wavelength plot
    L0, L1, T, B = 400, 770, 76, 326
    lo, hi = 0.5, 8.0
    f.math(L0-8, 66, r'λ_i \text{ (tokens, log)}', 14, 'muted')
    for e in range(1, 9):
        y = ymap(10**e, lo, hi, T, B)
        f.line(L0, y, L1, y, 'line', .6, opacity=.6)
        f.text(L0-8, y+4, fmt_tok(10**e) if e < 3 else fmt_tok(10**e), 13, 'muted', anchor='end')
    f.line(L0, B, L1, B, 'muted', 1.2)
    f.line(L0, T, L0, B, 'muted', 1.2)
    for i in (0, 16, 32, 48, 63):
        x = L0 + i/63*(L1-L0)
        f.line(x, B, x, B+5, 'muted', 1.2)
        f.text(x, B+20, str(i), 13, 'muted')
    f.text((L0+L1)/2, B+38, 'pair index i  (fast → slow)', 13, 'muted')
    # context lines
    for ctx, lab in ((8192, '8K · Llama 3 pre-train'), (131072, '128K')):
        y = ymap(ctx, lo, hi, T, B)
        f.line(L0, y, L1, y, 'red', 1.3, dash=True)
        f.text(L0+6, y-5, lab, 13, 'red', anchor='start')
    curves = ((1e4, 'muted', 1.8, False, '10K · Llama 2'), (5e5, 'teal', 2.8, False, '500K · Llama 3'), (1e7, 'teal', 1.8, True, '10M · Seed-OSS'))
    for b, c, w, dash, lab in curves:
        pts = [(round(L0 + i/63*(L1-L0), 1), round(ymap(lam(b, 128, i), lo, hi, T, B), 1)) for i in range(64)]
        f.path(pts, c, width=w, dash=dash)
        y_end = pts[-1][1]
        f.text(L1+8, y_end+4, fmt_tok(lam(b, 128, 63)), 14, c, True, 'start')
        f.text(L1+8, y_end+20, lab, 13, c, anchor='start')
    # brace: pairs slower than 8K for b = 500K
    first = next(i for i in range(64) if lam(5e5, 128, i) > 8192)
    xa = L0 + first/63*(L1-L0)
    f.brace(xa, B-4, L1, B-4, '', 'teal', 'above', 13, 4)
    f.text((xa+L1)/2, B-34, f'500K: {64-first} pairs < 1 turn in 8K', 13, 'teal', True)
    f.text((xa+L1)/2, B-18, '→ act as content channels', 13, 'teal')
    # bottom: dual bases
    f.hdivider(372, 20, 880)
    f.text(30, 398, 'Two bases in one model', 15, 'ink', True, 'start')
    f.pill(232, 398, 'Gemma 3 local ×5: b = 10K, W = 1,024', 'teal', 13)
    f.pill(552, 398, 'global ×1: b = 1M, PI ×8 → 128K', 'teal', 13)
    f.pill(232, 430, 'DeepSeek-V4 SWA layers: b = 10K, no YaRN', 'teal', 13)
    f.pill(578, 430, 'compressed layers: b = 160K + YaRN ×16', 'teal', 13)
    f.save()


# --------------------------------------------------------------------------
def extension():
    f = Fig('pos-extension', 'Context extension divides each RoPE frequency by a per-pair factor. For gpt-oss (d_h = 64, b = 150K, trained at 4,096, extended to 131,072, s = 32): PI divides all 32 pairs by 32; NTK-aware rises smoothly from 1 to 32; YaRN keeps pairs 0-8 (wavelength under 128 tokens), ramps pairs 9-17 and divides pairs 18-31 by 32. Llama 3.1 llama3 scaling (s = 8) keeps 29 pairs, blends 6 and divides 29. YaRN also multiplies logits by (0.1 ln s + 1)^2: 1.30 for s = 4, 1.46 for 8, 1.81 for 32, 1.87 for 40; llama3 type uses 1.', 460)
    strip(f, 30, 36, [r"\c{teal}{θ'_i} = (1-γ_i)\,\c{teal}{θ_i}/s + γ_i\,\c{teal}{θ_i}", r'γ_i = \text{clip}((L/λ_i - α)/(β - α), 0, 1)', r'\text{logits} \times (0.1 \ln s + 1)^2'], 20, 30)
    f.text(30, 68, 'gpt-oss · d_h = 64 · b = 150K · L = 4,096 → 131,072 · s = 32 · α = 1, β = 32', 14, 'muted', anchor='start')
    L0, L1, T, B = 92, 505, 110, 336
    d, b, L, s = 64, 150000, 4096, 32

    def yv(v):
        return B - math.log2(v)/5*(B-T)

    def xv(i):
        return L0 + i/31*(L1-L0)
    for v in (1, 2, 4, 8, 16, 32):
        f.line(L0, yv(v), L1, yv(v), 'line', .6, opacity=.6)
        f.text(L0-8, yv(v)+4, f'÷{v}', 13, 'muted', anchor='end')
    f.line(L0, B, L1, B, 'muted', 1.2)
    f.line(L0, T, L0, B, 'muted', 1.2)
    f.math(L0-8, T-14, r"θ_i / θ'_i \text{ per pair}", 14, 'muted')
    for i in (0, 8, 17, 31):
        f.line(xv(i), B, xv(i), B+5, 'muted', 1.2)
        f.text(xv(i), B+20, str(i), 13, 'muted')
        f.text(xv(i), B+37, 'λ ' + fmt_tok(lam(b, d, i)), 13, 'teal')
    f.text(L1+10, B+20, 'pair i', 13, 'muted', anchor='start')
    # ramp zone shading
    f.rect(xv(8.5), T, xv(17.5)-xv(8.5), B-T, 'teal', .07)
    f.text(xv(0)+8, yv(3.4), 'kept: λ < L/β', 13, 'teal', True, 'start')
    f.text(xv(13), T+16, 'ramp', 13, 'teal', True)
    f.text(xv(24.5), B-60, '÷32: λ > L = 4,096', 13, 'teal', True)
    # PI
    f.path([(xv(0), yv(32)), (xv(31), yv(32))], 'red', width=2.2)
    f.text(xv(0)+6, yv(32)+18, 'PI: every pair ÷32', 13, 'red', True, 'start')
    f.text(xv(0)+6, yv(32)+34, '(blurs fast pairs)', 13, 'red', anchor='start')
    # NTK-aware
    pts = [(round(xv(i), 1), round(yv(s**(2*i/(d-2))), 1)) for i in range(32)]
    f.path(pts, 'muted', width=2, dash=True)
    f.math(xv(17)+12, yv(s**(34/62))+26, r"\text{NTK-aware: } b' = b \cdot s^{d_h/(d_h-2)}", 14, 'muted')
    # YaRN
    yarn = []
    for i in range(32):
        r = L/lam(b, d, i)
        g = min(1, max(0, (r-1)/(32-1)))
        yarn.append(1/((1-g)/s + g))
    pts = [(round(xv(i), 1), round(yv(v), 1)) for i, v in enumerate(yarn)]
    f.path(pts, 'teal', width=3)
    for x, y in pts:
        f.circle(x, y, 2.6, color='teal', fill=True)
    f.callout(xv(10), yv(yarn[10]), xv(11), yv(1.45), 'YaRN (one dot per pair)', 'teal', 14)
    # right: per-pair strips
    f.divider(560, 56, 440)
    f.text(580, 96, 'Which pairs move (per-pair strip)', 15, 'ink', True, 'start')
    f.text(580, 122, 'gpt-oss · YaRN s = 32 · 32 pairs', 13, 'muted', anchor='start')
    for i, v in enumerate(yarn):
        op = 1 - .8*math.log2(v)/5
        f.rect(580+i*9, 130, 8, 22, 'teal', round(op, 2), 1)
    f.text(580, 170, '9 kept · 9 ramp · 14 ÷32', 13, 'teal', True, 'start')
    f.text(580, 198, 'Llama 3.1 · llama3 s = 8 · 64 pairs', 13, 'muted', anchor='start')
    d2, b2, L2, s2 = 128, 5e5, 8192, 8
    for i in range(64):
        l2 = lam(b2, d2, i)
        sm = min(1, max(0, (L2/l2 - 1)/(4-1)))
        v = 1/((1-sm)/s2 + sm)
        op = 1 - .8*math.log2(v)/3
        f.rect(580+i*4.5, 206, 3.8, 22, 'teal', round(op, 2), 1)
    f.text(580, 246, '29 kept · 6 blended · 29 ÷8 · α = 1, β = 4', 13, 'teal', True, 'start')
    f.hdivider(268, 570, 880)
    f.math(580, 296, r'\text{logit multiplier } (0.1 \ln s + 1)^2', 15)
    rows = [('Qwen3 · s = 4', 1.296, 'teal'), ('OLMo 3 · s = 8', 1.459, 'teal'), ('gpt-oss · s = 32', 1.813, 'teal'),
            ('DeepSeek-V3 · s = 40', 1.874, 'teal'), ('Llama 3.1 · llama3', 1.0, 'line')]
    f.bars(570, 308, rows, width=110, bar=16, gap=7, label_w=160, size=13, vmax=1.874, value_fmt=lambda v: f'×{v:.2f}')
    f.save()


# --------------------------------------------------------------------------
def partial():
    f = Fig('pos-partial', 'Rotated (teal) versus position-free (grey) dimensions per head, from configs: GPT-NeoX-20B 24 of 96, GLM-4.5 64 of 128, MiniMax-M2 64 of 128, Phi-4-mini 96 of 128, MiMo-V2-Flash 64 of 192, Qwen3-Next 64 of 256, Gemma 4 global 128 of 512, DeepSeek-V4 64 of 512, DeepSeek-V3 MLA lane 64 of 192. Right: at the same 25 percent, NeoX-style partial RoPE in Qwen3-Next keeps wavelengths 6 to 38M tokens, while p-RoPE in Gemma 4 keeps only 6 to 188 tokens and sets 192 pairs to zero frequency. Bottom right: reported Gemma 2B perplexity at 8K: NoPE 4.86, RoPE 4.46, 0.25-RoPE 4.53, 0.75-RoPE 4.44.', 450)
    strip(f, 30, 36, [r'\c{gold}{q}^{\top}\c{blue}{k} \to (R_t\,\c{gold}{q_{rot}})^{\top}(R_j\,\c{blue}{k_{rot}}) + \c{gold}{q_{nope}}^{\top}\c{blue}{k_{nope}}', r'd_{rot} = p \cdot d_h'], 21, 44)
    rows = [('GPT-NeoX-20B', 96, 24, 'L'), ('GLM-4.5', 128, 64, 'L'), ('MiniMax-M2', 128, 64, 'L'), ('Phi-4-mini', 128, 96, 'L'),
            ('MiMo-V2-Flash', 192, 64, 'L'), ('Qwen3-Next', 256, 64, 'L'), ('Gemma 4 global · p-RoPE', 512, 128, 'L'),
            ('DeepSeek-V4 (K=V)', 512, 64, 'R'), ('DeepSeek-V3 MLA lane', 192, 64, 'R')]
    x0, sc = 196, .58
    f.rect(30, 63, 12, 12, 'teal', .85, 2)
    f.text(48, 74, 'rotated', 13, 'teal', True, 'start')
    f.box(112, 63, 12, 12, '', 'line', False, rx=2, stroke=1.2)
    f.text(130, 74, 'position-free dims · rotated/head dim (config)', 13, 'muted', anchor='start')
    for k, (name, dh, dr, side) in enumerate(rows):
        y = 86 + k*36
        f.text(x0-10, y+16, name, 14, 'ink', anchor='end')
        w, wr = dh*sc, dr*sc
        f.box(x0, y, w, 22, '', 'line', False, rx=3, stroke=1.2)
        xr = x0 if side == 'L' else x0 + w - wr
        f.rect(xr, y, wr, 22, 'teal', .85, 3)
        if k == 8:
            f.line(x0+w-wr-1.5, y-3, x0+w-wr-1.5, y+25, 'ink', 1.6)
        f.text(x0+w+8, y+16, f'{dr}/{dh}', 14, 'teal', True, 'start')
    # right: spectrum comparison
    f.divider(560, 56, 440)
    f.text(578, 74, 'Same p = 0.25, different spectrum', 15, 'ink', True, 'start')
    A0, A1 = 600, 850
    lo, hi = 0.5, 8.0
    axy = 186
    for e in (1, 3, 5, 7):
        x = xmap(10**e, lo, hi, A0, A1)
        f.line(x, axy-4, x, axy+4, 'muted', 1.2)
        f.text(x, axy+20, fmt_tok(10**e), 13, 'muted')
    f.line(A0, axy, A1, axy, 'muted', 1.2)
    f.text(A1+6, axy+20, 'λ', 13, 'muted', anchor='start')
    f.text(578, 100, 'Qwen3-Next · partial · b = 10M', 13, 'muted', anchor='start')
    for i in range(32):
        f.circle(round(xmap(lam(1e7, 64, i), lo, hi, A0, A1), 1), 116, 3, color='teal', fill=True)
    f.text(578, 144, 'Gemma 4 · p-RoPE · b = 1M', 13, 'muted', anchor='start')
    for i in range(64):
        f.circle(round(xmap(lam(1e6, 512, i), lo, hi, A0, A1), 1), 160, 3, color='teal', fill=True)
    xg = xmap(188, lo, hi, A0, A1)
    f.text(xg+8, 165, 'max λ ≈ 188', 13, 'teal', True, 'start')
    f.box(770, 151, 82, 18, '192 pairs θ = 0', 'line', False, 11, rx=3)
    xc = xmap(262144, lo, hi, A0, A1)
    f.line(xc, 106, xc, axy, 'red', 1.3, dash=True)
    f.text(xc, 228, '256K context', 13, 'red', anchor='middle')
    # bottom right: Barbero ppl
    f.hdivider(240, 570, 880)
    f.text(578, 264, 'Gemma 2B · 8K · Wiki ppl (reported)', 14, 'ink', True, 'start')
    rows = [('NoPE', 4.859, 'red'), ('RoPE', 4.463, 'teal'), ('0.25-RoPE', 4.530, 'teal'), ('0.75-RoPE', 4.441, 'green')]
    f.bars(570, 276, [(n, v-4.4, c) for n, v, c in rows], width=150, bar=18, gap=9, label_w=90, size=13,
           value_fmt=lambda v: f'{v+4.4:.3f}')
    f.text(578, 398, 'bars start at 4.40 · lower is better', 13, 'muted', anchor='start')
    f.text(578, 420, 'p-RoPE: Barbero et al., ICLR 2025, Table 2', 13, 'muted', anchor='start')
    f.save()


# --------------------------------------------------------------------------
def nope_layers():
    f = Fig('pos-nope', 'Llama 4 interleaves three chunked RoPE layers (chunk 8,192) with one full-attention NoPE layer. Toy masks with 16 tokens and chunk 4. Layer strips: Llama 4 Scout 36 RoPE plus 12 NoPE of 48 layers; SmolLM3-3B 27 RoPE plus 9 NoPE of 36, all full attention; Command A pattern three sliding-window RoPE layers (W = 4,096) then one full NoPE layer. KV cache of Llama 4 Scout at 128K in BF16: 7.1 GiB with iRoPE versus 24 GiB if all 48 layers were global.', 430)
    strip(f, 30, 36, [r'\c{teal}{\text{RoPE layer: }} (R_t\,\c{gold}{q})^{\top}(R_j\,\c{blue}{k}) + m_{\text{chunk}}', r'\c{blue}{\text{NoPE layer: }} \c{gold}{q}^{\top}\c{blue}{k} + m_{\text{causal}}'], 21, 50)
    f.text(30, 70, 'Llama 4 block of 4 layers · toy: N = 16, chunk = 4 (real chunk 8,192)', 14, 'muted', anchor='start')
    n, cell = 16, 8
    for k in range(4):
        x = 30 + k*148
        nope = k == 3
        if nope:
            fn = (lambda i, j: ('blue', .8) if j <= i else None)
        else:
            fn = (lambda i, j: ('teal', .85) if j <= i and i//4 == j//4 else None)
        f.mask(x, 86, n, cell, fn)
        f.text(x+64, 236, f'L{k+1} · ' + ('NoPE · full' if nope else 'RoPE · chunk'), 13, 'blue' if nope else 'teal', True)
    f.text(30+3*148+64, 254, 'long-range retrieval', 13, 'blue')
    f.text(30+64+148, 254, 'local order, recency', 13, 'teal')
    # KV readout
    f.divider(624, 56, 262)
    f.text(640, 84, 'Llama 4 Scout KV @128K, BF16', 14, 'ink', True, 'start')
    f.text(640, 104, '8 KV heads × 128 × 2 × 2 B', 13, 'muted', anchor='start')
    f.text(640, 121, '= 4 KiB per token per layer', 13, 'muted', anchor='start')
    f.bars(634, 136, [('all global', 24.0, 'red'), ('iRoPE', 7.125, 'green')], width=110, bar=22, gap=12, label_w=76, size=14,
           value_fmt=lambda v: f'{v:.1f} GiB')
    f.text(640, 220, '12 NoPE × 131,072 tok = 6.0 GiB', 13, 'blue', anchor='start')
    f.text(640, 238, '36 RoPE × 8,192 tok = 1.1 GiB', 13, 'teal', anchor='start')
    f.hdivider(276)
    # layer strips

    def layer_row(y, name, pattern, note):
        f.text(30, y+17, name, 14, 'ink', True, 'start')
        for i, p in enumerate(pattern):
            f.rect(250+i*11, y, 9, 24, 'teal' if p else 'blue', .85, 2)
        f.text(250+len(pattern)*11+8, y+17, note, 13, 'muted', anchor='start')
    layer_row(296, 'Llama 4 Scout · 48', [1, 1, 1, 0]*12, '36 + 12')
    layer_row(336, 'SmolLM3-3B · 36', [1, 1, 1, 0]*9, '27 + 9, all full attention')
    layer_row(376, 'Command A pattern', [1, 1, 1, 0], 'RoPE layers SWA W = 4,096 · NoPE full · QK-norm removed')
    f.rect(250, 414, 11, 11, 'teal', .85, 2)
    f.text(266, 424, 'RoPE layer', 13, 'teal', anchor='start')
    f.rect(360, 414, 11, 11, 'blue', .85, 2)
    f.text(376, 424, 'NoPE layer (no rotation)', 13, 'blue', anchor='start')
    f.save()


# --------------------------------------------------------------------------
def temperature():
    f = Fig('pos-temperature', 'Left, toy: one key 3 logits above n - 1 equal keys. Plain softmax top weight falls from 0.019 at 1K keys to 2e-6 at 10M; Llama 4 temperature slows the fall; SSMax with s ln n keeps it near 0.8. Right, real configs: the Llama 4 query multiplier 1 + 0.1 ln(1 + floor((t+1)/8192)) is 1.07 at 8K, 1.28 at 128K, 1.49 at 1M and 1.72 at 10M; YaRN applies a constant logit multiplier of 1.81 (gpt-oss, s = 32) or 1.87 (DeepSeek-V3, s = 40).', 420)
    strip(f, 30, 36, [r'\c{gold}{q_t} \leftarrow \c{teal}{τ(t)}\,\c{gold}{q_t}', r'\c{teal}{τ}_{\text{Llama 4}} = 1 + 0.1 \ln(1 + ⌊(t+1)/8192⌋)', r'\c{teal}{τ}_{\text{SSMax}} = s_h \ln n'], 21, 40)
    # left plot
    L0, L1, T, B = 92, 420, 100, 330
    f.text(30, 72, 'Attention fading · toy: one key Δ = 3 above n − 1 equal keys', 14, 'ink', True, 'start')
    xlo, xhi = 3, 7
    ylo, yhi = -6, 0
    for e in range(ylo, yhi+1):
        y = B - (e-ylo)/(yhi-ylo)*(B-T)
        f.line(L0, y, L1, y, 'line', .6, opacity=.6)
        f.text(L0-8, y+4, '1' if e == 0 else f'1e{e}', 13, 'muted', anchor='end')
    for e in range(xlo, xhi+1):
        x = L0 + (e-xlo)/(xhi-xlo)*(L1-L0)
        f.text(x, B+20, fmt_tok(10**e), 13, 'muted')
    f.line(L0, B, L1, B, 'muted', 1.2)
    f.line(L0, T, L0, B, 'muted', 1.2)
    f.math(L0-8, T-12, r'\text{top weight } p_{max}', 14, 'muted')
    f.text((L0+L1)/2, B+38, 'keys n (log)', 13, 'muted')
    delta = 3.0

    def pmax(z, n):
        return math.exp(z)/(math.exp(z)+n-1)

    def pt(n, p):
        return (round(L0 + (math.log10(n)-xlo)/(xhi-xlo)*(L1-L0), 1), round(B - (math.log10(p)-ylo)/(yhi-ylo)*(B-T), 1))
    ns = [10**(xlo + k*(xhi-xlo)/200) for k in range(201)]
    plain = [pt(n, pmax(delta, n)) for n in ns]
    llama = [pt(n, pmax(delta*(1+.1*math.log1p(math.floor(n/8192))), n)) for n in ns]
    ss = [pt(n, pmax(.4*math.log(n)*delta, n)) for n in ns]
    f.path(plain, 'red', width=2.4)
    f.path(llama, 'teal', width=2.4)
    f.path(ss, 'green', width=2.4)
    k = 150
    f.text(plain[k][0]-6, plain[k][1]+22, 'plain softmax ∝ 1/n', 13, 'red', True, 'end')
    f.text(llama[k][0]+4, llama[k][1]-14, 'Llama 4 τ', 13, 'teal', True, 'start')
    f.text(ss[-1][0]-4, ss[-1][1]+22, 'SSMax · toy s = 0.4', 13, 'green', True, 'end')
    f.text(plain[0][0]+10, plain[0][1]-6, f'{pmax(delta, 1000):.3f}', 13, 'red', True, 'start')
    f.text(ss[0][0]+8, ss[0][1]+20, f'{pmax(.4*math.log(1000)*delta, 1000):.2f}', 13, 'green', True, 'start')
    f.text(plain[-1][0]+4, plain[-1][1]+4, '2e-6', 13, 'red', True, 'start')
    # right plot: real multipliers
    f.divider(462, 56, 400)
    R0, R1 = 530, 860
    f.text(480, 72, 'Logit multiplier vs position (configs)', 14, 'ink', True, 'start')

    def rx(t):
        return R0 + (math.log10(t)-3)/4*(R1-R0)

    def ry(v):
        return B - (v-1)/1.0*(B-T)
    for v in (1.0, 1.25, 1.5, 1.75, 2.0):
        f.line(R0, ry(v), R1, ry(v), 'line', .6, opacity=.6)
        f.text(R0-8, ry(v)+4, f'×{v:.2f}', 13, 'muted', anchor='end')
    for e in range(3, 8):
        f.text(rx(10**e), B+20, fmt_tok(10**e), 13, 'muted')
    f.line(R0, B, R1, B, 'muted', 1.2)
    f.line(R0, T, R0, B, 'muted', 1.2)
    f.text((R0+R1)/2, B+38, 'position t (log)', 13, 'muted')
    for v, lab in ((1.813, 'YaRN gpt-oss s = 32: ×1.81'), (1.874, 'DeepSeek-V3 s = 40: ×1.87')):
        f.line(R0, ry(v), R1, ry(v), 'muted', 1.6, dash=True)
    f.text(R0+6, ry(1.874)-8, 'YaRN, every position: ×1.87 DeepSeek-V3 · ×1.81 gpt-oss', 13, 'muted', anchor='start')
    steps = []
    ts = [10**(3 + k*4/400) for k in range(401)]
    for t in ts:
        steps.append((round(rx(t), 1), round(ry(1+.1*math.log1p(math.floor(t/8192))), 1)))
    f.path(steps, 'teal', width=2.6)
    for t in (8192, 131072, 1048576, 10485760):
        v = 1+.1*math.log1p(math.floor(t/8192))
        f.circle(rx(t), ry(v), 4, color='teal', fill=True)
        f.text(rx(t)-6, ry(v)-9, f'{v:.2f}', 13, 'teal', True, 'end')
    f.text(R1-4, ry(1.0)-12, 'Llama 4 · NoPE layers only', 13, 'teal', True, 'end')
    f.text(30, 400, 'Llama 4: attn_scale 0.1, floor_scale 8,192 (config); multiplies q, so logits scale by τ. YaRN folds (0.1 ln s + 1) into both q and k.', 13, 'muted', anchor='start')
    f.save()


def build():
    rope_spectrum()
    extension()
    partial()
    nope_layers()
    temperature()
