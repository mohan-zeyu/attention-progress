"""Share the KV heads: MHA → GQA → MQA."""
from figkit import Fig, _render_math


def head_sharing():
    f = Fig('kv-head-sharing', 'Eight query heads read eight, two or one KV head. Per token and layer the cache stores 2·H_kv·d_h elements: 16, 4 or 2 head-vectors.', 380)
    f.math(450, 34, r'\text{cache / token / layer} = \c{blue}{2} \cdot \c{blue}{H_{kv}} \cdot d_h \quad \text{(K and V)}', 22, anchor='middle')
    groups = {'MHA': 8, 'GQA': 2, 'MQA': 1}
    for p, (name, hkv) in enumerate(groups.items()):
        x0 = 30 + p*290
        if p:
            f.divider(x0-15, 60, 360)
        f.text(x0+125, 76, f'{name}', 20, bold=True)
        f.math(x0+125, 98, f'H_q = 8 \\to H_{{kv}} = {hkv}', 16, 'muted', 'middle')
        size = 8 // hkv
        for g in range(hkv):
            top = 112 + g*(8//hkv)*24
            f.parts.append(f'<rect x="{x0+4}" y="{top-2}" width="190" height="{size*24-4}" rx="8" fill="#f4f7f8" stroke="#cad5db"/>')
        kv_y = []
        for g in range(hkv):
            top = 112 + g*size*24
            kv_y.append(top + size*12 - 13)
        for i in range(8):
            y = 120 + i*24
            g = i // size
            f.circle(x0+22, y, 7, color='gold', fill=True)
            f.arrow(x0+32, y, x0+138, kv_y[g]+9, 'blue', width=1.2)
        for g, y in enumerate(kv_y):
            f.box(x0+142, y, 22, 18, 'K', 'blue', True, 11, rx=3)
            f.box(x0+167, y, 22, 18, 'V', 'purple', True, 11, rx=3)
        # per-token cache strip
        f.text(x0+125, 336, 'one token, one layer', 13, 'muted')
        for j in range(hkv):
            f.rect(x0+8+j*9, 300, 7, 18, 'blue', .85, 2)
            f.rect(x0+8+(hkv+j)*9 + 4, 300, 7, 18, 'purple', .85, 2)
        f.text(x0+262, 314, f'{2*hkv} vectors', 15, 'green', True, 'end')
        f.text(x0+125, 358, f'2·{hkv}·128 = {2*hkv*128:,} elems', 14, 'ink', anchor='middle')
    f.save()


def model_bytes():
    rows = [('Llama-2-7B · MHA 32/32', 512, 'red'), ('Llama-3-8B · GQA 32/8', 128, 'blue'),
            ('Llama-3-70B · GQA 64/8', 320, 'blue'), ('Falcon-7B · MQA 71/1', 8, 'green'),
            ('DeepSeek-V3 · MLA (ch. next)', 68.6, 'purple')]
    f = Fig('kv-bytes-models', 'KV cache per token summed over all layers in BF16, from each model config: Llama-2-7B 512 KiB, Llama-3-8B 128 KiB, Llama-3-70B 320 KiB, Falcon-7B 8 KiB, DeepSeek-V3 MLA 68.6 KiB. At 128K tokens multiply by 131,072.', 330)
    f.math(30, 34, r'\text{bytes / token} = L \cdot 2 H_{kv} d_h \cdot s', 21)
    f.text(870, 34, 'BF16 (s = 2 B), all layers', 14, 'muted', anchor='end')
    f.bars(30, 62, rows, width=300, bar=30, gap=16, label_w=250, size=15,
           value_fmt=lambda v: f'{v:g} KiB  →  {v*131072/1048576:.1f} GiB @128K')
    f.text(30, 318, 'Llama-3-70B: L=80, H_kv=8, d_h=128 → 80·2·8·128·2 B = 320 KiB per token.', 14, 'muted', anchor='start')
    f.save()


def decode_intensity():
    f = Fig('kv-decode-intensity', 'During decode one KV head is read from HBM once and reused by every query head in its group, so attention arithmetic intensity is about H_q/H_kv FLOPs per byte, far below the roughly 295 FLOP/byte an H100 needs to be compute-bound.', 432)
    f.math(30, 34, r'\text{intensity} \approx 4GNd_h \text{ FLOP} / 4Nd_h \text{ B} = \c{green}{G} = H_q / H_{kv}', 21)
    f.panel(30, 58, 250, 200, 'HBM · KV cache of one head')
    for r in range(5):
        f.rect(50, 94+r*28, 90, 20, 'blue', .8, 3)
        f.rect(150, 94+r*28, 90, 20, 'purple', .8, 3)
    f.text(155, 246, 'N tokens × d_h, read once', 13, 'muted')
    f.arrow(285, 158, 360, 158, 'blue', width=2.4)
    f.text(322, 148, 'bytes', 13, 'blue')
    f.panel(365, 58, 200, 200, 'SRAM')
    f.box(395, 133, 64, 50, 'K', 'blue', True, 18)
    f.box(469, 133, 64, 50, 'V', 'purple', True, 18)
    for i in range(4):
        y = 92 + i*44
        f.circle(660, y, 13, '$q', 'gold', True, 13)
        f.arrow(645, y, 540, 158, 'gold', width=1.2)
    f.text(686, 140, 'G query heads', 16, 'gold', True, 'start')
    f.text(686, 162, 'reuse the same bytes', 15, 'gold', anchor='start')
    f.hdivider(282)
    rows = [('MHA · G = 1', 1, 'red'), ('GQA · G = 8', 8, 'blue'), ('MQA · G = 64', 64, 'green'), ('H100 ridge point', 295, 'muted')]
    f.bars(30, 298, rows, width=420, bar=18, gap=8, label_w=170, size=14, value_fmt=lambda v: f'{v:g} FLOP/B')
    f.text(450, 424, 'Even G = 8 is ~37× below the ridge: decode attention stays bandwidth-bound, so bytes moved ≈ time.', 14, 'ink')
    f.save()


def _tied_pattern(f, pid='tied'):
    """Diagonal blue/purple stripes = one stored vector that serves as both key and value."""
    f.parts.append(f'<defs><pattern id="{pid}" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
                   f'<rect width="4" height="8" fill="#3f83b6"/><rect x="4" width="4" height="8" fill="#8d6bb5"/></pattern></defs>')


def _tied(f, x, y, w, h, pid='tied'):
    f.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2" fill="url(#{pid})" fill-opacity=".9"/>')


def _formula_row(f, y, pieces, size=21, gap=44, cx=450):
    """Several formula pieces centred as one row with real gaps between them."""
    widths = [_render_math(p, size, '#253340')[1] for p in pieces]
    x = cx - (sum(widths) + gap*(len(pieces)-1))/2
    for p, w in zip(pieces, widths):
        f.math(x, y, p, size)
        x += w + gap


def kv_tie():
    f = Fig('kv-tie', 'K=V tying: one projection u = W_K h serves as key (normed, 25% of dims rotated) and as value (normed, never rotated). Gemma 4 31B global layer: 4 KV heads × 512 store 4,096 elements with separate K and V; the report gives 37.5% less global KV. DeepSeek-V4: one 512-element entry is both key and value (1,024 → 512 elements). Kayyam et al.: K=V cuts KV to 50% of MHA at +3.1% perplexity; with GQA-4 to 12.5%, with MQA to 3.1%.', 450)
    _tied_pattern(f)
    _formula_row(f, 34, [r'\c{blue}{u_j} = W_K h_j', r'\c{blue}{k_j} = \c{teal}{RoPE}_j(\text{norm}_g(\c{blue}{u_j}))', r'\c{purple}{v_j} = \text{norm}(\c{blue}{u_j})'])
    # --- left: one projection, two reads (Gemma 4 31B global layer) ---
    f.panel(20, 54, 430, 200, 'One projection, two reads · Gemma 4 31B global')
    f.vec(40, 96, 7, 16, 'ink', values=[.25]*7)
    f.math(34, 232, r'h_j \in \R^{5376}', 15)
    f.arrow(58, 150, 106, 150, 'blue', width=1.8)
    f.box(108, 134, 52, 32, '$W_K', 'blue', True, 16)
    f.box(108, 180, 52, 28, '$W_V', 'line', False, 14, dash=True)
    f.line(110, 182, 158, 206, 'red', 1.6)
    f.text(134, 226, 'removed', 13, 'red')
    f.arrow(162, 150, 194, 150, 'blue', width=1.8)
    f.vec(198, 102, 6, 16, 'blue', values=[.8]*6)
    f.math(178, 232, r'\c{blue}{u_j} \in \R^{4 \times 512}', 15)
    f.arrow(218, 126, 292, 112, 'blue', width=1.4)
    f.arrow(218, 176, 292, 180, 'purple', width=1.4)
    for i in range(8):
        f.rect(298+i*16, 100, 14, 22, 'teal' if i < 2 else 'blue', .85, 2)
        f.rect(298+i*16, 168, 14, 22, 'purple', .85, 2)
    f.math(434, 117, r'\c{blue}{k_j}', 18)
    f.math(434, 185, r'\c{purple}{v_j}', 18)
    f.text(298, 142, 'norm (gain) · RoPE on 25%', 13, 'teal', anchor='start')
    f.text(298, 210, 'norm · never rotated', 13, 'purple', anchor='start')
    # --- right: Kayyam et al. study ---
    f.panel(466, 54, 414, 200, 'Study · KV cache as % of MHA')
    rows = [('separate K, V', 100, 'red'), ('K=V', 50, 'green'), ('K=V + GQA-4', 12.5, 'green'), ('K=V + MQA', 3.1, 'green')]
    f.bars(478, 96, rows, width=200, bar=20, gap=11, vmax=100, label_w=118, size=14, value_fmt=lambda v: f'{v:g}%')
    f.text(866, 226, 'K=V alone: +3.1% perplexity', 13, 'muted', anchor='end')
    f.text(866, 244, 'Kayyam et al., 300M / 1.2B, 10B tokens (reported)', 13, 'muted', anchor='end')
    # --- bottom: per-token, per-layer cache rows (elements) ---
    f.text(20, 284, 'Stored per token in one global layer (elements · BF16 bytes)', 15, 'ink', True, 'start')
    sc = 0.125
    x0, y = 200, 298
    rows = [
        ('Gemma 4 · K ≠ V', 'sep', 4, '4,096 · 8 KiB'),
        ('Gemma 4 · K = V', 'gem', 4, '2,560 · 5 KiB  (reported −37.5%)'),
        ('DeepSeek-V4 · K ≠ V', 'sep', 1, '1,024 · 2 KiB'),
        ('DeepSeek-V4 · K = V', 'v4', 1, '512 · 1 KiB  (one entry)'),
    ]
    for r, (name, kind, heads, label) in enumerate(rows):
        yy = y + r*34
        f.text(x0-12, yy+17, name, 14, 'ink', anchor='end')
        x = x0
        if kind == 'sep':
            for colour in ('blue', 'purple'):
                for h in range(heads):
                    f.rect(x, yy, 512*sc-2, 22, colour, .85, 2)
                    x += 512*sc
        elif kind == 'gem':
            _tied(f, x, yy, 2560*sc-2, 22)
            x += 2560*sc
        else:
            _tied(f, x, yy, 448*sc, 22)
            f.rect(x+448*sc, yy, 64*sc-1, 22, 'teal', 1, 0)
            x += 512*sc
        f.text(x+8, yy+17, label, 14, 'muted' if kind == 'sep' else 'green', kind != 'sep', 'start')
    f.callout(x0+478*sc, y+3*34+24, 330, 440, 'last 64 dims rotated; the output is rotated back by −t', 'teal', 13)
    f.rect(590, 272, 22, 14, 'blue', .85, 2); f.text(618, 284, 'K', 13, 'blue', True, 'start')
    f.rect(640, 272, 22, 14, 'purple', .85, 2); f.text(668, 284, 'V', 13, 'purple', True, 'start')
    _tied(f, 690, 272, 22, 14); f.text(718, 284, 'one vector = K and V', 13, 'ink', anchor='start')
    f.save()


def kv_designs():
    """Beyond GQA: cached elements per token/layer vs decode arithmetic intensity (BF16)."""
    import math
    f = Fig('kv-designs', 'Decode arithmetic intensity (FLOPs per KV byte, BF16) against cached elements per token per layer: MHA Llama-2-7B 8,192 elements, 1 FLOP/B; GQA Llama-3-70B 2,048, 8; GTA-8 toy 1,088, about 15; MFA Step-3 512, 64; GLA-2 toy 576, 128; MLA DeepSeek-V3 576, 242; K=V MQA DeepSeek-V4-Pro 512, 256. Ridge points: A800 156, H100 295.', 468)
    _tied_pattern(f)
    _formula_row(f, 34, [r'I \approx 2\c{green}{G} / m', r'\c{green}{G} = \text{query heads per cached head}', r'm = 2 \; (\c{blue}{K}, \c{purple}{V}) \text{ or } 1 \; (\text{tied / latent})'], size=20, gap=36)
    f.text(20, 72, 'design · model (H_q)', 13, 'muted', True, 'start')
    f.text(250, 72, 'cached / token / layer (elements)', 13, 'muted', True, 'start')
    f.text(520, 72, 'decode intensity I, FLOP/B (log scale)', 13, 'muted', True, 'start')
    rows = [
        ('MHA', 'Llama-2-7B (32/32)', [('blue', 4096), ('purple', 4096)], 8192, 1, '1', 'red', '32 K + 32 V heads × 128'),
        ('GQA-8', 'Llama-3-70B (64/8)', [('blue', 1024), ('purple', 1024)], 2048, 8, '8', 'blue', '8 K + 8 V heads × 128'),
        ('GTA-8', 'toy: 64 q, d_h 128', [('tied', 1024), ('teal', 64)], 1088, 32768/2176, '≈15', 'blue', '8 tied heads × 128 + 64 RoPE'),
        ('MFA', 'Step-3 (64 q × 256)', [('blue', 256), ('purple', 256)], 512, 64, '64', 'green', '1 K + 1 V head × 256'),
        ('GLA-2', 'toy: MLA shape, 128 q', [('tied', 256), ('tied2', 256), ('teal', 64)], 576, 128, '128', 'green', '2 latents × 256 + 64 RoPE'),
        ('MLA', 'DeepSeek-V3 (128 q)', [('tied', 512), ('teal', 64)], 576, 278528/1152, '242', 'green', '1 latent 512 + 64 RoPE'),
        ('K=V MQA', 'DeepSeek-V4-Pro (128 q)', [('tied', 448), ('teal', 64)], 512, 256, '256', 'green', '1 entry 512, last 64 rotated'),
    ]
    sc = 200/8192
    ax0, ax1 = 540, 860
    def ax(v):
        return ax0 + (ax1-ax0)*math.log2(v)/math.log2(512)
    top, rh = 84, 46
    bottom = top + len(rows)*rh
    # log axis grid
    for v in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        f.line(ax(v), top, ax(v), bottom, 'line', .8, opacity=.6)
        f.text(ax(v), bottom+16, f'{v}', 12, 'muted')
    for v, name in ((156, 'A800 156'), (295, 'H100 295')):
        f.line(ax(v), top-4, ax(v), bottom, 'red', 1.4, dash=True)
    f.text(ax(156)-4, bottom+34, 'ridge A800 156', 13, 'red', anchor='end')
    f.text(ax(295)+4, bottom+34, 'H100 295', 13, 'red', anchor='start')
    for r, (name, model, segs, total, ai, ai_s, c, desc) in enumerate(rows):
        y = top + r*rh
        if r:
            f.line(20, y-4, 880, y-4, 'line', .6, opacity=.6)
        f.text(20, y+16, name, 15, 'ink', True, 'start')
        f.text(20, y+34, model, 13, 'muted', anchor='start')
        x = 250
        for kind, n in segs:
            w = max(3, n*sc)
            if kind == 'tied':
                _tied(f, x, y+2, w-1, 18)
            elif kind == 'tied2':
                _tied(f, x, y+2, w-1, 18)
                f.line(x-1, y, x-1, y+22, 'ink', 1.4)
            else:
                f.rect(x, y+2, w-1, 18, kind, .9, 2)
            x += w
        f.text(x+6, y+16, f'{total:,}', 14, 'ink', True, 'start')
        f.text(250, y+35, desc, 12, 'muted', anchor='start')
        f.circle(ax(ai), y+17, 7, color=c, fill=True)
        f.text(ax(ai)+(-12 if ai > 100 else 12), y+22, ai_s, 14, c, True, 'end' if ai > 100 else 'start')
    f.text(250, bottom+16, 'teal = RoPE part · stripes = one vector is K and V', 12, 'teal', anchor='start')
    f.math(20, bottom+38, r'\text{BF16: } I = 2H_q(d_k+d_v) / (2 \text{ B} \cdot \text{cached elements})', 15, 'muted')
    f.save()


def _layer_cell(f, x, y, w, h, kind):
    """kind: G = owns a growing (global) cache, L = owns a window cache, S = reuses another layer's cache."""
    if kind == 'G':
        f.rect(x, y, w, h, 'blue', .95, 2)
    elif kind == 'L':
        f.rect(x, y, w, h, 'blue', .35, 2)
    elif kind == 'SG':
        f.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2" fill="white" stroke="#3f83b6" stroke-width="1.6"/>')
    else:
        f.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2" fill="white" stroke="#cad5db" stroke-width="1.2"/>')


def kv_cross_layer():
    f = Fig('kv-cross-layer', 'Which layers own a KV cache. CLA2 in Hunyuan-Large: 64 layers, every second layer reuses its neighbour, 32 caches, with GQA-8 about 5% of MHA (reported 95% saved). Gemma 3n E4B: 35 layers, layers 20 to 34 reuse layer 18 (local) or 19 (global): 20 caches, growing cache 14 to 8 KiB per token. Gemma 4 E2B: layers 15 to 34 reuse layers 13 and 14: 15 caches, 14 to 6 KiB per token. YOCO: a self-decoder with window caches, then one global KV read by every cross-decoder layer; reported about 80 times less KV at 65B.', 460)
    _formula_row(f, 34, [r'\c{blue}{K}^{(\ell)}, \c{purple}{V}^{(\ell)} \leftarrow \c{blue}{K}^{(src(\ell))}, \c{purple}{V}^{(src(\ell))}', r'\text{bytes / token} = \sum_{\ell \in \text{owners}} 2H_{kv}d_h s'], size=20, gap=50)
    # legend
    lx, ly = 20, 64
    for kind, lab in (('G', 'owns growing (global) cache'), ('L', 'owns window cache'), ('SG', 'reuses (global)'), ('S', 'reuses (local)')):
        _layer_cell(f, lx, ly-11, 12, 14, kind)
        f.text(lx+18, ly, lab, 13, 'ink', anchor='start')
        lx += 18 + len(lab)*6.6 + 22
    x0, sw = 170, 560   # strip area
    def row(y, name, sub, kinds, w, gap):
        f.text(20, y+15, name, 15, 'ink', True, 'start')
        f.text(20, y+33, sub, 13, 'muted', anchor='start')
        for i, k in enumerate(kinds):
            _layer_cell(f, x0+i*(w+gap), y, w, 22, k)
        return lambda i: x0 + i*(w+gap) + w/2
    # --- CLA2 · Hunyuan-Large ---
    y = 90
    kinds = ['G' if i % 2 == 0 else 'SG' for i in range(64)]
    cx = row(y, 'CLA2', 'Hunyuan-Large · 64 L', kinds, 6, 1.6)
    for i in range(1, 64, 2):
        f.path([(cx(i), y+24), (cx(i)-3.8, y+30), (cx(i-1), y+24)], 'blue', width=1)
    f.text(x0, y+46, 'layer 2i+1 reads layer 2i', 12, 'muted', anchor='start')
    f.text(880, y+15, '32 / 64 caches', 14, 'blue', True, 'end')
    f.text(880, y+33, 'with GQA-8: ≈5% of MHA', 13, 'green', True, 'end')
    # --- Gemma 3n E4B and Gemma 4 E2B ---
    def gemma(y, name, sub, first, full, src_l, src_g, before, after, window):
        kinds = []
        for i in range(35):
            g = i in full
            kinds.append(('G' if g else 'L') if i < first else ('SG' if g else 'S'))
        cx = row(y, name, sub, kinds, 12, 2)
        xa, xb = cx(first)-7, cx(34)+7
        f.brace(xa, y+24, xb, y+24, '', 'muted', 'below', 12, 5)
        f.path([(xa+3, y+31), (xa-6, y+44), (cx(src_g), y+26)], 'blue', arrow=True, width=1.4, curve=True)
        f.text(xa+12, y+47, f'layers {first}–34 reuse L{src_l} (local) / L{src_g} (global)', 12, 'muted', anchor='start')
        xd = cx(first)-8
        f.line(xd, y-6, xd, y+28, 'green', 1.6, dash=True)
        f.text(880, y+15, f'{first} / 35 caches', 14, 'blue', True, 'end')
        f.text(880, y+33, f'global {before} → {after} KiB/token', 13, 'green', True, 'end')
        f.text(880, y+49, f'window caches {window}', 12, 'muted', anchor='end')
    gemma(160, 'Gemma 3n E4B', '35 L · 2 KV × 256', 20, {4, 9, 14, 19, 24, 29, 34}, 18, 19, 14, 8, '28 → 16 MiB')
    gemma(236, 'Gemma 4 E2B', '35 L · 1 KV × 256/512', 15, {4, 9, 14, 19, 24, 29, 34}, 13, 14, 14, 6, '14 → 6 MiB')
    # --- YOCO ---
    y = 318
    f.text(20, y+15, 'YOCO', 15, 'ink', True, 'start')
    f.text(20, y+33, 'toy L = 32 · NeurIPS 2024', 13, 'muted', anchor='start')
    w, gap = 12, 2
    for i in range(16):
        _layer_cell(f, x0+i*(w+gap), y, w, 22, 'L')
    gx = x0 + 16*(w+gap) + 4
    f.box(gx, y-4, 44, 30, 'KV', 'blue', True, 14)
    for i in range(16):
        xx = gx + 52 + i*(w+gap)
        _layer_cell(f, xx, y, w, 22, 'SG')
        if i % 3 == 0:
            f.path([(xx+w/2, y+24), ((xx+gx+22)/2, y+44), (gx+22, y+28)], 'blue', width=1, curve=True)
    f.text(x0, y+46, 'self-decoder: SWA (W 1,024) or gated retention', 12, 'muted', anchor='start')
    f.text(gx+52, y+60, 'cross-decoder: every layer cross-attends to one global KV', 12, 'muted', anchor='start')
    f.line(gx+48, y-6, gx+48, y+28, 'green', 1.6, dash=True)
    f.text(880, y+15, '1 global KV', 14, 'blue', True, 'end')
    f.text(880, y+33, '65B: ≈80× less KV', 13, 'green', True, 'end')
    f.text(880, y+49, '(reported)', 12, 'muted', anchor='end')
    f.hdivider(400)
    f.line(30, 426, 30, 444, 'green', 1.6, dash=True)
    f.text(40, 440, 'prefill exit: layers above never produce KV, so prompt tokens can stop here (YOCO 2.87× prefill at 32K; Gemma 3n 2× vs Gemma 3 4B, reported)', 13, 'green', anchor='start')
    f.save()


def build():
    head_sharing()
    model_bytes()
    decode_intensity()
    kv_tie()
    kv_designs()
    kv_cross_layer()
