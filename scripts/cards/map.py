"""00 · The map: where the cost comes from, and which lever each route pulls."""
from figkit import Fig

N = 131072            # 128K tokens
D, L = 4096, 32       # Llama-3-8B: H_q·d_h = 32·128, 32 layers
PARAMS = 8.03e9
KV_TOKEN = L*2*8*128*2  # bytes per token, BF16, 8 KV heads


def costs():
    attn = 2*N*N*D*L/1e15          # causal: half of 4N²d per layer
    dense = 2*PARAMS*N/1e15
    kv = KV_TOKEN*N/2**30
    weights = PARAMS*2/2**30
    f = Fig('map-costs', f'Prefill pays compute that grows with N squared; decode pays bytes that grow with N for every generated token. Llama-3-8B at 128K: attention prefill {attn:.1f} PFLOP versus {dense:.1f} PFLOP for everything else; each decode step reads {kv:.0f} GiB of KV versus {weights:.0f} GiB of weights.', 400)
    f.text(225, 30, 'Prefill · compute-bound', 19, bold=True)
    f.text(675, 30, 'Decode · bandwidth-bound', 19, bold=True)
    f.divider(450, 16, 306)
    f.math(225, 62, r'\text{FLOP}_{attn} \approx \c{blue}{2N^2} d \cdot L', 20, anchor='middle')
    n = 11
    f.mask(40, 94, n, 15, lambda i, j: ('blue', .25 + .6*(j == i)) if j <= i else None)
    f.brace(40+n*15+4, 94, 40+n*15+4, 94+n*15-1.5, '$N', 'muted', 'right', 13, 4)
    f.text(122, 280, 'every query × every key', 13, 'muted')
    f.bars(215, 104, [('attention', round(attn, 1), 'blue'), ('rest of model', round(dense, 1), 'line')], width=95, bar=22, gap=10, label_w=110, size=13, value_fmt=lambda v: f'{v} PF')
    f.text(330, 196, 'Llama-3-8B, one 128K prompt', 13, 'muted')
    f.text(330, 214, 'attention > all other FLOPs', 13, 'blue', True)
    f.math(675, 62, r'\text{bytes / step} \approx \c{blue}{N} \cdot L \cdot 2H_{kv} d_h s \; + \; \text{weights}', 19, anchor='middle')
    f.circle(500, 140, 14, '$q_t', 'gold', True, 13)
    for i in range(10):
        f.rect(530+i*17, 125, 14, 30, 'blue', .75, 2)
        f.line(514, 140, 535+i*17, 140, 'gold', .8, opacity=.5)
    f.brace(530, 160, 530+10*17-3, 160, '$N \\text{ records, every step}', 'muted', 'below', 13, 6)
    f.bars(500, 214, [('KV cache', round(kv), 'blue'), ('weights', round(weights), 'line')], width=170, bar=22, gap=10, label_w=90, size=13, value_fmt=lambda v: f'{v} GiB')
    f.text(675, 296, 'read per generated token at 128K (×batch for KV)', 13, 'muted')
    f.hdivider(318, 30, 870)
    f.text(450, 346, 'Most routes in this book shrink one factor of the decode bill:', 15, 'ink')
    f.math(450, 376, r'\c{blue}{N} \text{ (records kept or read)} \;\cdot\; \c{blue}{2H_{kv}d_h} \text{ (record width)} \;\cdot\; L \;\cdot\; s \text{ (bytes)}', 18, anchor='middle')
    f.save()


LEVERS = [
    ('Make positions behave', 'enabler', 'teal', [('01', 'Positions'), ('02', 'Stable softmax')], 'pos'),
    ('Thinner records', 'width ↓', 'blue', [('03', 'Share KV heads'), ('04', 'Latent KV')], 'width'),
    ('Read fewer records', 'reads ↓, cache kept', 'green', [('05', 'Fixed patterns'), ('06', 'Sparse reads')], 'read'),
    ('Keep fewer records', 'N ↓, irreversible', 'red', [('07', 'Evict from cache'), ('08', 'Compress sequence')], 'keep'),
    ('Replace records by a state', 'O(1) per token', 'purple', [('09', 'Linear/SSM'), ('10', 'Delta rule'), ('11', 'TTT memory')], 'state'),
    ('Mix both memories', 'few full layers', 'gold', [('12', 'Hybrid stacks')], 'mix'),
    ('Move fewer bytes', 's ↓, traffic ↓', 'muted', [('13', 'Kernels'), ('14', 'Serving the cache')], 'bytes'),
]


def glyph(f, kind, x, y):
    """A 150×34 sketch of what the lever does to one layer's cache."""
    if kind == 'pos':
        for i in range(8):
            f.rect(x+i*18, y+8, 14, 22, 'blue', .55, 2)
            f.path([(x+i*18+3, y+4), (x+i*18+7, y), (x+i*18+11, y+4)], 'teal', width=1.4)
    elif kind == 'width':
        for i in range(8):
            f.rect(x+i*18, y+2, 14, 30, 'line', .9, 2)
            f.rect(x+i*18, y+20, 14, 12, 'blue', .85, 2)
    elif kind == 'read':
        for i in range(8):
            f.rect(x+i*18, y+2, 14, 30, 'green' if i in (0, 3, 6, 7) else 'line', .85, 2)
    elif kind == 'keep':
        for i in range(8):
            if i in (1, 2, 4):
                f.rect(x+i*18, y+2, 14, 30, 'red', .18, 2, stroke='red')
            else:
                f.rect(x+i*18, y+2, 14, 30, 'blue', .8, 2)
    elif kind == 'state':
        f.matrix(x+40, y, 3, 5, 11, 'purple')
        f.text(x+118, y+24, 'fixed', 12, 'purple', True, 'start')
    elif kind == 'mix':
        f.strip(x, y+4, [('', 'purple'), ('', 'purple'), ('', 'purple'), ('', 'blue')]*2, w=14, h=26, gap=4)
    elif kind == 'bytes':
        for i in range(8):
            f.rect(x+i*18, y+2, 14, 30, 'blue', .25, 2)
            f.rect(x+i*18, y+24, 14, 8, 'blue', .9, 2)
        f.text(x+150, y+30, 'FP8/4', 11, 'muted', True, 'start')


def levers():
    f = Fig('map-routes', 'Seven levers organise the book. Positions and stable softmax make attention trainable at length; MQA/GQA/MLA thin each record; fixed patterns and sparse reads touch fewer records; eviction and compression keep fewer; linear, delta and test-time-learning layers replace records with a fixed state; hybrids mix both; kernels and serving move fewer bytes.', 470)
    f.math(450, 32, r'\text{decode bytes} \approx \c{green}{N_{read}} \cdot \c{blue}{w_{record}} \cdot L \cdot \c{muted}{s} \qquad \text{state models: } \c{purple}{|S|} \cdot L \cdot s', 19, anchor='middle')
    f.text(40, 66, 'LEVER', 12, 'muted', True, 'start')
    f.text(330, 66, 'EFFECT ON ONE LAYER', 12, 'muted', True, 'start')
    f.text(540, 66, 'CHAPTERS', 12, 'muted', True, 'start')
    for r, (name, effect, c, chapters, kind) in enumerate(LEVERS):
        y = 80 + r*55
        f.rect(30, y, 840, 48, 'soft' if r % 2 == 0 else 'white', 1, 8)
        f.rect(30, y, 5, 48, c, 1, 2)
        f.text(48, y+21, name, 16, 'ink', True, 'start')
        f.text(48, y+40, effect, 13, c if c != 'muted' else 'muted', False, 'start')
        glyph(f, kind, 330, y+7)
        x = 540
        for num, label in chapters:
            w = f.pill(x, y+30, f'{num} {label}', c if c != 'muted' else 'ink', 12)
            x += w + 8
    f.save()


LANES = [
    ('KV width', 'blue', {2017: ['MHA'], 2019: ['MQA'], 2023: ['GQA'], 2024: ['MLA', 'CLA · YOCO'], 2026: ['K = V']}),
    ('Position · stability', 'teal', {2021: ['RoPE', 'ALiBi'], 2023: ['PI · YaRN'], 2024: ['soft-cap'], 2025: ['iRoPE', 'sink logit', 'output gate']}),
    ('Sparse · compress', 'green', {2019: ['Sparse Transf.'], 2020: ['Longformer', 'BigBird'], 2023: ['SWA', 'H2O', 'StreamingLLM'],
                                            2024: ['SnapKV', 'Quest', 'MInference'], 2025: ['NSA', 'MoBA', 'DSA'], 2026: ['CSA · HCA', 'MSA']}),
    ('Recurrent state', 'purple', {2020: ['Linear Transf.'], 2021: ['S4'], 2023: ['RetNet', 'Mamba', 'GLA'], 2024: ['Mamba-2', 'DeltaNet', 'GDN', 'TTT'],
                                   2025: ['RWKV-7', 'Titans', 'KDA'], 2026: ['GDN-2', 'KDN', 'Mamba-3']}),
    ('Hybrid · systems', 'gold', {2014: ['Bahdanau'], 2022: ['FlashAttn'], 2023: ['PagedAttn', 'Ring'], 2024: ['Jamba', 'FA-3'],
                                  2025: ['MiniMax-01', 'Qwen3-Next', 'Kimi Linear'], 2026: ['DeepSeek-V4', 'FA-4']}),
]


YEARX = {2014: 182, 2017: 238, 2019: 288, 2020: 338, 2021: 388, 2022: 438, 2023: 492, 2024: 592, 2025: 692, 2026: 792}


def timeline():
    f = Fig('map-timeline', 'Timeline 2014–2026 in five lanes: KV width (MHA, MQA, GQA, MLA, K=V), positions and stability (RoPE, YaRN, soft-capping, sink logits, output gates), sparse/evict/compress (Sparse Transformer to DSA and CSA/HCA), recurrent state (linear attention to KDA, GDN-2, Mamba-3), and hybrids/systems (FlashAttention, PagedAttention, MiniMax-01, Qwen3-Next, Kimi Linear, DeepSeek-V4). Years before 2023 are compressed.', 480)
    f.rect(YEARX[2023]-14, 34, 900-YEARX[2023]+4, 418, 'soft', .9, 10)
    f.text((YEARX[2023]+890)/2, 28, '2023–2026: most of the book (axis stretched)', 12, 'muted', True)
    for y, x in YEARX.items():
        f.line(x, 40, x, 448, 'line', 1, opacity=.7)
        f.text(x, 468, str(y), 13, 'ink' if y >= 2023 else 'muted', y >= 2023)
    for r, (lane, c, items) in enumerate(LANES):
        top = 42 + r*82
        f.rect(16, top, 150, 74, c, .12, 8)
        f.text(26, top+42, lane, 13, c if c != 'gold' else 'ink', True, 'start')
        ends = [0]*4
        for year in sorted(items):
            x = YEARX[year]
            for lab in items[year]:
                width = len(lab)*6.3 + 14
                k = next((i for i, e in enumerate(ends) if e < x - 2), 3)
                ends[k] = x + width
                y = top + 16 + k*18
                f.circle(x, y-4, 3.5, color=c, fill=True)
                f.text(x+6, y+1, lab, 11.5, 'ink', False, 'start')
    f.save()


def legend():
    f = Fig('map-legend', 'Colour legend used by every figure: gold query, blue key and KV cache, purple value and recurrent state, green output or kept, red erased or evicted, teal gates and positions, grey inactive. Notation: N tokens, L layers, H_q and H_kv heads, d_h head dim, s bytes per element, W window, S_t state.', 210)
    items = [('gold', 'query  q'), ('blue', 'key · KV cache  k'), ('purple', 'value · state  v, S'), ('green', 'output · kept'),
             ('red', 'erased · evicted · cost'), ('teal', 'gate · position'), ('line', 'inactive · skipped')]
    for i, (c, lab) in enumerate(items):
        x, y = 40 + (i % 4)*215, 30 + (i // 4)*44
        f.rect(x, y, 26, 26, c, .85, 5)
        f.text(x+36, y+18, lab, 15, 'ink', False, 'start')
    f.hdivider(122, 30, 870)
    f.math(450, 156, r'N \text{ tokens} \quad L \text{ layers} \quad H_q, H_{kv} \text{ heads} \quad G = H_q/H_{kv} \quad d_h \text{ head dim} \quad s \text{ bytes}', 17, anchor='middle')
    f.math(450, 190, r'\c{purple}{S_t} \in \mathbb{R}^{d_v \times d_k} \quad o_t = \c{purple}{S_t}\c{gold}{q_t} \quad \text{write } \c{purple}{v_t}\c{blue}{k_t^{\top}} \quad W \text{ window} \quad n_{sel} \text{ selected}', 17, anchor='middle')
    f.save()


def build():
    costs()
    levers()
    timeline()
    legend()
