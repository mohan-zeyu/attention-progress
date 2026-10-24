"""15 · Compare and choose."""
import math
from figkit import Fig

# Growing KV per token (BF16, growing layers only) computed from each config.json; see planning/research/D-hybrid-systems.md §2.
MODELS = [
    ('Llama 3.1-405B', 'all GQA 128/8', 516096, 'red'),
    ('GLM-4.6', 'all GQA 96/8', 376832, 'red'),
    ('MiniMax-M2', 'all GQA 48/8', 253952, 'red'),
    ('Gemma 2 27B', 'SWA 1:1, W 4096', 188416, 'green'),
    ('Gemma 3 27B', 'SWA 5:1, W 1024', 81920, 'green'),
    ('Kimi K2', 'all MLA (576)', 70272, 'blue'),
    ('DeepSeek-V3.2', 'MLA + DSA, in FP8', 40016, 'blue'),
    ('Llama 4 Scout', 'chunked 3:1', 49152, 'green'),
    ('MiniMax-01', 'lightning 7:1', 40960, 'purple'),
    ('Nemotron-H-56B', 'Mamba-2, 10 attn', 40960, 'purple'),
    ('gpt-oss-120b', 'SWA 1:1, W 128', 36864, 'green'),
    ('Qwen3-Next-80B', 'GDN 3:1', 24576, 'purple'),
    ('MiMo-V2-Flash', 'SWA 39:9, W 128', 23040, 'green'),
    ('Jamba v0.1', 'Mamba 7:1', 16384, 'purple'),
    ('Kimi Linear 48B', 'KDA 20 + MLA 7', 8064, 'purple'),
    ('Nemotron 3 Nano', 'Mamba-2, 6 attn', 6144, 'purple'),
    ('DeepSeek-V4.1-Flash', 'FP4 + index keys, reported', 890, 'gold'),
]


def kv_bytes():
    rows = sorted(MODELS, key=lambda r: -r[2])
    h = 76 + len(rows)*25 + 52
    f = Fig('choose-kv-bytes', 'Growing KV bytes per token, all layers, for 17 frontier designs on a log scale, from 516,096 B (Llama 3.1-405B, all GQA) down to 6,144 B (Nemotron 3 Nano) and a reported 890 B (DeepSeek-V4.1-Flash, FP4). Right column: the same cache at one million tokens.', h)
    f.math(450, 30, r'\text{growing KV / token} = \sum_{\ell \in \text{full layers}} \c{blue}{\text{record}_\ell} \cdot s \qquad (\text{bounded windows and states excluded})', 18, anchor='middle')
    x0, x1 = 330, 760
    lo, hi = math.log10(500), math.log10(600000)
    X = lambda v: x0 + (math.log10(v) - lo)/(hi - lo)*(x1 - x0)
    for tick, lab in [(1000, '1 KB'), (10000, '10 KB'), (100000, '100 KB')]:
        f.line(X(tick), 52, X(tick), h-46, 'line', 1, dash=True)
        f.text(X(tick), h-30, lab, 12, 'muted')
    f.text(830, 62, 'at 1M tokens', 12, 'muted', True)
    for i, (name, kind, b, c) in enumerate(rows):
        y = 70 + i*25
        f.text(24, y+13, name, 13.5, 'ink', True, 'start')
        f.text(178, y+13, kind, 11.5, 'muted', False, 'start')
        f.rect(x0, y+2, X(b)-x0, 15, c, .8, 3)
        f.text(X(b)+6, y+14, f'{b:,}', 12, c if c != 'gold' else 'ink', True, 'start')
        gib = b*2**20/2**30
        f.text(880, y+14, f'{gib:,.1f} GiB' if gib >= 1 else f'{gib*1024:.0f} MiB', 12, 'ink', False, 'end')
    lx = 24
    for lab, c in [('dense full attention', 'red'), ('MLA records', 'blue'), ('windows / chunks', 'green'), ('recurrent hybrids', 'purple'), ('compressed', 'gold')]:
        f.rect(lx, h-22, 12, 12, c, .8, 2)
        f.text(lx+18, h-12, lab, 12, 'ink', False, 'start')
        lx += 30 + len(lab)*6.6
    f.save()


PROPS = ['exact recall\nfar back', 'decode cost\nflat in N', 'nothing\ndiscarded', 'exposed by']
ROWS = [
    ('Full attention (GQA, MLA)', '03 04', ['yes', 'no', 'yes'], 'cost at 128K+'),
    ('Window / chunk layers alone', '05', ['no', 'yes', 'no'], 'needle beyond W'),
    ('Sparse reads (top-k)', '06', ['if found', 'reads only', 'yes'], 'multi-needle, aggregation'),
    ('Eviction', '07', ['if kept', 'yes', 'no'], 'question after the context'),
    ('Sequence compression', '08', ['blurred', 'N / r', 'no'], 'verbatim copy'),
    ('Recurrent state', '09–11', ['capacity', 'yes', 'no'], 'associative recall, copy'),
    ('Hybrid (few full layers)', '12', ['yes', 'mostly', 'yes'], 'cost of the full layers'),
]
TONE = {'yes': 'green', 'no': 'red'}


def properties():
    f = Fig('choose-properties', 'What each lever guarantees by construction. Full attention: exact recall, nothing discarded, cost grows. Windows alone: flat cost, no recall beyond W. Sparse reads: cache intact, recall only if the selector finds it. Eviction: flat cost, irreversible. Compression: blurred detail, still N/r growth. Recurrent state: flat cost, recall limited by capacity. Hybrids: exact recall through a few full layers.', 420)
    f.math(450, 30, r'\text{choose by what you cannot afford to lose:} \quad \c{green}{\text{recall}} \;\;\text{vs}\;\; \c{red}{\text{bytes per token}}', 19, anchor='middle')
    cols = [330, 450, 570, 735]
    for c, name in zip(cols, PROPS):
        for k, part in enumerate(name.split('\n')):
            f.text(c, 64+k*16, part, 13, 'muted', True)
    for i, (name, ch, cells, exposed) in enumerate(ROWS):
        y = 100 + i*44
        f.rect(20, y, 860, 38, 'soft' if i % 2 == 0 else 'white', 1, 6)
        f.text(34, y+17, name, 14.5, 'ink', True, 'start')
        f.text(34, y+33, 'ch. ' + ch, 12, 'muted', False, 'start')
        for c, cell in zip(cols, cells):
            tone = TONE.get(cell, 'gold')
            f.box(c-48, y+7, 96, 24, cell, tone, cell in ('yes', 'no'), 12.5, .85 if cell in ('yes', 'no') else 1, rx=12)
        f.text(cols[3], y+24, exposed, 13, 'ink', False, 'middle')
    f.text(450, 428, 'Design-level properties only. How often "if found / if kept / capacity" fails depends on training and budget; measure it.', 13, 'muted')
    f.save()


RECIPES = [
    ('DeepSeek-V4-Flash', '2026-04 · 43 layers', [
        ('01', 'partial RoPE 64 + YaRN ×16'), ('02', 'learned sinks'), ('03', 'MQA: 1 KV head, K = V (512)'),
        ('05', 'SWA W 128 branch in every layer'), ('06', 'indexer top-512 over compressed'), ('08', 'CSA r 4 / HCA r 128 alternate'),
        ('14', 'FP8 entries, BF16 RoPE dims')], 'reported: 10% of V3.2 KV at 1M'),
    ('Qwen3.8-Flash-Next', '2026-08 · 48 layers', [
        ('01', 'partial mRoPE 64/256'), ('02', 'sigmoid output gate'), ('03', 'GQA 24 Q / 2 KV, d_h 256'),
        ('06', 'QSA: pooled-key indexer, 2,048 budget'), ('10', 'Gated DeltaNet, 36 layers'), ('12', '12 × [3 GDN + 1 attention]')], '24,576 B/token growing KV'),
    ('Gemma 4 31B', '2026-04 · 60 layers', [
        ('01', 'local θ 1e4, global p-RoPE 25%'), ('02', 'QK-norm'), ('03', 'global: 4 KV heads, K = V, d_h 512'),
        ('05', '5 SWA (W 1,024) : 1 global')], '51,200 B/token (K=V: −37.5%, reported)'),
]


def recipes():
    f = Fig('choose-recipes', 'Three 2026 stacks combine several levers each. DeepSeek-V4-Flash: partial RoPE, sinks, one K=V head, a 128-token window branch, a top-512 indexer, CSA/HCA compression and FP8 storage. Qwen3.8-Flash-Next: partial mRoPE, output gates, GQA with 2 KV heads, a QSA indexer and 3:1 Gated DeltaNet hybrid. Gemma 4 31B: dual RoPE bases, QK-norm, K=V global layers and a 5:1 window ratio.', 400)
    f.math(450, 30, r'\text{a 2026 stack} = \text{positions} + \text{stability} + \c{blue}{\text{thin records}} + \c{green}{\text{fewer reads}} + \c{purple}{\text{state}} + \c{muted}{\text{bytes}}', 18, anchor='middle')
    colour = {'01': 'teal', '02': 'teal', '03': 'blue', '04': 'blue', '05': 'green', '06': 'green', '08': 'red', '10': 'purple', '12': 'gold', '14': 'muted'}
    for k, (name, sub, items, foot) in enumerate(RECIPES):
        x = 20 + k*290
        f.panel(x, 50, 280, 340)
        f.text(x+14, 76, name, 16, 'ink', True, 'start')
        f.text(x+14, 94, sub, 12, 'muted', False, 'start')
        for i, (ch, text) in enumerate(items):
            y = 108 + i*36
            c = colour[ch]
            f.box(x+14, y, 30, 26, ch, c, True, 12, .9, 5)
            f.text(x+52, y+18, text, 12.5, 'ink', False, 'start')
        f.text(x+140, 376, foot, 12.5, 'green', True)
    f.save()


def build():
    kv_bytes()
    properties()
    recipes()
