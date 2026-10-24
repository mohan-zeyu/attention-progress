"""12 · Hybrid stacks: figures (prefix hybrid-). Layer patterns are read from each model's config.json."""
import math
from figkit import Fig

EM = ' '  # SVG collapses ordinary spaces; em spaces survive

# ---------------------------------------------------------------- layer patterns (from config.json)
# codes: G global attention, S sliding window, R recurrent (Mamba / GDN / KDA / lightning),
#        P parallel attention + SSM in one layer, X sparse attention (top-k blocks)
def every(n, period, offset, hit='G', miss='R'):
    return [hit if i % period == offset else miss for i in range(n)]


def from_list(n, hits, hit='G', miss='R', one_indexed=True):
    hs = {h - 1 if one_indexed else h for h in hits}
    return [hit if i in hs else miss for i in range(n)]


def mixers(pattern):
    """Nemotron hybrid_override_pattern: M = Mamba-2, * = attention, - = MLP, E = MoE. Keep token mixers."""
    return ['G' if c == '*' else 'R' for c in pattern if c in 'M*']


NEMO_H56 = 'M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-'
NEMO_NANO2 = 'M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-'
NEMO3_NANO = 'MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME'

LAYOUTS = {
    'Jamba': every(32, 8, 4),
    'MiniMax-Text-01': every(80, 8, 7),
    'Gemma 3 27B': every(62, 6, 5, 'G', 'S'),
    'Nemotron-H-56B': mixers(NEMO_H56),
    'Falcon-H1-34B': ['P']*72,
    'gpt-oss-120b': every(36, 2, 1, 'G', 'S'),
    'Qwen3-Next-80B': every(48, 4, 3),
    'MiniMax-M2': ['G']*62,
    'Kimi Linear 48B': from_list(27, [4, 8, 12, 16, 20, 24, 27]),
    'Nemotron 3 Nano': mixers(NEMO3_NANO),
    'Qwen3.5-397B': every(60, 4, 3),
    'MiniMax-M3': ['G']*3 + ['X']*57,
    'Kimi K3': from_list(93, list(range(4, 93, 4)) + [93]),
}

TONE = {'G': ('blue', .9), 'S': ('blue', .3), 'R': ('purple', .8), 'X': ('green', .8)}


def draw_strip(f, x, y, cells, width, h=16, gap=1.0):
    n = len(cells)
    w = width/n
    for i, c in enumerate(cells):
        if c == 'P':
            f.rect(x + i*w, y, max(.8, w - gap), h*.35, 'blue', .9)
            f.rect(x + i*w, y + h*.35, max(.8, w - gap), h*.65, 'purple', .8)
        else:
            tone, op = TONE[c]
            f.rect(x + i*w, y, max(.8, w - gap), h, tone, op)
    return w


def kib(v):
    return f'{v:.1f}' if v < 10 else f'{v:.0f}'


# ---------------------------------------------------------------- card 1
def why_hybrid():
    f = Fig('hybrid-why', 'Per-sequence memory in elements against context length for the Qwen3-Next-80B-A3B stack '
            '(36 Gated DeltaNet + 12 gated attention layers): the fixed state is 18.9M elements and the 12 attention '
            'layers add 12,288 elements per token, so KV overtakes the state at N = 1,536. The same 48 layers as all '
            'attention would need 49,152 per token, as all GDN a flat 25.2M. Right: the whole recurrent state equals the '
            'KV of N* tokens: Jamba 448, Qwen3-Next 1,536, Kimi Linear 2,601, Qwen3.5 3,072, MiniMax-01 3,584, '
            'Nemotron 3 Nano 3,925, Granite 4.0-H 4,608, Kimi K3 7,851, Nemotron-H-56B 11,059.', 440)
    f.math(450, 34, r'M(N) = \c{purple}{\sum_{ℓ \in R} |S_ℓ|} + N \cdot \c{blue}{\sum_{ℓ \in G} c_ℓ}' + EM*2 +
           r'N^{*} = \c{purple}{\sum |S_ℓ|} \,/\, \c{blue}{\sum c_ℓ}', 21, anchor='middle')
    f.panel(24, 56, 470, 340, 'Qwen3-Next stack: elements per sequence (config)')
    px, py, pw, ph = 84, 96, 300, 236
    lx = lambda n: px + pw*(math.log10(n) - 3)/3
    ly = lambda e: py + ph - ph*(math.log10(e) - 6)/5
    for k, lab in enumerate(['1K', '10K', '100K', '1M']):
        x = lx(10**(3 + k))
        f.line(x, py, x, py + ph, 'line', .8)
        f.text(x, py + ph + 16, lab, 12, 'muted')
    for k, lab in enumerate(['1M', '10M', '100M', '1G', '10G', '100G']):
        y = ly(10**(6 + k))
        f.line(px, y, px + pw, y, 'line', .8)
        f.text(px - 6, y + 4, lab, 12, 'muted', anchor='end')
    f.text(px + pw/2, py + ph + 34, 'context N (tokens, log)', 13, 'muted')
    state, kv_h, kv_full, state_all = 18_874_368, 12_288, 49_152, 25_165_824
    Ns = [10**(3 + k/20) for k in range(61)]
    f.path([(round(lx(n), 1), round(ly(kv_full*n), 1)) for n in Ns], 'red', width=2.2)
    f.path([(round(lx(n), 1), round(ly(state + kv_h*n), 1)) for n in Ns], 'green', width=2.6)
    f.path([(round(lx(n), 1), round(ly(kv_h*n), 1)) for n in Ns], 'blue', width=1.4, dash=True)
    f.line(lx(1000), ly(state), lx(1e6), ly(state), 'purple', 1.4, True)
    f.line(lx(1000), ly(state_all), lx(1e6), ly(state_all), 'purple', 2.2)
    f.text(px + pw + 6, ly(kv_full*1e6) + 4, '48 attn: 51.5G', 12, 'red', True, 'start')
    f.text(px + pw + 6, ly(state + kv_h*1e6) + 14, 'hybrid: 12.9G', 12, 'green', True, 'start')
    f.text(px + pw + 6, ly(state_all) - 4, '48 GDN', 12, 'purple', True, 'start')
    f.text(px + pw + 6, ly(state_all) + 10, '25.2M flat', 12, 'purple', anchor='start')
    f.text(px + 6, ly(state) + 16, 'state of 36 GDN layers: 18.9M', 12, 'purple', anchor='start')
    xs = lx(1536)
    f.circle(xs, ly(state), 5, color='ink', fill=True)
    f.line(xs + 4, ly(state) + 4, xs + 40, ly(state) + 36, 'ink', 1)
    f.text(xs + 42, ly(state) + 42, 'N* = 1,536: KV = state', 13, 'ink', True, 'start')
    f.text(px + pw - 4, ly(kv_h*3e5) + 22, 'KV of 12 attn layers', 12, 'blue', anchor='end')
    # N* bars
    f.panel(508, 56, 368, 340, 'state = KV of N* tokens (config)')
    rows = [('Jamba', 448), ('Qwen3-Next', 1536), ('Kimi Linear', 2601), ('Qwen3.5-397B', 3072),
            ('MiniMax-01', 3584), ('Nemotron 3 Nano', 3925), ('Granite 4.0-H-S', 4608), ('Kimi K3', 7851),
            ('Nemotron-H-56B', 11059)]
    f.bars(518, 92, [(n, v, 'purple') for n, v in rows], width=130, bar=19, gap=12, label_w=128, size=13,
           value_fmt=lambda v: f'{v:,}')
    f.text(692, 388, 'elements; fp32 state doubles the byte N*', 12, 'muted')
    f.text(24, 428, 'Past N*, the few attention layers dominate memory. The hybrid cuts the KV slope 4×; it does not stop it growing.', 14, 'ink', anchor='start')
    f.save()


# ---------------------------------------------------------------- card 2
def seq_par():
    f = Fig('hybrid-seq-par', 'Sequential versus parallel hybrids. Sequential (Jamba, Nemotron-H, Samba): whole layers '
            'alternate, and only the attention layers keep a KV cache. Parallel (Hymba, Falcon-H1): every layer runs '
            'attention heads and SSM heads side by side; Falcon-H1 splits channels SSM 2 : attention 1 : MLP 5. '
            'Computed KV per token in BF16: Falcon-H1-34B 144 KiB (72 layers, 4 KV heads), Nemotron-H-56B 40 KiB '
            '(10 layers, 8 KV heads). Reported for Hymba-1.5B at 8K: 78.6 MB cache and 664 tok/s, 11.67 times smaller '
            'and 3.49 times faster than Llama-3.2-3B.', 450)
    f.math(450, 34, r'\text{Hymba: } Y = W_{out}(\c{teal}{β_1} \odot \text{norm}(\c{blue}{M_{attn}}X) + '
           r'\c{teal}{β_2} \odot \text{norm}(\c{purple}{M_{ssm}}X))', 21, anchor='middle')
    # sequential
    f.panel(24, 56, 270, 300, 'inter-layer (sequential)')
    seq = ['M', 'F', 'M', 'F', 'M', 'F', 'A', 'F']
    for i, c in enumerate(seq):
        y = 92 + i*28
        tone = {'M': 'purple', 'A': 'blue', 'F': 'line'}[c]
        lab = {'M': 'Mamba-2', 'A': 'attention', 'F': 'FFN'}[c]
        f.box(64, y, 120, 22, lab, tone, c != 'F', 13, .85, 4)
        if c == 'A':
            f.rect(194, y + 3, 50, 16, 'blue', .85, 2)
            f.text(250, y + 15, 'KV', 12, 'blue', True, 'start')
        if c == 'M':
            f.rect(194, y + 5, 14, 12, 'purple', .85, 2)
            f.text(212, y + 15, 'state', 12, 'purple', anchor='start')
    f.line(52, 88, 52, 318, 'muted', 1.4)
    f.text(46, 330, 'residual', 11, 'muted', anchor='start')
    f.lines(40, 346, ['Jamba 4/32 · Nemotron-H 10/118'], 12, 'muted')
    # parallel
    f.panel(308, 56, 300, 300, 'intra-layer (parallel), one layer')
    f.box(330, 92, 256, 24, 'norm(x)', 'line', False, 13, 1, 4)
    f.arrow(394, 116, 394, 136, 'muted')
    f.arrow(522, 116, 522, 136, 'muted')
    f.box(330, 138, 120, 70, '', 'blue', True, 13, .22, 4)
    f.text(390, 160, 'attention', 14, 'blue', True)
    f.text(390, 178, 'heads', 13, 'blue')
    f.text(390, 198, 'KV every layer', 12, 'blue')
    f.box(458, 138, 128, 70, '', 'purple', True, 13, .22, 4)
    f.text(522, 160, 'SSM heads', 14, 'purple', True)
    f.text(522, 178, 'Mamba / Mamba-2', 13, 'purple')
    f.text(522, 198, 'fixed state', 12, 'purple')
    f.arrow(390, 208, 450, 232, 'muted')
    f.arrow(522, 208, 470, 232, 'muted')
    f.box(400, 234, 116, 24, '$β_1, β_2 \\text{ + concat}', 'teal', False, 13, 1, 4)
    f.arrow(458, 258, 458, 274, 'muted')
    f.box(400, 276, 116, 24, 'MLP', 'line', False, 13, 1, 4)
    f.text(458, 322, 'Falcon-H1 channels (SA_M block):', 12, 'ink')
    tot, x0 = 240, 338
    for part, w8, tone in (('SSM 2', 2, 'purple'), ('attn 1', 1, 'blue'), ('MLP 5', 5, 'line')):
        w = tot*w8/8
        f.rect(x0, 330, w - 2, 16, tone, .85, 2)
        f.text(x0 + w/2 - 1, 343, part, 11, 'white' if tone != 'line' else 'ink', True)
        x0 += w
    # numbers
    f.panel(622, 56, 254, 300, 'KV per token, BF16 (config)')
    f.bars(632, 96, [('Falcon-H1-34B', 144, 'red'), ('Nemotron-H-56B', 40, 'blue')], width=90, bar=18, gap=10,
           label_w=114, size=12, value_fmt=lambda v: f'{v} KiB')
    f.lines(636, 172, ['Falcon-H1: 72 layers × 4 KV', 'Nemotron-H: 10 layers × 8 KV'], 12, 'muted')
    f.hdivider(212, 634, 864)
    f.text(636, 236, 'Hymba-1.5B, 8K (reported)', 13, 'ink', True, 'start')
    f.bars(632, 248, [('cache MB', 78.6, 'green'), ('Llama-3.2-3B', 918, 'red')], width=90, bar=16, gap=6,
           label_w=100, size=12, value_fmt=lambda v: f'{v:g}')
    f.lines(636, 310, ['3 global layers (0, 15, 31), rest SWA', 'KV shared by layer pairs; 664 tok/s'], 12, 'muted')
    f.text(24, 384, 'Parallel ≠ fewer KV layers: Falcon-H1 keeps KV in all 72 layers and saves through few KV heads.', 14, 'ink', anchor='start')
    f.text(24, 406, 'Sequential hybrids put KV in few layers. Hymba gets both: parallel heads + SWA + cross-layer KV sharing.', 14, 'ink', anchor='start')
    f.text(24, 430, 'Samba (Microsoft): Mamba-MLP-SWA(2048)-MLP repeated, no global layer at all.', 14, 'muted', anchor='start')
    f.save()


# ---------------------------------------------------------------- card 3
def nvidia():
    f = Fig('hybrid-nvidia', 'NVIDIA hybrids drawn from config.json token-mixer layers. Nemotron-H-56B: 10 attention among 64 '
            'mixers (118 layers with FFNs), reported 2.4 times the throughput of Qwen-2.5-72B and Llama-3.1-70B at 64K in '
            '/ 1K out. Nemotron Nano 2 9B: 4 attention among 31 mixers, up to 6 times Qwen3-8B. Nemotron 3 Nano: 6 '
            'attention among 29 mixers, 3.3 times Qwen3-30B-A3B. Jet-Nemotron-2B retrofits Qwen2.5-1.5B: 2 full and 2 '
            'SWA layers kept, 24 JetBlocks; KV at 64K falls from 1,792 MB to 154 MB and throughput rises from 241 to 2,885 '
            'tokens per second (reported, H100).', 462)
    f.math(450, 34, r'\text{KV/tok} = \c{blue}{L_{attn}} \cdot 2\c{blue}{H_{kv}} d_h s' + EM*2 +
           r'\text{state} = \c{purple}{L_{ssm}} \cdot H \cdot P \cdot N_{state}', 21, anchor='middle')
    W, x0 = 470, 210
    rows = [('Nemotron-H-56B', mixers(NEMO_H56), '10 / 64 mixers', '2.4× vs Llama-3.1-70B', 'KV 40 KiB/tok'),
            ('Nemotron Nano 2 9B', mixers(NEMO_NANO2), '4 / 31', 'up to 6× vs Qwen3-8B', 'KV 16 KiB/tok'),
            ('Nemotron 3 Nano', mixers(NEMO3_NANO), '6 / 29, H_kv = 2', '3.3× vs Qwen3-30B-A3B', 'KV 6 KiB/tok')]
    f.text(x0, 70, 'Mamba-2 mixers (purple) and attention (blue); FFN/MoE layers omitted · reported throughput', 13, 'muted', anchor='start')
    for i, (name, cells, frac, thr, kvs) in enumerate(rows):
        y = 86 + i*48
        f.text(x0 - 12, y + 13, name, 14, 'ink', True, 'end')
        draw_strip(f, x0, y, cells, W, 18)
        f.text(x0, y + 34, frac, 12, 'blue', anchor='start')
        f.text(x0 + W/2, y + 34, thr, 12, 'green', True)
        f.text(x0 + W, y + 34, kvs, 12, 'muted', anchor='end')
        f.text(x0 + W + 10, y + 13, f'{len(cells)}', 12, 'muted', anchor='start')
    f.text(30, 252, 'No position embeddings; first layer Mamba-2; attention ≈ 8 % of layers, evenly spread (Nemotron-H).', 13, 'ink', anchor='start')
    f.hdivider(266)
    # PostNAS retrofit
    f.text(30, 292, 'Jet-Nemotron-2B · PostNAS retrofit of Qwen2.5-1.5B (MLPs frozen, search where attention stays)', 14, 'ink', True, 'start')
    before = ['G']*28
    jet = ['R']*28
    for i in (14, 19):
        jet[i] = 'G'
    for i in (20, 21):
        jet[i] = 'S'
    f.text(x0 - 12, 318, 'Qwen2.5-1.5B', 13, 'ink', anchor='end')
    draw_strip(f, x0, 306, before, 360, 16)
    f.text(x0 - 12, 346, 'Jet-Nemotron-2B', 13, 'ink', True, 'end')
    draw_strip(f, x0, 334, jet, 360, 16)
    cw = 360/28
    f.text(x0 + 17*cw, 366, 'full: 15, 20', 12, 'blue', True)
    f.text(x0 + 23.5*cw, 366, 'SWA: 21, 22', 12, 'blue')
    f.text(x0, 386, 'all other layers → JetBlock (GDN + dynamic conv on V)', 12, 'purple', anchor='start')
    f.text(600, 314, 'KV cache MB @64K', 12, 'muted', anchor='start')
    f.bars(600, 320, [('Qwen2.5-1.5B', 1792, 'red'), ('Jet-2B', 154, 'green')], width=110, bar=13, gap=4, label_w=96,
           size=12, value_fmt=lambda v: f'{v:,}')
    f.text(600, 366, 'tok/s @64K, H100', 12, 'muted', anchor='start')
    f.bars(600, 372, [('Qwen2.5-1.5B', 241, 'red'), ('Jet-2B', 2885, 'green')], width=110, bar=13, gap=4, label_w=96,
           size=12, value_fmt=lambda v: f'{v:,}')
    f.text(30, 428, 'Reported: KV cache size, not parameter count, sets long-context generation throughput (Key Finding 4).', 13, 'ink', anchor='start')
    f.text(30, 450, 'Attention importance is task-specific: retrieval and MMLU depend on different 2–3 layers.', 13, 'muted', anchor='start')
    f.save()


# ---------------------------------------------------------------- card 4
def delta31():
    f = Fig('hybrid-delta31', 'Delta-rule hybrids from config.json. Qwen3-Next-80B: 12 x (3 GDN + 1 gated attention), 48 '
            'layers. Qwen3.5-397B: 15 x (3 + 1), 60 layers. Kimi Linear 48B: 20 KDA + 7 MLA, MLA at layers 4, 8, ..., '
            '24 and 27, so the last group is 2:1. Kimi K3: 69 KDA + 24 gated MLA over 93 layers. Kimi Linear ratio '
            'ablation, validation perplexity: 0:1 5.77, 1:1 5.66, 3:1 5.65, 7:1 5.70, 15:1 5.82. Qwen3.8 ablation on a '
            '25B-A3B model, average score: full attention 49.87, SWA hybrid 51.15, GDN hybrid 53.81.', 456)
    f.math(450, 34, r'\text{stack} = n \times [\,3 \times (\c{purple}{\text{GDN}} \to \text{MoE}) + 1 \times '
           r'(\c{blue}{\text{gated attn}} \to \text{MoE})\,]' + EM + r'n = 12 \text{ (3-Next)},\; 15 \text{ (3.5)}', 20, anchor='middle')
    W, x0 = 470, 170
    rows = [('Qwen3-Next-80B', LAYOUTS['Qwen3-Next-80B'], '36 GDN + 12 GQA (2 KV heads, d 256)', '24 KiB/tok'),
            ('Qwen3.5-397B', LAYOUTS['Qwen3.5-397B'], '45 GDN (64 V heads) + 15 GQA', '30 KiB/tok'),
            ('Kimi Linear 48B', LAYOUTS['Kimi Linear 48B'], '20 KDA + 7 NoPE MLA · last group 2 : 1', '7.9 KiB/tok'),
            ('Kimi K3', LAYOUTS['Kimi K3'], '69 KDA + 24 gated NoPE MLA · ends MLA, MLA', '27 KiB/tok')]
    for i, (name, cells, note, kvs) in enumerate(rows):
        y = 64 + i*42
        f.text(x0 - 12, y + 13, name, 14, 'ink', True, 'end')
        draw_strip(f, x0, y, cells, W, 18)
        f.text(x0, y + 33, note, 12, 'muted', anchor='start')
        f.text(x0 + W + 12, y + 14, kvs, 13, 'blue', True, 'start')
        f.text(x0 + W + 12, y + 30, f'{len(cells)} layers', 12, 'muted', anchor='start')
    f.hdivider(238)
    # ratio ablation
    f.panel(24, 250, 420, 190, 'Kimi Linear ratio ablation (reported, val. PPL ↓)')
    pts = [('0:1', 5.77), ('1:1', 5.66), ('3:1', 5.65), ('7:1', 5.70), ('15:1', 5.82)]
    px, py, pw, ph = 70, 290, 300, 110
    ly = lambda v: py + ph - ph*(v - 5.60)/(5.85 - 5.60)
    for v in (5.65, 5.75, 5.85):
        f.line(px, ly(v), px + pw, ly(v), 'line', .8)
        f.text(px - 6, ly(v) + 4, f'{v:.2f}', 11, 'muted', anchor='end')
    coords = []
    for i, (lab, v) in enumerate(pts):
        x = px + i*pw/4
        coords.append((x, ly(v)))
        f.text(x, py + ph + 16, lab, 12, 'ink', True)
        kvfrac = '1' if lab == '0:1' else f'1/{int(lab.split(":")[0]) + 1}'
        f.text(x, py + ph + 30, f'KV {kvfrac}', 11, 'blue')
    f.path(coords, 'purple', width=2)
    for (x, y), (lab, v) in zip(coords, pts):
        f.circle(x, y, 4.5 if lab != '3:1' else 6, color='green' if lab == '3:1' else 'purple', fill=True)
        f.text(x + 8, y - 6, f'{v:.2f}', 11, 'green' if lab == '3:1' else 'purple', True, 'start')
    f.text(px + pw + 36, py + 40, 'KDA : MLA', 12, 'muted')
    # Qwen3.8 + Kimi TPOT
    f.panel(456, 250, 420, 190, 'Qwen3.8 ablation, 25B-A3B (reported avg ↑)')
    f.bars(466, 288, [('full attention', 49.87, 'red'), ('3:1 SWA-128 hybrid', 51.15, 'blue'),
                      ('3:1 GDN hybrid', 53.81, 'green')], width=120, bar=17, gap=8, label_w=150, size=12,
           value_fmt=lambda v: f'{v:.2f}', vmax=56)
    f.text(476, 384, 'Kimi Linear decode at 1M (reported):', 12, 'ink', True, 'start')
    f.text(476, 402, 'TPOT 1.84 ms vs 11.48 ms for full MLA (6.3×)', 12, 'green', anchor='start')
    f.text(476, 420, 'KV cache up to 75 % smaller', 12, 'muted', anchor='start')
    f.save()


# ---------------------------------------------------------------- card 5
def minimax():
    f = Fig('hybrid-minimax', 'MiniMax layouts from config.json. Text-01 (Jan 2025) and M1 (Jun 2025): 80 layers, 70 '
            'lightning-attention plus 10 softmax, KV 40 KiB per token. M2 (Oct 2025): 62 full GQA layers, 248 KiB. M3 '
            '(2026): 3 dense plus 57 sparse MSA layers, 120 KiB retained per token, each query reads 16 blocks of 128 '
            'tokens plus a local block. The M2 blog gives four reasons for leaving the hybrid: multi-hop reasoning '
            'deficits that appeared only at scale, costly evaluation and late-surfacing precision problems, immature '
            'infrastructure (memory-bound kernels, low-precision state, prefix caching, speculative decoding), and a '
            'sliding-window hybrid that degraded with context length.', 452)
    f.math(450, 34, r'\text{KV/tok} = \c{blue}{L_{full}} \cdot 2H_{kv}d_h:' + EM + r'10·2·8·128 \to 62·2·8·128 \to 60·2·4·128', 21, anchor='middle')
    W, x0 = 420, 230
    rows = [('Text-01 · 2025-01', LAYOUTS['MiniMax-Text-01'], '7 lightning : 1 softmax', 40),
            ('M1 · 2025-06', LAYOUTS['MiniMax-Text-01'], 'same layout, RL reasoning', 40),
            ('M2 · 2025-10', LAYOUTS['MiniMax-M2'], 'all full GQA (48 / 8 heads)', 248),
            ('M3 · 2026', LAYOUTS['MiniMax-M3'], '3 dense + 57 MSA: reads 16 × 128 blocks', 120)]
    for i, (name, cells, note, kv) in enumerate(rows):
        y = 64 + i*44
        f.text(x0 - 12, y + 13, name, 14, 'ink', True, 'end')
        draw_strip(f, x0, y, cells, W, 18)
        f.text(x0, y + 33, note, 12, 'muted', anchor='start')
        w = 150*kv/248
        f.rect(x0 + W + 14, y + 2, w, 14, 'blue' if kv < 248 else 'red', .85, 2)
        f.text(x0 + W + 18 + w, y + 14, f'{kv} KiB', 12, 'blue' if kv < 248 else 'red', True, 'start')
    f.text(x0 + W + 14, 60, 'KV/tok, BF16', 12, 'muted', anchor='start')
    for k, (lab, tone) in enumerate((('lightning (linear)', 'purple'), ('full softmax', 'blue'), ('sparse MSA', 'green'))):
        f.rect(x0 + k*150, 240, 10, 10, tone, .85, 2)
        f.text(x0 + 15 + k*150, 249, lab, 12, 'muted', anchor='start')
    f.hdivider(262)
    f.text(30, 288, 'Why M2 went back to full attention (MiniMax blog, 2025-10-30)', 15, 'ink', True, 'start')
    reasons = [('1', 'Hidden cost', 'Multi-hop reasoning deficits showed only at scale; old benchmarks missed them.'),
               ('2', 'Evidence is expensive', 'Hard tasks need huge runs for a signal; precision bugs surfaced in M1 RL.'),
               ('3', 'Infrastructure', 'Memory-bound kernels; low-precision state; prefix caching; speculative decoding.'),
               ('4', 'SWA retrofit failed', 'Continued pretraining into an SWA hybrid degraded as context grew.')]
    for i, (n, head, body) in enumerate(reasons):
        y = 314 + i*28
        f.circle(42, y - 5, 10, n, 'red', True, 12)
        f.text(60, y, head, 14, 'red', True, 'start')
        f.text(232, y, body, 13, 'ink', anchor='start')
    f.text(30, 440, 'Same blog: the theoretical compute/memory crossover lies at a few thousand tokens.', 13, 'muted', anchor='start')
    f.save()


# ---------------------------------------------------------------- card 6
def layouts():
    order = [('Jamba v0.1', 'Jamba', 2024, 16, 'Mamba-1 1:7'),
             ('MiniMax-Text-01', 'MiniMax-Text-01', 2025, 40, 'lightning 7:1'),
             ('Gemma 3 27B', 'Gemma 3 27B', 2025, 80, 'SWA-1024 5:1'),
             ('Nemotron-H-56B *', 'Nemotron-H-56B', 2025, 40, 'Mamba-2, 10 attn'),
             ('Falcon-H1-34B', 'Falcon-H1-34B', 2025, 144, 'parallel'),
             ('gpt-oss-120b', 'gpt-oss-120b', 2025, 36, 'SWA-128 1:1'),
             ('Qwen3-Next-80B', 'Qwen3-Next-80B', 2025, 24, 'GDN 3:1'),
             ('MiniMax-M2', 'MiniMax-M2', 2025, 248, 'full'),
             ('Kimi Linear 48B', 'Kimi Linear 48B', 2025, 7.9, 'KDA ~3:1, MLA'),
             ('Nemotron 3 Nano *', 'Nemotron 3 Nano', 2025, 6, 'Mamba-2, 6 attn'),
             ('Qwen3.5-397B', 'Qwen3.5-397B', 2026, 30, 'GDN 3:1'),
             ('MiniMax-M3', 'MiniMax-M3', 2026, 120, 'sparse MSA'),
             ('Kimi K3', 'Kimi K3', 2026, 27, 'KDA ~3:1, MLA')]
    f = Fig('hybrid-layouts', 'Layer strips for 13 production models 2024–2026, read from config.json and scaled to equal '
            'width: blue global attention, light blue sliding window, purple recurrent, green sparse, split cells parallel. '
            'Right: KV per token for the growing layers in BF16, computed from config, log scale: Jamba 16 KiB, '
            'MiniMax-Text-01 40, Gemma 3 27B 80, Nemotron-H-56B 40, Falcon-H1-34B 144, gpt-oss-120b 36, Qwen3-Next 24, '
            'MiniMax-M2 248, Kimi Linear 7.9, Nemotron 3 Nano 6, Qwen3.5-397B 30, MiniMax-M3 120, Kimi K3 27; '
            'Llama-3.1-70B reference 320.', 470)
    f.math(450, 32, r'\text{KV/tok} = \sum_{ℓ \in \c{blue}{G}} c_ℓ' + EM + r'\text{(global layers only; SWA bounded by } W\text{, recurrent by } |S|)', 19, anchor='middle')
    W, x0, y0, rh = 400, 196, 66, 28
    f.text(x0, y0 - 6, 'layer 1 → L (equal width per model)', 12, 'muted', anchor='start')
    f.text(x0 + W + 10, y0 - 6, 'L', 12, 'muted', anchor='start')
    bx, bw = 700, 120
    lg = lambda v: bw*math.log10(v)/math.log10(320)
    f.text(bx, y0 - 6, 'KV KiB/tok (log)', 12, 'muted', anchor='start')
    for i, (label, key, year, kv, kind) in enumerate(order):
        y = y0 + i*rh
        cells = LAYOUTS[key]
        f.text(x0 - 10, y + 12, label, 13, 'ink', True, 'end')
        f.text(x0 - 10, y + 24, f'{year} · {kind}', 10.5, 'muted', anchor='end')
        draw_strip(f, x0, y, cells, W, 16, gap=.8 if len(cells) > 70 else 1.2)
        f.text(x0 + W + 10, y + 13, str(len(cells)), 12, 'muted', anchor='start')
        tone = 'red' if kv >= 100 else 'blue' if kv >= 20 else 'green'
        f.rect(bx, y + 1, max(3, lg(kv)), 14, tone, .85, 2)
        f.text(bx + max(3, lg(kv)) + 6, y + 13, kib(kv), 12, tone, True, 'start')
    y = y0 + len(order)*rh
    f.line(bx + lg(320), y0 - 2, bx + lg(320), y - 6, 'red', 1, True)
    f.text(bx + lg(320) - 4, y + 6, 'Llama-3.1-70B: 320', 11, 'red', anchor='end')
    for k, (lab, tone, op) in enumerate((('global', 'blue', .9), ('sliding window', 'blue', .3), ('recurrent', 'purple', .8),
                                          ('sparse', 'green', .8))):
        f.rect(x0 + k*100, y + 4, 10, 10, tone, op, 2)
        f.text(x0 + 15 + k*100, y + 13, lab, 12, 'muted', anchor='start')
    f.text(x0, y + 32, 'split cell = attention ∥ SSM in one layer · * token mixers only (FFN/MoE omitted)', 11, 'muted', anchor='start')
    f.save()


def build():
    why_hybrid()
    seq_par()
    nvidia()
    delta31()
    minimax()
    layouts()
