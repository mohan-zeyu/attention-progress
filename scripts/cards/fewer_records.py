"""08 · Compress the sequence: merge many positions into fewer learned records."""
import math
from figkit import Fig

M = 2**20                       # 1M tokens
ENTRY = 448*1 + 64*2            # V4 KV entry: 448 FP8 dims + 64 BF16 RoPE dims = 576 B
FLASH = [0, 0] + [4 if i % 2 == 0 else 128 for i in range(41)]     # config compress_ratios, 43 main layers
PRO = [128, 128] + [4 if i % 2 == 0 else 128 for i in range(59)]   # 61 main layers


def strip(f, parts, y=36, size=21, gap=34, x=30):
    """Formula strip: parts rendered left to right with real gaps (SVG collapses repeated spaces)."""
    for part in parts:
        x += f.math(x, y, part, size) + gap


def compressive():
    f = Fig('compress-compressive', 'Compressive Transformer. Toy: memories that leave the FIFO are pooled 3 to 1 into a compressed memory instead of being dropped. Real Enwik8 setting: a Transformer-XL with 2,304 memory slots reaches 2,304 past tokens per layer; the Compressive Transformer attends to fewer slots (768 memory + 1,152 compressed) yet reaches 4,224 past tokens per layer at rate c = 3.', 420)
    strip(f, [r'\c{blue}{cm} \leftarrow f_c(m_{oldest \; n_s}) ∈ ℝ^{⌊n_s / c⌋ × d}', r'\text{range} = L \cdot (n_m + \c{green}{c} \cdot n_{cm})'])
    # toy bands: old → new
    y = 116
    f.text(30, 74, 'toy: n_s = 3, n_m = 6, n_cm = 4, c = 3  (time →)', 14, 'muted', anchor='start')
    xs = 70
    f.box(xs - 52, y, 34, 34, '×', 'red', True, 16, .85)
    f.text(xs - 35, y + 54, 'drop', 13, 'red', True)
    for i in range(4):
        f.box(xs + i*50, y, 45, 34, '3 tok', 'blue', True, 13, .9)
    f.brace(xs, y + 40, xs + 4*50 - 5, y + 40, 'compressed n_cm', 'blue', 'below', 13)
    mx = xs + 4*50 + 22
    for i in range(6):
        old = i < 3
        f.box(mx + i*36, y, 31, 34, '', 'blue', old, 13, .35 if old else 1)
    f.brace(mx, y + 40, mx + 6*36 - 5, y + 40, 'memory n_m (FIFO)', 'blue', 'below', 13)
    sx = mx + 6*36 + 22
    for i in range(3):
        f.box(sx + i*36, y, 31, 34, '', 'gold', True, 13, .8)
    f.brace(sx, y + 40, sx + 3*36 - 5, y + 40, 'segment n_s', 'gold', 'below', 13)
    # arrow: three oldest memories → one compressed slot
    f.path([(mx + 50, y - 4), (mx - 10, y - 36), (xs + 3*50 + 22, y - 4)], 'teal', True, 2, curve=True)
    f.text(mx + 40, y - 22, 'f_c: 3 → 1', 13, 'teal', True, 'start')
    f.path([(xs + 18, y - 4), (xs - 8, y - 22), (xs - 34, y - 4)], 'red', True, 1.6, curve=True)
    # options panel
    px = 670
    f.text(px, 92, 'f_c tried:', 14, 'muted', anchor='start')
    for i, s in enumerate(['max / mean pool', '1D conv (best, WikiText)', 'dilated conv', 'most-used (≈ H2O-like)']):
        f.text(px, 114 + i*19, s, 14, 'ink', anchor='start')
    f.text(px, 194, '+ attention-reconstruction loss', 13, 'teal', anchor='start')
    f.hdivider(212, 30, 870)
    # Enwik8 A vs B
    f.text(30, 232, 'Enwik8, 24 layers, segment 768 (real config)', 15, 'ink', True, 'start')
    f.text(30, 256, 'slots each query attends', 14, 'muted', anchor='start')
    f.bars(30, 266, [('Transformer-XL', 768 + 2304, 'blue'), ('Compressive', 768 + 768 + 1152, 'blue')], width=180, bar=20, gap=8, label_w=130, size=14,
           vmax=4224, value_fmt=lambda v: f'{v:,}')
    f.text(470, 256, 'past tokens reachable per layer', 14, 'muted', anchor='start')
    f.bars(470, 266, [('Transformer-XL', 2304, 'line'), ('Compressive', 768 + 3*1152, 'green')], width=180, bar=20, gap=8, label_w=130, size=14,
           vmax=4224, value_fmt=lambda v: f'{v:,}')
    f.text(470, 344, '768 + 3 · 1,152 = 4,224 → × 24 layers ≈ 101K tokens', 13, 'green', anchor='start')
    f.text(30, 344, 'memory 2,304 vs 768 + compressed 1,152 (c = 3)', 13, 'muted', anchor='start')
    f.text(30, 390, 'reported: Enwik8 0.97 bpc (c = 3) · WikiText-103 17.1 ppl (conv, c = 4) · PG-19 33.6 vs Transformer-XL 36.3 (c = 2)', 14, 'ink', anchor='start')
    f.save()


def beacon():
    f = Fig('compress-beacon', 'Learned summary tokens. Gist: a toy 10-position mask where tokens after the gist token G cannot attend to the prompt, so only G\'s KV needs caching. Activation Beacon: every 1,024-token chunk of Llama-2 leaves 1,024/alpha beacon KVs (128 at alpha = 8) and drops the raw KV. At 128K a Llama-2-7B cache falls from 64 GiB to about 8.5 GiB.', 450)
    strip(f, [r'\text{chunk of } w \text{ tokens} \to w / \c{green}{\alpha} \text{ beacon KVs, raw KV dropped}', r'|\text{KV}| ≈ N / \c{green}{\alpha} + w'])
    # gist mask
    labels = ['P', 'P', 'P', 'P', 'G', 'x', 'x', 'x', 'y', 'y']
    n, c, x0, y0 = 10, 18, 70, 98
    f.text(30, 76, 'Gist · attention mask (toy)', 15, 'ink', True, 'start')
    def cell(i, j):
        if j > i:
            return None
        if i > 4 and j < 4:
            return ('red', .35)
        if j == 4:
            return ('green', .9)
        if j < 4:
            return ('gold', .6)
        return ('blue', .55)
    f.mask(x0, y0, n, c, cell)
    for k, lab in enumerate(labels):
        f.text(x0 + k*c + 8, y0 - 6, lab, 13, 'green' if lab == 'G' else 'muted', lab == 'G')
        f.text(x0 - 8, y0 + k*c + 14, lab, 13, 'green' if lab == 'G' else 'muted', lab == 'G', 'end')
    f.text(x0 + n*c + 12, y0 + 40, 'P prompt', 13, 'gold', anchor='start')
    f.text(x0 + n*c + 12, y0 + 58, 'G gist token', 13, 'green', anchor='start')
    f.text(x0 + n*c + 12, y0 + 76, 'x input', 13, 'blue', anchor='start')
    f.text(x0 + n*c + 12, y0 + 94, 'y output', 13, 'blue', anchor='start')
    f.text(30, y0 + n*c + 24, 'red: after G, prompt P is masked', 13, 'red', anchor='start')
    f.text(30, y0 + n*c + 42, '→ cache G only, not P', 13, 'green', True, 'start')
    f.divider(340, 62, 330)
    # beacon chunks
    bx = 360
    f.text(bx, 76, 'Activation Beacon · Llama-2, chunk w = 1,024, α = 8', 15, 'ink', True, 'start')
    cy = 96
    f.text(bx, cy + 15, 'chunk', 13, 'muted', anchor='start')
    for u in range(15):
        x = bx + 50 + u*28
        f.rect(x, cy, 21, 20, 'blue', .3, 2)
        f.rect(x + 21, cy, 5, 20, 'green', .95, 1)
    f.text(bx + 50 + 15*28 + 4, cy + 15, '…', 15, 'muted', anchor='start')
    f.brace(bx + 50, cy + 24, bx + 50 + 26, cy + 24, '', 'muted', 'below', 13)
    f.text(bx + 84, cy + 46, '8 raw + 1 beacon, repeated 128× per chunk', 13, 'muted', anchor='start')
    f.text(bx, cy + 82, 'cache while reading chunk k + 1', 13, 'muted', anchor='start')
    for k in range(4):
        y = cy + 92 + k*26
        f.text(bx + 14, y + 15, f'k = {k+1}', 13, 'muted', anchor='start')
        for q in range(k + 1):
            f.rect(bx + 60 + q*48, y, 44, 20, 'green', .85, 3)
        f.rect(bx + 60 + (k + 1)*48, y, 150, 20, 'blue', .25, 3)
        f.text(bx + 60 + (k + 1)*48 + 75, y + 15, '1,024 raw', 12, 'blue')
        f.text(bx + 60 + (k + 1)*48 + 158, y + 15, f'{128*(k+1) + 1024:,} slots', 13, 'ink', True, 'start')
    f.text(bx, cy + 214, 'green = 128 beacon KVs per finished chunk', 13, 'green', anchor='start')
    f.text(bx, cy + 232, 'training samples α ∈ {2, 4, 8, 16, 32} per chunk', 13, 'muted', anchor='start')
    f.hdivider(342, 30, 870)
    full = 131072*512/2**20
    bea = (131072/8 + 1024)*512/2**20
    f.text(30, 366, 'Llama-2-7B KV at 128K (512 KiB/token, BF16)', 14, 'muted', anchor='start')
    f.bars(30, 378, [('full', full, 'blue'), ('Beacon α = 8', bea, 'green')], width=240, bar=20, gap=8, label_w=110, size=14,
           value_fmt=lambda v: f'{v:.1f} GiB')
    f.text(560, 368, 'reported: Beacon 8× less KV, 2× faster;', 14, 'ink', anchor='start')
    f.text(560, 388, 'trained < 20K tokens, tested to 128K', 14, 'ink', anchor='start')
    f.text(560, 412, 'gist: up to 26× prompt compression,', 14, 'ink', anchor='start')
    f.text(560, 432, 'up to 40% fewer FLOPs (LLaMA-7B)', 14, 'ink', anchor='start')
    f.save()


def nsa_cmp():
    f = Fig('compress-nsa', 'NSA compression branch. Toy positions 0 to 159: windows of l = 32 tokens slide by d = 16, so neighbouring windows overlap by 16 and each emits one compressed key. At 64K this gives 4,095 compressed keys that are read densely and double as the block scorer; the raw KV is kept for the selection and window branches, so the cache does not shrink.', 420)
    strip(f, [r'\c{blue}{K^{cmp}_t} = \{ φ(k_{id+1 : id+l}) : 0 \le i \le ⌊(t-l)/d⌋ \}', r'l = 32, \; d = 16'])
    x0, sc = 50, 4.6           # px per token
    y = 132
    f.text(30, 70, 'toy positions 0 – 159 (real block l = 32, stride d = 16)', 14, 'muted', anchor='start')
    for t in range(160):
        f.rect(x0 + t*sc, y, sc - 1, 16, 'blue', .45 if (t // 16) % 2 == 0 else .25, 0)
    for k in (0, 32, 64, 96, 128, 160):
        f.text(x0 + k*sc, y + 32, str(k), 12, 'muted')
    nwin = (160 - 32)//16 + 1
    for i in range(nwin):
        a = i*16
        row = i % 2
        yy = y - 10 - row*12
        f.line(x0 + a*sc, yy, x0 + (a + 32)*sc - 2, yy, 'teal', 3)
        cx = x0 + (a + 16)*sc
        f.arrow(cx, y + 38, cx, y + 60, 'teal', width=1.2)
        f.box(cx - 14, y + 62, 28, 24, f'{i}', 'blue', True, 13)
    f.brace(x0, y - 36, x0 + 32*sc - 2, y - 36, 'l = 32', 'teal', 'above', 13, 2)
    f.brace(x0 + 32*sc, y - 36, x0 + 48*sc - 2, y - 36, 'd = 16', 'teal', 'above', 13, 2)
    f.text(x0 + 160*sc + 14, y + 80, f'{nwin} keys', 14, 'blue', True, 'start')
    f.text(x0 + 160*sc + 14, y + 12, 'φ: MLP with', 13, 'muted', anchor='start')
    f.text(x0 + 160*sc + 14, y + 28, 'intra-block', 13, 'muted', anchor='start')
    f.text(x0 + 160*sc + 14, y + 44, 'positions', 13, 'muted', anchor='start')
    f.hdivider(236, 30, 870)
    f.text(30, 262, 'NSA per query at 64K (GQA group reads, 27B model)', 15, 'ink', True, 'start')
    rows = [('compressed keys', (65536 - 32)//16 + 1, 'blue'), ('selected 16 × 64', 1024, 'green'), ('window', 512, 'gold')]
    f.bars(30, 276, rows, width=260, bar=20, gap=8, label_w=150, size=14, value_fmt=lambda v: f'{v:,}')
    f.text(30, 384, 'reads ≈ 5,631 of 65,536 (8.6%); cmp scores, summed per 64-block, pick the 16 blocks', 13, 'muted', anchor='start')
    f.text(560, 280, 'stored per layer', 14, 'muted', anchor='start')
    f.text(560, 304, 'raw KV 65,536 (kept)', 15, 'blue', True, 'start')
    f.text(560, 326, '+ compressed 4,095', 15, 'blue', anchor='start')
    f.text(560, 352, 'compression here is for', 14, 'ink', anchor='start')
    f.text(560, 372, 'reading, not storage →', 14, 'ink', anchor='start')
    f.text(560, 392, 'CSA drops the raw KV', 14, 'green', True, 'start')
    f.save()


def csa():
    f = Fig('compress-csa', 'CSA in DeepSeek-V4. Toy with m = 4: entry i pools the 8 positions of blocks i-1 and i with a per-channel softmax, and entries are emitted every 4 positions, so windows overlap. At 1M tokens a CSA layer stores 262,144 entries; an FP4 indexer picks the top 1,024 (Pro) or 512 (Flash), and attention reads those plus 128 raw window entries and a sink logit.', 430)
    strip(f, [r'\c{blue}{c_i} = \sum_{j ∈ \text{blk}_{i-1} ∪ \text{blk}_i} \text{softmax}_j(\c{teal}{z_j + B_j}) \odot x_j', r'A_t = \text{Top}_k(\c{gold}{I_{t,:}}) ∪ \text{win}_{128}'], size=20)
    x0, c, y = 120, 28, 136
    f.text(30, 70, 'toy, m = 4 (real m = 4, Flash and Pro)', 14, 'muted', anchor='start')
    f.text(x0 - 10, y + 16, 'tokens', 13, 'muted', anchor='end')
    for t in range(24):
        f.rect(x0 + t*c, y, c - 3, 22, 'blue', .5 if (t // 4) % 2 == 0 else .25, 2)
        f.text(x0 + t*c + (c-3)/2, y + 16, str(t), 12, 'ink')
    for i in range(6):
        a = max(0, (i - 1)*4)
        b = (i + 1)*4
        row = i % 2
        yy = y - 10 - row*12
        if i == 0:
            f.line(x0 - 4*c + 4, yy, x0 - 4, yy, 'line', 3, dash=True)
            f.text(x0 - 2*c, yy - 6, 'pad −∞', 12, 'muted')
        f.line(x0 + a*c, yy, x0 + b*c - 4, yy, 'teal', 3)
        cx = x0 + (i*4 + 2)*c - 2
        f.arrow(cx, y + 26, cx, y + 50, 'teal', width=1.2)
        f.box(cx - 17, y + 52, 34, 24, f'c{i}', 'blue', True, 13)
    f.brace(x0 + 4*c, y - 36, x0 + 12*c - 4, y - 36, '2m = 8 sources for c2', 'teal', 'above', 13, 2)
    f.brace(x0 + 12*c, y - 36, x0 + 16*c - 4, y - 36, 'stride m = 4', 'teal', 'above', 13, 2)
    # per-channel softmax inset
    ix, iy = 800, 100
    f.text(ix + 30, iy - 8, 'weights for c₂', 13, 'teal', True)
    vals = [[.05, .1, .2, .6, .3, .1, .05, .02], [.4, .3, .1, .05, .05, .05, .02, .03], [.1, .1, .1, .1, .2, .2, .1, .1]]
    for ch in range(3):
        col = vals[ch]
        s = sum(col)
        for r in range(8):
            f.rect(ix + ch*20, iy + r*13, 18, 11, 'teal', .1 + .9*col[r]/max(col), 1)
    f.text(ix + 30, iy + 122, '8 × c, softmax', 12, 'muted')
    f.text(ix + 30, iy + 136, 'down each column', 12, 'muted')
    f.hdivider(250, 30, 870)
    # read path at 1M
    f.text(30, 276, 'one query at 1M tokens (V4-Pro)', 15, 'ink', True, 'start')
    bx, by, bw = 30, 292, 520
    f.rect(bx, by, bw, 18, 'blue', .2, 2)
    import random
    rnd = random.Random(4)
    for _ in range(60):
        xx = bx + rnd.random()*bw
        f.rect(xx, by, 2, 18, 'green', .95, 0)
    f.rect(bx + bw - 6, by, 6, 18, 'gold', .95, 0)
    f.text(bx, by + 36, '262,144 entries (N / 4) · FP4 indexer, 64 heads × 128, scores all', 13, 'muted', anchor='start')
    f.text(bx, by + 56, 'green: top-k = 1,024 (Pro) / 512 (Flash) · gold: 128 raw window', 13, 'green', anchor='start')
    f.text(bx, by + 76, 'core read = 1,024 + 128 entries + per-head sink logit (ch 02)', 13, 'ink', anchor='start')
    f.text(bx, by + 96, 'entry = 512 dims, shared K = V (MQA); FP8 + BF16 RoPE dims', 13, 'muted', anchor='start')
    # cost readout: log bars
    rx = 570
    f.text(rx, 276, 'per layer, 1M tokens', 14, 'muted', anchor='start')
    def logbar(y, label, v, colr):
        w = 140*math.log10(v)/6
        f.text(rx + 84, y + 14, label, 13, 'ink', anchor='end')
        f.rect(rx + 90, y, w, 18, colr, .85, 3)
        f.text(rx + 94 + w, y + 14, f'{v:,}', 13, colr, True, 'start')
    logbar(290, 'MLA stores', 1048576, 'blue')
    logbar(314, 'CSA stores', 262144 + 128, 'blue')
    logbar(346, 'MLA reads', 1048576, 'red')
    logbar(370, 'CSA reads', 1152, 'green')
    f.text(rx + 90, 410, 'log scale', 12, 'muted', anchor='start')
    f.save()


def hca():
    f = Fig('compress-hca', 'HCA in DeepSeek-V4: every 128 tokens become one entry with no overlap, and queries read all entries densely plus a 128-token raw window. Entries read per query: at 8K, MLA 8,192, CSA 1,152, HCA 192; at 128K, 131,072, 1,152 and 1,152; at 1M, 1,048,576, 1,152 and 8,320. CSA also pays an indexer pass over N/4 entries.', 420)
    strip(f, [r'\c{blue}{c_i} = \sum_{j ∈ \text{blk}_i} \text{softmax}_j(\c{teal}{z_j + B_j}) \odot x_j, \; |\text{blk}_i| = 128', r'A_t = \{ c_s : s < ⌊t/128⌋ \} ∪ \text{win}_{128}'], size=20)
    # three rows at 1M
    bx, bw = 160, 480
    f.text(30, 78, 'one layer at 1M tokens', 15, 'ink', True, 'start')
    rows = [('MLA', '1,048,576 records, all read', 'blue', 1.0), ('CSA (m = 4)', '262,144 entries, 1,024 read', 'blue', .25), ('HCA (m′ = 128)', '8,192 entries, all read', 'green', .9)]
    for r, (name, sub, col, op) in enumerate(rows):
        y = 94 + r*40
        f.text(bx - 12, y + 16, name, 14, 'ink', True, 'end')
        if name.startswith('HCA'):
            for k in range(64):
                f.rect(bx + k*bw/64, y, bw/64 - 2, 22, 'green', .85, 1)
        elif name.startswith('CSA'):
            f.rect(bx, y, bw, 22, 'blue', .2, 2)
            for k in range(40):
                f.rect(bx + ((k*37) % 100)/100*bw, y, 2, 22, 'green', .95, 0)
        else:
            f.rect(bx, y, bw, 22, 'blue', .75, 2)
        f.text(bx + bw + 12, y + 16, sub, 13, 'ink', anchor='start')
    f.text(bx, 222, 'HCA = Compressive-Transformer-style dense read, with learned pooling and m′ = 128; no indexer', 13, 'muted', anchor='start')
    f.hdivider(238, 30, 870)
    # entries read vs N, log scale grouped bars
    f.text(30, 264, 'entries read per query (log scale) · V4-Pro settings, 128-token window included', 14, 'ink', True, 'start')
    Ns = [(8192, '8K'), (131072, '128K'), (M, '1M')]
    gx, gy = 80, 282
    for g, (N, lab) in enumerate(Ns):
        vals = [('MLA', N, 'blue'), ('CSA', min(N//4, 1024) + 128, 'green'), ('HCA', N//128 + 128, 'purple')]
        x = gx + g*280
        f.text(x - 12, gy + 46, lab, 15, 'ink', True, 'end')
        for k, (name, v, colr) in enumerate(vals):
            y = gy + k*26
            w = 130*(math.log10(v) - 2)/4
            f.rect(x, y, w, 20, colr, .85, 3)
            f.text(x + w + 6, y + 15, f'{name} {v:,}', 13, colr, True, 'start')
    f.text(30, 380, 'CSA also scores N/4 entries with its FP4 indexer: 2,048 · 32,768 · 262,144. HCA has no scorer.', 13, 'muted', anchor='start')
    f.text(30, 404, 'At 1M, HCA reads 7× more entries than CSA but each entry spans 128 tokens; CSA entries span 8 with overlap.', 13, 'muted', anchor='start')
    f.save()


def v4_layout():
    def per_tok(ratios):
        return sum(ENTRY/m for m in ratios if m)
    pro, flash = per_tok(PRO), per_tok(FLASH)
    v32 = 61*656
    gqa = 61*2*8*128*2
    f = Fig('compress-v4', f'DeepSeek-V4 layer layouts from config.json: Flash has 2 sliding-window-only layers, then 21 CSA and 20 HCA layers alternating; Pro has 2 HCA layers, then 30 CSA and 29 more HCA layers. Derived main-KV bytes per token (576-byte entries, scales and indexer keys left out): Pro about {pro:,.0f} B, Flash about {flash:,.0f} B, versus 40,016 B for DeepSeek-V3.2 MLA in FP8 and 249,856 B for a BF16 GQA8 baseline.', 450)
    strip(f, [r'\text{KV bytes / token} = \sum_{\ell} \c{blue}{b_{entry}} / \c{green}{m_{\ell}}', r'\c{blue}{b_{entry}} = 448 \cdot 1 + 64 \cdot 2 = 576 \text{ B}'], size=20)
    f.text(870, 36, 'FP8 + BF16 RoPE', 13, 'muted', anchor='end')
    w, x0 = 13, 74
    cols = {0: ('line', .9), 4: ('blue', .9), 128: ('blue', .3)}
    for r, (name, ratios) in enumerate([('V4-Flash · 43 layers', FLASH), ('V4-Pro · 61 layers', PRO)]):
        y = 78 + r*56
        n4, n128, n0 = ratios.count(4), ratios.count(128), ratios.count(0)
        f.text(30, y, name, 14, 'ink', True, 'start')
        f.text(200, y, f'{n4} CSA · {n128} HCA' + (f' · {n0} sliding-window-only' if n0 else ''), 13, 'muted', anchor='start')
        for l, m in enumerate(ratios):
            colr, op = cols[m]
            f.rect(x0 - 44 + l*w, y + 8, w - 2, 24, colr, op, 2)
    lx = 30
    for colr, op, lab, dx in [('blue', .9, 'CSA m = 4 (+ indexer)', 0), ('blue', .3, 'HCA m′ = 128', 210), ('line', .9, 'sliding window only', 360)]:
        f.rect(lx + dx, 192, 14, 14, colr, op, 2)
        f.text(lx + dx + 20, 204, lab, 13, 'ink', anchor='start')
    f.text(560, 204, 'every layer also keeps a 128-entry raw window', 13, 'muted', anchor='start')
    f.hdivider(222, 30, 870)
    f.text(30, 248, 'main KV per token (KiB per token = GiB per 1M tokens)', 14, 'ink', True, 'start')
    rows = [('BF16 GQA8, d_h 128, 61 L', gqa/1024, 'red'), ('V3.2 MLA, FP8 656 B × 61', v32/1024, 'blue'),
            ('V4-Pro (derived)', pro/1024, 'green'), ('V4-Flash (derived)', flash/1024, 'green')]
    f.bars(30, 262, rows, width=290, bar=22, gap=10, label_w=210, size=14, value_fmt=lambda v: f'{v:,.1f} KiB')
    rx = 660
    f.text(rx, 268, 'reported (V4 report):', 14, 'muted', anchor='start')
    f.text(rx, 290, 'Pro: 10% of V3.2 KV at 1M', 14, 'ink', anchor='start')
    f.text(rx, 310, 'Flash: 7% of V3.2 KV', 14, 'ink', anchor='start')
    f.text(rx, 330, '≈ 2% of BF16 GQA8', 14, 'ink', anchor='start')
    f.text(rx, 358, 'derived here (main KV only):', 13, 'muted', anchor='start')
    f.text(rx, 376, f'Pro {pro/v32:.0%} · Flash {flash/v32:.0%} of V3.2', 13, 'green', True, 'start')
    f.text(rx, 394, f'Pro {pro/gqa:.1%} of GQA8', 13, 'green', True, 'start')
    f.text(30, 410, f'Pro: 30 CSA × 576/4 + 31 HCA × 576/128 = {pro:,.1f} B · window: 61 × 128 × 576 B = 4.3 MiB per sequence', 13, 'muted', anchor='start')
    f.text(30, 432, 'indexer keys: FP4, 128 dims = 64 B per CSA entry → +16 B/token per CSA layer (not in bars) · V4.1-Flash: 890 B reported (ch 14)', 13, 'muted', anchor='start')
    f.save()


def build():
    compressive()
    beacon()
    nsa_cmp()
    csa()
    hca()
    v4_layout()
