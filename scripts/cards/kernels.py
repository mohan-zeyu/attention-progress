"""13 · Kernels: the same attention, fewer bytes moved and fuller GPUs."""
import math
from figkit import Fig, COL


# ---------------------------------------------------------------- helpers
def pair(f, x, y, title, a, b, unit, w=150, fmt='{:g}'):
    """Two bars (standard vs flash) under a small title; a/b = (label, value, colour)."""
    f.text(x, y, title, 14, 'ink', True, 'start')
    vmax = max(a[1], b[1])
    for i, (lab, v, c) in enumerate((a, b)):
        yy = y + 12 + i*26
        f.text(x+70, yy+15, lab, 13, 'muted', anchor='end')
        bw = max(2, w*v/vmax)
        f.rect(x+78, yy, bw, 20, c, .85, 3)
        f.text(x+84+bw, yy+15, fmt.format(v) + unit, 14, c, True, 'start')


def logbars(f, x, y, items, width, lo, hi, bar=20, gap=10, label_w=200, size=14):
    """Horizontal bars on a log axis: items = [(label, value, colour, shown)]."""
    span = math.log10(hi) - math.log10(lo)
    for i, (lab, v, c, shown) in enumerate(items):
        yy = y + i*(bar+gap)
        f.text(x+label_w-10, yy+bar*.7, lab, size, 'ink', anchor='end')
        w = max(3, width*(math.log10(v) - math.log10(lo))/span)
        f.rect(x+label_w, yy, w, bar, c, .85, 3)
        f.text(x+label_w+w+8, yy+bar*.7, shown, size, 'muted' if c == 'line' else c, True, 'start')


# ---------------------------------------------------------------- 1. FlashAttention tiles
def flash_tiles():
    f = Fig('kern-flash-tiles', 'FlashAttention keeps Q, K, V and O in HBM and streams tiles through on-chip SRAM, so the N×N score and probability matrices are never written. A100: HBM 1.5–2.0 TB/s, SRAM 192 KB per SM at about 19 TB/s. Example tiles of 128 rows with d = 64 fit in 144 KiB. Reported for GPT-2 medium attention (N = 1024, d = 64) on A100: HBM traffic 40.3 GB to 4.4 GB, runtime 41.7 ms to 7.3 ms, while FLOPs rise from 66.6 to 75.2 G because of recomputation.', 452)
    f.math(30, 36, r'\text{HBM accesses:  standard } Θ(Nd + \c{red}{N^2}) \quad \to \quad \text{FlashAttention } Θ(N^2 d^2 / \c{green}{M}), \quad M = \text{SRAM size}', 20)
    # HBM panel
    f.panel(30, 58, 330, 238, 'HBM · A100 40–80 GB · 1.5–2.0 TB/s')
    cols = [('Q', 'gold'), ('K', 'blue'), ('V', 'purple'), ('O', 'green')]
    for k, (name, c) in enumerate(cols):
        x = 56 + k*46
        f.rect(x, 104, 34, 142, c, .22, 3)
        band = 130 if name in ('Q', 'O') else 178
        f.rect(x, band, 34, 26, c, .9, 3)
        f.text(x+17, 266, name, 16, c, True)
    f.brace(56, 104, 56, 246, '$N', 'muted', 'left', 13, 6)
    f.math(73, 98, 'd', 13, 'muted', 'middle')
    f.math(46, 286, r'\text{tiles } Q_i, O_i \text{ (rows } i) \;\cdot\; K_j, V_j \text{ (rows } j)', 14, 'muted')
    # the N x N matrices that are never stored
    f.box(246, 106, 98, 98, '', 'red', dash=True)
    f.line(246, 106, 344, 204, 'red', 1.2, opacity=.6)
    f.line(344, 106, 246, 204, 'red', 1.2, opacity=.6)
    f.math(295, 228, r'\c{red}{S, P \in \R^{N\times N}}', 16, anchor='middle')
    f.text(295, 248, 'never written', 14, 'red', True)
    # arrows
    f.arrow(364, 140, 444, 140, 'blue', width=2.4)
    f.text(404, 130, 'load tiles', 13, 'blue', True)
    f.arrow(444, 196, 364, 196, 'green', width=2.4)
    f.math(404, 216, r'\text{write } O_i \text{ once}', 14, 'green', 'middle')
    # SRAM panel
    f.panel(450, 58, 420, 238, 'SRAM · 192 KB per SM × 108 · ≈19 TB/s')
    y0 = 102
    def t(x, w, c, lab, shape, kib):
        f.rect(x, y0, w, 66, c, .85 if c != 'ink' else .12, 4)
        f.math(x+w/2, y0+39, lab, 17, 'white' if c != 'ink' else 'ink', 'middle')
        f.math(x+w/2, y0+84, shape, 13, 'muted', 'middle')
        f.text(x+w/2, y0+101, kib, 13, 'muted')
    t(466, 52, 'gold', 'Q_i', r'B_r \times d', '16 KiB')
    f.text(527, y0+40, '·', 22, 'ink')
    t(536, 52, 'blue', r'K_j^\top', r'B_c \times d', '16 KiB')
    f.text(597, y0+38, '→', 18, 'ink')
    t(607, 70, 'ink', 'S_{ij}', r'B_r \times B_c', '64 KiB')
    f.text(686, y0+40, '·', 22, 'ink')
    t(695, 52, 'purple', 'V_j', r'B_c \times d', '16 KiB')
    f.text(756, y0+38, '→', 18, 'ink')
    t(766, 88, 'green', r'Õ_i, m_i, ℓ_i', r'B_r \times d', '32 KiB')
    f.text(466, 236, 'example: B = 128 rows, d = 64, BF16; S and Õ in FP32', 13, 'muted', anchor='start')
    f.text(466, 256, '16 + 16 + 64 + 16 + 32 = 144 KiB  <  192 KB', 14, 'green', True, 'start')
    f.text(466, 278, 'loop j over K/V tiles, i over Q tiles (FA-1 Alg. 1)', 13, 'muted', anchor='start')
    # reported numbers
    f.hdivider(310)
    f.text(30, 334, 'Reported (FA-1 Fig. 2): GPT-2 medium attention fwd + bwd, N = 1024, d = 64, 16 heads, batch 64, A100', 14, 'ink', anchor='start')
    pair(f, 30, 356, 'HBM read + write', ('standard', 40.3, 'red'), ('Flash', 4.4, 'green'), ' GB', 110)
    pair(f, 330, 356, 'runtime', ('standard', 41.7, 'red'), ('Flash', 7.3, 'green'), ' ms', 110)
    pair(f, 620, 356, 'FLOPs (recompute adds)', ('standard', 66.6, 'muted'), ('Flash', 75.2, 'gold'), ' G', 110)
    f.text(30, 440, 'Fewer bytes beats fewer FLOPs: 9× less traffic, 5.7× faster, 13% more arithmetic.', 14, 'green', True, 'start')
    f.save()


# ---------------------------------------------------------------- 2. online softmax toy
def online_softmax():
    f = Fig('kern-online-softmax', 'Toy worked example of the online softmax: one query row with scores 1, 3 | 2, 5 and scalar values 4, 2 | 6, 1 in two tiles. After tile 1: m = 3, ℓ = 1.135, unnormalised output 2.541. Tile 2 alone: m = 5, ℓ = 1.050, 1.299. Merging rescales the old statistics by e^(3−5) = 0.135, giving ℓ = 1.203, Õ = 1.643, o = 1.365, identical to a full softmax.', 372)
    f.math(30, 36, r"m′ = \max(m, \c{blue}{m_j})", 21)
    f.math(250, 36, r"\ell′ = \c{red}{e^{m−m′}}\ell + e^{\c{blue}{m_j}−m′}\c{blue}{\ell_j}", 21)
    f.math(560, 36, r"Õ′ = \c{red}{e^{m−m′}}Õ + \sum_j e^{s_j−m′} \c{purple}{v_j}", 21)
    # the row of scores
    f.text(30, 72, 'toy · one query row, 4 keys, 2 tiles', 14, 'muted', anchor='start')
    s, v = [1, 3, 2, 5], [4, 2, 6, 1]
    for k in range(4):
        x = 40 + k*58 + (14 if k > 1 else 0)
        op = .35 if k < 2 else .75
        f.rect(x, 104, 50, 34, 'blue', op, 4)
        f.text(x+25, 127, f's = {s[k]}', 14, 'ink' if k < 2 else 'white', True)
        f.rect(x, 142, 50, 28, 'purple', .2, 4)
        f.text(x+25, 161, f'v = {v[k]}', 13, 'purple', True)
    f.brace(40, 104, 148, 104, 'tile 1', 'blue', 'above', 13, 4)
    f.brace(170, 172, 278, 172, 'tile 2', 'blue', 'below', 13, 4)
    # table of statistics
    x0, y0 = 316, 80
    xs = [x0, 520, 642, 792]
    f.math(xs[1], y0, 'm', 17, 'ink', 'middle')
    f.math(xs[2], y0, r'\ell', 17, 'ink', 'middle')
    f.math(xs[3], y0, 'Õ', 17, 'ink', 'middle')
    rows = [
        ('after tile 1', '3', r'e^{−2} + 1 = 1.135', r'4e^{−2} + 2 = 2.541', 'ink'),
        ('tile 2 alone', '5', r'e^{−3} + 1 = 1.050', r'6e^{−3} + 1 = 1.299', 'blue'),
    ]
    for r, (lab, m, l, o, c) in enumerate(rows):
        y = y0 + 30 + r*30
        f.text(x0, y, lab, 14, c, anchor='start')
        f.text(xs[1], y, m, 15, c, True)
        f.math(xs[2], y, l, 15, c, 'middle')
        f.math(xs[3], y, o, 15, c, 'middle')
    y = y0 + 98
    f.rect(x0-8, y-20, 562, 30, 'red', .08, 4)
    f.text(x0, y, 'rescale the old (m, ℓ, Õ) by', 14, 'red', anchor='start')
    f.math(xs[2]+20, y, r'e^{3−5} = 0.135', 16, 'red', 'middle', bold=True)
    y += 34
    f.text(x0, y, 'merged', 14, 'ink', True, 'start')
    f.text(xs[1], y, '5', 15, 'ink', True)
    f.text(xs[2], y, '0.135·1.135 + 1.050', 14, 'ink')
    f.text(xs[3], y, '0.135·2.541 + 1.299', 14, 'ink')
    f.text(xs[2], y+21, '= 1.203', 15, 'ink', True)
    f.text(xs[3], y+21, '= 1.643', 15, 'ink', True)
    # result
    f.hdivider(262)
    f.math(30, 302, r"\c{green}{o} = Õ / \ell = 1.643 / 1.203 = \c{green}{1.365}", 22)
    f.math(450, 296, r'\text{check, full softmax: } \sum e^{s−5} = 1.203', 15, 'muted')
    f.math(450, 320, r'(4e^{−4} + 2e^{−2} + 6e^{−3} + 1) / 1.203 = \c{green}{1.365}', 15, 'muted')
    f.math(30, 352, r'\text{On chip between tiles: only } (m, \ell, Õ) \text{ per query row, i.e. } B_r (d + 2) \text{ numbers, never the score row.}', 15, 'ink')
    f.save()


# ---------------------------------------------------------------- 3. FA-2 -> FA-3 -> FA-4
def fa_generations():
    f = Fig('kern-fa-generations', 'Each FlashAttention version answers one GPU generation. The tensor cores outgrow everything else: matmul versus non-matmul throughput is 16× on A100 (312 vs 19.5 TFLOP/s FP32), 256× on H100 (989 vs 3.9 TFLOP/s exponential) and 512× on B200 (8192 vs 16 ops per clock per SM). FA-2 cuts non-matmul work and re-partitions; FA-3 overlaps with asynchronous TMA and warp specialization and adds FP8; FA-4 emulates part of exp on FMA units, skips most rescales and uses tensor memory. Reported utilization: FA-1 25–40% and FA-2 50–73% on A100, FA-2 35% on H100, FA-3 75% (740 TFLOP/s), FA-4 71% (1613 TFLOP/s on B200).', 456)
    f.math(30, 36, r'\text{matmul : non-matmul throughput} \quad \c{red}{16\times} \text{ (A100)} \;\to\; \c{red}{256\times} \text{ (H100)} \;\to\; \c{red}{512\times} \text{ (B200)}', 20)
    cols = [
        ('FA-2 · 2023', 'A100', ('matmul BF16', '312 TF'), ('FP32 non-matmul', '19.5 TF'), 16,
         ['divide by ℓ once, at the end', 'parallelize over N (Q blocks), not', '   only batch × heads', 'split Q (not K) over 4 warps:', '   no shared-memory reduction']),
        ('FA-3 · 2024', 'H100', ('matmul FP16', '989 TF'), ('exp (MUFU)', '3.9 TF'), 256,
         ['TMA producer warps feed', '   WGMMA consumer warps', 'ping-pong: softmax under the', '   other warpgroup’s GEMM', 'FP8: block scales + Hadamard']),
        ('FA-4 · 2026', 'B200', ('tensor ops/clk/SM', '8192'), ('MUFU exp ops/clk/SM', '16'), 512,
         ['exp2 as a polynomial on FMA', '   units for 10–25% of entries', 'rescale only if the max grows', '   by > 8 (log₂256)', 'tensor memory + 2-CTA MMA']),
    ]
    for k, (name, gpu, mm, nm, ratio, fixes) in enumerate(cols):
        x0 = 30 + k*284
        if k:
            f.divider(x0-14, 58, 290)
        f.text(x0, 74, name, 18, 'ink', True, 'start')
        f.text(x0+250, 74, gpu, 16, 'muted', True, 'end')
        f.rect(x0, 88, 220, 14, 'blue', .85, 3)
        f.text(x0, 120, f'{mm[0]} {mm[1]}', 13, 'blue', anchor='start')
        f.rect(x0, 128, max(1.5, 220/ratio), 14, 'red', .9, 2)
        f.text(x0+22, 140, f'{nm[0]} {nm[1]}', 13, 'red', anchor='start')
        f.text(x0+250, 140, f'{ratio}×', 18, 'red', True, 'end')
        for i, line in enumerate(fixes):
            y = 172 + i*22
            if not line.startswith('   '):
                f.circle(x0+4, y-5, 2.5, color='green', fill=True)
            f.text(x0+14, y, line.strip(), 14, 'ink', anchor='start')
    f.text(328, 284, 'ping-pong: 570 → 620–640 TFLOP/s', 13, 'muted', anchor='start')
    f.hdivider(300)
    f.text(30, 324, 'Reported utilization of peak (bar = % of peak)', 14, 'ink', True, 'start')
    rows = [('FA-1 · A100', 25, 40, 'line', '25–40%'), ('FA-2 · A100', 50, 73, 'blue', '50–73%'),
            ('FA-2 · H100', 35, 35, 'red', '35%'), ('FA-3 · H100 FP16', 75, 75, 'green', '75% = 740 TFLOP/s (FP8: ≈1.2 PFLOP/s)'),
            ('FA-4 · B200 BF16', 71, 71, 'green', '71% = 1613 TFLOP/s (1.3× cuDNN 9.13)')]
    for i, (lab, lo, hi, c, shown) in enumerate(rows):
        y = 336 + i*23
        f.text(190, y+14, lab, 14, 'ink', anchor='end')
        f.rect(200, y, 3.6*lo, 17, c, .85, 3)
        if hi > lo:
            f.rect(200+3.6*lo, y, 3.6*(hi-lo), 17, c, .35, 3)
        f.text(208+3.6*hi, y+14, shown, 14, 'muted' if c == 'line' else c, True, 'start')
    f.save()


# ---------------------------------------------------------------- 4. split-KV decoding
def split_kv():
    f = Fig('kern-split-kv', 'Split-KV decoding. Toy: batch 1 with 16 heads launches 16 thread blocks on a 108-SM A100, 15% of the SMs. Splitting each head’s KV into 6 chunks launches 96 blocks, 89%. Each chunk returns (m, ℓ, o) and a log-sum-exp reduction merges them exactly: chunk A (scores 1, 2; values 10, 20) gives o = 17.31, LSE = 2.313; chunk B (scores 3, 0; values 30, 0) gives o = 28.58, LSE = 3.049; weights 0.324 and 0.676 give 24.93, the full-softmax value. Reported for CodeLlama-34B on A100: up to 8× faster decoding and up to 50× faster attention than FlashAttention.', 452)
    f.math(30, 36, r'o = \sum_s \c{green}{w_s} o_s', 21)
    f.math(190, 36, r'\c{green}{w_s} = e^{\text{LSE}_s − \text{LSE}}', 21)
    f.math(400, 36, r'\text{LSE}_s = m_s + \log \ell_s', 21)
    f.math(640, 36, r'\text{LSE} = \log \sum_s e^{\text{LSE}_s}', 21)
    # SM grids
    def grid(x, y, busy, title, sub, c):
        f.text(x, y-10, title, 14, 'ink', True, 'start')
        for i in range(108):
            r, q = divmod(i, 12)
            f.rect(x + q*15, y + r*15, 13, 13, c if i < busy else 'line', .85 if i < busy else .45, 2)
        f.text(x, y+9*15+18, sub, 14, c, True, 'start')
    grid(30, 80, 16, 'one block per head', '16 / 108 SMs = 15%', 'red')
    grid(232, 80, 96, '6 KV splits per head', '96 / 108 SMs = 89%', 'green')
    f.text(30, 252, 'toy: batch 1, 16 heads, A100 (108 SMs)', 13, 'muted', anchor='start')
    # split diagram
    x0 = 470
    f.circle(x0+14, 104, 13, '$q', 'gold', True, 14)
    f.text(x0+34, 109, 'one query, N keys', 13, 'gold', True, 'start')
    ops = [.35, .55, .75, .95]
    for k in range(4):
        x = x0 + k*98
        f.rect(x, 128, 92, 26, 'blue', ops[k], 3)
        f.math(x+46, 146, f'K, V \\text{{ chunk }} {k+1}', 14, 'white' if k else 'ink', 'middle')
        f.arrow(x+46, 156, x+46, 178, 'muted', width=1.2)
        f.box(x+4, 180, 84, 30, '$m_s, \\ell_s, o_s', 'blue', size=14)
        f.line(x+46, 212, 664, 236, 'green', 1.4)
    f.brace(x0, 128, x0+386, 128, '', 'muted', 'above', 13, 6)
    f.circle(664, 246, 12, '', 'green', True)
    f.text(664, 251, '⊕', 16, 'white', True)
    f.text(684, 251, 'LSE merge → o', 14, 'green', True, 'start')
    f.text(470, 282, 'same bytes read, spread over more SMs', 13, 'muted', anchor='start')
    f.hdivider(298)
    # toy merge numbers
    f.text(30, 322, 'toy merge, 4 keys in 2 chunks', 14, 'ink', True, 'start')
    hdr = [('chunk', 30), ('scores', 120), ('values', 210), ('m', 290), ('ℓ', 340), ('o', 410), ('LSE', 480), ('w', 540)]
    for lab, x in hdr:
        f.text(x, 346, lab, 13, 'muted', True, 'start')
    data = [('A', '1, 2', '10, 20', '2', '1.368', '17.31', '2.313', '0.324'),
            ('B', '3, 0', '30, 0', '3', '1.050', '28.58', '3.049', '0.676')]
    for r, row in enumerate(data):
        for (lab, x), val in zip(hdr, row):
            f.text(x, 370 + r*22, val, 14, 'green' if lab == 'w' else 'ink', lab == 'chunk', 'start')
    f.math(30, 432, r'\c{green}{o} = 0.324 \cdot 17.31 + 0.676 \cdot 28.58 = \c{green}{24.93} = \text{softmax over all 4 keys}', 16)
    # reported
    f.text(610, 322, 'Reported · CodeLlama-34B · A100', 14, 'ink', True, 'start')
    f.bars(610, 336, [('decode', 8, 'green'), ('attention', 50, 'green')], width=100, bar=18, gap=10, label_w=80, size=14, value_fmt=lambda v: f'up to {v:g}×')
    f.text(610, 410, 'vs FlashAttention, 512 to 64K tokens', 13, 'muted', anchor='start')
    f.text(610, 430, 'batch 1: FlashAttention used < 1% of GPU', 13, 'red', anchor='start')
    f.save()


# ---------------------------------------------------------------- 5. FlashMLA / FlashInfer
def flashmla():
    f = Fig('kern-flashmla', 'FlashMLA runs absorbed MLA decode as MQA: 128 query heads become 128 rows of one matmul against a single 576-wide latent per token (512 latent plus 64 RoPE dims). Each cached token then serves 128 × 2 × (576 + 512) = 278,528 FLOPs: about 242 FLOP per byte in BF16 (1,152 B) and 425 in FP8 (656 B), against 1 for MHA, 8 for GQA-8 and an H100 ridge of about 295. Reported on H800: 3000 GB/s memory-bound, 580 then 660 TFLOP/s compute-bound. FlashInfer stores a ragged, paged batch as one block-sparse matrix.', 446)
    f.math(30, 36, r'I = \c{gold}{H_q} \cdot 2(\c{blue}{576} + \c{blue}{512}) \text{ FLOP} \;/\; \text{bytes per token} = 278,528 / \c{blue}{1152 \text{ B}} \approx \c{green}{242}', 20)
    # matmul picture
    f.text(30, 74, 'decode, one layer, absorbed weights (MQA mode)', 14, 'ink', True, 'start')
    f.rect(104, 100, 60, 124, 'gold', .85, 4)
    f.math(134, 168, 'Q', 18, 'white', 'middle')
    f.brace(104, 100, 104, 224, '$128 \\text{ heads}', 'gold', 'left', 13, 4)
    f.brace(104, 100, 164, 100, '$576', 'muted', 'above', 13, 4)
    f.text(176, 168, '·', 24, 'ink')
    for r in range(6):
        y = 102 + r*20
        f.rect(190, y, 100, 16, 'blue', .8, 2)
        f.rect(292, y, 14, 16, 'teal', .85, 2)
    f.math(240, 240, r'c^{KV} \; 512', 14, 'blue', 'middle')
    f.math(306, 240, r'k^R \; 64', 14, 'teal', 'start')
    f.text(250, 258, 'one latent per token, read once', 13, 'muted')
    f.brace(312, 102, 312, 218, '$N', 'muted', 'right', 13, 4)
    f.text(354, 168, '→', 20, 'ink')
    f.rect(376, 100, 50, 124, 'green', .85, 4)
    f.math(401, 168, 'O', 18, 'white', 'middle')
    f.brace(376, 100, 426, 100, '$512', 'muted', 'above', 13, 4)
    f.math(30, 284, r'\text{MMA rows} = s_q H_q / H_{kv} = 128 \to \text{full tensor-core tiles}', 14, 'muted')
    # intensity bars
    f.text(470, 74, 'FLOP per KV byte at decode (computed)', 14, 'ink', True, 'start')
    rows = [('MHA · G = 1', 1, 'red'), ('GQA · G = 8', 8, 'blue'), ('MLA · BF16 1152 B', 242, 'purple'),
            ('H100 ridge', 295, 'line'), ('MLA · FP8 656 B', 425, 'green')]
    f.bars(470, 90, rows, width=200, bar=22, gap=12, label_w=150, size=14, value_fmt=lambda v: f'{v:g}')
    f.text(470, 284, 'absorbed MLA sits near the ridge; MHA/GQA far below it', 13, 'muted', anchor='start')
    f.hdivider(298)
    # reported FlashMLA
    f.text(30, 324, 'FlashMLA · reported, H800 SXM5', 14, 'ink', True, 'start')
    f.bars(30, 338, [('memory-bound', 3000, 'blue')], width=150, bar=18, gap=8, label_w=130, size=14, value_fmt=lambda v: f'{v:g} GB/s')
    f.bars(30, 366, [('Feb 2025', 580, 'green'), ('Apr 2025', 660, 'green')], width=150, bar=18, gap=8, label_w=130, size=14, value_fmt=lambda v: f'{v:g} TFLOP/s')
    f.text(30, 438, 'paged KV, block 64 tokens · FP8 cache since Sep 2025', 13, 'muted', anchor='start')
    # FlashInfer BSR grid
    f.text(470, 324, 'FlashInfer · ragged paged batch = one block-sparse matrix', 14, 'ink', True, 'start')
    use = {0: [0, 3, 4, 8], 1: [1, 6], 2: [2, 5, 7, 9, 10]}
    for r in range(3):
        f.math(486, 360 + r*24, f'q_{{{"ABC"[r]}}}', 14, 'gold', 'middle')
        for c in range(11):
            on = c in use[r]
            f.rect(500 + c*24, 346 + r*24, 22, 20, 'blue' if on else 'line', .85 if on else .3, 2)
    f.math(500, 434, r'\text{rows: query tiles } (B_r) \cdot \text{ columns: KV pages } (B_c = \text{page})', 14, 'muted')
    f.save()


# ---------------------------------------------------------------- 6. chunkwise linear-attention kernels
def chunkwise():
    f = Fig('kern-chunkwise', 'Kernel view of chunkwise linear attention (flash-linear-attention). Each chunk of C = 64 tokens does a masked C×C attention and two state matmuls on tensor cores; a 128×128 state passes between chunks. Non-materialized: the state stays in SRAM (64 KiB in FP32) and chunks run in sequence. Materialized: all 512 chunk states for N = 32K go to HBM (32 MiB per head, computed) and chunks run in parallel; GLA reports about 10–20% more memory. Computed per head at N = 32K, d = 128: chunkwise 3.2 GFLOP, all matmul; step-by-step recurrence 2.1 GFLOP, all rank-1; causal softmax 275 GFLOP.', 450)
    f.math(30, 36, r'\c{green}{O_{[c]}} = (\c{gold}{Q_{[c]}} \c{blue}{K_{[c]}}^\top \odot M) \c{purple}{V_{[c]}} + \c{gold}{Q_{[c]}} \c{purple}{S_{[c−1]}}^\top', 21)
    f.math(560, 36, r'\c{purple}{S_{[c]}} = \c{purple}{S_{[c−1]}} + \c{purple}{V_{[c]}}^\top \c{blue}{K_{[c]}}', 21)
    # chunks along the sequence
    f.text(30, 74, 'N tokens in chunks of C = 64 (GLA benchmark setting; C a multiple of 16 for tensor cores)', 14, 'muted', anchor='start')
    n = 8
    for c in range(4):
        x = 40 + c*214
        f.mask(x, 92, n, 11, lambda i, j: ('blue', .3 + .07*(i+j)) if j <= i else None)
        f.text(x+44, 200, f'chunk {c+1}', 13, 'ink', True)
        f.math(x+44, 220, r'(QK^\top \odot M) V', 13, 'blue', 'middle')
        f.text(x+44, 238, '4C²d FLOP, matmul', 13, 'muted')
        if c < 3:
            sx = x + 112
            f.rect(sx, 120, 52, 40, 'purple', .85, 4)
            f.math(sx+26, 146, f'S_{{[{c+1}]}}', 15, 'white', 'middle')
            f.arrow(x+92, 140, sx-2, 140, 'purple', width=1.6)
            f.arrow(sx+54, 140, sx+96, 140, 'purple', width=1.6)
    f.text(186, 112, '128×128', 12, 'purple', True)
    f.text(186, 180, 'Q Sᵀ, Vᵀ K: 4Cd²', 12, 'purple')
    f.hdivider(256)
    # two schedules
    f.text(30, 280, 'two schedules (GLA §3.3)', 14, 'ink', True, 'start')
    f.box(30, 292, 400, 56, '', 'purple')
    f.text(44, 314, 'non-materialized: S stays in SRAM', 14, 'purple', True, 'start')
    f.text(44, 336, '64 KiB FP32 · chunks in sequence · parallel over B, H', 13, 'ink', anchor='start')
    f.box(30, 358, 400, 56, '', 'blue')
    f.text(44, 380, 'materialized: all N/C states → HBM', 14, 'blue', True, 'start')
    f.text(44, 402, '512 × 64 KiB = 32 MiB/head at 32K · chunks in parallel', 13, 'ink', anchor='start')
    f.text(30, 436, 'materialized: reported ≈10–20% more memory', 13, 'muted', anchor='start')
    # cost readout (log axis)
    f.text(470, 280, 'FLOPs per head, N = 32K, d = 128 (computed, log axis)', 14, 'ink', True, 'start')
    logbars(f, 470, 294, [('causal softmax', 274.9, 'red', '275 G'), ('chunkwise', 3.22, 'green', '3.2 G · matmul'),
                          ('recurrent', 2.15, 'line', '2.1 G · rank-1')], 190, 1, 400, bar=22, gap=12, label_w=120)
    f.text(470, 412, 'reported: FLA faster than FlashAttention-2', 13, 'green', True, 'start')
    f.text(470, 432, 'even at 1K tokens (GLA, ICML 2024)', 13, 'green', True, 'start')
    f.save()


# ---------------------------------------------------------------- 7. low-bit attention
def sage():
    f = Fig('kern-sage', 'Low-bit attention kernels. Bits per operand of the two matmuls: FA-2 16/16; FA-3 FP8 8/8; SageAttention 1 INT8 QK with FP16 PV; SageAttention 2 INT4 QK with FP8 PV; SageAttention 3 FP4 for both (NVFP4, 1×16 blocks). Smoothing K by subtracting its mean over tokens leaves softmax unchanged and shrinks the INT8 step for a toy channel 9.8, 10.2, 10.3, 9.7 from 0.081 to 0.0024, 34× finer. Reported kernel speedups over FlashAttention-2 on consumer GPUs: about 2.1×, 3× and 5×.', 440)
    f.math(30, 36, r'\text{softmax}_j(q^\top(\c{blue}{k_j} − \c{blue}{\mu_K})) = \text{softmax}_j(q^\top \c{blue}{k_j})', 21)
    f.math(540, 36, r'\c{gold}{Q}\c{blue}{K}^\top \approx \delta_Q \delta_K \cdot \c{gold}{Q}_{8} \c{blue}{K}_{8}^\top', 21)
    f.text(30, 74, 'bits per operand', 14, 'ink', True, 'start')
    f.math(233, 74, r'\c{gold}{Q}\c{blue}{K}^\top', 15, anchor='middle')
    f.math(383, 74, r'\c{green}{P}\c{purple}{V}', 15, anchor='middle')
    rows = [('FA-2', 16, 16, 'BF16', 'BF16'), ('FA-3 FP8 · H100', 8, 8, 'FP8', 'FP8'),
            ('Sage 1 · ICLR 25', 8, 16, 'INT8', 'FP16'), ('Sage 2 · ICML 25', 4, 8, 'INT4', 'FP8'),
            ('Sage 3 · NeurIPS 25', 4, 4, 'FP4', 'FP4')]
    for r, (name, bq, bp, lq, lp) in enumerate(rows):
        y = 88 + r*34
        f.text(170, y+19, name, 14, 'ink', r == 0, 'end')
        f.rect(185, y, bq*6, 26, 'blue', .85, 3)
        f.text(185 + bq*6 + 6, y+18, lq, 13, 'blue', True, 'start')
        f.rect(335, y, bp*6, 26, 'purple', .85, 3)
        f.text(335 + bp*6 + 6, y+18, lp, 13, 'purple', True, 'start')
    f.text(30, 274, 'Sage 1: FP16 accumulator in PV (2× faster on RTX 4090)', 13, 'muted', anchor='start')
    f.text(30, 294, 'Sage 3: NVFP4 = E2M1 values + E4M3 scale per 16', 13, 'muted', anchor='start')
    # smoothing toy
    x0 = 520
    f.text(x0, 74, 'toy · one key channel, 4 tokens', 14, 'ink', True, 'start')
    raw = [9.8, 10.2, 10.3, 9.7]
    sm = [-0.2, 0.2, 0.3, -0.3]
    for k in range(4):
        f.rect(x0 + k*62, 88, 56, 30, 'blue', .85, 3)
        f.text(x0 + k*62 + 28, 108, f'{raw[k]}', 14, 'white', True)
        f.rect(x0 + k*62, 170, 56, 30, 'blue', .3, 3)
        f.text(x0 + k*62 + 28, 190, f'{sm[k]:+.1f}'.replace('-', '−'), 14, 'ink', True)
    f.math(x0, 142, r'\text{INT8 step } Δ = 10.3/127 = \c{red}{0.081}', 15)
    f.arrow(x0+120, 150, x0+120, 166, 'blue', width=1.4)
    f.math(x0+132, 162, r'− \mu_K = 10.0', 14, 'blue')
    f.math(x0, 224, r'Δ = 0.3/127 = \c{green}{0.0024} \text{ (34× finer)}', 15)
    f.math(x0, 252, r'q^\top \mu_K \text{ is the same for every key } j', 14, 'muted')
    f.text(x0, 272, 'so the row softmax cancels it exactly', 13, 'muted', anchor='start')
    f.hdivider(310)
    f.text(30, 334, 'Reported kernel throughput vs FlashAttention-2 on the same GPU', 14, 'ink', True, 'start')
    f.bars(30, 346, [('Sage 1', 2.1, 'blue'), ('Sage 2 · RTX 4090', 3, 'blue'), ('Sage 3 · RTX 5090', 5, 'green')],
           width=200, bar=20, gap=8, label_w=160, size=14, value_fmt=lambda v: f'≈{v:g}×')
    f.text(520, 350, 'Sage 3: 1038 TOPS on RTX 5090', 14, 'green', True, 'start')
    f.text(520, 374, 'FA-3 FP8: ≈1.2 PFLOP/s on H100, 2.6× lower', 14, 'ink', anchor='start')
    f.text(520, 394, '   error than baseline FP8 (RMSE 9.1e-3 vs 2.4e-2)', 14, 'ink', anchor='start')
    f.text(520, 418, 'Sage 2 matches FA-3 FP8 speed on Hopper', 14, 'ink', anchor='start')
    f.save()


def build():
    flash_tiles()
    online_softmax()
    fa_generations()
    split_kv()
    flashmla()
    chunkwise()
    sage()
