"""14 · Serving the cache: bytes per record, pages, prefixes, transfers and tiers."""
import math
from figkit import Fig, COL

GiB, MiB = 2**30, 2**20


# ---------------------------------------------------------------- helpers
def logbars(f, x, y, items, width, lo, hi, bar=20, gap=10, label_w=200, size=14):
    """Horizontal bars on a log axis: items = [(label, value, colour, shown)]."""
    span = math.log10(hi) - math.log10(lo)
    for i, (lab, v, c, shown) in enumerate(items):
        yy = y + i*(bar+gap)
        f.text(x+label_w-10, yy+bar*.7, lab, size, 'ink', anchor='end')
        w = max(3, width*(math.log10(v) - math.log10(lo))/span)
        f.rect(x+label_w, yy, w, bar, c, .85, 3)
        f.text(x+label_w+w+8, yy+bar*.7, shown, size, 'muted' if c == 'line' else c, True, 'start')


def outline(f, x, y, w, h, color='ink', width=1.0, rx=2):
    f.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="none" stroke="{COL.get(color, color)}" stroke-width="{width}"/>')


def stacked(f, x, y, w, h, segs, total=100):
    """segs = [(value, colour, opacity, label_or_None)] drawn left to right."""
    cx = x
    for v, c, op, lab in segs:
        sw = w*v/total
        f.rect(cx, y, sw, h, c, op, 2)
        if lab and sw > 34:
            f.text(cx+sw/2, y+h*.7, lab, 13, 'white' if op > .6 else 'ink', True)
        cx += sw


# ---------------------------------------------------------------- 1. FP8 / FP4 KV formats
def kv_bytes():
    f = Fig('serve-kv-bytes', 'Bytes per token per layer of DeepSeek KV caches, from the FlashMLA formats. V3.2 BF16: 512 latent + 64 RoPE values × 2 B = 1,152 B. V3.2 FP8: 512 e4m3 + 4 fp32 scales (one per 128) + 64 BF16 RoPE values = 656 B, 57%. V4.1 (512 dims) FP8: 512 e4m3 including RoPE + 16 e8m0 scales (one per 32) = 528 B, 52% of BF16. V4.1 FP4: 512 e2m1 values in 256 B + 32 e4m3 scales (one per 16) = 288 B, 28%. Over V3.2’s 61 layers a 128K sequence drops from 8.58 GiB to 4.88 GiB.', 430)
    f.math(30, 36, r'\text{bytes / token / layer} = \c{blue}{n \cdot b / 8} \; + \; \c{red}{\text{scales}} \; + \; \c{teal}{\text{unquantized (RoPE)}}', 21)
    px = 560/1152
    rows = [
        ('V3.2 · BF16', [(1024, 'blue', .45, '512 latent × BF16 = 1024'), (128, 'teal', .85, '64 RoPE')], '1,152 B', ''),
        ('V3.2 · FP8', [(512, 'blue', .9, '512 × e4m3 = 512'), (16, 'red', .9, None), (128, 'teal', .85, '64 RoPE')], '656 B', '57% of its BF16'),
        ('V4.1 · FP8', [(512, 'blue', .9, '512 × e4m3, RoPE incl.'), (16, 'red', .9, None)], '528 B', '52% of 1,024 B BF16'),
        ('V4.1 · FP4', [(256, 'blue', .9, '512 × e2m1'), (32, 'red', .9, None)], '288 B', '28% of BF16'),
    ]
    notes = [None, '4 × fp32 scale, 1 per 128 = 16 B', '16 × e8m0, 1 per 32 = 16 B', '32 × e4m3, 1 per 16 = 32 B']
    for r, (name, segs, tot, pct) in enumerate(rows):
        y = 84 + r*60
        f.text(186, y+22, name, 15, 'ink', r == 0, 'end')
        cx = 200
        for v, c, op, lab in segs:
            w = v*px
            f.rect(cx, y, w, 30, c, op, 2)
            if lab:
                f.text(cx+w/2, y+20, lab, 13, 'white' if op > .6 else 'ink', True)
            cx += w
        f.text(cx+10, y+21, tot, 16, 'blue', True, 'start')
        f.text(cx+10+len(tot)*9.5+6, y+21, pct, 14, 'muted', True, 'start')
        if notes[r]:
            f.text(200 + 512*px if r == 1 else 200 + segs[0][0]*px, y+48, '↑ ' + notes[r], 13, 'red', anchor='start')
    f.brace(200, 82, 200+1152*px, 82, '$\\text{V3.2: 576 dims} = d_c + d_r = 512 + 64 \\quad \\text{(V4.1: 512 dims)}', 'muted', 'above', 13, 4)
    f.hdivider(354)
    f.text(30, 380, 'V3.2, all 61 layers (computed)', 14, 'ink', True, 'start')
    f.text(30, 404, 'BF16 70,272 B/token → 8.58 GiB at 128K', 14, 'muted', anchor='start')
    f.text(30, 424, 'FP8 40,016 B/token → 4.88 GiB at 128K', 14, 'green', True, 'start')
    f.text(470, 380, 'NVFP4 KV (NVIDIA, reported, Blackwell)', 14, 'ink', True, 'start')
    f.text(470, 404, '≈50% of FP8 · <1% accuracy loss', 14, 'muted', anchor='start')
    f.text(470, 424, 'dequantized to FP8 before the attention matmuls', 14, 'red', anchor='start')
    f.save()


# ---------------------------------------------------------------- 2. KIVI / KVQuant
def kivi():
    f = Fig('serve-kivi', 'Where the outliers are. Toy key cache (10 tokens × 12 channels) has two outlier channels, so KIVI quantizes keys per channel in groups of tokens and values per token, keeping the last R tokens in full precision (paper defaults: G = 32, R = 128, 2 bits). KVQuant quantizes keys before RoPE, because RoPE mixes channel pairs by position and smears the outlier channel. Reported LLaMA-7B Wikitext-2 perplexity: FP16 5.68, KVQuant 3-bit with 1% sparse outliers 5.75, uniform INT3 10.87.', 440)
    f.math(30, 36, r'Q_b(x) = z + Δ \cdot \text{round}((x − z)/Δ)', 21)
    f.math(390, 36, r'Δ = (\max − \min)/(2^b − 1)', 21)
    f.math(660, 36, r'z = \min \text{ of the group}', 21)
    out = (3, 8)
    cell = 15
    # K per channel
    x0, y0 = 40, 92
    f.text(x0, 76, 'K: per-channel groups', 14, 'blue', True, 'start')
    def kval(i, j):
        return .95 if j in out else .25 + .25*((i*5 + j*3) % 4)/3
    for i in range(10):
        for j in range(12):
            f.rect(x0 + j*cell, y0 + i*cell, cell-2, cell-2, 'blue', kval(i, j), 2)
    for j in range(12):
        outline(f, x0 + j*cell - 1, y0 - 1, cell, 4*cell, 'ink', .8)
        outline(f, x0 + j*cell - 1, y0 + 4*cell - 1, cell, 4*cell, 'ink', .8)
    outline(f, x0 - 2, y0 + 8*cell - 2, 12*cell + 2, 2*cell + 2, 'green', 2.2, 3)
    f.brace(x0 + 12*cell, y0, x0 + 12*cell, y0 + 4*cell - 3, '$G', 'muted', 'right', 13, 4)
    f.brace(x0 + 12*cell, y0 + 8*cell, x0 + 12*cell, y0 + 10*cell - 2, '$R', 'green', 'right', 13, 4)
    f.text(x0, y0 + 10*cell + 20, 'outlier channels stay inside', 13, 'muted', anchor='start')
    f.text(x0, y0 + 10*cell + 38, 'their own group', 13, 'muted', anchor='start')
    # V per token
    x1 = 270
    f.text(x1, 76, 'V: per-token', 14, 'purple', True, 'start')
    for i in range(10):
        for j in range(12):
            f.rect(x1 + j*cell, y0 + i*cell, cell-2, cell-2, 'purple', .25 + .3*((i*7 + j*5) % 5)/4, 2)
        if i < 8:
            outline(f, x1 - 1, y0 + i*cell - 1, 12*cell, cell, 'ink', .8)
    outline(f, x1 - 2, y0 + 8*cell - 2, 12*cell + 2, 2*cell + 2, 'green', 2.2, 3)
    f.text(x1, y0 + 10*cell + 20, 'green: last R tokens in FP16', 13, 'green', True, 'start')
    f.text(x1, y0 + 10*cell + 38, 'toy: 10 tokens × 12 channels', 13, 'muted', anchor='start')
    # KVQuant pre vs post RoPE
    x2 = 520
    f.text(x2, 76, 'KVQuant: quantize K before RoPE', 14, 'teal', True, 'start')
    for k, (title, post) in enumerate((('pre-RoPE', False), ('post-RoPE', True))):
        xx = x2 + k*170
        f.text(xx, y0 - 2, title, 13, 'muted', True, 'start')
        for i in range(8):
            for j in range(8):
                if not post:
                    v = .95 if j == 4 else .25
                else:
                    ang = i*0.7
                    v = .25 + .7*abs(math.cos(ang)) if j == 4 else (.25 + .7*abs(math.sin(ang)) if j == 5 else .25)
                f.rect(xx + j*cell, y0 + 6 + i*cell, cell-2, cell-2, 'blue', v, 2)
    f.arrow(x2 + 128, y0 + 66, x2 + 166, y0 + 66, 'teal', width=1.6)
    f.text(x2, y0 + 150, 'RoPE mixes channel pairs (4, 5) by position:', 13, 'muted', anchor='start')
    f.text(x2, y0 + 168, 'a clean outlier channel becomes a moving one', 13, 'muted', anchor='start')
    f.hdivider(300)
    f.text(30, 324, 'KVQuant · LLaMA-7B Wikitext-2 perplexity (reported)', 14, 'ink', True, 'start')
    f.bars(30, 336, [('FP16', 5.68, 'line'), ('3-bit + 1% outliers', 5.75, 'green'), ('uniform INT3', 10.87, 'red')],
           width=170, bar=18, gap=8, label_w=170, size=14, value_fmt=lambda v: f'{v:g}')
    f.text(30, 432, 'non-uniform levels (per-layer k-means) · first token kept FP16', 13, 'muted', anchor='start')
    f.text(520, 324, 'KIVI · 2-bit, tuning-free (reported)', 14, 'ink', True, 'start')
    f.text(520, 350, '2.6× lower peak memory (incl. weights)', 14, 'green', anchor='start')
    f.text(520, 372, 'up to 4× larger batch', 14, 'green', anchor='start')
    f.text(520, 394, '2.35–3.47× throughput', 14, 'green', anchor='start')
    f.text(520, 420, 'KVQuant: 1M context on one A100-80GB', 14, 'teal', anchor='start')
    f.save()


# ---------------------------------------------------------------- 3. PagedAttention
def paged():
    f = Fig('serve-paged', 'PagedAttention stores each request’s KV in fixed blocks addressed through a block table, like virtual memory. Toy with block size 4: request A uses physical blocks 7, 1, 3 and request B blocks 5, 2; only the last block of each request has empty slots. Reported KV memory breakdown: Orca (Max) 20.4% token states, 13.3% reservation, 57.3% internal and 8.9% external fragmentation; Orca (Pow2) 26.8%, 17.9%, 13.6%, 41.6%; vLLM 96.3% token states. OPT-13B needs 800 KB per token, so reserving 2,048 tokens costs 1.6 GB per request, while paging wastes at most 15 tokens, 12 MB, with block 16.', 450)
    f.math(30, 36, r'\text{slot}(t) = \c{blue}{\text{table}}[⌊t / b⌋] \cdot b + (t \text{ mod } b)', 21)
    f.math(480, 36, r'\text{waste} \le b − 1 \text{ tokens per sequence}', 21)
    b = 4
    reqs = [('A', [4, 4, 2], [7, 1, 3], 'blue'), ('B', [4, 1], [5, 2], 'purple')]
    phys = {}
    for r, (name, fills, table, c) in enumerate(reqs):
        y = 76 + r*86
        f.text(30, y+16, f'request {name}', 14, c, True, 'start')
        f.text(30, y+34, 'logical blocks', 13, 'muted', anchor='start')
        for k, n in enumerate(fills):
            x = 140 + k*92
            for s in range(b):
                filled = s < n
                f.rect(x + s*20, y, 18, 22, c if filled else 'red', .85 if filled else .15, 2)
            f.text(x + 38, y + 40, f'L{k}', 13, 'muted')
            phys[table[k]] = (name, n, c)
        # block table
        tx = 430
        f.text(tx, y - 6, 'block table', 13, 'muted', True, 'start')
        for k, (pb, n) in enumerate(zip(table, fills)):
            f.box(tx + k*58, y, 54, 26, f'{pb} · {n}/{b}', c, size=13)
        f.arrow(130 + len(fills)*92 - 4, y + 11, tx - 6, y + 13, 'muted', width=1.2)
    f.text(430, 246, 'entry = physical block · filled slots', 13, 'muted', anchor='start')
    # physical memory
    y = 270
    f.text(30, y + 16, 'GPU memory', 14, 'ink', True, 'start')
    for pb in range(8):
        x = 140 + pb*90
        if pb in phys:
            name, n, c = phys[pb]
            for s in range(b):
                f.rect(x + s*20, y, 18, 22, c if s < n else 'red', .85 if s < n else .15, 2)
            f.text(x + 38, y + 40, f'#{pb} · {name}', 13, c, True)
        else:
            for s in range(b):
                f.rect(x + s*20, y, 18, 22, 'line', .4, 2)
            f.text(x + 38, y + 40, f'#{pb} · free', 13, 'muted')
    f.text(870, 340, 'toy: b = 4 · red = unused slot in a last block', 13, 'red', anchor='end')
    f.hdivider(352)
    f.text(30, 374, 'KV memory use, reported (vLLM Fig. 2)', 14, 'ink', True, 'start')
    bars = [('Orca (Max)', [(20.4, 'green', .85, '20.4'), (13.3, 'red', .35, '13.3'), (57.3, 'red', .6, '57.3'), (8.9, 'red', .9, None)]),
            ('Orca (Pow2)', [(26.8, 'green', .85, '26.8'), (17.9, 'red', .35, '17.9'), (13.6, 'red', .6, '13.6'), (41.6, 'red', .9, '41.6')]),
            ('vLLM', [(96.3, 'green', .85, '96.3% token states')])]
    for i, (lab, segs) in enumerate(bars):
        yy = 386 + i*20
        f.text(126, yy + 13, lab, 13, 'ink', anchor='end')
        stacked(f, 134, yy, 330, 16, segs)
    f.text(480, 374, 'legend: green token states · red: reservation,', 13, 'muted', anchor='start')
    f.text(480, 392, 'internal, external fragmentation (light → dark)', 13, 'muted', anchor='start')
    f.text(480, 418, 'OPT-13B, 800 KB/token: reserve 2,048 tok = 1.6 GB', 13, 'red', True, 'start')
    f.text(480, 438, 'paged, b = 16: waste ≤ 15 tok = 12 MB (computed)', 13, 'green', True, 'start')
    f.save()


# ---------------------------------------------------------------- 4. RadixAttention / prefix cache
def radix():
    f = Fig('serve-radix', 'RadixAttention keeps finished requests’ KV in a radix tree keyed by token sequences. Toy: a 1,000-token system prompt is shared by two chats and a few-shot branch; a new turn of chat 1 matches 1,350 of 1,410 tokens and prefills only 60. Least-recently-used leaves are evicted first. Reported hit rates: 50–99% across SGLang benchmarks, 52.4% for LLaVA-NeXT-34B and 74.1% for Vicuna-33B in Chatbot Arena, and 56.3% of DeepSeek’s 608B daily input tokens hit its on-disk cache.', 450)
    f.math(30, 36, r'\text{prefill tokens} = N − \c{green}{N_{\text{hit}}}', 21)
    f.math(330, 36, r'\c{green}{N_{\text{hit}}} = \text{longest cached prefix (exact token match)}', 21)
    def node(x, y, w, label, c, dash=False, op=False):
        f.box(x, y, w, 30, label, c, fill=op, size=13, dash=dash, opacity=.85)
        return (x + w/2, y + 30)
    root = (90, 92)
    f.circle(90, 80, 9, '', 'ink', True)
    f.text(90, 66, 'root', 13, 'muted')
    sp = node(30, 108, 160, 'system prompt · 1,000', 'green', op=True)
    f.line(90, 89, 110, 108, 'green', 2)
    c1 = node(230, 108, 120, 'chat 1 · 200', 'green', op=True)
    c2 = node(230, 168, 120, 'chat 2 · 300', 'blue')
    fs = node(230, 228, 120, 'few-shot · 800', 'blue')
    for (x, y) in ((230, 123), (230, 183), (230, 243)):
        f.path([(190, 123), (210, 123), (210, y), (x, y)], 'green' if y == 123 else 'muted', width=2 if y == 123 else 1.4)
    t2 = node(390, 108, 120, 'turn 2 · 150', 'green', op=True)
    f.line(350, 123, 390, 123, 'green', 2)
    new = node(550, 108, 110, 'new · 60', 'gold', dash=True)
    f.line(510, 123, 550, 123, 'gold', 2, dash=True)
    q1 = node(390, 228, 80, 'q1 · 40', 'blue')
    q2 = node(390, 268, 80, 'q2 · 35', 'line', dash=True)
    f.line(350, 243, 390, 243, 'muted', 1.4)
    f.path([(370, 243), (370, 283), (390, 283)], 'muted', width=1.4)
    f.text(480, 288, 'LRU leaf: evicted first', 13, 'muted', anchor='start')
    f.text(680, 120, 'hit 1,350 of 1,410 tokens', 14, 'green', True, 'start')
    f.text(680, 140, 'prefill only 60 (4%)', 14, 'gold', True, 'start')
    f.text(30, 318, 'toy token counts · page = 1 token · scheduler serves longest shared prefix first (= DFS order)', 13, 'muted', anchor='start')
    f.hdivider(334)
    f.text(30, 358, 'Prefix-cache hit rates, reported', 14, 'ink', True, 'start')
    items = [('SGLang benchmarks', 50, 99, 'green', '50–99%'), ('LLaVA-NeXT-34B, Arena', 52.4, 52.4, 'green', '52.4%'),
             ('Vicuna-33B, Arena', 74.1, 74.1, 'green', '74.1%'), ('DeepSeek API, on disk', 56.3, 56.3, 'blue', '56.3% of 608B tok/day')]
    for i, (lab, lo, hi, c, shown) in enumerate(items):
        y = 370 + i*19
        f.text(220, y + 13, lab, 13, 'ink', anchor='end')
        f.rect(230, y, 3*lo, 15, c, .85, 2)
        if hi > lo:
            f.rect(230 + 3*lo, y, 3*(hi - lo), 15, c, .35, 2)
        f.text(238 + 3*hi, y + 13, shown, 13, c, True, 'start')
    f.text(870, 380, 'SGLang: up to 6.4× throughput', 14, 'green', True, 'end')
    f.text(870, 402, 'hit needs a byte-identical prefix:', 13, 'muted', anchor='end')
    f.text(870, 420, 'context truncation can halve it (LMCache)', 13, 'red', anchor='end')
    f.save()


# ---------------------------------------------------------------- 5. prefill/decode disaggregation
def disagg():
    x = 61*656*32768
    f = Fig('serve-disagg', 'Prefill/decode disaggregation. On one GPU a new long prefill stalls every running decode; split pools let prefill GPUs run prompts and ship the KV to decode GPUs. Computed transfer for a 32K-token prompt: DeepSeek-V3.2 FP8 (61 × 656 B = 40,016 B per token) is 1.22 GiB, 26 ms over one 400 Gb/s NIC or 6.9 ms at Mooncake’s reported 190 GB/s; Llama-3-70B BF16 (320 KiB per token) is 10 GiB, 215 ms over one NIC. Reported: DistServe 7.4× more requests, Splitwise 1.4× throughput at 20% lower cost, Mooncake 59–498% more capacity; DeepSeek uses 4 prefill nodes against 18 decode nodes per unit.', 456)
    f.math(30, 36, r'T_{\text{xfer}} = N_{\text{prompt}} \cdot \sum_\ell \c{blue}{c_\ell} \;/\; \text{BW}', 21)
    f.math(870, 36, r'c_\ell = \text{KV bytes per token in layer } \ell', 15, 'muted', 'end')
    f.text(870, 76, 'time →', 13, 'muted', anchor='end')
    # co-located
    f.text(30, 76, 'co-located', 14, 'ink', True, 'start')
    y = 86
    f.rect(140, y, 730, 28, 'line', .25, 4)
    for k in range(7):
        f.rect(146 + k*22, y + 6, 14, 16, 'green', .85, 2)
    f.rect(300, y + 3, 300, 22, 'gold', .9, 3)
    f.text(450, y + 19, 'prefill of a new 32K prompt', 13, 'white', True)
    f.brace(300, y + 28, 600, y + 28, 'decodes stall: TPOT spike', 'red', 'below', 13, 4)
    for k in range(12):
        f.rect(608 + k*22, y + 6, 14, 16, 'green', .85, 2)
    # disaggregated
    f.text(30, 168, 'disaggregated', 14, 'ink', True, 'start')
    y1, y2 = 178, 248
    f.text(132, y1 + 19, 'prefill pool', 13, 'gold', True, 'end')
    f.rect(140, y1, 730, 28, 'line', .25, 4)
    for k, w in enumerate((150, 300, 120)):
        x0 = 146 + sum((150, 300, 120)[:k]) + 10*k
        f.rect(x0, y1 + 3, w, 22, 'gold', .9, 3)
    f.text(132, y2 + 19, 'decode pool', 13, 'green', True, 'end')
    f.rect(140, y2, 730, 28, 'line', .25, 4)
    for k in range(32):
        f.rect(146 + k*22.6, y2 + 6, 14, 16, 'green', .85, 2)
    f.arrow(450, y1 + 28, 470, y2 - 2, 'blue', width=2.6)
    f.text(480, y1 + 50, 'KV over RDMA: 1.22 GiB', 14, 'blue', True, 'start')
    f.hdivider(296)
    # bytes table
    f.text(30, 320, '32K-token prompt, KV to ship (computed)', 14, 'ink', True, 'start')
    rows = [('V3.2 · FP8 · 40,016 B/tok', x/GiB, 'green', '1.22 GiB → 26 ms @ 50 GB/s'),
            ('Llama-3-70B · BF16 · 320 KiB/tok', 10.0, 'red', '10 GiB → 215 ms @ 50 GB/s')]
    f.bars(30, 332, [(a, v, c) for a, v, c, _ in rows], width=110, bar=20, gap=10, label_w=230, size=13,
           value_fmt=lambda v: '1.22 GiB → 26 ms' if v < 2 else '10 GiB → 215 ms')
    f.text(30, 410, '50 GB/s = one 400 Gb/s NIC · Mooncake transfer engine,', 13, 'muted', anchor='start')
    f.text(30, 428, 'reported 190 GB/s on 8 × 400 Gb/s → 6.9 ms for V3.2', 13, 'muted', anchor='start')
    # reported gains
    f.text(520, 320, 'Reported', 14, 'ink', True, 'start')
    lines = [('DistServe (OSDI 24)', '7.4× requests or 12.6× tighter SLO'),
             ('Splitwise (ISCA 24)', '1.4× throughput at 20% lower cost'),
             ('Mooncake (FAST 25)', '+59–498% effective capacity'),
             ('DeepSeek, Feb 2025', 'prefill 4 nodes · decode 18 nodes')]
    for i, (a, b) in enumerate(lines):
        yy = 344 + i*26
        f.text(520, yy, a, 13, 'ink', True, 'start')
        f.text(520, yy + 1, '', 13)
        f.text(870, yy, b, 13, 'green', anchor='end')
    f.save()


# ---------------------------------------------------------------- 6. Ring vs Ulysses
def cp():
    N, P, dh, s = 131072, 8, 128, 2
    ring = lambda hkv: 2*N*hkv*dh*s
    uly = lambda hq, hkv: N*(2*hq + 2*hkv)*dh*s/P
    f = Fig('serve-cp', 'Context parallelism over P GPUs. Ring Attention keeps each GPU’s query block and passes K/V blocks around a ring; communication hides behind compute when the block has at least F/B tokens: 1.0K on A100 NVLink, 24.5K on A100 InfiniBand (reported). DeepSpeed-Ulysses all-to-alls from a token split to a head split, moving 4Nh/P per link but needing P ≤ heads. Computed bytes received per GPU per layer at N = 128K, P = 8, BF16: MHA 64 heads, ring 4 GiB vs Ulysses 1 GiB; GQA 64/8, ring 512 MiB vs Ulysses 576 MiB.', 456)
    f.math(30, 36, r'\text{Ring: hidden if block } c \ge F / B', 20)
    f.math(400, 36, r'\text{Ulysses: } 4Nh/P \text{ per link}, \quad P \le H', 20)
    # ring
    f.panel(30, 56, 400, 230, 'Ring Attention · pass K, V around')
    for i in range(4):
        x = 48 + i*94
        f.box(x, 94, 76, 74, '', 'ink', stroke=1)
        f.text(x + 38, 110, f'GPU {i}', 13, 'ink', True)
        f.rect(x + 6, 118, 30, 40, 'gold', .9, 3)
        f.math(x + 21, 143, f'Q_{i}', 14, 'white', 'middle')
        j = (i - 1) % 4
        f.rect(x + 40, 118, 30, 40, 'blue', .85, 3)
        f.math(x + 55, 143, f'KV_{j}', 12, 'white', 'middle')
        if i < 3:
            f.arrow(x + 76, 131, x + 92, 131, 'blue', width=1.8)
    f.path([(330 + 46, 168), (330 + 46, 180), (86, 180), (86, 171)], 'blue', arrow=True, width=1.8)
    f.text(230, 204, 'step t: GPU i uses block (i − t) mod P, then sends it on', 13, 'muted')
    f.text(48, 228, 'min block c = F/B (reported, Table 2):', 13, 'ink', True, 'start')
    for i, (lab, v) in enumerate((('A100 NVLink', '1.0K'), ('A100 IB', '24.5K'), ('TPU v4', '1.0K'))):
        x = 48 + i*130
        f.text(x, 250, lab, 13, 'muted', anchor='start')
        f.text(x + 118, 250, v, 14, 'red' if 'IB' in lab else 'green', True, 'end')
    f.text(48, 274, 'causal mask unbalances work: stripe or use 2×P chunks', 13, 'muted', anchor='start')
    # Ulysses
    f.panel(450, 56, 420, 230, 'Ulysses · all-to-all, tokens → heads')
    gx, gy, c = 480, 104, 30
    for k, (title, by_row) in enumerate((('before: N/P tokens, all H', True), ('after: all N, H/P heads', False))):
        x0 = gx + k*210
        f.text(x0, gy - 8, title, 13, 'muted', True, 'start')
        for i in range(4):
            for j in range(4):
                owner = i if by_row else j
                f.rect(x0 + j*c, gy + i*c, c - 3, c - 3, 'blue', .25 + .2*owner, 3)
                if (by_row and j == 0) or (not by_row and i == 0):
                    f.text(x0 + j*c + 13, gy + i*c + 19, f'{owner}', 12, 'ink', True)
        f.text(x0 + 60, gy + 4*c + 16, 'heads →', 12, 'muted')
    f.arrow(gx + 128, gy + 60, gx + 200, gy + 60, 'gold', width=2.2)
    f.text(gx + 164, gy + 52, 'all-to-all', 12, 'gold', True)
    f.math(470, 268, r'\text{each GPU holds } H/P \text{ heads} \to P \le H \;(\text{K/V: } P \le H_{kv})', 15, 'red')
    f.hdivider(300)
    f.math(30, 322, r'\text{Bytes received per GPU per layer: } N = 128\text{K}, P = 8, d_h = 128, \text{ BF16 (computed)}', 15, 'ink')
    rows = [('MHA 64/64 · ring', ring(64)/MiB, 'red'), ('MHA 64/64 · Ulysses', uly(64, 64)/MiB, 'green'),
            ('GQA 64/8 · ring', ring(8)/MiB, 'blue'), ('GQA 64/8 · Ulysses', uly(64, 8)/MiB, 'blue')]
    f.bars(30, 334, rows, width=260, bar=18, gap=6, label_w=170, size=13,
           value_fmt=lambda v: f'{v/1024:g} GiB' if v >= 1024 else f'{v:g} MiB')
    f.math(30, 446, r'\text{ring} \approx 2N H_{kv} d_h s \text{ (flat in } P), \quad \text{Ulysses} \approx N(2H_q + 2H_{kv}) d_h s / P', 15, 'muted')
    f.text(870, 340, 'Ulysses: 2.5× faster at 4× longer', 13, 'green', True, 'end')
    f.text(870, 358, 'sequences (reported)', 13, 'green', True, 'end')
    f.text(870, 388, 'Llama 3: all-gather K/V instead;', 13, 'ink', anchor='end')
    f.text(870, 406, 'GQA K/V ≪ Q · CP = 16 at 131K', 13, 'ink', anchor='end')
    f.save()


# ---------------------------------------------------------------- 7. tiers
def tiers():
    x = 61*656*32768
    tiers = [('HBM3 · H100', 3350e9), ('NVLink 4, both ways', 900e9), ('PCIe 5 x16, one way', 64e9), ('RDMA NIC 400 Gb/s', 50e9), ('NVMe SSD, Gen4', 7e9)]
    f = Fig('serve-tiers', 'Where cached KV can live, with typical bandwidths of an H100-class node (spec values): HBM3 3,350 GB/s, NVLink 900 GB/s, PCIe 5 x16 64 GB/s per direction, a 400 Gb/s RDMA NIC 50 GB/s, one Gen4 NVMe SSD about 7 GB/s. Reloading a 32K-token DeepSeek-V3.2 FP8 prefix (1.22 GiB) takes 0.39 ms, 1.5 ms, 20 ms, 26 ms or 187 ms, all far below the roughly 2.5 GPU-seconds needed to recompute its prefill at H100 BF16 peak (computed). Reported: 56.3% of DeepSeek input tokens hit its on-disk cache; LMCache up to 15× throughput with vLLM.', 440)
    f.math(30, 36, r'\text{reload wins if } N \cdot \c{blue}{b_{\text{tok}}} / \text{BW}_{\text{tier}} \; < \; T_{\text{prefill}}(N)', 21)
    f.text(30, 74, 'bandwidth, typical spec (log axis)', 14, 'ink', True, 'start')
    f.text(600, 74, 'time to load 1.22 GiB (log axis)', 14, 'ink', True, 'start')
    items = []
    for name, bw in tiers:
        shown = f'{bw/1e9:,.0f} GB/s'
        items.append((name, bw/1e9, 'blue', shown))
    logbars(f, 30, 88, items, 230, 1, 4000, bar=22, gap=12, label_w=170)
    for i, (name, bw) in enumerate(tiers):
        t = x/bw*1e3
        y = 88 + i*34
        w = 240*math.log10(t/0.1)/math.log10(30000/0.1)
        f.rect(600, y, w, 22, 'green', .85, 3)
        f.text(608 + w, y + 16, f'{t:.2g} ms' if t < 10 else f'{t:.0f} ms', 14, 'green', True, 'start')
    y = 88 + 5*34
    t = 2*37e9*32768/989e12*1e3
    w = 240*math.log10(t/0.1)/math.log10(30000/0.1)
    f.text(590, y + 16, 'recompute', 13, 'red', True, 'end')
    f.rect(600, y, w, 22, 'red', .85, 3)
    f.text(608 + w, y + 16, f'≈{t/1e3:.1f} s', 14, 'red', True, 'start')
    f.text(870, y + 44, 'V3.2 FP8, 32K tokens · recompute: 2·37B·32K = 2.4 PFLOP at 989 TFLOP/s', 13, 'muted', anchor='end')
    f.text(30, 280, 'tokens of KV per TB (computed): V3.2 FP8 25M · Llama-3-70B BF16 3.1M', 13, 'blue', True, 'start')
    f.hdivider(330)
    f.text(30, 354, 'Reported', 14, 'ink', True, 'start')
    rows = [('DeepSeek, 24 h (Feb 2025)', '342B of 608B input tokens (56.3%) hit the on-disk KV cache'),
            ('3FS (DeepSeek)', '≈6.6 TiB/s aggregate reads, 180 storage nodes'),
            ('LMCache + vLLM', 'up to 15× throughput (multi-round QA, document analysis)'),
            ('Mooncake store', 'KV pool over DRAM and SSD/NVMe of the GPU cluster')]
    for i, (a, b) in enumerate(rows):
        yy = 378 + i*18
        f.text(30, yy, a, 13, 'ink', True, 'start')
        f.text(230, yy, b, 13, 'green' if i < 3 else 'muted', anchor='start')
    f.save()


def build():
    kv_bytes()
    kivi()
    paged()
    radix()
    disagg()
    cp()
    tiers()
