"""04 · Latent KV: MLA records, weight absorption, the decoupled RoPE lane, head count, conversion."""
from figkit import Fig, _render_math

# DeepSeek-V3 config (deepseek-ai/DeepSeek-V3 config.json)
HID, DC, DR, DH, HQ, LAYERS = 7168, 512, 64, 128, 128, 61


def _formula_row(f, y, pieces, size=21, gap=44, cx=450):
    widths = [_render_math(p, size, '#253340')[1] for p in pieces]
    x = cx - (sum(widths) + gap*(len(pieces)-1))/2
    for p, w in zip(pieces, widths):
        f.math(x, y, p, size)
        x += w + gap


def _hbar(f, x, y, segs, h=20, sc=1.0):
    """Horizontal segmented bar: segs = [(n_elements, colour), ...]; returns end x."""
    for n, c in segs:
        w = max(2, n*sc)
        f.rect(x, y, w-1, h, c, .9, 2)
        x += w
    return x


def _dots(f, x, y, n, cols, r=3.2, step=9, color='gold'):
    for i in range(n):
        f.parts.append(f'<circle cx="{x + (i % cols)*step}" cy="{y + (i//cols)*step}" r="{r}" fill="#d09124"/>')


def record():
    f = Fig('mla-record', 'MLA record in DeepSeek-V3: h_t (7168) is down-projected to a latent c_t of 512 and a shared RoPE key k^R of 64; only these 576 elements are cached per token and layer. Per-head keys (128 heads × 128, plus the shared 64) and values (128 × 128) are reconstructed by W_UK and W_UV and never cached. Elements per token per layer: MHA with the same shapes 40,960, GQA-8 2,048, MLA 576, K=V MQA 512. All 61 layers: 68.6 KiB per token in BF16.', 470)
    _formula_row(f, 34, [r'\c{blue}{c_t} = W_{DKV} h_t \in \R^{512}', r'\c{teal}{k_t^R} = \c{teal}{RoPE}_t(W_{KR} h_t) \in \R^{64}', r'\text{cache} = 512 + 64 = \c{blue}{576}'], 20, 40)
    # h_t
    f.vec(30, 70, 12, 15, 'ink', values=[.25]*12)
    f.math(37, 266, r'h_t', 17, anchor='middle')
    f.text(37, 284, '7168', 13, 'muted')
    f.arrow(48, 120, 86, 120, 'blue', width=1.6)
    f.arrow(48, 220, 86, 220, 'teal', width=1.6)
    f.box(88, 104, 74, 32, '$W_{DKV}', 'blue', True, 15)
    f.text(125, 152, '512 × 7168', 13, 'muted')
    f.box(88, 204, 74, 32, '$W_{KR}', 'teal', True, 15)
    f.text(125, 252, '64 × 7168 + RoPE', 13, 'muted')
    f.arrow(164, 120, 192, 120, 'blue', width=1.6)
    f.arrow(164, 220, 192, 220, 'teal', width=1.6)
    # cached record (vertical)
    f.parts.append('<rect x="190" y="68" width="62" height="196" rx="8" fill="#1f9a74" fill-opacity=".08" stroke="#1f9a74" stroke-width="1.6" stroke-dasharray="5 4"/>')
    f.vec(206, 82, 8, 15, 'blue', values=[.85]*8)
    f.math(224, 140, r'\c{blue}{c_t}', 16)
    f.vec(206, 206, 2, 15, 'teal', values=[.9]*2)
    f.math(224, 226, r'\c{teal}{k^R}', 15)
    f.brace(252, 82, 252, 200, '512', 'blue', 'right', 13, 6)
    f.brace(252, 206, 252, 234, '64', 'teal', 'right', 13, 6)
    f.text(221, 282, 'cached: 576', 14, 'green', True)
    # expansion (not cached)
    f.panel(310, 60, 570, 222, 'Reconstructed per head, never cached (MHA mode: train / prefill)')
    f.arrow(268, 110, 330, 112, 'line', dash=True, width=1.6)
    f.arrow(268, 150, 330, 190, 'line', dash=True, width=1.6)
    f.box(334, 96, 74, 32, '$W_{UK}', 'blue', False, 15)
    f.box(334, 174, 74, 32, '$W_{UV}', 'purple', False, 15)
    f.text(371, 144, '16384 × 512', 13, 'muted')
    f.text(371, 222, '16384 × 512', 13, 'muted')
    f.arrow(410, 112, 436, 112, 'blue', width=1.4)
    f.arrow(410, 190, 436, 190, 'purple', width=1.4)
    f.matrix(440, 96, 4, 16, 9, 'blue', values=[[.55]*16]*4)
    f.matrix(440, 174, 4, 16, 9, 'purple', values=[[.55]*16]*4)
    f.math(590, 116, r'\c{blue}{k^C_{t,i}} \in \R^{128}', 15)
    f.math(590, 194, r'\c{purple}{v_{t,i}} \in \R^{128}', 15)
    f.text(511, 152, '128 heads × 128', 13, 'muted')
    # one head key = [k^C ; k^R]
    f.text(706, 100, 'key of head i', 13, 'ink', True, 'start')
    x = _hbar(f, 706, 110, [(128, 'blue'), (64, 'teal')], 18, .8)
    f.brace(706, 130, x-1, 130, '192 = 128 + 64', 'muted', 'below', 13, 4)
    f.math(706, 178, r'\c{teal}{k^R} \text{ is shared by}', 14, 'teal')
    f.text(706, 196, 'all 128 heads', 13, 'teal', anchor='start')
    f.text(330, 262, 'cost of not caching these: one up-projection per token, or absorption (next card)', 13, 'muted', anchor='start')
    # bottom: elements per token per layer
    f.text(20, 316, 'Elements per token per layer (DeepSeek-V3 shapes, BF16 bytes)', 15, 'ink', True, 'start')
    rows = [('MHA · 128 × (192 + 128)', [(40960, 'red')], '40,960 · 80 KiB'),
            ('GQA-8 · 2 × 8 × 128', [(2048, 'blue')], '2,048 · 4 KiB'),
            ('MLA · 512 + 64', [(512, 'blue'), (64, 'teal')], '576 · 1,152 B'),
            ('K=V MQA (V4) · 512', [(448, 'purple'), (64, 'teal')], '512 · 1,024 B')]
    sc = 480/40960
    for r, (name, segs, lab) in enumerate(rows):
        y = 330 + r*28
        f.text(250, y+15, name, 14, 'ink', anchor='end')
        x = _hbar(f, 262, y, segs, 19, sc)
        f.text(x+8, y+15, lab, 14, 'red' if r == 0 else 'green' if r == 2 else 'muted', r == 2, 'start')
    f.text(20, 458, f'DeepSeek-V3, all {LAYERS} layers: {LAYERS} × 1,152 B = {LAYERS*1152:,} B = {LAYERS*1152/1024:.1f} KiB per token (V2 Table 1: 576 ≈ GQA with 2.25 groups).', 13, 'muted', anchor='start')
    f.save()


def absorb():
    f = Fig('mla-absorb', 'Weight absorption. The score q_i^T W_UK,i c_j is re-bracketed as (W_UK,i^T q_i)^T c_j, so each of the 128 heads gets a 576-dimensional query (512 absorbed + 64 RoPE) and all heads read the same cached 576-element row: MQA with one KV head, d_k 576, d_v 512. W_UV folds into the output projection. Per cached token and layer in BF16: 1,152 bytes and 278,528 FLOPs, intensity about 242 FLOP/B (2·H_q), versus 1 for an expanded MHA cache; H100 ridge 295. Toy check: 7·(2·4+3·5) = 161 = 14·4 + 21·5.', 470)
    _formula_row(f, 34, [r'\c{gold}{q_i}^{\top}(W_{UK,i}\c{blue}{c_j}) = (W_{UK,i}^{\top}\c{gold}{q_i})^{\top}\c{blue}{c_j}', r'\c{green}{o_i} = W_{UV,i} \sum_j p_{ij}\c{blue}{c_j}'], 20, 50)
    # left: MHA mode
    f.panel(20, 56, 420, 238, 'MHA mode · train / prefill')
    f.text(36, 102, 'cache row', 13, 'muted', anchor='start')
    _hbar(f, 110, 88, [(512, 'blue'), (64, 'teal')], 18, .3)
    f.arrow(190, 110, 190, 136, 'blue', width=1.4)
    f.text(198, 128, 'W_UK, W_UV per head: expand', 13, 'blue', anchor='start')
    for h in range(4):
        y = 144 + h*24
        f.circle(48, y+9, 7, color='gold', fill=True)
        f.arrow(58, y+9, 106, y+9, 'gold', width=1.1)
        _hbar(f, 110, y, [(128, 'blue'), (64, 'teal')], 16, .3)
        _hbar(f, 172, y, [(128, 'purple')], 16, .3)
    f.text(250, 170, '× 128 heads', 14, 'ink', True, 'start')
    f.text(250, 190, 'd_k 192, d_v 128', 13, 'muted', anchor='start')
    f.text(36, 260, 'expanded per token: 128 × (192 + 128) = 40,960 elements', 13, 'muted', anchor='start')
    f.text(36, 280, 'DeepSeek: training + prefill (V3.2 App. A)', 13, 'green', anchor='start')
    # right: MQA mode
    f.panel(460, 56, 420, 238, 'MQA mode · decode (absorbed)')
    f.circle(488, 104, 8, color='gold', fill=True)
    f.math(502, 109, r'\c{gold}{q_i} \in \R^{128}', 14)
    f.arrow(580, 104, 610, 104, 'gold', width=1.4)
    f.math(614, 109, r'W_{UK,i}^{\top}', 15)
    f.arrow(664, 104, 690, 104, 'gold', width=1.4)
    _hbar(f, 694, 95, [(512, 'gold'), (64, 'teal')], 18, .3)
    f.text(870, 132, 'query 576 = 512 + 64', 13, 'gold', anchor='end')
    _dots(f, 490, 150, 128, 16, 2.6, 7.4)
    f.text(490, 230, '128 query heads', 13, 'gold', True, 'start')
    for k in range(5):
        f.line(610, 150+k*16, 690, 194, 'gold', .9, opacity=.6)
    _hbar(f, 694, 186, [(512, 'blue'), (64, 'teal')], 18, .3)
    f.text(870, 224, 'one shared row: K 576, V 512', 13, 'blue', True, 'end')
    f.math(612, 262, r'\c{green}{o_i} = (W_{O,i}W_{UV,i})\, z_i', 15)
    f.math(612, 284, r'z_i = \sum_j p_{ij}\c{blue}{c_j} \in \R^{512} \text{ · } W_{UV} \text{ folds into } W_O', 13, 'muted')
    # bottom: cost readout
    f.text(20, 324, 'Per cached token per layer, BF16 (DeepSeek-V3: H_q = 128)', 15, 'ink', True, 'start')
    rows = [('expanded MHA cache', 1, 'red'), ('absorbed MQA mode', 278528/1152, 'green'), ('H100 ridge', 295, 'muted')]
    f.bars(20, 340, rows, width=250, bar=18, gap=9, label_w=170, size=14, value_fmt=lambda v: f'{v:.0f} FLOP/B')
    f.text(20, 448, 'reads 576 × 2 B = 1,152 B · FLOPs 2·128·(576 + 512) = 278,528', 13, 'muted', anchor='start')
    # toy check
    f.panel(560, 316, 320, 140, 'toy check: same score')
    f.math(576, 362, r'\text{toy: } W_{UK} = [2\;3],\; \c{blue}{c} = [4;5],\; \c{gold}{q} = 7', 15)
    f.math(576, 392, r'\c{gold}{q}(W_{UK}\c{blue}{c}) = 7 \cdot 23 = 161', 16)
    f.math(576, 420, r'(W_{UK}^{\top}\c{gold}{q})^{\top}\c{blue}{c} = 14\cdot4 + 21\cdot5 = 161', 16)
    f.text(576, 446, 'associativity: no approximation', 13, 'green', True, 'start')
    f.save()


def rope_lane():
    f = Fig('mla-rope', 'Why RoPE breaks absorption and how MLA fixes it. With RoPE on the content key, the score is q^T R_(j-t) W_UK c_j: the rotation depends on the key position j and sits between the query and W_UK, so W_UK cannot be folded into the query once. MLA adds a separate 64-dimensional lane: per-head rotated queries q^R and one rotated key k^R shared by all 128 heads. Score = absorbed content term over 512 dims + positional term over 64 dims, scaled by 1/sqrt(192). Shared k^R costs 64 elements per token; per-head rotated keys would cost 8,192.', 444)
    _formula_row(f, 34, [r'(\c{teal}{R_t}\c{gold}{q})^{\top}\c{teal}{R_j}W_{UK}\c{blue}{c_j} = \c{gold}{q}^{\top}\c{teal}{R_{j-t}}W_{UK}\c{blue}{c_j}'], 21)
    # top: broken absorption
    f.panel(20, 56, 860, 120, 'RoPE on the content key: absorption breaks')
    xs = [(60, r'\c{gold}{q}^{\top}', 'gold'), (160, r'\c{teal}{R_{j-t}}', 'teal'), (270, r'W_{UK}', 'blue'), (380, r'\c{blue}{c_j}', 'blue')]
    for x, m, c in xs:
        f.box(x, 88, 80, 40, '$' + m, c, False, 18)
    for a, b in ((140, 160), (240, 270), (350, 380)):
        f.text((a+b)/2, 113, '·', 20, 'ink')
    f.brace(60, 132, 350, 132, '', 'red', 'below', 13, 6)
    f.line(170, 134, 240, 160, 'red', 2)
    f.line(170, 160, 240, 134, 'red', 2)
    f.text(250, 160, 'cannot fold into one query: R depends on j', 14, 'red', True, 'start')
    f.text(500, 102, 'matrix products do not commute:', 14, 'ink', anchor='start')
    f.text(500, 122, 'W_UK would need a new absorbed query', 14, 'ink', anchor='start')
    f.text(500, 142, 'for every cached position j (V2 §2.1.3)', 14, 'ink', anchor='start')
    # bottom: decoupled lane
    f.panel(20, 190, 860, 240, 'Decoupled lane: content 512 (no RoPE) + position 64 (RoPE)')
    sc = 0.62
    f.text(150, 245, 'query, head i', 14, 'gold', True, 'end')
    xq = _hbar(f, 160, 230, [(512, 'gold')], 22, sc)
    _hbar(f, xq, 230, [(64, 'teal')], 22, sc)
    f.math(160 + 256*sc, 272, r'W_{UK,i}^{\top}\c{gold}{q_{t,i}^C} \text{ (absorbed, 512)}', 14, anchor='middle')
    f.math(xq + 20*sc, 272, r'\c{teal}{q_{t,i}^R}', 14, anchor='middle')
    f.text(150, 305, 'cached row', 14, 'blue', True, 'end')
    xk = _hbar(f, 160, 290, [(512, 'blue')], 22, sc)
    _hbar(f, xk, 290, [(64, 'teal')], 22, sc)
    f.math(160 + 256*sc, 332, r'\c{blue}{c_j} \text{ (no RoPE: absorbable)}', 14, anchor='middle')
    f.math(xk + 20*sc, 332, r'\c{teal}{k_j^R}', 14, anchor='middle')
    f.math(160, 374, r'e_{t,j,i} = ((W_{UK,i}^{\top}\c{gold}{q^C})^{\top}\c{blue}{c_j} + (\c{teal}{q^R})^{\top}\c{teal}{k_j^R}) / \sqrt192', 17)
    f.text(160, 402, '192 = d_h + d_r = 128 + 64 (the MHA-mode head size)', 13, 'muted', anchor='start')
    # cost of sharing k^R
    f.text(640, 238, 'RoPE key per token per layer', 13, 'ink', True, 'start')
    f.bars(640, 250, [('shared', 64, 'teal'), ('per head', 8192, 'red')], width=110, bar=18, gap=10, label_w=70, size=13, value_fmt=lambda v: f'{v:,}')
    f.math(640, 326, r'\text{one } \c{teal}{k^R} \text{ serves all 128 heads}', 14, 'teal')
    f.text(640, 346, '64 of 576 cached = 11%', 13, 'muted', anchor='start')
    f.save()


def heads():
    f = Fig('mla-heads', 'Head count changes FLOPs, not cache. DeepSeek-V3 (128 heads) and Kimi K2 (64 heads) cache the same 576 elements per token and layer. Decode FLOPs per cached token per layer: 278,528 versus 139,264; intensity in BF16 about 242 versus 121 FLOP/B, against an H100 ridge of 295. Kimi K2 reports that 64 to 128 heads costs +83% inference FLOPs at 128K for 0.5 to 1.2% lower validation loss.', 420)
    _formula_row(f, 34, [r'\text{cache} = d_c + d_r = \c{blue}{576} \text{ for any } H_q', r'\text{FLOP / cached token} = 2\c{gold}{H_q}(576 + 512)'], 20, 50)
    for k, (name, n, cols) in enumerate((('DeepSeek-V3 · 128 heads', 128, 32), ('Kimi K2 · 64 heads', 64, 32))):
        y = 70 + k*124
        f.text(20, y+16, name, 15, 'ink', True, 'start')
        _hbar(f, 20, y+30, [(512, 'blue'), (64, 'teal')], 20, .42)
        f.text(270, y+45, '576', 14, 'blue', True, 'start')
        for i in range(6):
            f.line(140, y+52, 60 + i*24, y+70, 'gold', .9, opacity=.6)
        _dots(f, 26, y+78, n, cols, 3.2, 9)
    f.divider(320, 64, 330)
    # right: three readouts
    x = 340
    f.text(x, 82, 'cache / token / layer (elements)', 14, 'ink', True, 'start')
    f.bars(x, 92, [('V3', 576, 'blue'), ('K2', 576, 'blue')], width=300, bar=16, gap=6, vmax=576, label_w=40, size=13, value_fmt=lambda v: f'{v:,}  identical')
    f.text(x, 160, 'decode FLOPs / cached token / layer', 14, 'ink', True, 'start')
    f.bars(x, 170, [('V3', 278528, 'gold'), ('K2', 139264, 'gold')], width=300, bar=16, gap=6, label_w=40, size=13, value_fmt=lambda v: f'{v:,}')
    f.text(x, 238, 'intensity, BF16 (FLOP/B)', 14, 'ink', True, 'start')
    f.bars(x, 248, [('V3', 278528/1152, 'green'), ('K2', 139264/1152, 'green'), ('H100', 295, 'muted')], width=300, bar=16, gap=6, vmax=295, label_w=40, size=13, value_fmt=lambda v: f'{v:.0f}')
    f.hdivider(342)
    f.text(20, 370, 'Kimi K2 §2.3 (reported): 64 → 128 heads = +83% inference FLOPs at 128K tokens,', 14, 'ink', anchor='start')
    f.text(20, 392, 'for only 0.5–1.2% lower validation loss. Same 61 layers, same 68.6 KiB/token cache.', 14, 'ink', anchor='start')
    f.save()


def convert():
    f = Fig('mla-convert', 'Converting a trained MHA or GQA model to MLA, Llama-2-7B per layer (32 heads × 128, d 4096). Step 1: keep RoPE on a few high-norm frequency pairs (MHA2MLA) or concentrate it in one head (TransMLA); the rest become NoPE. Step 2: joint SVD of the stacked NoPE key and value projections gives W_DKV (r × 4096) and the up-projections. Step 3: fine-tune with 0.3–0.6% of pre-training data (MHA2MLA) or 6B tokens (TransMLA). KV bytes per token and layer: MHA 16 KiB; MHA2MLA 5 KiB (−68.75%); with Int4 1.25 KiB (−92.19%); TransMLA 1,152 B (−92.97%).', 450)
    _formula_row(f, 34, [r'[\c{blue}{W_K^{nope}};\c{purple}{W_V}] \approx U_r\Sigma_r V_r^{\top}', r'W_{DKV} = \Sigma_r^{1/2}V_r^{\top}', r'[W_{UK};W_{UV}] = U_r\Sigma_r^{1/2}'], 20, 40)
    # step 1
    f.panel(20, 56, 260, 224, '1 · partial RoPE')
    vals = [[('.9', 'teal') if r in (0, 5) else None for _ in range(14)] for r in range(10)]
    vals = [[(.85, 'teal') if r in (0, 5) else (.55, 'blue') for _ in range(14)] for r in range(10)]
    f.matrix(40, 92, 10, 14, 11, 'blue', vals)
    f.text(32, 222, 'W_K: 4096 × 4096 (32 heads × 128)', 12, 'muted', anchor='start')
    f.text(200, 98, 'RoPE kept', 13, 'teal', True, 'start')
    f.text(200, 116, '(few pairs)', 12, 'teal', anchor='start')
    f.text(200, 160, 'NoPE', 13, 'blue', True, 'start')
    f.text(200, 178, '→ absorbable', 12, 'blue', anchor='start')
    f.text(32, 252, 'MHA2MLA: top-norm pairs per head', 12, 'ink', anchor='start')
    f.text(32, 270, 'TransMLA: fold RoPE into one head', 12, 'ink', anchor='start')
    f.arrow(282, 160, 300, 160, 'muted', width=1.6)
    # step 2
    f.panel(302, 56, 330, 224, '2 · joint SVD / PCA')
    f.matrix(320, 92, 6, 6, 11, 'blue', [[.55]*6]*6)
    f.matrix(320, 158, 6, 6, 11, 'purple', [[.55]*6]*6)
    f.text(354, 240, 'NoPE K ; V', 12, 'muted')
    f.text(400, 160, '≈', 22, 'ink')
    f.matrix(420, 92, 12, 2, 11, 'green', [[.5]*2]*12)
    f.text(431, 240, 'U Σ½', 12, 'muted')
    f.text(452, 160, '×', 18, 'ink')
    f.matrix(466, 140, 2, 12, 11, 'green', [[.8]*12]*2)
    f.brace(466, 140, 596, 140, 'r', 'green', 'above', 13, 6)
    f.text(531, 186, 'W_DKV: r × 4096', 13, 'green', True)
    f.text(531, 206, '→ cached latent c_t', 12, 'green')
    f.arrow(634, 160, 652, 160, 'muted', width=1.6)
    # step 3
    f.panel(654, 56, 226, 224, '3 · short fine-tune')
    f.text(670, 104, 'MHA2MLA', 14, 'ink', True, 'start')
    f.text(670, 124, '0.3–0.6% of', 13, 'ink', anchor='start')
    f.text(670, 142, 'pre-training data', 13, 'ink', anchor='start')
    f.text(670, 182, 'TransMLA', 14, 'ink', True, 'start')
    f.text(670, 202, '6B tokens', 13, 'ink', anchor='start')
    f.text(670, 244, 'not lossless: weights', 12, 'red', anchor='start')
    f.text(670, 262, 'change, then recover', 12, 'red', anchor='start')
    # bars
    f.text(20, 310, 'Llama-2-7B KV bytes per token per layer (reported reductions)', 15, 'ink', True, 'start')
    rows = [('MHA · 8,192 elements', 16384, 'red'), ('MHA2MLA · BF16', 5120, 'blue'),
            ('MHA2MLA + Int4', 1280, 'green'), ('TransMLA · 576 elements', 1152, 'green')]
    notes = {16384: '16 KiB', 5120: '5 KiB  (−68.75%)', 1280: '1.25 KiB  (−92.19%)', 1152: '1,152 B  (−92.97%)'}
    f.bars(20, 324, rows, width=420, bar=18, gap=8, label_w=200, size=14, value_fmt=lambda v: notes[v])
    f.text(880, 440, 'TransMLA 576 = 8,192 × (1 − 0.9297), the DeepSeek record size (computed)', 12, 'muted', anchor='end')
    f.save()


def build():
    record()
    absorb()
    rope_lane()
    heads()
    convert()
