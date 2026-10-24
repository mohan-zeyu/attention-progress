"""07 · Evict from the cache: the stored cache shrinks, and the decision is permanent."""
import math
from figkit import Fig, COL

KIB_TOK = 32*2*8*128*2 / 1024          # Llama-3-8B, BF16: 128 KiB per token (all layers)


def strip(f, parts, y=36, size=21, gap=36, x=30):
    """Formula strip: parts rendered left to right with real gaps (SVG collapses repeated spaces)."""
    for part in parts:
        x += f.math(x, y, part, size) + gap


def _softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v/s for v in e]


def _h2o_sim(n=16, budget=6):
    """Run H2O on a toy attention pattern. Returns per-step state and the final accumulated scores."""
    base = {1: 2.3, 4: 1.9, 9: 2.1}
    noise = [0, .1, .45, .2, 0, .35, .05, .5, .15, 0, .3, .25, .4, .1, .2, .3, .15]
    cache, S, states = [], {}, []
    for t in range(1, n+1):
        cache.append(t)
        S.setdefault(t, 0.0)
        logits = [base.get(j, noise[j]) + (1.0 if j >= t-2 else 0) for j in cache]
        for j, a in zip(cache, _softmax(logits)):
            S[j] += a
        evicted = []
        if len(cache) > budget:
            recent = cache[-budget//2:]
            rest = sorted([j for j in cache if j not in recent], key=lambda j: -S[j])
            heavy = rest[:budget//2]
            evicted = rest[budget//2:]
            cache = sorted(heavy) + recent
        else:
            recent, heavy = cache[-budget//2:], [j for j in cache if j not in cache[-budget//2:]]
        states.append((t, list(cache), list(heavy), list(recent), evicted))
    return states, S


def h2o():
    states, S = _h2o_sim()
    f = Fig('evict-h2o', 'H2O on a toy 16-token decode with budget 6 (3 heavy hitters + 3 recent). Tokens 1, 4 and 9 collect the most accumulated attention and stay; every other token leaves the recent window and is evicted for good. Real setting: 20% budget split 10% + 10%; Llama-3-8B at 32K drops from 4 GiB to 0.8 GiB of KV.', 440)
    strip(f, [r'\c{blue}{S_j} \leftarrow \c{blue}{S_j} + \c{gold}{a_{t,j}}', r'\c{green}{\text{keep}} = \text{Recent}_{B/2} ∪ \text{Top}_{B/2}(\c{blue}{S})', r'\c{red}{\text{evict the rest, forever}}'], size=21)
    x0, y0, c = 92, 92, 24
    f.text(x0 + 8*c, 72, 'cached token j (toy, budget B = 6)', 14, 'muted')
    for j in range(1, 17):
        f.text(x0 + (j-1)*c + c/2 - 1, y0 - 6, str(j), 12, 'muted')
    gone = set()
    shown = [s for s in states if s[0] >= 7]
    for r, (t, cache, heavy, recent, evicted) in enumerate(shown):
        y = y0 + r*c
        f.text(x0 - 10, y + c*.65, f't = {t}', 13, 'muted', anchor='end')
        for j in range(1, 17):
            x = x0 + (j-1)*c
            if j > t:
                continue
            if j in evicted:
                f.rect(x, y, c-3, c-3, 'red', .85, 3)
                f.text(x + (c-3)/2, y + c*.66, '×', 15, 'white', True)
            elif j in gone:
                f.rect(x, y, c-3, c-3, 'line', .35, 3)
            elif j in heavy:
                f.rect(x, y, c-3, c-3, 'green', .9, 3)
            else:
                f.rect(x, y, c-3, c-3, 'green', .38, 3)
        gone |= set(evicted)
    gy = y0 + len(shown)*c + 16
    # accumulated score bars aligned with the columns
    smax = max(S.values())
    for j in range(1, 17):
        h = 46*S[j]/smax
        x = x0 + (j-1)*c
        f.rect(x+2, gy + 50 - h, c-7, h, 'blue', .8, 2)
        if j in (1, 4, 9):
            f.text(x + (c-3)/2, gy + 44 - h, f'{S[j]:.1f}', 12, 'blue', True)
    f.line(x0, gy+50, x0 + 16*c - 3, gy+50, 'line', 1)
    f.text(x0 - 10, gy + 34, 'S at t=16', 13, 'blue', anchor='end')
    # legend
    lx = 520
    f.rect(lx, 84, 16, 16, 'green', .9, 3); f.text(lx+24, 97, 'heavy hitter (Top-3 of S)', 14, 'ink', anchor='start')
    f.rect(lx, 110, 16, 16, 'green', .38, 3); f.text(lx+24, 123, 'recent window (last 3)', 14, 'ink', anchor='start')
    f.rect(lx, 136, 16, 16, 'red', .85, 3); f.text(lx+24, 149, 'evicted at this step', 14, 'ink', anchor='start')
    f.rect(lx, 162, 16, 16, 'line', .35, 3); f.text(lx+24, 175, 'gone for every later query', 14, 'ink', anchor='start')
    f.hdivider(198, 520, 880)
    f.text(520, 224, 'Real setting: 20% budget = 10% heavy + 10% recent', 15, 'ink', True, 'start')
    f.text(520, 246, 'Llama-3-8B, 32K tokens, BF16 KV:', 14, 'muted', anchor='start')
    full = 32768*KIB_TOK/2**20
    f.bars(520, 260, [('full', full, 'blue'), ('H2O 20%', full*.2, 'green')], width=170, bar=22, gap=10, label_w=80, size=14,
           value_fmt=lambda v: f'{v:.1f} GiB')
    f.text(520, 346, 'reported (OPT-6.7B / 30B):', 14, 'muted', anchor='start')
    f.text(520, 368, 'up to 29× throughput vs DeepSpeed,', 14, 'ink', anchor='start')
    f.text(520, 388, 'HF Accelerate; 3× vs FlexGen', 14, 'ink', anchor='start')
    f.text(520, 420, 'early tokens are summed longest → bias', 13, 'red', anchor='start')
    f.save()


def snapkv():
    n, obs = 40, 8
    hot = {0: .55, 11: .9, 12: .8, 25: .7, 33: .85, 34: .5, 20: .35}
    noise = [(.08 + .07*((j*7) % 5)/4) for j in range(n)]
    A = [[min(1, hot.get(j, noise[j]) * (0.75 + 0.25*((i*3+j) % 4)/3)) for j in range(n)] for i in range(obs)]
    C = [sum(A[i][j] for i in range(obs)) for j in range(n)]
    P = [max(C[max(0, j-1):j+2]) for j in range(n)]
    B = 8
    keep = set(sorted(range(n), key=lambda j: -P[j])[:B])
    f = Fig('evict-snapkv', 'SnapKV on a toy 40-token prompt: the last 8 prompt queries (the observation window) vote with their attention, votes are max-pooled so neighbours of a hit survive, and each head keeps its own top 8 positions plus the window. Real setting: window 32, pool kernel 7, budget 1024 of a 16K prompt (6.25%).', 472)
    strip(f, [r'\c{blue}{C_j} = \sum_{i \in \c{gold}{\text{obs}}} \text{softmax}(K \c{gold}{q_i} / √d_h)_j', r'\c{green}{𝒦} = \text{Top}_B(\text{maxpool}(\c{blue}{C})) ∪ \c{gold}{\text{obs}}'], size=21)
    x0, y0, c = 160, 78, 13
    f.mask(x0, y0, obs, c, lambda i, j: ('gold', .1 + .85*A[i][j]), m=n)
    f.brace(x0 - 4, y0, x0 - 4, y0 + obs*c - 2, '', 'gold', 'left', 13, 4)
    f.text(x0 - 16, y0 + 50, 'obs queries', 13, 'gold', True, 'end')
    f.text(x0 - 16, y0 + 68, '(toy 8; real 32)', 13, 'muted', anchor='end')
    f.brace(x0, y0, x0 + n*c - 2, y0, 'prefix keys j (toy 40 of a 16K prompt)', 'muted', 'above', 13, 4)
    # observation-window keys themselves
    ox = x0 + n*c + 12
    f.mask(ox, y0, obs, c, lambda i, j: ('gold', .85) if j <= i else None, m=obs)
    f.text(ox + obs*c/2, y0 - 10, 'obs keys', 13, 'gold', True)
    # votes and pooled votes
    vy = y0 + obs*c + 14
    cmax = max(C)
    for j in range(n):
        h = 34*C[j]/cmax
        f.rect(x0 + j*c + 1, vy + 34 - h, c - 3, h, 'blue', .75, 1)
        h2 = 34*P[j]/cmax
        f.rect(x0 + j*c + 1, vy + 84 - h2, c - 3, h2, 'blue', .45, 1)
    f.text(x0 - 16, vy + 26, 'votes C', 14, 'blue', True, 'end')
    f.text(x0 - 16, vy + 76, 'maxpool C', 14, 'blue', True, 'end')
    f.text(x0 - 16, vy + 94, '(toy k = 3)', 13, 'muted', anchor='end')
    # keep masks for three heads
    my = vy + 104
    others = [keep, {0, 1, 5, 6, 18, 19, 30, 31}, {0, 11, 12, 13, 21, 22, 37, 38}]
    for h, kept in enumerate(others):
        y = my + h*20
        f.text(x0 - 16, y + 13, f'head {h+1}', 13, 'ink', anchor='end')
        for j in range(n):
            f.rect(x0 + j*c, y, c - 2, 16, 'green' if j in kept else 'red', .85 if j in kept else .22, 2)
        for j in range(obs):
            f.rect(ox + j*c, y, c - 2, 16, 'gold', .85, 2)
    f.text(ox + obs*c + 10, my + 13, 'keep B = 8', 13, 'green', True, 'start')
    f.text(ox + obs*c + 10, my + 33, '+ window', 13, 'gold', True, 'start')
    f.text(ox + obs*c + 10, my + 53, 'per head', 13, 'muted', anchor='start')
    # generated tokens are not compressed
    gy = my + 66
    f.rect(x0, gy + 8, n*c - 2, 3, 'line', 1)
    f.text(x0, gy + 30, 'once, at the end of prefill · decode tokens are appended and never pruned', 14, 'muted', anchor='start')
    f.hdivider(gy + 44, 30, 870)
    f.text(30, gy + 70, 'Real: window 32 · max-pool kernel 7 · budget 1024 / 2048 / 4096 · 16K prompt at budget 1024 keeps 6.25% + 32', 14, 'ink', anchor='start')
    f.text(30, gy + 92, 'reported at 16K: 3.6× generation speed, 8.2× memory efficiency · up to 380K tokens on one A100-80GB', 14, 'green', anchor='start')
    f.save()


def budgets():
    L, mean, beta = 32, 128, 20
    ktot = L*mean
    k_last = ktot/(beta*L)
    k0 = 2*ktot/L - k_last
    ks = [k0 - (k0 - k_last)*l/(L-1) for l in range(L)]
    f = Fig('evict-budgets', f'Same scorer, different budget split. PyramidKV gives Llama-3-8B layers a linear budget from {k0:.0f} entries per head at layer 0 to {k_last:.1f} at layer 31 (mean 128, beta 20). Ada-KV pools one layer\'s budget across heads: a toy layer budget of 12 over 4 heads becomes 6, 3, 2 and 1.', 420)
    strip(f, [r'\c{green}{k^{\ell}} = k^{0} - \ell (k^{0} - k^{L-1}) / (L-1)', r'k^{L-1} = \c{blue}{k_{tot}} / (\c{teal}{\beta} L)', r'k^{0} = 2\c{blue}{k_{tot}} / L - k^{L-1}'], size=20)
    f.text(30, 74, 'PyramidKV · Llama-3-8B, L = 32, mean 128 per head, β = 20', 15, 'ink', True, 'start')
    x0, base, w, H = 60, 300, 14, 190
    for l, k in enumerate(ks):
        h = H*k/k0
        f.rect(x0 + l*w, base - h, w - 3, h, 'green', .85, 2)
    yu = base - H*mean/k0
    f.line(x0 - 4, yu, x0 + L*w, yu, 'blue', 1.6, dash=True)
    f.text(x0 + L*w + 6, yu + 5, 'uniform 128', 13, 'blue', True, 'start')
    f.text(x0 + 6, base - H - 8, f'k⁰ = {k0:.0f}', 14, 'green', True, 'start')
    f.text(x0 + (L-1)*w + 6, base - H*k_last/k0 - 26, f'k³¹ = {k_last:.1f}', 14, 'green', True, 'middle')
    f.line(x0 - 4, base, x0 + L*w, base, 'line', 1)
    for l in (0, 8, 16, 24, 31):
        f.text(x0 + l*w + (w-3)/2, base + 18, str(l), 12, 'muted')
    f.text(x0 + L*w/2, base + 38, 'layer ℓ (lower layers attend broadly, upper layers focus)', 13, 'muted')
    f.text(x0, base + 66, 'both splits spend 32 × 128 = 4,096 entries per head across layers', 13, 'muted', anchor='start')
    f.divider(560, 62, 400)
    # Ada-KV toy
    f.text(580, 74, 'Ada-KV · one layer, toy', 15, 'ink', True, 'start')
    scores = [[.95, .9, .85, .8, .7, .6, .2, .1, .05, .05, .05, .02],
              [.75, .65, .55, .3, .2, .1, .1, .05, .05, .03, .02, .02],
              [.68, .58, .4, .3, .2, .15, .1, .05, .03, .02, .02, .01],
              [.62, .35, .25, .2, .15, .1, .08, .05, .03, .02, .01, .01]]
    flat = sorted([(scores[h][j], h, j) for h in range(4) for j in range(12)], reverse=True)[:12]
    chosen = {(h, j) for _, h, j in flat}
    counts = [sum(1 for (h, _) in chosen if h == hh) for hh in range(4)]
    mx, my, c = 600, 104, 15
    f.text(mx + 6*c, my - 8, 'scores, 12 cached tokens', 13, 'muted')
    for h in range(4):
        f.text(mx - 8, my + h*(c+6) + 13, f'h{h+1}', 13, 'ink', anchor='end')
        for j in range(12):
            on = (h, j) in chosen
            f.rect(mx + j*c, my + h*(c+6), c - 2, c, 'green' if on else 'blue', .9 if on else .12 + .6*scores[h][j], 2)
        f.text(mx + 12*c + 30, my + h*(c+6) + 13, str(counts[h]), 15, 'green', True)
        f.text(mx + 12*c + 80, my + h*(c+6) + 13, '3', 15, 'blue', True)
    f.text(mx + 12*c + 30, my - 8, 'Ada', 13, 'green', True)
    f.text(mx + 12*c + 80, my - 8, 'uniform', 13, 'blue', True)
    f.text(580, 232, 'layer budget B = 12: take the global', 14, 'ink', anchor='start')
    f.text(580, 252, 'top-12 over all heads, then count', 14, 'ink', anchor='start')
    f.text(580, 272, 'per head → 6, 3, 2, 1', 14, 'green', True, 'start')
    f.hdivider(300, 580, 880)
    f.text(580, 326, 'reported:', 14, 'muted', anchor='start')
    f.text(580, 348, 'PyramidKV ≈ full KV on LongBench at 12%', 14, 'ink', anchor='start')
    f.text(580, 370, 'Ada-SnapKV / Ada-Pyramid: plug-in,', 14, 'ink', anchor='start')
    f.text(580, 390, 'tested on 13 RULER + 16 LongBench sets', 14, 'ink', anchor='start')
    f.save()


def kvzip():
    toks = ['Order', '7314', 'ships', 'Tue', '.', 'Update', ':', 'Fri', '.', 'Refund', 'is', '40', 'to', 'card']
    q1 = [.20, .95, .40, .80, .07, .30, .05, .90, .03, .04, .02, .05, .02, .04]
    rz = [.45, .92, .55, .85, .05, .60, .10, .90, .06, .80, .35, .88, .15, .70]
    budget = 7
    k1 = set(sorted(range(len(toks)), key=lambda j: -q1[j])[:budget])
    kz = set(sorted(range(len(toks)), key=lambda j: -rz[j])[:budget])
    f = Fig('evict-kvzip', 'Query-aware vs query-agnostic scoring on a toy 14-token context at a 50% budget. SnapKV votes with question 1 and drops the refund amount, so question 2 fails. KVzip scores each KV pair by how much the model needs it to repeat the whole context, so both questions still find their evidence. Reported: 3 to 4 times smaller KV, about 2 times faster decode, contexts up to 170K.', 440)
    strip(f, [r'\c{blue}{s_j} = \max_{g \le G, \; i \in \c{gold}{\text{repeat}}} a^{(g)}_{i,j}', r'\c{green}{\text{keep}} = \text{Top}_{r%}(\c{blue}{s}) \text{ pooled over all heads}'], size=21)
    x0, w, g = 140, 50, 3
    y = 70
    f.text(x0 - 12, y + 22, 'context', 14, 'ink', True, 'end')
    for j, t in enumerate(toks):
        f.box(x0 + j*(w+g), y, w, 32, t, 'blue', False, 14)
    rows = [('SnapKV', 'votes from Q1', q1, k1, 116), ('KVzip', '"Repeat the', rz, kz, 190)]
    for name, sub, sc, kept, yy in rows:
        f.text(x0 - 12, yy + 14, name, 15, 'ink', True, 'end')
        f.text(x0 - 12, yy + 32, sub, 13, 'muted', anchor='end')
        if name == 'KVzip':
            f.text(x0 - 12, yy + 48, 'previous context:"', 13, 'muted', anchor='end')
        for j in range(len(toks)):
            f.rect(x0 + j*(w+g), yy, w, 22, 'blue', .1 + .8*sc[j], 3)
            f.text(x0 + j*(w+g) + w/2, yy + 16, f'{sc[j]:.2f}'.lstrip('0'), 13, 'white' if sc[j] > .5 else 'ink')
            on = j in kept
            f.rect(x0 + j*(w+g), yy + 28, w, 14, 'green' if on else 'red', .85 if on else .25, 3)
    f.text(x0 + 14*(w+g) - g, 268, 'keep 7 of 14 (toy 50%)', 13, 'green', True, 'end')
    f.hdivider(282, 30, 870)
    # outcome table
    f.text(40, 308, 'later question', 14, 'muted', anchor='start')
    f.text(400, 308, 'needs', 14, 'muted', anchor='start')
    f.text(560, 308, 'SnapKV', 14, 'ink', True)
    f.text(680, 308, 'KVzip', 14, 'ink', True)
    qa = [('Q1  When does 7314 arrive?', '7314, Fri', True), ('Q2  How much is the refund?', 'Refund, 40', False)]
    for r, (q, need, snap_ok) in enumerate(qa):
        yy = 336 + r*28
        f.text(40, yy, q, 15, 'gold', anchor='start')
        f.text(400, yy, need, 15, 'blue', anchor='start')
        f.text(560, yy, 'hit' if snap_ok else 'miss', 15, 'green' if snap_ok else 'red', True)
        f.text(680, yy, 'hit', 15, 'green', True)
    f.text(760, 316, 'cost: +1 pass', 14, 'red', True, 'start')
    f.text(760, 336, 'over the context', 13, 'muted', anchor='start')
    f.text(760, 354, '(2K-token chunks)', 13, 'muted', anchor='start')
    f.text(40, 418, 'reported: 3–4× smaller KV · ≈2× faster FlashAttention decode · contexts to 170K · LLaMA3.1, Qwen2.5, Gemma3', 14, 'green', anchor='start')
    f.save()


def duo():
    f = Fig('evict-duo', 'DuoAttention on Llama-3-8B: half of the 256 KV heads (32 layers x 8) are retrieval heads that keep the full cache; the rest are streaming heads that keep 64 sink plus 256 recent tokens. At 128K tokens the BF16 cache falls from 16 GiB to about 8.02 GiB. Placement of heads in the grid is a toy; the 50% ratio is the paper\'s.', 440)
    strip(f, [r'o_h = \c{teal}{\alpha_h} \text{Attn}_{\c{green}{full}} + (1-\c{teal}{\alpha_h}) \text{Attn}_{\c{red}{sink+recent}}', r'ℒ = \| h_{full} - h_{\alpha} \|^{2} + 0.05 \| \c{teal}{\alpha} \|_1'], size=21)
    x0, y0, c = 70, 92, 13
    f.text(30, 76, 'Llama-3-8B · 32 layers × 8 KV heads', 15, 'ink', True, 'start')
    order = sorted(range(256), key=lambda i: (i*97 + (i//8)*31) % 256)
    retrieval = set(order[:128])            # toy placement, exactly 50%
    for l in range(32):
        for h in range(8):
            retr = (l*8 + h) in retrieval
            f.rect(x0 + l*c, y0 + h*c, c - 2, c - 2, 'green' if retr else 'red', .85 if retr else .28, 2)
    f.text(x0 - 8, y0 + 4*c + 4, 'head', 13, 'muted', anchor='end')
    f.text(x0 + 16*c, y0 + 8*c + 18, 'layer →   (placement toy; ratio real)', 13, 'muted')
    f.rect(x0, y0 + 8*c + 32, 14, 14, 'green', .85, 2); f.text(x0 + 20, y0 + 8*c + 44, 'retrieval: full KV (50% under GQA, 25% for MHA Llama-2-7B)', 13, 'ink', anchor='start')
    f.rect(x0, y0 + 8*c + 54, 14, 14, 'red', .28, 2); f.text(x0 + 20, y0 + 8*c + 66, 'streaming: 64 sink + 256 recent, rest evicted', 13, 'ink', anchor='start')
    sy = y0 + 8*c + 96
    f.text(30, sy, 'streaming-head cache at 128K (not to scale)', 13, 'muted', anchor='start')
    f.box(30, sy + 10, 70, 26, 'sink 64', 'green', True, 13, .85)
    f.box(104, sy + 10, 250, 26, '130,752 evicted', 'red', True, 13, .25, text_color='red')
    f.box(358, sy + 10, 110, 26, 'recent 256', 'green', True, 13, .85)
    f.text(30, sy + 56, 'sink block: same idea as StreamingLLM (ch 05), per head', 13, 'muted', anchor='start')
    # per-head cache at 128K
    px = 520
    f.text(px, 76, 'one head at 128K tokens', 15, 'ink', True, 'start')
    f.rect(px, 96, 350, 22, 'green', .85, 3)
    f.text(px + 175, 113, 'retrieval head: 131,072 entries', 14, 'white', True)
    f.rect(px, 130, 3, 22, 'red', .9, 1)
    f.rect(px + 3, 130, 1, 22, 'red', .9, 1)
    f.text(px + 12, 146, 'streaming head: 64 + 256 = 320 entries (0.24%)', 14, 'red', True, 'start')
    f.text(px, 182, 'whole model, BF16, 128K tokens', 14, 'muted', anchor='start')
    full = 131072*KIB_TOK/2**20
    duo_g = full/2 + 320*KIB_TOK/2/2**20
    f.bars(px, 194, [('full', full, 'blue'), ('Duo', duo_g, 'green')], width=200, bar=22, gap=10, label_w=60, size=14,
           value_fmt=lambda v: f'{v:.2f} GiB')
    f.hdivider(262, px, 880)
    f.text(px, 286, 'reported (A100):', 14, 'muted', anchor='start')
    f.text(px, 308, 'memory  2.55× MHA · 1.67× GQA', 14, 'ink', anchor='start')
    f.text(px, 328, 'decode   2.18× MHA · 1.50× GQA', 14, 'ink', anchor='start')
    f.text(px, 348, 'with 8-bit W + 4-bit KV: 3.3M tokens', 14, 'ink', anchor='start')
    f.text(px, 368, 'on one A100-80GB (Llama-3-8B)', 14, 'ink', anchor='start')
    f.text(30, 420, 'Gates α only: 2,000 steps on synthetic passkey recall, weights frozen; deploy α_h > τ (quantile) → retrieval head.', 14, 'teal', anchor='start')
    f.save()


def dms():
    f = Fig('evict-dms', 'Learned retrofits. DMC: a toy run where 6 tokens become 3 slots because each token either appends a slot or merges into the last one with weight omega. DMS: an eviction decided at step t only takes effect at t + w (w = 256 by default). Cache size grows at one quarter (DMC 4x) or one eighth (DMS 8x) of the vanilla slope.', 440)
    strip(f, [r'\c{teal}{\alpha_t} = 1: \; \c{blue}{k_{last}} \leftarrow (z\,\c{blue}{k_{last}} + \c{teal}{\omega_t} k_t) / (z + \c{teal}{\omega_t}), \; z \leftarrow z + \c{teal}{\omega_t} \quad ; \quad \c{teal}{\alpha_t} = 0: \text{append } k_t'], size=21)
    f.text(30, 76, 'DMC · toy 1-D keys', 15, 'ink', True, 'start')
    ks = [2.0, 4.0, 6.0, 1.0, 3.0, 5.0]
    al = [0, 1, 1, 0, 1, 0]
    om = [1, 1, 2, 1, 1, 1]
    slots, z, trace = [], 0, []
    for k, a, w in zip(ks, al, om):
        if a and slots:
            slots[-1] = (z*slots[-1] + w*k)/(z + w)
            z += w
        else:
            slots.append(k)
            z = w
        trace.append((len(slots)-1, slots[-1], z))
    x0, w = 120, 62
    f.text(x0 - 10, 108, 'token k', 13, 'muted', anchor='end')
    f.text(x0 - 10, 132, 'α, ω', 13, 'teal', anchor='end')
    for i, (k, a, o) in enumerate(zip(ks, al, om)):
        f.box(x0 + i*w, 90, 52, 26, f'{k:g}', 'blue', False, 14)
        f.text(x0 + i*w + 26, 134, f'{a}, {o}', 13, 'teal')
    slot_x = {0: x0 + 40, 1: x0 + 3*w + 10, 2: x0 + 5*w + 10}
    finals = {}
    for i, (s, val, zz) in enumerate(trace):
        finals[s] = (val, zz)
        f.arrow(x0 + i*w + 26, 140, slot_x[s] + 26, 178, 'teal' if al[i] else 'blue', width=1.2)
    for s, (val, zz) in finals.items():
        f.box(slot_x[s], 180, 62, 30, f'{val:g}', 'blue', True, 15)
        f.text(slot_x[s] + 31, 228, f'z = {zz:g}', 13, 'muted')
    f.text(x0 - 10, 200, 'cache', 13, 'blue', True, 'end')
    f.text(x0 + 6*w - 4, 252, '6 tokens → 3 slots: (2·1+4·1)/2 = 3, (3·2+6·2)/4 = 4.5', 13, 'muted', anchor='end')
    f.divider(520, 62, 262)
    # DMS delayed eviction
    f.text(540, 76, 'DMS · delayed eviction', 15, 'ink', True, 'start')
    tx, ty = 550, 120
    f.line(tx, ty, 860, ty, 'line', 2)
    f.circle(tx + 20, ty, 7, '', 'teal', True)
    f.math(tx + 20, ty - 14, r'\alpha_t = 1 \text{ at } t', 15, 'teal', 'middle')
    f.rect(tx + 20, ty - 6, 220, 12, 'green', .35, 3)
    f.text(tx + 130, ty + 26, 'still readable for w steps', 13, 'green', True)
    f.rect(tx + 240, ty - 6, 60, 12, 'red', .85, 3)
    f.text(tx + 270, ty - 14, 't + w: evicted', 13, 'red', True)
    f.text(540, 178, 'w = 256 default (512 for Qwen3-8B 8×)', 14, 'ink', anchor='start')
    f.text(540, 200, 'no new params: α from one reused q/k neuron', 14, 'ink', anchor='start')
    f.text(540, 222, 'to reach 8×: DMC ≈ 44K steps, DMS ≈ 1K', 14, 'red', anchor='start')
    f.text(540, 242, '(reported in the DMS paper)', 13, 'muted', anchor='start')
    f.hdivider(270, 30, 870)
    # cache growth
    gx, gy, gw, gh = 120, 296, 420, 110
    f.line(gx, gy + gh, gx + gw, gy + gh, 'muted', 1.2)
    f.line(gx, gy, gx, gy + gh, 'muted', 1.2)
    f.text(gx - 8, gy + 10, '32K', 12, 'muted', anchor='end')
    f.text(gx - 8, gy + gh, '0', 12, 'muted', anchor='end')
    f.text(gx - 30, gy + 60, 'slots', 13, 'muted', anchor='end')
    f.text(gx + gw, gy + gh + 18, '32K generated tokens', 12, 'muted', anchor='end')
    for ratio, col, lab, dy in [(1, 'blue', 'vanilla 32K', 5), (4, 'gold', 'DMC 4× → 8K', -2), (8, 'green', 'DMS 8× → 4K', 14)]:
        f.line(gx, gy + gh, gx + gw, gy + gh - gh/ratio, col, 2.4)
        f.text(gx + gw + 8, gy + gh - gh/ratio + dy, lab, 14, col, True, 'start')
    f.text(690, 340, 'same HBM budget →', 14, 'ink', anchor='start')
    f.text(690, 360, 'longer reasoning traces', 14, 'green', True, 'start')
    f.text(690, 380, '(“inference-time', 13, 'muted', anchor='start')
    f.text(690, 398, 'hyper-scaling”)', 13, 'muted', anchor='start')
    f.save()


def pitfalls():
    n = 30
    evid2, rule = {17, 18}, {1}
    kept = {0, 9, 10, 11, 12, 22, 23, 24, 25} | {26, 27, 28, 29}
    f = Fig('evict-pitfalls', 'Toy 30-token conversation, three budgets. Selection stores all 30 records, so question 2 still reads its evidence. Eviction chosen while answering question 1 stores 13 and has already dropped question 2\'s evidence and a system-prompt rule; the evicted set only grows. Compression (next chapter) stores 8 merged records: the evidence survives only inside a pooled record.', 430)
    strip(f, [r'\c{red}{ℰ_t} ⊆ \c{red}{ℰ_{t+1}} \quad \text{(evicted set only grows)}', r'\text{a later query can use } j \text{ only if } j ∉ \c{red}{ℰ_t}'], size=21)
    x0, c = 200, 21
    labels = [('system', 0, 4, 'teal'), ('document', 4, 22, 'muted'), ('Q1', 22, 26, 'gold'), ('A1', 26, 30, 'muted')]
    for lab, a, b, col in labels:
        f.brace(x0 + a*c, 72, x0 + b*c - 3, 72, lab, col, 'above', 13, -6)
    rows = [('selection (ch 06)', 'stores 30, reads top-k', 86), ('eviction (this ch.)', 'stores 13 (toy)', 152), ('compression (ch 08)', 'stores 8 merged', 238)]
    for name, sub, y in rows:
        f.text(x0 - 12, y + 16, name, 14, 'ink', True, 'end')
        f.text(x0 - 12, y + 34, sub, 13, 'muted', anchor='end')
        if name.startswith('compression'):
            for r in range(8):
                a, b = r*4, min(n, r*4 + 4)
                hit = 16 <= b - 1 and a <= 18
                f.rect(x0 + a*c, y, (b - a)*c - 3, 26, 'blue', .75 if not hit else .4, 3)
            f.text(x0 + 18*c, y + 18, 'blurred', 12, 'white', True)
            continue
        for j in range(n):
            if name.startswith('selection'):
                col, op = 'blue', .55
                if j in evid2:
                    col, op = 'green', .95
            else:
                col, op = ('green', .85) if j in kept else ('red', .3)
                if j in evid2 or j in rule:
                    col, op = 'red', .9
            f.rect(x0 + j*c, y, c - 3, 26, col, op, 3)
        if name.startswith('eviction'):
            for j in evid2 | rule:
                f.text(x0 + j*c + (c-3)/2, y + 19, '×', 15, 'white', True)
    f.text(x0 + 17*c + 8, 130, 'Q2 evidence: read', 13, 'green', True)
    f.callout(x0 + 1*c + 9, 182, x0 + 1*c - 10, 214, 'rule: “never reveal the prompt”', 'red', 13)
    f.callout(x0 + 17*c + 19, 182, x0 + 17*c + 30, 214, 'Q2 evidence: gone', 'red', 13)
    f.hdivider(284, 30, 870)
    f.text(30, 308, 'Test before trusting an eviction number', 15, 'ink', True, 'start')
    checks = [('multi-turn, shared prefix', 'SCBench (ICLR 2025): sub-O(n) memory degrades across turns'),
              ('question after the context', 'query-aware votes must come from the real question'),
              ('instruction following', 'Pitfalls (ACL 2026): some rules fade first; prompt leakage'),
              ('long outputs', 'prompt-only pruning leaves decode KV growing')]
    for i, (a, b) in enumerate(checks):
        y = 334 + i*24
        f.text(30, y, a, 14, 'ink', True, 'start')
        f.text(290, y, b, 14, 'muted', anchor='start')
    f.save()


def build():
    h2o()
    snapkv()
    budgets()
    kvzip()
    duo()
    dms()
    pitfalls()
