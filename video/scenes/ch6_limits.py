"""Chapter 6 · Limits: learning is not recall (Fixed-Size States §6)."""
from common import *


def recall_error(d=8, Ns=range(1, 33), trials=120, seed=0):
    """Mean squared read error over the stored pairs (unit keys and values), additive vs exact least-squares write."""
    rng = np.random.default_rng(seed)
    add, ex = [], []
    for N in Ns:
        ea = ee = 0.0
        for _ in range(trials):
            K = rng.normal(size=(N, d))
            K /= np.linalg.norm(K, axis=1, keepdims=True)
            V = rng.normal(size=(N, d))
            V /= np.linalg.norm(V, axis=1, keepdims=True)
            Sa = V.T @ K
            Se = V.T @ np.linalg.pinv(K.T)
            ea += np.mean(np.sum((K @ Sa.T - V) ** 2, axis=1))
            ee += np.mean(np.sum((K @ Se.T - V) ** 2, axis=1))
        add.append(ea / trials)
        ex.append(ee / trials)
    return list(Ns), np.array(add), np.array(ex)


def buckets(t):
    """Fenwick-style partition of tokens 0..t-1 at token t: one bucket per set bit of t, largest first."""
    out, start = [], 0
    for bit in reversed(range(t.bit_length())):
        if t >> bit & 1:
            out.append((start, start + (1 << bit) - 1))
            start += 1 << bit
    return out


class Ch6Limits(VoiceScene):
    SCRIPT = 'ch6_limits'

    def construct(self):
        self.gap()
        self.loglinear()
        self.hybrids()
        self.journey()
        self.recap()

    def gap(self):
        with self.say('intro'):
            card = self.title_card(6, 'Limits', 'learning is not recall')
            self.play(FadeIn(card, shift=0.2 * UP), run_time=1)
            self.wait(2.4)
            self.play(FadeOut(card), run_time=0.6)
            self.header('6  Limits: learning is not recall')
            st = Mat(np.full((4, 4), 0.6), cell=0.5).move_to(LEFT * 3.6 + DOWN * 0.3)
            stl = MathTex(r'd_v\times d_k\text{ state}', font_size=32, color=PURPLE).next_to(st, UP, buff=0.3)
            stc = MathTex(r'\approx d_k\text{ exact pairs}', font_size=34, color=PURPLE).next_to(st, DOWN, buff=0.35)
            self.play(FadeIn(st), FadeIn(stl), run_time=0.6)
            self.play(Write(stc), run_time=0.8)
            recs = VGroup(*[VGroup(Square(0.3, fill_color=BLUE, fill_opacity=0.85, stroke_width=0),
                                   Square(0.3, fill_color=PURPLE, fill_opacity=0.85, stroke_width=0)).arrange(DOWN, buff=0.04) for _ in range(14)])
            recs.arrange(RIGHT, buff=0.08).move_to(RIGHT * 2.6 + DOWN * 0.3)
            rl = Text('KV cache', font_size=28, color=BLUE).next_to(recs, UP, buff=0.3)
            rc = MathTex(r'\text{all } N\text{ records}', font_size=34, color=BLUE).next_to(recs, DOWN, buff=0.35)
            self.play(FadeIn(rl), LaggedStart(*[FadeIn(r) for r in recs], lag_ratio=0.1), FadeIn(rc), run_time=2.2)
        with self.say('why'):
            self.clear_stage(run_time=0.5)
            plane = NumberPlane(x_range=[-0.3, 1.3, 1], y_range=[-0.3, 1.3, 1], x_length=3.6, y_length=3.6,
                                background_line_style={'stroke_color': LINE, 'stroke_width': 1},
                                axis_config={'stroke_color': MUTED, 'stroke_width': 1.5}).move_to(LEFT * 2.5 + DOWN * 0.4)
            a = [Arrow(plane.c2p(0, 0), plane.c2p(*v), buff=0, color=BLUE, stroke_width=6) for v in ((1, 0), (0, 1), (0.71, 0.71))]
            self.play(Create(plane), GrowArrow(a[0]), GrowArrow(a[1]), run_time=1)
            t1 = Text('2 dimensions: room for 2 orthogonal keys', font_size=26).move_to(RIGHT * 3.0 + UP * 0.6)
            self.play(FadeIn(t1), run_time=0.5)
            self.wait(1.5)
            self.play(GrowArrow(a[2]), a[2].animate.set_color(RED), run_time=0.8)
            t2 = Text('a third key must overlap', font_size=26, color=RED).next_to(t1, DOWN, buff=0.4)
            self.play(FadeIn(t2), run_time=0.5)
            self.term('cleaner writes, never a larger table', PURPLE)
        with self.say('sim', pause=1.5):
            self.clear_stage(run_time=0.5)
            self.term('simulation: 8 × 8 state, random unit keys', INK)
            Ns, add, ex = recall_error()
            ax = Axes(x_range=[0, 32, 8], y_range=[0, 2, 0.5], x_length=8.5, y_length=4.4, tips=False,
                      axis_config={'color': INK, 'stroke_width': 2}).move_to(DOWN * 0.3 + RIGHT * 0.6)
            xl = VGroup(*[Text(str(x), font_size=20, color=MUTED).next_to(ax.c2p(x, 0), DOWN, buff=0.15) for x in (0, 8, 16, 24, 32)])
            yl = VGroup(*[Text(f'{y:g}', font_size=20, color=MUTED).next_to(ax.c2p(0, y), LEFT, buff=0.15) for y in (0.5, 1, 1.5, 2)])
            xt = Text('pairs stored', font_size=24, color=MUTED).next_to(ax.x_axis, DOWN, buff=0.5).align_to(ax.x_axis, RIGHT)
            yt = Text('read error', font_size=24, color=MUTED).rotate(PI / 2).next_to(yl, LEFT, buff=0.25)
            dk = DashedLine(ax.c2p(8, 0), ax.c2p(8, 2), color=PURPLE, stroke_width=2)
            dkl = MathTex(r'd_k=8', font_size=30, color=PURPLE).next_to(ax.c2p(8, 2), UP, buff=0.1)
            self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(xt), FadeIn(yt), run_time=1)
            self.play(Create(dk), FadeIn(dkl), run_time=0.6)
            self.wait(4)
            pa = [ax.c2p(n, min(e, 2.0)) for n, e in zip(Ns, add) if e <= 2.05]
            pe = [ax.c2p(n, e) for n, e in zip(Ns, ex)]
            ca = VMobject(color=GOLD, stroke_width=5).set_points_smoothly([ax.c2p(0, 0)] + pa)
            ce = VMobject(color=GREEN, stroke_width=5).set_points_as_corners([ax.c2p(0, 0)] + pe)
            la = Text('additive writes', font_size=24, color=GOLD).next_to(pa[-1], LEFT, buff=0.2)
            le = Text('exact least-squares solve', font_size=24, color=GREEN).next_to(pe[-1], UP, buff=0.2).shift(LEFT * 1.2)
            self.play(Create(ca), run_time=3)
            self.play(FadeIn(la), run_time=0.4)
            self.wait(1.5)
            self.play(Create(ce), run_time=4)
            self.play(FadeIn(le), run_time=0.4)
            z = Text('exact up to 8', font_size=19, color=GREEN).next_to(ax.c2p(4, 0), DOWN, buff=0.15)
            self.play(FadeIn(z), run_time=0.4)
            self.sim_ax = ax
        with self.say('sim2'):
            ax = self.sim_ax
            zone = Rectangle(width=ax.c2p(32, 0)[0] - ax.c2p(8, 0)[0], height=ax.c2p(0, 2)[1] - ax.c2p(0, 0)[1],
                             fill_color=RED, fill_opacity=0.08, stroke_width=0).move_to(ax.c2p(8, 0), aligned_edge=DL)
            self.play(FadeIn(zone), run_time=0.8)
            self.term('past d_k, every fixed state starts to blur', RED)
        with self.say('numbers'):
            self.clear_stage(run_time=0.5)
            self.term('reported: 1B models, in-context recall', INK)
            base = DOWN * 2.3
            U = 4.4 / 55

            def vbar(x, v, col, name, num):
                b = Rectangle(width=1.4, height=v * U, fill_color=col, fill_opacity=0.85, stroke_width=0).move_to(base + RIGHT * x, aligned_edge=DOWN)
                n = Text(num, font_size=28, color=col, weight=MEDIUM).next_to(b, UP, buff=0.12)
                l = Text(name, font_size=22).next_to(b, DOWN, buff=0.15)
                return VGroup(b, n, l)
            axl = Line(base + LEFT * 5.6, base + RIGHT * 3.4, color=INK, stroke_width=2)
            mesa = vbar(-0.9, 41.79, PURPLE, 'Mesa, best fixed state', '41.79')
            tf = vbar(2.4, 52.27, BLUE, 'Transformer', '52.27')
            self.play(Create(axl), run_time=0.4)
            self.wait(2.5)
            self.play(GrowFromEdge(mesa[0], DOWN), FadeIn(mesa[1:]), run_time=1)
            self.wait(2)
            self.play(GrowFromEdge(tf[0], DOWN), FadeIn(tf[1:]), run_time=1)
        with self.say('numbers2', pause=1.0):
            addv = vbar(-4.4, 41.79 - 4, GOLD, 'plain additive writes', '≈ 4 lower')
            self.play(GrowFromEdge(addv[0], DOWN), FadeIn(addv[1:]), run_time=0.9)
            b1 = BraceBetweenPoints(base + RIGHT * 3.25 + UP * 41.79 * U, base + RIGHT * 3.25 + UP * 52.27 * U, RIGHT, color=BLUE)
            l1 = Text('+10.5:\nkeep the full cache', font_size=22, color=BLUE, line_spacing=0.9).next_to(b1, RIGHT, buff=0.1)
            b0 = BraceBetweenPoints(base + LEFT * 3.5 + UP * 37.79 * U, base + LEFT * 3.5 + UP * 41.79 * U, RIGHT, color=GOLD)
            l0 = Text('+≈4', font_size=22, color=GOLD).next_to(b0, RIGHT, buff=0.05)
            self.play(GrowFromCenter(b0), FadeIn(l0), run_time=0.5)
            self.play(GrowFromCenter(b1), FadeIn(l1), run_time=0.8)
        self.term(None)
        self.clear_stage()

    def loglinear(self):
        self.header('6  Log-linear attention: a middle ground')
        f = MathTex(r'\oc{t}=\sum_{\ell}\lambda_t^{(\ell)}\Sc{t}^{(\ell)}\qc{t},\qquad \Sc{t}^{(\ell)}=\sum_{s\in\mathcal B_t^{(\ell)}}\vc{s}\kct{s}',
                    font_size=42).to_edge(UP, buff=0.85)
        n, w = 17, 0.62
        toks = VGroup(*[VGroup(Square(0.5, fill_color=SOFT, fill_opacity=1, stroke_color=LINE, stroke_width=1.5),
                               Text(str(i), font_size=18, color=MUTED)) for i in range(n)])
        for i, tk in enumerate(toks):
            tk[1].move_to(tk[0])
            tk.move_to(LEFT * 5.0 + RIGHT * i * w + DOWN * 0.6)
        state = {'grp': VGroup()}

        def draw(t):
            g = VGroup()
            for a, b in buckets(t):
                r = SurroundingRectangle(VGroup(*[toks[i] for i in range(a, b + 1)]), color=PURPLE, buff=0.05, stroke_width=3)
                s = Square(0.34, fill_color=PURPLE, fill_opacity=0.85, stroke_width=0).next_to(r, UP, buff=0.12)
                lab = Text(f'{a}–{b}' if b > a else f'{a}', font_size=18, color=PURPLE).next_to(r, DOWN, buff=0.12)
                g.add(VGroup(r, s, lab))
            me = SurroundingRectangle(toks[t], color=GOLD, buff=0.05, stroke_width=3)
            ms = Square(0.34, fill_color=GOLD, fill_opacity=0.85, stroke_width=0).next_to(me, UP, buff=0.12)
            g.add(VGroup(me, ms, Text('itself', font_size=18, color=GOLD).next_to(me, DOWN, buff=0.12)))
            return g

        def go(t, rt=0.6):
            new = draw(t)
            cnt = Text(f'token {t}: {len(new)} states', font_size=28, weight=MEDIUM).move_to(DOWN * 2.6 + LEFT * 0.8)
            bn = Text(f'{t} = {t:b} in binary', font_size=24, color=MUTED).next_to(cnt, RIGHT, buff=0.6)
            anims = [FadeIn(toks[t])] if t < n else []
            self.play(FadeOut(state['grp']), FadeIn(new), FadeIn(cnt), FadeIn(bn), *anims, run_time=rt)
            state['grp'] = VGroup(new, cnt, bn)
            return new

        with self.say('loglin', pause=0.8):
            self.term('log-linear attention (2025)', PURPLE)
            self.wait(2)
            self.play(Write(f), run_time=2.5)
            self.wait(2)
            fl = Text('one state per power-of-two bucket, like a Fenwick tree', font_size=26, color=PURPLE).move_to(UP * 1.6)
            self.play(FadeIn(fl), run_time=0.6)
        with self.say('buckets', pause=1.0):
            self.play(FadeOut(fl), FadeIn(VGroup(*toks[:1])), run_time=0.4)
            for t in range(1, 14):
                go(t, rt=0.45)
            self.term('recent tokens: small buckets · far tokens: big ones', PURPLE)
        with self.say('merge'):
            self.wait(4.5)
            for t in (14, 15, 16):
                go(t, rt=0.8)
                self.wait(1.2)
            self.term('equal buckets merge, like carrying in binary', PURPLE)
        with self.say('scale', pause=1.0):
            self.clear_stage(run_time=0.5)
            self.term('after about a million tokens (2²⁰)', INK)
            items = [('KV cache', 1048576, '1,048,576 records', BLUE), ('log-linear attention', 22, 'at most 22 states', TEAL),
                     ('linear attention', 1, '1 state', PURPLE)]
            for i, (name, v, lab, c) in enumerate(items):
                y = 1.2 - i * 1.2
                t = Text(name, font_size=28, weight=MEDIUM).move_to([-2.7, y, 0], aligned_edge=RIGHT)
                b = Rectangle(width=max(0.06, 0.26 * np.log2(v)), height=0.55, fill_color=c, fill_opacity=0.85, stroke_width=0).move_to([-2.3, y, 0], aligned_edge=LEFT)
                l = Text(lab, font_size=26, color=c, weight=MEDIUM).next_to(b, RIGHT, buff=0.2)
                self.play(FadeIn(t), GrowFromEdge(b, LEFT), FadeIn(l), run_time=0.9)
                self.wait(2.0)
            ls = Text('log scale', font_size=20, color=MUTED).move_to([0.3, 2.0, 0])
            self.play(FadeIn(ls), run_time=0.3)
        with self.say('growth'):
            gr = Text('double the context → just one more state', font_size=28, color=TEAL).move_to(DOWN * 2.3)
            self.play(FadeIn(gr, shift=0.1 * UP), run_time=0.6)
        with self.say('scale2'):
            r = Text('reported perplexity gains: small · the gap to Transformers remains', font_size=26, color=RED).move_to(DOWN * 2.4)
            self.play(FadeOut(gr), FadeIn(r), run_time=0.6)
        self.term(None)
        self.clear_stage()

    def hybrids(self):
        self.header('6  Hybrids')
        n = 12
        layers = VGroup()
        for i in range(n):
            full = i % 4 == 3
            c = BLUE if full else PURPLE
            r = RoundedRectangle(corner_radius=0.08, width=4.2, height=0.36, fill_color=c, fill_opacity=0.8 if full else 0.3,
                                 stroke_color=c, stroke_width=2)
            layers.add(r)
        layers.arrange(UP, buff=0.08).move_to(LEFT * 3.0 + DOWN * 0.3)
        with self.say('hybrid'):
            self.play(LaggedStart(*[FadeIn(l, shift=0.1 * UP) for l in layers], lag_ratio=0.08), run_time=2)
            k1 = VGroup(RoundedRectangle(corner_radius=0.06, width=0.5, height=0.3, fill_color=PURPLE, fill_opacity=0.3, stroke_color=PURPLE),
                        Text('fixed-state layer', font_size=24, color=PURPLE)).arrange(RIGHT, buff=0.2)
            k2 = VGroup(RoundedRectangle(corner_radius=0.06, width=0.5, height=0.3, fill_color=BLUE, fill_opacity=0.8, stroke_color=BLUE),
                        Text('full-attention layer', font_size=24, color=BLUE)).arrange(RIGHT, buff=0.2)
            leg = VGroup(k1, k2).arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to(RIGHT * 3.0 + UP * 1.6)
            self.play(FadeIn(leg), run_time=0.6)
            il = Text('(layer mix shown for illustration)', font_size=20, color=MUTED).next_to(layers, DOWN, buff=0.2)
            self.play(FadeIn(il), run_time=0.4)
            self.term('the hybrid design: Part 5 of the essay', BLUE)
        with self.say('hybrid2'):
            full = VGroup(*[layers[i] for i in range(n) if i % 4 == 3])
            t1 = Text('exact recall from far back:\na number to repeat, a passage to copy', font_size=24, color=BLUE, line_spacing=0.9).move_to(RIGHT * 3.0 + DOWN * 0.3)
            arrs = VGroup(*[arrow(t1.get_left(), l.get_right(), color=BLUE) for l in full])
            self.play(FadeIn(t1), LaggedStart(*[GrowArrow(a) for a in arrs], lag_ratio=0.2), run_time=1.2)
            self.wait(2.5)
            t2 = Text('the many cheap state layers carry the rest', font_size=24, color=PURPLE).move_to(RIGHT * 3.0 + DOWN * 2.0)
            self.play(FadeIn(t2), run_time=0.6)
        self.term(None)
        self.clear_stage()

    def journey(self):
        self.header('The whole series')
        rows = [('1  Linear attention', r'\Sc{t}=\Sc{t-1}+\vc{t}\kct{t}'),
                ('2  Forgetting', r'\Sc{t}=\Ct{\alpha_t}\,\Sc{t-1}+\vc{t}\kct{t}'),
                ('3  Correcting', r'\Sc{t}=\Sc{t-1}\big(I-\beta_t\kc{t}\kct{t}\big)+\beta_t\vc{t}\kct{t}'),
                ('4  One axis', r'\Sc{t}=\Sc{t-1}\Ct{A_t}+\Cv{W_t}'),
                ('5  A model trained while reading', r'\Sc{t}=\alpha_t\Sc{t-1}-\beta_t\nabla_{\Cv{S}}\ell_t')]
        keys = ('journey', 'j2', 'j3', 'j4', 'j5')
        grp = VGroup()
        for i, ((name, tex), key) in enumerate(zip(rows, keys)):
            y = 2.2 - i * 1.05
            t = Text(name, font_size=28, weight=MEDIUM).move_to([-6.5, y, 0], aligned_edge=LEFT)
            m = MathTex(tex, font_size=38).move_to([2.2, y, 0])
            with self.say(key, pause=0.8 if key == 'j4' else 0):
                self.play(FadeIn(t, shift=0.2 * RIGHT), Write(m), run_time=1.4)
            grp.add(VGroup(t, m))
        self.journey_grp = grp

    def recap(self):
        self.play(FadeOut(self.journey_grp), run_time=0.6)
        self.header('Chapter 6 · Limits')
        card, lo = self.recap_card(r'\text{fixed state: }\approx d_k\text{ exact pairs}\qquad\text{KV cache: all }N',
                                   'A fixed state can be written better and better, but never made bigger.',
                                   'exact recall still needs real records; hybrids mix the two.')
        with self.say('recap', pause=0.6):
            self.play(FadeIn(card, shift=0.2 * UP), run_time=1)
        with self.say('leftover'):
            self.play(FadeIn(lo, shift=0.1 * UP), run_time=0.8)
        self.wait(1.5)
