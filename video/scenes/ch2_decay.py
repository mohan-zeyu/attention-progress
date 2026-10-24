"""Chapter 2 · Forgetting: decay the state (Fixed-Size States §2)."""
from common import *


def mask_mat(vals, cell=0.5, numbers=True, color=PURPLE):
    """Lower-triangular weight mask; the masked (future) half is greyed."""
    n = len(vals)
    m = Mat(vals, cell=cell, color=color, vmax=1, numbers=numbers, fmt='{:.2f}', font_size=int(cell * 30))
    for i in range(n):
        for j in range(i + 1, n):
            m.cell(i, j).set_fill(LINE, 0.3)
            if numbers:
                m.nums[i * n + j].set_opacity(0)
    return m


def decay_mask(alphas):
    """D[t, j] = prod_{s=j+1..t} alpha_s."""
    n = len(alphas)
    D = np.zeros((n, n))
    for t in range(n):
        for j in range(t + 1):
            D[t, j] = np.prod(alphas[j + 1:t + 1])
    return D


class Ch2Decay(VoiceScene):
    SCRIPT = 'ch2_decay'

    def construct(self):
        self.problem()
        self.rule_and_mask()
        self.retnet()
        self.selective()
        self.gla()
        self.mamba3()
        self.recap()

    def problem(self):
        rng = np.random.default_rng(5)
        with self.say('intro'):
            card = self.title_card(2, 'Forgetting', 'decay the state')
            self.play(FadeIn(card, shift=0.2 * UP), run_time=1)
            self.wait(2.4)
            self.play(FadeOut(card), run_time=0.6)
            self.header('2.1  One rule, one mask')
            S = Mat(rng.uniform(0.1, 0.5, (5, 5)), cell=0.6, vmax=4).move_to(RIGHT * 2.5 + DOWN * 0.3)
            lab = Text('state after many writes', font_size=26, color=MUTED).next_to(S, DOWN, buff=0.35)
            self.play(FadeIn(S), FadeIn(lab), run_time=0.6)
            vals = S.vals.copy()
            counter = Integer(0, color=INK, font_size=40).move_to(LEFT * 3.5 + UP * 0.4)
            ct = Text('writes', font_size=26, color=MUTED).next_to(counter, DOWN, buff=0.2)
            self.play(FadeIn(counter), FadeIn(ct), run_time=0.3)
            for n in range(8):
                v, k = rng.uniform(0, 1.6, 5), rng.uniform(0, 1, 5)
                vals = vals + np.outer(v, k / np.linalg.norm(k))
                self.play(S.set_values(np.minimum(vals, 4)), counter.animate.set_value(n + 1), run_time=0.45)
            self.term('crowded: every read is mostly crosstalk', RED)
            self.wait(1.5)
            self.play(S.set_values(np.minimum(vals, 4) * 0.55), run_time=1.5)
            fade = Text('let it fade before each write', font_size=28, color=PURPLE).next_to(counter, DOWN, buff=1.2)
            self.play(FadeIn(fade), run_time=0.5)
        self.term(None)
        self.clear_stage()

    def rule_and_mask(self):
        f = MathTex(r'\Sc{t}=\Ct{\alpha_t}\,\Sc{t-1}+\vc{t}\kct{t},\qquad 0<\Ct{\alpha_t}\le1', font_size=46).to_edge(UP, buff=0.9)
        rng = np.random.default_rng(2)
        S0 = rng.uniform(0.8, 3, (4, 4))
        with self.say('rule', pause=0.8):
            self.play(Write(f), run_time=1.4)
            S = Mat(S0, cell=0.6, vmax=3.5).move_to(LEFT * 2 + DOWN * 0.6)
            Sl = tlabel(r'\Sc{t-1}', S, DOWN)
            self.play(FadeIn(S), FadeIn(Sl), run_time=0.5)
            a = MathTex(r'\times\,\Ct{\alpha_t}=0.8', font_size=38).next_to(S, LEFT, buff=0.4)
            self.play(FadeIn(a), S.set_values(S0 * 0.8), run_time=1.2)
            v, k = np.array([1.5, 0.5, 2, 1]), np.array([0.5, 0.5, 0.5, 0.5])
            P = Mat(np.outer(v, k), cell=0.6, vmax=3.5).move_to(RIGHT * 3 + DOWN * 0.6)
            Pl = tlabel(r'\vc{t}\kct{t}', P, DOWN)
            plus = MathTex('+', font_size=56).move_to((S.get_right() + P.get_left()) / 2)
            self.play(FadeIn(P), FadeIn(Pl), FadeIn(plus), run_time=0.6)
            self.play(P.animate.move_to(S).set_opacity(0.3), FadeOut(plus), FadeOut(Pl), run_time=0.8)
            self.play(FadeOut(P), S.set_values(S0 * 0.8 + np.outer(v, k)), Transform(Sl, tlabel(r'\Sc{t}', S, DOWN)), run_time=0.6)
        with self.say('unroll', pause=0.8):
            self.play(FadeOut(VGroup(S, Sl, a)), run_time=0.5)
            slots = VGroup(*[Square(0.9, stroke_color=LINE, stroke_width=2) for _ in range(6)]).arrange(RIGHT, buff=0.35)
            slots.move_to(UP * 0.7)
            names = ['j', 'j+1', 'j+2', r'\cdots', 't-1', 't']
            sl = VGroup(*[MathTex(n, font_size=30, color=MUTED).next_to(s, DOWN, buff=0.15) for n, s in zip(names, slots)])
            self.play(FadeIn(slots), FadeIn(sl), run_time=0.6)
            blk = Square(0.7, fill_color=PURPLE, fill_opacity=0.9, stroke_width=0).move_to(slots[0])
            wl = MathTex(r'\vc{j}\kct{j}', font_size=30).next_to(slots[0], UP, buff=0.15)
            self.play(FadeIn(blk, scale=0.5), FadeIn(wl), run_time=0.6)
            amul = VGroup()
            op = 0.9
            for i in range(1, 6):
                op *= 0.8
                lab = MathTex(r'\times\Ct{\alpha_{' + ['j+1', 'j+2', r'\cdots', 't-1', 't'][i - 1] + '}}', font_size=28)
                lab.next_to(slots[i], UP, buff=0.15)
                amul.add(lab)
                self.play(blk.animate.move_to(slots[i]).set_fill(opacity=op), FadeIn(lab), run_time=0.7)
            g = MathTex(r'\oc{t}=\sum_{j\le t}\Big(\prod_{s=j+1}^{t}\Ct{\alpha_s}\Big)(\kct{j}\qc{t})\,\vc{j}',
                        r'\quad\Longleftrightarrow\quad', r'O=(\Cq{Q}\Ck{K}^{\top}\odot \Ct{D})\Cv{V}', font_size=40)
            g.move_to(DOWN * 1.5)
            if g.width > 13:
                g.scale_to_fit_width(13)
            self.play(Write(g[0]), run_time=2)
            self.wait(1.5)
            self.play(Write(g[1:]), run_time=1.5)
            dl = MathTex(r'\Ct{D_{tj}}=\prod_{s=j+1}^{t}\Ct{\alpha_s}', font_size=38).next_to(g, DOWN, buff=0.35)
            self.play(Write(dl), run_time=1)
        with self.say('mask'):
            self.play(FadeOut(VGroup(slots, sl, blk, wl, amul, f)), VGroup(g, dl).animate.scale(0.75).to_edge(UP, buff=0.75),
                      run_time=0.8)
            n = 8
            ones = np.tril(np.ones((n, n)))
            M1 = mask_mat(ones, cell=0.48).move_to(LEFT * 2.6 + DOWN * 0.95)
            ml = Text('linear attention: every α = 1', font_size=26).next_to(M1, RIGHT, buff=0.6).shift(UP * 1.2)
            self.play(FadeIn(M1, lag_ratio=0.01), FadeIn(ml), run_time=1.5)
            self.wait(4)
            D = decay_mask(np.full(n, 0.8))
            M2 = mask_mat(D, cell=0.48).move_to(M1)
            m2l = Text('with a decay, α = 0.8', font_size=26, color=TEAL).next_to(ml, DOWN, buff=0.35, aligned_edge=LEFT)
            self.play(Transform(M1, M2), FadeIn(m2l), run_time=2)
            self.wait(4)
            arr = Arrow(M1.cell(7, 7).get_bottom() + DOWN * 0.22, M1.cell(7, 0).get_bottom() + DOWN * 0.22, color=RED, buff=0,
                        stroke_width=4, max_tip_length_to_length_ratio=0.08)
            dt = Text('farther back → smaller weight', font_size=26, color=RED).next_to(m2l, DOWN, buff=0.35, aligned_edge=LEFT)
            self.play(GrowArrow(arr), FadeIn(dt), run_time=1)
            self.term('distance is built in: no other position encoding', TEAL)
        self.term(None)
        self.clear_stage()

    def retnet(self):
        self.header('2.2  A fixed decay per head')
        f = MathTex(r'\text{RetNet: }\;\Ct{D_{tj}}=\gamma^{\,t-j}', font_size=44).to_edge(UP, buff=0.85)
        ax = Axes(x_range=[0, 300, 50], y_range=[0, 1, 0.25], x_length=8, y_length=4.2, tips=False,
                  axis_config={'color': INK, 'stroke_width': 2}).move_to(DOWN * 0.5 + RIGHT * 0.5)
        xl = VGroup(*[Text(str(x), font_size=20, color=MUTED).next_to(ax.c2p(x, 0), DOWN, buff=0.15) for x in range(0, 301, 50)])
        yl = VGroup(*[Text(f'{y:g}', font_size=20, color=MUTED).next_to(ax.c2p(0, y), LEFT, buff=0.15) for y in (0.25, 0.5, 0.75, 1)])
        xt = Text('tokens back,  t − j', font_size=24, color=MUTED).next_to(ax.x_axis, DOWN, buff=0.5).align_to(ax.x_axis, RIGHT)
        with self.say('retnet', pause=0.8):
            self.play(Write(f), run_time=1.2)
            self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(xt), run_time=1)
            curve = ax.plot(lambda x: 0.99 ** x, x_range=[0, 300], color=TEAL, stroke_width=5)
            cl = MathTex(r'\gamma=0.99', font_size=34, color=TEAL).next_to(ax.c2p(220, 0.11), UP, buff=0.3)
            self.play(Create(curve), FadeIn(cl), run_time=2.5)
            self.wait(4)
            hz = MathTex(r'\text{horizon}\approx\frac{1}{1-\gamma}=100', font_size=36).move_to(ax.c2p(200, 0.75))
            self.play(Write(hz), run_time=1.2)
            self.wait(2.5)
            p = ax.c2p(100, 0.99 ** 100)
            v = DashedLine(ax.c2p(100, 0), p, color=RED, stroke_width=3)
            h = DashedLine(ax.c2p(0, 0.99 ** 100), p, color=RED, stroke_width=3)
            dot = Dot(p, color=RED, radius=0.08)
            pl = MathTex(r'0.99^{100}\approx0.37', font_size=36, color=RED).next_to(p, UR, buff=0.15)
            self.play(Create(v), Create(h), FadeIn(dot, scale=2), run_time=1)
            self.play(Write(pl), run_time=1)
        with self.say('heads'):
            self.play(FadeOut(VGroup(hz, v, h, dot, pl, cl, xt)), FadeOut(curve), run_time=0.5)
            ax2 = Axes(x_range=[0, 600, 100], y_range=[0, 1, 0.25], x_length=8, y_length=4.2, tips=False,
                       axis_config={'color': INK, 'stroke_width': 2}).move_to(ax)
            xl2 = VGroup(*[Text(str(x), font_size=20, color=MUTED).next_to(ax2.c2p(x, 0), DOWN, buff=0.15) for x in range(0, 601, 100)])
            self.play(ReplacementTransform(ax, ax2), ReplacementTransform(xl, xl2), run_time=0.8)
            horizons = [32, 64, 128, 256, 512]
            cols = [interpolate_color(ManimColor(TEAL), ManimColor(PURPLE), i / 4) for i in range(5)]
            curves, labs = VGroup(), VGroup()
            for hzn, c in zip(horizons, cols):
                g = 1 - 1 / hzn
                curves.add(ax2.plot(lambda x, g=g: g ** x, x_range=[0, 600], color=c, stroke_width=4))
                labs.add(Text(f'{hzn}', font_size=22, color=c).next_to(ax2.c2p(hzn, np.exp(-1)), LEFT if hzn == 32 else UR, buff=0.05))
            self.play(LaggedStart(*[Create(c) for c in curves], lag_ratio=0.3), run_time=3)
            ht = MathTex(r'1-\gamma\in\left[\tfrac{1}{512},\,\tfrac{1}{32}\right]', font_size=36).move_to(ax2.c2p(430, 0.95))
            self.play(Write(ht), LaggedStart(*[FadeIn(l) for l in labs], lag_ratio=0.2), run_time=2)
            hl = Text('horizons ≈ 32 … 512 tokens, one per head', font_size=26, color=MUTED).move_to(ax2.c2p(430, 0.8))
            self.play(FadeIn(hl), run_time=0.6)
            self.wait(3.5)
            self.term('Lightning attention (2025): fixed decay + fast kernel', TEAL)
        with self.say('blind'):
            self.clear_stage(run_time=0.5)
            self.term('decay ignores content', RED)
            t = ValueTracker(0)
            items = [('order 7314 ships Tuesday', BLUE, -1.0), ('hi, how are you?', MUTED, 1.0)]
            grp = VGroup()
            for name, c, y in items:
                lab = Text(name, font_size=28, color=c).move_to([-3.6, y, 0])
                b = always_redraw(lambda c=c, y=y: Rectangle(width=max(0.01, 5 * 0.97 ** t.get_value()), height=0.5,
                                                             fill_color=c, fill_opacity=0.8, stroke_width=0)
                                  .move_to([-0.6, y, 0], aligned_edge=LEFT))
                grp.add(lab, b)
            self.add(grp)
            self.play(t.animate.set_value(40), run_time=4.5, rate_func=linear)
        self.term(None)
        self.clear_stage()

    def selective(self):
        self.header('2.3  A decay chosen by the input')
        alphas = np.array([0.95, 0.9, 0.97, 0.15, 0.92, 0.96, 0.9, 0.94])
        f = MathTex(r'\text{Mamba-2: }\;\Ct{a_t}=e^{\Delta_t A_h},\qquad A_h<0', font_size=44).to_edge(UP, buff=0.85)
        with self.say('selective', pause=0.6):
            self.play(Write(f), run_time=1.4)
            toks = VGroup(*[Square(0.5, fill_color=SOFT, fill_opacity=1, stroke_color=LINE) for _ in alphas])
            toks.arrange(RIGHT, buff=0.15).move_to(LEFT * 3.6 + DOWN * 2.3)
            bars = VGroup(*[Rectangle(width=0.4, height=2.2 * a, fill_color=TEAL, fill_opacity=0.85, stroke_width=0)
                            .next_to(t, UP, buff=0.1) for t, a in zip(toks, alphas)])
            for b, t in zip(bars, toks):
                b.align_to(t.get_top() + UP * 0.1, DOWN)
            bl = MathTex(r'\Ct{a_t}\text{ per token}', font_size=30).next_to(bars, UP, buff=0.3)
            self.wait(2.5)
            self.play(FadeIn(toks), LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.15), FadeIn(bl), run_time=2)
            self.wait(3)
            D = decay_mask(alphas)
            M = mask_mat(D, cell=0.5).move_to(RIGHT * 3.3 + DOWN * 0.7)
            Ml = Text('the mask is no longer smooth', font_size=26, color=TEAL).next_to(M, UP, buff=0.25)
            self.play(FadeIn(M, lag_ratio=0.01), FadeIn(Ml), run_time=1.5)
            self.wait(2)
            hi = SurroundingRectangle(bars[3], color=RED, buff=0.06)
            cut = SurroundingRectangle(VGroup(*[M.cell(i, j) for i in range(3, 8) for j in range(0, 3)]), color=RED, buff=0.02)
            self.play(Create(hi), Create(cut), run_time=1)
            self.term('each token decides how much to forget', TEAL)
        with self.say('ssd'):
            self.clear_stage(run_time=0.5)
            self.term('state-space duality', TEAL)
            rows = [('SSM input  x', PURPLE, 'value  v'), ('B', BLUE, 'key  k'), ('C', GOLD, 'query  q')]
            grp = VGroup()
            for i, (a, c, b) in enumerate(rows):
                la = Text(a, font_size=32, color=c, weight=MEDIUM).move_to([-2.8, 1.2 - i * 1.1, 0])
                lb = Text(b, font_size=32, color=c, weight=MEDIUM).move_to([2.8, 1.2 - i * 1.1, 0])
                ar = DoubleArrow(la.get_right() + RIGHT * 0.3, lb.get_left() + LEFT * 0.3, color=c, buff=0, stroke_width=4,
                                 tip_length=0.2)
                grp.add(VGroup(la, ar, lb))
            hd = VGroup(Text('state-space model', font_size=26, color=MUTED).move_to([-2.8, 2.3, 0]),
                        Text('masked linear attention', font_size=26, color=MUTED).move_to([2.8, 2.3, 0]))
            self.play(FadeIn(hd), run_time=0.5)
            self.wait(3)
            for g in grp:
                self.play(FadeIn(g[0]), GrowFromCenter(g[1]), FadeIn(g[2]), run_time=1.0)
                self.wait(0.6)
            same = MathTex(r'\Sc{t}=\Ct{a_t}\,\Sc{t-1}+\Cv{x_t}\Ck{B_t}^{\top},\qquad \Cg{y_t}=\Sc{t}\,\Cq{C_t}', font_size=40)
            same.move_to(DOWN * 2.2)
            self.play(Write(same), run_time=1.5)
        with self.say('ssdcost'):
            self.clear_stage(run_time=0.5)
            self.term(None)
            fast = Text('one scalar per head → all matrix products', font_size=30).move_to(UP * 2.2)
            rep = Text('reported 2–8× faster than the scan of the first Mamba', font_size=28, color=GREEN).next_to(fast, DOWN, buff=0.3)
            self.play(FadeIn(fast), run_time=0.6)
            self.play(FadeIn(rep), run_time=0.6)
            self.wait(3)
            d = ValueTracker(0.3)
            A = -1.2
            knob = MathTex(r'\Delta_t', font_size=40).move_to(LEFT * 4.6 + DOWN * 1.0)
            track = Line(LEFT * 3.8, LEFT * 0.8, color=LINE, stroke_width=6).move_to(LEFT * 2.3 + DOWN * 1.0)
            dot = always_redraw(lambda: Dot(track.point_from_proportion(d.get_value() / 2.0), radius=0.13, color=INK))
            base = DOWN * 2.3

            def bar_at(x, h, c):
                return Rectangle(width=0.8, height=max(0.01, h), fill_color=c, fill_opacity=0.85, stroke_width=0) \
                    .move_to(base + RIGHT * x, aligned_edge=DOWN)
            wb = always_redraw(lambda: bar_at(1.6, 1.6 * d.get_value(), PURPLE))
            kb = always_redraw(lambda: bar_at(3.6, 3.0 * np.exp(d.get_value() * A), TEAL))
            wl = Text('write', font_size=26, color=PURPLE).next_to(base + RIGHT * 1.6, DOWN, buff=0.15)
            kl = MathTex(r'\text{keep } e^{\Delta_t A_h}', font_size=30, color=TEAL).next_to(base + RIGHT * 3.6, DOWN, buff=0.15)
            self.add(track, dot, wb, kb)
            self.play(FadeIn(knob), FadeIn(wl), FadeIn(kl), run_time=0.5)
            self.term('Δ_t sets both: a strong write also forgets more', RED)
            self.play(d.animate.set_value(1.9), run_time=3.5)
            self.wait(0.5)
        self.term(None)
        self.clear_stage()

    def gla(self):
        self.header('2.3  Per-channel decay: GLA')
        f = MathTex(r'\Sc{t}=\Sc{t-1}\Ct{\Diag(\alpha_t)}+\vc{t}\kct{t}', font_size=46).to_edge(UP, buff=0.85)
        rng = np.random.default_rng(4)
        S0 = rng.uniform(1.2, 3, (4, 4))
        al = np.array([0.99, 0.6, 0.95, 0.85])
        S = Mat(S0, cell=0.75, vmax=3.2).move_to(DOWN * 0.55)
        A = Mat(al[None, :], cell=0.75, color=TEAL, vmax=1, numbers=True, fmt='{:.2f}', font_size=22).next_to(S, UP, buff=0.3)
        Al = MathTex(r'\Ct{\alpha_t}', font_size=36).next_to(A, LEFT, buff=0.3)
        chl = VGroup(*[Text(f'ch {j + 1}', font_size=20, color=MUTED).next_to(S.cell(3, j), DOWN, buff=0.15) for j in range(4)])
        kt = Text('key channels', font_size=24, color=MUTED).next_to(chl, DOWN, buff=0.15)
        with self.say('gla', pause=0.8):
            self.play(Write(f), run_time=1.4)
            self.play(FadeIn(S), FadeIn(chl), FadeIn(kt), run_time=0.8)
            self.wait(2)
            self.play(FadeIn(A), FadeIn(Al), run_time=0.8)
            self.wait(1.5)
            for j in range(4):
                col = VGroup(*[S.cell(i, j) for i in range(4)])
                hi = SurroundingRectangle(VGroup(col, A.cell(0, j)), color=TEAL, buff=0.03)
                self.play(FadeIn(hi), *[S.cell(i, j).animate.set_fill(PURPLE, min(1, S0[i, j] * al[j] / 3.2) * 0.9)
                                        for i in range(4)], run_time=0.6)
                self.play(FadeOut(hi), run_time=0.2)
        S1 = S0 * al
        with self.say('glatopic'):
            self.term('topic switch: clear one channel, keep the rest', TEAL)
            new = Mat([[0.99, 0.05, 0.95, 0.85]], cell=0.75, color=TEAL, vmax=1, numbers=True, fmt='{:.2f}', font_size=22).move_to(A)
            self.play(Transform(A, new), run_time=0.6)
            col = VGroup(*[S.cell(i, 1) for i in range(4)])
            self.play(*[S.cell(i, 1).animate.set_fill(RED, 0.5) for i in range(4)], run_time=0.5)
            self.play(*[S.cell(i, 1).animate.set_fill(PURPLE, S1[i, 1] * 0.05 / 3.2 * 0.9) for i in range(4)], run_time=0.8)
        with self.say('glapower', pause=0.8):
            self.clear_stage(keep=[f], run_time=0.5)
            g = MathTex(r'\Ct{\alpha_t}=\sigma(W_\alpha x_t+b)^{1/16}\in(0,1)^{d_k}', font_size=42).next_to(f, DOWN, buff=0.4)
            self.play(Write(g), run_time=1.5)
            l1 = NumberLine(x_range=[0, 1, 0.25], length=8, color=INK, include_numbers=False).move_to(DOWN * 0.4 + RIGHT * 0.8)
            l2 = NumberLine(x_range=[0.7, 1, 0.05], length=8, color=TEAL, include_numbers=False).move_to(DOWN * 2.4 + RIGHT * 0.8)
            t1 = Text('sigmoid output', font_size=24, color=MUTED).next_to(l1, LEFT, buff=0.3)
            t2 = Text('decay α', font_size=24, color=TEAL).next_to(l2, LEFT, buff=0.3)
            n1 = VGroup(*[Text(f'{x:g}', font_size=20, color=MUTED).next_to(l1.n2p(x), DOWN, buff=0.15) for x in (0, 1)])
            n2 = VGroup(Text('0.7', font_size=20, color=MUTED).next_to(l2.n2p(0.7), DOWN, buff=0.15),
                        Text('1', font_size=20, color=MUTED).next_to(l2.n2p(1.0), RIGHT, buff=0.2))
            self.play(Create(l1), Create(l2), FadeIn(t1), FadeIn(t2), FadeIn(n1), FadeIn(n2), run_time=1)
            pairs = [(0.01, 0.75), (0.1, 0.866), (0.5, 0.958), (0.9, 0.993)]
            for s, a in pairs:
                assert abs(s ** (1 / 16) - a) < 0.001
                d1 = Dot(l1.n2p(s), color=INK, radius=0.07)
                d2 = Dot(l2.n2p(a), color=TEAL, radius=0.08)
                s1 = Text(f'{s:g}', font_size=22).next_to(d1, UP, buff=0.12)
                s2 = Text(f'{a:g}', font_size=22, color=TEAL).next_to(d2, DOWN, buff=0.15)
                ar = Line(d1.get_center(), d2.get_center(), buff=0.1, color=MUTED, stroke_width=2.5)
                self.play(FadeIn(d1), FadeIn(s1), run_time=0.5)
                self.play(Create(ar), FadeIn(d2), FadeIn(s2), run_time=0.9)
            self.term('decays stay close to 1: forget slowly by default', TEAL)
        with self.say('gllog'):
            self.clear_stage(keep=[f], run_time=0.5)
            lg = MathTex(r'\prod_{s}\Ct{\alpha_s}', r'\;=\;', r'\exp\Big(\sum_{s}\log\Ct{\alpha_s}\Big)', font_size=50)
            self.play(Write(lg[0]), run_time=0.8)
            self.play(Write(lg[1:]), run_time=1.2)
            self.term('chunks: products become sums in log space', TEAL)
        self.term(None)
        self.clear_stage()

    def mamba3(self):
        self.header('2.4  Mamba-3')
        with self.say('m3'):
            self.term('1  trapezoid write', PURPLE)
            ax = Axes(x_range=[0, 2, 1], y_range=[0, 0.6, 0.2], x_length=4, y_length=3.2, tips=False,
                      axis_config={'color': INK, 'stroke_width': 2})
            left = ax.copy().move_to(LEFT * 3.4 + DOWN * 0.5)
            right = ax.copy().move_to(RIGHT * 3.0 + DOWN * 0.5)
            note = MathTex(r'\text{toy: step }\Delta=0.5,\ \text{decay }0.8', font_size=36).to_edge(UP, buff=0.9)
            self.play(FadeIn(note), run_time=0.8)
            self.wait(4)

            def panel_of(a, title, weights, col):
                t = Text(title, font_size=28, color=col, weight=MEDIUM).next_to(a, UP, buff=0.35)
                ticks = VGroup(MathTex('x_{t-1}', font_size=30).next_to(a.c2p(0.5, 0), DOWN, buff=0.2),
                               MathTex('x_t', font_size=30).next_to(a.c2p(1.5, 0), DOWN, buff=0.2))
                bars, labs = VGroup(), VGroup()
                for x, w in zip((0.5, 1.5), weights):
                    if w == 0:
                        continue
                    b = Rectangle(width=0.8, height=a.c2p(0, w)[1] - a.c2p(0, 0)[1], fill_color=col, fill_opacity=0.85,
                                  stroke_width=0).move_to(a.c2p(x, 0), aligned_edge=DOWN)
                    bars.add(b)
                    labs.add(Text(f'{w:g}', font_size=26, color=col).next_to(b, UP, buff=0.1))
                return t, ticks, bars, labs

            for a, title, w, col in ((left, 'Mamba-2', (0, 0.5), BLUE), (right, 'trapezoid (Mamba-3)', (0.2, 0.25), PURPLE)):
                t, ticks, bars, labs = panel_of(a, title, w, col)
                self.play(Create(a.x_axis), FadeIn(t), FadeIn(ticks), run_time=0.8)
                self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in bars], lag_ratio=0.3), FadeIn(labs), run_time=1.2)
                self.wait(2.5)
        with self.say('m3rot'):
            self.clear_stage(run_time=0.5)
            self.term('2  rotations: a data-dependent position encoding', PURPLE)
            plane = NumberPlane(x_range=[-1.2, 1.2, 1], y_range=[-1.2, 1.2, 1], x_length=4.4, y_length=4.4,
                                background_line_style={'stroke_color': LINE, 'stroke_width': 1},
                                axis_config={'stroke_color': MUTED, 'stroke_width': 1.5})
            p1 = plane.copy().move_to(LEFT * 3.3 + DOWN * 0.4)
            p2 = plane.copy().move_to(RIGHT * 3.3 + DOWN * 0.4)
            t1 = Text('real decay: shrink', font_size=26, color=TEAL).next_to(p1, UP, buff=0.25)
            t2 = Text('complex: shrink + rotate', font_size=26, color=PURPLE).next_to(p2, UP, buff=0.25)
            self.play(FadeIn(p1), FadeIn(p2), FadeIn(t1), FadeIn(t2), run_time=0.8)
            d1 = VGroup(*[Dot(p1.c2p(1.0 * 0.85 ** n, 0), radius=0.07, color=TEAL) for n in range(12)])
            d2 = VGroup(*[Dot(p2.c2p(1.0 * 0.85 ** n * np.cos(0.5 * n), 1.0 * 0.85 ** n * np.sin(0.5 * n)), radius=0.07,
                              color=PURPLE) for n in range(12)])
            self.play(LaggedStart(*[FadeIn(d, scale=2) for d in d1], lag_ratio=0.3),
                      LaggedStart(*[FadeIn(d, scale=2) for d in d2], lag_ratio=0.3), run_time=4)
        with self.say('m3mimo'):
            self.clear_stage(run_time=0.5)
            self.term('3  MIMO: rank-4 writes', PURPLE)
            r1 = VGroup(Mat(np.ones((4, 1)), cell=0.4), Mat(np.ones((1, 4)), cell=0.4, color=BLUE)).arrange(RIGHT, buff=0.3)
            r4 = VGroup(Mat(np.ones((4, 4)), cell=0.4), Mat(np.ones((4, 4)), cell=0.4, color=BLUE)).arrange(RIGHT, buff=0.3)
            S1 = Mat(np.full((4, 4), 0.6), cell=0.4)
            S4 = Mat(np.full((4, 4), 0.6), cell=0.4)
            row1 = VGroup(Text('rank 1', font_size=26), r1, MathTex(r'\to', font_size=40), S1).arrange(RIGHT, buff=0.45)
            row4 = VGroup(Text('rank 4', font_size=26), r4, MathTex(r'\to', font_size=40), S4).arrange(RIGHT, buff=0.45)
            VGroup(row1, row4).arrange(DOWN, buff=0.7, aligned_edge=RIGHT).move_to(LEFT * 2.8 + DOWN * 0.3)
            self.play(FadeIn(row1), run_time=0.8)
            self.play(FadeIn(row4), run_time=0.8)
            sl = Text('same state size', font_size=24, color=PURPLE).next_to(S4, DOWN, buff=0.3)
            self.play(FadeIn(sl), run_time=0.5)
            cmp = VGroup(
                VGroup(Text('compute', font_size=24, color=MUTED), Rectangle(width=0.4, height=0.5, fill_color=PURPLE, fill_opacity=0.85, stroke_width=0),
                       Rectangle(width=0.4, height=2.0, fill_color=PURPLE, fill_opacity=0.85, stroke_width=0)),
                VGroup(Text('bytes read', font_size=24, color=MUTED), Rectangle(width=0.4, height=1.4, fill_color=BLUE, fill_opacity=0.85, stroke_width=0),
                       Rectangle(width=0.4, height=1.4, fill_color=BLUE, fill_opacity=0.85, stroke_width=0)))
            for i, g in enumerate(cmp):
                x = 3.9 + i * 1.7
                g[1].move_to([x - 0.3, -2.2, 0], aligned_edge=DOWN)
                g[2].move_to([x + 0.3, -2.2, 0], aligned_edge=DOWN)
                g[0].move_to([x, -2.55, 0])
            tags = VGroup(Text('≈4×', font_size=24, color=PURPLE).next_to(cmp[0][2], UP, buff=0.1),
                          Text('same', font_size=24, color=BLUE).next_to(cmp[1][2], UP, buff=0.1))
            self.wait(1.5)
            self.play(FadeIn(cmp), FadeIn(tags), run_time=1)
            fl = Text('decode is bound by bytes → latency about flat', font_size=26, color=GREEN).move_to(RIGHT * 2.2 + UP * 2.2)
            self.play(FadeIn(fl), run_time=0.6)
        self.term(None)
        self.clear_stage()

    def recap(self):
        self.header('Chapter 2 · Forgetting')
        card, lo = self.recap_card(r'\Sc{t}=\Ct{\alpha_t}\,\Sc{t-1}+\vc{t}\kct{t}',
                                   'Fade before each write: fixed (RetNet), by input (Mamba-2), per channel (GLA).',
                                   'decay fades everything; it cannot remove one wrong association.')
        with self.say('recap', pause=0.6):
            self.play(FadeIn(card, shift=0.2 * UP), run_time=1)
        with self.say('leftover'):
            self.play(FadeIn(lo, shift=0.1 * UP), run_time=0.8)
        self.wait(1)
