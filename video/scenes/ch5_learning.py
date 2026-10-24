"""Chapter 5 · The state as a small model, trained while reading (Fixed-Size States §5)."""
from common import *


def box(label, color, w=1.6, h=0.9, font_size=28):
    r = RoundedRectangle(corner_radius=0.12, width=w, height=h, fill_color=WHITE_, fill_opacity=1, stroke_color=color, stroke_width=3)
    t = MathTex(label, font_size=font_size) if label.startswith('$') is False and '\\' in label else Text(label, font_size=font_size, color=color)
    return VGroup(r, t.move_to(r))


def hbars(items, vmax, width=5.0, h=0.5, gap=0.9, x0=-1.5, y0=1.0, fmt='{:g}', font_size=26):
    """Horizontal bars: (label, value, color)."""
    g = VGroup()
    for i, (lab, v, c) in enumerate(items):
        y = y0 - i * gap
        b = Rectangle(width=max(1e-3, width * v / vmax), height=h, fill_color=c, fill_opacity=0.85, stroke_width=0)
        b.move_to([x0, y, 0], aligned_edge=LEFT)
        t = Text(lab, font_size=font_size, color=INK).next_to([x0, y, 0], LEFT, buff=0.25)
        n = Text(fmt.format(v), font_size=font_size, color=c, weight=MEDIUM).next_to(b, RIGHT, buff=0.15)
        g.add(VGroup(t, b, n))
    return g


class Ch5Learning(VoiceScene):
    SCRIPT = 'ch5_learning'

    def construct(self):
        self.learning_view()
        self.ttt()
        self.titans()
        self.atlas_mesa()
        self.lact()
        self.recap()

    def learning_view(self):
        f = MathTex(r'\Sc{t}=\alpha_t\Sc{t-1}-\beta_t\,\nabla_{\Cv{S}}\,\ell_t\big(\alpha_t\Sc{t-1}\big)', font_size=48)
        with self.say('intro', pause=0.8):
            card = self.title_card(5, 'The state as a small model', 'trained while reading')
            self.play(FadeIn(card, shift=0.2 * UP), run_time=1)
            self.wait(2.6)
            self.play(FadeOut(card), run_time=0.6)
            self.header('5.1  Every write is one learning step')
            self.term('S = the weights of a small linear model', PURPLE)
            self.wait(5)
            self.play(Write(f), run_time=2)
            self.wait(1)
            self.play(Circumscribe(f[0][-12:], color=RED), run_time=1.2)
        with self.say('model'):
            self.play(f.animate.scale(0.75).to_edge(UP, buff=0.8), run_time=0.6)
            k = VGroup(Square(0.7, fill_color=BLUE, fill_opacity=0.85, stroke_width=0), MathTex(r'k', font_size=34, color=WHITE_))
            k[1].move_to(k[0])
            S = VGroup(RoundedRectangle(corner_radius=0.15, width=1.8, height=1.3, fill_color=PURPLE, fill_opacity=0.2, stroke_color=PURPLE, stroke_width=3),
                       Text('model S', font_size=28, color=PURPLE))
            S[1].move_to(S[0])
            p = VGroup(Square(0.7, fill_color=GREEN, fill_opacity=0.85, stroke_width=0), MathTex(r'Sk', font_size=30, color=WHITE_))
            p[1].move_to(p[0])
            L = VGroup(RoundedRectangle(corner_radius=0.15, width=1.8, height=1.0, fill_color=WHITE_, fill_opacity=1, stroke_color=RED, stroke_width=3),
                       Text('loss', font_size=28, color=RED))
            L[1].move_to(L[0])
            v = VGroup(Square(0.7, fill_color=PURPLE, fill_opacity=0.85, stroke_width=0), MathTex(r'v', font_size=34, color=WHITE_))
            v[1].move_to(v[0])
            row = VGroup(k, S, p, L).arrange(RIGHT, buff=1.0).move_to(DOWN * 0.2)
            v.next_to(L, UP, buff=0.7)
            a1, a2, a3 = arrow(k.get_right(), S.get_left()), arrow(S.get_right(), p.get_left()), arrow(p.get_right(), L.get_left())
            a4 = arrow(v.get_bottom(), L.get_top())
            g = CurvedArrow(L.get_bottom() + DOWN * 0.05, S.get_bottom() + DOWN * 0.05, angle=-PI / 2.2, color=RED, stroke_width=4)
            gl = MathTex(r'-\beta\,\nabla_S\ell', font_size=32, color=RED).next_to(g, DOWN, buff=0.1)
            self.play(FadeIn(k), GrowArrow(a1), FadeIn(S), run_time=0.8)
            self.play(GrowArrow(a2), FadeIn(p), run_time=0.6)
            self.play(GrowArrow(a3), FadeIn(L), FadeIn(v), GrowArrow(a4), run_time=0.8)
            self.wait(1.5)
            self.play(Create(g), FadeIn(gl), run_time=1)
            self.play(Indicate(S, color=PURPLE), run_time=0.8)
            self.wait(1)
            self.play(Indicate(k, color=BLUE), Indicate(v, color=PURPLE), Indicate(S, color=PURPLE), run_time=1.2)
        with self.say('losses'):
            self.clear_stage(keep=[f], run_time=0.5)
            self.term(None)
            xs = (-4.6, -1.3, 2.1, 5.0)
            hdr = VGroup(*[Text(h, font_size=24, color=MUTED).move_to([x, 1.9, 0]) for h, x in
                           zip(('rule', 'loss  ℓ(S)', 'gradient', 'decay'), xs)])
            rule = Line([-6.5, 1.6, 0], [6.5, 1.6, 0], color=LINE, stroke_width=2)
            rows = [('Linear attention', r'-\langle S\kc{t},\vc{t}\rangle', r'-\vc{t}\kct{t}', '1'),
                    ('DeltaNet', r'\tfrac12\lVert S\kc{t}-\vc{t}\rVert^2', r'(S\kc{t}-\vc{t})\kct{t}', '1'),
                    ('RetNet, Mamba-2', r'-\langle S\kc{t},\vc{t}\rangle', r'-\vc{t}\kct{t}', 'scalar < 1'),
                    ('Gated DeltaNet', r'\tfrac12\lVert S\kc{t}-\vc{t}\rVert^2', r'(S\kc{t}-\vc{t})\kct{t}', 'scalar < 1')]
            self.trows = VGroup()
            for i, (r, l, gr, d) in enumerate(rows):
                y = 1.0 - i * 0.85
                self.trows.add(VGroup(Text(r, font_size=26, weight=MEDIUM).move_to([xs[0], y, 0]), MathTex(l, font_size=34).move_to([xs[1], y, 0]),
                                      MathTex(gr, font_size=34).move_to([xs[2], y, 0]), Text(d, font_size=26, color=TEAL).move_to([xs[3], y, 0])))
            self.play(FadeIn(hdr), Create(rule), run_time=0.6)
            self.play(FadeIn(self.trows[0][:2]), run_time=0.8)
            self.wait(2)
            self.play(FadeIn(self.trows[0][2:]), run_time=0.8)
            step = MathTex(r'-\beta\cdot(-\vc{t}\kct{t})=+\beta\,\vc{t}\kct{t}', font_size=34, color=PURPLE).move_to(DOWN * 2.6)
            sl = Text('= the linear attention write', font_size=24, color=PURPLE).next_to(step, RIGHT, buff=0.3)
            self.wait(1)
            self.play(Write(step), FadeIn(sl), run_time=1.2)
            self.wait(2.5)
            self.play(FadeIn(self.trows[1]), FadeOut(step), FadeOut(sl), run_time=0.8)
            step2 = Text('one step on the squared error = the delta rule', font_size=26, color=RED).move_to(DOWN * 2.6)
            self.wait(3)
            self.play(FadeIn(step2), run_time=0.6)
            self.thdr = VGroup(hdr, rule)
        with self.say('decay'):
            self.play(FadeOut(step2), run_time=0.3)
            self.play(FadeIn(self.trows[2]), run_time=0.6)
            self.play(FadeIn(self.trows[3]), run_time=0.6)
            self.wait(2.5)
            col = SurroundingRectangle(VGroup(self.trows[0][1], self.trows[1][1]), color=RED, buff=0.12)
            self.play(Create(col), run_time=0.6)
            self.term('they differ only in the loss', RED)
            dcol = SurroundingRectangle(VGroup(*[r[3] for r in self.trows]), color=TEAL, buff=0.12)
            self.play(Create(dcol), run_time=0.6)
            self.term('decay = weight decay', TEAL)
        with self.say('toy'):
            self.clear_stage(keep=[f], run_time=0.5)
            self.term('toy: d = 3, β = 1, new pair e₁ → e₃', INK)
            S0 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]], float)
            m0 = Mat(S0, cell=0.6, numbers=True, fmt='{:.0f}', vmax=1).move_to(LEFT * 4.6 + DOWN * 0.3)
            l0 = Text('state before', font_size=24, color=MUTED).next_to(m0, UP, buff=0.25)
            self.play(FadeIn(m0), FadeIn(l0), run_time=0.8)
            self.wait(4)
            la = Mat(S0, cell=0.6, numbers=True, fmt='{:.0f}', vmax=1).move_to(LEFT * 0.4 + DOWN * 0.3)
            de = Mat(S0, cell=0.6, numbers=True, fmt='{:.0f}', vmax=1).move_to(RIGHT * 4.0 + DOWN * 0.3)
            lla = Text('linear attention step', font_size=24, color=GOLD).next_to(la, UP, buff=0.25)
            lde = Text('delta step', font_size=24, color=PURPLE).next_to(de, UP, buff=0.25)
            self.play(TransformFromCopy(m0, la), TransformFromCopy(m0, de), FadeIn(lla), FadeIn(lde), run_time=1)
            self.wait(1.5)
            S1 = S0.copy()
            S1[2, 0] = 1
            S2 = S0.copy()
            S2[0, 0], S2[2, 0] = 0, 1
            self.play(la.set_values(S1), run_time=1)
            r1 = MathTex(r'S e_1=(1,0,1)', font_size=32).next_to(la, DOWN, buff=0.3)
            e1 = Text('error 1: keeps the old value', font_size=22, color=RED).next_to(r1, DOWN, buff=0.15)
            self.play(FadeIn(r1), FadeIn(e1), Circumscribe(VGroup(*[la.cell(i, 0) for i in range(3)]), color=GOLD), run_time=1.2)
            self.wait(2.5)
            self.play(de.set_values(S2), run_time=1)
            r2 = MathTex(r'S e_1=(0,0,1)', font_size=32).next_to(de, DOWN, buff=0.3)
            e2 = Text('error 0', font_size=22, color=GREEN).next_to(r2, DOWN, buff=0.15)
            self.play(FadeIn(r2), FadeIn(e2), Circumscribe(VGroup(*[de.cell(i, 0) for i in range(3)]), color=PURPLE), run_time=1.2)
        with self.say('next'):
            self.clear_stage(keep=[f], run_time=0.4)
            self.term(None)
            chips = VGroup(*[VGroup(panel(3.6, 1.0, color=c), Text(t, font_size=28, color=c)) for t, c in
                             (('a richer model', PURPLE), ('a better optimizer', TEAL), ('a better loss', RED))])
            for c in chips:
                c[1].move_to(c[0])
            chips.arrange(RIGHT, buff=0.5).move_to(DOWN * 0.3)
            self.play(LaggedStart(*[FadeIn(c, shift=0.2 * UP) for c in chips], lag_ratio=0.4), run_time=1.6)
        self.clear_stage()

    def ttt(self):
        self.header('5.2  TTT: the state is a model, trained in mini-batches')
        f = MathTex(r'W_t=W_{t-1}-\eta_t\nabla_W\big\lVert f(\Ck{\theta_Kx_t};W)-\Cv{\theta_Vx_t}\big\rVert^2,\qquad \Cg{z_t}=f(\Cq{\theta_Qx_t};W_t)',
                    font_size=38).to_edge(UP, buff=0.85)
        if f.width > 13:
            f.scale_to_fit_width(13).to_edge(UP, buff=0.85)
        with self.say('ttt', pause=0.8):
            self.term('TTT layers (2024)', PURPLE)
            # a tiny MLP drawing for f
            layers = [3, 4, 3]
            nodes = VGroup(*[VGroup(*[Circle(0.16, color=PURPLE, fill_color=WHITE_, fill_opacity=1, stroke_width=3) for _ in range(n)]).arrange(DOWN, buff=0.25)
                             for n in layers]).arrange(RIGHT, buff=0.9).move_to(RIGHT * 2.6 + DOWN * 0.6)
            edges = VGroup(*[Line(a.get_center(), b.get_center(), color=PURPLE, stroke_width=1.5, stroke_opacity=0.6)
                             for l in range(len(layers) - 1) for a in nodes[l] for b in nodes[l + 1]])
            fl = Text('f: linear, or a small MLP', font_size=24, color=PURPLE).next_to(nodes, DOWN, buff=0.35)
            self.play(FadeIn(edges), FadeIn(nodes), FadeIn(fl), run_time=1)
            self.wait(3)
            x = VGroup(Square(0.6, fill_color=SOFT, fill_opacity=1, stroke_color=INK, stroke_width=2), MathTex('x_t', font_size=30))
            x[1].move_to(x[0])
            x.move_to(LEFT * 5.2 + DOWN * 0.6)
            proj = VGroup(*[MathTex(t, font_size=34).move_to(LEFT * 2.4 + DOWN * 0.6 + UP * dy) for t, dy in
                            ((r'\Ck{\theta_Kx_t}', 1.1), (r'\Cv{\theta_Vx_t}', 0), (r'\Cq{\theta_Qx_t}', -1.1))])
            arr = VGroup(*[arrow(x.get_right(), p.get_left()) for p in proj])
            self.play(FadeIn(x), run_time=0.4)
            self.play(LaggedStart(*[AnimationGroup(GrowArrow(a), FadeIn(p)) for a, p in zip(arr, proj)], lag_ratio=0.4), run_time=1.8)
            self.wait(1)
            self.play(Write(f), run_time=2.5)
        with self.say('batch'):
            self.clear_stage(keep=[f], run_time=0.5)
            self.term('mini-batch: every gradient uses the batch-start weights', TEAL)
            b = 4
            toks = VGroup(*[Square(0.45, fill_color=SOFT, fill_opacity=1, stroke_color=LINE) for _ in range(2 * b)]).arrange(RIGHT, buff=0.12)
            toks.move_to(DOWN * 2.3)
            for i in range(b, 2 * b):
                toks[i].shift(RIGHT * 0.5)
            W0 = box('W₀', PURPLE, w=1.2, h=0.7).move_to(LEFT * 4.2 + UP * 0.8)
            W1 = box('W₁', PURPLE, w=1.2, h=0.7).move_to(RIGHT * 0.3 + UP * 0.8)
            W2 = box('W₂', PURPLE, w=1.2, h=0.7).move_to(RIGHT * 4.8 + UP * 0.8)
            self.play(FadeIn(toks), FadeIn(W0), run_time=0.6)
            bl = Text(f'b = {b} tokens (illustration)', font_size=22, color=MUTED).next_to(toks[:b], DOWN, buff=0.2)
            self.play(FadeIn(bl), run_time=0.4)
            g1 = VGroup(*[Line(W0.get_bottom(), toks[i].get_top(), color=RED, stroke_width=2.5) for i in range(b)])
            self.play(LaggedStart(*[Create(l) for l in g1], lag_ratio=0.1), run_time=1)
            s1 = arrow(W0.get_right(), W1.get_left(), color=PURPLE)
            mm = Text('one matrix product', font_size=22, color=RED).next_to(s1, UP, buff=0.12)
            self.play(GrowArrow(s1), FadeIn(W1), FadeIn(mm), run_time=0.8)
            self.wait(1.5)
            g2 = VGroup(*[Line(W1.get_bottom(), toks[i].get_top(), color=RED, stroke_width=2.5) for i in range(b, 2 * b)])
            self.play(LaggedStart(*[Create(l) for l in g2], lag_ratio=0.1), run_time=1)
            s2 = arrow(W1.get_right(), W2.get_left(), color=PURPLE)
            self.play(GrowArrow(s2), FadeIn(W2), run_time=0.8)
        with self.say('batchends'):
            self.clear_stage(keep=[f], run_time=0.5)
            self.term(None)
            line = NumberLine(x_range=[0, 1, 0.25], length=10, color=INK, include_numbers=False, include_ticks=False).move_to(DOWN * 0.4)
            e1 = VGroup(Dot(line.n2p(0), color=PURPLE, radius=0.1), MathTex('b=1', font_size=34).next_to(line.n2p(0), UP, buff=0.25),
                        Text('online learning,\nclose to the delta rule', font_size=24, color=PURPLE, line_spacing=0.9).next_to(line.n2p(0), DOWN, buff=0.35))
            e2 = VGroup(Dot(line.n2p(1), color=GOLD, radius=0.1), MathTex('b=T', font_size=34).next_to(line.n2p(1), UP, buff=0.25),
                        Text('linear f, step ½, zero start:\nexactly linear attention', font_size=24, color=GOLD, line_spacing=0.9).next_to(line.n2p(1), DOWN, buff=0.35))
            e1[2].align_to(line.n2p(0) + LEFT * 0.4, LEFT)
            e2[2].align_to(line.n2p(1) + RIGHT * 0.4, RIGHT)
            self.play(Create(line), run_time=0.6)
            self.play(FadeIn(e1), run_time=0.8)
            self.wait(2.5)
            self.play(FadeIn(e2), run_time=0.8)
            self.wait(5.5)
            m = VGroup(Triangle(color=TEAL, fill_opacity=1).scale(0.15).rotate(PI), MathTex('b=16', font_size=34, color=TEAL))
            m[1].next_to(m[0], UP, buff=0.12)
            m.move_to(line.n2p(0.12) + UP * 0.6)
            tl = Text('TTT', font_size=26, color=TEAL, weight=MEDIUM).next_to(m, UP, buff=0.1)
            self.play(FadeIn(m, shift=DOWN * 0.2), FadeIn(tl), run_time=0.8)
        with self.say('tttnum'):
            self.clear_stage(keep=[f], run_time=0.5)
            self.term('reported 125M ablation · perplexity, lower is better', INK)
            bars = hbars([('linear attention', 15.91, GOLD), ('TTT', 11.09, PURPLE)], 16, width=6, x0=-1.6, y0=0.6, gap=1.2)
            self.play(LaggedStart(*[AnimationGroup(FadeIn(g[0]), GrowFromEdge(g[1], LEFT), FadeIn(g[2])) for g in bars], lag_ratio=0.5), run_time=2)
            self.wait(2.5)
            mb = Text('mini-batch step alone: −1.70, the largest single gain', font_size=28, color=TEAL).move_to(DOWN * 2.0)
            self.play(FadeIn(mb), run_time=0.8)
        self.term(None)
        self.clear_stage()

    def titans(self):
        self.header('5.3  Titans: surprise, momentum and forgetting')
        f = MathTex(r'\mathcal M_t=(1-\Ct{\alpha_t})\,\mathcal M_{t-1}+m_t,\qquad m_t=\eta_t\,m_{t-1}-\theta_t\,\nabla\big\lVert\mathcal M_{t-1}(\kc{t})-\vc{t}\big\rVert^2',
                    font_size=38).to_edge(UP, buff=0.85)
        if f.width > 13:
            f.scale_to_fit_width(13).to_edge(UP, buff=0.85)
        with self.say('titans', pause=0.8):
            self.term('Titans (2024)', PURPLE)
            self.play(Write(f), run_time=2.5)
            fl = Text('forgetting', font_size=26, color=TEAL).move_to(f.get_left() + RIGHT * 2.2 + DOWN * 1.0)
            ml = Text('momentum', font_size=26, color=PURPLE).move_to(f.get_right() + LEFT * 4.4 + DOWN * 1.0)
            sl = Text('surprise', font_size=26, color=RED).move_to(f.get_right() + LEFT * 1.6 + DOWN * 1.0)
            self.play(LaggedStart(FadeIn(ml), FadeIn(fl), FadeIn(sl), lag_ratio=0.5), run_time=1.5)
            parts = VGroup(fl, ml, sl)
        n = 14
        sur = np.full(n, 0.25)
        sur[5] = 2.6
        mom = np.zeros(n)
        for t in range(n):
            mom[t] = (0.6 * mom[t - 1] if t else 0) + 0.4 * sur[t]
        x0 = -5.6
        toks = VGroup(*[Square(0.45, fill_color=SOFT, fill_opacity=1, stroke_color=LINE) for _ in range(n)]).arrange(RIGHT, buff=0.32)
        toks.move_to(DOWN * 2.6)
        with self.say('surprise'):
            self.play(FadeOut(parts), FadeIn(toks), run_time=0.5)
            sb = VGroup(*[Rectangle(width=0.32, height=0.8 * s, fill_color=RED, fill_opacity=0.85, stroke_width=0)
                          .next_to(toks[i], UP, buff=0.1) for i, s in enumerate(sur)])
            sl = Text('surprise: gradient size', font_size=24, color=RED).move_to(LEFT * 3.8 + UP * 1.4)
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in sb], lag_ratio=0.08), FadeIn(sl), run_time=1.6)
            self.term('wrong prediction → large gradient → strong write', RED)
            self.wait(3.5)
            mb = VGroup(*[Rectangle(width=0.32, height=max(1e-3, 0.8 * m), fill_color=PURPLE, fill_opacity=0.85, stroke_width=0)
                          for m in mom])
            for i, b in enumerate(mb):
                b.next_to(toks[i], UP, buff=0.1).shift(RIGHT * 0.36)
            ml = Text('momentum m_t keeps writing', font_size=24, color=PURPLE).move_to(RIGHT * 3.2 + UP * 1.4)
            self.play(LaggedStart(*[GrowFromEdge(b, DOWN) for b in mb], lag_ratio=0.08), FadeIn(ml), run_time=2)
            self.term(None)
            ctx = Brace(VGroup(*toks[5:10]), DOWN, color=PURPLE)
            ct = Text('context around the surprise is stored too', font_size=22, color=PURPLE).next_to(ctx, DOWN, buff=0.08)
            self.play(GrowFromCenter(ctx), FadeIn(ct), run_time=0.8)
        with self.say('forget'):
            self.clear_stage(keep=[f], run_time=0.5)
            self.term('forgetting α_t = weight decay', TEAL)
            bars = hbars([('full Titans', 27.01, PURPLE), ('without forgetting', 29.04, RED)], 30, width=6, x0=-1.2, y0=0.4, gap=1.2)
            hd = Text('reported ablation · perplexity', font_size=24, color=MUTED).move_to(UP * 1.7)
            self.play(FadeIn(hd), LaggedStart(*[AnimationGroup(FadeIn(g[0]), GrowFromEdge(g[1], LEFT), FadeIn(g[2])) for g in bars], lag_ratio=0.5), run_time=2)
            cm = Text('removing it costs the most', font_size=26, color=RED).move_to(DOWN * 2.0)
            self.play(FadeIn(cm), run_time=0.6)
        with self.say('deep'):
            self.clear_stage(keep=[f], run_time=0.5)
            self.term(None)
            layers = [3, 4, 4, 3]
            nodes = VGroup(*[VGroup(*[Circle(0.15, color=PURPLE, fill_color=WHITE_, fill_opacity=1, stroke_width=3) for _ in range(k)]).arrange(DOWN, buff=0.22)
                             for k in layers]).arrange(RIGHT, buff=0.7).move_to(LEFT * 3.2 + DOWN * 0.4)
            edges = VGroup(*[Line(a.get_center(), b.get_center(), color=PURPLE, stroke_width=1.5, stroke_opacity=0.6)
                             for l in range(len(layers) - 1) for a in nodes[l] for b in nodes[l + 1]])
            ml = Text('memory: a deep MLP', font_size=26, color=PURPLE).next_to(nodes, DOWN, buff=0.35)
            self.play(FadeIn(edges), FadeIn(nodes), FadeIn(ml), run_time=1)
            x = Text('no neat chunk algebra', font_size=28, color=RED).move_to(RIGHT * 3.0 + UP * 0.3)
            y = Text('trains slower than\nMamba-2 or Gated DeltaNet', font_size=26, color=MUTED, line_spacing=0.9).next_to(x, DOWN, buff=0.4)
            self.play(FadeIn(x), run_time=0.6)
            self.play(FadeIn(y), run_time=0.6)
        self.clear_stage()

    def atlas_mesa(self):
        self.header('5.4  ATLAS and Mesa: fit more than the newest pair')
        n = 12
        pairs = VGroup(*[VGroup(Square(0.38, fill_color=BLUE, fill_opacity=0.85, stroke_width=0),
                                Square(0.38, fill_color=PURPLE, fill_opacity=0.85, stroke_width=0)).arrange(DOWN, buff=0.05) for _ in range(n)])
        pairs.arrange(RIGHT, buff=0.15).move_to(UP * 0.9)
        pl = Text('past pairs (k, v)  →  newest', font_size=22, color=MUTED).next_to(pairs, UP, buff=0.25)

        def win(m, color, label, dy):
            r = SurroundingRectangle(VGroup(*pairs[n - m:]), color=color, buff=0.08)
            t = Text(label, font_size=24, color=color).next_to(r, DOWN, buff=0.15 + dy)
            return VGroup(r, t)

        with self.say('atlas'):
            self.play(FadeIn(pairs, lag_ratio=0.05), FadeIn(pl), run_time=1)
            w1 = win(1, RED, 'delta rule: the newest pair', 0)
            self.play(Create(w1[0]), FadeIn(w1[1]), run_time=0.8)
            self.wait(1.5)
            w2 = win(4, TEAL, 'ATLAS: the last c pairs', 0)
            self.play(ReplacementTransform(w1, w2), run_time=0.8)
            self.term('ATLAS (2025): window of c pairs · Muon optimizer', TEAL)
            cap = MathTex(r'\text{polynomial features: capacity } O(d_k)\;\to\;O(d_k^{\,p})', font_size=36).move_to(DOWN * 1.6)
            self.wait(2.5)
            self.play(Write(cap), run_time=1.5)
        with self.say('mesa', pause=0.8):
            w3 = win(n, PURPLE, 'Mesa: all pairs, solved exactly at read time', 0)
            self.play(ReplacementTransform(w2, w3), FadeOut(cap), run_time=0.8)
            self.term('Mesa (2025)', PURPLE)
            g = MathTex(r'\Cg{o_t}=G_t(H_t+\Lambda)^{-1}\qc{t},\qquad G_t=\sum\vc{}\kct{},\quad H_t=\sum\kc{}\kct{}', font_size=40).move_to(DOWN * 1.8)
            self.wait(2.5)
            self.play(Write(g), run_time=2.5)
        with self.say('mesatoy'):
            self.clear_stage(run_time=0.5)
            plane = NumberPlane(x_range=[-0.3, 1.2, 0.5], y_range=[-0.3, 1.2, 0.5], x_length=3.8, y_length=3.8,
                                background_line_style={'stroke_color': LINE, 'stroke_width': 1},
                                axis_config={'stroke_color': MUTED, 'stroke_width': 1.5}).move_to(LEFT * 4.0 + DOWN * 0.3)
            k1 = Arrow(plane.c2p(0, 0), plane.c2p(1, 0), buff=0, color=BLUE, stroke_width=6)
            k2 = Arrow(plane.c2p(0, 0), plane.c2p(0.6, 0.8), buff=0, color=BLUE, stroke_width=6)
            l1 = MathTex(r'\kc{1}\mapsto+1', font_size=30).next_to(plane.c2p(1, 0), DOWN, buff=0.15)
            l2 = MathTex(r'\kc{2}=(0.6,0.8)\mapsto-1', font_size=30).next_to(plane.c2p(0.6, 0.8), UP, buff=0.12)
            self.play(Create(plane), run_time=0.6)
            self.play(GrowArrow(k1), FadeIn(l1), run_time=0.8)
            self.wait(1.5)
            self.play(GrowArrow(k2), FadeIn(l2), run_time=0.8)
            self.wait(2.5)
            hd = MathTex(r'\text{read at }\kc{1}\ (\text{target }1)', font_size=32).move_to(RIGHT * 2.4 + UP * 2.1)
            bars = hbars([('linear attention', 0.40, GOLD), ('DeltaNet', 0.04, RED), ('Mesa', 1.00, GREEN)], 1.0, width=3.4,
                         x0=1.4, y0=1.0, gap=1.0, fmt='{:.2f}')
            tgt = DashedLine([1.4 + 3.4, 1.5, 0], [1.4 + 3.4, -1.5, 0], color=GREEN, stroke_width=2)
            self.play(FadeIn(hd), Create(tgt), run_time=0.6)
            for g in bars:
                self.play(FadeIn(g[0]), GrowFromEdge(g[1], LEFT), FadeIn(g[2]), run_time=0.8)
                self.wait(0.7)
            cost = Text('price: with k CG steps, about k × GLA', font_size=26, color=RED).move_to(RIGHT * 2.6 + DOWN * 2.4)
            self.wait(1)
            self.play(FadeIn(cost), run_time=0.6)
        self.term(None)
        self.clear_stage()

    def lact(self):
        self.header('5.5  LaCT: huge chunks make it fast')
        with self.say('lact'):
            self.term('TTT update: b = 16, width 64', INK)
            ax = Line(DOWN * 2.4 + LEFT * 2.8, DOWN * 2.4 + RIGHT * 2.8, color=INK, stroke_width=2)
            u = 4.0 / 290
            b1 = Rectangle(width=1.2, height=11 * u, fill_color=PURPLE, fill_opacity=0.85, stroke_width=0).move_to(LEFT * 1.6 + DOWN * 2.4, aligned_edge=DOWN)
            b2 = Rectangle(width=1.2, height=290 * u, fill_color=LINE, fill_opacity=0.9, stroke_color=MUTED, stroke_width=1.5).move_to(RIGHT * 1.6 + DOWN * 2.4, aligned_edge=DOWN)
            t1 = Text('≈ 11', font_size=28, color=PURPLE, weight=MEDIUM).next_to(b1, UP, buff=0.12)
            t2 = Text('≈ 290', font_size=28, color=INK, weight=MEDIUM).next_to(b2, UP, buff=0.12)
            n1 = Text('TTT update', font_size=22, color=PURPLE).next_to(b1, DOWN, buff=0.15)
            n2 = Text('what an H100 needs', font_size=22).next_to(b2, DOWN, buff=0.15)
            yl = Text('FLOPs per byte', font_size=24, color=MUTED).move_to(LEFT * 4.8 + UP * 0.5)
            self.play(Create(ax), FadeIn(yl), run_time=0.5)
            self.play(GrowFromEdge(b1, DOWN), FadeIn(t1), FadeIn(n1), run_time=1)
            self.wait(2.5)
            self.play(GrowFromEdge(b2, DOWN), FadeIn(t2), FadeIn(n2), run_time=1.2)
            idle = Text('small mini-batches\nleave the GPU idle', font_size=26, color=RED, line_spacing=0.9).move_to(RIGHT * 4.8 + UP * 0.6)
            self.play(FadeIn(idle), run_time=0.6)
        with self.say('lact2'):
            self.clear_stage(run_time=0.5)
            self.term('LaCT (2025): one update per 2K to 1M tokens', TEAL)
            small = VGroup(*[Rectangle(width=0.18, height=0.5, fill_color=PURPLE, fill_opacity=0.25, stroke_color=PURPLE, stroke_width=1.5) for _ in range(30)]).arrange(RIGHT, buff=0.04)
            big = Rectangle(width=small.width, height=0.5, fill_color=TEAL, fill_opacity=0.25, stroke_color=TEAL, stroke_width=2)
            VGroup(small, big).arrange(DOWN, buff=0.6, aligned_edge=LEFT).move_to(LEFT * 2.6 + UP * 1.2)
            sl = Text('TTT: many tiny updates', font_size=22, color=PURPLE).next_to(small, RIGHT, buff=0.3)
            bl = Text('LaCT: one big matrix product', font_size=22, color=TEAL).next_to(big, RIGHT, buff=0.3)
            self.play(FadeIn(small, lag_ratio=0.02), FadeIn(sl), run_time=0.8)
            self.play(FadeIn(big), FadeIn(bl), run_time=0.8)
            self.wait(1.5)
            gb = hbars([('earlier TTT', 5, PURPLE), ('LaCT', 70, TEAL)], 100, width=6, x0=-1.6, y0=-1.0, gap=1.0, fmt='{:g}%')
            gb[0][2].become(Text('under 5%', font_size=26, color=PURPLE, weight=MEDIUM).next_to(gb[0][1], RIGHT, buff=0.15))
            gb[1][2].become(Text('up to 70%', font_size=26, color=TEAL, weight=MEDIUM).next_to(gb[1][1], RIGHT, buff=0.15))
            gl = Text('reported GPU use', font_size=24, color=MUTED).next_to(gb, UP, buff=0.25).align_to(gb, LEFT)
            self.play(FadeIn(gl), LaggedStart(*[AnimationGroup(FadeIn(g[0]), GrowFromEdge(g[1], LEFT), FadeIn(g[2])) for g in gb], lag_ratio=0.5), run_time=2)
            self.wait(2)
            grow = Text('and the state can grow far larger', font_size=26, color=TEAL).move_to(DOWN * 2.95 + RIGHT * 3.6)
            self.play(FadeIn(grow), run_time=0.6)
        with self.say('lact3'):
            self.clear_stage(run_time=0.5)
            self.term('paired with sliding-window attention', BLUE)
            n = 24
            row = VGroup(*[Square(0.36, fill_color=TEAL, fill_opacity=0.18, stroke_color=TEAL, stroke_width=1) for _ in range(n)]).arrange(RIGHT, buff=0.04).move_to(UP * 0.4)
            cl = Text('one huge chunk: never sees its own update', font_size=24, color=TEAL).next_to(row, UP, buff=0.3)
            self.play(FadeIn(row), FadeIn(cl), run_time=0.6)
            w = SurroundingRectangle(VGroup(*row[0:5]), color=BLUE, buff=0.05, stroke_width=4)
            wl = Text('sliding window: the local part', font_size=24, color=BLUE).next_to(row, DOWN, buff=0.4)
            self.play(Create(w), FadeIn(wl), run_time=0.6)
            self.play(w.animate.move_to(VGroup(*row[n - 5:])), run_time=3, rate_func=linear)
        self.term(None)
        self.clear_stage()

    def recap(self):
        self.header('5  The family at a glance')
        rows = [('TTT-Linear / MLP', r'\lVert f(\theta_Kx)-\theta_Vx\rVert^2', 'GD, mini-batch b = 16', 'linear map or 2-layer MLP'),
                ('Titans', r'\lVert\mathcal M(k)-v\rVert^2', 'GD + momentum + decay', 'MLP'),
                ('ATLAS', r'\text{last } c\text{ pairs, polynomial features}', 'Muon', 'memory on φ(k)'),
                ('Mesa', r'\text{all pairs, discounted, ridge}', 'exact solve (CG)', 'G and H'),
                ('LaCT', r'-f_W(k)^{\top}v', 'Muon, once per huge chunk', 'SwiGLU MLP')]
        xs = (-5.4, -1.9, 1.7, 5.0)
        hdr = VGroup(*[Text(h, font_size=22, color=MUTED).move_to([x, 2.4, 0]) for h, x in zip(('layer', 'loss', 'optimizer', 'state'), xs)])
        rule = Line([-6.8, 2.1, 0], [6.8, 2.1, 0], color=LINE, stroke_width=2)
        tab = VGroup()
        for i, (a, b, c, d) in enumerate(rows):
            y = 1.5 - i * 0.78
            r = VGroup(Text(a, font_size=24, weight=MEDIUM).move_to([xs[0], y, 0]), MathTex(b, font_size=30).move_to([xs[1], y, 0]),
                       Text(c, font_size=22).move_to([xs[2], y, 0]), Text(d, font_size=22, color=PURPLE).move_to([xs[3], y, 0]))
            for m, wmax in zip(r, (2.6, 3.4, 3.2, 3.0)):
                if m.width > wmax:
                    m.scale_to_fit_width(wmax)
            tab.add(r)
        with self.say('table'):
            self.play(FadeIn(hdr), Create(rule), LaggedStart(*[FadeIn(r) for r in tab], lag_ratio=0.15), run_time=2.2)
        self.play(FadeOut(VGroup(hdr, rule, tab)), run_time=0.6)
        self.header('Chapter 5 · The state as a small model')
        card, lo = self.recap_card(r'\Sc{t}=\alpha_t\Sc{t-1}-\beta_t\,\nabla_{\Cv{S}}\,\ell_t\big(\alpha_t\Sc{t-1}\big)',
                                   'Every write is one gradient step on the newest pair.',
                                   'learning is not recall; a better learner still writes into a fixed state.')
        with self.say('recap', pause=0.6):
            self.play(FadeIn(card, shift=0.2 * UP), run_time=1)
        with self.say('leftover'):
            self.play(FadeIn(lo, shift=0.1 * UP), run_time=0.8)
        self.wait(1)
