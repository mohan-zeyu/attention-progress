"""Chapter 1 · Linear attention: a lookup table stored as a matrix (Fixed-Size States §1)."""
from common import *


class Ch1Linear(VoiceScene):
    SCRIPT = 'ch1_linear'

    def construct(self):
        self.problem()
        self.regroup()
        self.mechanics()
        self.size()
        self.remember()
        self.softmax_jobs()
        self.three_forms()
        self.recap()

    # -- 1.1 ---------------------------------------------------------------
    def problem(self):
        with self.say('intro'):
            card = self.title_card(1, 'Linear attention', 'a lookup table stored as a matrix')
            self.play(FadeIn(card, shift=0.2 * UP), run_time=1)
            self.wait(2.6)
            self.play(FadeOut(card), run_time=0.6)
            self.header('1.1  From the cache to the state')
            recs = VGroup(*[self.record() for _ in range(9)]).arrange(RIGHT, buff=0.16).move_to(DOWN * 0.4)
            cap = Text('KV cache: one record per past token', font_size=26, color=MUTED).next_to(recs, DOWN, buff=0.5)
            self.play(LaggedStart(*[FadeIn(r, shift=0.2 * UP) for r in recs], lag_ratio=0.25), FadeIn(cap), run_time=2.6)
            q = Square(0.5, fill_color=GOLD, fill_opacity=0.9, stroke_width=0).move_to(UP * 2.4)
            ql = MathTex(r'\qc{t}', font_size=36).next_to(q, RIGHT, buff=0.15)
            lines = VGroup(*[Line(q.get_bottom(), r[0].get_top(), color=GOLD, stroke_width=2, stroke_opacity=0.7)
                             for r in recs])
            self.play(FadeIn(q), FadeIn(ql), LaggedStart(*[Create(l) for l in lines], lag_ratio=0.1), run_time=2)
            self.term('softmax normalizes over every past key', RED)
            more = VGroup(*[self.record() for _ in range(4)])
            grown = VGroup(*recs, *more).copy().arrange(RIGHT, buff=0.16).move_to(DOWN * 0.4)
            for m, g in zip(more, grown[len(recs):]):
                m.move_to(g).shift(RIGHT * 0.6)
            self.play(*[r.animate.move_to(g) for r, g in zip(recs, grown)],
                      *[FadeIn(m) for m in more], FadeOut(lines), run_time=1.6)
            lines = VGroup(*[Line(q.get_bottom(), r[0].get_top(), color=GOLD, stroke_width=2, stroke_opacity=0.7)
                             for r in [*recs, *more]])
            self.play(Create(lines), run_time=1)
        self.term(None)
        self.clear_stage()

    def record(self):
        k = Square(0.45, fill_color=BLUE, fill_opacity=0.85, stroke_width=0)
        v = Square(0.45, fill_color=PURPLE, fill_opacity=0.85, stroke_width=0)
        return VGroup(k, v).arrange(DOWN, buff=0.07)

    def regroup(self):
        soft = MathTex(r'\oc{t}=\sum_{j\le t}', r'\frac{\exp(\kct{j}\qc{t})}{\sum_{l\le t}\exp(\kct{l}\qc{t})}',
                       r'\,\vc{j}', font_size=46).move_to(UP * 1.6)
        lin = MathTex(r'\oc{t}=\sum_{j\le t}\vc{j}\,(\kct{j}\qc{t})', font_size=46).move_to(UP * 1.6)
        with self.say('drop'):
            self.play(Write(soft), run_time=1.6)
            strike = Cross(soft[1], stroke_color=RED, stroke_width=5)
            self.play(Create(strike), run_time=0.6)
            self.term('drop the softmax', RED)
            self.play(FadeOut(strike), FadeOut(soft), run_time=0.6)
            self.play(FadeIn(lin), run_time=0.6)
        full = MathTex(r'\oc{t}=\sum_{j\le t}\vc{j}\,(\kct{j}\qc{t})', r'=', r'\Big(\sum_{j\le t}\vc{j}\kct{j}\Big)',
                       r'\qc{t}', r'=\Sc{t}\,\qc{t}', font_size=46).move_to(UP * 1.6)
        if full.width > 13:
            full.scale_to_fit_width(13)
        with self.say('regroup', pause=0.8):
            self.play(ReplacementTransform(lin, full[0]), run_time=0.8)
            self.play(Write(full[1:4]), run_time=1.6)
            br = Brace(full[2], DOWN, color=PURPLE)
            bt = Text('no query inside', font_size=26, color=PURPLE).next_to(br, DOWN, buff=0.1)
            self.wait(1.5)
            self.play(GrowFromCenter(br), FadeIn(bt), run_time=0.8)
            self.wait(1.5)
            self.play(Write(full[4]), run_time=1)
            self.term('the state  S_t', PURPLE)
            box = SurroundingRectangle(full[4][1:], color=PURPLE, buff=0.1)
            self.play(Create(box), run_time=0.6)
        self.clear_stage()
        self.term(None)

    # -- write, read, discard ------------------------------------------------
    def mechanics(self):
        rng = np.random.default_rng(3)
        S0 = rng.uniform(0.2, 1.4, (4, 4))
        v1, k1 = np.array([2.0, 1.0, 3.0, 0.5]), np.array([0.7, 0.1, 0.5, 0.5])
        VMAX = 3.6
        f = MathTex(r'\Sc{t}=\Sc{t-1}+\vc{t}\kct{t}', font_size=46).to_edge(UP, buff=0.9)
        S = Mat(S0, cell=0.6, vmax=VMAX).move_to(RIGHT * 3.4 + DOWN * 0.6)
        Sl = tlabel(r'\Sc{t-1}', S, DOWN)
        with self.say('write'):
            self.play(Write(f), FadeIn(S), FadeIn(Sl), run_time=1.4)
            v = Mat(v1[:, None], cell=0.6, color=PURPLE, vmax=VMAX).move_to(LEFT * 5.6 + DOWN * 0.6)
            k = Mat(k1[None, :], cell=0.6, color=BLUE, vmax=1).move_to(LEFT * 2.2 + UP * 1.6)
            vl, kl = tlabel(r'\vc{t}', v, DOWN), tlabel(r'\kct{t}', k, LEFT)
            self.play(FadeIn(v), FadeIn(vl), FadeIn(k), FadeIn(kl), run_time=1)
            P = Mat(np.zeros((4, 4)), cell=0.6, vmax=VMAX).move_to(LEFT * 2.2 + DOWN * 0.6)
            self.add(P)
            self.term('outer product  v kᵀ', PURPLE)
            outer = np.outer(v1, k1)
            for i in range(4):
                hi = VGroup(SurroundingRectangle(v.cell(i, 0), color=GOLD, buff=0.02),
                            SurroundingRectangle(k, color=GOLD, buff=0.02))
                self.play(FadeIn(hi), *[P.cell(i, j).animate.set_fill(PURPLE, min(1, outer[i, j] / VMAX) * 0.9)
                                        for j in range(4)], run_time=0.7)
                self.play(FadeOut(hi), run_time=0.25)
            P.vals = outer
            Pl = tlabel(r'\vc{t}\kct{t}', P, DOWN)
            plus = MathTex('+', font_size=60).move_to((P.get_right() + S.get_left()) / 2)
            self.play(FadeIn(Pl), FadeIn(plus), run_time=0.6)
            ghost = P.copy()
            self.play(ghost.animate.move_to(S).set_opacity(0.4), run_time=1.0)
            self.play(FadeOut(ghost), S.set_values(S0 + outer), Transform(Sl, tlabel(r'\Sc{t}', S, DOWN)), run_time=0.8)
        q1 = np.array([0.2, 0.8, 0.4, 0.4])
        with self.say('read'):
            self.play(FadeOut(VGroup(v, vl, k, kl, P, Pl, plus)), run_time=0.6)
            self.play(S.animate.shift(LEFT * 4.5), Sl.animate.shift(LEFT * 4.5), run_time=0.8)
            q = Mat(q1[:, None], cell=0.6, color=GOLD, vmax=1).next_to(S, RIGHT, buff=0.5)
            eq = MathTex('=', font_size=60).next_to(q, RIGHT, buff=0.45)
            out = S.vals @ q1
            o = Mat(np.zeros((4, 1)), cell=0.6, color=GREEN, vmax=out.max()).next_to(eq, RIGHT, buff=0.45)
            ql, ol = tlabel(r'\qc{t}', q, DOWN), tlabel(r'\oc{t}', o, DOWN)
            fr = MathTex(r'\oc{t}=\Sc{t}\,\qc{t}', font_size=46).move_to(f)
            self.play(FadeIn(q), FadeIn(ql), FadeIn(eq), FadeIn(o), FadeIn(ol), Transform(f, fr), run_time=1)
            self.term('read  o = S q', GREEN)
            for i in range(4):
                row = SurroundingRectangle(VGroup(*[S.cell(i, j) for j in range(4)]), color=GREEN, buff=0.02)
                self.play(FadeIn(row), o.cell(i, 0).animate.set_fill(GREEN, out[i] / out.max() * 0.9), run_time=0.5)
                self.play(FadeOut(row), run_time=0.2)
        with self.say('discard'):
            self.term('write · read · discard', INK)
            kv = VGroup(Mat(v1[:, None], cell=0.45, vmax=VMAX), Mat(k1[None, :], cell=0.45, color=BLUE, vmax=1))
            kv.arrange(RIGHT, buff=0.4).move_to(RIGHT * 4.3 + DOWN * 2.6)
            self.play(FadeOut(VGroup(q, ql, eq, o, ol)), FadeIn(kv), run_time=0.6)
            x = Cross(kv, stroke_color=RED, stroke_width=5)
            self.play(Create(x), run_time=0.4)
            self.play(FadeOut(kv), FadeOut(x), run_time=0.5)
            self.play(S.animate.move_to(RIGHT * 2.0 + DOWN * 0.6), Sl.animate.move_to(RIGHT * 2.0 + DOWN * 2.15),
                      Transform(f, MathTex(r'\Sc{t}=\Sc{t-1}+\vc{t}\kct{t}', font_size=46).move_to(f)), run_time=0.7)
            vals = S.vals.copy()
            for n in range(2):
                vv = rng.uniform(0.3, 3, 4)
                kk = rng.uniform(0, 1, 4)
                kk /= np.linalg.norm(kk)
                vm = Mat(vv[:, None], cell=0.5, vmax=VMAX).move_to(LEFT * 4.6 + DOWN * 0.6)
                km = Mat(kk[None, :], cell=0.5, color=BLUE, vmax=1).move_to(LEFT * 2.4 + UP * 1.0)
                pm = Mat(np.outer(vv, kk), cell=0.5, vmax=VMAX).move_to(LEFT * 2.4 + DOWN * 0.6)
                self.play(FadeIn(vm), FadeIn(km), run_time=0.4)
                self.play(FadeIn(pm), run_time=0.5)
                vals = np.minimum(vals + np.outer(vv, kk), VMAX)
                self.play(pm.animate.move_to(S).set_opacity(0.3), FadeOut(vm), FadeOut(km), run_time=0.7)
                self.play(FadeOut(pm), S.set_values(vals), run_time=0.5)
        self.term(None)
        self.clear_stage()

    # -- size --------------------------------------------------------------
    def size(self):
        ax = Axes(x_range=[0, 160, 32], y_range=[0, 40960, 8192], x_length=8.5, y_length=4.6,
                  axis_config={'color': INK, 'stroke_width': 2, 'include_ticks': True},
                  tips=False).move_to(DOWN * 0.3 + RIGHT * 0.6)
        xl = VGroup(*[Text(str(n), font_size=20, color=MUTED).next_to(ax.c2p(n, 0), DOWN, buff=0.15)
                      for n in (0, 32, 64, 96, 128, 160)])
        yl = VGroup(*[Text(f'{n:,}', font_size=20, color=MUTED).next_to(ax.c2p(0, n), LEFT, buff=0.15)
                      for n in (8192, 16384, 24576, 32768)])
        xt = Text('tokens N', font_size=24, color=MUTED).next_to(ax.x_axis, DOWN, buff=0.55)
        yt = Text('numbers per head', font_size=24, color=MUTED).rotate(PI / 2).next_to(yl, LEFT, buff=0.25)
        state = ax.plot(lambda x: 16384, x_range=[0, 160], color=PURPLE, stroke_width=5)
        st = Text('state: 128 × 128 = 16,384', font_size=26, color=PURPLE).next_to(ax.c2p(128, 16384), DOWN, buff=0.2)
        N = ValueTracker(0)
        cache = always_redraw(lambda: ax.plot(lambda x: 256 * x, x_range=[0, max(N.get_value(), 0.01)],
                                              color=BLUE, stroke_width=5))
        ct = Text('KV cache: 2 · N · 128', font_size=26, color=BLUE).move_to(ax.c2p(42, 33000))
        with self.say('size'):
            self.play(Create(ax), FadeIn(xl), FadeIn(yl), FadeIn(xt), FadeIn(yt), run_time=1.2)
            self.play(Create(state), Write(st), run_time=1.5)
            self.wait(2.5)
            self.add(cache)
            self.play(FadeIn(ct), N.animate.set_value(64), run_time=6, rate_func=linear)
            dot = Dot(ax.c2p(64, 16384), color=RED, radius=0.09)
            dl = Text('64 tokens', font_size=26, color=RED, weight=MEDIUM).next_to(dot, UL, buff=0.12)
            self.play(FadeIn(dot, scale=2), FadeIn(dl), run_time=0.6)
            self.play(N.animate.set_value(160), run_time=3, rate_func=linear)
        with self.say('size2'):
            self.play(*[FadeOut(m) for m in (ax, xl, yl, xt, yt, state, st, cache, ct, dot, dl)], run_time=0.6)
            s = 0.105
            grid = VGroup(Rectangle(width=64 * s, height=32 * s, fill_color=BLUE, fill_opacity=0.25,
                                    stroke_color=BLUE, stroke_width=2))
            for i in range(1, 64):
                grid.add(Line(grid[0].get_corner(DL) + RIGHT * i * s, grid[0].get_corner(UL) + RIGHT * i * s,
                              color=BLUE, stroke_width=0.6, stroke_opacity=0.5))
            for j in range(1, 32):
                grid.add(Line(grid[0].get_corner(DL) + UP * j * s, grid[0].get_corner(DR) + UP * j * s,
                              color=BLUE, stroke_width=0.6, stroke_opacity=0.5))
            grid.move_to(RIGHT * 1.6 + DOWN * 0.3)
            one = Square(s, fill_color=PURPLE, fill_opacity=1, stroke_width=0).move_to(LEFT * 5 + DOWN * 0.3)
            ol = Text('state\n16,384', font_size=22, color=PURPLE, line_spacing=0.8).next_to(one, DOWN, buff=0.35)
            ring = Circle(radius=0.22, color=PURPLE, stroke_width=3).move_to(one)
            gl = Text('KV cache at 128K tokens  =  2,048 states', font_size=28, color=BLUE).next_to(grid, UP, buff=0.3)
            self.play(FadeIn(one, scale=3), FadeIn(ol), Create(ring), run_time=0.6)
            self.play(FadeIn(grid, lag_ratio=0.002), FadeIn(gl), run_time=2.2)
            self.term('decode reads the same fixed matrix', PURPLE)
            self.play(Indicate(one, color=PURPLE, scale_factor=3), run_time=1.2)
        self.term(None)
        self.clear_stage()

    # -- 1.2 ---------------------------------------------------------------
    def remember(self):
        self.header('1.2  What the state remembers')
        f = MathTex(r'\Cv{S}\,\kc{i}=\sum_{j}\vc{j}\,(\kct{j}\kc{i})=', r'\underbrace{\vc{i}}_{\Cg{\text{wanted}}}',
                    r'+', r'\underbrace{\sum_{j\ne i}\vc{j}\,(\kct{j}\kc{i})}_{\Cr{\text{crosstalk}}}',
                    font_size=46).move_to(UP * 0.4)
        with self.say('lookup', pause=0.8):
            self.term('a lookup table: keys → values', PURPLE)
            self.play(Write(f[0]), run_time=2)
            self.wait(4)
            self.play(Write(f[1]), run_time=1)
            self.wait(3)
            self.play(Write(f[2:]), run_time=1.5)
            self.play(Circumscribe(f[3], color=RED), run_time=1.2)
        self.play(f.animate.scale(0.62).to_edge(UP, buff=0.75), run_time=0.7)

        # 2-D picture: keys as arrows, the read as a stacked bar
        plane = NumberPlane(x_range=[-1.4, 1.4, 1], y_range=[-1.4, 1.4, 1], x_length=4.6, y_length=4.6,
                            background_line_style={'stroke_color': LINE, 'stroke_width': 1},
                            axis_config={'stroke_color': MUTED, 'stroke_width': 1.5}).move_to(RIGHT * 3.4 + DOWN * 0.7)
        keys = [(np.array([1, 0]), 3.0, r'\kc{1}', r'\vc{1}=3'), (np.array([0, 1]), 1.0, r'\kc{2}', r'\vc{2}=1')]

        def key_arrow(vec, name, val, where):
            a = Arrow(plane.c2p(0, 0), plane.c2p(*vec), buff=0, color=BLUE, stroke_width=6,
                      max_tip_length_to_length_ratio=0.12)
            lab = MathTex(name, r'\;\mapsto\;', val, font_size=30).next_to(plane.c2p(*vec), where, buff=0.1)
            return VGroup(a, lab)

        base = Line(LEFT * 4.6, LEFT * 1.4, color=INK, stroke_width=2).move_to(LEFT * 3.0 + DOWN * 2.3)
        rl = MathTex(r'\text{read with }\kc{1}', font_size=30).next_to(base, DOWN, buff=0.2)
        unit = 0.62

        def stack(parts):
            g, y = VGroup(), base.get_center()[1]
            for val, col in parts:
                r = Rectangle(width=1.1, height=max(val * unit, 1e-3), fill_color=col, fill_opacity=0.85, stroke_width=0)
                r.move_to([base.get_center()[0], y + val * unit / 2, 0])
                y += val * unit
                g.add(r)
            return g

        with self.say('orth'):
            a1, a2 = key_arrow(keys[0][0], keys[0][2], keys[0][3], RIGHT), key_arrow(keys[1][0], keys[1][2], keys[1][3], LEFT)
            self.play(Create(plane), run_time=0.8)
            self.play(GrowArrow(a1[0]), FadeIn(a1[1]), GrowArrow(a2[0]), FadeIn(a2[1]), run_time=1)
            ra = Elbow(width=0.25, angle=0, color=MUTED).move_to(plane.c2p(0, 0), aligned_edge=DL)
            self.play(Create(ra), run_time=0.4)
            self.play(Create(base), FadeIn(rl), run_time=0.6)
            b1 = stack([(3, GREEN)])
            tot = Text('3.00 = v₁ exactly', font_size=26, color=GREEN).next_to(b1, UP, buff=0.15)
            self.play(GrowFromEdge(b1, DOWN), FadeIn(tot), run_time=1.2)
            self.term('orthogonal keys: a clean read', GREEN)
        with self.say('overlap'):
            k3 = np.array([np.sqrt(.5), np.sqrt(.5)])
            a3 = key_arrow(k3, r'\kc{3}', r'\vc{3}=2', UR)
            self.wait(3.5)
            self.play(GrowArrow(a3[0]), FadeIn(a3[1]), FadeOut(ra), run_time=1)
            proj = DashedLine(plane.c2p(*k3), plane.c2p(k3[0], 0), color=RED, stroke_width=3)
            pl = MathTex(r'\kct{3}\kc{1}=0.71', font_size=28, color=RED).next_to(plane.c2p(k3[0], 0), DOWN, buff=0.15)
            self.play(Create(proj), FadeIn(pl), run_time=0.8)
            b2 = stack([(3, GREEN), (2 * k3[0], RED)])
            tot2 = Text('4.41 = 3 + 0.71 × 2', font_size=26, color=RED).next_to(b2, UP, buff=0.15)
            self.play(ReplacementTransform(b1, b2[0]), GrowFromEdge(b2[1], DOWN), Transform(tot, tot2), run_time=1.2)
            self.term('crosstalk', RED)
        with self.say('gram'):
            self.play(*[FadeOut(m) for m in (plane, a1, a2, a3, proj, pl, b2, tot, base, rl)], run_time=0.6)
            G4 = np.eye(4)
            ks = []
            for j in range(8):
                kk = np.array([np.cos(1.1 * j), np.sin(.7 * j + .3), np.cos(2.3 * j + 1), np.sin(1.9 * j + .5)])
                ks.append(kk / np.linalg.norm(kk))
            ks = np.array(ks)
            G8 = np.abs(ks @ ks.T)
            m4 = Mat(G4, cell=0.5, color=BLUE, vmax=1).move_to(LEFT * 3.3 + DOWN * 0.6)
            m8 = Mat(G8, cell=0.5, color=BLUE, vmax=1).move_to(RIGHT * 2.9 + DOWN * 0.6)
            l4 = Text('4 keys in 4 dimensions', font_size=26).next_to(m4, UP, buff=0.3)
            l8 = Text('8 keys in 4 dimensions', font_size=26).next_to(m8, UP, buff=0.3)
            c4 = MathTex(r'|\kct{i}\kc{j}|', font_size=30).next_to(m4, DOWN, buff=0.3)
            self.play(FadeIn(l4), FadeIn(c4), LaggedStart(*[FadeIn(c) for c in m4.cells], lag_ratio=0.03), run_time=1.6)
            self.wait(3.5)
            self.play(FadeIn(l8), LaggedStart(*[FadeIn(c) for c in m8.cells], lag_ratio=0.02), run_time=2.4)
            off = VGroup(*[m8.cell(i, j) for i in range(8) for j in range(8) if i != j and G8[i, j] > 0.3])
            self.play(LaggedStart(*[Indicate(c, color=RED, scale_factor=1.15) for c in off], lag_ratio=0.03), run_time=2.5)
        with self.say('capacity'):
            self.play(FadeOut(VGroup(m4, l4, c4)), VGroup(m8, l8).animate.scale(0.75).move_to(LEFT * 4.2 + DOWN * 0.5),
                      run_time=0.8)
            st = Mat(np.full((4, 4), 0.6), cell=0.45).move_to(RIGHT * 0.4 + DOWN * 0.2)
            stl = MathTex(r'd_v\times d_k\text{ state}', font_size=30, color=PURPLE).next_to(st, UP, buff=0.25)
            stc = MathTex(r'\approx d_k\text{ exact pairs}', font_size=32, color=PURPLE).next_to(st, DOWN, buff=0.3)
            recs = VGroup(*[self.record().scale(0.55) for _ in range(12)]).arrange(RIGHT, buff=0.08)
            recs.move_to(RIGHT * 4.6 + DOWN * 0.2)
            rl2 = Text('KV cache', font_size=28, color=BLUE).next_to(recs, UP, buff=0.3)
            rc = MathTex(r'\text{all } N\text{ pairs}', font_size=32, color=BLUE).next_to(recs, DOWN, buff=0.3)
            self.play(FadeIn(st), FadeIn(stl), run_time=0.8)
            self.play(Write(stc), run_time=0.8)
            self.play(LaggedStart(*[FadeIn(r) for r in recs], lag_ratio=0.08), FadeIn(rl2), FadeIn(rc), run_time=1.6)
            self.term('capacity ≈ d_k pairs, whatever the length', PURPLE)
            self.wait(3)
            self.play(Circumscribe(VGroup(st, stc), color=PURPLE), run_time=1.5)
        self.term(None)
        self.clear_stage()

    # -- 1.3 ---------------------------------------------------------------
    def softmax_jobs(self):
        self.header('1.3  What the softmax was doing')
        e5, e3 = np.exp(5), np.exp(3)

        def chart(title, ws, labels, col, x):
            t = Text(title, font_size=30, weight=MEDIUM, color=col)
            g = value_bars(ws, labels, 1.0, [col, col], height=3.6, width=0.9, spacing=1.7, fmt='{:.0%}')
            grp = VGroup(t, g).arrange(DOWN, buff=0.4).move_to(RIGHT * x + DOWN * 0.3)
            t.next_to(g, UP, buff=0.5)
            return t, g

        with self.say('focus'):
            self.term('job 1: sharp focus', INK)
            sc = Text('scores 5 and 3', font_size=28, color=MUTED).to_edge(UP, buff=0.9)
            self.play(FadeIn(sc), run_time=0.5)
            ts, gs = chart('softmax', [e5 / (e5 + e3), e3 / (e5 + e3)], ['score 5', 'score 3'], GREEN, -3.4)
            tl, gl = chart('linear', [5 / 8, 3 / 8], ['score 5', 'score 3'], GOLD, 3.0)
            self.play(FadeIn(ts), LaggedStart(*[GrowFromEdge(b[0], DOWN) for b in gs], lag_ratio=0.3),
                      *[FadeIn(b[1:]) for b in gs], run_time=1.5)
            r1 = MathTex(r'e^5/e^3\approx 7.4\times', font_size=38, color=GREEN).next_to(gs, RIGHT, buff=0.25).shift(UP * 0.8)
            self.wait(2.5)
            self.play(Write(r1), run_time=1)
            self.wait(2.5)
            self.play(FadeIn(tl), LaggedStart(*[GrowFromEdge(b[0], DOWN) for b in gl], lag_ratio=0.3),
                      *[FadeIn(b[1:]) for b in gl], run_time=1.5)
            r2 = MathTex(r'5/3\approx 1.7\times', font_size=38, color=GOLD).next_to(gl, RIGHT, buff=0.25).shift(UP * 0.8)
            self.play(Write(r2), run_time=1)
        with self.say('drown'):
            # add ten weak matches (score 1 each) to both charts
            zs, zl = e5 + e3 + 10 * np.e, 5 + 3 + 10
            self.play(FadeOut(VGroup(r1, r2)), FadeOut(sc), run_time=0.4)
            new = Text('+ ten weak matches, score 1 each', font_size=28, color=MUTED).to_edge(UP, buff=0.9)
            anims = [FadeIn(new)]
            for g, ws, col in ((gs, [e5 / zs, e3 / zs, 10 * np.e / zs], GREEN), (gl, [5 / zl, 3 / zl, 10 / zl], GOLD)):
                extra = VGroup(bar(1, 1, 3.6, 0.9, MUTED), Text('0%', font_size=26).set_opacity(0), Text('10 × score 1', font_size=22))
                extra[0].move_to(g[1][0].get_bottom() + RIGHT * 1.7, aligned_edge=DOWN)
                extra[2].next_to(extra[0], DOWN, buff=0.15)
                g.add(extra)
                for i, w in enumerate(ws):
                    b0 = g[i][0]
                    bottom = b0.get_bottom()
                    tc = (RED if col == GOLD else col) if i == 0 else (MUTED if i == 2 else col)
                    anims.append(b0.animate.stretch_to_fit_height(3.6 * w).move_to(bottom, aligned_edge=DOWN))
                    anims.append(Transform(g[i][1], Text(f'{w:.0%}', font_size=26, color=tc).next_to(bottom + UP * 3.6 * w, UP, buff=0.12)))
                extra[0].stretch_to_fit_height(1e-3).move_to(extra[0].get_bottom(), aligned_edge=DOWN)
                anims.append(FadeIn(extra[2]))
            self.play(*anims, run_time=1.5)
            self.play(Indicate(gl[0], color=RED), run_time=1)
        self.clear_stage()
        with self.say('sum1'):
            self.term('job 2: weights that sum to 1', INK)
            origin = LEFT * 2.5 + DOWN * 1.2
            n = ValueTracker(1)
            arr = always_redraw(lambda: Arrow(origin, origin + np.array([0.9, 0.35, 0]) * n.get_value() ** 0.85,
                                              buff=0, color=GREEN, stroke_width=6))
            nt = always_redraw(lambda: Text(f'{int(n.get_value())} tokens in the state', font_size=26,
                                            color=MUTED).move_to(RIGHT * 3.5 + UP * 1.6))
            lab = MathTex(r'\lVert\oc{t}\rVert\text{ drifts}', font_size=36).move_to(RIGHT * 3.5 + UP * 0.8)
            self.add(arr, nt)
            self.play(FadeIn(lab), n.animate.set_value(6), run_time=5, rate_func=linear)
            warn = Text('unstable training', font_size=28, color=RED, weight=MEDIUM).next_to(lab, DOWN, buff=0.4)
            self.play(FadeIn(warn), run_time=0.5)
        self.clear_stage()
        with self.say('history'):
            self.term('Linear Transformer (2020)', INK)
            l1 = MathTex(r'\oc{t}=\frac{\sum_{j}\phi(\qc{t})^{\top}\phi(\kc{j})\,\vc{j}}{\sum_{j}\phi(\qc{t})^{\top}\phi(\kc{j})}',
                         font_size=44).move_to(UP * 1.1 + LEFT * 2.2)
            ex = MathTex(r'\exp(\qc{}^{\top}\kc{})\;\to\;\phi(\qc{})^{\top}\phi(\kc{})', font_size=40).move_to(UP * 2.4)
            fm = MathTex(r'\phi(x)=\operatorname{elu}(x)+1', font_size=40).next_to(l1, RIGHT, buff=0.8)
            self.play(Write(ex), run_time=1.5)
            self.wait(3)
            self.play(Write(fm), run_time=1.2)
            self.wait(1.5)
            self.play(Write(l1), run_time=1.5)
            den = Line(l1.get_corner(DL) + RIGHT * 1.0 + UP * 0.05, l1.get_corner(DR) + UP * 0.5, color=RED, stroke_width=5)
            dt = Text('normalizer', font_size=24, color=RED).next_to(l1, DOWN, buff=0.15)
            self.play(FadeIn(dt), run_time=0.8)
            self.wait(2.5)
            later = Text('later versions', font_size=28, color=MUTED, weight=MEDIUM).move_to(DOWN * 1.2 + LEFT * 3.5)
            self.play(Create(den), dt.animate.set_color(RED), run_time=0.6)
            l2 = MathTex(r'\lVert\qc{t}\rVert=\lVert\kc{t}\rVert=1,\qquad \oc{t}=\operatorname{Norm}(\Sc{t}\,\qc{t})',
                         font_size=42).next_to(later, RIGHT, buff=0.5)
            self.wait(1.0)
            self.play(FadeIn(later), Write(l2), run_time=1.8)
        self.clear_stage()
        with self.say('different'):
            self.term('a different model, not a faster softmax', RED)
            a = panel(4.0, 1.4).move_to(LEFT * 3.5)
            at = Text('trained with softmax', font_size=28).move_to(a)
            b = panel(4.0, 1.4, color=PURPLE).move_to(RIGHT * 3.5)
            bt = Text('linear attention', font_size=28, color=PURPLE).move_to(b)
            ar = arrow(a.get_right(), b.get_left(), color=MUTED)
            x = Cross(Square(0.5).move_to(ar), stroke_color=RED, stroke_width=6)
            self.play(FadeIn(a), FadeIn(at), FadeIn(b), FadeIn(bt), run_time=0.8)
            self.play(GrowArrow(ar), run_time=0.6)
            self.play(Create(x), run_time=0.5)
            tr = Text('train it this way from the start', font_size=28, color=PURPLE).next_to(b, DOWN, buff=0.4)
            self.wait(2)
            self.play(FadeIn(tr), run_time=0.6)
        self.term(None)
        self.clear_stage()

    # -- 1.4 ---------------------------------------------------------------
    def three_forms(self):
        self.header('1.4  Three ways to compute the same thing')
        n, cell = 12, 0.36

        def score_grid(fill=lambda i, j: BLUE):
            g = VGroup()
            for i in range(n):
                for j in range(n):
                    sq = Square(cell, stroke_color=LINE, stroke_width=1)
                    if j <= i:
                        sq.set_fill(fill(i, j), 0.55)
                    else:
                        sq.set_fill(LINE, 0.25)
                    g.add(sq.move_to([j * cell, -i * cell, 0]))
            return g.center()

        with self.say('parallel', pause=0.6):
            g = score_grid().move_to(LEFT * 3 + DOWN * 0.5)
            gl = MathTex(r'Q\Ck{K}^{\top}\odot M', font_size=34).next_to(g, UP, buff=0.25)
            self.play(FadeIn(g, lag_ratio=0.01), FadeIn(gl), run_time=1.5)
            f = MathTex(r'O=(\Cq{Q}\Ck{K}^{\top}\odot M)\Cv{V}', font_size=46).move_to(RIGHT * 3.2 + UP * 1)
            mask = Text('causal mask M', font_size=26, color=MUTED).next_to(f, DOWN, buff=0.4)
            self.play(Write(f), FadeIn(mask), run_time=1.5)
            self.wait(3)
            self.term('parallel form', BLUE)
            c = MathTex(r'\text{cost}\propto N^2 d', font_size=42, color=RED).next_to(mask, DOWN, buff=0.6)
            self.wait(4)
            self.play(Write(c), run_time=1)
        with self.say('recurrent'):
            self.play(FadeOut(VGroup(f, mask, c, g, gl)), run_time=0.6)
            chain = VGroup(*[Square(0.55, fill_color=PURPLE, fill_opacity=0.2, stroke_color=PURPLE, stroke_width=2)
                             for _ in range(8)]).arrange(RIGHT, buff=0.42).move_to(RIGHT * 2.7 + UP * 0.6)
            arrs = VGroup(*[Arrow(chain[i].get_right(), chain[i + 1].get_left(), color=PURPLE, buff=0.04, stroke_width=4,
                                  max_tip_length_to_length_ratio=0.45) for i in range(7)])
            lab = VGroup(*[MathTex(fr'\Sc{{{i + 1}}}', font_size=26).next_to(chain[i], DOWN, buff=0.12) for i in range(8)])
            self.play(FadeIn(chain), FadeIn(arrs), FadeIn(lab), run_time=0.8)
            self.term('recurrent form', PURPLE)
            for i in range(8):
                self.play(chain[i].animate.set_fill(PURPLE, 0.85), run_time=0.35)
            c = MathTex(r'\text{cost}\propto N d^2', font_size=40, color=GREEN).next_to(chain, DOWN, buff=1.0)
            w = Text('but each step waits for the last', font_size=28, color=RED).next_to(c, DOWN, buff=0.3)
            self.play(Write(c), run_time=0.8)
            self.play(FadeIn(w), run_time=0.6)
        with self.say('chunk'):
            g = score_grid().move_to(LEFT * 3.4 + DOWN * 0.5)
            self.play(FadeOut(VGroup(chain, arrs, lab, c, w)), FadeIn(g), run_time=0.8)
            self.term('chunkwise form', TEAL)
            C = 4
            lines = VGroup()
            for b in range(1, n // C):
                x = g.get_left()[0] + b * C * cell
                y = g.get_top()[1] - b * C * cell
                lines.add(Line([x, g.get_top()[1], 0], [x, g.get_bottom()[1], 0], color=INK, stroke_width=3))
                lines.add(Line([g.get_left()[0], y, 0], [g.get_right()[0], y, 0], color=INK, stroke_width=3))
            ct = Text('chunks of C tokens', font_size=26, color=MUTED).next_to(g, UP, buff=0.25)
            self.play(Create(lines), FadeIn(ct), run_time=1)
            self.wait(2.5)
            diag = VGroup(*[g[i * n + j] for i in range(n) for j in range(n) if j <= i and i // C == j // C])
            offd = VGroup(*[g[i * n + j] for i in range(n) for j in range(n) if i // C > j // C])
            s1 = Text('1  inside a chunk: parallel, C × C', font_size=25, color=GREEN)
            s2 = Text('2  earlier chunks: read the carried state', font_size=25, color=PURPLE)
            s3 = Text('3  update the state once per chunk', font_size=25, color=TEAL)
            steps = VGroup(s1, s2, s3).arrange(DOWN, aligned_edge=LEFT, buff=0.35)
            steps.move_to(RIGHT * 0.5 + UP * 0.6, aligned_edge=LEFT)
            self.wait(2)
            self.play(diag.animate.set_fill(GREEN, 0.75), FadeIn(s1), run_time=1)
            self.wait(2.5)
            self.play(offd.animate.set_fill(PURPLE, 0.7), FadeIn(s2), run_time=1)
            fq = MathTex(r'\Cq{Q}_{[c]}\Sc{[c-1]}^{\top}', font_size=34).next_to(steps, DOWN, buff=0.5)
            self.play(Write(fq), run_time=0.8)
            self.wait(2.5)
            states = VGroup(*[Square(0.4, fill_color=PURPLE, fill_opacity=0.85, stroke_width=0)
                              .move_to([g.get_right()[0] + 0.4, g.get_top()[1] - (b + 1) * C * cell, 0]) for b in range(n // C - 1)])
            fu = MathTex(r'\Sc{[c]}=\Sc{[c-1]}+\Cv{V}_{[c]}^{\top}\Ck{K}_{[c]}', font_size=34).next_to(fq, DOWN, buff=0.3)
            self.play(FadeIn(s3), LaggedStart(*[FadeIn(s, scale=0.5) for s in states], lag_ratio=0.4), Write(fu), run_time=1.5)
        with self.say('same'):
            self.clear_stage(run_time=0.5)
            o_par, o_rec, o_chk = self.three_ways()
            rows = VGroup()
            for name, col, vals in (('parallel', BLUE, o_par), ('recurrent', PURPLE, o_rec), ('chunkwise', TEAL, o_chk)):
                lab = Text(name, font_size=28, color=col, weight=MEDIUM)
                nums = VGroup(*[Text(f'{x:+.2f}', font_size=26) for x in vals]).arrange(RIGHT, buff=0.3)
                rows.add(VGroup(lab, nums))
            for i, r in enumerate(rows):
                r[0].move_to([-3.4, 1.0 - i * 0.9, 0], aligned_edge=RIGHT)
                r[1].next_to(r[0], RIGHT, buff=0.5)
            head = MathTex(r'\oc{1}\;\dots\;\oc{8}\quad(\text{first channel})', font_size=32).next_to(rows, UP, buff=0.5)
            self.play(FadeIn(head), run_time=0.5)
            for r in rows:
                self.play(FadeIn(r[0]), LaggedStart(*[FadeIn(x) for x in r[1]], lag_ratio=0.1), run_time=1.1)
            ok = Text('identical', font_size=30, color=GREEN, weight=SEMIBOLD).next_to(rows, DOWN, buff=0.5)
            self.play(FadeIn(ok, scale=1.2), run_time=0.5)
        with self.say('cost'):
            self.clear_stage(run_time=0.5)
            self.term('d = 128, C = 64, N = 32K, one head', INK)
            data = [('chunkwise', TEAL, 98304, '98,304', 512, '512'),
                    ('recurrent', PURPLE, 65536, '65,536', 32768, '32,768'),
                    ('quadratic', BLUE, 8.4e6, '8.4M', None, 'fully parallel')]
            h1 = Text('FLOPs per token', font_size=26, color=MUTED).move_to(LEFT * 0.6 + UP * 2.3)
            h2 = Text('sequential steps', font_size=26, color=MUTED).move_to(RIGHT * 4.4 + UP * 2.3)
            self.play(FadeIn(h1), FadeIn(h2), run_time=0.5)
            lo, hi = 4.0, 7.0  # log10 axis for FLOPs bars
            for r, (name, col, fl, fls, st, sts) in enumerate(data):
                y = 1.3 - r * 1.25
                lab = Text(name, font_size=28, color=col, weight=MEDIUM).move_to([-5.2, y, 0])
                w = 4.0 * (np.log10(fl) - lo) / (hi - lo)
                b1 = Rectangle(width=w, height=0.5, fill_color=col, fill_opacity=0.85, stroke_width=0)
                b1.move_to([-2.6, y, 0], aligned_edge=LEFT)
                t1 = Text(fls, font_size=24).next_to(b1, RIGHT, buff=0.15)
                if st:
                    w2 = 2.6 * np.log10(st) / np.log10(32768)
                    b2 = Rectangle(width=w2, height=0.5, fill_color=col, fill_opacity=0.85, stroke_width=0)
                    b2.move_to([2.9, y, 0], aligned_edge=LEFT)
                    t2 = Text(sts, font_size=24).next_to(b2, RIGHT, buff=0.15)
                else:
                    b2 = VMobject()
                    t2 = Text(sts, font_size=24, color=MUTED).move_to([2.9, y, 0], aligned_edge=LEFT)
                self.play(FadeIn(lab), GrowFromEdge(b1, LEFT), FadeIn(t1), GrowFromEdge(b2, LEFT) if st else FadeIn(b2),
                          FadeIn(t2), run_time=1.0)
                self.wait(4.2 if r < 2 else 1.5)
            note = Text('log scale', font_size=20, color=MUTED).next_to(h1, DOWN, buff=0.1)
            dec = Text('decode: the recurrent loop, one token at a time', font_size=28, color=PURPLE).move_to(DOWN * 2.6)
            self.play(FadeIn(note), FadeIn(dec), run_time=0.8)
        self.term(None)
        self.clear_stage()

    def three_ways(self):
        """The same causal linear attention computed three ways (first output channel)."""
        rng = np.random.default_rng(7)
        N, d, C = 8, 4, 4
        Q, K = rng.normal(size=(N, d)), rng.normal(size=(N, d))
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        K /= np.linalg.norm(K, axis=1, keepdims=True)
        V = rng.normal(size=(N, d))
        M = np.tril(np.ones((N, N)))
        par = ((Q @ K.T) * M) @ V
        S, rec = np.zeros((d, d)), []
        for t in range(N):
            S = S + np.outer(V[t], K[t])
            rec.append(S @ Q[t])
        rec = np.array(rec)
        S, chk = np.zeros((d, d)), []
        for c in range(0, N, C):
            q, k, v = Q[c:c + C], K[c:c + C], V[c:c + C]
            chk.append(((q @ k.T) * np.tril(np.ones((C, C)))) @ v + q @ S.T)
            S = S + v.T @ k
        chk = np.vstack(chk)
        assert np.allclose(par, rec) and np.allclose(par, chk)
        return par[:, 0], rec[:, 0], chk[:, 0]

    def recap(self):
        self.header('Chapter 1 · Linear attention')
        card, lo = self.recap_card(r'\Sc{t}=\Sc{t-1}+\vc{t}\kct{t},\qquad \oc{t}=\Sc{t}\,\qc{t}',
                                   'The whole history folds into one fixed matrix per head.',
                                   'the state only adds; nothing is forgotten or overwritten.')
        with self.say('recap', pause=0.6):
            self.play(FadeIn(card, shift=0.2 * UP), run_time=1)
        with self.say('leftover'):
            self.play(FadeIn(lo, shift=0.1 * UP), run_time=0.8)
        self.wait(1)
