"""Chapter 3 · Correcting: the delta rule (Fixed-Size States §3)."""
from common import *


class Orders(VGroup):
    """The doc's toy state: one key direction per order, the bar is the stored ship day."""
    def __init__(self, a, b, center=DOWN * 2.3, gap=2.6, u=0.42):
        super().__init__()
        self.U = u
        self.base_y = center[1]
        self.xs = [center[0] - gap / 2, center[0] + gap / 2]
        self.axis = Line([self.xs[0] - 1.2, self.base_y, 0], [self.xs[1] + 1.2, self.base_y, 0], color=INK, stroke_width=2)
        self.ticks = VGroup(*[Text(str(d), font_size=18, color=MUTED).move_to([self.xs[0] - 1.45, self.base_y + d * self.U, 0])
                              for d in range(0, 8)])
        self.grid = VGroup(*[DashedLine([self.xs[0] - 1.2, self.base_y + d * self.U, 0], [self.xs[1] + 1.2, self.base_y + d * self.U, 0],
                                        color=LINE, stroke_width=1, dash_length=0.08) for d in range(1, 8)])
        self.names = VGroup(*[Text(n, font_size=24, color=BLUE).next_to([x, self.base_y, 0], DOWN, buff=0.18)
                              for n, x in zip(('order 7314', 'order 2208'), self.xs)])
        self.vals = [a, b]
        self.bars = VGroup(self.bar(0, a), self.bar(1, b))
        self.labs = VGroup(self.lab(0, a), self.lab(1, b))
        self.add(self.grid, self.axis, self.ticks, self.names, self.bars, self.labs)

    def top(self, i, v):
        return np.array([self.xs[i], self.base_y + v * self.U, 0])

    def bar(self, i, v, color=PURPLE, op=0.85):
        return Rectangle(width=1.1, height=max(v * self.U, 1e-3), fill_color=color, fill_opacity=op,
                         stroke_width=0).move_to([self.xs[i], self.base_y, 0], aligned_edge=DOWN)

    def lab(self, i, v, color=PURPLE):
        return Text(f'{v:g}', font_size=28, color=color, weight=MEDIUM).next_to(self.top(i, v), UP, buff=0.12)

    def to(self, i, v, color=PURPLE, lab_color=None):
        self.vals[i] = v
        return [Transform(self.bars[i], self.bar(i, v, color)),
                Transform(self.labs[i], self.lab(i, v, lab_color or color))]


class Ch3Delta(VoiceScene):
    SCRIPT = 'ch3_delta'

    def construct(self):
        self.problem()
        self.rule()
        self.toy()
        self.geometry()
        self.gated_kda()
        self.sum_form()
        self.tracking()
        self.training()
        self.recap()

    def problem(self):
        with self.say('intro'):
            card = self.title_card(3, 'Correcting', 'the delta rule')
            self.play(FadeIn(card, shift=0.2 * UP), run_time=1)
            self.wait(2.4)
            self.play(FadeOut(card), run_time=0.6)
            self.header('3.1  Read, compare, overwrite')
            o = Orders(2, 4, center=DOWN * 1.6)
            stale = Text('stale', font_size=24, color=RED).next_to(o.labs[0], UP, buff=0.1)
            good = Text('correct', font_size=24, color=GREEN).next_to(o.labs[1], UP, buff=0.1)
            self.play(FadeIn(o), run_time=0.8)
            self.play(FadeIn(stale), FadeIn(good), run_time=0.5)
            self.wait(1.5)
            self.term('decay fades every entry', RED)
            self.play(*o.to(0, 1.6), *o.to(1, 3.2), stale.animate.shift(DOWN * 0.17), good.animate.shift(DOWN * 0.34), run_time=1.5)
            self.play(*o.to(0, 1.28), *o.to(1, 2.56), stale.animate.shift(DOWN * 0.13), good.animate.shift(DOWN * 0.27), run_time=1.5)
            self.wait(1)
            fix = Text('check the table before writing', font_size=30, color=PURPLE).move_to(UP * 2.2)
            self.play(FadeIn(fix), run_time=0.6)
        self.term(None)
        self.clear_stage()

    def rule(self):
        s1 = MathTex(r'\text{1. read: }\;\Cg{\hat v_t}=\Sc{t-1}\kc{t}', font_size=40)
        s2 = MathTex(r'\text{2. compare: }\;\Cr{e_t}=\vc{t}-\Cg{\hat v_t}', font_size=40)
        s3 = MathTex(r'\text{3. write: }\;\Sc{t}=\Sc{t-1}+\beta_t\,\Cr{e_t}\,\kct{t}', font_size=40)
        steps = VGroup(s1, s2, s3).arrange(DOWN, aligned_edge=LEFT, buff=0.55).move_to(LEFT * 3.0 + UP * 0.3)
        O = RIGHT * 2.3 + DOWN * 1.8
        vh, vt = np.array([2.2, 0.9, 0]), np.array([3.2, 2.9, 0])
        with self.say('steps'):
            self.term('DeltaNet (2021; parallel training 2024)', PURPLE)
            ax = VGroup(Line(O, O + RIGHT * 4.2, color=LINE), Line(O, O + UP * 3.6, color=LINE))
            self.play(FadeIn(ax), run_time=0.4)
            self.wait(2.5)
            self.play(Write(s1), run_time=1.2)
            a1 = Arrow(O, O + vh, buff=0, color=GREEN, stroke_width=6)
            l1 = MathTex(r'\Cg{\hat v_t}', font_size=36).next_to(a1.get_end(), RIGHT, buff=0.1)
            self.play(GrowArrow(a1), FadeIn(l1), run_time=1)
            self.wait(2.5)
            self.play(Write(s2), run_time=1.2)
            a2 = Arrow(O, O + vt, buff=0, color=PURPLE, stroke_width=6)
            l2 = MathTex(r'\vc{t}', font_size=36).next_to(a2.get_end(), UP, buff=0.1)
            e = Arrow(O + vh, O + vt, buff=0, color=RED, stroke_width=5)
            le = MathTex(r'\Cr{e_t}', font_size=36).next_to(e.get_center(), RIGHT, buff=0.15)
            self.play(GrowArrow(a2), FadeIn(l2), run_time=0.8)
            self.play(GrowArrow(e), FadeIn(le), run_time=0.8)
            self.wait(3)
            self.play(Write(s3), run_time=1.4)
            new = O + vh + 0.5 * (vt - vh)
            a3 = Arrow(O, new, buff=0, color=GREEN, stroke_width=6)
            l3 = MathTex(r'\beta=0.5', font_size=32, color=GREEN).move_to(new + RIGHT * 0.95 + DOWN * 0.3)
            self.play(Transform(a1, a3), FadeOut(l1), FadeIn(l3), run_time=1.2)
        f = MathTex(r'\Sc{t}=\Sc{t-1}+\beta_t(\vc{t}-\Sc{t-1}\kc{t})\kct{t}', r'=\Sc{t-1}\big(I-\beta_t\kc{t}\kct{t}\big)+\beta_t\vc{t}\kct{t}',
                    font_size=42).to_edge(UP, buff=0.85)
        with self.say('formula', pause=0.8):
            self.play(FadeOut(steps), run_time=0.4)
            self.play(Write(f[0]), run_time=1.4)
            self.play(Write(f[1]), run_time=1.6)
            self.wait(1.5)
            same = Arrow(O, O + vt, buff=0, color=GREEN, stroke_width=6)
            self.play(Transform(a1, same), FadeOut(e), FadeOut(le), FadeOut(l3), run_time=1)
            z = MathTex(r'\Cr{e_t}=0\;\Rightarrow\;\text{no change}', font_size=36).move_to(LEFT * 3.4 + DOWN * 0.3)
            self.play(Write(z), run_time=1)
            self.wait(1.5)
            r = Text('stale value → replaced', font_size=30, color=PURPLE).next_to(z, DOWN, buff=0.5)
            self.play(FadeIn(r), run_time=0.6)
        self.term(None)
        self.clear_stage()

    def toy(self):
        o = Orders(2, 4, center=LEFT * 2.2 + DOWN * 2.2, gap=3.6)
        with self.say('toy'):
            self.term('toy: order number → ship day', INK)
            self.play(FadeIn(o), run_time=1)
            days = VGroup(Text('day 2 = Tuesday', font_size=24, color=MUTED).next_to(o.labs[0], UP, buff=0.12))
            self.wait(4.5)
            self.play(FadeIn(days), run_time=0.5)
            self.wait(4)
            ft = Text('new fact:\norder 7314 → day 5 (Friday)', font_size=26, color=PURPLE, line_spacing=0.9)
            fact = VGroup(panel(ft.width + 0.6, ft.height + 0.5, color=PURPLE), ft)
            ft.move_to(fact[0])
            fact.move_to(RIGHT * 3.8 + UP * 1.0)
            self.play(FadeIn(fact, shift=LEFT * 0.3), run_time=0.8)
            target = DashedLine(o.top(0, 5) + LEFT * 0.75, o.top(0, 5) + RIGHT * 0.75, color=PURPLE, stroke_width=4)
            tl = Text('5', font_size=24, color=PURPLE).next_to(target, LEFT, buff=0.1)
            self.play(Create(target), FadeIn(tl), FadeOut(days), run_time=0.8)
        with self.say('toyla'):
            self.term('linear attention: add', RED)
            add = Rectangle(width=1.1, height=5 * o.U, fill_color=RED, fill_opacity=0.6, stroke_width=0)
            add.move_to(o.top(0, 2), aligned_edge=DOWN)
            l7 = Text('7: a mix', font_size=28, color=RED, weight=MEDIUM).next_to(add, UP, buff=0.12)
            self.play(FadeIn(add, shift=DOWN * 0.4), FadeOut(o.labs[0]), run_time=0.8)
            self.play(FadeIn(l7), run_time=0.5)
        with self.say('toydelta'):
            self.term('DeltaNet: read, compare, write the error', PURPLE)
            self.play(FadeOut(add), FadeOut(l7), FadeIn(o.labs[0]), run_time=0.6)
            rd = Text('read 2', font_size=26, color=GREEN).next_to(o.top(0, 2) + RIGHT * 0.6, RIGHT, buff=0.1).shift(DOWN * 0.35)
            self.play(Indicate(o.bars[0], color=GREEN), FadeIn(rd), run_time=0.8)
            br = BraceBetweenPoints(o.top(0, 2) + RIGHT * 0.6, o.top(0, 5) + RIGHT * 0.6, RIGHT, color=RED)
            bl = Text('error 3', font_size=26, color=RED).next_to(br, RIGHT, buff=0.1)
            self.play(GrowFromCenter(br), FadeIn(bl), run_time=0.8)
            b1 = MathTex(r'\beta=1', font_size=34).move_to(RIGHT * 3.8 + DOWN * 0.6)
            self.play(FadeIn(b1), *o.to(0, 5), run_time=1.2)
            self.wait(1.2)
            b2 = MathTex(r'\beta=0.5', font_size=34).move_to(b1)
            self.play(Transform(b1, b2), *o.to(0, 3.5), run_time=1.2)
            self.wait(1)
            keep = SurroundingRectangle(VGroup(o.bars[1], o.labs[1]), color=GREEN, buff=0.1)
            kl = Text('2208 stays at 4', font_size=26, color=GREEN).next_to(keep, RIGHT, buff=0.2)
            self.play(Create(keep), FadeIn(kl), run_time=0.8)
        with self.say('leak'):
            self.play(FadeOut(VGroup(br, bl, rd, b1, keep, kl, target, tl, fact)), *o.to(0, 2), run_time=0.6)
            o2 = Orders(2, 4, center=LEFT * 4.4 + DOWN * 2.0, gap=1.9, u=0.32)
            self.play(FadeOut(o), FadeIn(o2), run_time=0.6)
            o = o2
            plane = NumberPlane(x_range=[-0.3, 1.3, 1], y_range=[-0.3, 1.3, 1], x_length=3.6, y_length=3.6,
                                background_line_style={'stroke_color': LINE, 'stroke_width': 1},
                                axis_config={'stroke_color': MUTED, 'stroke_width': 1.5}).move_to(RIGHT * 0.4 + DOWN * 0.5)
            k1 = Arrow(plane.c2p(0, 0), plane.c2p(1, 0), buff=0, color=BLUE, stroke_width=5)
            k2 = Arrow(plane.c2p(0, 0), plane.c2p(0, 1), buff=0, color=BLUE, stroke_width=5)
            q = Arrow(plane.c2p(0, 0), plane.c2p(0.707, 0.707), buff=0, color=GOLD, stroke_width=6)
            kl1 = Text('7314', font_size=22, color=BLUE).next_to(plane.c2p(1, 0), DOWN, buff=0.15)
            kl2 = Text('2208', font_size=22, color=BLUE).next_to(plane.c2p(0, 1), LEFT, buff=0.15)
            ql = Text('query halfway', font_size=22, color=GOLD).next_to(plane.c2p(0.707, 0.707), UP, buff=0.12)
            self.play(Create(plane), GrowArrow(k1), GrowArrow(k2), FadeIn(kl1), FadeIn(kl2), run_time=1)
            self.play(GrowArrow(q), FadeIn(ql), run_time=0.8)
            before = MathTex(r'0.71\cdot2+0.71\cdot4=4.24', font_size=34).move_to(RIGHT * 4.6 + UP * 0.7)
            self.play(Write(before), run_time=1.2)
            self.play(*o.to(0, 5), run_time=0.8)
            after = MathTex(r'0.71\cdot5+0.71\cdot4=6.36', font_size=34, color=RED).next_to(before, DOWN, buff=0.5)
            self.play(Write(after), run_time=1.2)
            self.term('the edit leaks into overlapping queries', RED)
        self.term(None)
        self.clear_stage()

    def geometry(self):
        self.header('3.2  The geometry of I − βkkᵀ')
        f = MathTex(r'\operatorname{eig}\big(I-\beta\kc{}\kct{}\big)=\{1,\dots,1,\;1-\beta\}', font_size=42).to_edge(UP, buff=0.85)
        plane = NumberPlane(x_range=[-1.3, 1.3, 0.5], y_range=[-1.3, 1.3, 0.5], x_length=5.2, y_length=5.2,
                            background_line_style={'stroke_color': LINE, 'stroke_width': 1},
                            axis_config={'stroke_color': MUTED, 'stroke_width': 1.5}).move_to(LEFT * 2.6 + DOWN * 0.7)
        beta = ValueTracker(0)
        x0 = np.array([0.8, 0.6])
        kdir = Arrow(plane.c2p(0, 0), plane.c2p(1.2, 0), buff=0, color=BLUE, stroke_width=4)
        kl = MathTex(r'\kc{}', font_size=36).next_to(plane.c2p(1.2, 0), DOWN, buff=0.12)
        ghost = DashedLine(plane.c2p(0, 0), plane.c2p(*x0), color=MUTED, stroke_width=3)
        circ = always_redraw(lambda: ParametricFunction(
            lambda t: plane.c2p((1 - beta.get_value()) * np.cos(t), np.sin(t)), t_range=[0, TAU], color=TEAL, stroke_width=3))
        vec = always_redraw(lambda: Arrow(plane.c2p(0, 0), plane.c2p((1 - beta.get_value()) * x0[0] + 1e-4, x0[1]), buff=0,
                                          color=PURPLE, stroke_width=6))
        with self.say('geom', pause=0.6):
            self.play(Write(f), run_time=1.5)
            self.play(Create(plane), GrowArrow(kdir), FadeIn(kl), run_time=1)
            self.add(circ, vec)
            self.play(FadeIn(ghost), run_time=0.4)
            xl = MathTex(r'x=(0.8,\,0.6)', font_size=32, color=PURPLE).next_to(plane.c2p(*x0), UR, buff=0.1)
            self.play(FadeIn(xl), run_time=0.5)
            other = Text('other directions: untouched', font_size=26).move_to(RIGHT * 3.6 + UP * 1.9)
            along = MathTex(r'\text{along }\kc{}:\ \times(1-\beta)', font_size=36).next_to(other, DOWN, buff=0.3)
            self.wait(3)
            self.play(FadeIn(other), run_time=0.6)
            self.wait(1.5)
            self.play(Write(along), run_time=1)
        # eigenvalue gauge
        line = NumberLine(x_range=[-1, 1, 0.5], length=5, color=INK, include_numbers=False).move_to(RIGHT * 3.6 + DOWN * 1.2)
        nums = VGroup(*[Text(f'{v:g}', font_size=20, color=MUTED).next_to(line.n2p(v), DOWN, buff=0.15) for v in (-1, 0, 1)])
        gl = MathTex(r'1-\beta', font_size=32).next_to(line, UP, buff=0.3)
        gdot = always_redraw(lambda: Dot(line.n2p(1 - beta.get_value()), color=RED, radius=0.1))
        bt = always_redraw(lambda: MathTex(fr'\beta={beta.get_value():.1f}', font_size=36).move_to(RIGHT * 3.6 + DOWN * 2.6))
        with self.say('geom2'):
            self.play(FadeOut(xl), Create(line), FadeIn(nums), FadeIn(gl), run_time=0.8)
            self.add(gdot, bt)
            state = Text('', font_size=28)
            for b, name, col in ((0.5, 'shrink', TEAL), (1.0, 'erase: a projection', RED), (2.0, 'flip: a reflection', PURPLE)):
                self.play(beta.animate.set_value(b), run_time=2.0)
                new = Text(name, font_size=30, color=col, weight=MEDIUM).move_to(RIGHT * 3.6 + UP * 0.45)
                self.play(Transform(state, new) if len(state) else FadeIn(new), run_time=0.5)
                if not len(state):
                    state = new
                self.wait(2.3)
            self.term('β = 2: eigenvalue −1, needed for state tracking', PURPLE)
        self.term(None)
        self.clear_stage()

    def gated_kda(self):
        self.header('3.3  Gated DeltaNet: decay first, then correct')
        f = MathTex(r'\Sc{t}=\Ct{\alpha_t}\,\Sc{t-1}\big(I-\beta_t\kc{t}\kct{t}\big)+\beta_t\vc{t}\kct{t}', font_size=44).to_edge(UP, buff=0.85)
        with self.say('gated', pause=0.6):
            self.play(Write(f), run_time=1.6)
            rows = [('Mamba-2', True, False), ('DeltaNet', False, True), ('Gated DeltaNet', True, True)]
            h1 = Text('fade a whole head (α)', font_size=24, color=TEAL)
            h2 = Text('edit one association (β)', font_size=24, color=PURPLE)
            hdr = VGroup(h1, h2).arrange(RIGHT, buff=1.0).move_to(RIGHT * 1.6 + UP * 1.3)
            self.play(FadeIn(hdr), run_time=0.6)
            marks = VGroup()
            for i, (name, a, b) in enumerate(rows):
                y = 0.3 - i * 0.9
                n = Text(name, font_size=28, weight=MEDIUM).move_to([-2.4, y, 0], aligned_edge=RIGHT)
                m1 = (Text('✓', font_size=34, color=GREEN) if a else Text('–', font_size=34, color=MUTED)).move_to([h1.get_x(), y, 0])
                m2 = (Text('✓', font_size=34, color=GREEN) if b else Text('–', font_size=34, color=MUTED)).move_to([h2.get_x(), y, 0])
                marks.add(VGroup(n, m1, m2))
            self.wait(2)
            for m in marks:
                self.play(FadeIn(m), run_time=0.7)
                self.wait(1.0)
        with self.say('gatedtoy'):
            self.clear_stage(keep=[f], run_time=0.5)
            o = Orders(2, 4, center=LEFT * 2.6 + DOWN * 2.4)
            par = MathTex(r'\alpha=0.8,\ \beta=0.5', font_size=36).move_to(RIGHT * 3.6 + UP * 1.4)
            self.play(FadeIn(o), FadeIn(par), run_time=0.8)
            self.wait(2.5)
            s1 = Text('1  fade:  (2, 4) → (1.6, 3.2)', font_size=26, color=TEAL)
            s2 = Text('2  read 1.6,  error 5 − 1.6 = 3.4', font_size=26, color=RED)
            s3 = Text('3  write 0.5 × 3.4 = 1.7  →  3.3', font_size=26, color=PURPLE)
            steps = VGroup(s1, s2, s3).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(RIGHT * 3.4 + DOWN * 0.3)
            self.wait(2)
            self.play(FadeIn(s1), *o.to(0, 1.6), *o.to(1, 3.2), run_time=1.5)
            self.wait(1.5)
            tgt = DashedLine(o.top(0, 5) + LEFT * 0.75, o.top(0, 5) + RIGHT * 0.75, color=PURPLE, stroke_width=4)
            self.play(FadeIn(s2), Create(tgt), Indicate(o.bars[0], color=GREEN), run_time=1.2)
            self.wait(2)
            self.play(FadeIn(s3), *o.to(0, 3.3), run_time=1.5)
            self.wait(3)
            hit = SurroundingRectangle(VGroup(o.bars[1], o.labs[1]), color=RED, buff=0.1)
            hl = Text('2208 faded too: 4 → 3.2', font_size=26, color=RED).next_to(steps, DOWN, buff=0.5, aligned_edge=LEFT)
            self.play(Create(hit), FadeIn(hl), *o.to(1, 3.2, RED, RED), run_time=1)
        self.header('3.4  KDA: per-channel decay inside the delta rule')
        g = MathTex(r'\Sc{t}=\Sc{t-1}\Ct{\Diag(\alpha_t)}\big(I-\beta_t\kc{t}\kct{t}\big)+\beta_t\vc{t}\kct{t}', font_size=42).move_to(f)
        with self.say('kda', pause=0.6):
            self.play(FadeOut(VGroup(steps, hl, hit, tgt)), Transform(f, g), *o.to(0, 2), *o.to(1, 4), run_time=1.2)
            self.term('KDA, Kimi Delta Attention (2025)', TEAL)
            par2 = MathTex(r'\alpha=(0.8,\,1),\ \beta=0.5', font_size=36).move_to(par)
            self.play(Transform(par, par2), run_time=0.8)
            self.wait(5)
            k1 = Text('1  fade only channel 7314:  (1.6, 4)', font_size=26, color=TEAL)
            k2 = Text('2  read 1.6, error 3.4, write 1.7  →  3.3', font_size=26, color=PURPLE)
            ks = VGroup(k1, k2).arrange(DOWN, aligned_edge=LEFT, buff=0.35).move_to(RIGHT * 3.4 + DOWN * 0.1)
            self.play(FadeIn(k1), *o.to(0, 1.6), run_time=1.2)
            self.wait(2.5)
            self.play(FadeIn(k2), *o.to(0, 3.3), run_time=1.2)
            keep = SurroundingRectangle(VGroup(o.bars[1], o.labs[1]), color=GREEN, buff=0.1)
            kl = Text('2208 kept at 4', font_size=26, color=GREEN).next_to(ks, DOWN, buff=0.5, aligned_edge=LEFT)
            self.play(Create(keep), FadeIn(kl), *o.to(1, 4, PURPLE, GREEN), run_time=1)
        with self.say('kdafast'):
            self.clear_stage(keep=[f], run_time=0.5)
            t1 = MathTex(r'\text{KDA: }\;\Ct{\Diag(\alpha_t)}\big(I-\beta_t\,', r'\kc{t}', r'\kct{t}', r'\big)', font_size=40).move_to(UP * 0.8)
            t2 = MathTex(r'\text{general: }\;\Ct{\Diag(w)}-a\,b^{\top}', font_size=40).move_to(DOWN * 0.6)
            self.play(Write(t1), run_time=1.2)
            tied = VGroup(SurroundingRectangle(t1[1], color=BLUE, buff=0.05), SurroundingRectangle(t1[2], color=BLUE, buff=0.05))
            tl = Text('both low-rank vectors tied to k', font_size=26, color=BLUE).next_to(t1, UP, buff=0.4)
            self.play(Create(tied), FadeIn(tl), run_time=0.8)
            self.wait(2)
            self.play(Write(t2), run_time=1)
            sp = Text('reported ≈ 2× faster kernels', font_size=30, color=GREEN).move_to(DOWN * 2.0)
            self.play(FadeIn(sp), run_time=0.6)
        self.term(None)
        self.clear_stage()

    def sum_form(self):
        self.header('3.5  The sum form')
        with self.say('sum'):
            self.term('RWKV-7 and Comba (2025)', INK)
            hp = Text('product (fade, then erase)', font_size=28, color=TEAL, weight=MEDIUM)
            hs = Text('sum (both from the old state)', font_size=28, color=PURPLE, weight=MEDIUM)
            hp.move_to(LEFT * 3.4 + UP * 2.2)
            hs.move_to(RIGHT * 3.4 + UP * 2.2)
            self.play(FadeIn(hp), FadeIn(hs), run_time=0.8)
            p = VGroup(MathTex(r'0.8\cdot2', r'-0.5\cdot', r'(0.8\cdot2)', r'+0.5\cdot5', font_size=38),
                       MathTex(r'=1.6-0.8+2.5', font_size=38), MathTex(r'=3.3', font_size=44, color=TEAL)).arrange(DOWN, buff=0.4)
            s = VGroup(MathTex(r'0.8\cdot2', r'-0.5\cdot', r'2', r'+0.5\cdot5', font_size=38),
                       MathTex(r'=1.6-1.0+2.5', font_size=38), MathTex(r'=3.1', font_size=44, color=PURPLE)).arrange(DOWN, buff=0.4)
            p.next_to(hp, DOWN, buff=0.6)
            s.next_to(hs, DOWN, buff=0.6)
            self.wait(3)
            self.play(Write(p[0]), Write(s[0]), run_time=1.6)
            self.wait(2.5)
            r1 = SurroundingRectangle(p[0][2], color=RED, buff=0.06)
            r2 = SurroundingRectangle(s[0][2], color=RED, buff=0.06)
            el1 = Text('erase reads 1.6', font_size=24, color=RED).next_to(p[0], DOWN, buff=0.15)
            el2 = Text('erase reads 2', font_size=24, color=RED).next_to(s[0], DOWN, buff=0.15)
            self.play(Create(r1), Create(r2), FadeIn(el1), FadeIn(el2), run_time=1)
            for m in (p, s):
                m[1:].shift(DOWN * 0.35)
            self.wait(2)
            self.play(Write(p[1]), Write(s[1]), run_time=1.2)
            self.wait(1.5)
            self.play(Write(p[2]), Write(s[2]), run_time=1)
            o2 = Text('order 2208: 3.2 in both', font_size=26, color=MUTED).move_to(DOWN * 3.0 + RIGHT * 2.5)
            self.play(FadeIn(o2), run_time=0.6)
        with self.say('rwkv', pause=0.6):
            self.clear_stage(run_time=0.5)
            self.term(None)
            t = MathTex(r'A_t=', r'\operatorname{diag}(\Ct{w})', r'-', r'\Cr{\hat\kappa}', r'(a\odot', r'\Cr{\hat\kappa}', r')^{\top}', font_size=50).move_to(UP * 0.9)
            self.play(Write(t), run_time=1.5)
            l1 = Text('decay: a vector over key channels', font_size=24, color=TEAL).next_to(t[1], DOWN, buff=0.9).shift(LEFT * 1.0)
            l2 = Text('removal key ≠ write key', font_size=24, color=RED).next_to(t[3:6], UP, buff=0.5)
            l3 = Text('write strength: a vector', font_size=24).next_to(t[4], DOWN, buff=0.9).shift(RIGHT * 1.6)
            self.play(FadeIn(l1), run_time=0.6)
            self.play(FadeIn(l3), run_time=0.6)
            self.wait(1)
            self.play(FadeIn(l2), run_time=0.6)
            c = Text('general diagonal + low rank  →  ≈ 2× KDA’s work per chunk', font_size=28, color=PURPLE).move_to(DOWN * 2.2)
            self.wait(2.5)
            self.play(FadeIn(c), run_time=0.8)
        self.clear_stage()

    def tracking(self):
        self.header('3.6  Negative eigenvalues and state tracking')
        bits = [1, 0, 1, 1, 0, 1]
        boxes = VGroup(*[VGroup(Square(0.7, stroke_color=INK, stroke_width=2), Text(str(b), font_size=32)) for b in bits])
        boxes.arrange(RIGHT, buff=0.2).move_to(UP * 1.8)
        with self.say('parity'):
            self.play(FadeIn(boxes, lag_ratio=0.15), run_time=1.2)
            q = Text('number of 1-bits so far: even or odd?', font_size=28).next_to(boxes, DOWN, buff=0.4)
            self.play(FadeIn(q), run_time=0.6)
            self.wait(4)
            line = NumberLine(x_range=[-1, 1, 0.5], length=7, color=INK, include_numbers=False).move_to(DOWN * 1.2)
            nums = VGroup(*[Text(f'{v:g}', font_size=22, color=MUTED).next_to(line.n2p(v), DOWN, buff=0.15) for v in (-1, 0, 1)])
            self.play(Create(line), FadeIn(nums), run_time=0.8)
            d = Dot(line.n2p(1), color=PURPLE, radius=0.13)
            self.add(d)
            pos = Text('only positive eigenvalues: shrink or stay, never flip', font_size=26, color=RED).next_to(line, DOWN, buff=0.7)
            for v in (0.7, 0.45, 0.3):
                self.play(d.animate.move_to(line.n2p(v)), run_time=0.7)
            self.play(FadeIn(pos), run_time=0.6)
            self.term('provably cannot solve parity in finite precision', RED)
        with self.say('flip', pause=0.6):
            self.play(FadeOut(pos), FadeOut(q), d.animate.move_to(line.n2p(1)), run_time=0.6)
            self.term('β = 2σ(·): eigenvalue 1 − β reaches −1', PURPLE)
            self.wait(4.5)
            val = 1
            ptr = Triangle(color=GOLD, fill_opacity=1).scale(0.15).rotate(PI)
            for i, b in enumerate(bits):
                ptr.next_to(boxes[i], UP, buff=0.1)
                if i == 0:
                    self.play(FadeIn(ptr), run_time=0.3)
                if b:
                    val = -val
                    self.play(boxes[i][0].animate.set_fill(PURPLE, 0.25), d.animate.move_to(line.n2p(val)),
                              ptr.animate.next_to(boxes[i], UP, buff=0.1), run_time=0.9)
                else:
                    self.play(ptr.animate.next_to(boxes[i], UP, buff=0.1), run_time=0.6)
                self.wait(0.3)
            ev = Text('ends at +1: even count ✓', font_size=30, color=GREEN, weight=MEDIUM).next_to(line, DOWN, buff=0.7)
            self.play(FadeIn(ev), run_time=0.6)
        with self.say('product'):
            self.clear_stage(run_time=0.5)
            self.term('DeltaProduct (2025): several delta steps per token', PURPLE)
            O = DOWN * 0.6
            R = 2.6
            m1 = DashedLine(O - R * np.array([1, 0, 0]), O + R * np.array([1, 0, 0]), color=TEAL, stroke_width=3)
            a = np.deg2rad(30)
            m2 = DashedLine(O - R * np.array([np.cos(a), np.sin(a), 0]), O + R * np.array([np.cos(a), np.sin(a), 0]), color=BLUE, stroke_width=3)
            ml1 = Text('mirror 1', font_size=22, color=TEAL).next_to(m1.get_end(), DOWN, buff=0.1)
            ml2 = Text('mirror 2 (30°)', font_size=22, color=BLUE).next_to(m2.get_end(), UP, buff=0.1)
            self.play(Create(m1), Create(m2), FadeIn(ml1), FadeIn(ml2), run_time=1)

            def vec(deg, col):
                r = np.deg2rad(deg)
                return Arrow(O, O + 2.2 * np.array([np.cos(r), np.sin(r), 0]), buff=0, color=col, stroke_width=6)
            v0 = vec(20, PURPLE)
            self.play(GrowArrow(v0), run_time=0.6)
            v1 = vec(-20, MUTED)
            self.play(TransformFromCopy(v0, v1), run_time=1)
            v2 = vec(80, GREEN)
            self.play(TransformFromCopy(v1, v2), run_time=1)
            arc = Arc(radius=1.0, start_angle=np.deg2rad(20), angle=np.deg2rad(60), arc_center=O, color=GREEN, stroke_width=4)
            al = Text('two reflections = 60° rotation', font_size=28, color=GREEN, weight=MEDIUM).move_to(LEFT * 3.6 + UP * 2.0)
            self.play(Create(arc), FadeIn(al), run_time=0.8)
            ns = Text('the state does not grow', font_size=26, color=MUTED).move_to(RIGHT * 4.3 + DOWN * 2.6)
            self.play(FadeIn(ns), run_time=0.5)
        with self.say('capnote'):
            self.clear_stage(run_time=0.5)
            self.term('expressivity, not capacity', RED)
            st = Mat(np.full((4, 4), 0.6), cell=0.5).move_to(DOWN * 0.3)
            sl = MathTex(r'\approx d_k\text{ exact pairs, still}', font_size=36, color=PURPLE).next_to(st, DOWN, buff=0.35)
            self.play(FadeIn(st), Write(sl), run_time=1)
        self.term(None)
        self.clear_stage()

    def training(self):
        self.header('3.7  Training the delta rule')
        with self.say('train'):
            d = MathTex(r'\Ct{\Diag(\alpha_1)}\Ct{\Diag(\alpha_2)}=\Ct{\Diag(\alpha_2)}\Ct{\Diag(\alpha_1)}', font_size=38).move_to(UP * 1.7)
            dl = Text('diagonal: commute → cumulative products', font_size=26, color=TEAL).next_to(d, DOWN, buff=0.3)
            self.play(Write(d), run_time=1.4)
            self.play(FadeIn(dl), run_time=0.6)
            self.wait(2.5)
            h = MathTex(r'(I-\beta_1\kc{1}\kct{1})(I-\beta_2\kc{2}\kct{2})\;\neq\;(I-\beta_2\kc{2}\kct{2})(I-\beta_1\kc{1}\kct{1})',
                        font_size=36).move_to(DOWN * 0.4)
            hl = Text('delta rule: order matters', font_size=26, color=RED).next_to(h, DOWN, buff=0.3)
            self.play(Write(h), run_time=1.6)
            self.play(FadeIn(hl), run_time=0.5)
            self.wait(1.5)
            box1 = VGroup(panel(3.6, 0.9, color=PURPLE), Text('WY / UT form', font_size=28, color=PURPLE))
            box1[1].move_to(box1[0])
            box2 = VGroup(panel(3.6, 0.9, color=GREEN), Text('matrix products', font_size=28, color=GREEN))
            box2[1].move_to(box2[0])
            flow = VGroup(box1, box2).arrange(RIGHT, buff=1.2).move_to(DOWN * 2.6)
            ar = arrow(box1.get_right(), box2.get_left(), color=MUTED)
            self.play(FadeIn(box1), run_time=0.6)
            self.play(GrowArrow(ar), FadeIn(box2), run_time=0.8)
        self.clear_stage()

    def recap(self):
        self.header('Chapter 3 · Correcting')
        card, lo = self.recap_card(r'\Sc{t}=\Sc{t-1}\big(I-\beta_t\kc{t}\kct{t}\big)+\beta_t\vc{t}\kct{t}',
                                   'Read the old value, write only the error.',
                                   'the table is still d_v × d_k; a better write adds no room.')
        with self.say('recap', pause=0.6):
            self.play(FadeIn(card, shift=0.2 * UP), run_time=1)
        with self.say('leftover'):
            self.play(FadeIn(lo, shift=0.1 * UP), run_time=0.8)
        self.wait(1)
