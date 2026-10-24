"""Chapter 4 · One axis: the transition matrix (Fixed-Size States §4)."""
from common import *


def rot(deg):
    r = np.deg2rad(deg)
    return np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])


def householder(deg, beta=1.0):
    k = np.array([np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))])
    return np.eye(2) - beta * np.outer(k, k), k


class Ch4Transition(VoiceScene):
    SCRIPT = 'ch4_transition'

    def construct(self):
        self.shape_of_all()
        self.rows()
        self.toy_recall()
        self.commute()
        self.research()
        self.dial()
        self.recap()

    def shape_of_all(self):
        with self.say('intro', pause=0.8):
            card = self.title_card(4, 'One axis', 'the transition matrix')
            self.play(FadeIn(card, shift=0.2 * UP), run_time=1)
            self.wait(2.4)
            self.play(FadeOut(card), run_time=0.6)
            self.header('4  One axis: the transition matrix')
            tools = VGroup(*[Text(t, font_size=32, color=c, weight=MEDIUM) for t, c in
                             (('add', PURPLE), ('fade', TEAL), ('correct', RED))]).arrange(RIGHT, buff=1.2).move_to(UP * 1.6)
            self.play(LaggedStart(*[FadeIn(t, shift=0.2 * UP) for t in tools], lag_ratio=0.3), run_time=1.2)
            self.wait(2)
            self.f = MathTex(r'\Sc{t}=\Sc{t-1}', r'\Ct{A_t}', r'+', r'\Cv{W_t}', font_size=64).move_to(DOWN * 0.3)
            self.play(Write(self.f), run_time=1.5)
        f = self.f
        with self.say('parts'):
            bw = Brace(f[3], DOWN, color=PURPLE)
            tw = Text('write: almost always a scaled v kᵀ', font_size=26, color=PURPLE).next_to(bw, DOWN, buff=0.15)
            ba = Brace(f[1], UP, color=TEAL)
            ta = Text('transition: what happens to the old state', font_size=26, color=TEAL).next_to(ba, UP, buff=0.15)
            self.play(FadeOut(tools), GrowFromCenter(bw), FadeIn(tw), run_time=0.8)
            self.wait(2.5)
            self.play(GrowFromCenter(ba), FadeIn(ta), run_time=0.8)
            self.wait(3.5)
            self.term('the methods differ almost only in A_t', TEAL)
            self.wait(2)
            self.play(FadeOut(VGroup(bw, tw, ba, ta)), f.animate.scale(0.7).to_edge(UP, buff=0.8), run_time=0.8)
        with self.say('keyside'):
            rng = np.random.default_rng(1)
            S0 = rng.uniform(1.0, 3.0, (4, 4))
            S = Mat(S0, cell=0.6, vmax=3.2).move_to(LEFT * 2.5 + DOWN * 0.6)
            cl = VGroup(*[Text(f'{j + 1}', font_size=20, color=BLUE).next_to(S.cell(0, j), UP, buff=0.12) for j in range(4)])
            cl.add(Text('key directions', font_size=22, color=BLUE).next_to(cl, UP, buff=0.12))
            Sl = tlabel(r'\Sc{t-1}', S, DOWN)
            al = np.array([0.95, 0.4, 0.8, 0.6])
            A = Mat(np.diag(al), cell=0.6, color=TEAL, vmax=1).next_to(S, RIGHT, buff=0.5)
            Al = tlabel(r'\Ct{A_t}', A, DOWN)
            self.play(FadeIn(S), FadeIn(Sl), run_time=0.6)
            self.wait(2.5)
            self.play(FadeIn(A), FadeIn(Al), run_time=0.6)
            self.play(LaggedStart(*[FadeIn(c) for c in cl], lag_ratio=0.2), run_time=1.2)
            self.term('A_t acts on the key side: one column per key direction', BLUE)
            self.wait(1.5)
            for j in range(4):
                hi = SurroundingRectangle(VGroup(*[S.cell(i, j) for i in range(4)]), color=TEAL, buff=0.02)
                self.play(FadeIn(hi), *[S.cell(i, j).animate.set_fill(PURPLE, min(1, S0[i, j] * al[j] / 3.2) * 0.9) for i in range(4)],
                          run_time=0.5)
                self.play(FadeOut(hi), run_time=0.15)
        self.term(None)
        self.clear_stage(keep=[f])

    def rows(self):
        self.header('4  The table, row by row')
        plane = NumberPlane(x_range=[-1.4, 1.4, 0.5], y_range=[-1.4, 1.4, 0.5], x_length=4.8, y_length=4.8,
                            background_line_style={'stroke_color': LINE, 'stroke_width': 1},
                            axis_config={'stroke_color': MUTED, 'stroke_width': 1.5}).move_to(RIGHT * 3.7 + DOWN * 0.75)
        O = plane.c2p(0, 0)
        x0 = np.array([0.8, 0.6])

        def shape():
            c = Circle(radius=plane.c2p(1, 0)[0] - O[0], color=TEAL, stroke_width=3).move_to(O)
            mark = Dot(plane.c2p(0, 1), radius=0.07, color=TEAL)
            v = Arrow(O, plane.c2p(*x0), buff=0, color=PURPLE, stroke_width=6, max_tip_length_to_length_ratio=0.15)
            return VGroup(c, mark, v)

        ghost = shape()
        ghost[0].set_stroke(opacity=0.22)
        ghost[1:].set_opacity(0.22)
        cur = shape()
        kl = Text('key space', font_size=22, color=MUTED).next_to(plane, DOWN, buff=0.15)
        self.play(Create(plane), FadeIn(ghost), FadeIn(cur), FadeIn(kl), run_time=0.8)
        state = {'card': None, 'shape': cur, 'k': None}

        def card(A, methods, can, train):
            a = MathTex(A, font_size=46)
            m = Text(methods, font_size=28, weight=MEDIUM)
            c = Text(can, font_size=28, color=TEAL)
            t = Text('chunk training: ' + train, font_size=22, color=MUTED)
            g = VGroup(a, m, c, t).arrange(DOWN, buff=0.4, aligned_edge=LEFT).move_to(LEFT * 3.4 + DOWN * 0.5)
            for x in g:
                if x.width > 6.6:
                    x.scale_to_fit_width(6.6)
            g.align_to(LEFT * 6.6, LEFT)
            return g

        def show(A, methods, can, train, M, kdeg=None, run=1.6):
            new = card(A, methods, can, train)
            anims = [FadeOut(state['card'])] if state['card'] else []
            fresh = shape()
            anims += [FadeIn(new), FadeOut(state['shape']), FadeIn(fresh)]
            if state['k'] is not None:
                anims.append(FadeOut(state['k']))
                state['k'] = None
            if kdeg is not None:
                kv = np.array([np.cos(np.deg2rad(kdeg)), np.sin(np.deg2rad(kdeg))])
                state['k'] = VGroup(DashedLine(plane.c2p(*(-1.3 * kv)), plane.c2p(*(1.3 * kv)), color=BLUE, stroke_width=2),
                                    Arrow(O, plane.c2p(*(1.2 * kv)), buff=0, color=BLUE, stroke_width=4),
                                    MathTex(r'\kc{}', font_size=32).next_to(plane.c2p(*(1.25 * kv)), UR, buff=0.05))
                anims.append(FadeIn(state['k']))
            self.play(*anims, run_time=0.6)
            state['card'], state['shape'] = new, fresh
            if M is not None:
                Mt = np.asarray(M).T  # rows of S map r -> r A; as column vectors that is A^T
                self.play(ApplyMatrix(np.array([[Mt[0, 0], Mt[0, 1], 0], [Mt[1, 0], Mt[1, 1], 0], [0, 0, 1]]), fresh,
                                      about_point=O), run_time=run)

        with self.say('rowI'):
            show(r'\Ct{A_t}=I', 'linear attention', 'nothing: it only adds', 'plain sums', None)
            self.play(Indicate(state['shape'][2], color=PURPLE, scale_factor=1.05), run_time=1)
        with self.say('rowScalar'):
            show(r'\Ct{A_t}=\gamma I,\ \alpha_tI', 'RetNet, Lightning, Mamba-2', 'fade everything, equally', 'cumulative product',
                 0.6 * np.eye(2), run=2)
        with self.say('rowDiag'):
            show(r'\Ct{A_t}=\Diag(\alpha_t)', 'GLA', 'fade each key channel', 'cumulative sum in log space',
                 np.diag([0.4, 0.95]), run=2)
        with self.say('rowDelta'):
            H, _ = householder(30)
            show(r'\Ct{A_t}=I-\beta\kc{}\kct{}', 'DeltaNet', 'overwrite along k', 'WY / UT form', H, kdeg=30, run=2)
            self.wait(3)
            show(r'\Ct{A_t}=\alpha_t(I-\beta\kc{}\kct{})', 'Gated DeltaNet', 'fade, then overwrite', 'WY / UT form', 0.7 * H, kdeg=30, run=1.6)
            self.wait(2)
            show(r'\Ct{A_t}=\Diag(\alpha_t)(I-\beta\kc{}\kct{})', 'KDA', 'fade per channel, then overwrite', 'WY / UT form',
                 np.diag([0.5, 0.95]) @ H, kdeg=30, run=1.6)
        with self.say('rowRWKV'):
            kv = np.array([np.cos(np.deg2rad(30)), np.sin(np.deg2rad(30))])
            M = np.diag([0.9, 0.85]) - np.outer(kv, np.array([0.7, 0.7]) * kv)
            show(r'\Ct{A_t}=\operatorname{diag}(w)-\hat\kappa(a\odot\hat\kappa)^{\top}', 'RWKV-7, Comba', 'the sum form',
                 'general diagonal + low rank', M, kdeg=30, run=2)
        with self.say('rowProd'):
            show(r'\Ct{A_t}=\text{reflections, rotations}', 'DeltaProduct, Mamba-3', 'track structure such as parity',
                 'per sub-step; rotations', rot(-60), run=2)
        self.clear_stage(keep=[self.f])

    def toy_recall(self):
        self.header('4  Same write, different transitions')
        rows = [('before', 2, 4), ('add (A = I)', 7, 4), ('delta, β = 1', 5, 4), ('Gated DeltaNet', 3.3, 3.2),
                ('KDA', 3.3, 4), ('sum form', 3.1, 3.2)]
        U, x0 = 0.6, -2.4
        with self.say('toyrecall'):
            self.term('the order toy from chapter 3', INK)
            leg = VGroup(VGroup(Square(0.25, fill_color=PURPLE, fill_opacity=0.85, stroke_width=0), Text('order 7314 (target 5)', font_size=22, color=PURPLE)).arrange(RIGHT, buff=0.15),
                         VGroup(Square(0.25, fill_color=BLUE, fill_opacity=0.85, stroke_width=0), Text('order 2208 (was 4)', font_size=22, color=BLUE)).arrange(RIGHT, buff=0.15))
            leg.arrange(RIGHT, buff=0.8).move_to(UP * 2.35 + RIGHT * 0.8)
            t5 = DashedLine([x0 + 5 * U, 2.0, 0], [x0 + 5 * U, -3.0, 0], color=PURPLE, stroke_width=2)
            t4 = DashedLine([x0 + 4 * U, 2.0, 0], [x0 + 4 * U, -3.0, 0], color=BLUE, stroke_width=2)
            self.play(FadeIn(leg), Create(t5), Create(t4), run_time=0.8)
            grp = VGroup()
            for i, (name, a, b) in enumerate(rows):
                y = 1.75 - i * 0.82
                lab = Text(name, font_size=26, weight=MEDIUM if i else NORMAL, color=INK if i else MUTED).move_to([x0 - 0.35, y, 0], aligned_edge=RIGHT)
                ba = Rectangle(width=a * U, height=0.28, fill_color=PURPLE, fill_opacity=0.85, stroke_width=0).move_to([x0, y + 0.16, 0], aligned_edge=LEFT)
                bb = Rectangle(width=b * U, height=0.28, fill_color=BLUE, fill_opacity=0.85, stroke_width=0).move_to([x0, y - 0.16, 0], aligned_edge=LEFT)
                na = Text(f'{a:g}', font_size=22, color=PURPLE).next_to(ba, RIGHT, buff=0.1)
                nb = Text(f'{b:g}', font_size=22, color=BLUE).next_to(bb, RIGHT, buff=0.1)
                grp.add(VGroup(lab, ba, bb, na, nb))
            waits = [2.2, 1.6, 2.4, 3.6, 2.6, 2.4]
            for g, w in zip(grp, waits):
                self.play(FadeIn(g[0]), GrowFromEdge(g[1], LEFT), GrowFromEdge(g[2], LEFT), FadeIn(g[3:]), run_time=0.8)
                self.wait(w)
            self.term('same write, different transitions', TEAL)
        self.term(None)
        self.clear_stage(keep=[self.f])

    def commute(self):
        self.header('4  Why the training column splits')
        D1, D2 = np.diag([0.5, 0.9]), np.diag([0.9, 0.55])
        H1, _ = householder(0, 0.6)
        H2, _ = householder(60, 0.7)
        planes = []
        for x in (-3.4, 3.4):
            p = NumberPlane(x_range=[-1.3, 1.3, 0.5], y_range=[-1.3, 1.3, 0.5], x_length=4.0, y_length=4.0,
                            background_line_style={'stroke_color': LINE, 'stroke_width': 1},
                            axis_config={'stroke_color': MUTED, 'stroke_width': 1.5}).move_to(RIGHT * x + DOWN * 0.45)
            planes.append(p)
        x0 = np.array([0.8, 0.6])

        def vec(p, M, col):
            return Arrow(p.c2p(0, 0), p.c2p(*(M @ x0)), buff=0, color=col, stroke_width=6, max_tip_length_to_length_ratio=0.15)

        def ell(p, M, col):
            return ParametricFunction(lambda t: p.c2p(*(M @ np.array([np.cos(t), np.sin(t)]))), t_range=[0, TAU], color=col, stroke_width=3)

        with self.say('commute'):
            t1 = Text('diagonal: commute', font_size=28, color=TEAL, weight=MEDIUM).next_to(planes[0], UP, buff=0.25)
            t2 = Text('Householder-like: order matters', font_size=28, color=RED, weight=MEDIUM).next_to(planes[1], UP, buff=0.25)
            leg = VGroup(VGroup(Line(ORIGIN, RIGHT * 0.4, color=GOLD, stroke_width=5), Text('one order', font_size=22, color=GOLD)).arrange(RIGHT, buff=0.12),
                         VGroup(Line(ORIGIN, RIGHT * 0.4, color=PURPLE, stroke_width=5), Text('the other order', font_size=22, color=PURPLE)).arrange(RIGHT, buff=0.12))
            leg.arrange(RIGHT, buff=0.6).move_to(UP * 2.75)
            self.play(FadeOut(self.f), Create(planes[0]), Create(planes[1]), FadeIn(leg), run_time=0.8)
            self.wait(1.5)
            self.play(FadeIn(t1), run_time=0.5)
            self.wait(1)
            a, b = D2 @ D1, D1 @ D2
            self.play(Create(ell(planes[0], a, GOLD)), GrowArrow(vec(planes[0], a, GOLD)), run_time=1.2)
            self.play(Create(ell(planes[0], b, PURPLE)), GrowArrow(vec(planes[0], b, PURPLE)), run_time=1.2)
            same = Text('either order: same place', font_size=24, color=TEAL).next_to(planes[0], DOWN, buff=0.15)
            self.play(FadeIn(same), run_time=0.5)
            self.term('diagonal: a chunk needs only cumulative decays', TEAL)
            self.wait(2)
            self.play(FadeIn(t2), run_time=0.5)
            a, b = H2 @ H1, H1 @ H2
            self.play(Create(ell(planes[1], a, GOLD)), GrowArrow(vec(planes[1], a, GOLD)), run_time=1.2)
            self.play(Create(ell(planes[1], b, PURPLE)), GrowArrow(vec(planes[1], b, PURPLE)), run_time=1.2)
            diff = Text('swap the order: a different result', font_size=24, color=RED).next_to(planes[1], DOWN, buff=0.15)
            self.play(FadeIn(diff), run_time=0.5)
            self.term('delta rule: needs the WY / UT rewrite', RED)
        self.term(None)
        self.clear_stage()

    def research(self):
        self.header('4  The 2026 refinements')
        with self.say('research'):
            g = MathTex(r'\Sc{t}=\Sc{t-1}\big(I-', r'\beta_t', r'\,', r'\kc{t}\kct{t}', r'\big)+', r'\beta_t', r'\,\vc{t}\kct{t}', font_size=52).move_to(UP * 0.7)
            self.play(Write(g), run_time=1.4)
            l1 = Text('how strongly to erase', font_size=26, color=RED).next_to(g[1], UP, buff=0.9).shift(LEFT * 1.2)
            l2 = Text('where to erase', font_size=26, color=BLUE).next_to(g[3], DOWN, buff=0.9)
            l3 = Text('how strongly to write', font_size=26, color=PURPLE).next_to(g[5], UP, buff=0.9).shift(RIGHT * 1.0)
            a1 = arrow(l1.get_bottom(), g[1].get_top(), color=RED)
            a2 = arrow(l2.get_top(), g[3].get_bottom(), color=BLUE)
            a3 = arrow(l3.get_bottom(), g[5].get_top(), color=PURPLE)
            self.wait(2.5)
            self.play(FadeIn(l1), GrowArrow(a1), run_time=0.6)
            self.wait(0.6)
            self.play(FadeIn(l3), GrowArrow(a3), run_time=0.6)
            self.wait(0.6)
            self.play(FadeIn(l2), GrowArrow(a2), run_time=0.6)
            opts = VGroup(*[Text(t, font_size=24, color=INK) for t in
                            ('per-channel erase and write gates', 'an erase direction separate from the write key',
                             'a step size set by an uncertainty estimate')]).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
            opts.move_to(DOWN * 2.4 + LEFT * 2.0)
            self.wait(1.5)
            for o in opts:
                self.play(FadeIn(o, shift=0.1 * RIGHT), run_time=0.6)
                self.wait(1.4)
            box = Mat(np.full((3, 3), 0.6), cell=0.4).move_to(RIGHT * 4.6 + DOWN * 2.3)
            bl = Text('state size: unchanged', font_size=24, color=PURPLE).next_to(box, UP, buff=0.2)
            self.play(FadeIn(box), FadeIn(bl), run_time=0.6)
        self.clear_stage()

    def dial(self):
        self.header('4  The whole table')
        rows = [(r'I', 'linear attention', 'nothing, only adds'),
                (r'\gamma I,\ \alpha_tI', 'RetNet, Lightning, Mamba-2', 'fade everything'),
                (r'\Diag(\alpha_t)', 'GLA', 'fade each key channel'),
                (r'I-\beta kk^{\top}', 'DeltaNet', 'overwrite along k'),
                (r'\alpha_t(I-\beta kk^{\top})', 'Gated DeltaNet', 'fade, then overwrite'),
                (r'\Diag(\alpha_t)(I-\beta kk^{\top})', 'KDA', 'fade per channel, then overwrite'),
                (r'\operatorname{diag}(w)-\hat\kappa(a\odot\hat\kappa)^{\top}', 'RWKV-7, Comba', 'the sum form'),
                (r'\text{reflections, rotations}', 'DeltaProduct, Mamba-3', 'track structure such as parity')]
        xs = (-4.3, -0.2, 3.9)
        hdr = VGroup(MathTex(r'\Ct{A_t}', font_size=32).move_to([xs[0], 2.75, 0]),
                     Text('methods', font_size=24, color=MUTED).move_to([xs[1], 2.75, 0]),
                     Text('what it can do', font_size=24, color=MUTED).move_to([xs[2], 2.75, 0]))
        table = VGroup()
        for i, (a, m, c) in enumerate(rows):
            y = 2.15 - i * 0.66
            r = VGroup(MathTex(a, font_size=30, color=TEAL).move_to([xs[0], y, 0]),
                       Text(m, font_size=24).move_to([xs[1], y, 0]),
                       Text(c, font_size=24, color=INK).move_to([xs[2], y, 0]))
            for x in r:
                if x.width > 3.9:
                    x.scale_to_fit_width(3.9)
            table.add(r)
        rule = Line([-6.5, 2.45, 0], [6.5, 2.45, 0], color=LINE, stroke_width=2)
        with self.say('dial'):
            self.play(FadeIn(hdr), Create(rule), LaggedStart(*[FadeIn(r) for r in table], lag_ratio=0.12), run_time=2)
            ptr = Triangle(color=GOLD, fill_opacity=1).scale(0.14).rotate(-PI / 2).next_to(table[0], LEFT, buff=0.25)
            ptr.set_x(-6.7)
            self.play(FadeIn(ptr), run_time=0.3)
            for i, w in ((1, 1.2), (2, 1.4), (3, 1.4), (7, 2.0)):
                self.play(ptr.animate.set_y(table[i].get_y()), Indicate(table[i][2], color=GOLD, scale_factor=1.05), run_time=0.9)
                self.wait(w)
        self.table = VGroup(hdr, rule, table, ptr)

    def recap(self):
        self.play(FadeOut(self.table), run_time=0.6)
        self.header('Chapter 4 · One axis')
        card, lo = self.recap_card(r'\Sc{t}=\Sc{t-1}\Ct{A_t}+\Cv{W_t}',
                                   'The transition decides what the state can do.',
                                   'A_t changes how the table is edited, never how big it is.')
        with self.say('recap', pause=0.6):
            self.play(FadeIn(card, shift=0.2 * UP), run_time=1)
        with self.say('leftover'):
            self.play(FadeIn(lo, shift=0.1 * UP), run_time=0.8)
        self.wait(1)
