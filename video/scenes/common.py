"""Shared kit for the Fixed-Size States video series.

Palette, LaTeX macros, a Kokoro voiceover that keeps animations in sync with
the narration (and writes an .srt per chapter), and a few reusable visuals.

Colour meaning is the book's (scripts/figkit.py):
  gold = query · blue = key · purple = value / state · green = output / kept
  red = erased / cost · teal = transition · muted / line = inactive
"""
import hashlib
import json
import logging
import os
import re
import wave
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from manim import *

VIDEO = Path(__file__).resolve().parents[1]
SCRIPT_DIR = VIDEO / 'script'
CACHE = VIDEO / '.cache' / 'tts'
SRT_DIR = VIDEO / 'build' / 'srt'
MODELS = Path(os.environ.get('KOKORO_DIR', VIDEO / '.models'))
VOICE = os.environ.get('KOKORO_VOICE', 'af_heart')
SPEED = float(os.environ.get('KOKORO_SPEED', '1.0'))

# --- palette (scripts/figkit.py COL, style.css) -----------------------------
INK, MUTED, LINE = '#253340', '#687986', '#cad5db'
BLUE, GREEN, GOLD, RED = '#3f83b6', '#1f9a74', '#d09124', '#cf5f59'
PURPLE, TEAL = '#8d6bb5', '#2a9aa6'
PAPER, SOFT, BG, WHITE_ = '#f4f7f8', '#eef3f5', '#fbfcfc', '#ffffff'
FONT = 'Inter'

config.background_color = BG

TEX = TexTemplate()
TEX.add_to_preamble(r"""
\usepackage{xcolor}
\definecolor{cq}{HTML}{D09124}\definecolor{ck}{HTML}{3F83B6}\definecolor{cv}{HTML}{8D6BB5}
\definecolor{cg}{HTML}{1F9A74}\definecolor{cr}{HTML}{CF5F59}\definecolor{ct}{HTML}{2A9AA6}
\definecolor{cm}{HTML}{687986}
\newcommand{\Cq}[1]{\textcolor{cq}{#1}}\newcommand{\Ck}[1]{\textcolor{ck}{#1}}
\newcommand{\Cv}[1]{\textcolor{cv}{#1}}\newcommand{\Cg}[1]{\textcolor{cg}{#1}}
\newcommand{\Cr}[1]{\textcolor{cr}{#1}}\newcommand{\Ct}[1]{\textcolor{ct}{#1}}
\newcommand{\Cm}[1]{\textcolor{cm}{#1}}
\newcommand{\qc}[1]{\textcolor{cq}{q_{#1}}}
\newcommand{\kc}[1]{\textcolor{ck}{k_{#1}}}
\newcommand{\kct}[1]{\textcolor{ck}{k_{#1}^{\top}}}
\newcommand{\vc}[1]{\textcolor{cv}{v_{#1}}}
\newcommand{\Sc}[1]{\textcolor{cv}{S_{#1}}}
\newcommand{\oc}[1]{\textcolor{cg}{o_{#1}}}
\newcommand{\Diag}{\operatorname{Diag}}
""")
config.tex_template = TEX
Text.set_default(font=FONT, color=INK)
MathTex.set_default(color=INK)
Tex.set_default(color=INK)


# --- narration script --------------------------------------------------------
def load_script(name):
    """Segments of video/script/<name>.md: text after `<!-- key -->` until the next
    marker or heading. Quote lines (on-screen formulas) are not spoken."""
    segs, cur = {}, None
    for line in (SCRIPT_DIR / f'{name}.md').read_text().splitlines():
        m = re.match(r'\s*<!--\s*(\S+)\s*-->', line)
        if m:
            cur = m.group(1)
            segs[cur] = []
        elif line.startswith('#'):
            cur = None
        elif cur and not line.startswith('>'):
            segs[cur].append(line)
    return {k: ' '.join(' '.join(v).split()) for k, v in segs.items()}


# Spoken forms for names the phonemizer would otherwise spell or slur.
PRON = [(r'\bGLA\b', 'G L A'), (r'\bKDA\b', 'K D A'), (r'\bRWKV-7\b', 'R W K V seven'),
        (r'\bTTT\b', 'T T T'), (r'\bLaCT\b', 'lact'), (r'\bMamba-(\d)\b', r'Mamba \1'),
        (r'\bSSM\b', 'S S M'), (r'\bKV\b', 'K V'), (r'\bMLP\b', 'M L P'), (r'\bGPU\b', 'G P U'),
        (r'\bH100\b', 'H 100'), (r'\bWY\b', 'W Y'), (r'\bUT\b', 'U T'), (r'\bRetNet\b', 'Ret Net'),
        (r'\bDeltaNet\b', 'Delta Net'), (r'\bDeltaProduct\b', 'Delta Product'), (r'\bMIMO\b', 'my-mo'),
        (r'\bATLAS\b', 'Atlas'), (r'\belu\b', 'E L U'), (r'\bFLOPs\b', 'flops'), (r'\bd k\b', 'dee kay'),
        (r'\bd v\b', 'dee vee'), (r'\bphi\b', 'fie'), (r'\bQ K\b', 'Q, K'), (r'\bCG\b', 'C G'),
        (r'\bSwiGLU\b', 'swig-loo')]


def spoken(text):
    for pat, rep in PRON:
        text = re.sub(pat, rep, text)
    return text


_KOKORO = None
logging.getLogger('phonemizer').setLevel(logging.ERROR)


def _tts(sentence):
    global _KOKORO
    if _KOKORO is None:
        from kokoro_onnx import Kokoro
        _KOKORO = Kokoro(str(MODELS / 'kokoro-v1.0.onnx'), str(MODELS / 'voices-v1.0.bin'))
    audio, sr = _KOKORO.create(spoken(sentence), voice=VOICE, speed=SPEED, lang='en-us')
    audio = np.asarray(audio, dtype=np.float32)
    loud = np.flatnonzero(np.abs(audio) > 0.01)          # trim the model's edge silence
    if len(loud):
        audio = audio[max(0, loud[0] - int(.03 * sr)): loud[-1] + int(.08 * sr)]
    return audio, sr


def sentences(text):
    return [s for s in re.split(r'(?<=[.!?:])\s+(?=[A-Z0-9])', text) if s.strip()]


def voice_segment(text, gap=0.32):
    """Synthesise one narration segment, cached. Returns wav path, duration and
    per-sentence cues (start, end, text) relative to the segment start."""
    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f'{VOICE}|{SPEED}|{gap}|{text}'.encode()).hexdigest()[:16]
    wav, meta = CACHE / f'{key}.wav', CACHE / f'{key}.json'
    if wav.exists() and meta.exists():
        return {'wav': wav, **json.loads(meta.read_text())}
    parts, cues, t, sr = [], [], 0.0, 24000
    for s in sentences(text):
        a, sr = _tts(s)
        cues.append((t, t + len(a) / sr, s))
        parts += [a, np.zeros(int(gap * sr), np.float32)]
        t += len(a) / sr + gap
    pcm = (np.clip(np.concatenate(parts), -1, 1) * 32767).astype('<i2')
    with wave.open(str(wav), 'wb') as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr), w.writeframes(pcm.tobytes())
    info = {'dur': len(pcm) / sr, 'cues': cues}
    meta.write_text(json.dumps(info))
    return {'wav': wav, **info}


def _split_cue(a, b, text, width=84):
    """Break a long sentence into subtitle-sized cues, timed by character share."""
    if len(text) <= width:
        return [(a, b, text)]
    chunks, cur = [], ''
    for piece in re.split(r'(?<=[,;])\s+', text):
        if cur and len(cur) + len(piece) + 1 > width:
            chunks.append(cur)
            cur = piece
        else:
            cur = f'{cur} {piece}'.strip()
    chunks.append(cur)
    total, out, t = sum(len(c) for c in chunks), [], a
    for c in chunks:
        d = (b - a) * len(c) / total
        out.append((t, t + d, c))
        t += d
    return out


def _stamp(t):
    ms = int(round(t * 1000))
    return f'{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}'


class Tracker:
    def __init__(self, scene, t0, dur):
        self.scene, self.t0, self.dur = scene, t0, dur

    def left(self):
        return self.t0 + self.dur - self.scene.renderer.time


class VoiceScene(Scene):
    """A scene narrated from video/script/<SCRIPT>.md.

    `with self.say('key') as t:` starts that segment's audio; the animations in
    the block run alongside it and the block waits for the audio to finish."""
    SCRIPT = ''

    def setup(self):
        self.script = load_script(self.SCRIPT)
        self.cues, self.used, self.spans = [], set(), []
        self._chip = self._head = None

    @contextmanager
    def say(self, key, pause=0.0):
        seg = voice_segment(self.script[key])
        t0 = self.renderer.time
        # Scene.add_sound drops the sound when the previous play() came from manim's cache.
        self.renderer.file_writer.add_sound(str(seg['wav']), t0)
        tr = Tracker(self, t0, seg['dur'])
        yield tr
        if tr.left() > 1 / 30:
            self.wait(tr.left())
        if pause:
            self.wait(pause)
        for a, b, s in seg['cues']:
            self.cues += _split_cue(t0 + a, t0 + b, s)
        self.used.add(key)
        self.spans.append((key, t0, self.renderer.time))

    def tear_down(self):
        missing = set(self.script) - self.used
        if missing:
            print(f'WARNING: narration segments never used: {sorted(missing)}')
        if self.cues:
            SRT_DIR.mkdir(parents=True, exist_ok=True)
            out = [f'{i}\n{_stamp(a)} --> {_stamp(b)}\n{s}\n' for i, (a, b, s) in enumerate(self.cues, 1)]
            (SRT_DIR / f'{type(self).__name__}.srt').write_text('\n'.join(out))
            (SRT_DIR / f'{type(self).__name__}.spans.json').write_text(json.dumps(self.spans))

    # --- chrome -------------------------------------------------------------
    def header(self, text):
        """Small section label, top left."""
        new = Text(text, font_size=22, color=MUTED).to_corner(UL, buff=0.35)
        if self._head is None:
            self.play(FadeIn(new), run_time=0.4)
        else:
            self.play(FadeOut(self._head), FadeIn(new), run_time=0.4)
        self._head = new

    def term(self, text, color=INK):
        """Caption chip for a key term, bottom left. `None` clears it."""
        anims = [FadeOut(self._chip)] if self._chip is not None else []
        self._chip = None
        if text:
            label = Text(text, font_size=24, color=color, weight=MEDIUM)
            box = RoundedRectangle(corner_radius=0.12, width=label.width + 0.45, height=label.height + 0.3,
                                   fill_color=WHITE_, fill_opacity=1, stroke_color=color, stroke_width=2)
            self._chip = VGroup(box, label.move_to(box)).to_corner(DL, buff=0.35)
            anims.append(FadeIn(self._chip, shift=0.15 * UP))
        if anims:
            self.play(*anims, run_time=0.4)

    def clear_stage(self, keep=(), run_time=0.6):
        keep = set(keep) | {self._chip, self._head}
        gone = [m for m in self.mobjects if m not in keep]
        if gone:
            self.play(*[FadeOut(m) for m in gone], run_time=run_time)

    def title_card(self, number, title, subtitle):
        num = Text(f'Chapter {number}', font_size=30, color=MUTED)
        ttl = Text(title, font_size=56, weight=SEMIBOLD)
        sub = Text(subtitle, font_size=28, color=MUTED)
        bar = Line(LEFT * 3, RIGHT * 3, color=PURPLE, stroke_width=5)
        grp = VGroup(num, ttl, bar, sub).arrange(DOWN, buff=0.35)
        if grp.width > 12.5:
            grp.scale_to_fit_width(12.5)
        return grp

    def recap_card(self, formula, line, leftover):
        """Key formula plus one line, then the 'Left over:' line (returned separately)."""
        head = Text('Recap', font_size=30, color=MUTED, weight=MEDIUM)
        f = MathTex(formula, font_size=50)
        ln = Text(line, font_size=28)
        body = VGroup(head, f, ln).arrange(DOWN, buff=0.45)
        for m in (f, ln):
            if m.width > 11.8:
                m.scale_to_fit_width(11.8)
        box = RoundedRectangle(corner_radius=0.2, width=12.6, height=body.height + 0.9,
                               fill_color=WHITE_, fill_opacity=1, stroke_color=LINE, stroke_width=2)
        card = VGroup(box, body.move_to(box)).move_to(UP * 0.9)
        lo = VGroup(Text('Left over:', font_size=28, color=RED, weight=SEMIBOLD),
                    Text(leftover, font_size=28, color=INK)).arrange(RIGHT, buff=0.2)
        if lo.width > 12.6:
            lo.scale_to_fit_width(12.6)
        lo.next_to(card, DOWN, buff=0.55)
        return card, lo


# --- visuals -----------------------------------------------------------------
def fill_for(v, vmax, color, neg=RED):
    return (color if v >= 0 else neg), min(1.0, abs(v) / vmax) * 0.9 if vmax else 0


class Mat(VGroup):
    """A matrix (or vector) of shaded cells, optionally with numbers."""

    def __init__(self, values, cell=0.42, color=PURPLE, vmax=None, numbers=False, fmt='{:.1f}',
                 font_size=None, neg=RED, stroke=LINE, **kw):
        super().__init__(**kw)
        self.vals = np.atleast_2d(np.array(values, dtype=float))
        self.color, self.neg, self.fmt, self.numbers = color, neg, fmt, numbers
        self.vmax = vmax or (np.abs(self.vals).max() or 1.0)
        self.font_size = font_size or int(cell * 44)
        r, c = self.vals.shape
        self.cells = VGroup()
        self.nums = VGroup()
        for i in range(r):
            for j in range(c):
                col, op = fill_for(self.vals[i, j], self.vmax, color, neg)
                sq = Square(cell, stroke_color=stroke, stroke_width=1.5, fill_color=col, fill_opacity=op)
                sq.move_to(np.array([j * cell, -i * cell, 0]))
                self.cells.add(sq)
        self.add(self.cells)
        if numbers:
            for k, sq in enumerate(self.cells):
                self.nums.add(self._num(self.vals.flat[k], sq))
            self.add(self.nums)
        self.center()

    def _num(self, v, sq):
        col, op = fill_for(v, self.vmax, self.color, self.neg)
        txt = Text(self.fmt.format(v), font_size=self.font_size, color=WHITE_ if op > 0.5 else INK)
        return txt.move_to(sq)

    def cell(self, i, j):
        return self.cells[i * self.vals.shape[1] + j]

    def set_values(self, values, **anim):
        """Animations that move the cells (and numbers) to new values."""
        self.vals = np.atleast_2d(np.array(values, dtype=float)).reshape(self.vals.shape)
        anims = []
        for k, sq in enumerate(self.cells):
            col, op = fill_for(self.vals.flat[k], self.vmax, self.color, self.neg)
            anims.append(sq.animate.set_fill(col, op))
            if self.numbers:
                anims.append(Transform(self.nums[k], self._num(self.vals.flat[k], sq)))
        return AnimationGroup(*anims, **anim)


def tlabel(tex, mob, direction=UP, buff=0.18, font_size=34):
    return MathTex(tex, font_size=font_size).next_to(mob, direction, buff=buff)


def panel(width, height, color=LINE, fill=WHITE_):
    return RoundedRectangle(corner_radius=0.15, width=width, height=height, fill_color=fill,
                            fill_opacity=1, stroke_color=color, stroke_width=2)


def bar(value, vmax, height=3.0, width=0.6, color=PURPLE):
    h = max(1e-3, height * value / vmax)
    return Rectangle(width=width, height=h, fill_color=color, fill_opacity=0.85, stroke_width=0)


def value_bars(values, labels, vmax, colors, height=3.0, width=0.7, spacing=1.6, fmt='{:.2f}', font_size=26):
    """Bars standing on a common baseline, value printed above, label below."""
    grp = VGroup()
    for i, (v, lab, c) in enumerate(zip(values, labels, colors)):
        b = bar(v, vmax, height, width, c).move_to(RIGHT * i * spacing, aligned_edge=DOWN)
        top = Text(fmt.format(v), font_size=font_size, color=c).next_to(b, UP, buff=0.12)
        low = Text(lab, font_size=font_size - 4, color=INK).next_to(b, DOWN, buff=0.15)
        b.align_to(ORIGIN, DOWN)
        top.next_to(b, UP, buff=0.12)
        low.next_to(b, DOWN, buff=0.15)
        grp.add(VGroup(b, top, low))
    return grp


def arrow(a, b, color=MUTED, **kw):
    return Arrow(a, b, color=color, stroke_width=4, buff=0.1, max_tip_length_to_length_ratio=0.15, **kw)
