#!/usr/bin/env python3
"""Shared drawing kit for the mechanism-card figures.

Colour meaning is fixed across the book:
  gold = query · blue = key / cache record · purple = value / recurrent state
  green = output / selected / kept · red = erased / evicted / cost · line = inactive
"""
from pathlib import Path
from html import escape
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures' / 'cards'
COL = {'ink':'#253340','muted':'#687986','line':'#cad5db','blue':'#3f83b6','green':'#1f9a74','gold':'#d09124',
       'red':'#cf5f59','purple':'#8d6bb5','teal':'#2a9aa6','paper':'#f4f7f8','white':'#ffffff','soft':'#eef3f5'}
SANS = 'Helvetica Neue,Helvetica,Arial,sans-serif'
SERIF = 'STIX Two Math,Cambria Math,Times New Roman,serif'
GREEK = {'alpha':'α','beta':'β','gamma':'γ','delta':'δ','Delta':'Δ','eta':'η','theta':'θ','lambda':'λ','mu':'μ','sigma':'σ',
         'Sigma':'Σ','tau':'τ','phi':'φ','psi':'ψ','omega':'ω','top':'⊤','odot':'⊙','otimes':'⊗','cdot':'·','times':'×',
         'to':'→','leftarrow':'←','approx':'≈','le':'≤','ge':'≥','in':'∈','sum':'Σ','prod':'∏','infty':'∞','sqrt':'√',
         'ell':'ℓ','nabla':'∇','partial':'∂','neq':'≠','propto':'∝','dots':'…','mathbb{R}':'ℝ','R':'ℝ','quad':' ','qquad':'  ','circ':'∘',
         'subseteq':'⊆','supseteq':'⊇','subset':'⊂','notin':'∉','cup':'∪','cap':'∩','setminus':'∖','mid':'∣','emptyset':'∅',
         'rightarrow':'→','Rightarrow':'⇒','mapsto':'↦','pm':'±','sim':'∼','equiv':'≡','ll':'≪','gg':'≫','lfloor':'⌊','rfloor':'⌋',
         'langle':'⟨','rangle':'⟩','Vert':'‖','vert':'|','oplus':'⊕','forall':'∀','exists':'∃','kappa':'κ','rho':'ρ','xi':'ξ',
         'pi':'π','chi':'χ','zeta':'ζ','epsilon':'ε','varepsilon':'ε','Lambda':'Λ','Omega':'Ω','Phi':'Φ','Psi':'Ψ','Gamma':'Γ','Theta':'Θ',
         'iff':'⟺','lor':'∨','land':'∧','neg':'¬','cdots':'⋯','ldots':'…','prime':'′','hat':'^','Leftarrow':'⇐'}
SCRIPT = dict(zip('ABCDEFGHIJKLMNOPQRSTUVWXYZ', '𝒜ℬ𝒞𝒟ℰℱ𝒢ℋℐ𝒥𝒦ℒℳ𝒩𝒪𝒫𝒬ℛ𝒮𝒯𝒰𝒱𝒲𝒳𝒴𝒵'))
UPRIGHT = {'softmax','diag','exp','log','max','min','top','topk','sim','mean','norm','RoPE','KV','FLOP','KiB','MiB','GiB','TB','MB','GB',
           'sigmoid','SiLU','tr','rank','argmax','LN','MLP','ReLU','elu','cos','sin','dim','const','mod','pool','sign','clip','tanh','Attn','cat','B'}
INVENTORY = []


def colour(name):
    return COL.get(name, name)


def _math_tokens(expr):
    """Tiny TeX subset: _x _{..} ^x ^{..} \\c{colour}{..} \\name (greek/symbols) \\text{..} \\b{..} (bold)."""
    out, i = [], 0
    def group(j):
        if j < len(expr) and expr[j] == '{':
            depth, k = 0, j
            while k < len(expr):
                depth += expr[k] == '{'
                depth -= expr[k] == '}'
                if depth == 0:
                    return expr[j+1:k], k+1
                k += 1
            raise ValueError('unbalanced braces in ' + expr)
        if expr[j] == '\\':
            m = re.match(r'\\[A-Za-z]+', expr[j:])
            return m.group(0), j + len(m.group(0))
        return expr[j], j+1
    while i < len(expr):
        ch = expr[i]
        if ch in '_^':
            body, i = group(i+1)
            out.append(('sub' if ch == '_' else 'sup', body))
        elif ch == '\\':
            if expr.startswith('\\c{', i):
                name, j = group(i+2)
                body, i = group(j)
                out.append(('colour', (name, body)))
            elif expr.startswith('\\text{', i):
                body, i = group(i+5)
                out.append(('text', body))
            elif expr.startswith('\\b{', i):
                body, i = group(i+2)
                out.append(('bold', body))
            elif expr.startswith('\\mathbb{R}', i):
                out.append(('sym', 'ℝ')); i += 10
            elif expr.startswith('\\tilde{', i) or expr.startswith('\\bar{', i) or expr.startswith('\\hat{', i):
                mark = {'t': '\u0303', 'b': '\u0304', 'h': '\u0302'}[expr[i+1]]
                body, i = group(expr.index('{', i))
                out.append(('seq', body + mark if len(body) == 1 else body))
            elif expr.startswith('\\sqrt{', i):
                body, i = group(i+5)
                out.append(('sym', '√'))
                out.append(('seq', body if re.fullmatch(r'[A-Za-z0-9_{}^]+', body) else '(' + body + ')'))
            elif expr.startswith('\\mathcal{', i):
                body, i = group(i+8)
                out.append(('sym', ''.join(SCRIPT.get(ch, ch) for ch in body)))
            elif expr.startswith('\\frac{', i) or expr.startswith('\\tfrac{', i):
                num, j = group(i+(5 if expr.startswith('\\frac', i) else 6))
                den, i = group(j)
                wrap = lambda e: e if re.fullmatch(r'[A-Za-z0-9_^{}\\]+', e) else '(' + e + ')'
                out.append(('seq', wrap(num) + '/' + wrap(den)))
            elif i+1 < len(expr) and expr[i+1] in ',;: !{}|%#_&':
                nxt = expr[i+1]
                out.append(('raw', {',': ' ', ';': ' ', ':': ' ', ' ': ' ', '!': '', '{': '{', '}': '}', '|': '‖',
                                    '%': '%', '#': '#', '_': '_', '&': '&'}[nxt]))
                i += 2
            else:
                m = re.match(r'\\([A-Za-z]+)', expr[i:])
                if not m:
                    raise ValueError(f'unsupported TeX at {expr[i:i+8]!r} in {expr!r}')
                name = m.group(1)
                out.append(('sym', GREEK.get(name, name)))
                i += len(m.group(0))
        else:
            m = re.match(r'[A-Za-z]+|\d+(?:\.\d+)?|\s+|.', expr[i:])
            out.append(('raw', m.group(0)))
            i += len(m.group(0))
    return out


def _render_math(expr, size, fill, bold=False, level=0):
    """Return (tspans, approximate width)."""
    parts, width = [], 0.0
    for kind, body in _math_tokens(expr):
        if kind in ('sub', 'sup'):
            small = size * .68
            shift = size * (.28 if kind == 'sub' else -.42)
            inner, w = _render_math(body, small, fill, bold, level+1)
            parts.append(f'<tspan dy="{shift:.1f}" font-size="{small:.1f}">{inner}</tspan><tspan dy="{-shift:.1f}" font-size="{size:.1f}"> </tspan>')
            width += w + size*.08
        elif kind == 'colour':
            name, inner_expr = body
            inner, w = _render_math(inner_expr, size, colour(name), bold, level)
            parts.append(f'<tspan fill="{colour(name)}">{inner}</tspan>')
            width += w
        elif kind == 'seq':
            inner, w = _render_math(body, size, fill, bold, level)
            parts.append(inner)
            width += w
        elif kind == 'bold':
            inner, w = _render_math(body, size, fill, True, level)
            parts.append(f'<tspan font-weight="700">{inner}</tspan>')
            width += w * 1.05
        elif kind == 'text':
            parts.append(f'<tspan font-family="{SANS}" font-style="normal">{escape(body)}</tspan>')
            width += len(body) * size * .52
        elif kind == 'sym':
            parts.append(f'<tspan font-style="{"italic" if body in "αβγδηθλμσστφψωℓ" else "normal"}">{escape(body)}</tspan>')
            width += size * (.62 if body not in '  ' else .5)
        else:
            if re.fullmatch(r'[A-Za-z]+', body) and body not in UPRIGHT and len(body) <= 4:
                parts.append(f'<tspan font-style="italic">{escape(body)}</tspan>')
                width += len(body) * size * .52
            elif re.fullmatch(r'[A-Za-z]+', body):
                parts.append(escape(body))
                width += len(body) * size * .5
            else:
                parts.append(escape(body))
                width += sum(size*(.3 if c == ' ' else .62 if c in '=+−-×' else .5) for c in body)
    return ''.join(parts), width


class Fig:
    """A 900-wide SVG card figure. Coordinates are SVG user units."""

    def __init__(self, name, desc, height=320, width=900):
        self.name, self.desc, self.height, self.width = name, desc, height, width
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
                      f'<title id="title">{escape(name)}</title><desc id="desc">{escape(desc)}</desc>', '<defs>']
        for key in ('ink', 'muted', 'blue', 'green', 'gold', 'red', 'purple', 'teal', 'line'):
            self.parts.append(f'<marker id="a-{key}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="{COL[key]}"/></marker>')
        self.parts += ['</defs>', f'<rect width="{width}" height="{height}" fill="white"/>']

    # ---- text -------------------------------------------------------------
    def text(self, x, y, s, size=18, color='ink', bold=False, anchor='middle', italic=False, family=SANS, opacity=1):
        style = ' font-style="italic"' if italic else ''
        op = f' fill-opacity="{opacity}"' if opacity != 1 else ''
        self.parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{family}" font-size="{size}" font-weight="{700 if bold else 400}" fill="{colour(color)}"{style}{op}>{escape(str(s))}</text>')

    def lines(self, x, y, rows, size=16, color='ink', anchor='start', gap=1.35, bold=False):
        for i, row in enumerate(rows):
            self.text(x, y + i*size*gap, row, size, color, bold, anchor)

    def math(self, x, y, expr, size=22, color='ink', anchor='start', bold=False):
        """Colour-coded formula. Example: r'\\c{purple}{S_t} = \\c{gold}{α_t} S_{t-1}'."""
        inner, width = _render_math(expr, size, colour(color), bold)
        self.parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{SERIF}" font-size="{size}" fill="{colour(color)}"{" font-weight=\"700\"" if bold else ""}>{inner}</text>')
        return width

    def label(self, x, y, s, color='ink', size=13, anchor='middle'):
        """Small caps-like annotation label."""
        self.text(x, y, s.upper(), size, color, True, anchor)

    # ---- shapes -----------------------------------------------------------
    def box(self, x, y, w, h, label='', color='blue', fill=False, size=17, opacity=1, rx=6, dash=False, text_color=None, stroke=1.6):
        c = colour(color)
        d = ' stroke-dasharray="5 4"' if dash else ''
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{c if fill else COL["paper"]}" fill-opacity="{opacity}" stroke="{c}" stroke-width="{stroke}"{d}/>')
        if label != '':
            tc = text_color or ('white' if fill and opacity > .55 else ('muted' if color == 'line' else color))
            if str(label).startswith('$'):
                self.math(x+w/2, y+h/2+size*.36, str(label)[1:], size, tc, 'middle')
            else:
                self.text(x+w/2, y+h/2+size*.36, label, size, tc)

    def rect(self, x, y, w, h, color='blue', opacity=1, rx=0, stroke=None):
        s = f' stroke="{colour(stroke)}" stroke-width="1"' if stroke else ''
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{colour(color)}" fill-opacity="{opacity}"{s}/>')

    def panel(self, x, y, w, h, title='', color='muted', fill='soft'):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{colour(fill)}" stroke="{COL["line"]}" stroke-width="1"/>')
        if title:
            self.text(x+14, y+24, title, 15, color, True, 'start')

    def circle(self, x, y, r, label='', color='blue', fill=False, size=17):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{colour(color) if fill else COL["white"]}" stroke="{colour(color)}" stroke-width="2"/>')
        if label != '':
            if str(label).startswith('$'):
                self.math(x, y+size*.36, str(label)[1:], size, 'white' if fill else color, 'middle')
            else:
                self.text(x, y+size*.36, label, size, 'white' if fill else color)

    def pill(self, x, y, label, color='green', size=13, anchor='start'):
        w = len(label)*size*.56 + 16
        x0 = x - w/2 if anchor == 'middle' else x
        self.parts.append(f'<rect x="{x0}" y="{y-size-4}" width="{w}" height="{size+10}" rx="{(size+10)/2}" fill="{colour(color)}" fill-opacity=".14" stroke="{colour(color)}" stroke-width="1"/>')
        self.text(x0+w/2, y+1, label, size, color, True)
        return w

    def line(self, x1, y1, x2, y2, color='line', width=1.8, dash=False, arrow=False, start_arrow=False, opacity=1):
        extra = f' marker-end="url(#a-{color})"' if arrow else ''
        if start_arrow:
            extra += f' marker-start="url(#a-{color})"'
        if dash:
            extra += ' stroke-dasharray="5 4"'
        if opacity != 1:
            extra += f' stroke-opacity="{opacity}"'
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colour(color)}" stroke-width="{width}"{extra}/>')

    def arrow(self, x1, y1, x2, y2, color='muted', dash=False, width=1.8, both=False):
        self.line(x1, y1, x2, y2, color, width, dash, True, both)

    def path(self, points, color='muted', arrow=False, width=1.8, dash=False, curve=False):
        if curve and len(points) == 3:
            (x0, y0), (cx, cy), (x1, y1) = points
            d = f'M{x0} {y0}Q{cx} {cy} {x1} {y1}'
        else:
            d = 'M' + 'L'.join(f'{x} {y}' for x, y in points)
        extra = f' marker-end="url(#a-{color})"' if arrow else ''
        if dash:
            extra += ' stroke-dasharray="5 4"'
        self.parts.append(f'<path d="{d}" fill="none" stroke="{colour(color)}" stroke-width="{width}"{extra}/>')

    def brace(self, x1, y1, x2, y2, label='', color='muted', side='above', size=14, offset=8):
        """A dimension bracket from (x1,y1) to (x2,y2) with a label; horizontal or vertical."""
        c = colour(color)
        if y1 == y2:
            s = -1 if side == 'above' else 1
            y = y1 + s*offset
            self.parts.append(f'<path d="M{x1} {y-s*5}L{x1} {y}L{x2} {y}L{x2} {y-s*5}" fill="none" stroke="{c}" stroke-width="1.2"/>')
            if label:
                ty = y - 6 if side == 'above' else y + size + 2
                (self.math if label.startswith('$') else self.text)((x1+x2)/2, ty, label.lstrip('$'), size, color, anchor='middle')
        else:
            s = -1 if side == 'left' else 1
            x = x1 + s*offset
            self.parts.append(f'<path d="M{x-s*5} {y1}L{x} {y1}L{x} {y2}L{x-s*5} {y2}" fill="none" stroke="{c}" stroke-width="1.2"/>')
            if label:
                tx = x - 6 if side == 'left' else x + 6
                (self.math if label.startswith('$') else self.text)(tx, (y1+y2)/2+size*.35, label.lstrip('$'), size, color, anchor='end' if side == 'left' else 'start')

    def callout(self, x, y, tx, ty, text, color='muted', size=14, anchor='start'):
        """Leader line from point (x,y) to text at (tx,ty)."""
        self.line(x, y, tx, ty - size*.35, color, 1, True)
        self.circle(x, y, 2.5, color=color, fill=True)
        self.text(tx + (4 if anchor == 'start' else -4 if anchor == 'end' else 0), ty, text, size, color, False, anchor)

    # ---- data-shaped helpers --------------------------------------------
    def matrix(self, x, y, rows, cols, cell=18, color='purple', values=None, gap=2, label='', rlabel='', clabel='', size=14, outline=False):
        """Heat-map matrix. values[r][c] in [0,1] sets opacity; None leaves the cell empty."""
        for r in range(rows):
            for c in range(cols):
                v = values[r][c] if values is not None else .2 + .6*((r*7 + c*3) % 9)/8
                if v is None:
                    self.parts.append(f'<rect x="{x+c*cell}" y="{y+r*cell}" width="{cell-gap}" height="{cell-gap}" rx="2" fill="{COL["paper"]}" stroke="{COL["line"]}" stroke-width=".8"/>')
                else:
                    tone = color
                    if isinstance(v, tuple):
                        v, tone = v
                    self.parts.append(f'<rect x="{x+c*cell}" y="{y+r*cell}" width="{cell-gap}" height="{cell-gap}" rx="2" fill="{colour(tone)}" fill-opacity="{max(.05, min(1, v)):.2f}"/>')
        W, H = cols*cell - gap, rows*cell - gap
        if outline:
            self.parts.append(f'<rect x="{x-3}" y="{y-3}" width="{W+6}" height="{H+6}" rx="4" fill="none" stroke="{colour(color)}" stroke-width="1.4"/>')
        if label:
            (self.math if label.startswith('$') else self.text)(x+W/2, y+H+size+8, label.lstrip('$'), size+2, color, anchor='middle')
        if clabel:
            self.brace(x, y, x+W, y, clabel, 'muted', 'above', size)
        if rlabel:
            self.brace(x, y, x, y+H, rlabel, 'muted', 'left', size)
        return W, H

    def mask(self, x, y, n, cell, fn, m=None, gap=1.5):
        """Query×key grid (rows = queries). fn(i, j) -> colour name, (colour, opacity) or None (not computed)."""
        m = m or n
        for i in range(n):
            for j in range(m):
                v = fn(i, j)
                if v is None:
                    self.parts.append(f'<rect x="{x+j*cell}" y="{y+i*cell}" width="{cell-gap}" height="{cell-gap}" fill="{COL["paper"]}"/>')
                else:
                    tone, op = (v if isinstance(v, tuple) else (v, .85))
                    self.parts.append(f'<rect x="{x+j*cell}" y="{y+i*cell}" width="{cell-gap}" height="{cell-gap}" fill="{colour(tone)}" fill-opacity="{op}"/>')
        return m*cell, n*cell

    def vec(self, x, y, n, cell=16, color='blue', vertical=True, values=None, label='', size=14, fill=True):
        for i in range(n):
            v = values[i] if values else .75
            cx, cy = (x, y+i*cell) if vertical else (x+i*cell, y)
            if fill:
                self.parts.append(f'<rect x="{cx}" y="{cy}" width="{cell-2}" height="{cell-2}" rx="2" fill="{colour(color)}" fill-opacity="{v}"/>')
            else:
                self.parts.append(f'<rect x="{cx}" y="{cy}" width="{cell-2}" height="{cell-2}" rx="2" fill="{COL["paper"]}" stroke="{colour(color)}"/>')
        if label:
            if vertical:
                (self.math if label.startswith('$') else self.text)(x+(cell-2)/2, y+n*cell+size+4, label.lstrip('$'), size+2, color, anchor='middle')
            else:
                (self.math if label.startswith('$') else self.text)(x+n*cell/2, y+cell+size+6, label.lstrip('$'), size+2, color, anchor='middle')

    def bars(self, x, y, items, width=520, bar=26, gap=14, vmax=None, unit='', label_w=150, size=15, value_fmt=None):
        """Horizontal bars: items = [(label, value, colour), ...]."""
        vmax = vmax or max(v for _, v, _ in items)
        for i, (name, v, c) in enumerate(items):
            yy = y + i*(bar+gap)
            self.text(x+label_w-10, yy+bar*.68, name, size, 'ink', anchor='end')
            w = max(2, width*v/vmax)
            self.rect(x+label_w, yy, w, bar, c, .85, 4)
            shown = value_fmt(v) if value_fmt else f'{v:g}{unit}'
            self.text(x+label_w+w+8, yy+bar*.68, shown, size, 'muted' if c == 'line' else c, True, 'start')
        return len(items)*(bar+gap) - gap

    def strip(self, x, y, cells, w=26, h=34, gap=3, size=12):
        """Layer stack as a horizontal strip: cells = [(label, colour), ...]."""
        for i, (lab, c) in enumerate(cells):
            self.box(x+i*(w+gap), y, w, h, lab, c, True, size, .88, 4)
        return len(cells)*(w+gap) - gap

    def records(self, x, y, n, w=40, h=46, gap=8, color='blue', selected=None, labels=None, size=14):
        for i in range(n):
            tone = color if selected is None else ('green' if i in selected else 'line')
            self.box(x+i*(w+gap), y, w, h, labels[i] if labels else '', tone, size=size)
            if not labels:
                for j in range(3):
                    self.line(x+i*(w+gap)+8, y+h*.3+j*h*.18, x+i*(w+gap)+w-8, y+h*.3+j*h*.18, tone, 1.4)

    def legend(self, x, y, items, size=13, gap=22, swatch=12):
        """Horizontal legend: items = [(label, colour), ...]; returns the end x."""
        for lab, c in items:
            self.rect(x, y-swatch+2, swatch, swatch, c, .85, 2)
            self.text(x+swatch+6, y, lab, size, 'ink', False, 'start')
            x += swatch + 6 + len(lab)*size*.52 + gap
        return x

    def divider(self, x, y1=24, y2=None):
        self.line(x, y1, x, (y2 or self.height-20), 'line', 1.2)

    def hdivider(self, y, x1=20, x2=None):
        self.line(x1, y, x2 or self.width-20, y, 'line', 1.2)

    def save(self, out=OUT):
        out.mkdir(parents=True, exist_ok=True)
        (out/(self.name + '.svg')).write_text('\n'.join(self.parts + ['</svg>\n']))
        INVENTORY.append({'file': self.name + '.svg', 'description': self.desc})
