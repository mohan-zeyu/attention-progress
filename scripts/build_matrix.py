#!/usr/bin/env python3
"""Generate appendices/mechanism-matrix.md from the cards: one row per card, grouped by lever."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
LEVER = {'00': 'Map', '01': 'Positions & stability', '02': 'Positions & stability', '03': 'Thinner records', '04': 'Thinner records',
         '05': 'Read fewer records', '06': 'Read fewer records', '07': 'Keep fewer records', '08': 'Keep fewer records',
         '09': 'Fixed state', '10': 'Fixed state', '11': 'Fixed state', '12': 'Mix both', '13': 'Move fewer bytes',
         '14': 'Move fewer bytes', '15': 'Compare'}


def plain(text, keep_math=False):
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    if not keep_math:
        text = re.sub(r'\$([^$]+)\$', r'\1', text)
    return text.replace('**', '').replace('|', '/').strip()


def cards(path):
    source = path.read_text()
    for block in re.split(r'^## ', source, flags=re.M)[1:]:
        head = block.splitlines()[0]
        anchor = re.search(r'\{#([^\s}]+)', head)
        title = plain(re.sub(r'\s*\{#[^}]*\}\s*$', '', head), keep_math=True)
        meta = re.search(r'^::: meta\n(.+?)\n:::', block, re.M | re.S)
        meta = plain(meta.group(1)) if meta else ''
        used = ''
        if 'Used in:' in meta:
            meta, used = meta.split('Used in:', 1)
        years = sorted(set(re.findall(r'\b(20[12]\d)\b', meta)))
        when = years[0] if len(years) == 1 else (f'{years[0]}–{years[-1]}' if years else '')
        yield title, anchor.group(1) if anchor else '', when, used.strip(' ·—'), '.insight' in head


def build():
    chapters = sorted((ROOT/'chapters').glob('[0-9][0-9]-*.md'))
    lines = ['# Mechanism matrix', '',
             'Every card in the book, grouped by the lever it pulls. Generated from the chapter files by `scripts/build_matrix.py`. Insight cards (costs, comparisons) are in italics.', '']
    current = None
    for path in chapters:
        number = path.name[:2]
        lever = LEVER.get(number, '')
        if lever != current:
            lines += ['', f'## {lever}', '', '| Ch. | Card | Year | Used in |', '| --- | --- | --- | --- |']
            current = lever
        for title, anchor, when, used, insight in cards(path):
            label = f'*{title}*' if insight else title
            lines.append(f'| {number} | [{label}](../chapters/{path.name}#{anchor}) | {when} | {used} |')
    (ROOT/'appendices'/'mechanism-matrix.md').write_text('\n'.join(lines) + '\n')
    return sum(1 for line in lines if line.startswith('| ') and not line.startswith('| Ch.') and not line.startswith('| ---'))


if __name__ == '__main__':
    print(f'{build()} rows')
