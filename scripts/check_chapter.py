#!/usr/bin/env python3
"""Lint one chapter against planning/card-style.md: python3 scripts/check_chapter.py chapters/06-sparse-reads.md"""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def prose_words(text):
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.S)
    text = re.sub(r'\$[^$\n]+\$', '', text)
    text = re.sub(r'!\[.*?\]\([^)]*\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.S)
    text = re.sub(r'^::: meta\n.*?\n:::', '', text, flags=re.M | re.S)
    text = re.sub(r'^:::.*$', '', text, flags=re.M)
    text = re.sub(r'^\|.*\|$', '', text, flags=re.M)
    return len(re.findall(r"\b\w+(?:[’'-]\w+)*\b", text))


def check(path):
    path = Path(path).resolve()
    source = path.read_text()
    problems = []
    result = subprocess.run(['pandoc', '--from=markdown+tex_math_dollars-implicit_figures', '--to=html5', '--mathml',
                             f'--lua-filter={ROOT/"scripts"/"cards.lua"}'], input=source, text=True, capture_output=True)
    if result.stderr.strip():
        problems.append('pandoc: ' + result.stderr.strip().replace('\n', '\n        '))
    if not re.search(r'^# \d\d · ', source, re.M):
        problems.append('title must be "# NN · Title"')
    if '::: lede' not in source:
        problems.append('missing ::: lede')
    cards = re.split(r'^## ', source, flags=re.M)[1:]
    print(f'{path.name}: {len(cards)} cards')
    for card in cards:
        head = card.splitlines()[0]
        figures = re.findall(r'!\[(?:[^\[\]]|\[[^\]]*\])*\]\(([^)]+\.svg)\)', card)
        missing = [f for f in figures if not (path.parent/f).resolve().exists()]
        words = prose_words(card.split('\n', 1)[1] if '\n' in card else '')
        flags = []
        if '::: meta' not in card and '.insight' not in head:
            flags.append('no meta')
        if '$$' not in card:
            flags.append('no display formula')
        if not figures:
            flags.append('no figure')
        if missing:
            flags.append('missing figure files: ' + ', '.join(missing))
        if '::: tips' not in card:
            flags.append('no tips')
        if words > 110:
            flags.append(f'{words} prose words (budget 90)')
        print(f'  [{"OK " if not flags else "FIX"}] {head[:70]:70} {words:3d}w ' + ('; '.join(flags)))
        problems += [f'{head[:40]}: {flag}' for flag in flags]
    for problem in problems:
        print('  !', problem)
    return not problems


if __name__ == '__main__':
    ok = all([check(p) for p in sys.argv[1:]])
    sys.exit(0 if ok else 1)
