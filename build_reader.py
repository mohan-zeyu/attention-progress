#!/usr/bin/env python3
"""Build the offline card reader: one page per chapter, a print page for the PDF, a source register and a manifest."""
from pathlib import Path
from html import escape, unescape
from urllib.parse import urlsplit
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'reader'
FILTER = ROOT / 'scripts' / 'cards.lua'


def title(path):
    match = re.search(r'^#\s+(.+)$', path.read_text(), re.M)
    return match.group(1) if match else path.stem


def lede(path):
    match = re.search(r'^::: lede\n(.+?)\n:::', path.read_text(), re.M | re.S)
    return re.sub(r'\$([^$]+)\$', r'\1', match.group(1)).strip() if match else ''


def cards(path):
    return len(re.findall(r'^## ', path.read_text(), re.M))


def destination(path):
    rel = path.relative_to(ROOT)
    return OUT / ('index.html' if rel == Path('README.md') else rel.with_suffix('.html'))


def relative(path, current):
    return os.path.relpath(path, current.parent).replace(os.sep, '/')


def canonical(url):
    match = re.search(r'arxiv\.org/(?:abs|html|pdf)/(\d{4}\.\d{4,5})', url)
    return 'https://arxiv.org/abs/' + match.group(1) if match else url.rstrip('/')


def merge_sources(chapters):
    merged, by_title = {}, {}
    for path in sorted((ROOT/'bibliography').glob('*.json')):
        if path.name == 'papers.json':
            continue
        for item in json.loads(path.read_text()):
            key = canonical(item['url'])
            title_key = re.sub(r'\W+', '', item['title'].lower())
            key = by_title.setdefault(title_key, key) if key not in merged else key
            if key not in merged:
                merged[key] = dict(item, url=key)
            else:
                old = merged[key]
                old['chapters'] = sorted(set(old['chapters'] + item['chapters']))
                if item.get('evidence_note') and item['evidence_note'] not in old.get('evidence_note', ''):
                    old['evidence_note'] = (old.get('evidence_note', '') + ' ' + item['evidence_note']).strip()
                if old.get('reading_depth') == 'abstract' and item.get('reading_depth') not in (None, 'abstract'):
                    old['reading_depth'] = item['reading_depth']
    records = sorted(merged.values(), key=lambda x: (x['year'] or 9999, x['title']))
    (ROOT/'bibliography'/'papers.json').write_text(json.dumps(records, indent=2, ensure_ascii=False)+'\n')
    chapter_paths = {p.name[:2]: p for p in chapters}
    lines = ['# Primary sources', '',
             f'**{len(records)} papers, technical reports, model cards and official implementations** support the cards. '
             '“sections” = the relevant full-text sections or config files were checked; “abstract” = abstract-level characterization; '
             '“official report” = official project, model card or code. Worked numbers are teaching constructions; no experiments were reproduced.', '',
             '| Year | Source | Venue / org | Chapters | Checked |', '| --- | --- | --- | --- | --- |']
    for item in records:
        label = item['title'].replace('|', '\\|')
        links = ', '.join(f'[{n}](../chapters/{chapter_paths[n].name})' for n in item['chapters'] if n in chapter_paths)
        note = (item.get('evidence_note') or '').replace('|', '\\|').replace('\n', ' ')
        venue = (item.get('venue') or '').replace('|', '\\|')
        year = item['year'] if item['year'] is not None else 'Living project'
        lines.append(f'| {year} | [{label}]({item["url"]}) | {venue} | {links} | **{item.get("reading_depth", "")}** {note} |')
    (ROOT/'bibliography'/'sources.md').write_text('\n'.join(lines)+'\n')
    return records


def nav(current, chapters):
    def link(path, label, search=''):
        target = destination(path)
        active = ' aria-current="page"' if target == current else ''
        return f'<li data-search="{escape(search or label.lower(), quote=True)}"><a href="{escape(relative(target, current))}"{active}>{escape(label)}</a></li>'
    rows = [link(ROOT/'README.md', 'Book contents')]
    for p in chapters:
        terms = ' '.join(sorted(set(re.findall(r'[A-Za-z][A-Za-z0-9²-]*', p.read_text().lower()))))
        rows.append(link(p, title(p), terms))
    reference_rows = [link(ROOT/name, label) for name, label in [
        ('appendices/mechanism-matrix.md', 'Mechanism matrix'),
        ('appendices/glossary.md', 'Notation and glossary'),
        ('bibliography/reference-note.md', 'Reference article and coverage'),
        ('bibliography/sources.md', 'Primary sources')] if (ROOT/name).exists()]
    return ('<a class="skip" href="#book-text">Skip to chapter</a><nav class="book-nav" aria-label="Book navigation">'
            '<details open id="book-navigation"><summary>Attention<br>by pictures</summary>'
            '<input id="chapter-search" type="search" placeholder="Find a mechanism" aria-label="Find a chapter or mechanism">'
            '<span class="nav-label">Chapters</span><ul id="chapter-list">' + ''.join(rows) +
            '</ul><p id="no-chapters" hidden>No matching chapters.</p><span class="nav-label">Reference</span><ul>' +
            ''.join(reference_rows) + '</ul></details></nav>')


def normalize_math(source):
    source = re.sub(r'\\\[(.*?)\\\]', lambda m: '$$\n'+m.group(1).strip()+'\n$$', source, flags=re.S)
    return re.sub(r'\\\((.*?)\\\)', lambda m: '$'+m.group(1).strip()+'$', source, flags=re.S)


def rewrite_links(document, source, current):
    def rewrite(match):
        attr, quote, raw = match.groups()
        value = unescape(raw)
        parts = urlsplit(value)
        if parts.scheme or value.startswith(('#', '//')):
            return match.group(0)
        resolved = (source.parent/parts.path).resolve()
        if resolved.suffix == '.md':
            resolved = destination(resolved)
        elif resolved.suffix in ('.svg', '.png') and ROOT/'figures' in resolved.parents:
            resolved = OUT/resolved.relative_to(ROOT)
        rewritten = relative(resolved, current)
        if parts.query:
            rewritten += '?'+parts.query
        if parts.fragment:
            rewritten += '#'+parts.fragment
        return f'{attr}={quote}{escape(rewritten, quote=True)}{quote}'
    return re.sub(r'(href|src)=(["\'])(.*?)\2', rewrite, document)


def word_count(source):
    """Approximate prose length, excluding URLs, figure alt text, math and HTML comments."""
    text = normalize_math(source.read_text())
    text = re.sub(r'\$\$.*?\$\$', '', text, flags=re.S)
    text = re.sub(r'\$[^$\n]+\$', '', text)
    text = re.sub(r'!\[.*?\]\([^)]*\)', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.S)
    text = re.sub(r'^:::.*$', '', text, flags=re.M)
    return len(re.findall(r"\b\w+(?:[’'-]\w+)*\b", text))


def render(source, current, prefix='', toc=True):
    args = ['pandoc', '--from=markdown+tex_math_dollars-implicit_figures', '--to=html5', '--mathml', '--section-divs', f'--lua-filter={FILTER}']
    if toc:
        args += ['--toc', '--toc-depth=2']
    result = subprocess.run(args, input=normalize_math(source.read_text()), text=True, capture_output=True, check=True)
    if result.stderr.strip():
        print(f'[pandoc] {source.relative_to(ROOT)}: {result.stderr.strip()}')
    body = rewrite_links(result.stdout, source, current)
    body = re.sub(r'(<nav id="TOC".*?</nav>)', r'<div class="page-toc">\1</div>', body, flags=re.S)
    body = re.sub(r'(<table>.*?</table>)', r'<div class="table-wrap">\1</div>', body, flags=re.S)
    body = re.sub(r'<p>(<img\b[^>]*\bsrc="([^"]+\.svg)"[^>]*>)</p>',
                  lambda m: '<figure class="card-figure"><a class="diagram-link" href="'+m.group(2)+'" target="_blank" rel="noopener" title="Open full-size diagram">'+m.group(1)+'</a></figure>', body)
    if prefix:
        body = re.sub(r'\bid="([^"]+)"', lambda m: 'id="'+prefix+m.group(1)+'"', body)
        body = re.sub(r'href="#([^"]+)"', lambda m: 'href="#'+prefix+m.group(1)+'"', body)
    return body


DELTA_DEMO = '''<div class="delta-demo" role="group" aria-label="Try a partial delta update">
<label for="edit-strength">Write strength β <output id="edit-beta" for="edit-strength">1.00</output></label>
<input id="edit-strength" type="range" min="0" max="1" step="0.05" value="1" aria-label="Delta update strength beta">
<div class="demo-track" aria-hidden="true"><span id="edit-fill"></span><i class="old-marker"></i></div>
<output id="edit-result" for="edit-strength" aria-live="polite">2 + 1.00 × 3 = 5</output>
</div>'''


def landing(current, chapters):
    first = relative(destination(chapters[0]), current)
    tiles = []
    for p in chapters:
        label = title(p)
        number, text = label.split(' · ', 1) if ' · ' in label else ('', label)
        tiles.append(f'<a class="chapter-card" href="{escape(relative(destination(p), current))}"><span>{escape(number)}</span>'
                     f'<strong>{escape(text)}</strong><small>{escape(lede(p))}</small><em>{cards(p)} cards</em></a>')
    pdf_link = '<a class="pdf-link" href="../attention-visual-guide.pdf">PDF version</a>' if (ROOT/'attention-visual-guide.pdf').exists() else ''
    total = sum(cards(p) for p in chapters)
    cover = ROOT/'figures'/'cards'/'map-routes.svg'
    cover_html = (f'<a class="diagram-link cover-diagram" href="figures/cards/map-routes.svg" target="_blank" rel="noopener" title="Open full-size diagram">'
                  f'<img src="figures/cards/map-routes.svg" alt="The routes of attention development."></a>') if cover.exists() else ''
    return ('<header class="book-cover"><p class="eyebrow">A visual field guide · ' + str(len(chapters)) + ' chapters · ' + str(total) + ' mechanism cards</p>'
            '<h1>Attention,<br>one change at a time.</h1>'
            '<p>Each card covers one mechanism: what it changes, the formula, a diagram with real numbers, and the catch.</p>'
            '<a class="start-reading" href="' + escape(first) + '">Start with the map →</a>' + pdf_link + '</header>' + cover_html +
            '<div class="chapter-grid">' + ''.join(tiles) + '</div>')


def footer(source, current, chapters):
    if source in chapters:
        i = chapters.index(source)
        before = chapters[i-1] if i else ROOT/'README.md'
        after = chapters[i+1] if i+1 < len(chapters) else ROOT/'appendices'/'mechanism-matrix.md'
        return (f'<footer class="chapter-footer"><a href="{escape(relative(destination(before), current))}">← {escape(title(before))}</a>'
                f'<a class="next" href="{escape(relative(destination(after), current))}">{escape(title(after) if after.exists() else "")} →</a></footer>')
    return f'<footer class="chapter-footer"><a href="{escape(relative(OUT/"index.html", current))}">← Book contents</a></footer>'


SCRIPT = '''<script>
const bookNav = document.getElementById('book-navigation');
if (window.innerWidth <= 900) bookNav.open = false;
const search = document.getElementById('chapter-search');
search.addEventListener('input', () => {
  const term = search.value.trim().toLowerCase(); let visible = 0;
  document.querySelectorAll('#chapter-list li').forEach(row => {
    row.hidden = !row.dataset.search.includes(term); if (!row.hidden) visible++;
  });
  document.getElementById('no-chapters').hidden = visible > 0;
});
function revealSection() {
  let id;
  try { id = decodeURIComponent(window.location.hash.slice(1)); } catch { return; }
  const target = id && document.getElementById(id);
  const more = target && target.closest('details.more');
  if (more) { more.open = true; requestAnimationFrame(() => target.scrollIntoView()); }
}
window.addEventListener('hashchange', revealSection);
revealSection();
const strength = document.getElementById('edit-strength');
if (strength) {
  function updateEdit() {
    const beta = Number(strength.value);
    const prediction = 2 + beta * 3;
    document.getElementById('edit-beta').value = beta.toFixed(2);
    document.getElementById('edit-result').value = '2 + ' + beta.toFixed(2) + ' × 3 = ' + Number(prediction.toFixed(2));
    document.getElementById('edit-fill').style.width = (prediction / 5 * 100) + '%';
  }
  strength.addEventListener('input', updateEdit);
  updateEdit();
}
</script>'''


def page(doc_title, css, body, navigation='', script=''):
    return ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{escape(doc_title)}</title><style>{css}</style></head><body>{navigation}<main id="book-text">{body}</main>{script}</body></html>\n')


def print_page(chapters, css):
    """All chapters on one page, for the PDF export."""
    current = OUT/'print.html'
    parts = ['<header class="book-cover print-cover"><p class="eyebrow">A visual field guide</p><h1>Attention,<br>one change at a time.</h1>'
             f'<p>{len(chapters)} chapters · {sum(cards(p) for p in chapters)} mechanism cards · formulas, diagrams, numbers and catches.</p></header>']
    for p in chapters:
        body = render(p, current, prefix=p.stem+'-', toc=False)
        parts.append(f'<article class="print-chapter card-chapter">{body}</article>')
    body = '\n'.join(parts).replace('<!-- delta-demo -->', '')
    current.write_text(page('Attention, one change at a time', css, body).replace('<body>', '<body class="print">'))


def build():
    if not shutil.which('pandoc'):
        raise SystemExit('pandoc is required to rebuild this reader')
    OUT.mkdir(exist_ok=True)
    chapters = sorted((ROOT/'chapters').glob('[0-9][0-9]-*.md'))
    sys.path.insert(0, str(ROOT/'scripts'))
    import build_matrix
    build_matrix.build()
    records = merge_sources(chapters)
    sources = [ROOT/'README.md'] + chapters + sorted((ROOT/'appendices').glob('*.md')) + \
        [p for p in (ROOT/'bibliography'/'reference-note.md', ROOT/'bibliography'/'sources.md') if p.exists()]
    css = (ROOT/'style.css').read_text()
    for stale in (OUT/'chapters').glob('*.html'):
        stale.unlink()
    lengths = []
    for source in sources:
        current = destination(source)
        current.parent.mkdir(parents=True, exist_ok=True)
        if source == ROOT/'README.md':
            body = landing(current, chapters)
        else:
            body = render(source, current, toc=source in chapters)
            body = body.replace('<!-- delta-demo -->', DELTA_DEMO)
            if source in chapters:
                body = '<div class="card-chapter">' + body + '</div>'
                lengths.append({'chapter': source.name[:2], 'cards': cards(source), 'words': word_count(source)})
        doc_title = title(source) + ' · Attention, one change at a time'
        current.write_text(page(doc_title, css, body + footer(source, current, chapters), nav(current, chapters), SCRIPT))
    if (OUT/'figures').exists():
        shutil.rmtree(OUT/'figures')
    for figure in (ROOT/'figures'/'cards').rglob('*.svg'):
        target = OUT/'figures'/figure.relative_to(ROOT/'figures')
        target.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(figure, target)
    print_page(chapters, css)
    manifest = {'title': 'Attention, one change at a time', 'entry': 'index.html', 'checked_through': '2026-10-21',
                'format': 'mechanism cards: formula, diagram, numbers, tips', 'chapter_pages': len(chapters),
                'cards': sum(item['cards'] for item in lengths), 'card_figures': len(list((ROOT/'figures'/'cards').glob('*.svg'))),
                'primary_sources': len(records), 'words': sum(item['words'] for item in lengths),
                'chapters': [{'title': title(p), 'page': destination(p).relative_to(OUT).as_posix()} for p in chapters], 'lengths': lengths}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    (ROOT/'index.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="0; url=reader/index.html"><title>Attention, one change at a time</title></head><body><p><a href="reader/index.html">Open the attention book</a></p></body></html>\n')
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('chapters', 'lengths')}, ensure_ascii=False))
    check_links()


def check_links():
    """Report internal links whose target page, anchor or figure does not exist."""
    pages = {p: p.read_text() for p in OUT.rglob('*.html') if p.name != 'print.html'}
    ids = {p: set(re.findall(r'\bid="([^"]+)"', text)) for p, text in pages.items()}
    broken = []
    for page_path, text in pages.items():
        for raw in re.findall(r'(?:href|src)="([^"]+)"', text):
            value = unescape(raw)
            parts = urlsplit(value)
            if parts.scheme or value.startswith('//'):
                continue
            target = (page_path.parent/parts.path).resolve() if parts.path else page_path
            if not target.exists():
                broken.append(f'{page_path.relative_to(OUT)} → {value} (missing file)')
            elif parts.fragment and target.suffix == '.html' and parts.fragment not in ids.get(target, set()):
                broken.append(f'{page_path.relative_to(OUT)} → {value} (missing anchor)')
    for line in sorted(set(broken)):
        print('[link]', line)
    if not broken:
        print('links: all internal links resolve')


if __name__ == '__main__':
    build()
