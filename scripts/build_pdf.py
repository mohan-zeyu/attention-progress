#!/usr/bin/env python3
"""Print reader/print.html (all cards, MathML and SVG intact) to attention-visual-guide.pdf with headless Chrome."""
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', 'google-chrome', 'chromium', 'chromium-browser']


def chrome():
    for candidate in CANDIDATES:
        if Path(candidate).exists() or shutil.which(candidate):
            return candidate
    raise SystemExit('Chrome or Chromium is required to export the PDF')


def build():
    source = ROOT/'reader'/'print.html'
    if not source.exists():
        raise SystemExit('Run build_reader.py first')
    target = ROOT/'attention-visual-guide.pdf'
    subprocess.run([chrome(), '--headless=new', '--disable-gpu', '--no-pdf-header-footer', '--virtual-time-budget=8000',
                    f'--print-to-pdf={target}', source.as_uri()], check=True, capture_output=True)
    pages = subprocess.run(['pdfinfo', str(target)], capture_output=True, text=True).stdout if shutil.which('pdfinfo') else ''
    count = next((line.split()[-1] for line in pages.splitlines() if line.startswith('Pages:')), '?')
    print(f'Exported {target.name}: {count} pages')


if __name__ == '__main__':
    build()
    sys.exit(0)
