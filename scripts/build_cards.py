#!/usr/bin/env python3
"""Draw every mechanism-card figure: one module per chapter in scripts/cards/, each with build()."""
from pathlib import Path
import importlib.util
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figkit  # noqa: E402


def main(only=None):
    modules = sorted((HERE/'cards').glob('*.py'))
    for path in modules:
        if path.name.startswith('_') or (only and only not in path.stem):
            continue
        spec = importlib.util.spec_from_file_location('cards_'+path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        before = len(figkit.INVENTORY)
        module.build()
        print(f'{path.stem}: {len(figkit.INVENTORY)-before} figures')
    if not only:
        (figkit.OUT/'inventory.json').write_text(json.dumps(figkit.INVENTORY, indent=2, ensure_ascii=False)+'\n')
    print(f'Drew {len(figkit.INVENTORY)} card figures')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
