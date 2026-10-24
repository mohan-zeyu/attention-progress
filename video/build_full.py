#!/usr/bin/env python3
"""Join the six chapter renders into renders/full_series.mp4 (with chapter markers) and full_series.srt.

Each chapter's audio is padded to its video length first, so the audio cannot drift
across chapter boundaries when the streams are concatenated without re-encoding video.
"""
import json
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / 'renders'
CHAPTERS = [('ch1_linear', 'Ch1Linear', '1 · Linear attention'), ('ch2_decay', 'Ch2Decay', '2 · Forgetting: decay'),
            ('ch3_delta', 'Ch3Delta', '3 · Correcting: the delta rule'), ('ch4_transition', 'Ch4Transition', '4 · One axis: the transition matrix'),
            ('ch5_learning', 'Ch5Learning', '5 · The state as a model trained while reading'), ('ch6_limits', 'Ch6Limits', '6 · Limits')]


def duration(path, stream='v'):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', f'{stream}:0', '-show_entries', 'stream=duration',
                          '-of', 'csv=p=0', str(path)], capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def parse_srt(text):
    for block in text.strip().split('\n\n'):
        lines = block.splitlines()
        a, b = lines[1].split(' --> ')
        yield a, b, '\n'.join(lines[2:])


def secs(stamp):
    h, m, s = stamp.replace(',', '.').split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def stamp(t):
    ms = int(round(t * 1000))
    return f'{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}'


def main():
    tmp = Path(tempfile.mkdtemp())
    parts, meta, cues, spans, t = [], [';FFMETADATA1'], [], [], 0.0
    for name, scene, title in CHAPTERS:
        src, padded = R / f'{name}.mp4', tmp / f'{name}.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(src), '-c:v', 'copy', '-af', 'apad', '-c:a', 'aac', '-b:a', '160k',
                        '-shortest', str(padded)], check=True)
        d = duration(padded)
        parts.append(padded)
        meta += ['[CHAPTER]', 'TIMEBASE=1/1000', f'START={int(t * 1000)}', f'END={int((t + d) * 1000)}', f'title={title}']
        for a, b, s in parse_srt((R / f'{name}.srt').read_text()):
            cues.append((secs(a) + t, secs(b) + t, s))
        sp = HERE / 'build' / 'srt' / f'{scene}.spans.json'
        if sp.exists():
            spans += [(f'{name}:{k}', a + t, b + t) for k, a, b in json.loads(sp.read_text())]
        t += d
    (tmp / 'list.txt').write_text(''.join(f"file '{p}'\n" for p in parts))
    (tmp / 'meta.txt').write_text('\n'.join(meta) + '\n')
    out = R / 'full_series.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(tmp / 'list.txt'), '-i', str(tmp / 'meta.txt'),
                    '-map', '0', '-map_metadata', '1', '-c', 'copy', '-movflags', '+faststart', str(out)], check=True)
    (R / 'full_series.srt').write_text('\n'.join(f'{i}\n{stamp(a)} --> {stamp(b)}\n{s}\n' for i, (a, b, s) in enumerate(cues, 1)))
    (HERE / 'build' / 'full_series.spans.json').write_text(json.dumps(spans))
    print(f'{out.name}: {duration(out):.2f}s video, {duration(out, "a"):.2f}s audio, {out.stat().st_size / 1e6:.1f} MB; '
          f'chapters sum {t:.2f}s')


if __name__ == '__main__':
    main()
