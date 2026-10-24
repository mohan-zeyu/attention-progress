#!/usr/bin/env python3
"""Check a rendered chapter: every narration segment must be audible where the scene placed it.

usage: check_audio.py renders/<chapter>.mp4 build/srt/<Scene>.spans.json
"""
import json, subprocess, sys
import numpy as np

mp4, spans = sys.argv[1], sys.argv[2]
sr = 8000
pcm = subprocess.run(['ffmpeg', '-v', 'error', '-i', mp4, '-map', '0:a', '-ac', '1', '-ar', str(sr), '-f', 's16le', '-'],
                     capture_output=True, check=True).stdout
a = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768
probe = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,duration', '-of', 'json', mp4],
                                  capture_output=True, check=True).stdout)
durs = {s['codec_type']: float(s['duration']) for s in probe['streams']}
bad = 0
for key, t0, t1 in json.load(open(spans)):
    seg = a[int((t0 + 0.05) * sr): int((t0 + 1.2) * sr)]
    rms = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
    pre = a[max(0, int((t0 - 0.25) * sr)): int((t0 - 0.02) * sr)]
    lead = float(np.sqrt(np.mean(pre ** 2))) if len(pre) else 0.0
    flag = 'MISSING' if rms < 0.01 else ''
    bad += bool(flag)
    print(f'{key:12s} start {t0:7.2f}s  speech rms {rms:.3f}  before {lead:.3f} {flag}')
print(f"video {durs.get('video', 0):.2f}s  audio {durs.get('audio', 0):.2f}s  segments missing: {bad}")
sys.exit(1 if bad else 0)
