# Fixed-Size States — the video series

A narrated, animated walk through all six sections of [`Fixed-Size States.md`](../Fixed-Size%20States.md): linear attention and everything built on it. One video per section, with an English voiceover, on-screen captions for key terms, and an `.srt` subtitle file for each video.

## Watch

| # | Video | Covers (doc section) | Length | Size |
| --- | --- | --- | --- | --- |
| 1 | [`renders/ch1_linear.mp4`](renders/ch1_linear.mp4) | §1 Linear attention: write/read/discard, 16,384 vs 2·N·128, crosstalk, 7.4× vs 1.7×, three equal forms | 6:34 | 15.3 MB |
| 2 | [`renders/ch2_decay.mp4`](renders/ch2_decay.mp4) | §2 Decay: one mask, RetNet, Lightning, Mamba-2 and SSD, GLA per-channel decay, Mamba-3 | 5:16 | 12.5 MB |
| 3 | [`renders/ch3_delta.mp4`](renders/ch3_delta.mp4) | §3 Delta rule: read/compare/overwrite, the order toy, I−βkkᵀ, Gated DeltaNet, KDA, sum form, parity, WY/UT | 6:34 | 16.1 MB |
| 4 | [`renders/ch4_transition.mp4`](renders/ch4_transition.mp4) | §4 The transition matrix S_t = S_{t−1}A_t + W_t, row by row, commuting, 2026 refinements | 4:04 | 10.8 MB |
| 5 | [`renders/ch5_learning.mp4`](renders/ch5_learning.mp4) | §5 The state as a model: gradient-step view, TTT, Titans, ATLAS, Mesa, LaCT | 5:38 | 14.2 MB |
| 6 | [`renders/ch6_limits.mp4`](renders/ch6_limits.mp4) | §6 Limits: d_k exact pairs, recall numbers, log-linear attention, hybrids, series recap | 4:06 | 9.2 MB |
| all | [`renders/full_series.mp4`](renders/full_series.mp4) | all six back to back, with chapter markers | 32:11 | 85.7 MB |

All files are 1920×1080, 30 fps, H.264 video with AAC audio. Each video has a matching `.srt` next to it (`full_series.srt` for the joined file). The subtitles come straight from the voiceover timings. To read along, the narration text for each chapter is in [`script/`](script/).

Each chapter opens with the problem it solves, animates the mechanism, and ends with a recap card (the key formula plus one line) and a **Left over:** line that leads into the next chapter.

## How it is built

```
video/
  scenes/common.py        palette (from scripts/figkit.py + style.css), LaTeX colour macros, voiceover + SRT, helpers
  scenes/chN_*.py         one Manim scene per chapter
  script/chN_*.md         narration, split into <!-- segment --> blocks; the scenes speak these blocks
  render.sh               render chapters and copy mp4 + srt into renders/
  build_full.py           join the chapters into renders/full_series.mp4 (+ chapter markers, full_series.srt)
  check_audio.py          confirm every narration segment is audible where the scene placed it
  requirements.txt        exact Python package pins (pip freeze of the build venv)
```

The script files are the single source of narration. A scene says `with self.say('write'):` and the animations inside that block run while that segment plays. The block then waits for the audio to finish, and `pause=` adds a short beat after a key formula. Lines in quote blocks (`> $…$`) are the on-screen formulas for readers and are not spoken. `PRON` in `common.py` maps names the phonemizer would garble ("GLA" → "G L A", "LaCT" → "lact"), so the script can keep the normal spelling.

## Re-render

Tested on Ubuntu 24.04 (noble), Python 3.13.

```bash
# system packages: LaTeX for MathTex, pango/cairo for Manim, ffmpeg, the Inter font
sudo apt-get install -y --no-install-recommends texlive-latex-base texlive-latex-recommended texlive-latex-extra \
  texlive-fonts-recommended texlive-science texlive-lang-english dvisvgm cm-super libpango1.0-dev pkg-config \
  ffmpeg fonts-inter

# Python packages, in a venv (Debian's own setuptools fails to build the `srt` dependency)
python3 -m venv .venv && . .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r video/requirements.txt

# Kokoro TTS model files (offline after this); KOKORO_DIR overrides the location
mkdir -p video/.models && cd video/.models
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
cd ../..

cd video
QUALITY=l ./render.sh ch3_delta:Ch3Delta          # quick 480p15 preview (stays in build/)
./render.sh ch1_linear:Ch1Linear ch2_decay:Ch2Decay ch3_delta:Ch3Delta \
            ch4_transition:Ch4Transition ch5_learning:Ch5Learning ch6_limits:Ch6Limits   # 1080p30 finals
python3 check_audio.py renders/ch3_delta.mp4 build/srt/Ch3Delta.spans.json
python3 build_full.py
```

Synthesized speech is cached in `video/.cache/tts/` and keyed on the text, voice and speed, so changing one sentence only re-synthesizes that segment. Manim also caches unchanged animations. On a 4-core cloud VM, a 6-minute chapter took about 2 minutes at 480p15 and about 5 minutes at 1080p30.

Model checksums (sha256):
`kokoro-v1.0.onnx` 7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5,
`voices-v1.0.bin` bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d.

### Pinned versions

| Tool | Version |
| --- | --- |
| Manim Community | 0.19.0 (ManimPango 0.7.0, pycairo 1.29.2, av 13.1.0) |
| TTS | kokoro-onnx 0.4.9, onnxruntime 1.31.0, phonemizer-fork 3.3.2, espeakng-loader 0.2.4 |
| Python / NumPy | 3.13.17 / 2.5.4 |
| LaTeX | TeX Live 2023 (Ubuntu `texlive-latex-base 2023.20240207-1`), dvisvgm 3.2.1 |
| ffmpeg | 6.1.1 |
| Font | Inter 4.0 (`fonts-inter`) |

### TTS engine

**Kokoro-82M v1.0** (an offline neural TTS), run through `kokoro-onnx` on CPU with the fp32 ONNX model, voice **`af_heart`** (American English) at speed 1.0. It synthesizes at about 4.5× real time on 4 CPU cores. Kokoro was the first choice, and it worked. In the build environment Hugging Face was blocked, but the GitHub release mirror of the model files was reachable. Neither gTTS, edge-tts, espeak-ng nor captions-only was needed.

### Two gotchas, already handled

- `manim -qh` renders 1080p at **60** fps. `render.sh` passes `--frame_rate 30` to get 1080p30.
- `Scene.add_sound()` silently drops a sound when the previous `play()` came from Manim's partial-movie cache. On re-renders that removed whole narration segments. `VoiceScene.say()` adds the audio through the file writer instead, and `check_audio.py` flags any segment that is not audible where it should be.

## Faithfulness notes

Every claim, name and number in the narration comes from `Fixed-Size States.md`. The chapter intros follow Part 4 of `Cheaper Attention.md`, and Chapter 6's "a number to repeat or a passage to copy" comes from essay §4.5. A few pictures use illustrative values that the doc does not give. They are labelled on screen or are plain arithmetic:

- Ch 1: the outer-product and read values; the 2-D crosstalk read (v₁ = 3, v₃ = 2, overlap 0.71); the "ten weak matches" shares (computed from the scores); the eight outputs computed three ways (a seeded random example, checked equal in code). The 4-key versus 8-key overlap grids reuse the formula in `scripts/state_figures.py`.
- Ch 2: the α = 0.8 mask, the per-token Mamba-2 decays, the Δ-knob bars and the Mamba-3 rotation spiral are shapes, not data.
- Ch 4: the key-space ellipses use chosen matrices to show what each kind of A_t does; the order-toy bars reuse the doc's numbers.
- Ch 5: the mini-batch picture uses b = 4 for legibility (the doc's TTT uses b = 16, which is stated); the Titans surprise/momentum bars are illustrative shapes.
- Ch 6: the capacity curve is a small simulation computed in the scene (8 × 8 state, random unit keys, additive versus exact least-squares writes), labelled as a simulation; the "plain additive" bar is drawn about 4 points under Mesa, as the doc states; the hybrid layer mix is labelled as an illustration. "Double the context, one more state" follows from the doc's power-of-two buckets.

## What was checked

- Each chapter was previewed at 480p15. One frame per narration segment was tiled into contact sheets and looked at, and overlaps, clipping and stray labels were fixed.
- Each final 1080p render was checked again on about 8 sampled frames.
- `ffprobe` on every file: 1920×1080, 30 fps, H.264 + AAC. The audio runs the full length (it ends with the final hold, at most 1.7 s before the video ends).
- `check_audio.py` passes on every chapter and on `full_series.mp4`: each segment starts with silence and turns to speech at exactly the time the scene recorded, so the audio stays in sync across chapter boundaries.
- Not checked: no human has listened to the audio. Pronunciation was steered with the `PRON` table, but odd stresses on names (for example "Mamba", "Comba", "Mesa") are possible.
