#!/usr/bin/env bash
# Render chapters and copy the results into renders/.
#   ./render.sh ch1_linear:Ch1Linear [more file:Scene ...]   final 1080p30
#   QUALITY=l ./render.sh ch1_linear:Ch1Linear               quick 480p15 preview (not copied)
set -euo pipefail
cd "$(dirname "$0")"
Q=${QUALITY:-h}
for spec in "$@"; do
  file=${spec%%:*}; scene=${spec##*:}
  if [ "$Q" = h ]; then manim -qh --frame_rate 30 "scenes/$file.py" "$scene"; else manim -q"$Q" "scenes/$file.py" "$scene"; fi
  if [ "$Q" = h ]; then
    cp "build/media/videos/$file/1080p30/$scene.mp4" "renders/$file.mp4"
    cp "build/srt/$scene.srt" "renders/$file.srt"
  fi
done
