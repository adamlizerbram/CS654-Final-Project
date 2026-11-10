set -euo pipefail
unset SDL_VIDEODRIVER || true
unset SDL_AUDIODRIVER || true

rm -rf manish_autorunner/status/clips
mkdir -p manish_autorunner/status/clips

python -m eval.evaluate \
  --agent random \
  --episodes 3 \
  --out results/video_smoke.csv \
  --summary-out results/summary_video_smoke.csv \
  --video-dir manish_autorunner/status/clips \
  --video-episodes 2 \
  --video-vertical

python -m eval.evaluate \
  --agent idle \
  --episodes 2 \
  --out results/video_smoke.csv \
  --append \
  --summary-out results/summary_video_smoke.csv \
  --video-dir manish_autorunner/status/clips \
  --video-episodes 1 \
  --video-vertical

ls -lh manish_autorunner/status/clips
