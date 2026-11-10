set -euo pipefail
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy

rm -rf manish_autorunner/status/videos
mkdir -p manish_autorunner/status/videos

python -c "import gymnasium, highway_env, pygame; print('gym/highway/pygame ok')"
python -c "import moviepy; print('moviepy:', getattr(moviepy, '__version__', 'ok'))" || pip install -q moviepy

python -m eval.evaluate \
  --agent random \
  --episodes 3 \
  --out results/video_smoke.csv \
  --summary-out results/summary_video_smoke.csv \
  --video-dir manish_autorunner/status/videos \
  --video-episodes 2

python -m eval.evaluate \
  --agent idle \
  --episodes 2 \
  --out results/video_smoke.csv \
  --append \
  --summary-out results/summary_video_smoke.csv \
  --video-dir manish_autorunner/status/videos \
  --video-episodes 1

ls -lh manish_autorunner/status/videos
