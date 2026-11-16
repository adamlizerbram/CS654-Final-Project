set -euo pipefail
unset SDL_VIDEODRIVER || true
unset SDL_AUDIODRIVER || true

# 1. Download fine-tuned Laya weights from Track 4 if not present yet
if [ ! -d "models/final/laya-highway" ]; then
  mkdir -p models/final
  hf download adamlizerbram/cs654-final-project --include "laya-highway/*" --local-dir models/final
fi

# 2. Preview benchmark readiness
python -m eval.run_benchmark --dry-run

# 3. Run all 100 eval episodes for the 3 ready conditions
python -m eval.run_benchmark \
  --only random laya_zeroshot laya_finetuned \
  --episodes 100 \
  --skip-existing

# 4. Record 1 vertical demo clip of fine-tuned Laya driving
rm -rf manish_autorunner/status/clips
mkdir -p manish_autorunner/status/clips
python -m eval.evaluate \
  --agent-spec agents.laya_agent:LayaAgent \
  --agent-kwargs '{"model_name": "models/final/laya-highway", "name": "laya_finetuned"}' \
  --episodes 1 \
  --out results/laya_demo.csv \
  --video-dir manish_autorunner/status/clips \
  --video-episodes 1 \
  --video-vertical
