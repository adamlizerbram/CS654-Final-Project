set -euo pipefail
unset SDL_VIDEODRIVER || true
unset SDL_AUDIODRIVER || true

# 1. Download exact shared weights (DQN seeds 0..4 + fine-tuned Laya) from Hugging Face
mkdir -p models/final
hf download adamlizerbram/cs654-final-project \
  --include "dqn_seed*.zip" "laya-highway/*" \
  --local-dir models/final

# 2. Drop any previous locally-trained dqn/shield rows while keeping random/laya_* rows
python - << 'EOF'
import csv
from pathlib import Path
from eval.results_schema import COLUMNS

raw_path = Path("results/raw_episodes.csv")
if raw_path.exists():
    keep = {"random", "laya_zeroshot", "laya_finetuned"}
    with raw_path.open("r", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("method") in keep]
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"Kept {len(rows)} non-DQN baseline rows in {raw_path}")
EOF

# 3. Run 5-seed DQN baseline using Noah's Hugging Face checkpoints
python -m eval.run_benchmark --only dqn --episodes 100 --skip-existing

# 4. Record 1-episode vertical demo clips of Noah's DQN and shielded DQN
rm -rf manish_autorunner/status/clips
mkdir -p manish_autorunner/status/clips
python -m eval.evaluate \
  --agent-spec agents.dqn_agent:DQNAgent \
  --agent-kwargs '{"model_path": "models/final/dqn_seed0.zip"}' \
  --model-seed 0 \
  --episodes 1 \
  --out results/demo_dqn.csv \
  --video-dir manish_autorunner/status/clips \
  --video-episodes 1 \
  --video-vertical

python -m eval.evaluate \
  --agent-spec agents.laya_shield:LayaShieldAgent \
  --agent-kwargs '{"dqn_path": "models/final/dqn_seed0.zip", "laya_model": "models/final/laya-highway", "threshold": 0.5}' \
  --method shield_finetuned \
  --model-seed 0 \
  --episodes 1 \
  --out results/demo_shield.csv \
  --video-dir manish_autorunner/status/clips \
  --video-episodes 1 \
  --video-vertical

# 5. Run full shield sweeps on Noah's exact DQN checkpoints
python -m eval.run_benchmark --only shield_finetuned shield_zeroshot --episodes 100 --skip-existing
