set -euo pipefail
unset SDL_VIDEODRIVER || true
unset SDL_AUDIODRIVER || true

# 1. Ensure fine-tuned Laya weights exist
if [ ! -d "models/final/laya-highway" ]; then
  mkdir -p models/final
  hf download adamlizerbram/cs654-final-project --include "laya-highway/*" --local-dir models/final
fi

# 2. Train any missing DQN seed checkpoints (0..4) in parallel
pids=()
for seed in 0 1 2 3 4; do
  if [ ! -f "models/final/dqn_seed${seed}.zip" ]; then
    echo "Training DQN seed ${seed}..."
    python -m agents.train_dqn --seed "${seed}" &
    pids+=($!)
  fi
done
for pid in "${pids[@]+"${pids[@]}"}"; do
  wait "$pid"
done

# 3. Preview readiness (all 78 jobs should now be READY)
python -m eval.run_benchmark --dry-run

# 4. Run 5-seed DQN baseline (500 episodes total)
python -m eval.run_benchmark --only dqn --episodes 100 --skip-existing

# 5. Record quick 1-episode vertical demo clips of DQN and shielded DQN
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

# 6. Run full shield sweeps (checkpoints after every job via --skip-existing)
python -m eval.run_benchmark --only shield_finetuned shield_zeroshot --episodes 100 --skip-existing
