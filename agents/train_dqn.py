"""Train one DQN baseline checkpoint. Run from the repo root:

    python -m agents.train_dqn --seed 0
"""

import argparse
from pathlib import Path
from time import perf_counter

import torch
from stable_baselines3 import DQN

from sim.make_env import load_yaml, make_env


def main():
    parser = argparse.ArgumentParser(description="Train DQN on the shared highway environment.")
    parser.add_argument("--seed", type=int, choices=load_yaml("seeds.yaml")["dqn_train"], default=0)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--output-dir", type=Path, default=Path("models/final"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    checkpoint = args.output_dir / f"dqn_seed{args.seed}.zip"
    if checkpoint.exists():
        parser.error(f"{checkpoint} already exists; use another --output-dir for a rerun")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    env = make_env()
    try:
        # Use the HighwayEnv DQN quickstart settings with the team's shared environment.
        model = DQN(
            "MlpPolicy",
            env,
            policy_kwargs={"net_arch": [256, 256]},
            learning_rate=5e-4,
            buffer_size=15_000,
            learning_starts=200,
            batch_size=32,
            gamma=0.8,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=50,
            seed=args.seed,
            device="cpu",
            verbose=1,
        )
        start = perf_counter()
        model.learn(total_timesteps=args.steps, log_interval=50)
        elapsed = perf_counter() - start
        model.save(checkpoint)
        print(f"Trained {model.num_timesteps:,} steps in {elapsed:.1f}s. Saved {checkpoint}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
