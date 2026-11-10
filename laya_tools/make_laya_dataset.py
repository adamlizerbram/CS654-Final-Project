"""Generate Laya fine-tuning data from highway-env by simulating each action.

For every visited state and every legal action:
  1. copy the environment,
  2. apply the action, then keep driving (IDLE) for a short horizon,
  3. label the action "unsafe" if the copy crashes or the gap to another
     car gets dangerously small.

Writes two kinds of JSONL rows in Laya's training format:
  - shield rows:  state + proposed action -> noul "unsafe" (true/false)
  - policy rows:  state -> choice "action" (best safe action by progress)

Usage (from the repo root):
  python -m laya_tools.make_laya_dataset --episodes 300 --out data/train.jsonl --seed-offset 0
  python -m laya_tools.make_laya_dataset --episodes 50  --out data/eval.jsonl  --seed-offset 10000
Use different seed ranges for train and eval so test scenes are never seen in training.
"""
import argparse
import copy
import json
import random
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401

from laya_tools.highway_to_laya import (ACTIONS, available_actions, action_choice_question,
                             describe_state, describe_with_action, safety_question)

NAME_TO_ID = {v: k for k, v in ACTIONS.items()}
HORIZON = 3          # lookahead steps (1 step = 1 s at the default policy_frequency)
MIN_SAFE_GAP = 5.0   # meters; a smaller bumper-to-bumper gap counts as unsafe
# Tie-break order when several actions are safe and make similar progress.
PREFERENCE = ["IDLE", "FASTER", "LANE_LEFT", "LANE_RIGHT", "SLOWER"]


def min_gap(env):
    u = env.unwrapped
    ego = u.vehicle
    gaps = [abs(v.position[0] - ego.position[0]) - ego.LENGTH
            for v in u.road.vehicles
            if v is not ego and v.lane_index == ego.lane_index]
    return min(gaps, default=float("inf"))


def simulate(env, action_name):
    """Return (unsafe, progress_m) for taking action_name now, then IDLE."""
    sim = copy.deepcopy(env)
    start_x = sim.unwrapped.vehicle.position[0]
    actions = [NAME_TO_ID[action_name]] + [NAME_TO_ID["IDLE"]] * (HORIZON - 1)
    for a in actions:
        _, _, terminated, truncated, info = sim.step(a)
        if info.get("crashed") or min_gap(sim) < MIN_SAFE_GAP:
            return True, 0.0
        if terminated or truncated:
            break
    return False, sim.unwrapped.vehicle.position[0] - start_x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="highway-fast-v0")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--out", default="train.jsonl")
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--explore", type=float, default=0.5,
                   help="probability of a random action when driving the data-collection car")
    args = p.parse_args()

    env = gym.make(args.env)
    rng = random.Random(args.seed_offset)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_rows = n_unsafe = 0
    with open(args.out, "w") as f:
        for ep in range(args.episodes):
            env.reset(seed=args.seed_offset + ep)
            done = False
            while not done:
                legal = available_actions(env)
                results = {a: simulate(env, a) for a in legal}

                # Shield rows: one per (state, action).
                for a, (unsafe, _) in results.items():
                    f.write(json.dumps({
                        "state": describe_with_action(env, a),
                        "questions": safety_question(),
                        "expected": {"unsafe": unsafe},
                    }) + "\n")
                    n_rows += 1
                    n_unsafe += unsafe

                # Policy row: the safe action with the most progress.
                safe = [a for a in legal if not results[a][0]]
                if safe:
                    best = max(safe, key=lambda a: (round(results[a][1]), -PREFERENCE.index(a)))
                    f.write(json.dumps({
                        "state": describe_state(env),
                        "questions": action_choice_question(env),
                        "expected": {"action": best},
                    }) + "\n")
                    n_rows += 1

                # Drive the data-collection car: mix of random and safest-looking actions,
                # so the dataset contains risky situations, not only easy ones.
                if rng.random() < args.explore or not safe:
                    a = rng.choice(legal)
                else:
                    a = best
                _, _, terminated, truncated, _ = env.step(NAME_TO_ID[a])
                done = terminated or truncated
            print(f"episode {ep + 1}/{args.episodes}: {n_rows} rows, {n_unsafe} unsafe labels")


if __name__ == "__main__":
    main()
