"""Sample real Laya unsafe scores and plot action acceptance by threshold.

These are model scores, not collision rates or driving-performance results.
Run on SDSU: python -m figures.track5_threshold_analysis --finetuned PATH --out-dir results/track5/threshold_analysis
"""
from __future__ import annotations

import argparse
import csv
import gc
import math
from collections import defaultdict
from pathlib import Path

THRESHOLDS = (0.01, 0.05, 0.10, 0.30, 0.50, 0.70, 0.90, 0.95, 0.99)
SEEDS = (100000, 100001, 100002)
COLUMNS = ("model", "model_source", "env_seed", "state_description", "action_id", "action_name", "unsafe_score", "threshold", "accepted")


def collect_scores(models, seeds=SEEDS, env_factory=None, agent_factory=None):
    """Score all legal actions on the same seeded reset states for both models."""
    from laya_tools.highway_to_laya import ACTIONS, describe_state
    if env_factory is None:
        from sim.make_env import make_env
        env_factory = make_env
    if agent_factory is None:
        from agents.laya_shield import LayaShieldAgent
        agent_factory = LayaShieldAgent
    baseline = {}
    records = []
    for label, source in models:
        env = env_factory()
        try:
            agent = agent_factory(laya_model=source, q_values=lambda env, obs: [0.0] * env.action_space.n)
            for seed in seeds:
                obs, _ = env.reset(seed=int(seed))
                state = describe_state(env)
                actions = tuple(sorted(env.unwrapped.action_type.get_available_actions()))
                if not actions:
                    raise ValueError(f"No available actions for seed {seed}")
                if label == models[0][0]:
                    baseline[seed] = (state, actions)
                elif baseline.get(seed) != (state, actions):
                    raise ValueError(f"State or available actions differ for seed {seed}")
                for action in actions:
                    value = float(agent._unsafe_score(env, obs, action))
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError(f"Invalid unsafe score for {label}, seed {seed}, action {action}")
                    records.append({"model": label, "model_source": source, "env_seed": int(seed),
                                    "state_description": state, "action_id": int(action),
                                    "action_name": ACTIONS[action], "unsafe_score": value})
                    print(f"model={label} seed={seed} action={ACTIONS[action]} score={value:.6f}", flush=True)
        finally:
            env.close()
        del agent
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    return records


def threshold_rows(records, thresholds=THRESHOLDS):
    return [{**record, "threshold": threshold, "accepted": record["unsafe_score"] <= threshold}
            for record in records for threshold in thresholds]


def acceptance(records, thresholds=THRESHOLDS):
    grouped = defaultdict(list)
    for row in records:
        grouped[row["model"]].append(row["unsafe_score"])
    return {model: [100 * sum(score <= threshold for score in scores) / len(scores)
                    for threshold in thresholds] for model, scores in grouped.items()}


def write_outputs(records, out_dir):
    """Create a new output directory to avoid overwriting earlier experiments."""
    import matplotlib.pyplot as plt
    if not records:
        raise ValueError("No scores to analyze")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    csv_path = out_dir / "action_thresholds.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(threshold_rows(records))
    grouped = defaultdict(list)
    for row in records:
        grouped[row["model"]].append(row["unsafe_score"])
    seeds = ", ".join(map(str, sorted({r["env_seed"] for r in records})))
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        for label, scores in grouped.items():
            ax.hist(scores, bins=[i / 10 for i in range(11)], alpha=0.5, label=f"{label} (n={len(scores)} actions)")
        ax.set(xlabel="Laya estimated unsafe score", ylabel="Candidate-action count", title=f"Unsafe-score distribution\nSeeds: {seeds}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "unsafe_score_distribution.png", dpi=150)
    finally:
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        for label, percentages in acceptance(records).items():
            ax.plot(THRESHOLDS, percentages, marker="o", label=f"{label} (n={len(grouped[label])} actions)")
        ax.set(xlabel="Unsafe-score threshold (score ≤ threshold accepted)", ylabel="Candidate actions accepted (%)",
               ylim=(-5, 105), title=f"Action-level threshold acceptance\nSeeds: {seeds}")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "threshold_acceptance.png", dpi=150)
    finally:
        plt.close(fig)
    return csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finetuned", required=True, help="Existing local fine-tuned Laya checkpoint")
    parser.add_argument("--out-dir", required=True, help="New output directory; must not exist")
    args = parser.parse_args()
    models = [("zero-shot", "convaiinnovations/laya"), ("fine-tuned", args.finetuned)]
    records = collect_scores(models)
    path = write_outputs(records, args.out_dir)
    print(f"Wrote {len(records)} model/action scores and {len(records) * len(THRESHOLDS)} threshold rows to {path}")
    for model, rates in acceptance(records).items():
        print(f"{model} accepted percentages: " + ", ".join(f"{t:g}={v:.1f}%" for t, v in zip(THRESHOLDS, rates)))


if __name__ == "__main__":
    main()
