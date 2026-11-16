"""Generate MOCK / SYNTHETIC shield decisions and a labeled visualization."""
import csv
from pathlib import Path
from types import SimpleNamespace

from agents.laya_shield import LayaShieldAgent

NAMES = ("LANE_LEFT", "IDLE", "LANE_RIGHT")
Q_VALUES = (4.0, 3.0, 5.0)
UNSAFE = (0.20, 0.10, 0.85)
THRESHOLDS = (0.05, 0.50, 0.90)


def decisions():
    env = SimpleNamespace(action_space=SimpleNamespace(n=3))
    env.unwrapped = SimpleNamespace(action_type=SimpleNamespace(get_available_actions=lambda: [0, 1, 2]))
    obs = {"description": "MOCK highway state; three available actions"}
    preferred = max(range(len(Q_VALUES)), key=lambda a: (Q_VALUES[a], -a))
    rows = []
    for threshold in THRESHOLDS:
        calls = []

        def score(env, obs, action):
            calls.append(action)
            return UNSAFE[action]

        shield = LayaShieldAgent(threshold=threshold, q_values=lambda env, obs: Q_VALUES,
                                unsafe_score=score)
        selected = shield.act(env, obs)
        fallback = len(calls) == len(Q_VALUES) and all(UNSAFE[a] > threshold for a in calls)
        rows.append({"data_type": "MOCK / SYNTHETIC", "threshold": threshold,
                     "dqn_preferred_action": NAMES[preferred], "shield_selected_action": NAMES[selected],
                     "override": shield.last_override, "all_unsafe_fallback": fallback})
    return rows


def write_outputs(rows, output_dir=Path("results/track5")):
    """Create new outputs only; never overwrite an existing artifact."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "mock_synthetic_shield_decisions.csv"
    png_path = output_dir / "mock_synthetic_shield_decisions.png"
    if csv_path.exists() or png_path.exists():
        raise FileExistsError("Mock output exists; use a fresh output directory")
    with csv_path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    try:
        axes[0].barh(NAMES, Q_VALUES, color="#326da8")
        axes[0].set(xlabel="Mock DQN Q-value", title="DQN preference: LANE_RIGHT", xlim=(0, 6))
        for i, value in enumerate(Q_VALUES):
            axes[0].text(value + 0.04, i, f"{value:.1f}", va="center")
        axes[1].barh(NAMES, UNSAFE, color="#d18145")
        for i, value in enumerate(UNSAFE):
            axes[1].text(value + 0.02, i, f"{value:.2f}", va="center")
        for row, color in zip(rows, ("#9a2230", "#8051a4", "#25745d")):
            axes[1].axvline(row["threshold"], color=color, linestyle="--", linewidth=1.4,
                            label=f"threshold {row['threshold']:.2f}")
        axes[1].set(xlim=(0, 1.16), xlabel="Mock Laya unsafe score", title="Threshold and shield decision")
        axes[1].legend(fontsize="small", loc="lower right")
        legend = [f"{r['threshold']:.2f}: {r['shield_selected_action']} | override={r['override']}"
                  + (" | fallback" if r['all_unsafe_fallback'] else "") for r in rows]
        fig.text(0.5, 0.04, "\n".join(legend), ha="center", va="bottom", fontsize=9)
        fig.suptitle("MOCK / SYNTHETIC — Shield decisions, not model predictions", fontsize=12)
        fig.tight_layout(rect=(0, 0.18, 1, 0.93))
        with png_path.open("xb") as f:
            fig.savefig(f, format="png", dpi=150)
    finally:
        plt.close(fig)
    return csv_path, png_path


def main():
    rows = decisions()
    print("MOCK / SYNTHETIC — illustrative values, not model predictions")
    print("DQN preferred: LANE_RIGHT (2), Q=5.00")
    print("Candidates (action id, name, Q-value, unsafe score):")
    for action in sorted(range(len(Q_VALUES)), key=lambda a: (-Q_VALUES[a], a)):
        print(f"  {action} {NAMES[action]:10s} Q={Q_VALUES[action]:.2f} risk={UNSAFE[action]:.2f}")
    for row in rows:
        print(f"threshold={row['threshold']:.2f} | selected={row['shield_selected_action']} | "
              f"override={row['override']} | all-unsafe fallback={row['all_unsafe_fallback']}")
    csv_path, png_path = write_outputs(rows)
    print(f"Wrote MOCK / SYNTHETIC outputs: {csv_path}, {png_path}")


if __name__ == "__main__":
    main()
