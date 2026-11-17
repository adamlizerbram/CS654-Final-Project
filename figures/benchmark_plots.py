"""Generate clean, easy-to-read report figures from results/summary.csv.

  python -m figures.benchmark_plots
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_CSV = REPO_ROOT / "results" / "summary.csv"
CRASH_RATE_PNG = REPO_ROOT / "figures" / "crash_rate_comparison.png"
DISTANCE_SPEED_PNG = REPO_ROOT / "figures" / "distance_and_speed.png"
THRESHOLD_SWEEP_PNG = REPO_ROOT / "figures" / "threshold_sweep.png"


def _load_summary(csv_path: Path = SUMMARY_CSV) -> list[dict[str, Any]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def _comparison_rows(rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any], str]]:
    by_key = {(r["method"], r["threshold"]): r for r in rows}
    specs = [
        ("Laya (Zero-Shot)", ("laya_zeroshot", ""), "#ef4444"),
        ("Random Baseline", ("random", ""), "#94a3b8"),
        ("DQN + Zero-Shot Shield (thr=0.5)", ("shield_zeroshot", "0.5"), "#f97316"),
        ("Laya (Fine-Tuned)", ("laya_finetuned", ""), "#3b82f6"),
        ("DQN Alone", ("dqn", ""), "#0ea5e9"),
        ("DQN + Fine-Tuned Shield (thr=0.5)", ("shield_finetuned", "0.5"), "#22c55e"),
    ]
    out = []
    for label, key, color in specs:
        if key in by_key:
            out.append((label, by_key[key], color))
    return out


def plot_crash_rate_bars(
    rows: list[dict[str, Any]],
    out_path: Path = CRASH_RATE_PNG,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = _comparison_rows(rows)
    labels = [x[0] for x in items]
    vals = [_f(x[1], "crash_rate_mean") * 100.0 for x in items]
    colors = [x[2] for x in items]

    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=200)
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, vals, color=colors, height=0.62)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.invert_yaxis()  # worst at top, best (4.2%) at bottom
    ax.set_xlim(0, 114)
    ax.set_xlabel("Crash Rate (%) — Lower is Safer", fontsize=10)
    ax.set_title("How Often Each Driver Crashes (100 Unseen Highway Scenes)", fontsize=11.5, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, v in zip(bars, vals):
        ax.text(v + 1.5, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9.5, fontweight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_distance_and_speed(
    rows: list[dict[str, Any]],
    out_path: Path = DISTANCE_SPEED_PNG,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = _comparison_rows(rows)
    labels = [x[0] for x in items]
    dists = [_f(x[1], "distance_mean") for x in items]
    speeds = [_f(x[1], "avg_speed_mean") for x in items]
    colors = [x[2] for x in items]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=200, sharey=True)
    y_pos = list(range(len(labels)))

    b1 = ax1.barh(y_pos, dists, color=colors, height=0.62)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=9.5)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 780)
    ax1.set_xlabel("Average Distance Driven (m) — Farther is Better", fontsize=9.5)
    ax1.set_title("Distance Before Episode Ends", fontsize=10.5, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.35)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    for bar, v in zip(b1, dists):
        ax1.text(v + 10, bar.get_y() + bar.get_height() / 2, f"{v:.0f} m", va="center", fontsize=9)

    b2 = ax2.barh(y_pos, speeds, color=colors, height=0.62)
    ax2.set_xlim(0, 31)
    ax2.set_xlabel("Average Speed (m/s) — Higher is Faster", fontsize=9.5)
    ax2.set_title("Driving Speed", fontsize=10.5, fontweight="bold")
    ax2.grid(axis="x", linestyle="--", alpha=0.35)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for bar, v in zip(b2, speeds):
        ax2.text(v + 0.4, bar.get_y() + bar.get_height() / 2, f"{v:.1f} m/s", va="center", fontsize=9)

    fig.suptitle("Driving Performance Comparison (100 Unseen Highway Scenes)", fontsize=11.5, fontweight="bold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_threshold_sweep(
    rows: list[dict[str, Any]],
    out_path: Path = THRESHOLD_SWEEP_PNG,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dqn_row = next((r for r in rows if r["method"] == "dqn"), None)

    def _series(method: str) -> tuple[list[float], list[float]]:
        sub = sorted(
            (r for r in rows if r["method"] == method and r["threshold"] != ""),
            key=lambda r: float(r["threshold"]),
        )
        return (
            [float(r["threshold"]) for r in sub],
            [_f(r, "crash_rate_mean") * 100.0 for r in sub],
        )

    zs_t, zs_c = _series("shield_zeroshot")
    ft_t, ft_c = _series("shield_finetuned")

    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=200)
    if zs_t:
        ax.plot(zs_t, zs_c, "v-", color="#f97316", linewidth=2.2, markersize=7, label="DQN + Zero-Shot Laya Shield")
    if ft_t:
        ax.plot(ft_t, ft_c, "o-", color="#22c55e", linewidth=2.4, markersize=7, label="DQN + Fine-Tuned Laya Shield")
    if dqn_row:
        ax.axhline(_f(dqn_row, "crash_rate_mean") * 100.0, color="#0ea5e9", linestyle="--", linewidth=1.8, label="DQN Alone (31.0%)")

    ax.set_title("Effect of Shield Risk Threshold on Crash Rate", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Safety Threshold — lower = stricter shield overrides")
    ax.set_ylabel("Crash Rate (%) — Lower is Safer")
    ax.set_ylim(0, 75)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="center right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot() -> Path:
    rows = _load_summary()
    plot_crash_rate_bars(rows)
    plot_distance_and_speed(rows)
    return plot_threshold_sweep(rows)


def main() -> int:
    rows = _load_summary()
    p1 = plot_crash_rate_bars(rows)
    p2 = plot_distance_and_speed(rows)
    p3 = plot_threshold_sweep(rows)
    print(f"Wrote {p1}, {p2}, and {p3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
