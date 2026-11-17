"""Generate report figures from results/summary.csv (no GPU required).

  python -m figures.benchmark_plots
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_CSV = REPO_ROOT / "results" / "summary.csv"
CRASH_VS_SPEED_PNG = REPO_ROOT / "figures" / "crash_vs_speed.png"
THRESHOLD_SWEEP_PNG = REPO_ROOT / "figures" / "threshold_sweep.png"


def _load_summary(csv_path: Path = SUMMARY_CSV) -> list[dict[str, Any]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, Any], key: str) -> float:
    return float(row[key])


def plot_crash_vs_speed(
    rows: list[dict[str, Any]],
    out_path: Path = CRASH_VS_SPEED_PNG,
) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_key = {(r["method"], r["threshold"]): r for r in rows}
    points = [
        ("random", "", "Random", "#7f8c8d", "x", (10, -6)),
        ("laya_zeroshot", "", "Laya (Zero-Shot)", "#d62728", "s", (-112, -6)),
        ("laya_finetuned", "", "Laya (Fine-Tuned)", "#1f77b4", "^", (-118, -22)),
        ("dqn", "", "DQN Alone", "#0ea5e9", "D", (8, 8)),
        ("shield_zeroshot", "0.7", "Shield (Zero-Shot, thr=0.7)", "#ff7f0e", "v", (8, -4)),
        ("shield_finetuned", "0.5", "Shield (Fine-Tuned, thr=0.5)", "#2ca02c", "o", (8, 8)),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=200)
    for method, thr, label, color, marker, offset in points:
        row = by_key.get((method, thr))
        if row is None:
            continue
        x = _f(row, "avg_speed_mean")
        x_err = _f(row, "avg_speed_ci95")
        y = _f(row, "crash_rate_mean") * 100.0
        y_err = _f(row, "crash_rate_ci95") * 100.0
        ax.errorbar(
            x, y, xerr=x_err, yerr=y_err,
            fmt=marker, color=color, markersize=8, capsize=4, elinewidth=1.5,
            label=label, zorder=4,
        )
        ax.annotate(
            f"{label}\n({y:.1f}% crash)",
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            color=color,
            fontweight="semibold",
        )

    ax.set_title("Safety vs. Speed Trade-off on highway-fast-v0 (100 Eval Seeds)")
    ax.set_xlabel("Average Speed (m/s, mean ± 95% CI)")
    ax.set_ylabel("Crash Rate (%, mean ± 95% CI)")
    ax.set_ylim(-6, 112)
    ax.set_xlim(21.0, 26.8)
    ax.grid(True, linestyle="--", alpha=0.35)
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
    laya_ft_row = next((r for r in rows if r["method"] == "laya_finetuned"), None)

    def _series(method: str) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float]]:
        sub = sorted(
            (r for r in rows if r["method"] == method and r["threshold"] != ""),
            key=lambda r: float(r["threshold"]),
        )
        thrs = [float(r["threshold"]) for r in sub]
        crash = [_f(r, "crash_rate_mean") * 100.0 for r in sub]
        crash_ci = [_f(r, "crash_rate_ci95") * 100.0 for r in sub]
        reward = [_f(r, "reward_mean") for r in sub]
        reward_ci = [_f(r, "reward_ci95") for r in sub]
        ov = [_f(r, "override_rate_mean") * 100.0 for r in sub]
        return thrs, crash, crash_ci, reward, reward_ci, ov

    zs_t, zs_c, zs_cc, zs_r, zs_rc, zs_o = _series("shield_zeroshot")
    ft_t, ft_c, ft_cc, ft_r, ft_rc, ft_o = _series("shield_finetuned")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=200)

    # Panel 1: Crash Rate
    ax = axes[0]
    if zs_t:
        ax.errorbar(zs_t, zs_c, yerr=zs_cc, fmt="v-", color="#ff7f0e", capsize=3, label="Shield (Zero-Shot Laya)")
    if ft_t:
        ax.errorbar(ft_t, ft_c, yerr=ft_cc, fmt="o-", color="#2ca02c", capsize=3, label="Shield (Fine-Tuned Laya)")
    if dqn_row:
        ax.axhline(_f(dqn_row, "crash_rate_mean") * 100.0, color="#0ea5e9", linestyle="--", label="DQN Alone")
    if laya_ft_row:
        ax.axhline(_f(laya_ft_row, "crash_rate_mean") * 100.0, color="#1f77b4", linestyle=":", label="Laya Fine-Tuned Alone")
    ax.set_title("Crash Rate vs. Threshold")
    ax.set_xlabel("P(unsafe) Threshold")
    ax.set_ylabel("Crash Rate (%)")
    ax.set_ylim(-2, 78)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=7.5, loc="center right")

    # Panel 2: Episode Reward
    ax = axes[1]
    if zs_t:
        ax.errorbar(zs_t, zs_r, yerr=zs_rc, fmt="v-", color="#ff7f0e", capsize=3, label="Shield (Zero-Shot)")
    if ft_t:
        ax.errorbar(ft_t, ft_r, yerr=ft_rc, fmt="o-", color="#2ca02c", capsize=3, label="Shield (Fine-Tuned)")
    if dqn_row:
        ax.axhline(_f(dqn_row, "reward_mean"), color="#0ea5e9", linestyle="--", label="DQN Alone")
    ax.set_title("Episode Reward vs. Threshold")
    ax.set_xlabel("P(unsafe) Threshold")
    ax.set_ylabel("Mean Episode Return")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=7.5, loc="lower right")

    # Panel 3: Override Rate
    ax = axes[2]
    if zs_t:
        ax.plot(zs_t, zs_o, "v-", color="#ff7f0e", label="Zero-Shot Override %")
    if ft_t:
        ax.plot(ft_t, ft_o, "o-", color="#2ca02c", label="Fine-Tuned Override %")
    ax.set_title("Shield Override Rate")
    ax.set_xlabel("P(unsafe) Threshold")
    ax.set_ylabel("Actions Overridden (%)")
    ax.set_ylim(40, 80)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=7.5, loc="lower left")

    fig.suptitle("Laya Safety Shield Threshold Sweep (5 DQN Seeds × 100 Eval Episodes)", fontsize=12, fontweight="semibold")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot() -> Path:
    rows = _load_summary()
    plot_crash_vs_speed(rows)
    return plot_threshold_sweep(rows)


def main() -> int:
    rows = _load_summary()
    p1 = plot_crash_vs_speed(rows)
    p2 = plot_threshold_sweep(rows)
    print(f"Wrote {p1} and {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
