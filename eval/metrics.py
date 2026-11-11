"""Metrics aggregation and summary table generation for Track 1 evaluation.

Reads episode-level CSV files conforming to ``eval.results_schema.COLUMNS`` and
produces summary tables (``results/summary*.csv``) with mean and 95% confidence
intervals across model seeds (for multi-seed RL methods) or across episodes
(for single-model methods like Laya alone).

Usage (from the repo root):
  python -m eval.metrics --input results/raw_episodes.csv --out results/summary.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

# Two-tailed 95% Student's t critical values for df = 1..30; 1.96 for df > 30.
_T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}

SUMMARY_COLUMNS = [
    "method",
    "threshold",
    "n_seeds",
    "n_episodes",
    "ci_basis",
    "crash_rate_mean",
    "crash_rate_ci95",
    "avg_speed_mean",
    "avg_speed_ci95",
    "distance_mean",
    "distance_ci95",
    "reward_mean",
    "reward_ci95",
    "steps_mean",
    "steps_ci95",
    "override_rate_mean",
    "override_rate_ci95",
    "total_overrides",
]


def _mean_and_ci95(values: Sequence[float]) -> tuple[float, float]:
    """Return (mean, half_width_95_ci) using Student's t distribution."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = statistics.fmean(values)
    if n == 1:
        return mean, 0.0
    sd = statistics.stdev(values)
    t_crit = _T_CRIT_95.get(n - 1, 1.96)
    ci95 = t_crit * (sd / math.sqrt(n))
    return mean, ci95


def _is_empty(val: Any) -> bool:
    return val is None or str(val).strip() == ""


def _normalize_threshold(val: Any) -> str:
    if _is_empty(val):
        return ""
    return f"{float(val):g}"


def load_episode_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load raw episode rows from one or more CSV paths."""
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
    return rows


def _episode_override_rate(row: dict[str, Any]) -> float | None:
    if _is_empty(row.get("overrides")):
        return None
    steps = float(row["steps"])
    if steps <= 0:
        return 0.0
    return float(row["overrides"]) / steps


def _mean_dict(ep_rows: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    crash = statistics.fmean(float(r["crashed"]) for r in ep_rows)
    speed = statistics.fmean(float(r["avg_speed"]) for r in ep_rows)
    dist = statistics.fmean(float(r["distance"]) for r in ep_rows)
    reward = statistics.fmean(float(r["reward"]) for r in ep_rows)
    steps = statistics.fmean(float(r["steps"]) for r in ep_rows)
    ov_rates = [rate for r in ep_rows if (rate := _episode_override_rate(r)) is not None]
    ov_rate = statistics.fmean(ov_rates) if ov_rates else None
    return {
        "crash_rate": crash,
        "avg_speed": speed,
        "distance": dist,
        "reward": reward,
        "steps": steps,
        "override_rate": ov_rate,
    }


def summarize_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate episode rows by (method, threshold) with 95% confidence intervals."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["method"]), _normalize_threshold(row.get("threshold")))
        grouped[key].append(row)

    summary: list[dict[str, Any]] = []
    for (method, threshold), group_rows in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            float(item[0][1]) if item[0][1] != "" else -1.0,
        ),
    ):
        seed_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in group_rows:
            seed_key = "" if _is_empty(r.get("model_seed")) else str(r["model_seed"]).strip()
            seed_groups[seed_key].append(r)

        non_empty_seeds = [k for k in seed_groups if k != ""]
        use_seed_level = len(non_empty_seeds) > 1 and "" not in seed_groups
        if use_seed_level:
            units = [_mean_dict(seed_groups[k]) for k in sorted(non_empty_seeds)]
            ci_basis = "seeds"
        else:
            units = [_mean_dict([r]) for r in group_rows]
            ci_basis = "episodes"

        crash_m, crash_ci = _mean_and_ci95([float(u["crash_rate"]) for u in units])  # type: ignore[arg-type]
        speed_m, speed_ci = _mean_and_ci95([float(u["avg_speed"]) for u in units])  # type: ignore[arg-type]
        dist_m, dist_ci = _mean_and_ci95([float(u["distance"]) for u in units])  # type: ignore[arg-type]
        reward_m, reward_ci = _mean_and_ci95([float(u["reward"]) for u in units])  # type: ignore[arg-type]
        steps_m, steps_ci = _mean_and_ci95([float(u["steps"]) for u in units])  # type: ignore[arg-type]

        ov_values = [float(u["override_rate"]) for u in units if u["override_rate"] is not None]
        if ov_values:
            ov_m, ov_ci = _mean_and_ci95(ov_values)
            ov_mean_out: Any = round(ov_m, 4)
            ov_ci_out: Any = round(ov_ci, 4)
            total_ov: Any = sum(
                int(float(r["overrides"])) for r in group_rows if not _is_empty(r.get("overrides"))
            )
        else:
            ov_mean_out = ""
            ov_ci_out = ""
            total_ov = ""

        summary.append(
            {
                "method": method,
                "threshold": threshold,
                "n_seeds": len(non_empty_seeds),
                "n_episodes": len(group_rows),
                "ci_basis": ci_basis,
                "crash_rate_mean": round(crash_m, 4),
                "crash_rate_ci95": round(crash_ci, 4),
                "avg_speed_mean": round(speed_m, 4),
                "avg_speed_ci95": round(speed_ci, 4),
                "distance_mean": round(dist_m, 4),
                "distance_ci95": round(dist_ci, 4),
                "reward_mean": round(reward_m, 4),
                "reward_ci95": round(reward_ci, 4),
                "steps_mean": round(steps_m, 4),
                "steps_ci95": round(steps_ci, 4),
                "override_rate_mean": ov_mean_out,
                "override_rate_ci95": ov_ci_out,
                "total_overrides": total_ov,
            }
        )

    return summary


def write_summary_csv(summary_rows: Sequence[dict[str, Any]], out_path: str | Path) -> Path:
    """Write aggregated summary metrics to CSV."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    return path


def format_summary_table(summary_rows: Sequence[dict[str, Any]]) -> str:
    """Format summary rows as a compact human-readable table."""
    if not summary_rows:
        return "(no evaluation rows)"
    header = (
        f"{'method':<18} {'thr':<5} {'ep':>4} {'seeds':>5} "
        f"{'crash_rate':>16} {'avg_speed':>16} {'reward':>16} {'override_rate':>16}"
    )
    lines = [header, "-" * len(header)]
    for r in summary_rows:
        crash = f"{r['crash_rate_mean']:.3f} ± {r['crash_rate_ci95']:.3f}"
        speed = f"{r['avg_speed_mean']:.2f} ± {r['avg_speed_ci95']:.2f}"
        rew = f"{r['reward_mean']:.2f} ± {r['reward_ci95']:.2f}"
        if r["override_rate_mean"] == "":
            ov = "-"
        else:
            ov = f"{r['override_rate_mean']:.3f} ± {r['override_rate_ci95']:.3f}"
        thr = r["threshold"] if r["threshold"] != "" else "-"
        lines.append(
            f"{r['method']:<18} {thr:<5} {r['n_episodes']:>4} {r['n_seeds']:>5} "
            f"{crash:>16} {speed:>16} {rew:>16} {ov:>16}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate episode CSVs into a summary table with 95% CIs."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input episode-level CSV path(s).",
    )
    parser.add_argument(
        "--out",
        default="results/summary.csv",
        help="Output summary CSV path (default: results/summary.csv).",
    )
    args = parser.parse_args(argv)

    rows = load_episode_rows(args.input)
    summary_rows = summarize_rows(rows)
    out_path = write_summary_csv(summary_rows, args.out)
    print(format_summary_table(summary_rows))
    print(f"\nWrote summary to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
