"""Plot a Track 5 synthetic fixture using Track 1's existing aggregation.

The fixture and this plot demonstrate the reporting pipeline, not model results.
All output is deliberately marked SYNTHETIC. No models or simulator are loaded.
Existing output files are never overwritten; use a new --out path for a rerun.

Usage (from the repository root):
  MPLBACKEND=Agg python -m figures.plot_synthetic_crash_speed
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

from eval.metrics import summarize_rows
from eval.results_schema import COLUMNS


DEFAULT_INPUT = Path("tests/fixtures/track5_synthetic_episodes.csv")
DEFAULT_OUTPUT = Path("results/track5_synthetic_crash_speed.png")


def load_synthetic_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read the exact shared CSV schema and require explicit synthetic labels."""
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise ValueError(f"Expected CSV columns {COLUMNS}, got {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        raise ValueError("Synthetic CSV must contain at least one episode.")
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Every episode must have exactly the shared schema columns.")
        if not row["method"].startswith("synthetic_"):
            raise ValueError("Synthetic fixture method names must start with 'synthetic_'.")
    return rows


def plot_crash_speed(input_path: str | Path, out_path: str | Path) -> Path:
    """Create a new PNG with one point per (method, threshold) summary."""
    import matplotlib.pyplot as plt

    output = Path(out_path)
    if output.suffix.lower() != ".png":
        raise ValueError("Synthetic figure output must have a .png extension.")
    summary = summarize_rows(load_synthetic_rows(input_path))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    try:
        for row in summary:
            label = row["method"]
            if row["threshold"] != "":
                label += f" (threshold={row['threshold']})"
            ax.scatter(row["avg_speed_mean"], row["crash_rate_mean"], label=label)
        ax.set_xlabel("Average speed (m/s)")
        ax.set_ylabel("Crash rate (fraction of episodes)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("SYNTHETIC — Crash rate versus average speed")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize="small")
        fig.text(0.5, 0.02, "Illustrative test data only — not experimental results", ha="center")
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects existing results, even with a custom --out.
        with output.open("xb") as f:
            fig.savefig(f, format="png", dpi=150)
        return output
    finally:
        plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot SYNTHETIC Track 5 episode data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output = plot_crash_speed(args.input, args.out)
    print(f"Wrote SYNTHETIC demonstration plot to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
