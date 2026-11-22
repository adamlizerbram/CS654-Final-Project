"""Regenerate every report figure from committed results/summary*.csv tables (no GPU needed).

  python -m figures.make_figures

Each track owns one module in figures/ exposing plot() -> Path; add it to MODULES.
"""
from __future__ import annotations

import importlib

MODULES = [
    "figures.calibration",      # Track 4
    "figures.benchmark_plots",  # Track 1
]


def main() -> int:
    for name in MODULES:
        print(f"{name}: wrote {importlib.import_module(name).plot()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
