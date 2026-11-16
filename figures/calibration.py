"""Reliability of zero-shot vs fine-tuned Laya on held-out highway rows (Track 4).

  python -m figures.calibration --compute   # needs both models (GPU): writes results/summary_calibration.csv
  python -m figures.calibration             # plots figures/reliability.png and figures/finetune_bars.png
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from laya_tools.finetune_laya import held_out

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "results" / "summary_calibration.csv"
PNG_PATH = REPO_ROOT / "figures" / "reliability.png"
COLUMNS = ["model", "kind", "bin_lo", "bin_hi", "n", "mean_pred", "observed", "gap"]
N_BINS = 10
MIN_PLOT_N = 20  # sparser bins stay in the CSV but are too noisy to draw
KINDS = {
    "p_unsafe": ("Shield: P(unsafe)", "Predicted P(unsafe)", "Fraction actually unsafe"),
    "choice_top1": ("Action choice: top-1", "Top-1 confidence", "Accuracy"),
}
MODEL_STYLE = {"zero-shot": ("#2a78d6", "o"), "fine-tuned": ("#eb6834", "s")}
SURFACE = "#fcfcfb"
FINETUNE_CSV = REPO_ROOT / "results" / "summary_finetune.csv"
FINETUNE_PNG = REPO_ROOT / "figures" / "finetune_bars.png"
QUESTIONS = {"noul": "Safety\nP(unsafe)", "choice": "Action\nchoice"}
METRICS = {"accuracy": "Accuracy", "ece": "ECE"}


def score(model_path: str, rows: list[dict]) -> list[tuple[str, float, float]]:
    """(kind, predicted, outcome) per row: (P(unsafe), unsafe) or (top-1 probability, correct)."""
    import laya

    agent = laya.load(model_path)
    groups = defaultdict(list)
    for r in rows:
        groups[json.dumps(r["questions"])].append(r)  # predict_batch shares one question set per call

    out = []
    for group in groups.values():
        results = agent.predict_batch([r["state"] for r in group], group[0]["questions"], batch_size=64)
        for r, res in zip(group, results):
            (ans,) = res["answers"].values()
            if ans["type"] == "noul":
                out.append(("p_unsafe", ans["noul"], float(r["expected"]["unsafe"])))
            else:
                out.append(("choice_top1", max(ans["probabilities"].values()),
                            float(ans["choice"] == r["expected"]["action"])))
    return out


def reliability(model: str, scored: list[tuple[str, float, float]]) -> list[dict]:
    """Equal-width bins on [0, 1]; gap = predicted - observed (> 0: overconfident top-1, or overstated risk)."""
    bins = defaultdict(list)
    for kind, pred, outcome in scored:
        bins[(kind, min(int(pred * N_BINS), N_BINS - 1))].append((pred, outcome))
    rows = []
    for (kind, b), pairs in sorted(bins.items()):
        mean_pred = sum(p for p, _ in pairs) / len(pairs)
        observed = sum(o for _, o in pairs) / len(pairs)
        rows.append({"model": model, "kind": kind, "bin_lo": b / N_BINS, "bin_hi": (b + 1) / N_BINS,
                     "n": len(pairs), "mean_pred": round(mean_pred, 4), "observed": round(observed, 4),
                     "gap": round(mean_pred - observed, 4)})
    return rows


def compute(models: dict[str, str], train: Path, val: Path, out: Path = CSV_PATH) -> None:
    rows = held_out(train, val)
    print(f"scoring {len(rows)} held-out val rows")
    table = [row for name, path in models.items() for row in reliability(name, score(path, rows))]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(table)
    print(f"wrote {out}")


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors="#898781", labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")


def plot_finetune(csv_path: Path = FINETUNE_CSV, out: Path = FINETUNE_PNG) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with csv_path.open(encoding="utf-8") as f:
        table = {(r["model"], r["question"]): r for r in csv.DictReader(f)}

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), gridspec_kw={"wspace": 0.25})
    fig.patch.set_facecolor(SURFACE)
    width = 0.36
    for ax, (metric, title) in zip(axes, METRICS.items()):
        for i, (model, (color, _)) in enumerate(MODEL_STYLE.items()):
            xs = [q + (i - 0.5) * width for q in range(len(QUESTIONS))]
            values = [float(table[(model, q)][metric]) for q in QUESTIONS]
            bars = ax.bar(xs, values, width=width, color=color, edgecolor=SURFACE, linewidth=2, label=model)
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9, color="#0b0b0b")
        ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
        ax.set_xticks(range(len(QUESTIONS)), QUESTIONS.values())
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
        ax.set_axisbelow(True)
        _style(ax)
        ax.tick_params(axis="x", labelcolor="#0b0b0b")
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")
    items = {q: int(table[("fine-tuned", q)]["items"]) for q in QUESTIONS}
    fig.text(0.5, -0.06, f"Validation rows from laya-train's report: {items['noul']:,} safety, {items['choice']:,} action "
             f"(includes 13 rows that also appear in train).", ha="center", fontsize=9, color="#52514e")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def plot_reliability(csv_path: Path = CSV_PATH, out: Path = PNG_PATH) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with csv_path.open(encoding="utf-8") as f:
        table = list(csv.DictReader(f))

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), sharex="col",
                             gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08, "wspace": 0.25})
    fig.patch.set_facecolor(SURFACE)
    width = 1 / N_BINS / 2.4
    for col, (kind, (title, xlabel, ylabel)) in enumerate(KINDS.items()):
        top, bottom = axes[0, col], axes[1, col]
        top.plot([0, 1], [0, 1], color="#c3c2b7", linewidth=1, linestyle="--", label="perfect calibration")
        for i, (model, (color, marker)) in enumerate(MODEL_STYLE.items()):
            rows = [r for r in table if r["kind"] == kind and r["model"] == model]
            if not rows:
                continue
            by_bin = {round(float(r["bin_lo"]) * N_BINS): r for r in rows if int(r["n"]) >= MIN_PLOT_N}
            nan = float("nan")  # breaks the line where a bin is skipped
            top.plot([float(by_bin[b]["mean_pred"]) if b in by_bin else nan for b in range(N_BINS)],
                     [float(by_bin[b]["observed"]) if b in by_bin else nan for b in range(N_BINS)],
                     color=color, marker=marker, markersize=6, linewidth=2, label=model)
            centers = [(float(r["bin_lo"]) + float(r["bin_hi"])) / 2 for r in rows]
            offset = (i - 0.5) * width
            bottom.bar([c + offset for c in centers], [int(r["n"]) for r in rows], width=width * 0.9,
                       color=color, linewidth=0)
        top.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
        top.set_ylabel(ylabel, color="#52514e")
        top.set_xlim(0, 1)
        top.set_ylim(0, 1)
        bottom.set_xlabel(xlabel, color="#52514e")
        bottom.set_ylabel("Rows", color="#52514e")
        for ax in (top, bottom):
            ax.grid(color="#e1e0d9", linewidth=0.6)
            _style(ax)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.text(0.5, 0.005, f"Below the diagonal: predicted above observed (overstated risk / overconfident). "
             f"Bins with < {MIN_PLOT_N} rows not drawn. Bottom: rows per bin.",
             ha="center", fontsize=9, color="#52514e")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out

def plot() -> list[Path]:
    return [plot_reliability(), plot_finetune()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--compute", action="store_true")
    p.add_argument("--zero-shot", default="convaiinnovations/laya")
    p.add_argument("--fine-tuned", default="models/final/laya-highway")
    p.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    p.add_argument("--val", type=Path, default=Path("data/val.jsonl"))
    args = p.parse_args(argv)
    if args.compute:
        compute({"zero-shot": args.zero_shot, "fine-tuned": args.fine_tuned}, args.train, args.val)
    for path in plot():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
