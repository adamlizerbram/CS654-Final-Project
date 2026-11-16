"""Fine-tune Laya on highway-env: parallel data collection, merge, laya-train dry run, train.

One checkpoint serves both roles: shield rows (noul "unsafe") and policy rows (choice "action").
laya-train writes train_report.json (before/after accuracy, ECE, Brier per question type) into --out.
After training, this also writes results/summary_finetune.csv (from that report) and
results/summary_finetune_loss.csv (parsed from laya-train's printed loss, kept as --out/train.log).

Usage (from the repo root):
  python -m laya_tools.finetune_laya --dry-run
  python -m laya_tools.finetune_laya --workers 8 --out models/final/laya-highway
  python -m laya_tools.finetune_laya --skip-data --epochs 2 --device cuda   # unknown flags go to laya-train
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
QUESTION_KEY = {"shield": "unsafe", "policy": "action"}
SUMMARY_CSV = REPO_ROOT / "results" / "summary_finetune.csv"
SUMMARY_COLUMNS = ["model", "question", "items", "accuracy", "ece", "brier"]
LOSS_CSV = REPO_ROOT / "results" / "summary_finetune_loss.csv"
LOSS_COLUMNS = ["kind", "epoch", "step", "loss"]
STEP_RE = re.compile(r"^epoch (\d+)/\d+ step (\d+) loss (\S+)$", re.M)
EPOCH_RE = re.compile(r"^epoch (\d+)/\d+ mean loss (\S+)$", re.M)


def shard_plan(episodes: int, seed_offset: int, workers: int) -> list[tuple[int, int]]:
    """(seed_offset, episodes) per worker, covering seed_offset..seed_offset+episodes-1 exactly once."""
    workers = max(1, min(workers, episodes))
    base, extra = divmod(episodes, workers)
    plan, start = [], seed_offset
    for i in range(workers):
        n = base + (i < extra)
        plan.append((start, n))
        start += n
    return plan


def check_seed_ranges(train_episodes: int, val_episodes: int, seeds: dict) -> None:
    train_end = seeds["laya_train_offset"] + train_episodes
    val_end = seeds["laya_val_offset"] + val_episodes
    if train_end > seeds["laya_val_offset"]:
        raise ValueError(f"train seeds reach {train_end - 1}, into the val range at {seeds['laya_val_offset']}")
    if val_end > seeds["eval_start"]:
        raise ValueError(f"val seeds reach {val_end - 1}, into the eval range at {seeds['eval_start']}")


def keep_row(row: dict, task: str) -> bool:
    return task == "both" or QUESTION_KEY[task] in row["questions"]


def merge_shards(shards: list[Path], out: Path, task: str) -> int:
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for shard in shards:
            with shard.open(encoding="utf-8") as src:
                for line in src:
                    if line.strip() and keep_row(json.loads(line), task):
                        f.write(line if line.endswith("\n") else line + "\n")
                        n += 1
    return n


def row_key(row: dict) -> tuple[str, str]:
    return row["state"], json.dumps(row["questions"], sort_keys=True)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def held_out(train: Path, val: Path) -> list[dict]:
    """Val rows whose input (state + question) never appears in train."""
    seen = {row_key(r) for r in read_jsonl(train)}
    return [r for r in read_jsonl(val) if row_key(r) not in seen]


def drop_overlap(train: Path, val: Path) -> int:
    rows = held_out(train, val)
    n_dropped = len(read_jsonl(val)) - len(rows)
    val.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return n_dropped


def summarize_report(report: Path, out: Path = SUMMARY_CSV) -> Path:
    """Per-question zero-shot (before) vs fine-tuned (after) metrics from train_report.json."""
    data = json.loads(report.read_text(encoding="utf-8"))
    rows = [{"model": model, "question": q, **{k: data[phase]["by_type"][q][k] for k in SUMMARY_COLUMNS[2:]}}
            for model, phase in (("zero-shot", "before"), ("fine-tuned", "after")) for q in ("noul", "choice")]
    return write_csv(out, SUMMARY_COLUMNS, rows)


def parse_loss(log: str) -> list[dict]:
    """laya-train's printed loss: one micro-batch every log_every steps, plus each epoch's mean."""
    rows = [{"kind": "step", "epoch": int(e), "step": int(s), "loss": float(v)} for e, s, v in STEP_RE.findall(log)]
    rows += [{"kind": "epoch_mean", "epoch": int(e), "step": "", "loss": float(v)} for e, v in EPOCH_RE.findall(log)]
    return rows


def write_csv(out: Path, columns: list[str], rows: list[dict]) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return out


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s: str) -> int:
        for stream in self.streams:
            stream.write(s)
        return len(s)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def collect(split: str, episodes: int, seed_offset: int, workers: int, data_dir: Path, task: str) -> Path:
    shard_dir = data_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for off, n in shard_plan(episodes, seed_offset, workers):
        shard = shard_dir / f"{split}_{off}.jsonl"
        log = (shard_dir / f"{split}_{off}.log").open("w", encoding="utf-8")
        cmd = [sys.executable, "-m", "laya_tools.make_laya_dataset",
               "--episodes", str(n), "--seed-offset", str(off), "--out", str(shard)]
        jobs.append((shard, log, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)))
    print(f"[{split}] {len(jobs)} workers, seeds {seed_offset}..{seed_offset + episodes - 1}, logs in {shard_dir}")

    failed = []
    for shard, log, proc in jobs:
        rc = proc.wait()
        log.close()
        print(f"[{split}] {shard.name}: rc={rc}")
        if rc != 0:
            failed.append(shard.name)
    if failed:
        raise RuntimeError(f"data collection failed for {failed}; see the matching .log files")

    out = data_dir / f"{split}.jsonl"
    n = merge_shards([s for s, _, _ in jobs], out, task)
    print(f"[{split}] wrote {n} rows ({task}) to {out}")
    return out


def load_seeds() -> dict:
    import yaml

    return yaml.safe_load((REPO_ROOT / "configs" / "seeds.yaml").read_text())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--train-episodes", type=int, default=300)
    p.add_argument("--val-episodes", type=int, default=50)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--task", choices=["both", "shield", "policy"], default="both",
                   help="which rows to train on; 'both' gives one checkpoint for agent and shield")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--skip-data", action="store_true", help="reuse data-dir/train.jsonl and val.jsonl")
    p.add_argument("--base", default="convaiinnovations/laya")
    p.add_argument("--out", default="models/final/laya-highway")
    p.add_argument("--dry-run", action="store_true", help="collect data and run laya-train --dry-run only")
    return p


def main(argv: list[str] | None = None) -> int:
    args, laya_args = build_parser().parse_known_args(argv)
    if args.skip_data:
        train, val = args.data_dir / "train.jsonl", args.data_dir / "val.jsonl"
        missing = [str(p) for p in (train, val) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"--skip-data but missing {missing}")
    else:
        seeds = load_seeds()
        check_seed_ranges(args.train_episodes, args.val_episodes, seeds)
        train = collect("train", args.train_episodes, seeds["laya_train_offset"], args.workers, args.data_dir, args.task)
        val = collect("val", args.val_episodes, seeds["laya_val_offset"], args.workers, args.data_dir, args.task)
        print(f"[val] dropped {drop_overlap(train, val)} rows whose input also appears in train")

    from laya.train_cli import main as laya_train

    common = ["--data", str(train), "--eval-data", str(val), "--base", args.base, "--shuffle-options", *laya_args]
    rc = laya_train([*common, "--dry-run"])
    if rc != 0 or args.dry_run:
        return rc

    log = io.StringIO()
    with contextlib.redirect_stdout(Tee(sys.stdout, log)):
        rc = laya_train([*common, "--out", args.out])
    if rc == 0:
        out = Path(args.out)
        # Per-epoch resume copy duplicates the final weights; drop it so models/final holds one checkpoint.
        shutil.rmtree(out / "checkpoint_latest", ignore_errors=True)
        (out / "train.log").write_text(log.getvalue(), encoding="utf-8")
        print(f"Metrics: {out / 'train_report.json'}")
        print(f"wrote {summarize_report(out / 'train_report.json')}")
        loss = parse_loss(log.getvalue())
        if not loss:
            print(f"warning: no loss lines found in {out / 'train.log'}; laya-train's log format may have changed")
        print(f"wrote {write_csv(LOSS_CSV, LOSS_COLUMNS, loss)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
