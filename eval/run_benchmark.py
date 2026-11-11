"""Benchmark orchestrator for Track 1 (all conditions, seeds, and thresholds).

Reads a benchmark manifest (default: ``configs/benchmark.yaml``), expands every
enabled (method, model_seed, threshold) job, skips or flags unimplemented
tracks cleanly, supports resuming from an existing raw CSV, and writes both
episode-level results and the aggregated summary table.

Usage (from the repo root):
  python -m eval.run_benchmark --dry-run
  python -m eval.run_benchmark --only random --episodes 10
  python -m eval.run_benchmark --skip-existing
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from eval.evaluate import (
    BUILTIN_AGENTS,
    _default_eval_seeds,
    evaluate_agent,
    load_agent,
    write_results_csv,
)
from eval.metrics import (
    _normalize_threshold,
    format_summary_table,
    load_episode_rows,
    summarize_rows,
    write_summary_csv,
)


@dataclass(frozen=True)
class BenchmarkJob:
    method: str
    agent_spec: str
    model_seed: int | str | None
    threshold: float | None
    kwargs: dict[str, Any]
    requires_paths: tuple[str, ...]


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load a benchmark manifest from JSON or YAML."""
    manifest_path = Path(path)
    text = manifest_path.read_text(encoding="utf-8")
    if manifest_path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to parse YAML manifests; install pyyaml or pass a .json manifest."
        ) from exc
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Benchmark manifest {manifest_path} must be a mapping.")
    return data


def _format_value(value: Any, *, model_seed: Any, threshold: float | None) -> Any:
    if isinstance(value, str):
        if value == "{threshold}":
            return threshold
        if value == "{model_seed}":
            return model_seed
        return value.format(model_seed=model_seed, threshold=threshold)
    if isinstance(value, dict):
        return {
            k: _format_value(v, model_seed=model_seed, threshold=threshold)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            _format_value(v, model_seed=model_seed, threshold=threshold)
            for v in value
        ]
    return value


def expand_jobs(
    manifest: dict[str, Any],
    only_methods: Sequence[str] | None = None,
) -> list[BenchmarkJob]:
    """Expand a benchmark manifest into concrete evaluation jobs."""
    selected = set(only_methods) if only_methods else None
    jobs: list[BenchmarkJob] = []

    for cond in manifest.get("conditions", []):
        if not cond.get("enabled", True):
            continue
        method = str(cond["method"])
        if selected is not None and method not in selected:
            continue

        agent_spec = str(cond["agent_spec"])
        model_seeds: list[Any] = list(cond.get("model_seeds") or [None])
        thresholds: list[Any] = list(cond.get("thresholds") or [None])
        raw_kwargs = dict(cond.get("kwargs") or {})
        raw_requires = list(cond.get("requires_paths") or [])

        for m_seed in model_seeds:
            for thr in thresholds:
                thr_val = float(thr) if thr is not None else None
                kwargs = _format_value(
                    raw_kwargs, model_seed=m_seed, threshold=thr_val
                )
                req_paths = tuple(
                    str(_format_value(p, model_seed=m_seed, threshold=thr_val))
                    for p in raw_requires
                )
                jobs.append(
                    BenchmarkJob(
                        method=method,
                        agent_spec=agent_spec,
                        model_seed=m_seed,
                        threshold=thr_val,
                        kwargs=kwargs,
                        requires_paths=req_paths,
                    )
                )

    return jobs


def check_job_readiness(job: BenchmarkJob) -> tuple[bool, str]:
    """Check whether a job's agent module and required artifacts exist."""
    if job.agent_spec not in BUILTIN_AGENTS:
        if ":" not in job.agent_spec:
            return False, f"invalid agent_spec {job.agent_spec!r}"
        module_name, _ = job.agent_spec.split(":", 1)
        try:
            spec = importlib.util.find_spec(module_name)
        except ModuleNotFoundError:
            spec = None
        if spec is None:
            return False, f"module {module_name!r} not implemented yet"

    missing_paths = [p for p in job.requires_paths if not Path(p).exists()]
    if missing_paths:
        return False, f"missing artifact(s): {', '.join(missing_paths)}"

    return True, "ready"


def load_completed_keys(csv_path: str | Path) -> set[tuple[str, str, str, int]]:
    """Return (method, model_seed, normalized_threshold, env_seed) keys already in CSV."""
    path = Path(csv_path)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    completed: set[tuple[str, str, str, int]] = set()
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add(
                (
                    str(row["method"]),
                    str(row.get("model_seed", "")).strip(),
                    _normalize_threshold(row.get("threshold")),
                    int(row["env_seed"]),
                )
            )
    return completed


def run_benchmark(
    manifest: dict[str, Any],
    *,
    raw_out: str | Path | None = None,
    summary_out: str | Path | None = None,
    only_methods: Sequence[str] | None = None,
    episodes: int | None = None,
    seeds: Iterable[int] | None = None,
    env_factory: Callable[..., Any] | None = None,
    skip_existing: bool = False,
    dry_run: bool = False,
    strict: bool = False,
    video_dir: str | Path | None = None,
    video_episodes: int = 0,
    quiet: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute ready benchmark jobs, write raw/summary CSVs, and return rows."""
    eval_cfg = manifest.get("eval", {})
    raw_path = Path(raw_out or eval_cfg.get("raw_out", "results/raw_episodes.csv"))
    sum_path = Path(summary_out or eval_cfg.get("summary_out", "results/summary.csv"))
    n_episodes = episodes if episodes is not None else eval_cfg.get("episodes")

    all_seeds = list(_default_eval_seeds() if seeds is None else seeds)
    if n_episodes is not None:
        all_seeds = all_seeds[: int(n_episodes)]

    jobs = expand_jobs(manifest, only_methods=only_methods)
    completed = load_completed_keys(raw_path) if skip_existing else set()

    if dry_run:
        for idx, job in enumerate(jobs, start=1):
            ready, reason = check_job_readiness(job)
            status = "READY" if ready else f"PENDING ({reason})"
            print(
                f"[{idx}/{len(jobs)}] {status} | method={job.method} "
                f"model_seed={job.model_seed} threshold={job.threshold} "
                f"spec={job.agent_spec}"
            )
        return [], []

    new_rows: list[dict[str, Any]] = []
    wrote_file = skip_existing and raw_path.exists() and raw_path.stat().st_size > 0

    for idx, job in enumerate(jobs, start=1):
        ready, reason = check_job_readiness(job)
        if not ready:
            msg = (
                f"[{idx}/{len(jobs)}] Skipping {job.method} "
                f"(model_seed={job.model_seed}, threshold={job.threshold}): {reason}"
            )
            if strict:
                raise RuntimeError(msg)
            if not quiet:
                print(msg)
            continue

        seed_str = "" if job.model_seed is None else str(job.model_seed)
        thr_str = _normalize_threshold(job.threshold)
        pending_seeds = [
            s for s in all_seeds if (job.method, seed_str, thr_str, int(s)) not in completed
        ]
        if not pending_seeds:
            if not quiet:
                print(
                    f"[{idx}/{len(jobs)}] Already complete: {job.method} "
                    f"(model_seed={job.model_seed}, threshold={job.threshold})"
                )
            continue

        if not quiet:
            print(
                f"[{idx}/{len(jobs)}] Running {job.method} "
                f"(model_seed={job.model_seed}, threshold={job.threshold}) "
                f"for {len(pending_seeds)} episode(s)..."
            )
        agent = load_agent(job.agent_spec, kwargs=job.kwargs)
        rows = evaluate_agent(
            agent,
            method=job.method,
            model_seed=job.model_seed,
            seeds=pending_seeds,
            env_factory=env_factory,
            video_dir=video_dir,
            video_episodes=video_episodes,
            quiet=quiet,
        )
        write_results_csv(rows, raw_path, append=wrote_file)
        wrote_file = True
        new_rows.extend(rows)
        for r in rows:
            completed.add((job.method, seed_str, thr_str, int(r["env_seed"])))

    if raw_path.exists() and raw_path.stat().st_size > 0:
        all_raw_rows = load_episode_rows([raw_path])
        summary_rows = summarize_rows(all_raw_rows)
        write_summary_csv(summary_rows, sum_path)
        if not quiet:
            print("\n" + format_summary_table(summary_rows))
            print(f"\nRaw CSV: {raw_path}\nSummary CSV: {sum_path}")
        return new_rows, summary_rows

    return new_rows, []


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Orchestrate Track 1 benchmark across methods, seeds, and shield thresholds."
    )
    parser.add_argument(
        "--config",
        default="configs/benchmark.yaml",
        help="Path to benchmark YAML/JSON manifest (default: configs/benchmark.yaml).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Subset of condition method names to run.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override number of evaluation episodes per job.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Override raw episode CSV output path.",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Override aggregated summary CSV output path.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip (method, model_seed, threshold, env_seed) rows already present in --out.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List all expanded benchmark jobs and readiness without running episodes.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail immediately if any selected condition is missing its module or artifacts.",
    )
    parser.add_argument(
        "--video-dir",
        default=None,
        help="Optional directory to record evaluation videos.",
    )
    parser.add_argument(
        "--video-episodes",
        type=int,
        default=1,
        help="Episodes to record per job when --video-dir is set (0 = all).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-episode logs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    manifest = load_manifest(args.config)
    run_benchmark(
        manifest,
        raw_out=args.out,
        summary_out=args.summary_out,
        only_methods=args.only,
        episodes=args.episodes,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
        strict=args.strict,
        video_dir=args.video_dir,
        video_episodes=args.video_episodes,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
