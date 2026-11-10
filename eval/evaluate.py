"""Evaluation harness for highway-env driving agents.

Runs any agent implementing ``agents.base.Agent`` (``act(env, obs) -> int``)
across a deterministic list of evaluation seeds and writes per-episode metrics
matching ``eval.results_schema.COLUMNS``.

Optional hooks supported for richer agents (e.g. Track 5 safety shield):
  - ``agent.reset()``: called once at the start of each episode if present.
  - ``agent.threshold``: numeric P(unsafe) threshold recorded in ``threshold``.
  - ``agent.overrides``: integer count of overrides in the current episode.
  - ``agent.last_override``: boolean fallback checked after each ``act()`` call
    when ``agent.overrides`` is not exposed.

Usage (from the repo root):
  python -m eval.evaluate --agent random --episodes 10 --out results/random.csv
  python -m eval.evaluate --agent-spec agents.mock_agent:MockAgent --episodes 20
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import random
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from eval.results_schema import COLUMNS


class RandomAgent:
    """Uniformly random legal-action baseline for harness verification."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def act(self, env: Any, obs: Any) -> int:
        unwrapped = getattr(env, "unwrapped", env)
        action_type = getattr(unwrapped, "action_type", None)
        if action_type is not None and hasattr(action_type, "get_available_actions"):
            legal = list(action_type.get_available_actions())
            if legal:
                return int(self._rng.choice(legal))
        return int(env.action_space.sample())


class IdleAgent:
    """Always selects IDLE (action 1 in DiscreteMetaAction) for sanity checks."""

    name = "idle"

    def act(self, env: Any, obs: Any) -> int:
        return 1


BUILTIN_AGENTS: dict[str, type] = {
    "random": RandomAgent,
    "idle": IdleAgent,
}


def load_agent(spec: str, kwargs: dict[str, Any] | None = None) -> Any:
    """Instantiate an agent from a built-in name or 'module.path:SymbolName'."""
    init_kwargs = kwargs or {}
    if spec in BUILTIN_AGENTS:
        return BUILTIN_AGENTS[spec](**init_kwargs)
    if ":" not in spec:
        raise ValueError(
            f"Agent spec {spec!r} must be one of {sorted(BUILTIN_AGENTS)} "
            "or use 'module.path:SymbolName' format."
        )
    module_name, symbol_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    factory = getattr(module, symbol_name)
    return factory(**init_kwargs)


def _get_ego_vehicle(env: Any) -> Any:
    unwrapped = getattr(env, "unwrapped", env)
    return getattr(unwrapped, "vehicle", None)


def _get_ego_x(env: Any) -> float | None:
    ego = _get_ego_vehicle(env)
    pos = getattr(ego, "position", None)
    if pos is None:
        return None
    return float(pos[0])


def _get_ego_speed(env: Any, info: dict[str, Any]) -> float:
    if "speed" in info and info["speed"] is not None:
        return float(info["speed"])
    ego = _get_ego_vehicle(env)
    if ego is not None and hasattr(ego, "speed"):
        return float(ego.speed)
    return 0.0


def _is_ego_crashed(env: Any, info: dict[str, Any]) -> bool:
    if info.get("crashed"):
        return True
    ego = _get_ego_vehicle(env)
    return bool(getattr(ego, "crashed", False))


def _default_make_env(render_mode: str | None = None) -> Any:
    from sim.make_env import make_env

    return make_env(render_mode=render_mode)


def _default_eval_seeds() -> Sequence[int]:
    from sim.make_env import eval_seeds

    return list(eval_seeds())


def _create_env(env_factory: Callable[..., Any], render_mode: str | None) -> Any:
    if render_mode is None:
        return env_factory()
    try:
        return env_factory(render_mode=render_mode)
    except TypeError:
        return env_factory()


def run_episode(
    agent: Any,
    env: Any,
    env_seed: int,
    *,
    method: str | None = None,
    model_seed: int | str | None = None,
) -> dict[str, Any]:
    """Run a single evaluation episode and return a row matching COLUMNS."""
    obs, info = env.reset(seed=int(env_seed))
    if hasattr(agent, "reset") and callable(agent.reset):
        agent.reset()

    start_x = _get_ego_x(env)
    steps = 0
    total_reward = 0.0
    speed_sum = 0.0
    crashed = False
    step_overrides = 0
    saw_last_override_attr = False

    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = int(agent.act(env, obs))
        if hasattr(agent, "last_override"):
            saw_last_override_attr = True
            if agent.last_override:
                step_overrides += 1

        obs, reward, terminated, truncated, info = env.step(action)
        info = info or {}
        steps += 1
        total_reward += float(reward)
        speed_sum += _get_ego_speed(env, info)
        if _is_ego_crashed(env, info):
            crashed = True

    avg_speed = speed_sum / steps if steps > 0 else 0.0
    end_x = _get_ego_x(env)
    if start_x is not None and end_x is not None:
        distance = float(end_x - start_x)
    else:
        distance = float(speed_sum)

    threshold_val = getattr(agent, "threshold", None)
    threshold_out: Any = "" if threshold_val is None else float(threshold_val)

    overrides_attr = getattr(agent, "overrides", None)
    is_shield = (
        threshold_val is not None
        or overrides_attr is not None
        or saw_last_override_attr
    )
    if not is_shield:
        overrides_out: Any = ""
    elif overrides_attr is not None:
        overrides_out = int(overrides_attr)
    else:
        overrides_out = int(step_overrides)

    method_name = method or getattr(agent, "name", type(agent).__name__)
    model_seed_out: Any = "" if model_seed is None else model_seed

    return {
        "method": method_name,
        "model_seed": model_seed_out,
        "env_seed": int(env_seed),
        "crashed": int(crashed),
        "steps": int(steps),
        "avg_speed": round(avg_speed, 4),
        "distance": round(distance, 4),
        "reward": round(total_reward, 4),
        "threshold": threshold_out,
        "overrides": overrides_out,
    }


def _wrap_video_env(
    env: Any,
    *,
    video_dir: str | Path,
    prefix: str,
    video_episodes: int,
    video_vertical: bool,
    quiet: bool,
) -> Any:
    import gymnasium as gym
    import numpy as np
    from gymnasium.wrappers import RecordVideo

    if video_vertical:
        class _VerticalRenderWrapper(gym.Wrapper):
            def render(self) -> Any:
                frame = super().render()
                if frame is None:
                    return None
                # Rotate 90 deg CCW so cars travel bottom-to-top in portrait orientation.
                return np.ascontiguousarray(np.rot90(frame, k=1))

        env = _VerticalRenderWrapper(env)

    video_path = Path(video_dir)
    video_path.mkdir(parents=True, exist_ok=True)
    trigger = (
        (lambda ep_idx: True)
        if video_episodes <= 0
        else (lambda ep_idx: ep_idx < video_episodes)
    )
    wrapped = RecordVideo(
        env,
        video_folder=str(video_path),
        name_prefix=prefix,
        episode_trigger=trigger,
        disable_logger=quiet,
    )
    unwrapped = getattr(wrapped, "unwrapped", None)
    if unwrapped is not None and hasattr(unwrapped, "set_record_video_wrapper"):
        unwrapped.set_record_video_wrapper(wrapped)
    return wrapped


def evaluate_agent(
    agent: Any,
    *,
    method: str | None = None,
    model_seed: int | str | None = None,
    seeds: Iterable[int] | None = None,
    episodes: int | None = None,
    env_factory: Callable[..., Any] | None = None,
    video_dir: str | Path | None = None,
    video_episodes: int = 1,
    video_vertical: bool = False,
    quiet: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate an agent over a sequence of environment seeds."""
    seed_list = list(_default_eval_seeds() if seeds is None else seeds)
    if episodes is not None:
        if episodes < 0:
            raise ValueError(f"episodes must be non-negative, got {episodes}")
        seed_list = seed_list[:episodes]

    if not seed_list:
        return []

    factory = _default_make_env if env_factory is None else env_factory
    render_mode = "rgb_array" if video_dir is not None else None
    env = _create_env(factory, render_mode=render_mode)

    method_label = method or getattr(agent, "name", type(agent).__name__)
    if video_dir is not None:
        seed_tag = f"_seed{model_seed}" if model_seed is not None else ""
        thr_val = getattr(agent, "threshold", None)
        thr_tag = f"_thr{thr_val:g}" if thr_val is not None else ""
        prefix = f"{method_label}{seed_tag}{thr_tag}"
        env = _wrap_video_env(
            env,
            video_dir=video_dir,
            prefix=prefix,
            video_episodes=video_episodes,
            video_vertical=video_vertical,
            quiet=quiet,
        )

    rows: list[dict[str, Any]] = []
    try:
        total = len(seed_list)
        for idx, env_seed in enumerate(seed_list, start=1):
            row = run_episode(
                agent,
                env,
                env_seed,
                method=method_label,
                model_seed=model_seed,
            )
            rows.append(row)
            if not quiet:
                print(
                    f"[{method_label}] episode {idx}/{total} "
                    f"seed={row['env_seed']} crashed={row['crashed']} "
                    f"steps={row['steps']} avg_speed={row['avg_speed']:.2f} "
                    f"distance={row['distance']:.1f} reward={row['reward']:.2f}"
                )
    finally:
        env.close()

    return rows


def write_results_csv(
    rows: Sequence[dict[str, Any]],
    out_path: str | Path,
    *,
    append: bool = False,
) -> Path:
    """Write or append episode rows to CSV following eval.results_schema.COLUMNS."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    for row in rows:
        missing = [col for col in COLUMNS if col not in row]
        if missing:
            raise ValueError(f"Row missing required schema columns: {missing}")

    file_exists_nonempty = path.exists() and path.stat().st_size > 0
    if append and file_exists_nonempty:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
        if existing_header != COLUMNS:
            raise ValueError(
                f"Existing CSV header {existing_header} in {path} does not match "
                f"expected schema {COLUMNS}."
            )
        mode = "a"
        write_header = False
    else:
        mode = "w"
        write_header = True

    with path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    return path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Track 1 evaluation harness for a driving agent."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--agent",
        choices=sorted(BUILTIN_AGENTS),
        help="Built-in baseline agent name.",
    )
    group.add_argument(
        "--agent-spec",
        help="Dotted import path 'module.path:ClassOrFactory' for an Agent.",
    )
    parser.add_argument(
        "--agent-kwargs",
        default="{}",
        help="JSON object of keyword arguments passed to the agent constructor.",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="Override the 'method' column (defaults to agent.name).",
    )
    parser.add_argument(
        "--model-seed",
        default=None,
        help="DQN training seed to record in 'model_seed' (empty if omitted).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of evaluation seeds to run (defaults to configs/seeds.yaml).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="Override starting eval seed (otherwise read from configs/seeds.yaml).",
    )
    parser.add_argument(
        "--out",
        default="results/eval.csv",
        help="Output CSV path (default: results/eval.csv).",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append rows to --out instead of overwriting.",
    )
    parser.add_argument(
        "--video-dir",
        default=None,
        help="Directory to save optional Gymnasium RecordVideo MP4s.",
    )
    parser.add_argument(
        "--video-episodes",
        type=int,
        default=1,
        help="Number of episodes to record when --video-dir is set (0 = all).",
    )
    parser.add_argument(
        "--video-vertical",
        action="store_true",
        help="Rotate recorded video frames 90 deg CCW so cars drive bottom-to-top.",
    )
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Optional path to also write aggregated summary CSV.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-episode progress logs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    spec = args.agent if args.agent is not None else args.agent_spec
    kwargs = json.loads(args.agent_kwargs)
    if not isinstance(kwargs, dict):
        raise ValueError("--agent-kwargs must decode to a JSON object.")

    agent = load_agent(spec, kwargs=kwargs)

    seeds: Iterable[int] | None = None
    if args.seed_start is not None:
        count = args.episodes if args.episodes is not None else len(_default_eval_seeds())
        seeds = range(args.seed_start, args.seed_start + count)

    model_seed: int | str | None = args.model_seed
    if isinstance(model_seed, str):
        try:
            model_seed = int(model_seed)
        except ValueError:
            pass

    rows = evaluate_agent(
        agent,
        method=args.method,
        model_seed=model_seed,
        seeds=seeds,
        episodes=args.episodes,
        video_dir=args.video_dir,
        video_episodes=args.video_episodes,
        video_vertical=args.video_vertical,
        quiet=args.quiet,
    )
    out_path = write_results_csv(rows, args.out, append=args.append)
    if not args.quiet:
        print(f"Wrote {len(rows)} episode rows to {out_path}")

    if args.summary_out is not None:
        from eval.metrics import load_episode_rows, summarize_rows, write_summary_csv

        all_rows = load_episode_rows([out_path])
        summary_rows = summarize_rows(all_rows)
        write_summary_csv(summary_rows, args.summary_out)
        if not args.quiet:
            print(f"Wrote summary ({len(summary_rows)} conditions) to {args.summary_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
