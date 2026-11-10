"""Unit tests for Track 1 evaluation harness, metrics, and benchmark orchestration."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

from eval.evaluate import (
    IdleAgent,
    RandomAgent,
    evaluate_agent,
    load_agent,
    write_results_csv,
)
from eval.metrics import (
    SUMMARY_COLUMNS,
    format_summary_table,
    load_episode_rows,
    summarize_rows,
    write_summary_csv,
)
from eval.results_schema import COLUMNS
from eval.run_benchmark import (
    check_job_readiness,
    expand_jobs,
    load_manifest,
    run_benchmark,
)


class _FakeActionType:
    def get_available_actions(self) -> list[int]:
        return [1, 3, 4]


class _FakeActionSpace:
    def sample(self) -> int:
        return 1


class _FakeHighwayEnv:
    """Minimal highway-env compatible simulator for unit testing without gym."""

    def __init__(self, max_steps: int = 4, crash_on_action: int | None = 3) -> None:
        self.max_steps = max_steps
        self.crash_on_action = crash_on_action
        self.action_space = _FakeActionSpace()
        self.closed = False
        self._step_idx = 0
        self._seed = 0
        self.unwrapped = SimpleNamespace(
            vehicle=SimpleNamespace(position=[0.0, 0.0], speed=20.0, crashed=False),
            action_type=_FakeActionType(),
        )

    def reset(self, seed: int | None = None):
        self._seed = 0 if seed is None else int(seed)
        self._step_idx = 0
        self.unwrapped.vehicle.position = [100.0, 0.0]
        self.unwrapped.vehicle.speed = 20.0 + (self._seed % 3)
        self.unwrapped.vehicle.crashed = False
        return [0.0], {}

    def step(self, action: int):
        self._step_idx += 1
        crashed = self.crash_on_action is not None and action == self.crash_on_action
        speed = self.unwrapped.vehicle.speed + (2.0 if action == 3 else 0.0)
        self.unwrapped.vehicle.speed = speed
        self.unwrapped.vehicle.position[0] += speed
        self.unwrapped.vehicle.crashed = crashed
        terminated = crashed
        truncated = (not crashed) and (self._step_idx >= self.max_steps)
        reward = -1.0 if crashed else 0.8
        info = {"speed": speed, "crashed": crashed}
        return [float(self._step_idx)], reward, terminated, truncated, info

    def close(self) -> None:
        self.closed = True


class _ShieldWithCounter:
    name = "shield_counter"

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.overrides = 0
        self._step = 0

    def reset(self) -> None:
        self.overrides = 0
        self._step = 0

    def act(self, env, obs) -> int:
        self._step += 1
        if self._step % 2 == 1:
            self.overrides += 1
        return 1


class _ShieldWithStepFlag:
    name = "shield_flag"

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold
        self.last_override = False
        self._step = 0

    def act(self, env, obs) -> int:
        self._step += 1
        self.last_override = self._step == 2
        return 1


class TestEvaluateHarness(unittest.TestCase):
    def test_idle_and_random_agents_match_schema(self) -> None:
        rows = evaluate_agent(
            IdleAgent(),
            seeds=[100000, 100001],
            env_factory=lambda: _FakeHighwayEnv(max_steps=3),
            quiet=True,
        )
        self.assertEqual(len(rows), 2)
        for row, seed in zip(rows, [100000, 100001]):
            self.assertEqual(list(row.keys()), COLUMNS)
            self.assertEqual(row["method"], "idle")
            self.assertEqual(row["model_seed"], "")
            self.assertEqual(row["env_seed"], seed)
            self.assertEqual(row["crashed"], 0)
            self.assertEqual(row["steps"], 3)
            self.assertAlmostEqual(row["reward"], 2.4)
            self.assertEqual(row["threshold"], "")
            self.assertEqual(row["overrides"], "")
            self.assertAlmostEqual(row["distance"], row["avg_speed"] * 3)

        rand_rows = evaluate_agent(
            RandomAgent(seed=42),
            seeds=[100000, 100001],
            env_factory=lambda: _FakeHighwayEnv(max_steps=4, crash_on_action=3),
            quiet=True,
        )
        self.assertEqual(len(rand_rows), 2)
        self.assertEqual(rand_rows[0]["method"], "random")

    def test_shield_override_tracking_variants(self) -> None:
        rows_counter = evaluate_agent(
            _ShieldWithCounter(threshold=0.5),
            model_seed=2,
            seeds=[100000, 100001],
            env_factory=lambda: _FakeHighwayEnv(max_steps=4),
            quiet=True,
        )
        self.assertEqual(rows_counter[0]["model_seed"], 2)
        self.assertEqual(rows_counter[0]["threshold"], 0.5)
        self.assertEqual(rows_counter[0]["overrides"], 2)
        self.assertEqual(rows_counter[1]["overrides"], 2)

        rows_flag = evaluate_agent(
            _ShieldWithStepFlag(threshold=0.7),
            model_seed=0,
            seeds=[100000],
            env_factory=lambda: _FakeHighwayEnv(max_steps=3),
            quiet=True,
        )
        self.assertEqual(rows_flag[0]["threshold"], 0.7)
        self.assertEqual(rows_flag[0]["overrides"], 1)

    def test_write_and_append_results_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "eval.csv"
            rows_a = evaluate_agent(
                IdleAgent(),
                seeds=[100000],
                env_factory=lambda: _FakeHighwayEnv(max_steps=2),
                quiet=True,
            )
            rows_b = evaluate_agent(
                IdleAgent(),
                seeds=[100001],
                env_factory=lambda: _FakeHighwayEnv(max_steps=2),
                quiet=True,
            )
            write_results_csv(rows_a, csv_path, append=False)
            write_results_csv(rows_b, csv_path, append=True)

            loaded = load_episode_rows([csv_path])
            self.assertEqual(len(loaded), 2)
            self.assertEqual([int(r["env_seed"]) for r in loaded], [100000, 100001])

            bad_csv = Path(tmp) / "bad.csv"
            bad_csv.write_text("wrong,header\n1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                write_results_csv(rows_a, bad_csv, append=True)

    def test_load_agent_builtin_and_dotted(self) -> None:
        agent = load_agent("idle")
        self.assertIsInstance(agent, IdleAgent)
        agent2 = load_agent("eval.evaluate:RandomAgent", {"seed": 7})
        self.assertIsInstance(agent2, RandomAgent)
        with self.assertRaises(ValueError):
            load_agent("not_a_spec")


class TestMetricsAggregation(unittest.TestCase):
    def test_summarize_multi_seed_and_single_model(self) -> None:
        rows = [
            # Multi-seed method: 2 seeds x 2 episodes
            {"method": "dqn", "model_seed": 0, "env_seed": 100000, "crashed": 0, "steps": 10, "avg_speed": 25.0, "distance": 250.0, "reward": 8.0, "threshold": "", "overrides": ""},
            {"method": "dqn", "model_seed": 0, "env_seed": 100001, "crashed": 1, "steps": 5, "avg_speed": 25.0, "distance": 125.0, "reward": 2.0, "threshold": "", "overrides": ""},
            {"method": "dqn", "model_seed": 1, "env_seed": 100000, "crashed": 0, "steps": 10, "avg_speed": 27.0, "distance": 270.0, "reward": 9.0, "threshold": "", "overrides": ""},
            {"method": "dqn", "model_seed": 1, "env_seed": 100001, "crashed": 0, "steps": 10, "avg_speed": 27.0, "distance": 270.0, "reward": 9.0, "threshold": "", "overrides": ""},
            # Single-model method (empty model_seed)
            {"method": "laya_zeroshot", "model_seed": "", "env_seed": 100000, "crashed": 0, "steps": 10, "avg_speed": 22.0, "distance": 220.0, "reward": 7.0, "threshold": "", "overrides": ""},
            {"method": "laya_zeroshot", "model_seed": "", "env_seed": 100001, "crashed": 0, "steps": 10, "avg_speed": 24.0, "distance": 240.0, "reward": 7.5, "threshold": "", "overrides": ""},
            # Shielded method across 2 seeds
            {"method": "shield_zeroshot", "model_seed": 0, "env_seed": 100000, "crashed": 0, "steps": 10, "avg_speed": 24.0, "distance": 240.0, "reward": 7.5, "threshold": 0.5, "overrides": 2},
            {"method": "shield_zeroshot", "model_seed": 1, "env_seed": 100000, "crashed": 0, "steps": 10, "avg_speed": 24.0, "distance": 240.0, "reward": 7.5, "threshold": 0.5, "overrides": 4},
        ]
        summary = summarize_rows(rows)
        by_method = {r["method"]: r for r in summary}

        self.assertEqual(by_method["dqn"]["ci_basis"], "seeds")
        self.assertEqual(by_method["dqn"]["n_seeds"], 2)
        self.assertEqual(by_method["dqn"]["n_episodes"], 4)
        self.assertAlmostEqual(by_method["dqn"]["crash_rate_mean"], 0.25)
        self.assertAlmostEqual(by_method["dqn"]["avg_speed_mean"], 26.0)
        self.assertEqual(by_method["dqn"]["override_rate_mean"], "")

        self.assertEqual(by_method["laya_zeroshot"]["ci_basis"], "episodes")
        self.assertEqual(by_method["laya_zeroshot"]["n_seeds"], 0)
        self.assertAlmostEqual(by_method["laya_zeroshot"]["avg_speed_mean"], 23.0)

        self.assertEqual(by_method["shield_zeroshot"]["threshold"], "0.5")
        self.assertAlmostEqual(by_method["shield_zeroshot"]["override_rate_mean"], 0.3)
        self.assertEqual(by_method["shield_zeroshot"]["total_overrides"], 6)

        with tempfile.TemporaryDirectory() as tmp:
            out = write_summary_csv(summary, Path(tmp) / "summary.csv")
            with out.open("r", encoding="utf-8") as f:
                header = next(csv.reader(f))
            self.assertEqual(header, SUMMARY_COLUMNS)

        table_text = format_summary_table(summary)
        self.assertIn("shield_zeroshot", table_text)


class TestBenchmarkOrchestration(unittest.TestCase):
    def test_expand_jobs_and_run_benchmark_with_skip_existing(self) -> None:
        manifest = {
            "eval": {"episodes": 2},
            "conditions": [
                {"method": "idle_test", "agent_spec": "idle"},
                {
                    "method": "pending_dqn",
                    "agent_spec": "agents.nonexistent_dqn:DQNAgent",
                    "model_seeds": [0, 1],
                    "requires_paths": ["models/final/dqn_seed{model_seed}.zip"],
                },
                {
                    "method": "shield_test",
                    "agent_spec": "tests.test_eval:_ShieldWithCounter",
                    "model_seeds": [0],
                    "thresholds": [0.3, 0.6],
                    "kwargs": {"threshold": "{threshold}"},
                },
            ],
        }
        jobs = expand_jobs(manifest)
        self.assertEqual(len(jobs), 5)
        self.assertEqual(jobs[3].kwargs["threshold"], 0.3)
        self.assertEqual(jobs[4].kwargs["threshold"], 0.6)

        ready_idle, _ = check_job_readiness(jobs[0])
        ready_dqn, reason_dqn = check_job_readiness(jobs[1])
        self.assertTrue(ready_idle)
        self.assertFalse(ready_dqn)
        self.assertIn("not implemented", reason_dqn)

        with tempfile.TemporaryDirectory() as tmp:
            raw_csv = Path(tmp) / "raw.csv"
            sum_csv = Path(tmp) / "summary.csv"
            new_rows, summary_rows = run_benchmark(
                manifest,
                raw_out=raw_csv,
                summary_out=sum_csv,
                seeds=[100000, 100001],
                env_factory=lambda: _FakeHighwayEnv(max_steps=3),
                skip_existing=True,
                quiet=True,
            )
            # idle_test (2 eps) + shield_test x 2 thresholds (4 eps) = 6 rows
            self.assertEqual(len(new_rows), 6)
            self.assertEqual(len(summary_rows), 3)
            self.assertTrue(raw_csv.exists())
            self.assertTrue(sum_csv.exists())

            # Re-run with skip_existing=True: should produce 0 new rows.
            rerun_rows, _ = run_benchmark(
                manifest,
                raw_out=raw_csv,
                summary_out=sum_csv,
                seeds=[100000, 100001],
                env_factory=lambda: _FakeHighwayEnv(max_steps=3),
                skip_existing=True,
                quiet=True,
            )
            self.assertEqual(len(rerun_rows), 0)
            self.assertEqual(len(load_episode_rows([raw_csv])), 6)

            # Strict mode fails on pending track.
            with self.assertRaises(RuntimeError):
                run_benchmark(
                    manifest,
                    raw_out=raw_csv,
                    summary_out=sum_csv,
                    seeds=[100000],
                    env_factory=lambda: _FakeHighwayEnv(max_steps=3),
                    strict=True,
                    quiet=True,
                )

    def test_load_json_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bench.json"
            path.write_text(
                json.dumps({"conditions": [{"method": "idle", "agent_spec": "idle"}]}),
                encoding="utf-8",
            )
            loaded = load_manifest(path)
            self.assertEqual(loaded["conditions"][0]["method"], "idle")

    def test_cli_entry_with_mocked_sim_module(self) -> None:
        fake_sim = types.ModuleType("sim.make_env")
        fake_sim.make_env = lambda render_mode=None: _FakeHighwayEnv(max_steps=2)
        fake_sim.eval_seeds = lambda: range(100000, 100005)
        prev = sys.modules.get("sim.make_env")
        sys.modules["sim.make_env"] = fake_sim
        try:
            from eval.evaluate import main as eval_main

            with tempfile.TemporaryDirectory() as tmp:
                out_csv = Path(tmp) / "cli.csv"
                sum_csv = Path(tmp) / "cli_summary.csv"
                rc = eval_main(
                    [
                        "--agent",
                        "idle",
                        "--episodes",
                        "3",
                        "--out",
                        str(out_csv),
                        "--summary-out",
                        str(sum_csv),
                        "--quiet",
                    ]
                )
                self.assertEqual(rc, 0)
                self.assertEqual(len(load_episode_rows([out_csv])), 3)
                self.assertTrue(sum_csv.exists())
        finally:
            if prev is None:
                sys.modules.pop("sim.make_env", None)
            else:
                sys.modules["sim.make_env"] = prev


if __name__ == "__main__":
    unittest.main()
