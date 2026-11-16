"""Track 5 synthetic plotting tests; no models or highway-env required."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from eval.metrics import summarize_rows
from eval.results_schema import COLUMNS
from figures.plot_synthetic_crash_speed import load_synthetic_rows, main


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "track5_synthetic_episodes.csv"


class TestTrack5SyntheticFigures(unittest.TestCase):
    def test_fixture_matches_shared_schema(self) -> None:
        rows = load_synthetic_rows(FIXTURE)
        self.assertEqual(len(rows), 12)
        for row in rows:
            self.assertEqual(list(row), COLUMNS)
            self.assertTrue(row["method"].startswith("synthetic_"))
            self.assertIn(int(row["crashed"]), (0, 1))
            self.assertGreater(int(row["steps"]), 0)
            if row["overrides"] != "":
                self.assertGreaterEqual(int(row["overrides"]), 0)
                self.assertLessEqual(int(row["overrides"]), int(row["steps"]))

    def test_groups_by_method_and_threshold(self) -> None:
        summary = summarize_rows(load_synthetic_rows(FIXTURE))
        groups = {(row["method"], row["threshold"]): row for row in summary}
        expected = {
            ("synthetic_dqn", ""): (26.0, 0.5),
            ("synthetic_shield", "0.3"): (22.0, 0.25),
            ("synthetic_shield", "0.7"): (25.0, 0.5),
        }
        self.assertEqual(set(groups), set(expected))
        for key, (speed, crash_rate) in expected.items():
            row = groups[key]
            self.assertEqual(row["n_episodes"], 4)
            self.assertEqual(row["n_seeds"], 2)
            self.assertEqual(row["ci_basis"], "seeds")
            self.assertAlmostEqual(row["avg_speed_mean"], speed)
            self.assertAlmostEqual(row["crash_rate_mean"], crash_rate)

    def test_unshielded_fields_stay_blank(self) -> None:
        rows = load_synthetic_rows(FIXTURE)
        unshielded = [row for row in rows if row["method"] == "synthetic_dqn"]
        self.assertEqual(len(unshielded), 4)
        for row in unshielded:
            self.assertEqual(row["threshold"], "")
            self.assertEqual(row["overrides"], "")
        summary = summarize_rows(unshielded)[0]
        for field in ("threshold", "override_rate_mean", "override_rate_ci95", "total_overrides"):
            self.assertEqual(summary[field], "")

    def test_rejects_wrong_schema_empty_data_and_unlabeled_data(self) -> None:
        valid = FIXTURE.read_text(encoding="utf-8")
        cases = [
            "wrong,header\n1,2\n",
            ",".join(COLUMNS) + "\n",
            valid.replace("synthetic_dqn", "dqn"),
            valid.replace("synthetic_dqn,0,100000", "synthetic_dqn,100000", 1),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.csv"
            for content in cases:
                with self.subTest(content=content.splitlines()[0]):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_synthetic_rows(path)

    def test_cli_generates_figure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "nested" / "synthetic.png"
            rc = main(["--input", str(FIXTURE), "--out", str(output)])
            self.assertEqual(rc, 0)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 1000)
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_existing_output_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "existing_experiment.png"
            original = b"existing experimental result"
            output.write_bytes(original)
            with self.assertRaises(FileExistsError):
                main(["--input", str(FIXTURE), "--out", str(output)])
            self.assertEqual(output.read_bytes(), original)

    def test_rejects_non_png_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "summary.csv"
            with self.assertRaises(ValueError):
                main(["--input", str(FIXTURE), "--out", str(output)])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
