"""Lightweight tests; no models or HighwayEnv required."""
import csv
import tempfile
import unittest
from pathlib import Path

from figures.track5_threshold_analysis import COLUMNS, acceptance, threshold_rows, write_outputs


class TestThresholdAnalysis(unittest.TestCase):
    def setUp(self):
        self.records = [
            dict(model="zero-shot", model_source="mock", env_seed=100000, state_description="fake", action_id=1, action_name="IDLE", unsafe_score=0.5),
            dict(model="zero-shot", model_source="mock", env_seed=100000, state_description="fake", action_id=2, action_name="LANE_RIGHT", unsafe_score=0.9),
            dict(model="fine-tuned", model_source="mock", env_seed=100000, state_description="fake", action_id=1, action_name="IDLE", unsafe_score=0.1),
        ]

    def test_threshold_boundary_and_percentages(self):
        rows = threshold_rows(self.records, [0.5, 0.9])
        self.assertEqual([r["accepted"] for r in rows], [True, True, False, True, True, True])
        self.assertEqual(acceptance(self.records, [0.5, 0.9]), {"zero-shot": [50.0, 100.0], "fine-tuned": [100.0, 100.0]})

    def test_csv_and_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_outputs(self.records, Path(tmp) / "result")
            with path.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.assertEqual(tuple(reader.fieldnames), COLUMNS)
                self.assertEqual(len(list(reader)), 27)
            for name in ("unsafe_score_distribution.png", "threshold_acceptance.png"):
                self.assertEqual((path.parent / name).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            with self.assertRaises(FileExistsError):
                write_outputs(self.records, path.parent)


if __name__ == "__main__":
    unittest.main()
