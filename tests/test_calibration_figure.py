from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from figures.calibration import compute, plot, reliability


class ReliabilityTest(unittest.TestCase):
    def test_bins_and_signed_gap(self):
        scored = [("p_unsafe", 0.95, 1.0), ("p_unsafe", 0.91, 0.0), ("p_unsafe", 0.05, 0.0),
                  ("choice_top1", 1.0, 1.0)]
        rows = {(r["kind"], r["bin_lo"]): r for r in reliability("m", scored)}
        top = rows[("p_unsafe", 0.9)]
        self.assertEqual(top["n"], 2)
        self.assertAlmostEqual(top["gap"], 0.93 - 0.5)
        self.assertEqual(rows[("p_unsafe", 0.0)]["gap"], 0.05)
        self.assertEqual(rows[("choice_top1", 0.9)]["n"], 1)  # p = 1.0 lands in the last bin


@unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib not installed")
class PlotTest(unittest.TestCase):
    def test_writes_png_from_csv(self):
        import figures.calibration as cal

        scored = [("p_unsafe", 0.8, 1.0), ("choice_top1", 0.6, 0.0)]
        with tempfile.TemporaryDirectory() as d:
            csv_path, png = Path(d) / "s.csv", Path(d) / "r.png"
            original = cal.score
            cal.score = lambda path, rows: scored
            try:
                train, val = Path(d) / "train.jsonl", Path(d) / "val.jsonl"
                train.write_text("")
                val.write_text("")
                compute({"zero-shot": "a", "fine-tuned": "b"}, train, val, out=csv_path)
            finally:
                cal.score = original
            self.assertGreater(plot(csv_path, png).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
