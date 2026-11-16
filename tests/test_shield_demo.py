"""Focused tests for the synthetic demonstration and its artifacts."""
import csv
import tempfile
import unittest
from pathlib import Path

from examples.demo_shield import decisions, write_outputs


class TestShieldDemo(unittest.TestCase):
    def test_expected_actual_shield_decisions(self):
        rows = decisions()
        self.assertEqual([(r["threshold"], r["shield_selected_action"], r["override"], r["all_unsafe_fallback"]) for r in rows],
                         [(0.05, "IDLE", True, True), (0.5, "LANE_LEFT", True, False),
                          (0.9, "LANE_RIGHT", False, False)])
        self.assertTrue(all(r["dqn_preferred_action"] == "LANE_RIGHT" for r in rows))

    def test_csv_png_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = decisions()
            csv_path, png_path = write_outputs(rows, Path(tmp))
            with csv_path.open(newline="", encoding="utf-8") as f:
                saved = list(csv.DictReader(f))
            self.assertEqual(len(saved), 3)
            self.assertTrue(all(r["data_type"] == "MOCK / SYNTHETIC" for r in saved))
            self.assertEqual(png_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            original = png_path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_outputs(rows, Path(tmp))
            self.assertEqual(png_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
