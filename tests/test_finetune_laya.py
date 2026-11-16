from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from laya_tools.finetune_laya import (check_seed_ranges, drop_overlap, merge_shards, parse_loss, shard_plan,
                                      summarize_report)

SEEDS = {"laya_train_offset": 0, "laya_val_offset": 10000, "eval_start": 100000}
SHIELD = {"state": "s", "questions": {"unsafe": {"type": "noul"}}, "expected": {"unsafe": True}}
POLICY = {"state": "s", "questions": {"action": {"type": "choice"}}, "expected": {"action": "IDLE"}}


class ShardPlanTest(unittest.TestCase):
    def test_covers_every_seed_once(self):
        plan = shard_plan(10, 100, 3)
        seeds = [s for off, n in plan for s in range(off, off + n)]
        self.assertEqual(seeds, list(range(100, 110)))
        self.assertEqual([n for _, n in plan], [4, 3, 3])

    def test_more_workers_than_episodes(self):
        self.assertEqual(shard_plan(2, 0, 8), [(0, 1), (1, 1)])


class SeedRangeTest(unittest.TestCase):
    def test_accepts_disjoint_ranges(self):
        check_seed_ranges(10000, 90000, SEEDS)

    def test_rejects_train_into_val(self):
        with self.assertRaises(ValueError):
            check_seed_ranges(10001, 50, SEEDS)

    def test_rejects_val_into_eval(self):
        with self.assertRaises(ValueError):
            check_seed_ranges(300, 90001, SEEDS)


class MergeShardsTest(unittest.TestCase):
    def _merge(self, task):
        with tempfile.TemporaryDirectory() as d:
            shards = []
            for i, rows in enumerate([[SHIELD, POLICY], [SHIELD]]):
                p = Path(d) / f"s{i}.jsonl"
                p.write_text("".join(json.dumps(r) + "\n" for r in rows))
                shards.append(p)
            out = Path(d) / "out.jsonl"
            n = merge_shards(shards, out, task)
            rows = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual(n, len(rows))
        return rows

    def test_both_keeps_all(self):
        self.assertEqual(self._merge("both"), [SHIELD, POLICY, SHIELD])

    def test_filters_by_task(self):
        self.assertEqual(self._merge("shield"), [SHIELD, SHIELD])
        self.assertEqual(self._merge("policy"), [POLICY])


class DropOverlapTest(unittest.TestCase):
    def test_drops_only_val_rows_seen_in_train(self):
        other = dict(SHIELD, state="t")
        with tempfile.TemporaryDirectory() as d:
            train, val = Path(d) / "train.jsonl", Path(d) / "val.jsonl"
            train.write_text(json.dumps(SHIELD) + "\n")
            val.write_text("".join(json.dumps(r) + "\n" for r in [SHIELD, other, POLICY]))
            self.assertEqual(drop_overlap(train, val), 1)
            self.assertEqual([json.loads(l) for l in val.read_text().splitlines()], [other, POLICY])


class SummarizeReportTest(unittest.TestCase):
    def test_before_after_per_question(self):
        def metrics(acc):
            return {"items": 5, "accuracy": acc, "ece": 1 - acc, "brier": 0.1, "loss": 0.2}

        report = {phase: {"by_type": {"noul": metrics(acc), "choice": metrics(acc / 2)}}
                  for phase, acc in (("before", 0.5), ("after", 0.9))}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "train_report.json"
            path.write_text(json.dumps(report))
            with summarize_report(path, Path(d) / "f.csv").open() as f:
                rows = {(r["model"], r["question"]): r for r in csv.DictReader(f)}
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[("fine-tuned", "choice")]["accuracy"], "0.45")
        self.assertEqual(rows[("zero-shot", "noul")]["accuracy"], "0.5")
        self.assertNotIn("loss", rows[("zero-shot", "noul")])


class ParseLossTest(unittest.TestCase):
    def test_steps_and_epoch_means(self):
        log = ("Loading model...\nepoch 1/4 step 100 loss 1.2345\nepoch 1/4 step 200 loss -0.0100\n"
               "epoch 1/4 mean loss 0.9000\nepoch 2/4 step 100 loss nan\nTraining complete!\n")
        rows = parse_loss(log)
        steps = [r for r in rows if r["kind"] == "step"]
        self.assertEqual([(r["epoch"], r["step"], r["loss"]) for r in steps[:2]], [(1, 100, 1.2345), (1, 200, -0.01)])
        self.assertEqual(len(steps), 3)
        self.assertEqual([r for r in rows if r["kind"] == "epoch_mean"],
                         [{"kind": "epoch_mean", "epoch": 1, "step": "", "loss": 0.9}])


if __name__ == "__main__":
    unittest.main()
