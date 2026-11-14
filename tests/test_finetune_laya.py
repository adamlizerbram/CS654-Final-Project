from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from laya_tools.finetune_laya import check_seed_ranges, merge_shards, shard_plan

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


if __name__ == "__main__":
    unittest.main()
