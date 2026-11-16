"""Independent shield tests using only fake models and environment."""
import unittest
from types import SimpleNamespace

from agents.laya_shield import LayaShieldAgent
from eval.evaluate import run_episode


class FakeEnv:
    def __init__(self, legal=(0, 1, 2)):
        self.action_space = SimpleNamespace(n=3)
        self.unwrapped = SimpleNamespace(action_type=SimpleNamespace(get_available_actions=lambda: legal))

    def reset(self, seed=None):
        self.steps = 0
        return None, {}

    def step(self, action):
        self.steps += 1
        return None, 1, False, self.steps == 2, {"speed": 20, "crashed": False}


class TestShield(unittest.TestCase):
    def make(self, scores, threshold=0.5, q=(2, 1, 3)):
        return LayaShieldAgent(threshold=threshold, q_values=lambda env, obs: q,
                               unsafe_score=lambda env, obs, a: scores[a])

    def test_preferred_and_boundary(self):
        shield = self.make([0.1, 0.1, 0.5])
        self.assertEqual(shield.act(FakeEnv(), None), 2)
        self.assertFalse(shield.last_override)
        self.assertEqual(shield.overrides, 0)

    def test_rejection_and_reset(self):
        shield = self.make([0.2, 0.1, 0.85])
        self.assertEqual(shield.act(FakeEnv(), None), 0)
        self.assertEqual(shield.overrides, 1)
        self.assertTrue(shield.last_override)
        shield.reset()
        self.assertEqual(shield.overrides, 0)
        self.assertFalse(shield.last_override)

    def test_all_unsafe_lowest_risk_and_tie_rank(self):
        shield = self.make([0.7, 0.7, 0.9])
        self.assertEqual(shield.act(FakeEnv(), None), 0)

    def test_legal_filter_and_ties(self):
        shield = self.make([0.2, 0.1, 0.2], q=(2, 2, 3))
        self.assertEqual(shield.act(FakeEnv((0, 1)), None), 0)
        self.assertTrue(shield.last_override)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            self.make([0, 0, 0], threshold=float('nan'))
        for shield, env in [(self.make([0, 0, float('nan')]), FakeEnv()),
                            (self.make([0, 0, 0], q=(1, 2)), FakeEnv()),
                            (self.make([0, 0, 0]), FakeEnv((3,))),
                            (self.make([0, 0, 0]), FakeEnv(()))]:
            with self.subTest(shield=shield, env=env), self.assertRaises(ValueError):
                shield.act(env, None)

    def test_evaluator_resets_and_records_overrides(self):
        shield = self.make([0.2, 0.1, 0.85])
        shield.act(FakeEnv(), None)
        row = run_episode(shield, FakeEnv(), 100000)
        self.assertEqual(row['steps'], 2)
        self.assertEqual(row['overrides'], 2)
        self.assertEqual(row['threshold'], 0.5)


if __name__ == '__main__':
    unittest.main()
