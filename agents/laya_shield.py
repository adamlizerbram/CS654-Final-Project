"""Inference-time shield: rank DQN actions, then filter by Laya risk.

The default adapters load SB3 and Laya lazily. For independent tests or other
implementations, inject a Q-value callable and an unsafe-score callable.
Scores are model estimates, not guarantees of safety.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any


class LayaShieldAgent:
    name = "laya_shield"

    def __init__(
        self,
        dqn_path: str | None = None,
        laya_model: str = "convaiinnovations/laya",
        threshold: float = 0.5,
        *,
        q_values: Callable[[Any, Any], Sequence[float]] | None = None,
        unsafe_score: Callable[[Any, Any, int], float] | None = None,
    ) -> None:
        self.threshold = float(threshold)
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be finite and in [0, 1]")
        if q_values is None:
            if dqn_path is None:
                raise ValueError("Provide dqn_path or a q_values callable")
            from stable_baselines3 import DQN
            model = DQN.load(dqn_path)

            def q_values(env: Any, obs: Any) -> Sequence[float]:
                import torch
                obs_tensor, _ = model.policy.obs_to_tensor(obs)
                with torch.no_grad():
                    return model.q_net(obs_tensor)[0].cpu().tolist()

        if unsafe_score is None:
            import laya
            from laya_tools.highway_to_laya import ACTIONS, describe_with_action, safety_question
            model_laya = laya.load(laya_model)

            def unsafe_score(env: Any, obs: Any, action: int) -> float:
                state = describe_with_action(env, ACTIONS[action])
                result = model_laya.predict(state, safety_question())
                return float(result["answers"]["unsafe"]["noul"])

        self._q_values = q_values
        self._unsafe_score = unsafe_score
        self.reset()

    def reset(self) -> None:
        self.overrides = 0
        self.last_override = False

    def act(self, env: Any, obs: Any) -> int:
        self.last_override = False
        values = [float(value) for value in self._q_values(env, obs)]
        n = int(env.action_space.n)
        if len(values) != n or not all(math.isfinite(value) for value in values):
            raise ValueError("Q-values must contain one finite value per action")
        unwrapped = getattr(env, "unwrapped", env)
        action_type = getattr(unwrapped, "action_type", None)
        if action_type is not None and hasattr(action_type, "get_available_actions"):
            available = list(action_type.get_available_actions())
            if any(isinstance(a, bool) or int(a) != a or not 0 <= int(a) < n for a in available):
                raise ValueError("environment returned an invalid available action")
            legal = {int(a) for a in available}
        else:
            legal = set(range(n))
        if not legal:
            raise ValueError("environment has no available actions")
        # Compare to DQN's preferred action before legal-action filtering.
        preferred = max(range(n), key=lambda a: (values[a], -a))
        ranked = sorted(legal, key=lambda a: (-values[a], a))
        scored = []
        for action in ranked:
            score = float(self._unsafe_score(env, obs, action))
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("unsafe scores must be finite and in [0, 1]")
            scored.append((action, score))
            if score <= self.threshold:
                selected = action
                break
        else:
            # All unsafe: choose lowest estimated risk, breaking ties by Q rank.
            selected = min(scored, key=lambda pair: pair[1])[0]
        self.last_override = selected != preferred
        self.overrides += int(self.last_override)
        return selected
