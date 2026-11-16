"""Load a trained Stable-Baselines3 DQN for evaluation or shielding."""

from pathlib import Path

import torch
from stable_baselines3 import DQN


class DQNAgent:
    name = "dqn"

    def __init__(self, model_path: str | Path):
        self.model = DQN.load(model_path, device="cpu")

    def act(self, env, obs) -> int:
        """Return the greedy action used by the standalone DQN baseline."""
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action.item())

    def q_values(self, env, obs) -> list[float]:
        """Return one score per action ID for the current observation."""
        obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            return self.model.q_net(obs_tensor)[0].tolist()

    def rank_actions(self, env, obs) -> list[tuple[int, float]]:
        """Return (action ID, Q-value) pairs, highest score first."""
        return sorted(enumerate(self.q_values(env, obs)), key=lambda pair: -pair[1])
