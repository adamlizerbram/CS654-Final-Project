
from __future__ import annotations

import laya

from laya_tools.highway_to_laya import (
    ACTIONS,
    describe_state,
    action_choice_question,
    available_actions,
)


class LayaAgent:
    name = "laya_zeroshot"

    def __init__(
        self,
        model_name: str = "convaiinnovations/laya",
        name: str | None = None,
    ):
        if name is not None:
            self.name = name
        elif model_name != "convaiinnovations/laya":
            self.name = "laya_finetuned"
        self.model_name = model_name
        self.model = laya.load(model_name)
        self.action_to_id = {
            idx_name: idx for idx, idx_name in ACTIONS.items()
        }

    def act(self, env, obs) -> int:
        state = describe_state(env)
        question = action_choice_question(env)
        result = self.model.predict(state, question)

        chosen_action = result["answers"]["action"]["choice"]
        if chosen_action not in available_actions(env):
            raise ValueError(
                f"Bad action from Laya: {chosen_action!r}"
            )

        return self.action_to_id[chosen_action]
