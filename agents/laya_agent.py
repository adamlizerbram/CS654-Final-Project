
import laya

from laya_tools.highway_to_laya import (
    ACTIONS,
    describe_state,
    action_choice_question,
    available_actions,
)


class LayaAgent:
    name = "laya_zeroshot"

    def __init__(self):
        self.model = laya.load("convaiinnovations/laya")
        self.action_to_id = {
            name: idx for idx, name in ACTIONS.items()
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
