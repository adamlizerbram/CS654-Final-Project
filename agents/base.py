from typing import Protocol


class Agent(Protocol):
    name: str

    def act(self, env, obs) -> int:
        """Return an action id (see laya_tools.highway_to_laya.ACTIONS). Laya-based agents read env.unwrapped."""
        ...
