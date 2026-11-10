from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401
import yaml

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def load_yaml(name):
    return yaml.safe_load((CONFIGS / name).read_text())


def make_env(render_mode=None):
    cfg = load_yaml("env.yaml")
    return gym.make(cfg["id"], config=cfg["config"], render_mode=render_mode)


def eval_seeds():
    s = load_yaml("seeds.yaml")
    return range(s["eval_start"], s["eval_start"] + s["eval_episodes"])
