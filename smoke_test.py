import time
import traceback
from importlib.metadata import version

PACKAGES = ["highway-env", "gymnasium", "stable-baselines3", "torch", "laya"]


def check_imports():
    import gymnasium, highway_env, laya, stable_baselines3, torch  # noqa: F401
    for p in PACKAGES:
        print(f"    {p} {version(p)}")


def check_highway():
    from sim.make_env import make_env
    env = make_env()
    env.reset(seed=0)
    for _ in range(20):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            env.reset()
    env.close()


def check_dqn():
    from stable_baselines3 import DQN
    from sim.make_env import make_env
    env = make_env()
    DQN("MlpPolicy", env, learning_starts=200, verbose=0).learn(1_000)
    env.close()


def check_laya():
    import laya
    from laya_tools.highway_to_laya import describe_with_action, safety_question
    from sim.make_env import make_env
    env = make_env()
    env.reset(seed=0)
    for _ in range(5):
        env.step(1)
    state = describe_with_action(env, "LANE_LEFT")
    result = laya.load("convaiinnovations/laya").predict(state, safety_question())
    p = result["answers"]["unsafe"]["noul"]
    assert 0.0 <= p <= 1.0, f"P(unsafe) out of range: {p}"
    print(f"    state: {state}\n    P(unsafe) = {p:.2f}")


def check_device():
    import torch
    if torch.cuda.is_available():
        print(f"    CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        print("    MPS (Apple Silicon)")
    else:
        print("    CPU only (fine for highway-env; Laya fine-tuning will be slow)")


CHECKS = [
    ("imports", check_imports),
    ("highway-env", check_highway),
    ("SB3 DQN", check_dqn),
    ("Laya", check_laya),
    ("device", check_device),
]

if __name__ == "__main__":
    failed = []
    for name, fn in CHECKS:
        print(f"[{name}]")
        start = time.time()
        try:
            fn()
            print(f"    ok ({time.time() - start:.1f} s)")
        except Exception:
            traceback.print_exc()
            failed.append(name)
            if name == "imports":
                break
    print("ALL CHECKS PASSED" if not failed else f"FAILED: {', '.join(failed)}")
    raise SystemExit(1 if failed else 0)
