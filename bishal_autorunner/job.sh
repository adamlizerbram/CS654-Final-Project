set -euo pipefail
source_sha=e60c15e3b320809936ed3a7080da5054b8427c2c
runner_root="$PWD"
echo "Runner revision: $(git rev-parse HEAD)"
git fetch origin refs/heads/feat/track5-safety-shield
git merge-base --is-ancestor "$source_sha" FETCH_HEAD
test "$(git rev-parse "${source_sha}^{commit}")" = "$source_sha"
job_dir=$(mktemp -d "${TMPDIR:-/tmp}/track5-laya-probe.XXXXXXXX")
trap 'rm -rf -- "$job_dir"' EXIT
git archive "$source_sha" | tar -x -C "$job_dir"
cd "$job_dir"
echo "Source revision: $source_sha"
python - <<'TRACK5_PROBE'
"""SDSU-only real Laya probe; DQN values are mock, Laya scores are real."""
import math
from pathlib import Path

from agents.laya_shield import LayaShieldAgent
from laya_tools.highway_to_laya import ACTIONS, describe_state
from sim.make_env import make_env


def probe(label, checkpoint):
    env = None
    try:
        print(f"[{label}] loading checkpoint", flush=True)
        # A fixed ranking avoids loading DQN. No unsafe_score is injected, so
        # the constructor installs the shield's actual Laya scoring adapter.
        shield = LayaShieldAgent(laya_model=checkpoint, threshold=1.0,
                                q_values=lambda env, obs: list(range(env.action_space.n)))
        print(f"[{label}] model loaded", flush=True)
        env = make_env()
        obs, _ = env.reset(seed=100000)
        print(f"[{label}] real HighwayEnv state: {describe_state(env)}", flush=True)
        actions = list(env.unwrapped.action_type.get_available_actions())[:2]
        if not actions:
            raise RuntimeError("No available candidate actions")
        for action in actions:
            score = shield._unsafe_score(env, obs, action)
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Invalid score for action {action}: {score!r}")
            print(f"[{label}] REAL LAYA SCORE action={action} ({ACTIONS[action]}) "
                  f"unsafe={score:.6f}", flush=True)
        # Also exercise the evaluator-facing path; mock Q-values are never
        # presented as trained DQN predictions.
        chosen = shield.act(env, obs)
        if chosen not in env.unwrapped.action_type.get_available_actions():
            raise ValueError("Shield selected an unavailable action")
        print(f"[{label}] shield.act succeeded; selected={chosen}, "
              f"override={shield.last_override}; DQN Q-values MOCK", flush=True)
        return True
    except Exception as exc:
        # Keep logs useful without dumping environment variables or credentials.
        print(f"[{label}] FAILED: {type(exc).__name__} (details omitted from Git log)", flush=True)
        return False
    finally:
        if env is not None:
            env.close()


def main():
    print("REAL LAYA INFERENCE / MOCK DQN Q-VALUES — not synthetic Laya scores", flush=True)
    zero_ok = probe("zero-shot", "convaiinnovations/laya")
    paths = [Path("/home/jovyan/CS654-Final-Project/models/final/laya-highway"),
             Path("/home/jovyan/CS654-Track5-Runner/models/final/laya-highway")]
    local = next((path for path in paths if path.is_dir()), None)
    if local:
        print(f"[fine-tuned] local checkpoint directory found: {local}", flush=True)
        fine_ok = probe("fine-tuned", str(local))
    else:
        print("[fine-tuned] no local checkpoint directory found", flush=True)
        try:
            from huggingface_hub import HfApi
            info = HfApi().model_info("adamlizerbram/cs654-final-project", files_metadata=False)
            files = [entry.rfilename for entry in (info.siblings or [])]
            found = any(name.startswith("laya-highway/") for name in files)
            print(f"[fine-tuned] Hugging Face repository reachable; "
                  f"laya-highway files present={found}; weights not downloaded", flush=True)
        except Exception as exc:
            print(f"[fine-tuned] repository check FAILED: {type(exc).__name__} (details omitted from Git log)", flush=True)
        fine_ok = None
    print(f"RESULT zero-shot={'PASS' if zero_ok else 'FAIL'}; "
          f"fine-tuned={'PASS' if fine_ok else 'FAIL' if fine_ok is False else 'NOT RUN'}", flush=True)
    return 0 if zero_ok and fine_ok is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
TRACK5_PROBE
