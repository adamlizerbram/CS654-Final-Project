set -euo pipefail
source_sha=e60c15e3b320809936ed3a7080da5054b8427c2c
echo "Runner revision: $(git rev-parse HEAD)"
git fetch origin refs/heads/main
git merge-base --is-ancestor "$source_sha" FETCH_HEAD
test "$(git rev-parse "${source_sha}^{commit}")" = "$source_sha"
job_dir=$(mktemp -d "${TMPDIR:-/tmp}/track5-finetuned.XXXXXXXX")
trap 'rm -rf -- "$job_dir"' EXIT
git archive "$source_sha" | tar -x -C "$job_dir"
cd "$job_dir"
echo "Source revision: $source_sha"
python - <<'TRACK5_PROBE'
"""SDSU only: compare real Laya scores; DQN Q-values are mock."""
import gc
import hashlib
import math
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from agents.laya_shield import LayaShieldAgent
from laya_tools.highway_to_laya import ACTIONS, describe_state
from sim.make_env import make_env

MODEL_REV = "8f150026ebd947d30db1afaf048fb832b3bfe330"
MODEL_SHA256 = "95ea832b40e5719984ca7400ce93e36f26c659dca366d482d4151158542a2607"
MODEL_ROOT = Path("/home/jovyan/CS654-Track5-Runner/models/final")
MODEL_PATH = MODEL_ROOT / "laya-highway"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run():
    env = make_env()
    try:
        obs, _ = env.reset(seed=100000)
        candidates = [1, 2]
        if not set(candidates).issubset(env.unwrapped.action_type.get_available_actions()):
            raise RuntimeError("Expected candidate actions not available")
        print("REAL LAYA SCORES; DQN Q-VALUES MOCK; seed=100000", flush=True)
        print("State:", describe_state(env), flush=True)
        results = {}
        for label, source in (("zero-shot", "convaiinnovations/laya"),
                              ("fine-tuned", str(MODEL_PATH))):
            print(f"[{label}] loading source={source}", flush=True)
            shield = LayaShieldAgent(laya_model=source, threshold=1.0,
                                     q_values=lambda env, obs: list(range(env.action_space.n)))
            print(f"[{label}] loaded", flush=True)
            results[label] = {}
            for action in candidates:
                score = shield._unsafe_score(env, obs, action)
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"Invalid {label} score for action {action}")
                results[label][action] = score
                print(f"[{label}] action={action} ({ACTIONS[action]}) unsafe={score:.6f}", flush=True)
            del shield
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        for action in candidates:
            print(f"COMPARE {ACTIONS[action]} zero-shot={results['zero-shot'][action]:.6f} "
                  f"fine-tuned={results['fine-tuned'][action]:.6f}", flush=True)
        print("RESULT both models loaded; four finite scores in [0, 1]", flush=True)
    finally:
        env.close()


def main():
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(MODEL_ROOT).free
    print(f"Model source: adamlizerbram/cs654-final-project revision={MODEL_REV}", flush=True)
    print(f"Model directory: {MODEL_PATH}; free bytes={free}", flush=True)
    if free < 2_000_000_000:
        raise RuntimeError("Need at least 2 GB free before checkpoint download")
    # Pinned revision and local_dir reuse existing matching files; no Git writes.
    snapshot_download(repo_id="adamlizerbram/cs654-final-project", revision=MODEL_REV,
                      allow_patterns=["laya-highway/*"], local_dir=str(MODEL_ROOT))
    weights = MODEL_PATH / "model.safetensors"
    if sha256(weights) != MODEL_SHA256:
        raise ValueError("Fine-tuned weights SHA-256 does not match pinned metadata")
    print("Fine-tuned weights SHA-256 verified", flush=True)
    run()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Avoid publishing URLs or tokens embedded in exception messages.
        print(f"RESULT FAIL: {type(exc).__name__} (details omitted from Git log)", flush=True)
        raise SystemExit(1)
TRACK5_PROBE
