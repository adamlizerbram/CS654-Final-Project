set -euo pipefail
python - << 'EOF'
import inspect
import highway_env.envs.common.abstract as ab
from gymnasium.wrappers import RecordVideo

print("--- _automatic_rendering ---")
print(inspect.getsource(ab.AbstractEnv._automatic_rendering))
print("--- RecordVideo methods ---")
print([m for m in dir(RecordVideo) if not m.startswith("__")])
print(inspect.getsource(RecordVideo.step))
EOF
