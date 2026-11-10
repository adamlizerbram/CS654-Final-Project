set -euo pipefail
python - << 'EOF'
import inspect
import highway_env.envs.common.abstract as ab
from sim.make_env import make_env

env = make_env(render_mode="rgb_array")
print("metadata:", env.metadata)
print("has set_record_video_wrapper:", hasattr(env.unwrapped, "set_record_video_wrapper"))
print("AbstractEnv._simulate:")
print(inspect.getsource(ab.AbstractEnv._simulate))
if hasattr(ab.AbstractEnv, "set_record_video_wrapper"):
    print("AbstractEnv.set_record_video_wrapper:")
    print(inspect.getsource(ab.AbstractEnv.set_record_video_wrapper))
env.close()
EOF
