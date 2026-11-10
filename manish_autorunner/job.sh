set -euo pipefail
python - << 'EOF'
import inspect, os, numpy as np
from sim.make_env import make_env
import highway_env.envs.common.graphics as gfx

print("SDL_VIDEODRIVER:", os.environ.get("SDL_VIDEODRIVER"))
for sdl in [None, "dummy"]:
    if sdl is None:
        os.environ.pop("SDL_VIDEODRIVER", None)
    else:
        os.environ["SDL_VIDEODRIVER"] = sdl
    env = make_env(render_mode="rgb_array")
    obs, _ = env.reset(seed=100000)
    print("sdl=", sdl, "render_mode=", env.render_mode, "offscreen=", env.unwrapped.config.get("offscreen_rendering"))
    f0 = env.render()
    env.step(1)
    f1 = env.render()
    print("  f0:", None if f0 is None else (f0.shape, f0.dtype, int(f0.min()), int(f0.max()), float(f0.mean())))
    print("  f1:", None if f1 is None else (f1.shape, f1.dtype, int(f1.min()), int(f1.max()), float(f1.mean())))
    env.close()

print("--- EnvViewer.__init__ ---")
print(inspect.getsource(gfx.EnvViewer.__init__))
print("--- EnvViewer.get_image ---")
print(inspect.getsource(gfx.EnvViewer.get_image))
EOF
