set -euo pipefail
python -m figures.benchmark_plots
cp figures/crash_vs_speed.png manish_autorunner/status/crash_vs_speed.png
cp figures/threshold_sweep.png manish_autorunner/status/threshold_sweep.png
ls -lh manish_autorunner/status/*.png
