set -euo pipefail
# Replace only after the reviewed development commit is pushed.
source_sha=9bd061a8a0b1a05b31ca6c76e270518f3b90e88c
feature_branch=codex/track5-laya-threshold-analysis
runner_root="$PWD"
echo "Runner revision: $(git rev-parse HEAD)"
git fetch origin "refs/heads/$feature_branch"
git merge-base --is-ancestor "$source_sha" FETCH_HEAD
test "$(git rev-parse "${source_sha}^{commit}")" = "$source_sha"
model_dir=/home/jovyan/CS654-Track5-Runner/models/final/laya-highway
test -f "$model_dir/model.safetensors"
printf '%s  %s\n' '95ea832b40e5719984ca7400ce93e36f26c659dca366d482d4151158542a2607' "$model_dir/model.safetensors" | sha256sum -c -
job_dir=$(mktemp -d "${TMPDIR:-/tmp}/track5-threshold.XXXXXXXX")
trap 'rm -rf -- "$job_dir"' EXIT
git archive "$source_sha" | tar -x -C "$job_dir"
cd "$job_dir"
echo "Source revision: $source_sha"
echo "Fine-tuned source: $model_dir"
export MPLBACKEND=Agg
export HF_HUB_OFFLINE=1
python -m unittest discover -s tests -p test_track5_threshold_analysis.py -v
python -m figures.track5_threshold_analysis --finetuned "$model_dir" --out-dir results/track5/threshold_analysis
status_dir="$runner_root/bishal_autorunner/status"
for name in action_thresholds.csv unsafe_score_distribution.png threshold_acceptance.png; do
  destination="$status_dir/track5_${source_sha}_${name}"
  test ! -e "$destination"
  cp "results/track5/threshold_analysis/$name" "$job_dir/$name"
  mv "$job_dir/$name" "$destination"
  echo "Analysis artifact: $destination"
done
