set -euo pipefail
source_sha=e60c15e3b320809936ed3a7080da5054b8427c2c
runner_root="$PWD"
echo "Runner revision: $(git rev-parse HEAD)"
git fetch origin refs/heads/feat/track5-safety-shield
git merge-base --is-ancestor "$source_sha" FETCH_HEAD
test "$(git rev-parse "${source_sha}^{commit}")" = "$source_sha"
job_dir=$(mktemp -d "${TMPDIR:-/tmp}/track5-demo.XXXXXXXX")
trap 'rm -rf -- "$job_dir"' EXIT
git archive "$source_sha" | tar -x -C "$job_dir"
cd "$job_dir"
echo "Source revision: $source_sha"
export MPLBACKEND=Agg
python -m unittest discover -s tests -p test_shield_demo.py -v
python -m unittest discover -s tests -p test_laya_shield.py -v
python -m examples.demo_shield
status_dir="$runner_root/bishal_autorunner/status"
for ext in csv png; do
  destination="$status_dir/mock_synthetic_shield_e60c15e.$ext"
  test ! -e "$destination"
  cp "results/track5/mock_synthetic_shield_decisions.$ext" "$job_dir/artifact.$ext"
  mv "$job_dir/artifact.$ext" "$destination"
  echo "MOCK / SYNTHETIC artifact: $destination"
done
