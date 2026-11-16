set -euo pipefail
source_sha=313a3c9b443b57d8a387de07c55d2e491991a850
echo "Runner revision: $(git rev-parse HEAD)"
git fetch origin refs/heads/feat/track5-safety-shield
git merge-base --is-ancestor "$source_sha" FETCH_HEAD
test "$(git rev-parse "${source_sha}^{commit}")" = "$source_sha"
job_dir=$(mktemp -d "${TMPDIR:-/tmp}/track5-shield.XXXXXXXX")
trap 'rm -rf -- "$job_dir"' EXIT
git archive "$source_sha" | tar -x -C "$job_dir"
cd "$job_dir"
echo "Source revision: $source_sha"
echo "Python: $(python --version)"
python -m unittest discover -s tests -p test_laya_shield.py -v
