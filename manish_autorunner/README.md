# Personal JupyterHub Autorunner (`manish/autorunner` branch)

Temporary personal runner folder for Manish Thapa to run commands on SDSU NRP Nautilus JupyterHub and sync live logs/summaries (`manish_autorunner/status/`) via Git without cluttering `main` or PR branches.

## Start on JupyterHub

```bash
cd ~/RL/CS654-Final-Project
git fetch origin
git checkout manish/autorunner
conda activate ./.conda/cs654

# Keep watching in the background (survives browser disconnect):
nohup python manish_autorunner/autorun.py --watch > logs/autorun_watcher.log 2>&1 &
```

Or run a single command with automatic status/summary sync:
```bash
python manish_autorunner/autorun.py --cmd "python -m eval.run_benchmark --only random --episodes 20 --skip-existing"
```

## How It Works
1. Edit `manish_autorunner/job.sh` locally on Mac and push to `manish/autorunner`.
2. JupyterHub pulls `job.sh`, runs it inside `./.conda/cs654`, and pushes `manish_autorunner/status/state.json`, `manish_autorunner/status/latest.log` (last 400 lines), and any `results/summary*.csv` back to `manish/autorunner`.
3. Pull on Mac (`git pull`) to inspect live logs and summary tables.

## Cleanup Later
When the project is done (or before final merge):
```bash
git checkout main
git branch -D manish/autorunner
git push origin --delete manish/autorunner
```
