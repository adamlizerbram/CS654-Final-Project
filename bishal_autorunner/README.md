# Track 5 JupyterHub Autorunner (`bishal/autorunner`)

Temporary personal runner for Track 5: Safety Shield Wrapper & Event Metrics.

Adapted from Manish's `manish/autorunner` implementation. This runner allows jobs to execute on SDSU JupyterHub while synchronizing execution status and logs through GitHub.

## Start on JupyterHub

```bash
cd ~/CS654-Track5-Runner
mkdir -p logs

# The runner uses the existing Conda environment at:
# ~/CS654-Final-Project/.conda/cs654

nohup python bishal_autorunner/autorun.py --watch > logs/autorun_watcher.log 2>&1 &
```

The SSH agent must be running with the GitHub authentication key loaded before starting the watcher.

## How It Works

1. Update `bishal_autorunner/job.sh` and push the change to `bishal/autorunner`.
2. The JupyterHub watcher detects the updated job and executes it.
3. Execution status and logs are saved under `bishal_autorunner/status/`.
4. The runner commits and pushes status updates to `bishal/autorunner`.
5. Codex can inspect the synchronized results through GitHub.

## Branches

- `feat/track5-safety-shield` — Track 5 development, tests, and plotting scripts.
- `bishal/autorunner` — Job execution, status tracking, and logs.

Before executing Track 5 tests, ensure the runner checkout contains the intended development code. The autorunner does not automatically synchronize unmerged Track 5 changes.

## Notes

- `job.sh` should contain `# IDLE` when no job is scheduled.
- The watcher runs while the SDSU JupyterHub server remains available.
- The runner uses the existing project Conda environment rather than installing another copy.
- Keep credentials and personal development notes out of GitHub.
