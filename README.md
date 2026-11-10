# CS654-Final-Project

Final project for CS 654 Reinforcement Learning: DQN, Laya, and a Laya-shielded DQN in [highway-env](https://highway-env.farama.org/). See [plan.md](plan.md) for the plan and work split.

## Requirements

- conda ([Miniforge](https://github.com/conda-forge/miniforge) or any other distribution)
- About 5 GB of free disk space: packages, plus the Laya weights (about 2 GB, downloaded on first use)

Python 3.11 and all packages come from [environment.yml](environment.yml).

## Setup

Run these from the repo root:

1. Clone the repo:
   ```bash
   git clone https://github.com/adamlizerbram/CS654-Final-Project.git
   cd CS654-Final-Project
   ```
2. Create the env inside the repo, in `.conda/` (git-ignored), and make the prompt show `(cs654)` instead of the full path (once per machine):
   ```bash
   conda env create -f environment.yml -p ./.conda/cs654
   conda config --set env_prompt '({name}) '
   ```
3. Activate it. Do this in every new terminal:
   ```bash
   conda activate ./.conda/cs654
   ```
4. Run the smoke test, which checks the packages, highway-env, SB3 DQN, Laya and the device (GPU or CPU):
   ```bash
   python smoke_test.py
   ```
   The first run downloads the Laya weights. It should end with `ALL CHECKS PASSED`; otherwise it prints `FAILED: <checks>` with a traceback.

### Tokens (optional)

`HF_TOKEN` avoids Hugging Face rate limits; `GH_TOKEN` lets git push and pull over HTTPS without a password prompt (useful on the cluster).

```bash
cp .env.example .env            # then fill in the tokens
set -a; source .env; set +a     # load them; do this in every new terminal
git config --global credential.https://github.com.helper '!f() { echo username=x-access-token; echo "password=$GH_TOKEN"; }; f'   # once per machine
```

### Updating the env

After someone changes `environment.yml`, update your env:
```bash
conda env update -f environment.yml -p ./.conda/cs654 --prune
```

## Common errors

- `CondaError: Run 'conda init' before 'conda activate'`: this may happen in a new session on the instructional cluster. Run:
  ```bash
  conda init bash
  source ~/.bashrc
  conda activate ./.conda/cs654
  ```
- `ModuleNotFoundError: laya_tools` or `sim`: run from the repo root, and run scripts as modules (`python -m laya_tools.make_laya_dataset`).
- `ModuleNotFoundError` for `highway_env`, `laya`, etc.: the env isn't active; run `conda activate ./.conda/cs654`.
- `You are sending unauthenticated requests to the HF Hub`, or the Laya download fails: set `HF_TOKEN` (see [Tokens](#tokens-optional)).
- `laya: this checkpoint ships invalid temperatures ... choice:11+`: harmless. It only affects choice questions with 11 or more options; ours have at most 5.

## Outputs

`models/`, `data/`, `results/`, `videos/` and `logs/` are git-ignored. Final weights in `models/final/` and summary tables in `results/summary*.csv` are committed.
