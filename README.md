# CS654-Final-Project

Final project for CS 654 Reinforcement Learning: DQN, Laya, and a Laya-shielded DQN in [highway-env](https://highway-env.farama.org/). See [plan.md](plan.md) for the plan and work split.

## Requirements

- conda ([Miniforge](https://github.com/conda-forge/miniforge) or any other distribution)
- About 5 GB of free disk space: packages, plus the Laya weights (about 2 GB, downloaded on first use)

Python 3.11 and all packages come from [environment.yml](environment.yml).

## Setup

Run these from the repo root, on your own machine or in a terminal on the cluster.

1. Clone the repo:
   ```bash
   git clone https://github.com/adamlizerbram/CS654-Final-Project.git
   cd CS654-Final-Project
   ```
2. Create the env inside the repo, in `.conda/` (git-ignored):
   ```bash
   conda env create -f environment.yml -p ./.conda
   ```
3. Activate it. Do this in every new terminal:
   ```bash
   conda activate ./.conda
   ```
4. Run the smoke test, which checks the packages, highway-env, SB3 DQN, Laya and the device (GPU or CPU):
   ```bash
   python smoke_test.py
   ```
   The first run downloads the Laya weights. It should end with `ALL CHECKS PASSED`; otherwise it prints `FAILED: <checks>` with a traceback.

After someone changes `environment.yml`, update your env:
```bash
conda env update -f environment.yml -p ./.conda --prune
```

## Common errors

- `ModuleNotFoundError: laya_tools` or `sim`: run from the repo root, and run scripts as modules (`python -m laya_tools.make_laya_dataset`).
- `ModuleNotFoundError` for `highway_env`, `laya`, etc.: the env isn't active; run `conda activate ./.conda`.
- Laya download fails or is rate-limited: run `huggingface-cli login`, or set `HF_TOKEN`.
- Lots of warnings during the Laya check: harmless if the check ends with `ok`.

## Outputs

`models/`, `data/`, `results/`, `videos/` and `logs/` are git-ignored. Final weights in `models/final/` and summary tables in `results/summary*.csv` are committed.
