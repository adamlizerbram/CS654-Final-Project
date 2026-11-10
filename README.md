# CS654-Final-Project

My final project for CS 654 Reinforcement Learning: DQN, Laya, and a Laya-shielded DQN in [highway-env](https://highway-env.farama.org/). See [plan.md](plan.md) for the plan and work split.

## Prerequisites

- [Miniforge](https://github.com/conda-forge/miniforge) (or any conda). macOS: `brew install --cask miniforge`, then `conda init zsh` and open a new terminal.
- About 5 GB of free disk space: about 3 GB for packages and about 2 GB for the Laya weights, downloaded on first use.
- Python 3.11 comes from `environment.yml`; you don't install it yourself.

## Install

```bash
conda env create -f environment.yml
conda activate cs654
```

After someone changes `environment.yml`: `conda env update -f environment.yml --prune`.

## Smoke test

Run from the repo root, so `laya_tools` can be imported:

```bash
python smoke_test.py
```

Expected output (the first run takes about 1–2 min while the Laya weights download):

```
[imports]
    highway-env 1.12.1
    gymnasium 1.4.0
    stable-baselines3 2.9.2
    torch 2.15.0
    laya 0.4.30
    ok
[highway-env]
    ok
[SB3 DQN]
    ok
[Laya]
    state: Highway with 4 lanes, ... Proposed action: change one lane to the left.
    P(unsafe) = 0.62
    ok
[device]
    MPS (Apple Silicon)      # or CUDA: <GPU name>, or CPU only
    ok
ALL CHECKS PASSED
```

It ends with `ALL CHECKS PASSED` (exit 0), or with `FAILED: <checks>` and a traceback for each failed check (exit 1).

## Common errors

| Error | Fix |
|---|---|
| `ModuleNotFoundError: laya_tools` | Run from the repo root. Run scripts as modules: `python -m laya_tools.make_laya_dataset ...` |
| `ModuleNotFoundError` for highway_env, laya, etc. | Activate the env first: `conda activate cs654` |
| `torch` install fails on an Intel Mac | `pip install "numpy<2" "torch==2.2.2" "transformers==4.57.6"` inside the env |
| Laya download fails or is rate-limited | `huggingface-cli login`, or set `HF_TOKEN` |
| Lots of warnings during the Laya check | Harmless if the check ends with `ok` |

## Local vs. cluster

- **Local:** fine for development, the smoke test, short DQN runs and Laya inference. Apple Silicon uses MPS automatically.
- **SDSU instructional cluster:** use it for multi-seed DQN training, Laya fine-tuning and the final evaluation runs. It has A100 GPUs and 80 GB of storage per user, accessed through JupyterHub. See [logging in](https://sdsu-research-ci.github.io/instructionalcluster/students/loggingin) and [launching a container](https://sdsu-research-ci.github.io/instructionalcluster/students/launchcontainer). Request a GPU only for Laya fine-tuning; highway-env and DQN are CPU-bound.
- Large outputs (`models/`, `data/`, `results/`, `videos/`) are git-ignored. Share them through the cluster or a shared drive. Final weights go in `models/final/` and summary tables in `results/summary*.csv`; both are committed.
