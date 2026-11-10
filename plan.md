# Project Plan: DQN, Laya, and Laya-Shielded DQN in highway-env

**Timeline:** Nov 10 – Dec 8, 2026 (4 weeks) · **Experiment freeze:** Nov 29 · **Report and presentation:** Nov 30 – Dec 8

Optional and *stretch* items start only if the Nov 18 checkpoint is met.

**Conditions compared:**
1. DQN alone
2. Laya alone (zero-shot)
3. Laya alone (fine-tuned)
4. DQN with a Laya safety shield (zero-shot and fine-tuned Laya)
5. *Optional:* lookahead oracle (upper bound), Double/Dueling DQN

---

## Repository structure

```
project/
├── sim/            # highway-env setup shared by every method (make_env)
├── agents/         # everything that picks actions: DQN, Laya agent, Laya shield, oracle
├── laya_tools/     # Laya-specific code: highway_to_laya.py, make_laya_dataset.py
├── eval/           # evaluation harness and video recording
├── figures/        # plotting scripts and results tables
├── configs/        # env config, DQN hyperparameters, seed lists, shield threshold
├── README.md
├── environment.yml # conda env; pip-only packages (e.g. Laya) go under its pip: section
├── smoke_test.py
└── .gitignore
```

**Git-ignored** (large or easy to regenerate; share via the cluster or a shared drive):
`.conda/`, `models/`, `data/`, `results/`, `videos/`, `checkpoints/`, `runs/`, `wandb/`, `*.pt`, `*.pth`, `*.ckpt`

**Committed exceptions:** final weights in `models/final/` and summary tables matching `results/summary*.csv`, so graders can reproduce the report without retraining.

---

## Step 0: Shared conventions (Nov 10–11, whole group)

Agree on these before anyone writes code, so everyone can work in parallel:

- [ ] **Environment config** (`configs/env.yaml`, drafted): `highway-fast-v0` defaults pinned. Every method gets its env from `sim.make_env.make_env()`.
- [ ] **Agent interface** (`agents/base.py`, drafted): `agent.act(env, obs) -> int` for DQN, Laya, and the shield.
- [x] **Text format for Laya:** `laya_tools/highway_to_laya.py`
- [ ] **Results CSV columns** (`eval/results_schema.py`, drafted): `method, model_seed, env_seed, crashed, steps, avg_speed, distance, reward, threshold, overrides`
- [ ] **Seeds** (`configs/seeds.yaml`, drafted): DQN 0–4, Laya train data from 0, Laya validation from 10000, evaluation 100000–100099. Ranges never overlap.

---

## Step 1: Environment setup (Nov 10–13)

- [x] Python 3.11 (Laya needs 3.10+; torch 2.15 needs 3.11+)
- [x] Conda env from `environment.yml`: `conda env create -f environment.yml`
- [x] Install packages (versions pinned in `environment.yml`):
  - highway-env: https://highway-env.farama.org/
  - Gymnasium: https://gymnasium.farama.org/
  - Stable-Baselines3 DQN: https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
  - Laya (install, API, `laya-train`): https://github.com/NandhaKishorM/laya
  - Laya weights: `convaiinnovations/laya` on Hugging Face
- [x] Train and evaluate every method on `highway-fast-v0`: about 11× faster than `highway-v0`, with similar baseline crash rates (random policy 90% vs. 92%). Note its coarser 5 Hz physics in the report's limitations.
- [x] **`smoke_test.py`** checks, in order:
  1. All packages import; versions print
  2. highway-env resets and runs 20 random steps
  3. SB3 DQN trains about 1,000 steps without errors
  4. Laya loads and answers one question about a described highway state
  5. GPU detection (CUDA / MPS / CPU fallback)
  6. Prints `ALL CHECKS PASSED`, or names the failed check
- [x] **`README.md`**: prerequisites, install commands, running the smoke test, common errors; same commands locally and on the cluster; env in `./.conda/cs654`
- [ ] Cluster access set up: https://sdsu-research-ci.github.io/instructionalcluster/architecture

**Checkpoint (Fri Nov 13):** the smoke test passes on all five machines.

---

## Step 2: Connect the models to the simulator (Nov 12–17)

- [ ] **Laya agent:** describe the state as text, ask the action choice question, return the top action
- [ ] **Laya shield:** score "unsafe" probabilities for DQN's ranked actions in one batched call; return the best safe action
- [ ] **Evaluation harness:** run any agent for N episodes on the evaluation seeds; write the results CSV; optional videos

---

## Step 3: Train DQN (Nov 12–22)

- [ ] Start from the highway-env quickstart SB3 DQN example: https://highway-env.farama.org/quickstart
- [ ] Train 3–5 seeds; save each model to `models/`; copy the models used in the report to `models/final/`
- [ ] *Stretch:* Double/Dueling DQN. SB3 provides only vanilla DQN; eleurent/rl-agents includes Double, Dueling, and prioritized replay: https://github.com/eleurent/rl-agents

---

## Step 4: Fine-tune Laya (Nov 12–22)

- [ ] Generate data with `make_laya_dataset.py`, running several copies in parallel with different `--seed-offset` values
- [ ] Separate train and eval files with non-overlapping seeds
- [ ] `laya-train --dry-run` to check the data (no model loaded)
- [ ] Train and calibrate: `laya-train --data train.jsonl --eval-data eval.jsonl --base convaiinnovations/laya --out ./laya-highway --shuffle-options`
- [ ] Compare accuracy and calibration, zero-shot vs. fine-tuned
- [ ] Record the label settings (3 s horizon, 5 m safety gap) and don't tune them against test results

**Checkpoint (Wed Nov 18):** one DQN model and one Laya model each drive a full episode through the evaluation harness.

---

## Step 5: Combine them (Nov 19–24)

- [ ] Shield at test time: DQN proposes an action; Laya overrides it if P(unsafe) > threshold
- [ ] Sweep the threshold (e.g. 0.3–0.9) to map the safety vs. speed trade-off
- [ ] *Optional:* shield during training (works cleanly with DQN, which learns from the executed action)

**Checkpoint (Tue Nov 24):** the shield works and the threshold sweep is done.

---

## Step 6: Final evaluation (Nov 24–29)

- [ ] Run every condition on the same evaluation seeds
- [ ] *Optional:* lookahead oracle as an upper-bound reference
- [ ] Record demo videos, e.g. the shield overriding DQN

---

## Step 7: Analysis (Nov 25–29)

- [ ] Results table: mean ± confidence interval across seeds; save as `results/summary*.csv`
- [ ] Crash rate vs. average speed plot, one point per method
- [ ] Shield threshold sweep curve
- [ ] Calibration (reliability) plot: zero-shot vs. fine-tuned Laya
- [ ] Override rate: how often the shield changed DQN's action, and whether it prevented crashes

**Experiment freeze: Nov 29.** No new runs after this date.

---

## Step 8: Report and presentation (Nov 30 – Dec 8)

- [ ] Setup and method sections drafted during Nov 17–24 (they don't need results)
- [ ] Full draft report by Thu Dec 3
- [ ] Slides by Sat Dec 5; rehearsal Mon Dec 7
- [ ] Final submission: Tue Dec 8

Thanksgiving (Nov 26–27): no deliverables due.

---

## Work split (5 autonomous tracks)

To eliminate blockers and allow all five members to develop and test in parallel against shared interfaces (`agents/base.py`, `sim/make_env.py`, `eval/results_schema.py`, `configs/seeds.yaml`), the project is divided into five equal, autonomous tracks:

| Track | Assignee | Scope | Week 1 (Nov 10–16) | Weeks 2–3 (Nov 17–24) | Nov 24–29 | Report section |
|---|---|---|---|---|---|---|
| **Track 1: Evaluation Harness, Benchmark Orchestration & Metrics** | **Manish Thapa** | Evaluation runner, benchmark orchestrator, schema enforcement, CI metrics & summary tables | Repo/env verification, `eval/results_schema.py`, mock/baseline agents, `eval/evaluate.py` | Benchmark orchestration (`eval/run_benchmark.py`, `configs/benchmark.yaml`), `eval/metrics.py`, video capture | Run unified final evaluation across all conditions and generate `results/summary*.csv` | Setup and methodology |
| **Track 2: DQN Baseline & RL Policy** | **Placeholder 1** | SB3 DQN training pipeline, seed sweeps, `Agent` wrapper + Q-value ranking | Run SB3 quickstart on `highway-fast-v0`; hyperparameter tuning | Train seeds `0–4` (`models/final/dqn_seed*.zip`); expose ranked Q-values; *stretch:* Double/Dueling DQN | Finalize DQN baseline runs and support Q-value integration | DQN method and results |
| **Track 3: Zero-Shot Laya Agent & Driving Prompts** | **Placeholder 2** | Zero-shot Laya policy wrapper and action-choice prompt tuning/validation | Install Laya; implement `Agent` wrapper over `laya_tools/highway_to_laya.py` | Zero-shot validation on `highway-fast-v0`; prompt consistency checks | Generate zero-shot Laya evaluation results | Laya method |
| **Track 4: Laya Fine-Tuning & Calibration** | **Adam Lizerbram** | Dataset generation, `laya-train` fine-tuning, and probability calibration | Parallel data collection via `laya_tools/make_laya_dataset.py` (train offset `0`, val offset `10000`) | Run `laya-train --dry-run` and fine-tune (`models/final/laya-highway`); calibration checks | Compare zero-shot vs. fine-tuned accuracy and reliability plots | Fine-tuning and calibration |
| **Track 5: Safety Shield Wrapper & Event Metrics** | **Bishal Rauniyar** | Shield wrapper over DQN + Laya `P(unsafe)`, threshold sweep, event logging & plots | Plotting scripts on schema-compliant CSVs; report outline | Implement shield wrapper (`threshold`, `overrides` / `last_override` hooks); run `0.3–0.9` threshold sweep | Produce crash-vs-speed, threshold sweep, override-prevention plots & demo videos | Combined approach and overall analysis |

**Autonomous track interface contracts (no cross-track blocking):**
- Every policy or shield implements `agents.base.Agent` (`act(env, obs) -> int`, plus `name: str`).
- Shielded agents optionally expose `reset()`, `threshold`, and `overrides` (or per-step `last_override`) so Track 1's harness records threshold sweep and override counts automatically.
- Track 1 tests its harness and orchestrator using built-in/mock agents (`RandomAgent`, `IdleAgent`, `MockAgent`) independently of Tracks 2–5.

**Handoffs:**
- **Nov 13 / Nov 17:** Track 1's smoke test & evaluation harness (`eval/evaluate.py`, `eval/run_benchmark.py`) → all tracks
- **Nov 19:** Track 2's trained DQN + Track 3/5's Laya & shield wrappers + Track 4's fine-tuned Laya → unified benchmark & threshold sweeps

**Whole group:** Step 0 conventions, weekly check-in, report review, presentation.

---

## Key references

- highway-env documentation: https://highway-env.farama.org/
- Stable-Baselines3 DQN: https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
- rl-agents (Double/Dueling DQN): https://github.com/eleurent/rl-agents
- Laya: https://github.com/NandhaKishorM/laya
- Mnih et al., "Human-level control through deep reinforcement learning" (DQN), *Nature*, 2015
- van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning," AAAI 2016: https://arxiv.org/abs/1509.06461
- Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning," ICML 2016: https://arxiv.org/abs/1511.06581
- Alshiekh et al., "Safe Reinforcement Learning via Shielding," AAAI 2018: https://arxiv.org/abs/1708.08611
