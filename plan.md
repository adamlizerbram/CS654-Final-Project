# Project Plan: DQN, Laya, and Laya-Shielded DQN in highway-env

**Timeline:** Oct 8 – Dec 8, 2026 · **Experiment freeze:** Nov 20 · **Report and presentation:** Nov 21 – Dec 8

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

## Step 0: Shared conventions (Oct 8–9, whole group)

Agree on these before anyone writes code, so everyone can work in parallel:

- [ ] **Environment config** (`configs/env.yaml`): lanes, traffic density, episode length. Every method uses it.
- [ ] **Agent interface:** `agent.act(env, obs) -> action` for DQN, Laya, and the shield.
- [ ] **Text format for Laya:** `laya_tools/highway_to_laya.py` (already written).
- [ ] **Results CSV columns:** `method, seed, episode, crashed, avg_speed, distance, reward`
- [ ] **Seeds:** fixed lists for training and evaluation. Evaluation seeds are never used for training.

---

## Step 1: Environment setup (Week 1: Oct 8–14)

- [ ] Python 3.10+ (required by Laya)
- [ ] Conda env from `environment.yml`: `conda env create -f environment.yml`
- [ ] Install packages:
  - highway-env: https://highway-env.farama.org/
  - Gymnasium: https://gymnasium.farama.org/
  - Stable-Baselines3 DQN: https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
  - Laya (install, API, `laya-train`): https://github.com/NandhaKishorM/laya
  - Laya weights: `convaiinnovations/laya` on Hugging Face
- [ ] Train on `highway-fast-v0` (faster, slightly less accurate); evaluate on `highway-v0`. Same for every method.
- [ ] **`smoke_test.py`** checks, in order:
  1. All packages import; versions print
  2. highway-env resets and runs 20 random steps
  3. SB3 DQN trains about 1,000 steps without errors
  4. Laya loads and answers one question about a described highway state
  5. GPU detection (CUDA / MPS / CPU fallback)
  6. Prints `ALL CHECKS PASSED`, or names the failed check
- [ ] **`README.md`**: prerequisites, install commands, running the smoke test, expected output, common errors, local vs. cluster notes
- [ ] Cluster access set up: https://sdsu-research-ci.github.io/instructionalcluster/architecture

**Checkpoint:** the smoke test passes on all five machines.

---

## Step 2: Connect the models to the simulator (Week 2: Oct 15–21)

- [ ] **Laya agent:** describe the state as text, ask the action choice question, return the top action
- [ ] **Laya shield:** score "unsafe" probabilities for DQN's ranked actions in one batched call; return the best safe action
- [ ] **Evaluation harness:** run any agent for N episodes on the evaluation seeds; write the results CSV; optional videos

---

## Step 3: Train DQN (Weeks 2–4: Oct 15 – Nov 4)

- [ ] Start from the highway-env quickstart SB3 DQN example: https://highway-env.farama.org/quickstart
- [ ] Train 3–5 seeds; save each model to `models/`; copy the models used in the report to `models/final/`
- [ ] *Stretch:* Double/Dueling DQN. SB3 provides only vanilla DQN; eleurent/rl-agents includes Double, Dueling, and prioritized replay: https://github.com/eleurent/rl-agents

---

## Step 4: Fine-tune Laya (Weeks 2–4: Oct 15 – Nov 4)

- [ ] Generate data with `make_laya_dataset.py`, running several copies in parallel with different `--seed-offset` values
- [ ] Separate train and eval files with non-overlapping seeds
- [ ] `laya-train --dry-run` to check the data (no model loaded)
- [ ] Train and calibrate: `laya-train --data train.jsonl --eval-data eval.jsonl --base convaiinnovations/laya --out ./laya-highway --shuffle-options`
- [ ] Compare accuracy and calibration, zero-shot vs. fine-tuned
- [ ] Record the label settings (3 s horizon, 5 m safety gap) and don't tune them against test results

**Checkpoint (end of week 3, Oct 28):** one DQN model and one Laya model each drive a full episode through the evaluation harness.

---

## Step 5: Combine them (Weeks 4–5: Oct 29 – Nov 11)

- [ ] Shield at test time: DQN proposes an action; Laya overrides it if P(unsafe) > threshold
- [ ] Sweep the threshold (e.g. 0.3–0.9) to map the safety vs. speed trade-off
- [ ] *Optional:* shield during training (works cleanly with DQN, which learns from the executed action)

**Checkpoint (end of week 5, Nov 11):** the shield works and the threshold sweep is done.

---

## Step 6: Final evaluation (Week 6: Nov 12–18)

- [ ] Run every condition on the same evaluation seeds
- [ ] *Optional:* lookahead oracle as an upper-bound reference
- [ ] Record demo videos, e.g. the shield overriding DQN

---

## Step 7: Analysis (Nov 12–20)

- [ ] Results table: mean ± confidence interval across seeds; save as `results/summary*.csv`
- [ ] Crash rate vs. average speed plot, one point per method
- [ ] Shield threshold sweep curve
- [ ] Calibration (reliability) plot: zero-shot vs. fine-tuned Laya
- [ ] Override rate: how often the shield changed DQN's action, and whether it prevented crashes

**Experiment freeze: Nov 20.** No new runs after this date.

---

## Step 8: Report and presentation (Nov 21 – Dec 8)

- [ ] Draft report by Dec 1
- [ ] Slides and rehearsal
- [ ] Buffer for Thanksgiving (Nov 26–27)
- [ ] Final submission: Dec 8

---

## Work split (5 people)

| Person | Role | Weeks 1–2 (Oct 8–21) | Weeks 3–5 (Oct 22 – Nov 11) | Weeks 6–7 (Nov 12–20) | Report section |
|---|---|---|---|---|---|
| **A** | Infrastructure and evaluation | Repo, `.gitignore`, environment, `smoke_test.py`, README, cluster access | Evaluation harness, results CSV, batch job scripts | Run the final evaluation | Setup and methodology |
| **B** | DQN | Run the SB3 quickstart; basic hyperparameter tuning | Train all seeds; *stretch:* Double/Dueling | Help integrate the shield with Q-values | DQN method and results |
| **C** | Laya agent | Install Laya; build the Laya agent; zero-shot tests | Laya shield wrapper (with E) | Zero-shot Laya results | Laya method |
| **D** | Laya fine-tuning | Dataset generation pipeline; parallel data runs | Fine-tuning, calibration checks | Fine-tuned Laya accuracy and calibration results | Fine-tuning and calibration |
| **E** | Integration and analysis | Plotting scripts on fake CSV data; report outline | Shield logic and threshold sweep (with B and C) | Plots, demo videos | Combined approach and overall analysis |

**Handoffs:**
- **Week 2:** A's smoke test and evaluation harness → everyone
- **Week 4:** B's trained DQN + C's shield wrapper + D's fine-tuned Laya → E's shield experiments

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
