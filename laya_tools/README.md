# Laya tools (Track 4: fine-tuning and calibration)

| File | Purpose |
|---|---|
| `highway_to_laya.py` | highway-env state → fixed-format text; the safety (`noul`) and action (`choice`) questions |
| `make_laya_dataset.py` | Labels states by simulating each legal action (one worker) |
| `finetune_laya.py` | Parallel data collection, merge, removing val/train overlap, `laya-train` dry run, fine-tune |

One checkpoint serves both roles. It is trained on safety rows and action rows together, and `laya-train` fits a separate temperature for each question type.

## Label settings

Fixed on Nov 14, 2026, before any evaluation. Do not tune these against evaluation results.

| Setting | Value | Defined in |
|---|---|---|
| Lookahead | 3 steps = 3 s: the action, then IDLE ×2 | `HORIZON` |
| Unsafe if | crash, or same-lane bumper gap < 5 m at any lookahead step | `MIN_SAFE_GAP` |
| Action label | safe action with the most forward progress (m, rounded); no row if none is safe | `make_laya_dataset.py` |
| Tie-break | IDLE > FASTER > LANE_LEFT > LANE_RIGHT > SLOWER | `PREFERENCE` |
| Collection policy | 50% random legal action, otherwise the labeled action | `--explore 0.5` |
| Sensing range in text | 80 m | `SENSING_RANGE` |
| Seeds | train 0–299, val 10000–10049 (eval 100000+ never used here) | `configs/seeds.yaml` |

Val rows whose input (state + question) also appears in train are dropped. `finetune_laya.py` does this after collection, and `figures/calibration.py` does it when scoring.

## Weights

The fine-tuned checkpoint is in `laya-highway/` of [adamlizerbram/cs654-final-project](https://huggingface.co/adamlizerbram/cs654-final-project) (public, Apache-2.0, like the base model). `models/` is git-ignored.

```bash
hf download adamlizerbram/cs654-final-project --include "laya-highway/*" --local-dir models/final   # -> models/final/laya-highway
```

`laya.load("adamlizerbram/cs654-final-project", subfolder="laya-highway")` also works without downloading first.

## Commands

```bash
python -m laya_tools.finetune_laya --dry-run          # data + laya-train --dry-run
python -m laya_tools.finetune_laya --workers 16       # full run -> models/final/laya-highway
python -m figures.calibration --compute               # GPU: reliability table -> results/summary_calibration.csv
python -m figures.make_figures                        # all figures from committed CSVs
```

Unknown flags pass through to `laya-train`, e.g. `--epochs 8 --loss soft-ce`.

## Results

### Training run (Nov 14, 2026, A100, about 85 min)

Defaults: 4 epochs, `rlcd` loss, `--shuffle-options`, effective batch 64, 1,052 optimizer updates. Data: 16,805 train items, 400 calibration items, 3,451 val items.

| Question | Val items | Accuracy before → after | ECE before → after |
|---|---|---|---|
| Safety (`noul`) | 2,754 | 0.503 → 0.947 | 0.303 → 0.025 |
| Action (`choice`) | 697 | 0.144 → 0.891 | 0.586 → 0.048 |

These numbers come from `models/final/laya-highway/train_report.json`. That run's val set still included 13 items that also appear in train (0.4%). Excluding them changes overall accuracy by at most 0.4 points.

Calibration fitted temperature T = 5.0 for both `noul` and `choice`, the maximum Laya allows. Even after maximum softening, both stay slightly overconfident: mean confidence is 0.971 vs 0.947 accuracy for safety, and 0.927 vs 0.891 for action choice.

### Reliability (`figures/reliability.png`, `results/summary_calibration.csv`)

These figures come from the 3,438 val rows whose input does not appear in train, scored in 10 equal-width bins.

| | Zero-shot | Fine-tuned |
|---|---|---|
| P(unsafe) rows in [0.3, 0.9) | 1,572 / 2,744 (57%) | 20 / 2,744 (0.7%) |
| Actual unsafe rate from the lowest to the highest plotted bin (≥ 20 rows) | 0.24 → 0.42 | 0.04 → 0.94 |
| Unsafe actions given P(unsafe) < 0.1 | 0 / 1,077 | 67 / 1,077 (6.2%) |
| Top bin, action choice: confidence vs accuracy | 0.93 vs 0.19 | 0.95 vs 0.92 (622 of 694 rows) |

- **Zero-shot:** P(unsafe) barely separates safe from unsafe actions, and it overstates risk everywhere above 0.3.
- **Fine-tuned:** P(unsafe) is nearly binary and well calibrated in both clusters: predicted 0.026 vs 0.041 observed near 0, and 0.974 vs 0.940 near 1.
- **For the shield:**
  - Any threshold between 0.1 and 0.9 gives almost the same overrides.
  - The 6.2% of unsafe actions scored below 0.1 pass at every threshold in that range.
  - Thresholds below 0.1 are the only ones that change anything; whether they catch those misses is untested.
