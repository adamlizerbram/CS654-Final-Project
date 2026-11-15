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

## Commands

```bash
python -m laya_tools.finetune_laya --dry-run          # data + laya-train --dry-run
python -m laya_tools.finetune_laya --workers 16       # full run -> models/final/laya-highway
python -m figures.calibration --compute               # GPU: reliability table -> results/summary_calibration.csv
python -m figures.make_figures                        # all figures from committed CSVs
```

Unknown flags pass through to `laya-train`, e.g. `--epochs 8 --loss soft-ce`.

## Training run (Nov 14, 2026, A100)

Defaults: 4 epochs, `rlcd` loss, `--shuffle-options`, effective batch 64, 1,052 optimizer updates. Data: 16,805 train items, 400 calibration items, 3,451 val items.

| Question | Val items | Accuracy before → after | ECE before → after |
|---|---|---|---|
| Safety (`noul`) | 2,754 | 0.503 → 0.947 | 0.303 → 0.025 |
| Action (`choice`) | 697 | 0.144 → 0.891 | 0.586 → 0.048 |

These numbers come from `models/final/laya-highway/train_report.json`. Both fitted temperatures sit at the 5.0 maximum. That run's val set still included 13 items that also appear in train (0.4%).
