# Part 2 handover for Parts 3/4

Part 2 (basic model training) is finished on branch `part2-model-training`. This note says what the
reference configuration is, how to reuse the code, what to share outside git, and what Parts 3/4 own.
Code details: `src/model_training/README.md`.

## Reference configuration

| Setting | Value |
|---|---|
| Input | `data/processed/quality_features/` (Part 1), `data/splits/` (51 train / 13 test users) |
| Window | W = 50 points, never across `feature_block_id`; test windows cut blind |
| Features | 27 window features; `dt_mean`, `dt_max` excluded by default (`models.EXCLUDED_FEATURES`) |
| Models | majority baseline, logistic regression, decision tree, k-NN, RBF-SVM (20k subsample), random forest, HistGradientBoosting (300 iterations, no early stopping) |
| Tuning | grid search, 3-fold `StratifiedGroupKFold` on training users, linked users in one fold, macro-F1 |
| Model selection | highest CV macro-F1 → **SVM (RBF), C = 1, gamma = scale** |
| Final step | sliding majority vote over **k = 9** windows inside each block (k chosen on training users only, paired one-SE rule) |
| Main metric | test point-level macro-F1, with user-level bootstrap 95% CIs (linked users drawn together) |

**Reference results** (13 test users, final model):

| | No smoothing | Smoothed (k = 9) |
|---|---|---|
| Point macro-F1 (95% CI) | 0.760 (0.404–0.792) | 0.821 (0.460–0.856) |
| Point accuracy (95% CI) | 0.796 (0.712–0.847) | 0.845 (0.744–0.898) |

The paired gain from smoothing is +0.061 macro-F1 (95% CI 0.018–0.108). The CIs are wide because one
linked user group holds 78.9% of test points; compare settings with **paired** differences on the
same resamples (`models.user_bootstrap`), not by eye.

## How to reuse the code

Run from `src/model_training` (`pip install -r requirements.txt` first; versions are pinned).

```powershell
python build_windows.py --window 100                       # -> windows_w100.csv (other window size)
python build_windows.py --points <folder> --out windows_x.csv   # e.g. your resampled / noisy points
python train_basic.py --windows windows_w100.csv           # same models, grids, selection, metrics
```

From your own script:

```python
from models import (load_windows, prepare, cv_groups, train_and_evaluate, results_table,
                    select_best, user_bootstrap)
from smoothing import smooth_majority

df = load_windows("windows_w100.csv")
tr = df[(df.split == "train") & (df.n_points >= 10)]
te = df[df.split == "test"].reset_index(drop=True)
Xtr, ytr, utr, _ = prepare(tr)                 # 27 features; prepare(tr, exclude=()) keeps dt_*
Xte, yte, ute, mte = prepare(te)
res = train_and_evaluate(Xtr, ytr, cv_groups(utr, "../../data/splits/report.json"), Xte, yte, mte)
print(results_table(res))
best = select_best(res)                        # by CV macro-F1 on training users
smoothed = smooth_majority(best.y_pred, te["feature_block_id"], te["start_time"], k=9)
```

Rules that keep results comparable and honest:
- Never use test users to tune, select a model, choose features or choose k.
- k = 9 was chosen for the SVM at W = 50. If you change the window size, sampling rate or noise, the
  right k changes too: rerun the training-only choice (`tune_smoothing.py` with your window files),
  or report the unsmoothed result next to the smoothed one with k fixed in advance.
- Keep windows inside `feature_block_id`.
- If memory is short, set `$env:LOKY_MAX_CPU_COUNT = "8"`; results are identical.

## Files to share outside git

`results/`, the window tables and the data are git-ignored. Share them as a zip together with the
commit hash they came from:

| File / folder | Size note | Needed for |
|---|---|---|
| `data/processed/quality_features/` | ~450 MB (Part 1's zip) | rebuilding windows |
| `src/model_training/windows_w50.csv` | pure training + blind test windows | retraining without rebuilding |
| `src/model_training/windows_w50_blind.csv` | blind windows for all users | choosing smoothing k |
| `src/model_training/results/basic_w50_no_dt/` | headline run + `smoothing/` | numbers, predictions, segments |
| `src/model_training/results/basic_w50/` | run with `dt_*` (ablation) | the ablation figure |
| `src/model_training/results/figures/` | fig1–fig5 + `captions.md` | report and poster |

## What the dt ablation already showed

`dt_mean` and `dt_max` describe the sampling interval, not the movement, and they shift between users:
the median car-window `dt_mean` is 4.12 s for training users but 2.0 s for test users. Removing them
did not lower CV macro-F1 for any model, and the selected SVM's test macro-F1 changed by −0.0002.
They are therefore excluded by default. **A reduced-sampling experiment should not add them back**,
since they would change directly with the sampling rate.

## Proposed scope split

Part 2 deliberately stops at one configuration. Proposed ownership of the "realistic conditions"
experiments the course guide asks for:

| Part 3/4 topic | Starting point in Part 2 |
|---|---|
| Window size (e.g. W = 25, 100, 200) | `build_windows.py --window`; rerun the smoothing choice per W |
| Reduced sampling rate | resample the Part 1 points, then `build_windows.py --points` |
| Noisy or incomplete GPS | perturb or drop points before windowing; same pipeline |
| Feature-set experiments | `train_basic.py --drop-features ...` or `prepare(..., drop=[...])`; `feature_importance.csv` is from test users and must not be used to choose features |

Part 2 keeps: the reference configuration above, the smoothing step and the segment output. Open
Part 2 items that could move if someone has time: a smarter final step than majority voting
(transition recall drops from 0.91 to 0.48 at k = 9), and point-level labels for segment evaluation.
