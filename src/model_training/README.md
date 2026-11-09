# Part 2 – Basic model training (GeoLife transportation-mode classification)

Input: teammate A's per-point data (`data/processed/quality_features/<user>.parquet`) and fixed split
(`data/splits/train_users.json`, `test_users.json`, `report.json`).
Classes: walk, bike, bus, car, train (Part 1 delivers these 5 and does **not** merge taxi/subway).

## Current version: v2 (`pipeline_v2.py`)
The headline results now come from the hardened v2 pipeline; the sections below describe v1
(`train_basic.py`, `tune_smoothing.py`, `evaluate_smoothing.py`), kept for reference.
- **Results and conclusions:** [`docs/part2_v2_findings.md`](../../docs/part2_v2_findings.md)
- **Notes for Part 1 and Parts 3/4:** [`docs/part2_notes_for_teammates.md`](../../docs/part2_notes_for_teammates.md)
- **Outputs:** `results/basic_w50_no_dt_v2/` (not in git), figures in its `figures/` with `captions.md`.

Final model: SVM (RBF, Nystroem 1,000 components, all training windows), balanced class weights,
soft voting over k = 11 windows. Test point macro-F1 0.779 -> 0.844 (smoothing gain +0.065, 95% CI
0.032-0.135). Changes against v1: blind-cut validation windows, 5 user-grouped folds, all models on the
same 20k subsample, per-model smoothing variant and k, selection on the full pipeline, bootstrap over
the 9 independent test groups.

```powershell
python build_windows.py --no-pure-train --out windows_w50_blind.csv   # blind windows (once)
python make_all_train_splits.py         # scratch split listing all 64 users as train (only for --stage allusers)
python build_windows.py --splits scratch\splits_all_train --out windows_w50_pure_all.csv   # label-cut windows, all users
python pipeline_v2.py --stage cv        # tune, choose variants and k, select the model (Nystroem SVM ~1 h)
python pipeline_v2.py --stage final     # fit on all training users, evaluate test users once
python pipeline_v2.py --stage allusers  # report-only grouped CV over all users with windows (61 of 64)
python notes_v2.py                      # analyses behind docs/part2_notes_for_teammates.md
python make_figures_v2.py               # fig1-fig5 (v2) -> results/basic_w50_no_dt_v2/figures/
```
`pipeline_v2.py --stage cv --resume` skips models whose CV result file already exists.

## Folder layout expected by the defaults
```
AIAA3111/
├── data/processed/quality_features/   <- Part 1 Parquet files (one per user) + report.json (not in git)
├── data/splits/                       <- Part 1's fixed split
└── src/model_training/                <- this folder
```

## Setup (Windows, once)
```powershell
cd src\model_training
pip install -r requirements.txt
```
(or, inside teammate A's repo environment: `uv add scikit-learn matplotlib joblib`)

## Run
```powershell
python build_windows.py                 # ~4.35M points -> windows_w50.csv (+ _report.json), a few minutes
python train_basic.py --no-tune         # quick check -> results/basic_w50_no_dt_notune/ (selects no model)
python train_basic.py                   # full run with grid search -> results/basic_w50_no_dt/ (headline run)
python train_basic.py --keep-dt         # ablation with dt_mean, dt_max -> results/basic_w50/
python build_windows.py --no-pure-train --out windows_w50_blind.csv   # blind windows for all users
python tune_smoothing.py                # choose smoothing k on training users (~10 min; --reuse-oof: seconds)
python evaluate_smoothing.py            # apply the chosen k to test users, build segments
python make_figures.py                  # report/poster figures -> results/figures/
```
`requirements.txt` pins the exact versions used for the reported results.
If memory runs short (16 GB machine, 20 CPU threads), limit the parallel workers with `$env:LOKY_MAX_CPU_COUNT = "8"` before training; a run with 8 workers reproduced the 20-worker scores exactly.

Try a different window length: `python build_windows.py --window 100`, then
`python train_basic.py --windows windows_w100.csv`.

Pipeline test without real data: `python make_mock_features.py --out mock_data`, then
`python build_windows.py --points mock_data/quality_features --splits mock_data/splits --out mock_data/w.csv`
and `python train_basic.py --windows mock_data/w.csv --splits-report mock_data/splits/report.json`.

## Method
**Why windows?** At inference time the input is an unsegmented stream of points (see teammate A's
note). Each `feature_block` is therefore cut into consecutive windows of W points (default 50). A
window is summarised into 29 features (27 go to the models, see below), classified, and its prediction is given to every point in it.
- Windows never cross `feature_block_id`.
- **Test users:** windows are cut *blind* (labels not used), as at real inference time.
- **Train users:** also cut where the true mode changes, so training windows are pure
  (`--no-pure-train` turns this off). Training windows with < 10 points are dropped.
- A tail shorter than W/2 is merged into the previous window, so every point is covered exactly once.

**Features per window:** the following statistics, plus duration, path length, straightness (displacement / path length), overall average speed, sampling interval and the valid-speed fraction:

| Quantity | Statistics |
|---|---|
| Speed | mean, std, min, max, p10, p50, p90, p95 |
| \|Acceleration\| | mean, std, p90, max |
| \|Jerk\| | mean, p90 |
| Heading-change rate | mean, p50, p90 |
| Event rates | stop rate (< 0.5 m/s), slow rate (< 2 m/s), sharp-turn rate (> 30°), large-acceleration rate (> 1 m/s²) |

**Why the sampling-interval features are excluded:** `dt_mean` and `dt_max` stay in the window table for QA but are not model inputs. They describe how often the device recorded, not how the person moved, and that shifts between users: the median car-window `dt_mean` is 4.12 s for train users but 2.0 s for test users. In the ablation, removing them did not lower CV macro-F1 for any model and left the selected model's test score essentially unchanged. They would also change directly in any reduced-sampling experiment. Use `--keep-dt` to reproduce the comparison.

**Models:**
- Baseline: majority class.
- Others: logistic regression, decision tree, k-NN, RBF-SVM (trained on a 20k stratified subsample because SVM training cost grows roughly quadratically with data size), random forest, HistGradientBoosting.
- HistGradientBoosting uses a fixed 300 iterations with early stopping off, because its internal validation split is random rather than grouped by user.
- Class weights are balanced wherever supported.

**Protocol:**
- Part 1's fixed users: 51 train and 13 test.
- Hyperparameters come from a small grid search with 3-fold `StratifiedGroupKFold` on **training users only**, optimising macro-F1.
- Groups = users, with Part 1's duplicate-linked users (`report.json → duplicate_linked_groups`) kept in the same fold.
- Test users are never used for tuning.
- **Model selection:** the selected model is the one with the highest CV macro-F1 on training users (`select_best`, marked `selected_by_cv` in `results.csv`). Test scores are reported for every model but never used to choose one. `results.csv` is sorted by test score for reading only, so its first row is not necessarily the selected model.
- `--no-tune` has no CV scores, so it selects no model and writes no `best_model.joblib` or confusion matrix.

**Temporal smoothing (final step of the model, `smoothing.py`):**
- Each window's prediction is replaced by the majority vote over k consecutive windows centred on it, **within the same `feature_block_id` only**; near a block edge the vote uses only the windows inside the block. Ties keep the original prediction.
- Why: without smoothing the SVM's predictions change mode 3,163 times inside test blocks while the true mode changes 529 times; 858 windows are isolated one-window flips, 82% of them wrong.
- **k is chosen from training users only** (`tune_smoothing.py`). Training windows used for fitting are cut at label changes and do not look like real input, so the training users' windows are cut blind as well (`windows_w50_blind.csv`). With the same user-grouped 3 folds as the grid search, each model is refitted with its tuned parameters on two folds and predicts the blind windows of the third. This gives out-of-fold predictions, and every k in {1, 3, 5, 7, 9, 11, 13, 15, 21} is scored on them.
- **Choice rule (paired one-SE):** best = the k with the highest mean fold macro-F1. For each k, take the per-fold difference to the best k; chosen = the smallest k whose mean difference is within one SE of that paired difference. The unpaired SE of the fold scores does not work here: fold difficulty differs a lot (SVM fold macro-F1 at k = 1: 0.639, 0.697, 0.556, a spread of 0.14). That spread is the same for every k, and with it even k = 1 counts as "within one SE", although k = 9 beats k = 1 in every fold by at least 0.033.
- Chosen k: SVM 9, logistic regression 13, random forest 9, HGB 13, k-NN 7, decision tree 9 (`smoothing_choice.json`). `evaluate_smoothing.py` reads this file and applies k once to the test users.
- **Trade-off (limitation):** smoothing removes real short segments too. On training out-of-fold predictions, SVM transition recall (true mode changes found within ±1 window) falls from 0.91 at k = 1 to 0.48 at k = 9, while transition precision rises from 0.14 to 0.25. Point F1 favours smoothing; detecting short trips or exact change points does not. A smarter final step (e.g. an HMM or change-point model) is future work.

**Segment output (`segments.py`):** consecutive windows with the same (smoothed) predicted mode inside a block are merged into one segment, which is teammate A's output format. Segment metrics are kept simple and work at window granularity:
- predicted vs true segment counts
- segment recall: the share of true segments matched by a predicted segment of the same mode with point-overlap IoU >= 0.5 (segment precision is the same for predicted segments)
- transition precision and recall: a predicted change counts as correct if a true change lies within ±1 window in the same block, and vice versa

True segments are runs of windows with the same majority true mode, so a change inside a window is placed at the nearest window boundary.

**Metrics:**
- **Main:** point-level macro-F1 on test users. The confusion matrix is built from each window's per-class point counts, so mixed windows are scored correctly.
- **Also reported:** point accuracy and balanced accuracy, plus window-level accuracy and macro-F1 (window label = majority mode).
- **Uncertainty:** 95% percentile bootstrap CIs for test point macro-F1 and accuracy, resampling test users with duplicate-linked users drawn together. All models share the same resamples, so model differences are paired. A resample may miss a class, so macro-F1 averages over the classes present in it. There are few independent user groups, so read these CIs as approximate.
- **CV spread:** mean and standard deviation of macro-F1 across the 3 folds, for each model's chosen parameters.
- **Per user:** point accuracy per test user and the unweighted user average, because a few users hold most test points.

## Outputs (`results/basic_w<W>_no_dt/`)
| file | content |
|---|---|
| results.csv | one row per model: point/window metrics, CV macro-F1 mean and std, best params, fit time |
| per_class_f1_points.csv | point-level F1 per mode per model |
| bootstrap_ci.csv | per model: test point macro-F1 and accuracy with 95% CIs, plus user-averaged accuracy |
| bootstrap_pairwise.csv | per model pair: paired difference with 95% CI and whether the CI includes 0 |
| per_user_test.csv | per test user: points, share of test points, resampling group, accuracy of each model |
| window_predictions_test.csv | every model's prediction for every test window (error analysis) |
| feature_importance.csv | random-forest impurity + permutation importance (macro-F1 drop), computed on test users: for reporting only, never for feature selection |
| model_comparison.png, confusion_matrix_points.png | figures for report/poster; the confusion matrix shows the CV-selected model |
| best_model.joblib | fitted CV-selected pipeline + feature list |
| run_config.json | arguments, sizes, feature list |

`results/basic_w50_no_dt/smoothing/`:
| file | content |
|---|---|
| smoothing_cv.csv | training out-of-fold scores per model and k: per-fold and pooled macro-F1, paired difference to the best k and its SE, mode changes, segment metrics |
| smoothing_choice.json | the rule and the chosen k per model (written by `tune_smoothing.py`, read by `evaluate_smoothing.py`) |
| oof_predictions_train.csv | out-of-fold predictions on blind training windows |
| smoothing_test.csv | test: baseline vs smoothed macro-F1 and accuracy with 95% CIs, mode changes, isolated flips |
| smoothing_bootstrap_pairwise.csv | test: paired gain from smoothing with 95% CI per model |
| smoothing_per_class_f1.csv | test: per-class F1, baseline and smoothed |
| segment_metrics_test.csv | test: segment counts, segment recall/precision, transition precision/recall |
| segments_test.csv | **final output**: single-mode segments of the selected model after smoothing |
| segments_test_baseline.csv | the same without smoothing |
| window_predictions_test_smoothed.csv, confusion_matrix_final_points.csv, per_user_test_smoothed.csv | smoothed window predictions, final confusion matrix, per-user accuracy |

`results/figures/`: fig1–fig5 PNG at 200 dpi plus `captions.md` (see `make_figures.py`).

## Reuse in Part 3/4
```python
from models import load_windows, prepare, cv_groups, train_and_evaluate, results_table
df = load_windows("windows_w100.csv")                      # e.g. another window size / resampled data
tr, te = df[(df.split == "train") & (df.n_points >= 10)], df[df.split == "test"]
Xtr, ytr, utr, _ = prepare(tr); Xte, yte, _, mte = prepare(te)
res = train_and_evaluate(Xtr, ytr, cv_groups(utr, "../../data/splits/report.json"), Xte, yte, mte)
print(results_table(res))

from models import select_best
from smoothing import smooth_majority                      # final step; Part 2 chose k = 9 for SVM
best = select_best(res)                                    # by CV macro-F1, never by test score
smoothed = smooth_majority(best.y_pred, te["feature_block_id"], te["start_time"], k=9)
# k = 9 belongs to the SVM; for another model or setting, rerun the training-only choice
# (tune_smoothing.py) instead of reusing it.
```
Same users, models, grids and metrics → results directly comparable with Part 2.
See `docs/part2_handover.md` for the reference configuration and the proposed scope of Parts 3/4.
