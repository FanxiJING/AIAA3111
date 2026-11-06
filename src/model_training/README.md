# Part 2 – Basic model training (GeoLife transportation-mode classification)

Input: teammate A's per-point data (`data/processed/quality_features/<user>.parquet`) and fixed split
(`data/splits/train_users.json`, `test_users.json`, `report.json`).
Classes: walk, bike, bus, car, train (Part 1 delivers these 5 and does **not** merge taxi/subway).

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
```
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

## Reuse in Part 3/4
```python
from models import load_windows, prepare, cv_groups, train_and_evaluate, results_table
df = load_windows("windows_w100.csv")                      # e.g. another window size / resampled data
tr, te = df[(df.split == "train") & (df.n_points >= 10)], df[df.split == "test"]
Xtr, ytr, utr, _ = prepare(tr); Xte, yte, _, mte = prepare(te)
res = train_and_evaluate(Xtr, ytr, cv_groups(utr, "../../data/splits/report.json"), Xte, yte, mte)
print(results_table(res))
```
Same users, models, grids and metrics → results directly comparable with Part 2.
