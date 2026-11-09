# Part 2 handover for Parts 3/4

Part 2 (basic model training) is finished. The **reference version is v2** (`pipeline_v2.py`, branch
`part2-fixes`, commit `93e50ac` or later; tag `part2-v2-reference` once created). v1
(`train_basic.py` + `tune_smoothing.py`, branch `part2-model-training`) is kept for comparison only.
Results and conclusions: `docs/part2_v2_findings.md`. Things to watch in your experiments:
`docs/part2_notes_for_teammates.md`. Code details: `src/model_training/README.md`.

## Reference configuration (v2)

| Setting | Value |
|---|---|
| Input | `data/processed/quality_features/` (Part 1), `data/splits/` (51 train / 13 test users) |
| Window | W = 50 points, never across `feature_block_id`; test windows cut blind |
| Features | 27 window features; `dt_mean`, `dt_max` excluded by default (`models.EXCLUDED_FEATURES`) |
| Models | majority baseline, logistic regression, decision tree, k-NN, RBF-SVM, random forest, HistGradientBoosting, all on the same 20k-window subsample; plus an RBF-SVM approximated with Nystroem features (1,000 components) + LinearSVC on all windows |
| Validation | 5-fold `StratifiedGroupKFold` on training users, linked users in one fold; models fit on label-cut windows and scored on **blind-cut** windows of the held-out users, point-level macro-F1 |
| Per-model final step | own smoothing k, soft vs hard voting, class weights on/off; a variant is adopted only if its paired CV gain is larger than one SE |
| Model selection | on the full pipeline (model + variant + k), paired one-SE rule → **SVM (RBF, Nystroem, all data), C = 1, balanced class weights** |
| Final step | **soft voting** (average class scores) over **k = 11** windows inside each block |
| Main metric | test point-level macro-F1; bootstrap 95% CIs over the **9 independent test groups** (linked users drawn together); also report user-averaged scores |

**Reference results** (13 test users, final model):

| | No smoothing | Smoothed (soft, k = 11) |
|---|---|---|
| Point macro-F1 (95% CI) | 0.779 | **0.844** (0.471–0.868) |
| Point accuracy (95% CI) | 0.809 | 0.853 (0.777–0.904) |
| User-averaged macro-F1 | – | 0.731 |
| Transition recall (±1 window) | 0.879 | 0.261 |

Smoothing gain +0.065 macro-F1 (95% CI 0.032–0.135). The top five models (Nystroem SVM, HGB, SVM 20k,
RF, DT: 0.834–0.844) are statistically tied. One linked user group holds ~79% of test points, so
compare settings with **paired** differences on the same resamples (`models.user_bootstrap`), not by eye.

## How to reuse the code

Run from `src/model_training` (`pip install -r requirements.txt` first; versions are pinned in
`results/basic_w50_no_dt_v2/run_config.json`).

Rebuild windows for your condition:
```powershell
python build_windows.py --window 100 --out windows_w100.csv                       # other window size
python build_windows.py --points <folder> --out windows_x.csv                     # resampled / noisy points
python build_windows.py --points <folder> --no-pure-train --out windows_x_blind.csv   # blind windows, for choosing k
```

Evaluate the reference model on your windows from your own script (hyperparameters fixed to v2):
```python
from types import SimpleNamespace
from models import load_windows, prepare, cv_groups, user_bootstrap, MODES
from pipeline_v2 import model_specs, fit_predict, smooth, SPLITS_REPORT

df = load_windows("windows_w100.csv")
tr = df[(df.split == "train") & (df.n_points >= 10)].reset_index(drop=True)
te = df[df.split == "test"].reset_index(drop=True)
Xtr, ytr, utr, _ = prepare(tr)                 # 27 features; prepare(tr, exclude=()) keeps dt_*
Xte, _, ute, mte = prepare(te)
spec = model_specs(Xtr.shape[1])["SVM (RBF, Nystroem, all data)"]
raw, scores, _ = fit_predict(spec, True, {"clf__C": 1.0}, Xtr, ytr, utr, Xte)   # True = balanced weights
final = smooth("soft", raw, scores, te["feature_block_id"], te["start_time"], k=11)
ci, pairs = user_bootstrap([SimpleNamespace(name="raw", y_pred=raw),
                            SimpleNamespace(name="final", y_pred=final)],
                           mte, cv_groups(ute, SPLITS_REPORT), MODES)
print(ci)
```

The full v2 pipeline (`python pipeline_v2.py --stage cv / final / allusers`) reads the fixed files
`windows_w50.csv`, `windows_w50_blind.csv`, `windows_w50_pure_all.csv` and writes to
`results/basic_w50_no_dt_v2/`. To rerun the whole selection for another condition, change those paths
and `OUT` near the top of `pipeline_v2.py` (there are no command-line options for them yet), and use
a new output folder so the reference results are not overwritten.

Rules that keep results comparable and honest:
- Never use test users to tune, select a model, choose features or choose k.
- k = 11 with soft voting was chosen for this model at W = 50. If you change the window size,
  sampling rate or noise, the right k changes too: choose it again on training users only
  (`pipeline_v2.py --stage cv` with your window files), or report the unsmoothed result next to the
  smoothed one with k fixed in advance.
- Keep windows inside `feature_block_id`.
- Report point-weighted and user-averaged scores together.
- The Nystroem CV stage takes about an hour on one core; leave the terminal alone while it runs.

## Where the results are

- **On GitHub (branch `part2-fixes`):** all of `src/model_training/results/` (~25 MB) is committed,
  although the folder is listed in `src/model_training/.gitignore` (added with `git add -f`). New
  result folders you create stay ignored unless you force-add them too.
- **Full bundle (zip, shared separately):** `part2_results_93e50ac_full.zip` (64 MB) holds the results
  plus the three window tables and their `_report.json`, the three Part 2 docs and a manifest. Unzip
  it into the repo root so the paths match.
- **Not in either:** Part 1's data (`data/processed/quality_features/`, ~450 MB); use Part 1's zip.
- `results/basic_w50_no_dt_v2/run_config.json` lists package versions and the seed.

| File / folder | Size | Needed for |
|---|---|---|
| `data/processed/quality_features/` | ~450 MB (Part 1's zip) | rebuilding windows |
| `src/model_training/windows_w50.csv` | 46 MB | label-cut training + blind test windows; retraining |
| `src/model_training/windows_w50_blind.csv` | 45 MB | blind windows for all users; CV scoring and k |
| `src/model_training/windows_w50_pure_all.csv` | 46 MB | label-cut windows for all users; all-user CV only |
| `src/model_training/results/basic_w50_no_dt_v2/` | 3.4 MB | **reference v2 run**: CV per model, test tables, predictions, segments, notes, figures |
| `src/model_training/results/basic_w50_no_dt/` | 14 MB | v1 headline run (comparison) |
| `src/model_training/results/basic_w50/` | 5 MB | v1 run with `dt_*` (dt ablation) |
| `src/model_training/results/figures/` | 0.5 MB | v1 figures |

## Known effects to account for in Parts 3/4
- **Sampling rate leaks in through window length.** A 50-point car window lasts a median 198 s in
  training but 98 s in test. Lowering the sampling rate also lengthens every window, so window length
  and sampling rate change together. Consider fixed-duration windows to separate them.
- **dt features stay out.** `dt_mean` and `dt_max` describe the sampling interval (car: 4.12 s train
  vs 2.0 s test). Removing them did not lower CV macro-F1 for any model. A reduced-sampling experiment
  should not add them back.
- **Smoothing trades transitions for clean segments.** Transition recall 0.88 → 0.26; segment
  precision 0.29 → 0.70.
- Details and numbers: `docs/part2_notes_for_teammates.md`.

## Proposed scope split

Part 2 deliberately stops at one configuration. Proposed ownership of the "realistic conditions"
experiments the course guide asks for:

| Part 3/4 topic | Starting point in Part 2 |
|---|---|
| Window size (e.g. W = 25, 100, 200), fixed-duration windows | `build_windows.py --window`; choose k again per setting |
| Reduced sampling rate | resample the Part 1 points, then `build_windows.py --points` |
| Noisy or incomplete GPS | perturb or drop points before windowing; same pipeline |
| Feature-set experiments | `prepare(..., drop=[...])`; feature importance from test users must not be used to choose features |
| Unseen users | already covered: all Part 2 evaluation is on users not seen in training; the all-user CV adds a second estimate |

Part 2 keeps: the reference configuration above, the smoothing step and the segment output. Possible
Part 2 extensions, reported next to the reference and not replacing it: a transition-aware final step
(HMM or change-point detection) and a selection criterion that also constrains transition recall.
