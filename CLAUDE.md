# Data Mining group project (AIAA 3111, HKUST-GZ) – context for Claude Code

## Project
GeoLife GPS transportation-mode classification: walk, bike, bus, car, train (5 classes, no merging).
Team split: Part 1 = pre-processing & feature engineering (teammate A, done, repo
https://github.com/FanxiJING/AIAA3111), **Part 2 = basic model training (this user)**,
Parts 3/4 = multi-variable experiments (other teammates, will reuse Part 2's code).
Deadlines: poster 27 Nov, final zip (Report.pdf + code + README + data instructions) 11 Dec.
The report needs an AIGC disclosure section, so keep a short log of what AI tools did.

## Folder layout (this repo)
- `data/processed/quality_features/<user>.parquet` – Part 1 output, not in git, ONE ROW PER GPS POINT (4,347,764 points, 64 users).
  Key columns: user_id (string, keep leading zeros), trip_id, feature_block_id, timestamp, latitude,
  longitude, mode, delta_t_s, distance_m, speed_mps, acceleration_mps2, jerk_mps3,
  heading_change_rad, heading_change_rate_rad_s (+ *_valid flags, QA columns).
  Definitions: FEATURES.md / README.md in the repo root.
- `data/splits/` – Part 1's fixed split: train_users.json (51), test_users.json (13), report.json
  (duplicate_linked_groups must stay in the same CV fold), manifest.csv.
- `src/model_training/` – Part 2 code (see its README.md):
  - `build_windows.py` – cuts each feature_block into W-point windows (default 50), never across
    feature_block_id; test windows cut blind (no labels), train windows also cut at label changes;
    29 summary features per window (27 are model inputs; dt_mean, dt_max are excluded in models.py)
    + per-class point counts -> `windows_w50.csv`.
  - `models.py` – model zoo (majority baseline, LogReg, DecisionTree, kNN, RBF-SVM on 20k subsample,
    RandomForest, HistGradientBoosting), user-grouped CV, point-level metrics from window counts.
  - `train_basic.py` – trains/tunes all models (grid search on training users only, macro-F1),
    selects by CV macro-F1, evaluates on test users; main metric = POINT-level macro-F1, with
    user-level bootstrap CIs. Outputs in results/basic_w<W>_no_dt/ (results/basic_w<W>/ with --keep-dt).
  - `make_mock_features.py` – synthetic point data for testing only (never report its numbers).
  - `smoothing.py` – sliding majority vote over k windows inside each block (final model step).
  - `segments.py` – single-mode segment output and segment metrics.
  - `tune_smoothing.py` – chooses k on training users only (blind windows, out-of-fold, paired one-SE
    rule) -> `smoothing_choice.json`; `evaluate_smoothing.py` applies it once to test users.
  - `make_figures.py` – fig1–fig5 + captions -> `results/figures/`.
- `src/model_training/scratch/` – temp files and small analysis scripts, not in git.
- `docs/aigc_log_part2.md` – log of AI tool use for the report's Disclosures section.
- `docs/part2_handover.md` – reference configuration and scope split for Parts 3/4.
- `docs/part2_report_draft.md` – DRAFT Methodology/Experiments text (team must rewrite and verify).

## Status (2026-11-08)
- Part 2 pipeline has run on the real data. Point totals match Part 1 (3,486,792 / 860,972).
- Headline run: `results/basic_w50_no_dt/` (27 features). Selected by CV: SVM (RBF), CV macro-F1
  0.653 +/- 0.060. Final model = SVM + smoothing k = 9 (paired one-SE rule on training out-of-fold
  predictions): test point macro-F1 0.821 (95% CI 0.460–0.856), accuracy 0.845 (0.744–0.898);
  without smoothing 0.760 (0.404–0.792). Paired gain +0.061 (0.018–0.108).
- Segments (SVM, test): 1,946 predicted vs 1,738 true; segment recall 0.727, precision 0.649;
  transition recall 0.48 at k = 9 vs 0.89 unsmoothed (over-smoothing trade-off, documented).
- SVM, Random Forest, HGB and k-NN cannot be separated: their paired macro-F1 differences have CIs
  that include 0. Only 9 independent test groups exist; one linked group holds 78.9% of test points.
- `results/basic_w50/` keeps the with-dt ablation run. Results and windows are not in git.
- Versions pinned in requirements.txt (pandas 3.1.1, scikit-learn 1.9.1); pandas 2.x untested.

## Next steps
1. Team rewrites `docs/part2_report_draft.md`; fill the [citation needed] items with real sources.
2. Poster: use `results/figures/` (fig1, fig3, fig5 are the strongest for the poster).
3. Share windows + results zip with Parts 3/4 (see `docs/part2_handover.md`).
4. Future work: transition-aware final step (HMM / change-point) instead of majority voting.
5. Optional: window-size comparison (`--window 25/100/200`); point-level deep model
   (1D-CNN + BiLSTM, per teammate A's note) – GPU available via the course HPC web portal
   (login.hpc.hkust-gz.edu.cn, campus network only, billed, always click Stop).

## Rules
- Never use test users for tuning, model selection, feature selection or choosing the smoothing k.
- Windows must not cross feature_block_id.
- Don't put credentials (HPC password, AIGC API key) in any file.
