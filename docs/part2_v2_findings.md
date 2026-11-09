# Part 2 v2 – findings for the report (notes, not report text)

Recorded 2026-11-09. Code: `src/model_training/pipeline_v2.py` (branch `part2-fixes`, commit `517bda2`).
Results: `src/model_training/results/basic_w50_no_dt_v2/`. Figures: `.../figures/` (fig1–fig5 + `captions.md`).
These are working notes with the numbers and where they come from. The report text must be written by the team.

## 1. What v2 changed against v1
| Item | v1 | v2 |
|---|---|---|
| Validation windows | cut at true mode changes (uses labels) | **blind-cut**, like test input |
| CV folds | 3 user-grouped | **5** user-grouped (StratifiedGroupKFold, linked users together) |
| Training data per model | SVM on 20k subsample, others ~70k | all on the **same 20k subsample**; extra Nystroem SVM on all windows |
| Smoothing | majority vote, k = 9 for all test rows | per model: own k, soft vs hard voting, class weights; variant adopted only if paired CV gain > 1 SE |
| Model selection | before smoothing | on the **full pipeline** (model + variant + k), paired one-SE rule |
| Test CIs | bootstrap | bootstrap over the **9 independent test groups** |

Effect of the blind-validation fix: unsmoothed SVM CV macro-F1 0.653 (v1) → 0.621 (v2). v1's CV was somewhat optimistic.

## 2. Model selection (training users only)
- Selected: **SVM (RBF, Nystroem, all data)**, C = 1, class weights balanced, **soft voting, k = 11**. CV macro-F1 **0.700 ± 0.034**.
- Next: SVM (RBF, 20k) 0.686, Logistic Regression 0.685, Random Forest 0.670, Decision Tree 0.659, HGB 0.657, k-NN 0.616, majority 0.076.
- vs SVM (RBF, 20k): paired gain +0.015, SE 0.0075, ahead or tied in all 5 folds → switch allowed by the pre-declared rule.
- Before smoothing the two SVMs are equal (0.623 vs 0.621); the gap comes from soft voting, which helps the Nystroem SVM (+0.009, SE 0.004, adopted) and hurts the 20k SVM (−0.008, not adopted). Do not claim "more data made it better".
- Class weights help every model strongly (turning them off: −0.05 for both SVMs).
- Secondary check, grouped CV over all users with windows (61 users, report only): Nystroem SVM 0.755 ± 0.037 first, RF 0.747, HGB 0.743, DT 0.734, SVM 20k 0.735, LR 0.720, k-NN 0.690. Same winner as the training-user CV.

## 3. Test results (13 test users, evaluated once)
| Model | Test macro-F1 (95% CI) | Accuracy | User-avg macro-F1 | Smoothing gain (95% CI) |
|---|---|---|---|---|
| **SVM (Nystroem, all data)** – selected | **0.844** (0.471–0.868) | 0.853 | 0.731 | **+0.065** (0.032–0.135) |
| HGB | 0.844 (0.488–0.871) | 0.860 | 0.755 | +0.088 (0.031–0.203) |
| SVM (RBF, 20k) | 0.838 (0.471–0.868) | 0.850 | 0.774 | +0.057 (0.016–0.132) |
| Random Forest | 0.835 (0.470–0.858) | 0.851 | 0.732 | +0.060 (0.029–0.149) |
| Decision Tree | 0.834 (0.484–0.870) | 0.844 | 0.760 | +0.096 (0.054–0.208) |
| Logistic Regression | 0.793 (0.467–0.839) | 0.828 | 0.751 | +0.076 (0.040–0.147) |
| k-NN | 0.770 (0.379–0.777) | 0.826 | 0.666 | +0.046 (0.008–0.071) |
| Majority baseline | 0.106 | 0.359 | 0.291 | 0 |

- v1 headline for comparison: SVM + majority vote k = 9, 0.760 → 0.821.
- **Smoothing helps significantly for all six trained models** (every gain CI above 0).
- **Top five are tied**: all pairwise CIs among Nystroem SVM, HGB, SVM 20k, RF, DT include 0. Only k-NN, LR and majority are clearly worse.
- Per-class F1, selected model, raw → final: walk 0.863 → 0.866, bike 0.842 → 0.913, bus 0.730 → 0.784, **car 0.612 → 0.750** (v1 final 0.669), train 0.848 → 0.909.

## 4. Limitations to state (findings, not bugs)
### 4.1 Over-smoothing blurs transitions
- Selected model, test: transition recall 0.879 → **0.261**, transition precision 0.202 → 0.304; segment precision 0.288 → **0.695**; predicted segments 4,223 → 1,682 (true 1,738). v1 at k = 9: transition recall 0.48.
- The same trade-off appears on training OOF (not only on test), selected model by k: transition recall 0.93 (k=1), 0.61 (5), 0.32 (9), **0.26 (11)**, 0.09 (21); segment precision 0.23 → 0.53 → 0.60 → 0.63 → 0.69. See fig5_v2.
- HGB keeps transition recall 0.50 at the same test macro-F1, but this was seen on test, so it is **not** a reason to switch models.
- The selection criterion optimises point macro-F1 only. Future work: transition-aware final step (HMM, change-point detection) or a criterion that also constrains transition recall.

### 4.2 Point-weighted vs user-averaged scores
- Selected model: point macro-F1 0.844 but user-averaged 0.731; SVM 20k has the highest user-averaged score (0.774).
- The test set is dominated by one linked group (users 167 + 126 + 069 + 129 = 679,162 of 860,972 points). Per user the selected model ranges from 0.054 (user 091, 703 points) to 0.943 (user 089).
- Report both numbers; the ranking depends on the weighting.

### 4.3 Sampling-rate artefact remains through window duration
- Windows have a fixed number of points (50), so their duration depends on the device's sampling interval.
- Car: median window duration **198 s (train) vs 98 s (test)**; median dt_mean 4.13 s vs 2.00 s (KS 0.47).
- Car jerk_mean test/train ratio 1.26 (KS 0.22), jerk_p90 1.23 (KS 0.21); acceleration features differ little (ratios 0.94–0.95).
- Within the same dt band the car jerk shift mostly disappears (1.5–2.5 s: 0.137 test vs 0.161 train), so it is a sampling effect, not driving behaviour.
- Dropping dt_mean/dt_max (v1 Finding 02) removed the direct feature, not the effect on window length and on time-difference features.
- Files: `notes/window_duration.csv`, `notes/accel_jerk_shift.csv`, `notes/accel_jerk_by_dt.csv`.

### 4.4 Possible label noise in car errors
- Largest car→train error: user 167, trip 167_000149, 22.6 min, median speed 15.5 m/s (~56 km/h), stop rate 0. Plausible as train or expressway driving; the label may be wrong. Not relabelled.
- Users 167 and 126 contain the same trip at the same timestamps (e.g. 2008-12-13 01:52–02:08), confirming they are duplicates; the grouped split already keeps them together.
- File: `notes/car_errors_top20.csv`.

### 4.5 Other
- 49 of 51 training users produce training windows (users 100 and 147 have 9 and 7 points). All 13 test users are scored on blind windows.
- The all-user CV uses **61 of 64 users**: it keeps users with at least one label-cut window of ≥ 10 points, which also drops test user 060 (17 points, split by a mode change into pieces under 10). So 61 = 64 − {060, 100, 147}; together 33 points.
- Bug found and fixed on 2026-11-09: the smoothing-gain CI was 0 for every model because raw and final predictions shared a name in `user_bootstrap` (commit `517bda2`). No other number was affected.

## 5. Suggested report and poster use
- Methodology: blind validation, 5 grouped folds, fair subsample, per-model smoothing variants, full-pipeline selection with paired one-SE rule.
- Experiments: tables in §2 and §3; fig1_v2 (model comparison), fig2_v2 (per class), fig3_v2 (confusion), fig4_v2 (example trip), fig5_v2 (k trade-off).
- Limitations: §4.1–4.4.
- Poster key claim: smoothing gives a significant gain for every model (selected +0.065, CI 0.032–0.135), car F1 0.61 → 0.75, at the cost of about three-quarters of real mode changes.
