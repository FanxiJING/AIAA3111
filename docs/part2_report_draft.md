# DRAFT — Part 2 (model training): Methodology and Experiments

> **DRAFT for the team to verify and rewrite. Do not submit as is.**
> Drafted with an AI tool (see `docs/aigc_log_part2.md`); the course forbids submitting AI-generated
> text directly. Every number below comes from the files named next to it in
> `src/model_training/results/`; please re-check them against those files after any re-run.
> Only sources that appear in the repository docs are cited; everything else that needs a source is
> marked **[citation needed]**. Section numbers follow the course template (3 Methodology,
> 4 Experiments); merge with Parts 1, 3 and 4.

## 3 Methodology (Part 2)

### 3.x Problem setting at inference time

At inference time the input is an unsegmented stream of GPS points; the desired output is a set of
contiguous sub-sequences, each with one transport mode (teammate A's handover note). Following the
idea in the CNN-BiLSTM framework for joint mode and transition detection cited in that note [2], we
predict a mode for every point and then merge consecutive points with the same mode into segments.
Unlike that work, Part 2 uses classical classifiers on fixed-length windows.

### 3.x Windowing

Each feature block from Part 1 (`feature_block_id`, a stretch of points with valid motion features)
is cut into consecutive windows of W = 50 points. Windows never cross a block boundary. A remainder
shorter than W/2 is merged into the previous window, so every point belongs to exactly one window.
Every point inherits the prediction of its window.

- **Test users:** windows are cut *blind*, without using labels, as at real inference time.
  Test windows are 99.05% pure on average (96.4% fully pure), so mixed-mode windows are rare.
- **Training users:** windows used for fitting are additionally cut where the true mode changes, so
  each training window has a single label. Windows with fewer than 10 points are dropped.

### 3.x Window features

Each window is summarised by 27 features computed from Part 1's per-point motion features:
speed (mean, std, min, max, 10th/50th/90th/95th percentiles), absolute acceleration (mean, std,
90th percentile, max), absolute jerk (mean, 90th percentile), heading-change rate (mean, median, 90th
percentile), mean heading change, event rates (stop < 0.5 m/s, slow < 2 m/s, sharp turn > 30°,
large acceleration > 1 m/s²), duration, path length, straightness (displacement / path length),
overall average speed and the share of points with a valid speed. Speed, acceleration, jerk and
heading change are standard inputs for GPS mode detection [1, 3].

Two further statistics, the mean and maximum sampling interval (`dt_mean`, `dt_max`), are computed
but **not** used: they describe the recording device rather than the movement, and they differ
between users (median car-window `dt_mean` 4.12 s for training users vs 2.0 s for test users).
Section 4.x shows that removing them does not hurt.

### 3.x Classifiers

- majority-class baseline
- logistic regression (standardised features)
- decision tree
- k-nearest neighbours (standardised)
- RBF-kernel SVM (standardised), trained on a 20,000-window stratified subsample because SVM training
  cost grows roughly quadratically with the number of samples
- random forest (200 trees)
- histogram-based gradient boosting (300 iterations, early stopping off because its internal
  validation split is random rather than grouped by user)

Missing feature values are imputed with the training median, and class weights are balanced where
the model supports it. All models are from scikit-learn **[citation needed]**.

### 3.x Tuning and model selection

Hyperparameters are tuned by grid search with 3-fold `StratifiedGroupKFold` on the training users
only, maximising macro-F1. Folds are grouped by user, and users that share duplicate records
(Part 1's `duplicate_linked_groups`) are kept in the same fold to avoid leakage. The **final model is
the one with the highest CV macro-F1**; this rule was fixed before any test results were seen, and
test scores are never used to choose a model, features or any setting.

### 3.x Temporal smoothing

Windows inside one block are consecutive in time, but the classifier predicts each window
independently. Without smoothing, the selected SVM changes its prediction 3,163 times inside test
blocks while the true mode changes only 529 times; 858 windows are isolated one-window flips, and
82% of these are wrong. We therefore add a final step: each window's prediction is replaced by the
majority vote over the k consecutive windows centred on it, within the same block only (near a block
edge the vote uses only the windows inside the block); ties keep the original prediction.

**Choosing k on training users only.** Training windows used for fitting are cut at label changes, so
they do not look like real input. For tuning, the training users' windows are therefore cut blind as
well, exactly like the test windows. Using the same three user-grouped folds, each model is refitted
with its tuned hyperparameters on two folds and predicts the blind windows of the third. Every
k ∈ {1, 3, 5, 7, 9, 11, 13, 15, 21} is applied to these out-of-fold predictions and scored by point
macro-F1 per fold.

We choose k with a *paired* one-standard-error rule **[citation needed]**: the best k has the highest
mean fold macro-F1; for every k we take its per-fold difference to the best k, and choose the
smallest k whose mean difference is within one standard error of that paired difference. The usual
unpaired version (SE of the fold scores themselves) fails here. The three folds differ strongly in
difficulty (SVM fold macro-F1 at k = 1: 0.639, 0.697 and 0.556, a spread of 0.14). That spread is the
same for every k and makes the unpaired SE so large (≈ 0.04) that even k = 1 counts as "within one
SE", although k = 9 beats k = 1 in every fold by at least 0.033. The paired rule chooses k = 9 for the
SVM (logistic regression 13, random forest 9, gradient boosting 13, k-NN 7, decision tree 9). These
values were fixed in `smoothing_choice.json` before any test segment metric was computed.

### 3.x Segment output

Consecutive windows with the same smoothed prediction inside a block are merged into one segment
(user, trip, block, start and end time, number of points, predicted mode, majority true mode). Segment
evaluation works at window granularity: true segments are runs of windows with the same majority true
mode. We report segment counts; segment recall, the share of true segments matched by a predicted
segment of the same mode with point-overlap IoU ≥ 0.5 (segment precision likewise); and transition
precision and recall, where a predicted mode change counts as correct if a true change lies within
±1 window in the same block (and vice versa).

## 4 Experiments (Part 2)

### 4.x Setup

- **Data:** Part 1's output, 4,347,764 labelled GPS points from 64 users, five classes (walk 1,321,226;
  bus 1,192,765; bike 775,977; train 556,382; car 501,414 points). GeoLife dataset **[citation needed]**.
- **Split (Part 1, fixed):** 51 training users (3,486,792 points) and 13 test users (860,972 points),
  with duplicate-linked users on the same side.
- **Windows:** 70,441 training windows (70,086 after dropping 355 windows with < 10 points, which
  removes users 100 and 147 entirely, so **49 of the 51 training users contribute windows**) and
  17,380 blind test windows. Point totals match Part 1 exactly.
- **Metrics:** the main metric is point-level macro-F1 on test users, computed from each window's
  per-class point counts so mixed windows are scored correctly; we also report point accuracy. The
  95% CIs come from a bootstrap that resamples test users with replacement (5,000 resamples)
  **[citation needed]**, drawing duplicate-linked users together. The 13 test users form only
  **9 independent groups**, and one linked group (users 069, 126, 129, 167) holds **78.9%** of test
  points, so the CIs are wide and skewed. Model differences are tested as paired differences on the
  same resamples.

### 4.x Results

Table 1 – CV on training users and test results for every model (from `results.csv`,
`bootstrap_ci.csv`, `smoothing/smoothing_test.csv`).

| Model | CV macro-F1 (mean ± std over 3 folds) | Test macro-F1, no smoothing (95% CI) | k | Test macro-F1, smoothed (95% CI) | Test accuracy, smoothed (95% CI) |
|---|---|---|---|---|---|
| Majority baseline | – | 0.106 (0.094–0.177) | – | – | 0.359 (0.289–0.612) |
| **SVM (RBF), selected** | **0.653 ± 0.060** | 0.760 (0.404–0.792) | 9 | **0.821 (0.460–0.856)** | 0.845 (0.744–0.898) |
| Logistic regression | 0.631 ± 0.052 | 0.717 (0.375–0.757) | 13 | 0.785 (0.462–0.829) | 0.822 (0.715–0.931) |
| Random forest | 0.618 ± 0.049 | 0.770 (0.415–0.791) | 9 | 0.830 (0.467–0.853) | 0.850 (0.779–0.864) |
| Gradient boosting | 0.607 ± 0.051 | 0.767 (0.400–0.796) | 13 | 0.844 (0.474–0.875) | 0.854 (0.761–0.935) |
| k-NN | 0.585 ± 0.062 | 0.752 (0.386–0.761) | 7 | 0.804 (0.417–0.819) | 0.841 (0.585–0.862) |
| Decision tree | 0.581 ± 0.040 | 0.703 (0.369–0.725) | 9 | 0.787 (0.424–0.811) | 0.817 (0.728–0.837) |

Observations (to verify and reword):
- **Every model beats the majority baseline** by a wide margin; the paired CIs of all differences
  exclude 0.
- **The top models cannot be separated.** Without smoothing, the paired differences between SVM,
  random forest, gradient boosting and k-NN all have 95% CIs that include 0
  (`bootstrap_pairwise.csv`). Only SVM vs logistic regression, gradient boosting vs logistic
  regression (macro-F1 only), and SVM / random forest / gradient boosting vs the decision tree are
  clearly different. These are 15 uncorrected comparisons. Random forest and gradient boosting score
  slightly higher than the SVM on test, but the SVM remains the final model because the selection
  rule uses CV only.
- **Selection vs smoothing.** With smoothing included, logistic regression is marginally ahead of the
  SVM on training out-of-fold macro-F1 (0.752 vs 0.746 at k = 9, pooled over folds). The gap is small
  compared with the fold spread (CV std ≈ 0.05), and the selection rule was fixed before smoothing was
  added, so the SVM is kept.
- **CV is lower than test.** CV macro-F1 (0.58–0.65) is below test (0.70–0.77 without smoothing). The
  SVM's CV value lies inside its test CI. A likely reason is the concentration of the test set: the
  dominant linked group may simply be easier than the average CV fold. We did not test this further.
- **Smoothing helps every model.** The paired gain in test macro-F1 is positive for all six models
  with CIs above 0. For the SVM it is +0.061 (0.018–0.108) in macro-F1 and +0.048 (0.002–0.072) in
  accuracy (`smoothing_bootstrap_pairwise.csv`). For k-NN the accuracy gain's CI touches 0. Predicted
  mode changes fall from 3,163 to 737 (true: 529) and isolated flips from 858 to 44.

Table 2 – Per-class test F1 of the SVM (`smoothing/smoothing_per_class_f1.csv`).

| | walk | bike | bus | car | train |
|---|---|---|---|---|---|
| No smoothing | 0.865 | 0.836 | 0.698 | 0.565 | 0.836 |
| Smoothed, k = 9 | 0.877 | 0.902 | 0.768 | 0.669 | 0.891 |

Table 3 – Segment-level results of the SVM on test users (`smoothing/segment_metrics_test.csv`).

| | Segments (true: 1,738) | Segment recall (IoU ≥ 0.5) | Segment precision | Transition precision (±1 window) | Transition recall |
|---|---|---|---|---|---|
| No smoothing | 4,372 | 0.724 | 0.288 | 0.203 | 0.894 |
| Smoothed, k = 9 | 1,946 | 0.727 | 0.649 | 0.388 | 0.482 |

Smoothing brings the number of predicted segments close to the true number and more than doubles
segment precision, but it halves transition recall: real short segments are smoothed away together
with the noise. The same trade-off appears on the training out-of-fold predictions (transition recall
0.91 at k = 1, 0.48 at k = 9), so it is a property of the method, not of the test set.

### 4.x Ablation: sampling-interval features

Removing `dt_mean` and `dt_max` (29 → 27 features) did not lower CV macro-F1 for any model
(changes +0.000 to +0.011). On test, without smoothing, the SVM changed by −0.0002, gradient boosting
by +0.010 and the decision tree by −0.020; the other models changed by less than 0.002. For gradient
boosting and k-NN the grid search also picked different hyperparameters, so their changes are not
purely due to the features. These test values are single estimates without CIs. Because the features
encode the device's sampling rate, which differs between users and would change directly in a
reduced-sampling experiment, the 27-feature set is the default (Figure 2).

### 4.x Failure analysis (final model: SVM, smoothed)

- **Bus is the hardest class to recall.** The largest confusions in point counts are bus → walk
  (29,809 points, 14.1% of bus points) and bus → car (19,136, 9.1%). A likely cause is that buses in
  traffic or at stops move at walking or car-like speeds; window features do not see road type or
  stop locations **[citation needed for GIS features in mode detection]**.
- **Car has the lowest precision** (0.651): car → train is 14.0% of car points, and 11% of car points
  are predicted as bus.
- **Wrong segments.** 511 of the 1,946 predicted test segments have the wrong mode, covering 12.3% of
  test points. They are mostly short (median 2 windows), but 56 wrong segments of ≥ 10 windows hold
  43,655 points, almost all from users 167, 052 and 126. These are whole sub-trips classified
  consistently wrong, which smoothing cannot fix.
- **Small users vary widely.** User 091 (703 points; 28 walk and 11 bus windows) has point accuracy
  0.35 without and 0.28 with smoothing; nine of its windows are predicted as car. Per-user accuracy
  ranges from 0.28 to 1.00 after smoothing, and the unweighted user average rises from 0.768 to 0.808
  (`smoothing/per_user_test_smoothed.csv`).
- **Mixed windows** (3.6% of test points) are classified correctly far less often than pure windows
  (window accuracy 0.52 vs 0.86 after smoothing).
- **Example trip** (Figure 5, chosen by a fixed rule): two isolated car predictions inside a bus
  stretch are removed by smoothing; the bus → walk and walk → bike changes are each predicted one
  window early, both with and without smoothing.

### 4.x Limitations

- **Test set dominated by one linked group.** One group of four linked users holds 78.9% of test
  points, and there are only 9 independent test groups. Test scores mainly reflect these users, and
  the bootstrap CIs are approximate.
- **49 of 51 training users contribute windows.** Users 100 and 147 have only 9 and 7 points.
- **The SVM is trained on a 20,000-window subsample** (of about 70,000), and its grid search ran on
  that subsample, so its comparison with the other models is not strictly like-for-like.
- **Smoothing trade-off.** Majority voting over k = 9 windows (about 15 minutes of travel at the
  median window duration) removes real short segments: transition recall drops from 0.91 to 0.48 on
  training out-of-fold predictions. Future work: a final step that models transitions explicitly,
  for example a hidden Markov model or change-point detection **[citation needed]**, or the joint
  mode and transition model of [2].
- **Window granularity.** Segment and transition metrics are measured at window level, so a mode
  change inside a window is placed at the nearest window boundary.
- **One configuration.** Window size, sampling rate, noise and feature-set effects are left to
  Parts 3/4.
- **Software versions.** Results were produced with pandas 3.1.1 and scikit-learn 1.9.1 only.

## Figures (in `src/model_training/results/figures/`; captions in `captions.md`)

1. `fig1_model_comparison.png` – test macro-F1 with 95% CIs, before and after smoothing, selected model marked
2. `fig2_dt_ablation.png` – CV and test macro-F1 with and without the sampling-interval features
3. `fig3_smoothing_per_class.png` – per-class F1 of the SVM before and after smoothing
4. `fig4_confusion_final.png` – point-level confusion matrix of the final model
5. `fig5_example_trip.png` – one test trip (median-length trip with a true mode change): true vs predicted modes

## References (only sources found in the repository docs; verify before use)

[1] Dabiri et al., 2020, IEEE TKDE. Linked from `FEATURES.md`:
https://people.cs.vt.edu/~clu/Publication/2020/TKDE-Dabiri-2020.pdf
— *full title and author list to be checked from the PDF.*

[2] "Convolutional Neural Network-Bidirectional Long Short-Term Memory-Based Framework for Joint
Transportation Mode and Transition Point Detection From GPS Trajectories", IEEE journal article
named in teammate A's handover note (`关于输入输出的一点说明.docx`).
— *authors, journal, year and DOI to be added from IEEE Xplore.*

[3] Ribeiro et al., 2024, linked from `FEATURES.md` (feature extraction, §3.1):
https://link.springer.com/article/10.1007/s41060-024-00510-3
— *full title and author list to be checked.*

Still needed: GeoLife dataset paper, scikit-learn, the bootstrap, the one-standard-error rule, HMM or
change-point smoothing, GIS features — all marked **[citation needed]** above.
