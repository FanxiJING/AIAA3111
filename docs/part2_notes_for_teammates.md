# Notes for Part 1 and Parts 3/4 (from Part 2 v2)

Recorded 2026-11-09. Details and numbers: `docs/part2_v2_findings.md`. Results folder (not in Git, shared separately):
`src/model_training/results/basic_w50_no_dt_v2/`. Code on branch `part2-fixes`.

## Reference setup for your experiments
- Final model: SVM (RBF, Nystroem 1,000 components, LinearSVC C = 1), class weights balanced, soft voting over k = 11 windows within a feature block.
- Features: 27 window features (dt_mean, dt_max excluded). Windows: 50 points, never across `feature_block_id`; test windows cut blind.
- Test macro-F1 0.844 (95% CI 0.471–0.868). Top five models are tied, so compare your conditions with the same model and the same evaluation, not across models.
- Report both point-weighted and user-averaged scores: one linked user group holds ~79% of test points.

## 1. Sampling rate still leaks in through window length (most important, Parts 3/4 and Part 1)
- A 50-point window lasts a different time on different devices. Car: median **198 s in training vs 98 s in test**, because training car data was recorded at ~4 s intervals and test at ~2 s.
- Car jerk features are shifted (test/train ≈ 1.25, KS ≈ 0.22); within the same sampling-interval band the shift mostly disappears, so it is a sampling effect.
- Consequence for the reduced-sampling experiment: lowering the sampling rate also makes every window longer, so window length and sampling rate change together. Either keep this in mind when explaining the results, or cut windows by **fixed duration** (e.g. 60 s or 120 s) to separate the two.
- Suggested experiment: fixed-duration windows vs 50-point windows, same model and evaluation.

## 2. Possible label noise (Part 1)
- Some test segments labelled car look like train or expressway driving, e.g. user 167, trip 167_000149: 22.6 min, median ~56 km/h, no stops, predicted train. Listed in `notes/car_errors_top20.csv`.
- Not relabelled (that would bias the evaluation). Worth one sentence in the data-quality section.
- Users 167 and 126 contain the same trip at the same timestamps; the duplicate linking in `data/splits/report.json` is needed and works.

## 3. Small users (Part 1)
- Users 100 and 147 (training, 9 and 7 points) and 060 (test, 17 points) are too small for label-cut windows of ≥ 10 points. They are dropped from training and from the all-user CV (61 of 64 users); 060 is still scored in the fixed test split on its blind window.

## 4. Smoothing trade-off (Parts 3/4, interpretation)
- Smoothing raises point macro-F1 significantly for every model but removes about three-quarters of real mode changes (transition recall 0.88 → 0.26 for the selected model). If an experiment changes how often modes switch (e.g. window size), expect the best k to change too; choose k on training users only (`pipeline_v2.py --stage cv`).
