# Post-hoc diagnostics specification

Date: 2026-11-23, Asia/Shanghai.

This is a descriptive, post-hoc analysis of an already completed experiment.
It is not a new preregistration, a model-selection procedure, or a new noise
experiment. No model fitting, parameter selection, coordinate perturbation,
feature regeneration, or modification of primary artifacts is allowed here.
G4, correlated noise, coordinate denoising, noise-augmented training, and new
smoothing lengths remain excluded.

## Inputs and protection

- Read the 16 existing feature caches and 48 existing frozen-model predictions.
- Read their 48 saved smoothed prediction files; do not rerun primary stages.
- Preserve original class order, metadata, point counts and window membership.
- Use the fitted imputer and scaler, not test-fitted normalization.
- Reuse the exact existing 5000-by-9 bootstrap multiplicity matrix.
- Hash primary files and original input/code files before and after analysis.
- Write all new artifacts below this directory, including logs and plot caches.
- Use the original audit environment because the bundled document runtime lacks
  scikit-learn, scipy, joblib, pyarrow and matplotlib.

## Feature and sampling diagnostics

Window-equal summaries: clean/noisy medians, paired signed/absolute shifts,
missing fractions, and absolute paired shifts divided by fitted training
standard deviations. These summarize distributions, not causal importance.
Stratify by majority class, linked user group, and original dt_mean:
`(0,1]`, `(1,2]`, `(2,5]`, `(5,inf)` seconds. Missing/nonpositive values have a
separate explicit bucket. These descriptive bins are chosen after seeing the
main results; dt columns are never introduced as model inputs.

## Frozen RBF diagnostics

For every window and all three models, compute the maximum RBF similarity to
the 1000 fitted basis points, Nystroem representation L2 norm, maximum absolute
decision-score residual from the classifier intercept, and predicted class.
Reconstructed scores must reproduce saved scores (absolute tolerance 1e-7).
Diagnostic near-intercept flag: max absolute residual <= 1e-6; it is not a
new classifier threshold. Report window-equal and point-weighted quantities
separately. Similarity to a finite basis is not distance to all training data.
Decision values are not calibrated probabilities.

## Smoothing diagnostics

Focus detailed sequence analysis on G3, all 16 conditions and all 3 fixed
smoothers. A true run is a consecutive sequence of window-majority labels in
one original user/block. Run length bins: 1, 2-5, 6-10, >=11 windows.
Duration bins: <=60, (60,180], (180,600], >600 seconds.

A run is "absent" when no predicted window in that true run has its mode.
"Newly absent from raw" means raw had at least one such window and smoothing
has none; report its denominator explicitly. It does not require that raw was
fully correct, so also report the subset with all raw windows correct.
These are sequence diagnostics, not point-level class F1.

Compute actual support spans from the first included window's start to the
last included window's end; centered smoothing uses at most 5 future windows,
trailing uses at most 10 past windows. Window count is not seconds.

Boundary matches reuse scipy maximum-cardinality bipartite matching, the same
block restriction and +/-1 window tolerance as the primary protocol. The
matching does not optimize timestamp differences; ties follow the existing
solver. Report signed boundary timestamp error and earliest score-availability
latency only for matched boundaries, together with missed-boundary counts.
Availability latency includes the required completed window(s). It is not
whole-system real-time latency. Do not infer all-transition latency from
this matched subset. Boundary matching remains class-blind; directional
class agreement is an additional descriptive column.

## Heterogeneity and uncertainty

Recompute fixed-five-class point metrics from per-window true point counts,
not majority-label sample weights. Show 13 users and 9 linked groups
separately, class support and confusion counts, and paired changes.
Leave-one-linked-group-out values are descriptive sensitivity checks, not
alternative primary estimates. Do not average seed confusion matrices before
computing bootstrap F1; compute each seed's metric first and then average.
The 5 perturbation seeds are not independent users.

## Interpretation and figures

All explanatory analyses are post-hoc and association-based. Do not attribute
the entire G3/G2 difference to jerk or turning features. Do not assert a
significant robustness ranking when the paired interval crosses zero.
Independent added E/N Gaussian noise is not calibrated real GPS temporal
noise. Sigma=0 means no added noise. Endo's cited noise experiment used KT,
not GeoLife; our settings are project-defined. Smoothing is in decision-score
space, not coordinate denoising.

Figures report seed mean +/- sample seed SD where applicable, not user CIs.
Sources and verification limits are recorded in SOURCES.md.
