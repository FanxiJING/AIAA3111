# Figure Captions

All three figures are exploratory diagnostics of the existing frozen
clean-training/noisy-test study. Noise is synthetic independent Gaussian
position error: east and north each have standard deviation sigma; horizontal
RMS is sigma times sqrt(2). No added noise at sigma=0 does not imply ground truth.
These plots do not establish robustness to real temporally correlated GPS error.
Our perturbation construction is project-specific; the earlier Endo study used
the KT dataset and does not validate this GeoLife protocol.

No smoothing is applied in these figures. The original S1/S2 analyses smooth
decision scores within blocks, not GPS coordinates, and are reported separately.
This round does not claim a new smoothing experiment.

## Figure 1

[PNG](figures/figure1_feature_recipe.png) /
[PDF](figures/figure1_feature_recipe.pdf).

Point-level fixed-five-class raw Macro-F1 under the original gamma=1/dimension
recipe (left) and the previously fitted common gamma=1/27 controls (right).
G3 is shared between panels. Nonzero-noise markers average five seedwise F1s;
error bars show plus/minus one sample seed SD. They are not across-user
confidence intervals. At sigma=0 a single condition is used.
The same 13 test users, nine linked groups and windows are retained in all cases.
Showing all conditions does not authorize selecting a model from test scores.

## Figure 2

[PNG](figures/figure2_kernel_and_predictions.png) /
[PDF](figures/figure2_kernel_and_predictions.pdf).

Median nearest-landmark RBF similarity (left) and point-weighted share predicted
as train (right), averaged across five noise seeds for nonzero sigma. Solid lines
are original models; dashed lines are existing common-gamma G1/G2 controls.
The left statistic is a median across windows, not points. High similarity alone
does not establish an in-distribution sample or a correct prediction.

The right panel deliberately reports one common class and is not an
intercept-collapse rate for every model. Original G2 instead has a walk-preferred
intercept and predicts walk for 81.43% of points at sigma=20m; its train share is
only 0.48%. Original G3 predicts train for 85.65% at that condition.
Use `kernel_summary.csv` and `model_configs.json` for model-specific intercept
concentration. Do not summarize this figure as all models collapsing to train.

## Figure 3

[PNG](figures/figure3_distance_and_error_flows.png) /
[PDF](figures/figure3_distance_and_error_flows.pdf).

Left: for G3, signed per-feature changes in gamma-weighted squared standardized
distance to each window's fixed clean-nearest landmark, aggregated into four
disjoint feature buckets. Curves average windows first, then noise seeds.
The reference is not reselected after perturbation. Contributions are additive
geometric quantities, not causal attribution of classification loss.
Heavy tails affect means; bucket medians are supplied separately and cannot
be added to obtain a median total.

Right: for G3 at sigma=20m, points changing from correct to wrong and from wrong
to correct relative to that model's clean predictions, divided by each true
class's point support. Counts use actual class counts inside mixed windows.
Bars average five seeds. They decompose correct-point changes rather than
Macro-F1. The train-class improvement occurs alongside false-positive inflows;
it is not evidence of overall robustness.

## Visual Verification

PNG/PDF export and automatic nonblank/text-in-canvas checks passed for all three.
Manual visual review has not passed: the image tool could not provide inspection.
Automatic checks alone do not certify absence of every overlap or presentation
problem. These captions and images remain drafts for team review.
