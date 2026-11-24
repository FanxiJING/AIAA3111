# Prospective Supplemental Protocol

Fixed on 2026-11-23, before any outcomes from these supplemental conditions.
The original test results have ALREADY been inspected. This is an exploratory,
post-hoc extension, not preregistration, independent validation, or test tuning.
The original fixed experiment and v2 outputs remain authoritative and unchanged.

## Questions And Conditions

1. Temporal noise structure: evaluate the original frozen G3 on stationary
   Ornstein-Uhlenbeck (OU) E/N position noise, with time constant **30 seconds**.
   This single illustrative time constant is fixed without a parameter search;
   it is not estimated from or calibrated to real GPS errors. Compare with the
   original iid position-noise conditions.
2. Gamma control: fit G1 and G2 on the original clean training windows with
   **gamma=1/27**, matching original G3. All other model settings are unchanged.
   Evaluate all original iid caches. G3 is reused, not refitted. Compare each
   fixed-gamma group with its original gamma=1/d counterpart, and compare the
   three groups at fixed gamma. No winner will be selected for deployment.
3. Group influence: omit each of the nine linked test groups in turn, then
   recompute pooled, fixed-five-class metrics without refitting any model.
   Show class supports, retained points and absolute/noisy-minus-clean scores.
   This is deterministic sensitivity analysis, not nine independent test sets.

## What Remains Fixed

Original train/test users, linked groups, label-cut fitting windows, blind test
windows, 50-point size, tail merging, point-count labels, original 27 features,
and inherited feature caveats are retained. No added training noise, point
deletion, test tuning, coordinate smoothing, new splits or feature corrections.
All fitting uses 70,086 eligible clean windows. Standardisation and imputation
are fitted only on training data. Nystroem: 1000 components, seed42; LinearSVC:
C=1, balanced, dual=False, max_iter=5000, seed42.

All conditions use raw scores, centered soft k=11, and causal trailing soft k=11,
each confined to the original block. Point metrics use all five cnt_* columns.
Boundary truth is window majority, class-blind one-to-one matching within +/-1
window. The original zero-denominator convention is retained and disclosed.
Existing 5,000 paired bootstrap draws across nine linked groups are reused.
Compute each seed's metric before averaging; seed SD is not user uncertainty.
Report absolute metrics and differences from the matching clean baseline.

## OU Generation

Per axis, within each original block:

    q[0] = z[0]
    a[i] = exp(-delta_t_seconds[i] / 30)
    q[i] = a[i] * q[i-1] + sqrt(1-a[i]^2) * z[i]
    error[i] = sigma * q[i]

E/N innovations are independent standard normals. Start each block from its
stationary marginal; no state crosses block boundaries. Use actual timestamps,
explicit timedelta seconds (never assume datetime integer units), and stable
original point order. Nonnegative delta_t is required. Theoretical per-axis
marginal SD is sigma, covariance sigma^2 exp(-abs(dt)/30); realized finite-path
RMS may differ and will not be renormalised. Do not resample unfavorable draws.

sigma = 5,10,20m; seeds = 20261123,20261124,20261125,20261126,20261127.
Generate the same user-wise innovation array as the original iid generator:
default_rng(SeedSequence([seed, int(user_id)])).standard_normal((n_points,2)).
Thus OU and iid share innovations, and all sigmas share one OU standard field.
Use the original spherical E/N displacement mapping, then recompute all
motion derivatives and original window features from the perturbed coordinates.
High speeds remain in the data and are counted. Zero noise reuses the original
clean cache/predictions after an explicit identity/recomputation check.

There are 15 new OU feature caches, 15 new G3 OU score files, two newly fitted
gamma-control models, and 32 new gamma-control iid score files. Reused results
are identified by source path and hash. No fabricated clean replicates.

## Interpretation And Delivery

Descriptive effect estimates and paired percentile intervals only; no multiple
testing significance claims, universal ranking, causal attribution to a single
feature, or real-device accuracy claims. A fixed numeric gamma does not make
effective kernel distances identical across different input dimensions.
LOO subsets can lack classes or boundaries; retain fixed-five scoring and
report supports. A time constant sensitivity condition is not coordinate denoising.

Deliver all scores, predictions, manifests, per-seed/per-user/per-class tables,
paired contrasts, LOO sensitivity tables, figures, report and executed validation.
Run cap: seven hours from supplemental initialization, checkpoint on errors or
deadline. Resume only complete artifacts whose input/protocol/code hashes match.
Original source/input/output content hashes are checked before and after; this
new directory and the explicitly updated AIGC log are excluded from protection.

## Sources Consulted Before Running

- Rasmussen & Williams (2006), Gaussian Processes for Machine Learning,
  chapter4 pp85-86: exponential covariance / OU process.
  https://gaussianprocess.org/gpml/chapters/RW4.pdf
- scikit-learn 1.9.1, Nystroem API: gamma and random_state definitions.
  https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html
- Original GPS review and reading limits:
  ../literature_review/SOURCES.md and ../literature_review/REVIEW_REPORT.md.
- Scientific-critical-thinking procedural support: Kassis, T., Agarwal, V.,
  He, Y., Patel, D., & Brueckner, A. M. (2026), Scientific Agent Skills:
  A Library of Procedural Knowledge for Research Agents, arXiv:2609.00065,
  DOI10.48550/arXiv.2609.00065. Metadata checked on 2026-11-23
  (current revision September2,2026); software disclosure, not empirical evidence.
