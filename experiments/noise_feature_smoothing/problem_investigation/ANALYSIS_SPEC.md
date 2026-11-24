# Read-only problem investigation

Fixed 2026-11-23 before producing the new diagnostic outputs. This is post-hoc
explanation of inspected test results, not independent validation or tuning.

1. Inspect every G3 test window in clean, all 15 iid and all 15 OU30 conditions.
   Use the existing fitted imputer, scaler and 1,000 RBF landmarks. Report
   training 0.5/99.5 percentile exceedance, standardised feature shifts, maximum
   landmark similarity and decision convergence toward the classifier intercept.
   Similarity <1e-6 is a descriptive threshold, not a calibrated OOD detector.
   Reconstruct decision scores algebraically and check against all saved scores;
   additionally check 64 evenly spaced windows through the actual Pipeline.
2. Decompose the existing raw point confusion matrices by all nine groups,
   all five classes, all sigma values and all existing seeds. Report pooled,
   linked0 and remaining-group results. Compute F1 per seed before averaging.
   Do not equate pooled Macro-F1 with group-averaged Macro-F1.
3. Use existing G3 sequences for all 31 conditions and all three smoothers.
   Keep the primary class-blind one-to-one +/-1-window metric. For diagnostic
   offsets use maximum-cardinality, then minimum-total-distance ordered matching.
   Report label-pair compatibility among matches, true-segment label presence,
   losses/gains versus raw, and actual support span of the 11-window operator.
   A separate +/-10-window label-pair-compatible match is descriptive only:
   extra matches do not prove that delay caused the primary misses.
   Length bands <=5, 6-10, >=11 windows are fixed to explain the k=11 operator.
4. Check displacement units, derivative differencing, wraparound, stationary
   block resets, class ordering and causal smoothing with known-answer tests.
   Describe inherited feature issues; do not correct their frozen definitions.
5. Hash protected input files before and after analysis, preserving content and
   membership. Only this folder and the declared AIGC log may be written.
6. Research primary sources for GPS error autocorrelation, RBF/Nystroem and
   temporal oversegmentation. Record exact retrieval and reading limits.
   A source supports only its own scope, not validation of this experiment.

No model fitting, parameter selection, new perturbation experiments, sample
deletion, coordinate filtering, new bootstrap draws or replacement main metrics.
No p-values or additional confirmatory confidence intervals are introduced.

Scientific-critical-thinking informed evidence appraisal. Software reference:
Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents.
arXiv:2609.00065. DOI:10.48550/arXiv.2609.00065.
