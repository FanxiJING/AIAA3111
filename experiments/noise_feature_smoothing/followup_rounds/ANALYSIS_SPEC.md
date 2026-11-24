# Three Follow-up Rounds

2026-11-23. These are post-hoc diagnostics of observed test results.
They are not new independent-user experiments or prospective validation.
No model fitting, tuning, noise generation, new bootstrap draws, G4, changed
features, changed windows, or replacement of frozen metrics is permitted.

## Round 1: Broaden The Mechanism Checks

Use five existing frozen models: original G1/G2/G3 and supplementary
common-gamma G1/G2. All use the 16 existing iid feature caches (80 pairs).
Reconstruct every saved score through the landmark-kernel algebra and
sample actual Pipeline scores. Verify feature and metadata alignment.
Measure nearest landmark RBF similarity, distance, score-minus-intercept
norm, and point-weighted class concentration.

For each model and window, retain its clean nearest landmark. Decompose
the change in squared standardized distance to this fixed reference into
speed (10), acceleration (5), jerk/turn (7), and remaining geometry/coverage
(5) columns present in that model. Multiply by that model's gamma.
Signed contributions can be negative and must sum to the total change.
This describes a geometric change; it is not a causal feature ablation.
Also check distances to independently reselected noisy nearest landmarks.
Numerical thresholds for tiny kernels remain descriptive, not OOD tests.

## Round 2: Paired Outcomes And Error Flows

Reconstruct five-class point confusion matrices for nine linked groups,
retaining mixed-window true point counts. Reuse the exact original
5,000 paired group bootstrap draws; score seedwise, then average.
Export absolute F1/accuracy, positive clean-minus-noisy F1 loss, and all
three feature-group pair contrasts under both original and common gamma.
Also export noisy-minus-clean differences of between-group gaps.
These are unadjusted exploratory intervals, not model-selection criteria.

For each condition compare predictions with that model's own clean
prediction. Count unchanged-correct, unchanged-wrong, rescued, damaged
and changed-still-wrong points separately by true class. Record complete
clean-predicted/noisy-predicted flows with each true class's point counts.
All cases and seeds remain included. Compare all group omissions without
refitting; do not substitute an omitted-group result for the main test.

## Round 3: Delivery And Checks

Export summary figures, captions, tables, a Chinese synthesis and a round
checkpoint. Independently verify metrics, feature distance decomposition,
score samples, error flows, bootstrap contrasts and source hashes.
Run automatic nonblank/layout checks; manual visual QA is only passed if
an image-capable tool actually provides inspection.
Retain any unresolved inherited limitations in the final report.

Each round writes a checkpoint and acceptance record. Execution-only
repairs are logged; at most three retries per round. This run ends after
these three rounds and their verification, rather than repeating identical
experiments indefinitely. Existing files are protected by content and
membership hashes; only this new directory and the disclosure log are excluded.

Primary-source reading limits and the scientific-critical-thinking software
reference are inherited explicitly from
[the investigation source record](../problem_investigation/SOURCES.md).
