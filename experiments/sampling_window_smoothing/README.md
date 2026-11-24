# GeoLife Part 3. Sampling, windows, and score smoothing

Question. How does changing the GPS sampling interval affect point classification and mode-switch
detection? Does a fixed-duration window or a shorter causal smoother help at lower sampling rates?

## Experiment

- The original five modes, 27 v2 features, training/test users, linked-user grouping, trajectory
  blocks, and SVM recipe are shared across comparisons. There is no injected positional noise.
- Keep native sampling, or greedily keep the first fix in each block and then the first fix at
  least 15 or 30 seconds after the last retained fix. Report realized intervals and point retention.
  Motion features are rebuilt on retained coordinates. A gap longer than 1,200 seconds resets
  the motion derivatives while keeping the original block for windowing and evaluation.
- Use 50-point and 120-second windows across sampling conditions. Compare 25-point, 100-point,
  and 60-second windows as extra native-rate controls. Fixed-point windows use the v2 tail rule.
  Time buckets start at the first available point in a block or fitting run, omit empty buckets,
  and keep nonempty tails. Fitting runs are split at retained label changes. Fitting requires
  at least 10 points for point-count windows and 3 for time windows. This allows the 120-second
  window to train at 30 seconds and means the comparison includes the fitting eligibility rule.
  Every blind evaluation window is retained.
- Train at the native rate and evaluate at all available rates for the same window definition.
  Also train matching 15- and 30-second models for the two crossed window definitions.
- Compare raw scores, centered 11-window averaging, trailing 3/5/11-window averaging, and
  trailing 120/300/600-second averaging. Use the supplied grouped v2 folds for every condition.
  Among causal settings within 0.02 of the best mean fold macro-F1 for each target rate, select
  the one with the highest mean release-time switch F1. Resolve ties by macro-F1 and then a
  deterministic ordering. The release-time metric determines selection only on training folds.
  The full test grid is reported as exploratory comparison.

## Shared Metrics

- Retrospective point-level macro-F1 and accuracy use the same 860,972 original test fixes for
  every condition. Attribute each original point to the latest window start inside its block.
  Five classes always contribute to the macro-F1 denominator. The separate user mean also
  uses all five classes and assigns zero F1 to absent classes.
- Match predicted switches one to one to original point-label changes within 120 seconds and
  within the original block. The metric checks switch timing irrespective of transition class.
  The window-start version measures retrospective alignment. The release-time version uses
  window availability and the latest prediction when several are released at the same second.
- For point windows, release waits until the next window reaches the size needed to resolve the
  tail merge, or until the last retained observation. For time windows, release occurs at the
  bucket deadline or original block end. The replay assumes block termination is known and uses
  recorded elapsed times. This includes the latency of the v2 tail rule for fixed-point windows.
  Release-time errors describe matched switches only. The main macro-F1 score uses retrospective
  assignment. The report also gives an exploratory release-time point score and prediction
  coverage for the selected configurations and the raw references.
- Paired 95% percentile intervals reuse 5,000 resamples of the same nine linked test-user groups.
  The small number of correlated groups limits precision. The selected settings can depend on
  the chosen 120-second tolerance and 0.02 validation margin.

## Reproduction

From the project root with the Part 2 environment:

```powershell
& .\.venv-audit\Scripts\python.exe experiments/sampling_window_smoothing/run.py --jobs 5
& .\.venv-audit\Scripts\python.exe experiments/sampling_window_smoothing/report.py
& .\.venv-audit\Scripts\python.exe experiments/sampling_window_smoothing/test_run.py
```

Use `--output` for a fresh run or `--resume` with the same inputs and experiment configuration to
reuse complete window caches, CV blocks, and fitted models. The run saves fold assignments,
a feature and input manifest, fitted models,
and aligned scores. It checks native 50-point windows and both headline v2 F1 values.

Reference. `docs/part2_handover.md` and `docs/part2_notes_for_teammates.md`.

## Compare with separate sampling experiments

Match the user split, the rule for retaining fixes, the model's fitting rate, the window
definition, and the scoring timeline before comparing numbers. The 15 s and 30 s settings
are minimum elapsed-time rules, so the retained intervals can exceed those values.
`results/windows/*_sampling.csv` documents retained point counts and per-user interval
summaries. GPS point retention measures data reduction. Battery or energy use was not measured.

Research workflow: Kassis et al. (2026), *Scientific Agent Skills: A Library of Procedural
Knowledge for Research Agents*, arXiv:2609.00065, DOI 10.48550/arXiv.2609.00065.
Bibliographic details are from the local skill documentation. The official record was unavailable
when checked.
