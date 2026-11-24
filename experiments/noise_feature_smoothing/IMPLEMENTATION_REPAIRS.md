# Implementation repairs and validation

The fixed scientific protocol was not changed.

- Replaced the failed quoted absolute-path Windows launch with a Python subprocess
  argument list and an explicit repository working directory.
- Counted the earlier failed launch toward the seven-hour budget. A supervisor
  checks the wall clock every second and saves state before stopping its worker.
- Each stage has one initial attempt plus at most three repair retries. A failed
  stage pauses for a recorded implementation repair; no blind repeated training.
- Rebuilt sigma=0 features from coordinates instead of copying the reference table.
- Replaced nearest-first greedy transition matching with maximum-cardinality
  bipartite matching, as required by the fixed boundary metric.
- Removed prediction on an empty evaluation matrix while fitting G1/G2. Models
  still use pipeline_v2.model_specs and the exact fixed fitting parameters.
- Reused the saved stage-0 G3 pipeline; saved feature and score class order.
- Preserved per-class point counts in prediction files for point-level evaluation.
- Fixed bootstrap Macro-F1 to average all five fixed classes, including absent
  classes with F1=0. Reused one 5000x9 draw matrix for all conditions.
- Calculated F1 separately per seed before averaging bootstrap results. Added
  paired noisy-minus-clean changes and G3/G2 change contrasts.
- Calculated noise-induced label changes against each model/smoother's own clean
  predictions, weighted by original window point counts.
- Kept wrong isolated flips separate from all isolated flips.
- Added per-user results, per-class F1, genuine timestamp example plots and
  explanatory example-selection records. G4 remains skipped.

Primary API references consulted (installed official API docstrings were read;
web lookup returned no page content):

- SciPy `scipy.sparse.csgraph.maximum_bipartite_matching` API:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.maximum_bipartite_matching.html
- scikit-learn `Pipeline` API:
  https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
- NumPy `Generator.multinomial` API:
  https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.multinomial.html

No new scientific method or external-paper parameter setting was introduced.
The original plan's interpretation limits continue to apply.
