# Figure captions (Part 2 v2)

**fig1_v2_model_comparison.png** — Test point macro-F1 of every model after its own smoothing step, v1 (orange) and v2 (blue), with 95% bootstrap CIs over the 9 independent test groups. Models are ordered by v2 CV macro-F1 on blind-cut training windows (5 user-grouped folds); the shaded row is the selected model. v2 fits all models on the same 20,000-window subsample except the Nystroem SVM (all windows). The Nystroem SVM has no v1 counterpart.

**fig2_v2_per_class.png** — Per-class point F1 of the selected v2 model (SVM (RBF, Nystroem, all data)) on the 13 test users: raw window predictions and the final output after its smoothing step (k = 11).

**fig3_v2_confusion.png** — Point-level confusion matrix of the selected v2 model (SVM (RBF, Nystroem, all data), final output) on the 13 test users; each cell shows the share of the true class and the point count.

**fig4_v2_example_trip.png** — The same fixed rule as v1: among the 332 test trips with at least one true mode change between consecutive windows, the trip with the median number of windows (126_000094). Rows: true window mode, raw prediction and final output of SVM (RBF, Nystroem, all data).

**fig5_v2_k_tradeoff.png** — Training-user CV only (blind windows, 5 folds) for the selected model (SVM (RBF, Nystroem, all data), soft voting). Left: mean fold point macro-F1 for each smoothing window k. Right: precision and recall of mode changes within ±1 window. The dashed line is the k chosen by the paired one-SE rule; larger k raises point F1 a little but misses more real mode changes.
