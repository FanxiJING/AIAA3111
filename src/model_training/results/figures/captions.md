# Figure captions (Part 2)

**fig1_model_comparison.png** — Point-level macro-F1 on the 13 test users for every model, before (orange) and after (blue) temporal smoothing. Whiskers are 95% bootstrap CIs from resampling test users, with duplicate-linked users drawn together (9 groups). Models are ordered by CV macro-F1 on training users, the selection rule; the shaded row is the selected model. The CIs are wide and skewed because one linked group holds 78.9% of test points. k per model was chosen on training users only (paired one-SE rule).

**fig2_dt_ablation.png** — Ablation of the sampling-interval features dt_mean and dt_max, without smoothing. Removing them did not lower CV macro-F1 for any model; on test the selected SVM changed by -0.0002. These features shift between users (median car-window dt_mean 4.12 s in train vs 2.0 s in test), so they are excluded by default. Test values here are single estimates; this run predates the bootstrap.

**fig3_smoothing_per_class.png** — Per-class point F1 of the selected model (SVM (RBF)) on test users, without smoothing and with the sliding majority vote over k = 9 windows inside each feature block. The largest gains are for car and bus, the two classes most often confused with each other.

**fig4_confusion_final.png** — Point-level confusion matrix of the final model (SVM (RBF) with smoothing, k = 9) on the 13 test users. Each cell shows the share of the true class's points and the point count; rows sum to 1. Every point inherits its window's prediction.

**fig5_example_trip.png** — One test trip, chosen by a fixed rule before looking at its predictions: among the 332 test trips with at least one true mode change between consecutive windows, the trip with the median number of windows (trip 126_000094). Rows show the true window mode, the SVM (RBF) prediction without smoothing, and with smoothing (k = 9). Each bar is one 50-point window; gaps are periods with no retained GPS points.
