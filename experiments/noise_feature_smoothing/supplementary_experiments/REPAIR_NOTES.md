# Rendering recovery, 2026-11-23

The initial full run completed all model fitting, coordinate/feature generation,
prediction, and 288 metric rows. At 19:12:28 Asia/Shanghai its figure canvas
assertion failed on tick labels `0.9` and `0.1`. The original traceback is retained
in `error.log` and `run.log`.

The check iterated all tick label objects, including ticks outside the axis view
interval. Those objects are not drawn by the axis but can have nominal bounding
boxes outside the canvas. This is a plotting acceptance bug, not a numeric or
experimental failure.

`complete_delivery.py` imports the unchanged original runner and replaces only
the in-memory `save_figure` function. It checks labels on ticks inside each view
interval, plus axis labels, titles, figure text and legends. It changes no tick
positions, axis limits, data, estimates, parameters, seeds, or plotted series.
It resumes only report generation and final acceptance, then records the
specific excluded tick labels in `diagnostics/render_recovery.json`.

The original runner and protocol hashes, as well as all original input/output
hashes, remain checked. No experimental stage is rerun. The final checkpoint
and `acceptance.json` describe the resumed outcome; the original failure log
continues to document the first attempt. This is the first reporting repair.

Automatic layout/nonblank checks and independent numeric verification are
separate from manual visual review.
