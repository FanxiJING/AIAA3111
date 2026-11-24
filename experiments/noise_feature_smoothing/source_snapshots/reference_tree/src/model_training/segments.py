"""
Segment output: merge consecutive windows with the same predicted mode inside a feature block into
single-mode segments (teammate A's output format), and score them against the true segments.

Granularity is the window: true segments are runs of windows with the same majority true mode, so
a mode change inside a window is placed at the nearest window boundary.

    from segments import make_segments, segment_metrics
    segs = make_segments(windows_meta, smoothed_pred)
    m = segment_metrics(windows_meta, smoothed_pred, windows_meta["label"])
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models import MODES

SEGMENT_COLS = ["user_id", "trip_id", "feature_block_id", "start_time", "end_time", "n_points",
                "predicted_mode", "majority_true_mode", "n_windows"]


def _sorted(meta: pd.DataFrame, **cols) -> pd.DataFrame:
    d = meta.copy()
    for name, values in cols.items():
        d[name] = np.asarray(values, dtype=object)
    return d.sort_values(["feature_block_id", "start_time"], kind="stable").reset_index(drop=True)


def _runs(d: pd.DataFrame, col: str) -> pd.Series:
    """Run id per window: a new run starts at a new block or where `col` changes."""
    new = d.feature_block_id.ne(d.feature_block_id.shift()) | d[col].ne(d[col].shift())
    return new.cumsum()


def make_segments(meta: pd.DataFrame, pred, labels=MODES) -> pd.DataFrame:
    """meta needs user_id, trip_id, feature_block_id, start_time, end_time, n_points, cnt_<mode>.
    One row per predicted segment; majority_true_mode comes from the summed per-class point counts."""
    cnt = [f"cnt_{m}" for m in labels]
    d = _sorted(meta[["user_id", "trip_id", "feature_block_id", "start_time", "end_time",
                      "n_points", *cnt]], pred=pred)
    g = d.groupby(_runs(d, "pred"), sort=True)
    seg = g.agg(user_id=("user_id", "first"), trip_id=("trip_id", "first"),
                feature_block_id=("feature_block_id", "first"), start_time=("start_time", "min"),
                end_time=("end_time", "max"), n_points=("n_points", "sum"),
                predicted_mode=("pred", "first"), n_windows=("pred", "size"))
    seg["majority_true_mode"] = g[cnt].sum().idxmax(axis=1).str.removeprefix("cnt_")
    return seg[SEGMENT_COLS].reset_index(drop=True)


def segment_metrics(meta: pd.DataFrame, pred, true, iou: float = 0.5, tol: int = 1) -> dict:
    """Simple segment-level scores (meta needs feature_block_id, start_time, n_points):
    * segment counts, predicted and true
    * segment recall    = share of true segments matched by a predicted segment of the same mode
                          with point-overlap IoU >= iou; segment precision = the same for predicted
    * transition precision / recall: a predicted mode change counts as correct if a true change lies
      within +/- tol windows in the same block, and vice versa (tolerance match, not one-to-one)
    """
    d = _sorted(meta[["feature_block_id", "start_time", "n_points"]], pred=pred, true=true)
    d["pseg"], d["tseg"] = _runs(d, "pred"), _runs(d, "true")
    plen = d.groupby("pseg").n_points.sum()
    tlen = d.groupby("tseg").n_points.sum()
    ov = d[d.pred == d.true].groupby(["tseg", "pseg"]).n_points.sum().reset_index()
    ov["iou"] = ov.n_points / (tlen.loc[ov.tseg].to_numpy() + plen.loc[ov.pseg].to_numpy() - ov.n_points)
    hit = ov[ov.iou >= iou]

    same_block = d.feature_block_id.eq(d.feature_block_id.shift()).to_numpy()
    idx = np.arange(len(d))
    true_tr = idx[same_block & d.true.ne(d.true.shift()).to_numpy()]
    pred_tr = idx[same_block & d.pred.ne(d.pred.shift()).to_numpy()]

    def near(points, targets):
        # a transition at i sits between windows i-1 and i; i-1..i+1 are then in one block
        t = set(targets.tolist())
        return np.array([any(j in t for j in range(i - tol, i + tol + 1)) for i in points], dtype=bool)

    return {"true_segments": len(tlen), "pred_segments": len(plen),
            "segment_recall_iou50": hit.tseg.nunique() / len(tlen),
            "segment_precision_iou50": hit.pseg.nunique() / len(plen),
            "true_transitions": len(true_tr), "pred_transitions": len(pred_tr),
            "transition_precision": near(pred_tr, true_tr).mean() if len(pred_tr) else np.nan,
            "transition_recall": near(true_tr, pred_tr).mean() if len(true_tr) else np.nan}
