"""
Temporal smoothing of window predictions, the final step of the Part 2 model.

Windows inside one feature block are consecutive in time, so a single window whose prediction
differs from its neighbours is usually wrong. smooth_majority() replaces each window's prediction
by the majority over k consecutive windows centred on it, inside the same block only.

    from smoothing import smooth_majority, count_changes
    smoothed = smooth_majority(pred, blocks=df["feature_block_id"], times=df["start_time"], k=5)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models import MODES


def _block_order(blocks, times) -> np.ndarray:
    """Positions that sort the windows by block, then by start time."""
    key = pd.DataFrame({"block": np.asarray(blocks, dtype=object), "time": np.asarray(times)})
    return key.sort_values(["block", "time"], kind="stable").index.to_numpy()


def smooth_majority(pred, blocks, times, k: int, labels=MODES) -> np.ndarray:
    """Sliding majority vote over k consecutive windows (k odd) within each block.

    * never crosses blocks: near a block edge the vote uses only the windows inside the block
    * ties keep the window's original prediction
    * k <= 1 returns the predictions unchanged
    Returns an array in the same order as `pred`.
    """
    pred = np.asarray(pred, dtype=object)
    if k <= 1:
        return pred.copy()
    if k % 2 == 0:
        raise ValueError("k must be odd so the vote is centred on the window")
    order = _block_order(blocks, times)
    p, b = pred[order], np.asarray(blocks, dtype=object)[order]
    codes = pd.Categorical(p, categories=labels).codes
    if (codes < 0).any():
        raise ValueError(f"predictions outside {labels}: {set(p[codes < 0])}")

    n, h = len(p), k // 2
    idx = np.arange(n)
    onehot = np.zeros((n + 1, len(labels)))
    onehot[idx + 1, codes] = 1
    cum = onehot.cumsum(axis=0)
    new_block = np.r_[True, b[1:] != b[:-1]]
    block_no = np.cumsum(new_block) - 1
    starts = np.flatnonzero(new_block)
    ends = np.r_[starts[1:] - 1, n - 1]
    lo = np.maximum(idx - h, starts[block_no])
    hi = np.minimum(idx + h, ends[block_no])
    counts = cum[hi + 1] - cum[lo]
    top = counts.max(axis=1)
    tie = (counts == top[:, None]).sum(axis=1) > 1
    voted = np.where(tie, codes, counts.argmax(axis=1))

    out = np.empty(n, dtype=object)
    out[order] = np.asarray(labels, dtype=object)[voted]
    return out


def count_changes(seq, blocks, times) -> int:
    """Number of times the label changes between consecutive windows of the same block."""
    order = _block_order(blocks, times)
    s, b = np.asarray(seq, dtype=object)[order], np.asarray(blocks, dtype=object)[order]
    return int(((s[1:] != s[:-1]) & (b[1:] == b[:-1])).sum())


def isolated_flips(pred, true, blocks, times) -> tuple[int, int]:
    """One-window flips: a window whose prediction differs from both neighbours in its block while
    the two neighbours agree. Returns (number of flips, number of those that are wrong)."""
    order = _block_order(blocks, times)
    p = np.asarray(pred, dtype=object)[order]
    t = np.asarray(true, dtype=object)[order]
    b = np.asarray(blocks, dtype=object)[order]
    same = (b[1:-1] == b[:-2]) & (b[1:-1] == b[2:])
    flip = same & (p[1:-1] != p[:-2]) & (p[:-2] == p[2:])
    return int(flip.sum()), int((flip & (p[1:-1] != t[1:-1])).sum())
