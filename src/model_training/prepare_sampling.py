"""Prepare real GPS downsampling conditions and point/time windows; never train on mock data.

Example (from repository root):
  python src/model_training/prepare_sampling.py --points data/processed/quality_features

0 seconds preserves the original sampling; positive intervals use timestamp-based greedy
decimation inside each original block. All motion features are recomputed afterwards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from feature_enginneering.motion_features import build_features
from build_windows import MODES, assign_windows, load_split, read_points, window_features

KEYS = ["user_id", "trip_id", "feature_block_id"]
BASE = KEYS + ["timestamp", "latitude", "longitude", "mode"]


def downsample(points: pd.DataFrame, seconds: float) -> pd.DataFrame:
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError("Sampling interval must be finite and nonnegative")
    d = points[BASE].sort_values(KEYS + ["timestamp"], kind="stable").reset_index(drop=True)
    if d[BASE].isna().any().any() or not d["mode"].isin(MODES).all():
        raise ValueError("Expected nonmissing, five-class quality_features input")
    delta = d.groupby(KEYS, sort=False).timestamp.diff().dt.total_seconds()
    if (delta.dropna() <= 0).any():
        raise ValueError("Timestamps must increase strictly inside each block")
    keep = []
    timestamps = d.timestamp.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    threshold_ns = seconds * 1_000_000_000
    for positions in d.groupby(KEYS, sort=False).indices.values():
        last = None
        for pos in positions:
            t = timestamps[pos]
            if last is None or t - last >= threshold_ns:
                keep.append(pos)
                last = t
    d = d.loc[keep].reset_index(drop=True)
    d["input_feature_block_id"] = d.feature_block_id
    # Decimation can turn a formerly valid gap into >20 minutes. Break it before
    # derivatives; never discard additional points or join original blocks.
    gaps = d.groupby(KEYS, sort=False).timestamp.diff().dt.total_seconds()
    extra = gaps.gt(1200).groupby([d[k] for k in KEYS], sort=False).cumsum()
    d["feature_block_id"] = d.feature_block_id.astype(str) + "_s" + extra.astype(int).astype(str)
    d["feature_block_start"] = d.groupby(KEYS, sort=False).cumcount().eq(0)
    return build_features(d)


def duration_windows(d: pd.DataFrame, seconds: float, pure: bool) -> pd.Series:
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("Window duration must be finite and positive")
    starts = d[KEYS].ne(d[KEYS].shift()).any(axis=1)
    if pure:
        starts |= d["mode"].ne(d["mode"].shift())
    run = starts.cumsum()
    elapsed = (d.timestamp - d.timestamp.groupby(run).transform("first")).dt.total_seconds()
    buckets = np.floor(elapsed / seconds).astype(np.int64)
    # Half-open time bins, anchored at each run's first retained observation.
    # Empty bins are omitted, partial tails retained, and windows never cross blocks.
    return pd.Series(pd.factorize(pd.MultiIndex.from_arrays([run, buckets]))[0], index=d.index)


def make_table(d, split, window, duration, pure):
    d = d.copy()
    d["window_id"] = (duration_windows(d, duration, pure and split == "train") if duration is not None
                      else assign_windows(d, window, pure and split == "train"))
    w = window_features(d)
    w.insert(1, "split", split)
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--points", type=Path, default=PROJECT / "data/processed/quality_features")
    ap.add_argument("--splits", type=Path, default=PROJECT / "data/splits")
    ap.add_argument("--intervals", type=float, nargs="+", default=[0, 5, 10, 30])
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--duration", type=float, default=None, help="seconds; omit for the main fixed-point experiment")
    ap.add_argument("--out", type=Path, default=PROJECT / "outputs/sampling")
    a = ap.parse_args()
    if a.window <= 0 or any(not np.isfinite(s) or s < 0 for s in a.intervals):
        ap.error("Use a positive window size and finite, nonnegative sampling intervals")
    if a.duration is not None and (not np.isfinite(a.duration) or a.duration <= 0):
        ap.error("Duration must be finite and positive")
    split_of = load_split(a.splits)
    files = sorted(a.points.glob("*.parquet"))
    files = [p for p in files if p.stem in split_of]
    missing = set(split_of) - {p.stem for p in files}
    if missing:
        ap.error("Missing real per-user Parquet inputs: " + ", ".join(sorted(missing)))
    intervals = sorted(set(a.intervals))
    directories = {s: a.out / (f"interval_{s:g}s_" + (f"time_{a.duration:g}s" if a.duration else f"points_{a.window}"))
                   for s in intervals}
    if any(p.exists() for p in directories.values()):
        ap.error("Condition directory already exists; use a new --out to preserve previous runs")
    for p in directories.values():
        p.mkdir(parents=True)
    manifest = [{"file": str(p.resolve()), "bytes": p.stat().st_size,
                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
    for interval, output in directories.items():
        tables, blind_tables, stats, memberships = [], [], [], []
        for p in files:
            before = read_points(p)
            sampled = downsample(before, interval)
            split = split_of[p.stem]
            table = make_table(sampled, split, a.window, a.duration, True)
            blind = make_table(sampled, split, a.window, a.duration, False)
            tables.append(table)
            blind_tables.append(blind)
            if split == "test":
                membership = sampled[["user_id", "timestamp", "mode", "input_feature_block_id"]].copy()
                membership["window_id"] = (duration_windows(sampled, a.duration, False) if a.duration is not None
                                           else assign_windows(sampled, a.window, False))
                memberships.append(membership)
            stats.append({"user_id": p.stem, "split": split, "input_points": len(before),
                          "retained_points": len(sampled), "windows": len(table), "blind_windows": len(blind),
                          "median_retained_delta_s": sampled.delta_t_s.median()})
            print(f"interval={interval:g}s {p.stem} {split}: {len(before):,} -> {len(sampled):,} points", flush=True)
        table, blind = pd.concat(tables, ignore_index=True), pd.concat(blind_tables, ignore_index=True)
        table.to_csv(output / "windows.csv", index=False)
        blind.to_csv(output / "windows_blind.csv", index=False)
        test_windows = table[table.split == "test"].reset_index(drop=True)
        test_windows["window_id"] = test_windows.groupby("user_id", sort=False).cumcount()
        lookup = pd.Series(test_windows.index.to_numpy(),
                           index=pd.MultiIndex.from_frame(test_windows[["user_id", "window_id"]]))
        membership = pd.concat(memberships, ignore_index=True)
        membership["test_window_index"] = lookup.reindex(
            pd.MultiIndex.from_frame(membership[["user_id", "window_id"]])).to_numpy()
        if membership.test_window_index.isna().any() or len(membership) != test_windows.n_points.sum():
            raise ValueError("Test point-to-window mapping is incomplete")
        membership.to_csv(output / "test_point_membership.csv", index=False)
        pd.DataFrame(stats).to_csv(output / "sampling_by_user.csv", index=False)
        durations = blind.groupby("split").duration_s.describe().reset_index()
        durations.to_csv(output / "blind_window_durations.csv", index=False)
        config = {"interval_seconds": interval, "window_points": a.window if a.duration is None else None,
                  "window_seconds": a.duration, "input_manifest": manifest,
                  "split_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(a.splits.glob("*.json"))},
                  "evaluation_support": "retained GPS observations only; differs across intervals",
                  "rules": "greedy time decimation within original blocks; recompute derivatives; no label-based sampling; blind validation/test windows",
                  "data_points": int(table.n_points.sum()), "blind_points": int(blind.n_points.sum())}
        (output / "condition.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(f"[prepared] {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
