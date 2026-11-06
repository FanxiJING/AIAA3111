"""
Part 2, step 1 - turn Part 1's per-point data into fixed-length windows with summary features.

Why windows: the model must work on an unsegmented point stream (see teammate A's note
"关于输入输出的一点说明"). We cut every feature_block into consecutive windows of W points,
summarise the point features of each window, classify the window, and give the window's
prediction to every point inside it -> point-level labels without knowing segment boundaries.

Rules
-----
* windows never cross (user_id, feature_block_id)   [teammate A's requirement]
* test users : windows are cut blind - mode labels are NOT used to place boundaries
* train users: additionally cut where the true mode changes (labels are known there), so the
               training windows are pure. Disable with --no-pure-train.
* a tail shorter than W/2 is merged into the previous window, so every point is in a window
* window label = majority mode; per-class point counts are kept for point-level evaluation

Run (needs pyarrow - teammate A's uv env has it):
    uv run python build_windows.py                                  # defaults below
    uv run python build_windows.py --points ../../data/processed/quality_features --splits ../../data/splits --window 50

Output: windows_w<W>.csv (one row per window, ~70k rows for W=50) + windows_w<W>_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]                  # project root (Desktop folder or AIAA3111 repo)


def _first_existing(*cands):
    for c in cands:
        if c.exists():
            return c
    return cands[0]


# works in both layouts: Desktop folder (quality_features/, splits/) and the AIAA3111 repo
# (data/processed/quality_features/, data/splits/)
DEFAULT_POINTS = _first_existing(PROJECT / "data" / "processed" / "quality_features", PROJECT / "quality_features")
DEFAULT_SPLITS = _first_existing(PROJECT / "data" / "splits", PROJECT / "splits")
MODES = ["walk", "bike", "bus", "car", "train"]

POINT_COLS = ["user_id", "trip_id", "feature_block_id", "timestamp", "latitude", "longitude",
              "mode", "delta_t_s", "distance_m", "speed_mps", "acceleration_mps2", "jerk_mps3",
              "heading_change_rad", "heading_change_rate_rad_s"]

STOP_SPEED = 0.5            # m/s  - "stopped"
SLOW_SPEED = 2.0            # m/s  - walking-ish
SHARP_TURN = np.pi / 6      # 30 degrees
BIG_ACC = 1.0               # m/s^2


def read_points(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        have = set(pq.read_schema(path).names)
        # ignore_metadata: Part 1's files restore Arrow-backed strings under pandas 3, where
        # "mode != mode.shift()" gives NA on the first row instead of True and breaks cumsum
        df = pq.read_table(path, columns=[c for c in POINT_COLS if c in have]).to_pandas(ignore_metadata=True)
    else:
        df = pd.read_csv(path, dtype={"user_id": str})
    df["user_id"] = df["user_id"].astype(str).str.zfill(3)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def assign_windows(df: pd.DataFrame, W: int, cut_at_label_change: bool) -> pd.Series:
    """Return an integer window id per row (rows must be sorted by block, time)."""
    run_key = df["feature_block_id"].astype(str)
    if cut_at_label_change:
        new_run = (df["mode"] != df["mode"].shift()) | (run_key != run_key.shift())
        run = new_run.cumsum()
    else:
        run = (run_key != run_key.shift()).cumsum()
    pos = df.groupby(run).cumcount().to_numpy()
    n = run.map(run.value_counts()).to_numpy()
    chunk = pos // W
    last = (n - 1) // W
    tail = n - last * W
    merge = (chunk == last) & (last > 0) & (tail < W / 2)
    chunk = np.where(merge, last - 1, chunk)
    key = run.to_numpy().astype(np.int64) * 1_000_000 + chunk
    return pd.Series(pd.factorize(key)[0], index=df.index)


def _q(s, q):
    return s.quantile(q)


def window_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(
        abs_acc=df["acceleration_mps2"].abs(),
        abs_jerk=df["jerk_mps3"].abs(),
        # float, so missing values stay NaN and are skipped by the group means
        is_stop=(df["speed_mps"] < STOP_SPEED).astype(float).where(df["speed_mps"].notna()),
        is_slow=(df["speed_mps"] < SLOW_SPEED).astype(float).where(df["speed_mps"].notna()),
        is_turn=(df["heading_change_rad"] > SHARP_TURN).astype(float).where(df["heading_change_rad"].notna()),
        is_bigacc=(df["acceleration_mps2"].abs() > BIG_ACC).astype(float).where(df["acceleration_mps2"].notna()),
    )
    g = df.groupby("window_id", sort=True)
    sp, acc, jk, hcr = g["speed_mps"], g["abs_acc"], g["abs_jerk"], g["heading_change_rate_rad_s"]

    f = pd.DataFrame({
        "speed_mean": sp.mean(), "speed_std": sp.std(), "speed_min": sp.min(), "speed_max": sp.max(),
        "speed_p10": sp.quantile(0.10), "speed_p50": sp.median(),
        "speed_p90": sp.quantile(0.90), "speed_p95": sp.quantile(0.95),
        "acc_mean": acc.mean(), "acc_std": g["acceleration_mps2"].std(),
        "acc_p90": acc.quantile(0.90), "acc_max": acc.max(),
        "jerk_mean": jk.mean(), "jerk_p90": jk.quantile(0.90),
        "hcr_mean": hcr.mean(), "hcr_p50": hcr.median(), "hcr_p90": hcr.quantile(0.90),
        "heading_change_mean": g["heading_change_rad"].mean(),
        "stop_rate": g["is_stop"].mean(), "slow_rate": g["is_slow"].mean(),
        "sharp_turn_rate": g["is_turn"].mean(), "big_acc_rate": g["is_bigacc"].mean(),
        "dt_mean": g["delta_t_s"].mean(), "dt_max": g["delta_t_s"].max(),
        "path_length_m": g["distance_m"].sum(min_count=1),
        "valid_speed_frac": sp.count() / g.size(),
    })
    t0, t1 = g["timestamp"].min(), g["timestamp"].max()
    f["duration_s"] = (t1 - t0).dt.total_seconds()
    # straight-line displacement / path length (1 = straight, ~0 = wandering)
    lat0, lat1 = np.radians(g["latitude"].first()), np.radians(g["latitude"].last())
    dlon = np.radians(g["longitude"].last() - g["longitude"].first())
    dlat = lat1 - lat0
    h = np.sin(dlat / 2) ** 2 + np.cos(lat0) * np.cos(lat1) * np.sin(dlon / 2) ** 2
    disp = 2 * 6_371_008.8 * np.arcsin(np.sqrt(h.clip(0, 1)))
    f["straightness"] = (disp / f["path_length_m"].replace(0, np.nan)).clip(upper=1.0)
    f["avg_speed_overall"] = f["path_length_m"] / f["duration_s"].replace(0, np.nan)

    # ---- meta columns (never used as features) ----
    counts = pd.crosstab(df["window_id"], df["mode"]).reindex(columns=MODES, fill_value=0)
    meta = pd.DataFrame({
        "user_id": g["user_id"].first(), "trip_id": g["trip_id"].first(),
        "feature_block_id": g["feature_block_id"].first(),
        "start_time": t0, "end_time": t1, "n_points": g.size(),
    })
    meta["label"] = counts.idxmax(axis=1)
    meta["purity"] = counts.max(axis=1) / meta["n_points"]
    counts.columns = [f"cnt_{m}" for m in MODES]
    return pd.concat([meta, counts, f], axis=1).reset_index(drop=True)


def load_split(splits: Path) -> dict[str, str]:
    train = json.loads((splits / "train_users.json").read_text())
    test = json.loads((splits / "test_users.json").read_text())
    return {**{str(u): "train" for u in train}, **{str(u): "test" for u in test}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", default=str(DEFAULT_POINTS),
                    help="folder with <user>.parquet (or .csv) from Part 1")
    ap.add_argument("--splits", default=str(DEFAULT_SPLITS),
                    help="folder with train_users.json / test_users.json from Part 1")
    ap.add_argument("--window", type=int, default=50, help="points per window")
    ap.add_argument("--no-pure-train", action="store_true",
                    help="do not cut training windows at label changes")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    split_of = load_split(Path(a.splits))
    files = sorted(p for p in Path(a.points).iterdir() if p.suffix in (".parquet", ".csv"))
    out = Path(a.out or HERE / f"windows_w{a.window}.csv")
    parts, report = [], {"window": a.window, "pure_train": not a.no_pure_train, "users": {}}

    for p in files:
        uid = p.stem
        if uid not in split_of:
            continue                                   # empty users are in neither list
        df = read_points(p)
        if df.empty:
            continue
        df = df[df["mode"].isin(MODES)]
        df = df.sort_values(["feature_block_id", "timestamp"], kind="stable").reset_index(drop=True)
        split = split_of[uid]
        df["window_id"] = assign_windows(df, a.window, cut_at_label_change=(split == "train" and not a.no_pure_train))
        w = window_features(df)
        w.insert(1, "split", split)
        parts.append(w)
        report["users"][uid] = {"split": split, "points": int(len(df)), "windows": int(len(w))}
        print(f"{uid} {split:<5} points={len(df):>8,} windows={len(w):>6,}")

    allw = pd.concat(parts, ignore_index=True)
    allw.to_csv(out, index=False)
    for s in ("train", "test"):
        sw = allw[allw.split == s]
        report[s] = {"windows": int(len(sw)), "points": int(sw.n_points.sum()),
                     "window_labels": sw.label.value_counts().to_dict(),
                     "point_labels": {m: int(sw[f"cnt_{m}"].sum()) for m in MODES},
                     "mean_purity": round(float(sw.purity.mean()), 4),
                     "pure_window_share": round(float((sw.purity == 1).mean()), 4)}
    Path(str(out).replace(".csv", "_report.json")).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ("train", "test")}, indent=2))
    print(f"[done] {len(allw):,} windows -> {out}")


if __name__ == "__main__":
    main()
