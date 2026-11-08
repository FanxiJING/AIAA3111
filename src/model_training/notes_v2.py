"""
Report-only analyses behind docs/part2_notes_for_teammates.md (nothing here feeds a model choice).

    python notes_v2.py      # after pipeline_v2.py --stage final

Outputs in results/basic_w50_no_dt_v2/notes/ :
    accel_jerk_shift.csv        train vs test medians and KS statistic of acceleration/jerk features, by class
    accel_jerk_by_dt.csv        the same medians within sampling-interval bands (is the shift a dt effect?)
    window_duration.csv         window duration in seconds by split and class
    car_errors_top20.csv        the 20 largest test segments whose true mode is car but predicted train/bus
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from models import MODES, load_windows

HERE = Path(__file__).resolve().parent
V2 = HERE / "results" / "basic_w50_no_dt_v2"
OUT = V2 / "notes"
FEATS = ["acc_mean", "acc_std", "acc_p90", "acc_max", "jerk_mean", "jerk_p90", "big_acc_rate"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # blind windows on both sides, so the comparison is like-for-like
    w = load_windows(HERE / "windows_w50_blind.csv")
    rows = []
    for m in MODES:
        a, b = w[(w.split == "train") & (w.label == m)], w[(w.split == "test") & (w.label == m)]
        for f in FEATS + ["dt_mean", "speed_p50"]:
            x, y = a[f].dropna(), b[f].dropna()
            rows.append({"class": m, "feature": f, "train_median": x.median(), "test_median": y.median(),
                         "ratio_test_to_train": y.median() / x.median() if x.median() else np.nan,
                         "ks_statistic": ks_2samp(x, y).statistic, "n_train": len(x), "n_test": len(y)})
    pd.DataFrame(rows).round(4).to_csv(OUT / "accel_jerk_shift.csv", index=False)

    w["dt_band"] = pd.cut(w.dt_mean, [0, 1.5, 2.5, 5.5, np.inf], labels=["<=1.5 s", "1.5-2.5 s", "2.5-5.5 s", ">5.5 s"])
    by_dt = w.groupby(["label", "dt_band", "split"], observed=True)[["acc_mean", "jerk_mean"]] \
        .agg(["median", "size"]).round(4)
    by_dt.columns = ["_".join(c) for c in by_dt.columns]
    by_dt.reset_index().to_csv(OUT / "accel_jerk_by_dt.csv", index=False)

    dur = w.groupby(["split", "label"]).duration_s.describe(percentiles=[0.1, 0.5, 0.9]).round(1)
    allw = w.groupby("split").duration_s.describe(percentiles=[0.1, 0.5, 0.9]).round(1).assign(label="all")
    pd.concat([dur.reset_index(), allw.reset_index()]).to_csv(OUT / "window_duration.csv", index=False)

    sel = json.loads((V2 / "selection.json").read_text())["selected"]
    seg = pd.read_csv(V2 / "segments_test.csv", dtype={"user_id": str})
    car = seg[(seg.majority_true_mode == "car") & seg.predicted_mode.isin(["train", "bus"])].copy()
    te = w[w.split == "test"]
    feat = []
    for _, s in car.iterrows():
        x = te[(te.feature_block_id == s.feature_block_id) & (te.start_time >= s.start_time) & (te.end_time <= s.end_time)]
        feat.append({"duration_min": (pd.Timestamp(s.end_time) - pd.Timestamp(s.start_time)).total_seconds() / 60,
                     "speed_p50_median": x.speed_p50.median(), "speed_p90_median": x.speed_p90.median(),
                     "stop_rate_mean": x.stop_rate.mean(), "dt_mean_median": x.dt_mean.median()})
    car = pd.concat([car.reset_index(drop=True), pd.DataFrame(feat)], axis=1)
    car = car.sort_values("n_points", ascending=False).head(20)
    car.insert(0, "model", sel)
    car.round(2).to_csv(OUT / "car_errors_top20.csv", index=False)
    print(f"[done] notes in {OUT}; {len(car)} car errors listed")


if __name__ == "__main__":
    main()
