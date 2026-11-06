"""
MOCK per-point data in the same layout as Part 1's quality_features (one file per user, CSV
instead of Parquet) plus mock split lists - only for testing the pipeline without the real data.
Numbers are synthetic: never report results from it.

    python make_mock_features.py --out mock_data
    python build_windows.py --points mock_data/quality_features --splits mock_data/splits
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# mode: (mean speed m/s, speed noise, heading noise rad, stop prob)
MODES = {"walk": (1.3, 0.5, 0.6, 0.15), "bike": (4.0, 1.2, 0.3, 0.06),
         "bus": (6.0, 4.0, 0.15, 0.25), "car": (10.0, 5.0, 0.12, 0.12),
         "train": (20.0, 6.0, 0.03, 0.05)}


def user_points(uid, rng, n_trips):
    rows, block = [], 0
    t = pd.Timestamp("2008-06-01") + pd.Timedelta(days=int(rng.integers(0, 300)))
    lat, lon = 39.9 + rng.normal(0, .05), 116.4 + rng.normal(0, .05)
    for trip in range(n_trips):
        t += pd.Timedelta(hours=int(rng.integers(2, 30)))
        for _ in range(rng.integers(1, 3)):                # blocks per trip
            block += 1
            seq = rng.choice(list(MODES), size=rng.integers(1, 4))
            heading = rng.uniform(0, 2 * np.pi)
            for m in seq:                                  # mode segments inside a block
                mu, sd, hn, stop = MODES[m]
                for _ in range(rng.integers(30, 400)):
                    dt = float(rng.choice([1, 2, 5], p=[.5, .3, .2]))
                    v = 0.0 if rng.random() < stop else abs(rng.normal(mu, sd))
                    heading += rng.normal(0, hn)
                    lat += v * dt * np.cos(heading) / 111_000
                    lon += v * dt * np.sin(heading) / (111_000 * np.cos(np.radians(lat)))
                    t += pd.Timedelta(seconds=dt)
                    rows.append((uid, f"{uid}_{trip:06d}", f"{uid}_q{block:06d}", t, lat, lon, m, dt, v))
    df = pd.DataFrame(rows, columns=["user_id", "trip_id", "feature_block_id", "timestamp",
                                     "latitude", "longitude", "mode", "delta_t_s", "speed_mps"])
    g = df.groupby("feature_block_id")
    first = g.cumcount() == 0
    df.loc[first, ["delta_t_s", "speed_mps"]] = np.nan
    df["distance_m"] = df.speed_mps * df.delta_t_s
    df["acceleration_mps2"] = g["speed_mps"].diff() / df.delta_t_s
    df["jerk_mps3"] = df.groupby("feature_block_id")["acceleration_mps2"].diff() / df.delta_t_s
    brg = np.arctan2(g["longitude"].diff(), g["latitude"].diff())
    d = brg.groupby(df.feature_block_id).diff()
    df["heading_change_rad"] = np.abs(np.arctan2(np.sin(d), np.cos(d)))
    df["heading_change_rate_rad_s"] = df.heading_change_rad / df.delta_t_s
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mock_data")
    ap.add_argument("--users", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    (out / "quality_features").mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(parents=True, exist_ok=True)
    users = [f"{i:03d}" for i in range(a.users)]
    for u in users:
        user_points(u, rng, int(rng.integers(3, 12))).to_csv(out / "quality_features" / f"{u}.csv", index=False)
    test = users[::5]
    (out / "splits" / "test_users.json").write_text(json.dumps(test))
    (out / "splits" / "train_users.json").write_text(json.dumps([u for u in users if u not in test]))
    (out / "splits" / "report.json").write_text(json.dumps({"duplicate_linked_groups": [users[1:3]]}))
    print(f"wrote mock data for {len(users)} users to {out}")


if __name__ == "__main__":
    main()
