"""Sampling must change geometry-derived features, preserve blocks, and ignore labels."""
import sys
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/model_training"))
from prepare_sampling import downsample, duration_windows, make_table


def points(times=(0, 2, 4, 6, 8, 10)):
    n = len(times)
    return pd.DataFrame({"user_id": ["001"] * n, "trip_id": ["001_1"] * n,
                         "feature_block_id": ["001_q1"] * n,
                         "timestamp": pd.Timestamp("2020-01-01") + pd.to_timedelta(times, unit="s"),
                         "latitude": [30.] * n, "longitude": 120 + np.arange(n) * 0.0001,
                         "mode": ["walk"] * n, "speed_mps": [999.] * n})


class TestSampling(unittest.TestCase):
    def test_sampling_is_time_based_and_features_are_recomputed(self):
        result = downsample(points(), 5)
        self.assertEqual(result.timestamp.dt.second.tolist(), [0, 6])
        self.assertEqual(result.delta_t_s.dropna().tolist(), [6.])
        self.assertTrue(result.speed_mps.iloc[1] < 100)
        self.assertTrue(pd.isna(result.speed_mps.iloc[0]))

    def test_sampling_and_blind_windows_ignore_labels(self):
        original = points()
        changed = original.copy()
        changed["mode"] = ["car", "bus", "car", "train", "bike", "walk"]
        a, b = downsample(original, 3), downsample(changed, 3)
        self.assertEqual(a.timestamp.tolist(), b.timestamp.tolist())
        self.assertEqual(duration_windows(a, 5, False).tolist(), duration_windows(b, 5, False).tolist())

    def test_block_boundaries_reset_motion_history(self):
        d = points()
        d.loc[3:, "feature_block_id"] = "001_q2"
        result = downsample(d, 5)
        self.assertEqual(len(result), 2)
        self.assertTrue(result.speed_mps.isna().all())

    def test_new_long_gap_is_split_before_derivatives(self):
        result = downsample(points((0, 800, 1600)), 1000)
        self.assertEqual(result.feature_block_id.nunique(), 2)
        self.assertTrue(result.delta_t_s.isna().all())

    def test_duration_boundary_and_point_conservation(self):
        d = downsample(points(), 0)
        self.assertEqual(duration_windows(d, 4, False).tolist(), [0, 0, 1, 1, 2, 2])
        d.loc[2:, "mode"] = "bus"
        for duration in (None, 4):
            for pure in (True, False):
                w = make_table(d, "train", 4, duration, pure)
                self.assertEqual(w.n_points.sum(), len(d))
                self.assertEqual(w.filter(like="cnt_").sum().sum(), len(d))

    def test_invalid_interval_and_nonmonotonic_data_fail(self):
        for interval in (-1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                downsample(points(), interval)
        with self.assertRaises(ValueError):
            downsample(points((0, 0)), 5)

    def test_cli_writes_both_window_tables_and_manifest(self):
        project = Path(__file__).resolve().parents[1]
        scratch = project / "outputs/test_tmp"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            root = Path(temp)
            source, splits = root / "points", root / "splits"
            source.mkdir()
            splits.mkdir()
            for uid, split in [("001", "train"), ("002", "test")]:
                d = points()
                d["user_id"], d["trip_id"], d["feature_block_id"] = uid, uid + "_1", uid + "_q1"
                d.to_parquet(source / f"{uid}.parquet", index=False)
                (splits / f"{split}_users.json").write_text(json.dumps([uid]))
            for duration in ([], ["--duration", "4"]):
                output = root / ("time" if duration else "count")
                command = [sys.executable, str(project / "src/model_training/prepare_sampling.py"),
                           "--points", str(source), "--splits", str(splits), "--intervals", "0", "5",
                           "--out", str(output), *duration]
                run = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                for directory in output.iterdir():
                    table = pd.read_csv(directory / "windows.csv")
                    blind = pd.read_csv(directory / "windows_blind.csv")
                    config = json.loads((directory / "condition.json").read_text())
                    expected = 12 if config["interval_seconds"] == 0 else 4
                    self.assertEqual(int(table.n_points.sum()), expected)
                    self.assertEqual(int(blind.n_points.sum()), expected)
                    self.assertEqual(len(config["input_manifest"]), 2)


if __name__ == "__main__":
    unittest.main()
