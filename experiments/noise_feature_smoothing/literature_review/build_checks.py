"""Deterministic, descriptive checks of existing GeoLife noise-study outputs."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
MODES = ("walk", "bike", "bus", "car", "train")
SMOOTHERS = ("S0_raw_k1", "S1_centered_soft_k11", "S2_trailing_soft_k11")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")
    temp.replace(path)


def protected_files():
    files = []
    for folder in ("src", "tests", "data", "experiments"):
        files.extend(p for p in (ROOT / folder).rglob("*")
                     if p.is_file() and HERE not in p.parents
                     and "__pycache__" not in p.parts and p.suffix != ".pyc"
                     and "matplotlib_cache" not in p.parts)
    files.extend(ROOT / p for p in ("CLAUDE.md", "README.md", "pyproject.toml", "uv.lock"))
    return sorted(files)


def point_confusion(counts, predictions):
    counts = np.asarray(counts, dtype=np.int64)
    pred = pd.Index(MODES).get_indexer(predictions)
    if counts.ndim != 2 or counts.shape[1] != 5 or len(pred) != len(counts):
        raise ValueError("Unaligned counts and predictions")
    if (pred < 0).any() or (counts < 0).any():
        raise ValueError("Unknown classes or negative counts")
    return np.stack([counts[pred == j].sum(axis=0) for j in range(5)], axis=1)


def metrics(cm):
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    den = cm.sum(axis=0) + cm.sum(axis=1)
    f1 = np.divide(2 * tp, den, out=np.zeros(5), where=den > 0)
    return float(f1.mean()), float(tp.sum() / cm.sum())


def stationary_reference(sigma, dt, threshold):
    """Local-plane incremental-noise reference; not an observed moving-speed law."""
    dt = np.asarray(dt, dtype=float)
    if sigma < 0 or threshold < 0 or np.any(dt <= 0):
        raise ValueError("Need nonnegative sigma/threshold and positive dt")
    if sigma == 0:
        return np.zeros_like(dt), np.zeros_like(dt), np.zeros_like(dt)
    mean = sigma * np.sqrt(np.pi) / dt
    rms = 2 * sigma / dt
    exceed = np.exp(-(threshold * dt) ** 2 / (4 * sigma ** 2))
    return mean, rms, exceed


def constant_references(counts):
    support = np.asarray(counts).sum(axis=0).astype(np.int64)
    total = int(support.sum())
    rows = []
    for j, mode in enumerate(MODES):
        cm = np.zeros((5, 5), dtype=np.int64)
        cm[:, j] = support
        f1, accuracy = metrics(cm)
        p = support[j] / total
        np.testing.assert_allclose(f1, (2 * p / (1 + p)) / 5, atol=1e-15)
        rows.append({"constant_mode": mode, "points": total,
                     "true_class_points": int(support[j]),
                     "true_class_fraction": float(p), "macro_f1": f1, "accuracy": accuracy,
                     "predicted_class_fraction": 1.0,
                     "interpretation": "descriptive_reference_not_a_fitted_or_selected_model"})
    return rows


def main():
    started = time.perf_counter()
    HERE.mkdir(exist_ok=True)
    (HERE / "tables").mkdir(exist_ok=True)
    (HERE / "acceptance.json").unlink(missing_ok=True)
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected_files()}
    write_json(HERE / "protected_input_manifest.json", before)

    windows = pd.read_csv(EXP / "feature_cache/sigma00_seed0.csv", dtype={"user_id": str})
    counts = windows[[f"cnt_{m}" for m in MODES]].to_numpy(np.int64)
    assert len(windows) == 17380 and counts.sum() == 860972
    np.testing.assert_array_equal(counts.sum(axis=1), windows.n_points)
    constant = pd.DataFrame(constant_references(counts))
    constant.to_csv(HERE / "tables/constant_class_references.csv", index=False)

    primary = pd.read_csv(EXP / "metrics/all_metrics.csv")
    reconstructed, comparison = [], []
    invariant = ["user_id", "feature_block_id", "start_time", "end_time",
                 "n_points", "label", "purity", *[f"cnt_{m}" for m in MODES]]
    for path in sorted((EXP / "predictions").glob("*_smoothed.csv")):
        group, condition = path.stem.replace("_smoothed", "").split("_", 1)
        prediction = pd.read_csv(path, dtype={"user_id": str})
        pd.testing.assert_frame_equal(prediction[invariant], windows[invariant],
                                      check_dtype=False)
        for smoother in SMOOTHERS:
            cm = point_confusion(counts, prediction[smoother])
            f1, accuracy = metrics(cm)
            ref = primary[(primary["group"] == group) & (primary.condition == condition)
                          & (primary.smoother == smoother)]
            assert len(ref) == 1
            np.testing.assert_allclose([f1, accuracy],
                                       ref[["macro_f1", "accuracy"]].to_numpy()[0],
                                       rtol=0, atol=1e-12)
            pred_support = cm.sum(axis=0)
            j = int(np.argmax(pred_support))
            row = {"group": group, "condition": condition, "sigma": int(ref.sigma.iloc[0]),
                   "seed": int(ref.seed.iloc[0]), "smoother": smoother,
                   "macro_f1": f1, "accuracy": accuracy,
                   "most_predicted_class": MODES[j],
                   "most_predicted_point_fraction": float(pred_support[j] / counts.sum())}
            reconstructed.append(row)
            for c in constant.to_dict("records"):
                comparison.append({**row, "constant_mode": c["constant_mode"],
                                   "constant_macro_f1": c["macro_f1"],
                                   "constant_accuracy": c["accuracy"],
                                   "f1_minus_constant": f1 - c["macro_f1"],
                                   "accuracy_minus_constant": accuracy - c["accuracy"]})
    reconstructed = pd.DataFrame(reconstructed)
    assert len(reconstructed) == 144
    reconstructed.to_csv(HERE / "tables/reconstructed_metrics_and_concentration.csv",
                         index=False)
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(HERE / "tables/constant_reference_contrasts.csv", index=False)
    summary = comparison.groupby(["group", "sigma", "smoother", "constant_mode"]).agg(
        seed_count=("seed", "size"),
        macro_f1_seed_mean=("macro_f1", "mean"),
        macro_f1_seed_sd=("macro_f1", "std"),
        accuracy_seed_mean=("accuracy", "mean"),
        f1_minus_constant_mean=("f1_minus_constant", "mean"),
        accuracy_minus_constant_mean=("accuracy_minus_constant", "mean"),
        most_predicted_point_fraction_mean=("most_predicted_point_fraction", "mean")
    ).reset_index()
    summary["macro_f1_seed_sd"] = summary.macro_f1_seed_sd.fillna(0)
    summary.to_csv(HERE / "tables/constant_reference_summary.csv", index=False)

    intervals, user_rows = [], []
    for uid in sorted(windows.user_id.unique()):
        columns = ["user_id", "trip_id", "feature_block_id", "timestamp", "delta_t_s",
                   "speed_mps"]
        points = pd.read_parquet(ROOT / f"data/processed/quality_features/{uid}.parquet",
                                 columns=columns)
        points = points.sort_values(["feature_block_id", "timestamp"], kind="stable")
        dt = points.groupby(["user_id", "trip_id", "feature_block_id"],
                            sort=False).timestamp.diff().dt.total_seconds()
        np.testing.assert_allclose(dt, points.delta_t_s, equal_nan=True, atol=0, rtol=0)
        valid = dt.dropna().to_numpy(float)
        assert np.all((valid > 0) & (valid <= 1200))
        intervals.append(valid)
        user_rows.append({"user_id": uid, "points": len(points), "valid_intervals": len(valid),
                          "block_starts_without_interval": int(dt.isna().sum()),
                          "dt_median_s": float(np.median(valid)),
                          "dt_p90_s": float(np.quantile(valid, 0.9))})
    assert sum(row["points"] for row in user_rows) == 860972
    pd.DataFrame(user_rows).to_csv(HERE / "tables/intervals_by_user.csv", index=False)
    dt = np.concatenate(intervals)
    strata = [("all", np.ones(len(dt), bool)), ("<=1s", dt <= 1),
              ("1-2s", (dt > 1) & (dt <= 2)), ("2-5s", (dt > 2) & (dt <= 5)),
              (">5s", dt > 5)]
    interval_rows = []
    for label, mask in strata:
        values = dt[mask]
        interval_rows.append({"interval_stratum": label, "intervals": int(mask.sum()),
                              "fraction_all_valid_intervals": float(mask.mean()),
                              "dt_median_s": float(np.median(values)),
                              "dt_p10_s": float(np.quantile(values, .1)),
                              "dt_p90_s": float(np.quantile(values, .9))})
    pd.DataFrame(interval_rows).to_csv(HERE / "tables/actual_interval_support.csv",
                                      index=False)
    assert sum(r["intervals"] for r in interval_rows[1:]) == len(dt)

    analytical = []
    for sigma, delta in itertools.product((0, 5, 10, 20), (1, 2, 5, 10, 30, 60)):
        mean, rms, _ = stationary_reference(sigma, delta, 0.5)
        row = {"sigma_axis_m": sigma, "horizontal_position_rms_m": np.sqrt(2) * sigma,
               "dt_s": delta, "stationary_mean_speed_mps": float(mean),
               "stationary_rms_speed_mps": float(rms)}
        for name, threshold in (("stop", 0.5), ("slow", 2.0), ("speed100", 100.0)):
            _, _, exceed = stationary_reference(sigma, delta, threshold)
            row[f"p_below_{name}_threshold"] = float(1 - exceed)
        analytical.append(row)
    pd.DataFrame(analytical).to_csv(HERE / "tables/analytical_stationary_reference.csv",
                                   index=False)
    weighted = []
    for sigma, (label, mask) in itertools.product((5, 10, 20), strata):
        values = dt[mask]
        mean, rms, _ = stationary_reference(sigma, values, .5)
        row = {"sigma_axis_m": sigma, "interval_stratum": label, "intervals": len(values),
               "mean_of_stationary_expected_speeds_mps": float(np.mean(mean)),
               "pooled_stationary_rms_speed_mps": float(np.sqrt(np.mean(rms ** 2))),
               "interpretation": "hypothetical_zero_displacement_at_observed_dt_not_observed_speed"}
        for name, threshold in (("stop", .5), ("slow", 2.), ("speed100", 100.)):
            _, _, exceed = stationary_reference(sigma, values, threshold)
            row[f"mean_p_below_{name}_threshold"] = float(np.mean(1 - exceed))
        weighted.append(row)
    pd.DataFrame(weighted).to_csv(HERE / "tables/dt_weighted_stationary_reference.csv",
                                 index=False)

    # Linear coordinate stencils illustrate amplification only; scalar speed
    # differences in motion_features.py are nonlinear and use irregular dt.
    stencil_rows = []
    for order in (1, 2, 3):
        from math import comb
        coef = np.array([(-1) ** (order - k) * comb(order, k) for k in range(order + 1)])
        stencil_rows.append({"difference_order": order, "coordinate_stencil": str(coef.tolist()),
                             "variance_multiplier": int(coef @ coef),
                             "dt_power_in_variance": 2 * order,
                             "applies_to_project_scalar_acc_or_jerk": False})
    pd.DataFrame(stencil_rows).to_csv(HERE / "tables/coordinate_difference_reference.csv",
                                     index=False)
    after = {str(p.relative_to(ROOT)): sha(p) for p in protected_files()}
    changed = [p for p in before.keys() | after.keys() if before.get(p) != after.get(p)]
    assert not changed, changed
    own_code = {p.name: sha(p) for p in HERE.glob("*.py")}
    write_json(HERE / "acceptance.json", {
        "passed": True, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": time.perf_counter() - started,
        "protected_files": len(before), "changed_protected_files": changed,
        "code_sha256": own_code, "metric_rows_reconstructed": len(reconstructed),
        "constant_reference_contrasts": len(comparison),
        "points": int(counts.sum()), "windows": len(windows),
        "valid_original_intervals": len(dt),
        "intervals_without_dt": 860972 - len(dt),
        "original_dt_reconstructed_exactly": True,
        "training_performed": False, "new_noise_generated": False,
        "new_random_numbers_generated": False, "protocol_changed": False,
        "analysis_status": "post_hoc_descriptive_not_model_selection",
        "bootstrap_recalculated": False,
    })
    print(json.dumps(json.loads((HERE / "acceptance.json").read_text(encoding="utf-8")),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
