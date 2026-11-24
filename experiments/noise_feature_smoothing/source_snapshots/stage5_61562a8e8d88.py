"""Run the fixed GeoLife noise/feature/smoothing experiment.

This entry point is intentionally outside the reference Part 2 pipeline.  It imports
the reference feature and model implementations, writes only below the experiment
directory, and records every condition needed to reproduce the run.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import time
import traceback
import argparse
import importlib.metadata
import ctypes
from pathlib import Path
from types import SimpleNamespace

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "src" / "model_training"
sys.path.insert(0, str(MODEL_DIR))
sys.path.insert(0, str(ROOT))

from build_windows import assign_windows, read_points, window_features
from models import MODES, cv_groups, metrics_from_cm, point_confusion, prepare
from pipeline_v2 import fit_predict, model_specs, smooth
from segments import segment_metrics
from smoothing import count_changes, isolated_flips, smooth_soft
from src.feature_enginneering.motion_features import EARTH_RADIUS_M, build_features

OUT = ROOT / "experiments" / "noise_feature_smoothing"
CACHE = OUT / "feature_cache"
MODELS = OUT / "models"
PRED = OUT / "predictions"
METRICS = OUT / "metrics"
FIGURES = OUT / "figures"
CHECKPOINT = OUT / "checkpoint.json"
LOG = OUT / "run.log"
DEADLINE_HOURS = 7.0
SEEDS = [20261123, 20261124, 20261125, 20261126, 20261127]
SIGMAS = [0, 5, 10, 20]
GROUPS = ["G1", "G2", "G3"]
SMOOTHERS = ["S0_raw_k1", "S1_centered_soft_k11", "S2_trailing_soft_k11"]
FEATURE_GROUPS = {
    "G1": ["speed_mean", "speed_std", "speed_min", "speed_max", "speed_p10",
           "speed_p50", "speed_p90", "speed_p95", "stop_rate", "slow_rate"],
    "G2": ["speed_mean", "speed_std", "speed_min", "speed_max", "speed_p10",
           "speed_p50", "speed_p90", "speed_p95", "stop_rate", "slow_rate",
           "acc_mean", "acc_std", "acc_p90", "acc_max", "big_acc_rate"],
}
META_COLS = ["user_id", "trip_id", "feature_block_id", "start_time", "end_time",
             "n_points", "label", "purity", *[f"cnt_{m}" for m in MODES]]
SEED0 = 42
MODEL_NAME = "SVM (RBF, Nystroem, all data)"
LOADED_SOURCE = Path(__file__).read_bytes()

_START = time.monotonic()
_DEADLINE = _START + max(0, float(os.environ.get(
    "GEOLIFE_DEADLINE_EPOCH", time.time() + DEADLINE_HOURS * 3600)) - time.time())
_CHECK = {"stage": "starting", "completed": [], "attempts": {}, "started_at": time.time()}


def log(message: str):
    text = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(text, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def deadline_check():
    if time.monotonic() >= _DEADLINE:
        save_checkpoint("deadline", "global 7-hour deadline reached")
        raise TimeoutError("global 7-hour deadline reached")


def save_checkpoint(stage: str, status: str, **extra):
    _CHECK.update({"stage": stage, "status": status, "updated_at": time.time(),
                   "elapsed_seconds": time.monotonic() - _START,
                   "remaining_seconds": max(0.0, _DEADLINE - time.monotonic())})
    _CHECK.update(extra)
    supervisor_path = OUT / "supervisor.json"
    if supervisor_path.exists():
        state = json.loads(supervisor_path.read_text(encoding="utf-8"))
        _CHECK.update(completed=state.get("completed", []), attempts=state.get("retries", {}),
                      started_at=state["started_at_epoch"],
                      elapsed_seconds=time.time() - state["started_at_epoch"],
                      remaining_seconds=max(0, state["deadline_epoch"] - time.time()))
    json_write(CHECKPOINT, _CHECK)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def df_hash(df: pd.DataFrame) -> str:
    x = df.copy()
    for c in x.columns:
        if pd.api.types.is_datetime64_any_dtype(x[c]):
            x[c] = x[c].astype("int64")
    return hashlib.sha256(pd.util.hash_pandas_object(x, index=True).values.tobytes()).hexdigest()


def json_write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    temp.replace(path)


def process_memory_bytes():
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    *[(name, ctypes.c_size_t) for name in (
                        "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    assert psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    return int(counters.WorkingSetSize)


def protocol():
    return {
        "protocol_version": "noise-feature-smoothing-fixed-v1",
        "clean_training_noisy_test": True,
        "window_points": 50,
        "sigmas_m": SIGMAS,
        "seeds": SEEDS,
        "noise_axes": "independent east/north Gaussian standard deviation in metres",
        "horizontal_rms": "sigma * sqrt(2)",
        "feature_groups": FEATURE_GROUPS | {"G3": "all original v2 27 features"},
        "model": {"recipe": MODEL_NAME, "C": 1.0, "balanced": True, "seed": 42,
                  "nystroem_components": 1000, "gamma": "1 / feature dimension"},
        "smoothing": SMOOTHERS,
        "bootstrap": {"clusters": 9, "n_boot": 5000, "seed": 42},
        "g4": "skipped by protocol",
    }


def init():
    for p in (CACHE, MODELS, PRED, METRICS, FIGURES):
        p.mkdir(parents=True, exist_ok=True)
    json_write(OUT / "protocol.json", protocol())
    files = [
        MODEL_DIR / "build_windows.py", MODEL_DIR / "models.py",
        MODEL_DIR / "pipeline_v2.py", MODEL_DIR / "smoothing.py",
        MODEL_DIR / "segments.py",
        ROOT / "src" / "feature_enginneering" / "motion_features.py",
        MODEL_DIR / "windows_w50.csv",
        Path(__file__),
        ROOT / "data" / "splits" / "report.json",
        ROOT / "data" / "splits" / "train_users.json",
        ROOT / "data" / "splits" / "test_users.json",
    ]
    files += sorted((ROOT / "data" / "processed" / "quality_features").glob("*.parquet"))
    manifest = {
        "root": str(ROOT), "output": str(OUT), "python": sys.version,
        "platform": platform.platform(), "pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "files": {str(p.relative_to(ROOT)): sha256_file(p) for p in files if p.exists()},
        "protocol": protocol(),
        "dependencies": {p: importlib.metadata.version(p) for p in
                         ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "pyarrow")},
    }
    json_write(OUT / "manifest.json", manifest)


def load_tables():
    pure = pd.read_csv(MODEL_DIR / "windows_w50.csv", dtype={"user_id": str})
    pure.user_id = pure.user_id.str.zfill(3)
    test = pure[pure.split == "test"].reset_index(drop=True)
    train = pure[(pure.split == "train") & (pure.n_points >= 10)].reset_index(drop=True)
    for d in (test, train):
        d["start_time"] = pd.to_datetime(d["start_time"])
        d["end_time"] = pd.to_datetime(d["end_time"])
    assert len(test) == 17380 and test.n_points.sum() == 860972
    assert test.user_id.nunique() == 13
    assert cv_groups(test.user_id, ROOT / "data/splits/report.json").nunique() == 9
    assert not set(train.user_id) & set(test.user_id)
    assert test.groupby("feature_block_id").user_id.nunique().max() == 1
    return train, test


def condition_name(sigma: int, seed: int) -> str:
    return f"sigma{sigma:02d}_seed{seed if sigma else 0}"


def perturb(points: pd.DataFrame, sigma: float, seed: int) -> tuple[pd.DataFrame, dict]:
    result = points.copy()
    if sigma == 0:
        return result, {"east_std_m": 0.0, "north_std_m": 0.0, "radial_rms_m": 0.0}
    user = str(points.user_id.iloc[0])
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(user)]))
    east, north = (rng.standard_normal((len(points), 2)) * sigma).T
    lat = np.radians(points.latitude.to_numpy(float))
    lon = np.radians(points.longitude.to_numpy(float))
    angle = np.hypot(east, north) / EARTH_RADIUS_M
    bearing = np.arctan2(east, north)
    lat2 = np.arcsin(np.sin(lat) * np.cos(angle)
                     + np.cos(lat) * np.sin(angle) * np.cos(bearing))
    lon2 = lon + np.arctan2(np.sin(bearing) * np.sin(angle) * np.cos(lat),
                            np.cos(angle) - np.sin(lat) * np.sin(lat2))
    result["latitude"] = np.degrees(lat2)
    result["longitude"] = (np.degrees(lon2) + 180) % 360 - 180
    return result, {
        "east_std_m": float(np.std(east)), "north_std_m": float(np.std(north)),
        "radial_rms_m": float(np.sqrt(np.mean(east * east + north * north))),
    }


def build_test_condition(sigma: int, seed: int, baseline: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    parts, diagnostics = [], []
    point_total = 0
    high_speed = 0
    for uid in sorted(baseline.user_id.unique()):
        deadline_check()
        p = read_points(ROOT / "data" / "processed" / "quality_features" / f"{uid}.parquet")
        p = p.sort_values(["feature_block_id", "timestamp"], kind="stable").reset_index(drop=True)
        original_ids = assign_windows(p, 50, cut_at_label_change=False)
        noisy, noise_diag = perturb(p, sigma, seed)
        invariant = ["user_id", "trip_id", "feature_block_id", "timestamp", "mode"]
        pd.testing.assert_frame_equal(noisy[invariant], p[invariant])
        assert np.isfinite(noisy[["latitude", "longitude"]].to_numpy(float)).all()
        noisy = build_features(noisy)
        noisy["window_id"] = assign_windows(noisy, 50, cut_at_label_change=False)
        pd.testing.assert_series_equal(noisy.window_id, original_ids, check_names=False)
        assert len(noisy) == len(p)
        np.testing.assert_allclose(noisy.delta_t_s, p.delta_t_s, equal_nan=True)
        w = window_features(noisy)
        parts.append(w)
        point_total += len(noisy)
        high_speed += int(noisy.speed_mps.gt(100).sum())
        diagnostics.append({"user_id": uid, "points": len(noisy), **noise_diag,
                            "speed_gt100": int(noisy.speed_mps.gt(100).sum()),
                            "missing_speed": int(noisy.speed_mps.isna().sum())})
        log(f"stage2 {condition_name(sigma, seed)} user={uid} windows={len(w):,}")
    out = pd.concat(parts, ignore_index=True)
    out.user_id = out.user_id.astype(str).str.zfill(3)
    out["start_time"] = pd.to_datetime(out["start_time"])
    out["end_time"] = pd.to_datetime(out["end_time"])
    base_meta = baseline[META_COLS].copy()
    for c in ("start_time", "end_time"):
        base_meta[c] = pd.to_datetime(base_meta[c])
    pd.testing.assert_frame_equal(out[META_COLS].reset_index(drop=True),
                                  base_meta.reset_index(drop=True), check_dtype=False)
    manifest = {"sigma": sigma, "seed": seed, "users": 13, "points": point_total,
                "windows": len(out), "speed_gt100": high_speed,
                "user_diagnostics": diagnostics, "feature_hash": df_hash(out)}
    cols = list(prepare(baseline)[0].columns)
    manifest.update({"feature_missing_fraction": out[cols].isna().mean().to_dict(),
                     "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "code_sha256": sha256_file(Path(__file__)),
                     "input_manifest_sha256": sha256_file(OUT / "manifest.json")})
    if sigma == 0:
        np.testing.assert_allclose(out[cols], baseline[cols], rtol=1e-10,
                                   atol=1e-7, equal_nan=True)
        manifest["zero_feature_max_absolute_error"] = float(np.nanmax(
            np.abs(out[cols].to_numpy() - baseline[cols].to_numpy())))
    return out, manifest


def save_condition(df: pd.DataFrame, manifest: dict, sigma: int, seed: int):
    name = condition_name(sigma, seed)
    path = CACHE / f"{name}.csv"
    df.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    manifest["file_sha256"] = sha256_file(path)
    json_write(CACHE / f"{name}.json", manifest)


def stage0():
    deadline_check()
    train, test = load_tables()
    xtr, ytr, utr, _ = prepare(train)
    xte, _, ute, meta = prepare(test)
    cols = list(xtr.columns)
    assert len(cols) == 27 and list(xte.columns) == cols
    spec = model_specs(len(cols))[MODEL_NAME]
    raw, scores, est = fit_predict(spec, True, {"clf__C": 1.0}, xtr, ytr, utr, xte)
    final = smooth("soft", raw, scores, test.feature_block_id, test.start_time, 11)
    MODELS.mkdir(exist_ok=True)
    joblib.dump(est, MODELS / "full.joblib")
    json_write(MODELS / "G3_config.json", {
        "group": "G3", "feature_columns": cols, "classes": MODES,
        "params": {"clf__C": 1.0}, "class_weight": "balanced", "seed": 42,
        "model": MODEL_NAME, "n_features": len(cols),
        "estimator_classes": est.classes_.tolist(),
        "pipeline_parameters": est.get_params(),
    })
    json_write(MODELS / "feature_columns_G3.json", cols)
    pd.DataFrame({"user_id": test.user_id, "feature_block_id": test.feature_block_id,
                  "start_time": test.start_time, "raw": raw, "final": final}).to_csv(
        PRED / "stage0_G3_sigma0_predictions.csv", index=False)
    np.save(PRED / "stage0_G3_sigma0_scores.npy", scores)
    mraw = metrics_from_cm(point_confusion(meta, raw, MODES))
    mfin = metrics_from_cm(point_confusion(meta, final, MODES))
    ref = pd.read_csv(MODEL_DIR / "results" / "basic_w50_no_dt_v2" / "window_predictions_test.csv",
                      dtype={"user_id": str})
    ref_raw = ref[f"{MODEL_NAME} | raw"].to_numpy(object)
    ref_final = ref[MODEL_NAME].to_numpy(object)
    assert len(ref) == len(test)
    for col in ("user_id", "feature_block_id", "start_time"):
        a, b = ref[col], test[col]
        if col == "user_id":
            a = a.str.zfill(3)
        elif col == "start_time":
            a = pd.to_datetime(a)
        np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())
    common = {"raw_macro_f1": mraw["macro_f1"], "final_macro_f1": mfin["macro_f1"],
              "raw_accuracy": mraw["accuracy"], "final_accuracy": mfin["accuracy"],
              "reference_raw_macro_f1": 0.7789, "reference_final_macro_f1": 0.8443,
              "raw_label_diff": int(np.sum(raw != ref_raw)),
              "final_label_diff": int(np.sum(final != ref_final)),
              "reference_row_count": len(ref), "new_row_count": len(test)}
    json_write(METRICS / "stage0_baseline.json", common)
    log(f"stage0 baseline: raw={mraw['macro_f1']:.4f}, final={mfin['macro_f1']:.4f}, "
        f"label_diff raw={common['raw_label_diff']} final={common['final_label_diff']}")
    diff = test[META_COLS].copy()
    diff["raw_diff"] = raw != ref_raw
    diff["centered_diff"] = final != ref_final
    diff["new_raw"], diff["reference_raw"] = raw, ref_raw
    diff["new_centered"], diff["reference_centered"] = final, ref_final
    diff.to_csv(METRICS / "stage0_window_comparison.csv", index=False)
    assert abs(mraw["macro_f1"] - 0.7789) < 0.001
    assert abs(mfin["macro_f1"] - 0.8443) < 0.001
    assert common["raw_label_diff"] < len(test) * 0.02
    assert common["final_label_diff"] < len(test) * 0.02
    return {"train": train, "test": test, "G3": (est, cols)}


def trailing_soft(scores, blocks, times, k=11):
    scores = np.asarray(scores, float)
    order = pd.DataFrame({"block": np.asarray(blocks, object),
                          "time": np.asarray(times)}).sort_values(
                              ["block", "time"], kind="stable").index.to_numpy()
    s = scores[order]
    b = np.asarray(blocks, object)[order]
    cum = np.vstack([np.zeros((1, s.shape[1])), s.cumsum(axis=0)])
    starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
    block_no = np.cumsum(np.r_[True, b[1:] != b[:-1]]) - 1
    idx = np.arange(len(s))
    lo = np.maximum(idx - (k - 1), starts[block_no])
    mean = (cum[idx + 1] - cum[lo]) / (idx - lo + 1)[:, None]
    out = np.empty(len(s), object)
    out[order] = np.asarray(MODES, object)[mean.argmax(axis=1)]
    return out


def one_to_one_transitions(meta, pred, true, tol=1):
    d = meta[["user_id", "feature_block_id", "start_time"]].copy()
    d["pred"], d["true"] = np.asarray(pred, object), np.asarray(true, object)
    d = d.sort_values(["feature_block_id", "start_time"], kind="stable").reset_index()
    rows = []
    for block, g in d.groupby(["user_id", "feature_block_id"], sort=False):
        p = g.pred.to_numpy(object); t = g.true.to_numpy(object)
        pidx = np.flatnonzero(p[1:] != p[:-1]) + 1
        tidx = np.flatnonzero(t[1:] != t[:-1]) + 1
        edges = np.abs(pidx[:, None] - tidx[None, :]) <= tol
        matching = maximum_bipartite_matching(csr_matrix(edges), perm_type="column")
        rows.append((len(tidx), len(pidx), int((matching >= 0).sum())))
    true_n = sum(x[0] for x in rows); pred_n = sum(x[1] for x in rows); hit = sum(x[2] for x in rows)
    precision = hit / pred_n if pred_n else 0.0
    recall = hit / true_n if true_n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"true_transitions": true_n, "pred_transitions": pred_n, "matched_transitions": hit,
            "transition_precision": precision, "transition_recall": recall, "transition_f1": f1}


def stage1(context):
    deadline_check()
    tick = time.monotonic()
    train, test = context["train"], context["test"]
    uid = sorted(test.user_id.unique())[0]
    p = read_points(ROOT / "data" / "processed" / "quality_features" / f"{uid}.parquet")
    p = p.sort_values(["feature_block_id", "timestamp"], kind="stable").reset_index(drop=True)
    noisy, nd = perturb(p, 20, SEEDS[0])
    invariant = ["user_id", "trip_id", "feature_block_id", "timestamp", "mode"]
    pd.testing.assert_frame_equal(p[invariant], noisy[invariant])
    old_ids = assign_windows(p, 50, cut_at_label_change=False)
    before = len(noisy)
    noisy = build_features(noisy)
    noisy["window_id"] = assign_windows(noisy, 50, cut_at_label_change=False)
    pd.testing.assert_series_equal(noisy.window_id, old_ids, check_names=False)
    np.testing.assert_allclose(noisy.delta_t_s, p.delta_t_s, equal_nan=True)
    assert len(noisy) == len(p)
    w = window_features(noisy)
    base = test[test.user_id == uid].reset_index(drop=True)
    pd.testing.assert_frame_equal(w[META_COLS], base[META_COLS], check_dtype=False)
    est, cols = context["G3"]
    assert not np.allclose(w[cols], base[cols], equal_nan=True)
    # Poison stale derivatives: recomputation must depend on coordinates only.
    poisoned = perturb(p, 20, SEEDS[0])[0]
    for c in ("speed_mps", "acceleration_mps2", "jerk_mps3",
              "distance_m", "heading_change_rad", "heading_change_rate_rad_s"):
        poisoned[c] = 1234567.0
    rebuilt = build_features(poisoned)
    for c in ("speed_mps", "acceleration_mps2", "jerk_mps3",
              "distance_m", "heading_change_rad", "heading_change_rate_rad_s"):
        np.testing.assert_allclose(rebuilt[c], noisy[c], equal_nan=True)
    meta = w[META_COLS]
    x = w[cols].replace([np.inf, -np.inf], np.nan)
    raw = est.predict(x).astype(object)
    # Use the public helper by treating the fitted estimator as a fixed spec is unnecessary;
    # class scores are extracted through its decision function with class-column alignment.
    raw_scores = est.decision_function(x)
    aligned = np.full((len(x), len(MODES)), raw_scores.min() - 1)
    for j, c in enumerate(est.classes_):
        aligned[:, MODES.index(c)] = raw_scores[:, j]
    outs = {
        "S0_raw_k1": raw,
        "S1_centered_soft_k11": smooth_soft(aligned, w.feature_block_id, w.start_time, 11),
        "S2_trailing_soft_k11": trailing_soft(aligned, w.feature_block_id, w.start_time, 11),
    }
    rows = []
    for name, pred in outs.items():
        m = metrics_from_cm(point_confusion(meta, pred, MODES))
        rows.append({"smoother": name, **{k: m[k] for k in ("accuracy", "macro_f1")},
                     **one_to_one_transitions(meta, pred, w.label),
                     "isolated_flips": isolated_flips(pred, w.label, w.feature_block_id, w.start_time)[0]})
    report = {"user_id": uid, "sigma": 20, "seed": SEEDS[0], "points": before,
              "windows": len(w), "speed_gt100": int(noisy.speed_mps.gt(100).sum()),
              "noise": nd, "rows": rows}
    elapsed = time.monotonic() - tick
    report.update({"elapsed_seconds": elapsed,
                   "seconds_per_point": elapsed / len(p),
                   "process_rss_bytes": process_memory_bytes(),
                   "estimated_full_condition_seconds": elapsed * test.n_points.sum() / len(p),
                   "estimated_16_conditions_seconds": elapsed * test.n_points.sum() / len(p) * 16,
                   "invariance_and_derivative_recompute": "passed"})
    json_write(METRICS / "stage1_single_user.json", report)
    log(f"stage1 single user={uid} passed with {len(w):,} windows")
    return context


def stage2(context):
    deadline_check()
    _, test = context["train"], context["test"]
    for sigma in SIGMAS:
        for seed in ([0] if sigma == 0 else SEEDS):
            name = condition_name(sigma, seed)
            path = CACHE / f"{name}.csv"
            if path.exists() and (CACHE / f"{name}.json").exists():
                log(f"stage2 cached {name}")
                continue
            df, manifest = build_test_condition(sigma, seed, test)
            save_condition(df, manifest, sigma, seed)
    files = sorted(CACHE.glob("sigma*.csv"))
    assert len(files) == 16
    checks = []
    for p in files:
        d = pd.read_csv(p, dtype={"user_id": str})
        checks.append({"file": p.name, "rows": len(d), "points": int(d.n_points.sum()),
                       "hash": sha256_file(p)})
        assert len(d) == len(test) and int(d.n_points.sum()) == int(test.n_points.sum())
    pd.DataFrame(checks).to_csv(METRICS / "stage2_cache_index.csv", index=False)
    log("stage2 all 16 shared feature caches passed")
    return context


def train_group(group, train):
    if group == "G3":
        cols = list(prepare(train)[0].columns)
    else:
        cols = FEATURE_GROUPS[group]
    x = train[cols].replace([np.inf, -np.inf], np.nan)
    y = train.label
    spec = model_specs(len(cols))[MODEL_NAME]
    est = spec.make(True).set_params(clf__C=1.0).fit(x, y)
    joblib.dump(est, MODELS / f"{group}.joblib")
    json_write(MODELS / f"{group}_config.json", {
        "group": group, "feature_columns": cols, "n_features": len(cols),
        "model": MODEL_NAME, "class_weight": "balanced", "seed": 42, "C": 1.0,
        "classes": MODES, "estimator_classes": est.classes_.tolist(),
        "pipeline_parameters": est.get_params(),
    })
    return est, cols


def score_aligned(est, x):
    raw = est.predict(x).astype(object)
    s = est.decision_function(x)
    s = np.asarray(s, float)
    out = np.full((len(x), len(MODES)), s.min() - 1)
    for j, c in enumerate(est.classes_):
        out[:, MODES.index(c)] = s[:, j]
    return raw, out


def stage3(context):
    deadline_check()
    train, test = context["train"], context["test"]
    models = {}
    for group in GROUPS:
        path = MODELS / ("full.joblib" if group == "G3" else f"{group}.joblib")
        if path.exists():
            est = joblib.load(path)
            cols = json.loads((MODELS / f"{group}_config.json").read_text())["feature_columns"]
        else:
            est, cols = train_group(group, train)
        models[group] = (est, cols)
    rows = []
    for sigma in SIGMAS:
        for seed in ([0] if sigma == 0 else SEEDS):
            name = condition_name(sigma, seed)
            d = pd.read_csv(CACHE / f"{name}.csv", dtype={"user_id": str})
            d["start_time"] = pd.to_datetime(d.start_time)
            d["end_time"] = pd.to_datetime(d.end_time)
            meta = d[META_COLS]
            for group, (est, cols) in models.items():
                deadline_check()
                x = d[cols].replace([np.inf, -np.inf], np.nan)
                raw, scores = score_aligned(est, x)
                base = PRED / f"{group}_{name}.csv"
                out = meta[["user_id", "feature_block_id", "start_time", "end_time",
                            "n_points", "label", "purity"]].copy()
                for col in [f"cnt_{m}" for m in MODES]:
                    out[col] = meta[col]
                out["raw"] = raw
                for j, mode in enumerate(MODES):
                    out[f"score_{mode}"] = scores[:, j]
                out.to_csv(base, index=False, date_format="%Y-%m-%d %H:%M:%S")
                m = metrics_from_cm(point_confusion(meta, raw, MODES))
                rows.append({"group": group, "condition": name, "sigma": sigma,
                             "seed": seed, "smoother": "S0_raw_k1",
                             "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                             **dict(zip([f"f1_{m}" for m in MODES], m["per_class_f1"]))})
                log(f"stage3 {group} {name} raw={m['macro_f1']:.4f}")
    raw_df = pd.DataFrame(rows)
    raw_df["clean_macro_f1"] = raw_df.groupby("group").macro_f1.transform(
        lambda x: x.iloc[0])
    raw_df["macro_f1_drop"] = raw_df.macro_f1 - raw_df.clean_macro_f1
    raw_df.to_csv(METRICS / "stage3_raw_metrics.csv", index=False)
    assert len(list(PRED.glob("G1_sigma*.csv"))) == 16
    assert len(list(PRED.glob("G2_sigma*.csv"))) == 16
    assert len(list(PRED.glob("G3_sigma*.csv"))) == 16
    log("stage3 48 prediction score files and raw metrics passed")
    context["models"] = models
    return context


def read_prediction(path):
    d = pd.read_csv(path, dtype={"user_id": str})
    d.start_time = pd.to_datetime(d.start_time); d.end_time = pd.to_datetime(d.end_time)
    scores = d[[f"score_{m}" for m in MODES]].to_numpy(float)
    raw = d.raw.to_numpy(object)
    return d, raw, scores


def stage4(context):
    deadline_check()
    _, test = context["train"], context["test"]
    rows, old_rows, user_rows = [], [], []
    for group in GROUPS:
        clean_predictions = {}
        for sigma in SIGMAS:
            for seed in ([0] if sigma == 0 else SEEDS):
                name = condition_name(sigma, seed)
                d, raw, scores = read_prediction(PRED / f"{group}_{name}.csv")
                preds = {
                    "S0_raw_k1": raw,
                    "S1_centered_soft_k11": smooth_soft(scores, d.feature_block_id, d.start_time, 11),
                    "S2_trailing_soft_k11": trailing_soft(scores, d.feature_block_id, d.start_time, 11),
                }
                saved = d.copy()
                for smoother, pred in preds.items():
                    deadline_check()
                    saved[smoother] = pred
                    if sigma == 0:
                        clean_predictions[smoother] = pred.copy()
                    m = metrics_from_cm(point_confusion(d, pred, MODES))
                    flips = isolated_flips(pred, d.label, d.feature_block_id, d.start_time)
                    tr = one_to_one_transitions(d, pred, d.label)
                    old = segment_metrics(d, pred, d.label)
                    rows.append({"group": group, "condition": name, "sigma": sigma,
                                 "seed": seed, "smoother": smoother, "accuracy": m["accuracy"],
                                 "macro_f1": m["macro_f1"], "isolated_flips": flips[0],
                                 "isolated_flips_wrong": flips[1],
                                 "isolated_flips_per100_windows": flips[0] / len(d) * 100,
                                 "wrong_isolated_flips_per100_windows": flips[1] / len(d) * 100,
                                 "prediction_change_from_clean_point_fraction": float(
                                     np.average(pred != clean_predictions[smoother],
                                                weights=d.n_points)),
                                 **dict(zip([f"f1_{mode}" for mode in MODES], m["per_class_f1"])),
                                 **tr})
                    old_rows.append({"group": group, "condition": name, "sigma": sigma,
                                     "seed": seed, "smoother": smoother, **old})
                    for uid, sub in d.groupby("user_id", sort=False):
                        ix = sub.index.to_numpy()
                        um = metrics_from_cm(point_confusion(sub, pred[ix], MODES))
                        uf = isolated_flips(pred[ix], sub.label,
                                            sub.feature_block_id, sub.start_time)
                        user_rows.append({
                            "group": group, "condition": name, "sigma": sigma,
                            "seed": seed, "smoother": smoother, "user_id": uid,
                            "points": int(sub.n_points.sum()), "windows": len(sub),
                            "macro_f1": um["macro_f1"], "accuracy": um["accuracy"],
                            **dict(zip([f"f1_{mode}" for mode in MODES], um["per_class_f1"])),
                            "isolated_flips": uf[0], "isolated_flips_wrong": uf[1],
                            "wrong_isolated_flips_per100_windows": uf[1] / len(sub) * 100,
                            "prediction_change_from_clean_point_fraction": float(np.average(
                                pred[ix] != clean_predictions[smoother][ix], weights=sub.n_points)),
                            **one_to_one_transitions(sub, pred[ix], sub.label)})
                saved.to_csv(PRED / f"{group}_{name}_smoothed.csv", index=False)
    out = pd.DataFrame(rows)
    out["clean_macro_f1"] = out.groupby(["group", "smoother"]).macro_f1.transform(
        lambda x: x.iloc[0])
    out["macro_f1_drop"] = out.macro_f1 - out.clean_macro_f1
    out.to_csv(METRICS / "stage4_smoothing_metrics.csv", index=False)
    pd.DataFrame(old_rows).to_csv(METRICS / "stage4_legacy_segment_metrics.csv", index=False)
    pd.DataFrame(user_rows).to_csv(METRICS / "per_user_results.csv", index=False)
    out[["group", "sigma", "seed", "smoother", *[f"f1_{m}" for m in MODES]]].to_csv(
        METRICS / "per_class_f1.csv", index=False)
    out[["group", "sigma", "seed", "smoother", "true_transitions", "pred_transitions",
         "matched_transitions", "transition_precision", "transition_recall",
         "transition_f1"]].to_csv(METRICS / "transition_metrics.csv", index=False)
    assert set(out.smoother) == set(SMOOTHERS) and len(out) == 3 * 16 * 3
    assert len(user_rows) == 144 * 13
    assert out.true_transitions.eq(529).all()
    assert np.isfinite(out.select_dtypes("number").to_numpy()).all()
    log("stage4 smoothing, one-to-one transitions, and legacy metrics passed")
    return context


def bootstrap_scores(cms):
    tp = np.diagonal(cms, axis1=1, axis2=2).astype(float)
    support = cms.sum(axis=2)
    predicted = cms.sum(axis=1)
    f1 = np.divide(2 * tp, support + predicted,
                   out=np.zeros_like(tp), where=(support + predicted) > 0)
    macro = f1.mean(axis=1)
    acc = tp.sum(axis=1) / cms.sum(axis=(1, 2))
    return macro, acc


CAPTION = ("Added noise is independent Gaussian E/N noise, not a calibrated model of real "
           "GPS temporal errors. Noise settings are project-defined; Endo's noise study used "
           "KT. Smoothing averages decision scores, not GPS coordinates.")


def plot_outputs(metrics):
    plt.figure(figsize=(8, 5))
    for group in GROUPS:
        q = metrics[(metrics.group == group) & (metrics.smoother == "S0_raw_k1")]
        means = q.groupby("sigma").macro_f1.mean()
        stds = q.groupby("sigma").macro_f1.std().fillna(0)
        plt.errorbar(means.index, means.values, yerr=stds.values, marker="o", capsize=4, label=group)
    plt.xlabel("Added coordinate noise sigma (m)")
    plt.ylabel("Point-level Macro-F1")
    plt.title("Feature groups: clean training, noisy test")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout(rect=(0, .15, 1, 1))
    plt.gcf().text(.03, .02, "Error bars: sample SD over 5 perturbation seeds; not user CIs.\n"
                   + CAPTION, fontsize=7, wrap=True)
    plt.savefig(FIGURES / "figure1_raw_macro_f1_vs_sigma.png", dpi=180); plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, col, title in zip(axes, [
            "macro_f1", "wrong_isolated_flips_per100_windows", "transition_f1"],
            ["Point Macro-F1", "Wrong isolated flips / 100 windows", "Transition F1"]):
        for j, sigma in enumerate((0, 20)):
            q = metrics[(metrics.group == "G3") & (metrics.sigma == sigma)]
            means = [q[q.smoother == s][col].mean() for s in SMOOTHERS]
            stds = [q[q.smoother == s][col].std(ddof=1) if sigma else 0 for s in SMOOTHERS]
            ax.bar(np.arange(3) + (j - .5) * .36, means, .36, yerr=stds,
                   capsize=3, label=f"sigma={sigma} m")
        ax.set_xticks(np.arange(3), ["Raw", "Centered", "Trailing"])
        ax.set_title(title, fontsize=11); ax.grid(axis="y", alpha=.25)
    axes[0].legend()
    fig.suptitle("G3: classification, stability and transition trade-offs")
    fig.tight_layout(rect=(0, .19, 1, .94))
    fig.text(.02, .025, "Centered k=11 uses future windows. Trailing k=11 uses current/history.\n"
             "Bars at 20 m: mean +/- seed SD.\n" + CAPTION, fontsize=7, wrap=True)
    fig.savefig(FIGURES / "figure2_G3_smoothing_tradeoff.png", dpi=180); plt.close(fig)


def make_case_figure():
    import matplotlib.dates as mdates
    clean = pd.read_csv(PRED / "G3_sigma00_seed0_smoothed.csv", dtype={"user_id": str})
    noisy = pd.read_csv(PRED / f"G3_sigma20_seed{SEEDS[0]}_smoothed.csv",
                        dtype={"user_id": str})
    for d in (clean, noisy):
        d["start_time"] = pd.to_datetime(d.start_time)
    candidates = np.flatnonzero((clean.raw == clean.label) & (noisy.raw != clean.label))
    assert len(candidates), "No noise-failure case exists under the declared selection rule"
    first = int(candidates[0])
    short = None
    for _, sub in clean.groupby(["user_id", "feature_block_id"], sort=False):
        y = sub.label.to_numpy()
        starts = np.flatnonzero(np.r_[True, y[1:] != y[:-1]])
        ends = np.r_[starts[1:], len(sub)]
        for a, b in zip(starts, ends):
            if a > 0 and b < len(sub) and b - a <= 5:
                q = sub.iloc[a:b]
                if (q.raw == q.label).any() and (q.S1_centered_soft_k11 != q.label).all():
                    short = (int(q.index[0]), int(q.index[-1]))
                    break
        if short is not None:
            break
    assert short is not None, "No erased-short-mode case exists under the declared selection rule"
    choices = [("Noise failure: first clean-correct / noisy-wrong window", first, first, noisy),
               ("Short true mode erased by centered smoothing", *short, clean)]
    feature_clean = pd.read_csv(CACHE / "sigma00_seed0.csv")
    feature_noisy = pd.read_csv(CACHE / f"sigma20_seed{SEEDS[0]}.csv")
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), gridspec_kw={"height_ratios": [2, 1, 2, 1]})
    case_rows = []
    mapping = {m: i for i, m in enumerate(MODES)}
    for panel, (title, a, b, source) in enumerate(choices):
        block = source.feature_block_id.iloc[a]
        indices = source.index[source.feature_block_id == block].to_numpy()
        lo, hi = max(indices.min(), a - 7), min(indices.max() + 1, b + 8)
        q = source.iloc[lo:hi].copy()
        ax, speed = axes[2 * panel:2 * panel + 2]
        for col, label, style in [("label", "Window majority truth", "-"),
                                  ("raw", "Raw", "--"),
                                  ("S1_centered_soft_k11", "Centered soft", ":"),
                                  ("S2_trailing_soft_k11", "Trailing soft", "-.")]:
            ax.step(q.start_time, q[col].map(mapping), where="post", label=label,
                    linestyle=style, linewidth=1.7)
        ax.axvspan(source.start_time.iloc[a], pd.to_datetime(source.end_time.iloc[b]),
                   alpha=.1, color="red")
        ax.set_yticks(range(5), MODES); ax.set_title(title, fontsize=11)
        ax.legend(ncol=4, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.0))
        for features, label in [(feature_clean, "Observed coordinates"),
                                (feature_noisy, "Added noise sigma=20 m")]:
            speed.plot(q.start_time, features.speed_mean.iloc[lo:hi], label=label)
        speed.set_ylabel("Window speed\nmean (m/s)", fontsize=9)
        speed.legend(fontsize=8)
        speed.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        speed.set_xlabel(f"GPS timestamp {q.start_time.iloc[0]:%Y-%m-%d}; user "
                         f"{q.user_id.iloc[0]}, block {block}", fontsize=9)
        q["case"] = panel + 1
        q["observed_speed_mean"] = feature_clean.speed_mean.iloc[lo:hi].to_numpy()
        q["noisy_speed_mean"] = feature_noisy.speed_mean.iloc[lo:hi].to_numpy()
        case_rows.append(q)
    fig.tight_layout(rect=(0, .12, 1, 1), h_pad=1.5)
    fig.text(.015, .02, "Explanatory test examples selected by the fixed first-match rule; "
             "they do not estimate event frequency.\n" + CAPTION, fontsize=8, wrap=True)
    fig.savefig(FIGURES / "figure3_case_timeline.png", dpi=180)
    plt.close(fig)
    pd.concat(case_rows, ignore_index=True).to_csv(METRICS / "figure3_case_data.csv", index=False)
    json_write(METRICS / "case_selection.json", {
        "rule": "First ordered clean-correct/noisy-wrong G3 window at 20m seed20261123; "
                "first interior truth run of <=5 windows with raw correct >=1 window "
                "and centered wrong throughout, sigma0.",
        "noise_case_index": first, "short_run_indices": short,
        "scope": "explanatory test examples; not representative frequencies"})


def stage5(context):
    deadline_check()
    train, test = context["train"], context["test"]
    metrics = pd.read_csv(METRICS / "stage4_smoothing_metrics.csv")
    groups = np.asarray(cv_groups(test.user_id, ROOT / "data/splits/report.json"))
    clusters = pd.unique(groups)
    assert len(clusters) == 9
    draws = np.random.default_rng(42).multinomial(9, np.full(9, 1 / 9), size=5000)
    np.savez_compressed(METRICS / "paired_bootstrap_draws.npz", draws=draws, clusters=clusters)
    distributions, seed_rows = {}, []
    for group in GROUPS:
        for sigma in SIGMAS:
            for seed in ([0] if sigma == 0 else SEEDS):
                deadline_check()
                d = pd.read_csv(PRED / f"{group}_{condition_name(sigma, seed)}_smoothed.csv",
                                dtype={"user_id": str})
                np.testing.assert_array_equal(d.user_id, test.user_id)
                for smoother in SMOOTHERS:
                    pred = d[smoother].to_numpy(object)
                    per = np.stack([point_confusion(d.loc[groups == c], pred[groups == c], MODES)
                                    for c in clusters])
                    values = bootstrap_scores(np.tensordot(draws, per, axes=1))
                    distributions[(group, smoother, sigma, seed)] = values
                    row = {"group": group, "smoother": smoother, "sigma": sigma, "seed": seed}
                    for label, vals in zip(["macro_f1", "accuracy"], values):
                        row[f"{label}_ci_low"], row[f"{label}_ci_high"] = np.percentile(vals, [2.5, 97.5])
                    seed_rows.append(row)
    pd.DataFrame(seed_rows).to_csv(METRICS / "bootstrap_per_seed.csv", index=False)
    rows, means = [], {}
    for group in GROUPS:
        for smoother in SMOOTHERS:
            clean = distributions[(group, smoother, 0, 0)]
            for sigma in SIGMAS:
                seeds = [0] if sigma == 0 else SEEDS
                q = metrics[(metrics.group == group) & (metrics.smoother == smoother)
                            & (metrics.sigma == sigma)]
                row = {"group": group, "smoother": smoother, "sigma": sigma,
                       "seed_count": len(seeds)}
                for j, label in enumerate(["macro_f1", "accuracy"]):
                    # F1 is computed separately for every seed before averaging.
                    mean_dist = np.mean([distributions[(group, smoother, sigma, s)][j]
                                         for s in seeds], axis=0)
                    means[(group, smoother, sigma, label)] = mean_dist
                    delta = mean_dist - clean[j]
                    row.update({
                        f"{label}_seed_mean": float(q[label].mean()),
                        f"{label}_seed_sd": float(q[label].std(ddof=1)) if sigma else 0.0,
                        f"{label}_ci_low": float(np.percentile(mean_dist, 2.5)),
                        f"{label}_ci_high": float(np.percentile(mean_dist, 97.5)),
                        f"{label}_delta_from_clean": float(q[label].mean() -
                            metrics[(metrics.group == group) & (metrics.smoother == smoother)
                                    & (metrics.sigma == 0)][label].iloc[0]),
                        f"{label}_delta_ci_low": float(np.percentile(delta, 2.5)),
                        f"{label}_delta_ci_high": float(np.percentile(delta, 97.5))})
                rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(METRICS / "stage5_bootstrap_all.csv", index=False)
    summary[summary.sigma.isin([0, 20])].to_csv(
        METRICS / "table2_sigma0_sigma20_bootstrap.csv", index=False)
    contrasts = []
    for smoother in SMOOTHERS:
        for a, b in [("G3", "G2"), ("G3", "G1"), ("G2", "G1")]:
            delta_a = means[(a, smoother, 20, "macro_f1")] - means[(a, smoother, 0, "macro_f1")]
            delta_b = means[(b, smoother, 20, "macro_f1")] - means[(b, smoother, 0, "macro_f1")]
            dist = delta_a - delta_b
            aa = summary[(summary.group == a) & (summary.smoother == smoother) & (summary.sigma == 20)]
            bb = summary[(summary.group == b) & (summary.smoother == smoother) & (summary.sigma == 20)]
            contrasts.append({"comparison": f"{a} minus {b}", "smoother": smoother,
                              "delta_difference": float(aa.macro_f1_delta_from_clean.iloc[0]
                                                        - bb.macro_f1_delta_from_clean.iloc[0]),
                              "ci_low": np.percentile(dist, 2.5), "ci_high": np.percentile(dist, 97.5)})
    pd.DataFrame(contrasts).to_csv(METRICS / "paired_group_delta_contrasts.csv", index=False)
    metrics.to_csv(METRICS / "all_metrics.csv", index=False)
    protocol_df = pd.DataFrame([
        ["dataset", "GeoLife processed quality_features", f"{test.n_points.sum()} test points; {len(test)} windows"],
        ["split", "fixed user split", f"{train.user_id.nunique()} train / 13 test / 9 linked groups"],
        ["window", "blind W=50 points", "inside original block; original tail merging"],
        ["noise", "E,N iid Gaussian; sigma is per-axis SD", "0/5/10/20 m; RMS=sqrt(2)*sigma; paired 5 seeds"],
        ["G1", ", ".join(FEATURE_GROUPS["G1"]), "10 features"],
        ["G2", ", ".join(FEATURE_GROUPS["G2"]), "15 features"],
        ["G3", ", ".join(context["G3"][1]), "27 original v2 features"],
        ["model", "median impute; standardize; Nystroem RBF; LinearSVC", "1000 components; C=1; balanced; seed42; gamma=1/d"],
        ["smoothing", "decision score domain: raw / centered / trailing", "k=1 / 11 / 11; centered uses future"],
        ["transition", "window-majority truth; max one-to-one boundary matching", "+/-1 window, same user and block"],
        ["uncertainty", "9 linked user groups; 5000 paired bootstrap draws", "Seed mean/SD reported separately; fixed 5-class F1"],
    ], columns=["item", "setting", "details"])
    protocol_df.to_csv(METRICS / "table1_protocol.csv", index=False)
    plot_outputs(metrics)
    make_case_figure()
    assert len(summary) == 36 and len(seed_rows) == 144
    for name in ["figure1_raw_macro_f1_vs_sigma.png", "figure2_G3_smoothing_tradeoff.png",
                 "figure3_case_timeline.png"]:
        assert (FIGURES / name).stat().st_size > 10000
    for name in ["table1_protocol.csv", "table2_sigma0_sigma20_bootstrap.csv",
                 "all_metrics.csv", "per_user_results.csv", "per_class_f1.csv"]:
        assert (METRICS / name).stat().st_size > 0
    shares = test.groupby(groups).n_points.sum() / test.n_points.sum()
    lines = ["# GeoLife noise / feature / smoothing execution report", "",
             "All six stages passed their implemented acceptance checks.",
             "Reference results were read only; all new artifacts are under this directory.",
             "", "## Fixed protocol and interpretation", "", CAPTION,
             "Sigma=0 means no added noise, not true coordinates. Clean training, noisy test.",
             "Decision scores are not calibrated probabilities. Centered smoothing uses future "
             "windows; trailing smoothing still requires completed windows and is not a full real-time system.",
             "Seeds describe perturbation variability; bootstrap CIs use 9 linked user groups.",
             f"Largest linked group contributes {shares.max():.4%} of test points.",
             "Transition truth uses window majority labels (529 boundaries), losing some point-level changes.",
             "Delta is noisy minus own clean score; negative values are degradation. Always inspect absolute scores.",
             "Examples use deterministic error selection and are explanatory, not representative frequencies.",
             "", "## Results: raw Macro-F1", ""]
    for _, r in summary[summary.smoother == "S0_raw_k1"].iterrows():
        lines.append(f"- {r['group']} sigma={r.sigma:g}m: {r.macro_f1_seed_mean:.4f} "
                     f"+/- {r.macro_f1_seed_sd:.4f} seed SD; user CI "
                     f"[{r.macro_f1_ci_low:.4f}, {r.macro_f1_ci_high:.4f}]; "
                     f"delta {r.macro_f1_delta_from_clean:+.4f}.")
    (OUT / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("stage5 paired bootstrap, tables, and three figures completed")
    return context


def run_stage(name, fn, context):
    for attempt in range(1, 4):
        deadline_check()
        _CHECK["attempts"][name] = attempt
        save_checkpoint(name, "running", attempt=attempt)
        try:
            log(f"START {name} attempt {attempt}/3")
            context = fn(context)
            _CHECK["completed"].append(name)
            save_checkpoint(name, "passed", attempt=attempt, completed=_CHECK["completed"])
            log(f"PASS {name} attempt {attempt}/3; remaining={max(0, _DEADLINE-time.monotonic()):.0f}s")
            return context
        except TimeoutError:
            raise
        except Exception as exc:
            log(f"FAIL {name} attempt {attempt}/3: {type(exc).__name__}: {exc}")
            (OUT / "errors").mkdir(exist_ok=True)
            (OUT / "errors" / f"{name}_attempt{attempt}.txt").write_text(
                traceback.format_exc(), encoding="utf-8")
            save_checkpoint(name, "failed", attempt=attempt, error=str(exc))
            if attempt >= 3:
                raise
            time.sleep(2)
    return context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, choices=range(6), required=True)
    args = parser.parse_args()
    if args.stage == 0 and not (OUT / "manifest.json").exists():
        init()
    name = f"stage{args.stage}"
    source_dir = OUT / "source_snapshots"
    source_dir.mkdir(exist_ok=True)
    source_path = source_dir / f"{name}_{hashlib.sha256(LOADED_SOURCE).hexdigest()[:12]}.py"
    source_path.write_bytes(LOADED_SOURCE)
    save_checkpoint(name, "running")
    try:
        if args.stage == 0:
            stage0()
        else:
            train, test = load_tables()
            context = {"train": train, "test": test}
            config = json.loads((MODELS / "G3_config.json").read_text(encoding="utf-8"))
            context["G3"] = (joblib.load(MODELS / "full.joblib"), config["feature_columns"])
            [None, stage1, stage2, stage3, stage4, stage5][args.stage](context)
        deadline_check()
        json_write(OUT / f"{name}_acceptance.json", {
            "stage": name, "passed": True, "finished_at": time.time(),
            "code_sha256": sha256_file(Path(__file__)),
            "elapsed_seconds": time.monotonic() - _START})
        save_checkpoint(name, "passed")
        log(f"PASS {name}")
    except TimeoutError as exc:
        save_checkpoint(_CHECK.get("stage", "unknown"), "deadline", error=str(exc))
        log(f"STOPPED AT DEADLINE: {exc}")
    except Exception as exc:
        save_checkpoint(_CHECK.get("stage", "unknown"), "terminated", error=str(exc))
        log(f"TERMINATED AFTER FAILURE: {exc}")
        raise


if __name__ == "__main__":
    main()
