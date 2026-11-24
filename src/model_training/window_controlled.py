"""Full-coverage window comparison on original GPS, with a common physical smoothing span.

Stages: prepare; cv --scheme points50/time120; test --scheme points50/time120.
All positive-length training windows are kept. Model parameters are fixed, not reselected.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata as md
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import pipeline_v2 as p
from build_windows import assign_windows, read_points
from prepare_sampling import downsample, duration_windows
from models import MODES, cv_groups, load_windows, metrics_from_cm, point_confusion, prepare

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "outputs/window_controlled_v1"
MODEL = "SVM (RBF, Nystroem, all data)"
SPAN_SECONDS = 1200
SOURCES = {"points50": PROJECT / "outputs/sampling_main_v1/interval_0s_points_50",
           "time120": PROJECT / "outputs/sampling_duration_v1/interval_0s_time_120s"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(labels):
    codes = pd.Categorical(labels, categories=MODES).codes.astype(np.int64)
    if (codes < 0).any():
        raise ValueError("Unknown class")
    return codes


def cm(truth, predictions):
    return np.bincount(np.asarray(truth) * 5 + predictions, minlength=25).reshape(5, 5)


def smooth_points(members, window_scores, span_seconds=SPAN_SECONDS):
    """Mean class scores of retained GPS observations within +/- span/2, inside each block.

    Both window schemes use the same observation timestamps and time limits. This is
    centered offline smoothing, not a causal online predictor.
    """
    if not np.isfinite(span_seconds) or span_seconds < 0:
        raise ValueError("Invalid smoothing duration")
    times = members.timestamp.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    window = members.window_index.to_numpy(dtype=np.int64)
    result = np.empty(len(members), dtype=np.int64)
    half = int(span_seconds * 1e9 / 2)
    for positions in members.groupby(["user_id", "feature_block_id"], sort=False).indices.values():
        t = times[positions]
        if (np.diff(t) <= 0).any():
            raise ValueError("Non-increasing timestamps inside a block")
        values = window_scores[window[positions]]
        prefix = np.vstack([np.zeros((1, values.shape[1])), values.cumsum(axis=0)])
        left = np.searchsorted(t, t - half, side="left")
        right = np.searchsorted(t, t + half, side="right")
        result[positions] = ((prefix[right] - prefix[left]) / (right - left)[:, None]).argmax(axis=1)
    return result


def read_members(scheme, split):
    d = pd.read_parquet(ROOT / scheme / f"{split}_members.parquet")
    d["user_id"] = d.user_id.astype(str).str.zfill(3)
    d["timestamp"] = pd.to_datetime(d.timestamp)
    return d


def check_members(members, windows):
    counts = pd.crosstab(members.window_index, members["mode"]).reindex(
        index=windows.index, columns=MODES, fill_value=0).to_numpy()
    if not np.array_equal(counts, windows[[f"cnt_{m}" for m in MODES]].to_numpy()):
        raise ValueError("Point membership and window labels disagree")
    if members.duplicated(["user_id", "timestamp"]).any():
        raise ValueError("Duplicate observation membership")
    idx = members.window_index.to_numpy(dtype=int)
    times = members.timestamp.to_numpy(dtype="datetime64[ns]")
    if not ((times >= pd.to_datetime(windows.start_time).to_numpy()[idx]) &
            (times <= pd.to_datetime(windows.end_time).to_numpy()[idx])).all():
        raise ValueError("Observation outside its window")


def prepare_data():
    if (ROOT / "protocol.json").exists():
        raise ValueError("Prepared experiment already exists; do not overwrite")
    ROOT.mkdir(parents=True, exist_ok=True)
    train_ids = json.loads((PROJECT / "data/splits/train_users.json").read_text())
    folds = json.loads((PROJECT / "outputs/sampling_results_v1/frozen_training_folds.json").read_text())
    groups = cv_groups(pd.Series(train_ids), PROJECT / "data/splits/report.json")
    for user, group in zip(train_ids, groups):
        if user not in folds:
            linked = [u for u, g in zip(train_ids, groups) if g == group and u in folds]
            folds[user] = folds[linked[0]] if linked else int(hashlib.sha256(f"42:{group}".encode()).hexdigest(), 16) % 5
    if set(folds) != set(train_ids):
        raise ValueError("Fold map must contain exactly the 51 training users")
    (ROOT / "folds_51.json").write_text(json.dumps(folds, indent=2))
    targets, lookups, parts, coverage = {}, {}, {s: [] for s in SOURCES}, []
    for scheme, source in SOURCES.items():
        (ROOT / scheme).mkdir(exist_ok=True)
        pure, blind = load_windows(source / "windows.csv"), load_windows(source / "windows_blind.csv")
        train = pure[pure.split == "train"].reset_index(drop=True)
        targets[scheme] = blind[blind.split == "train"].reset_index(drop=True)
        local = targets[scheme].groupby("user_id", sort=False).cumcount()
        lookups[scheme] = pd.Series(targets[scheme].index.to_numpy(),
                                   index=pd.MultiIndex.from_arrays([targets[scheme].user_id, local]))
        test = pd.read_csv(source / "test_point_membership.csv", dtype={"user_id": str})
        test = test.rename(columns={"input_feature_block_id": "feature_block_id", "test_window_index": "window_index"})
        test = test[["user_id", "feature_block_id", "timestamp", "mode", "window_index"]]
        test.timestamp = pd.to_datetime(test.timestamp)
        check_members(test, pure[pure.split == "test"].reset_index(drop=True))
        test.to_parquet(ROOT / scheme / "test_members.parquet", index=False)
        coverage.append({"scheme": scheme, "train_users": train.user_id.nunique(),
                         "train_points": int(train.n_points.sum()), "train_windows": len(train),
                         "min_train_points": 1, "test_points": len(test)})
    for path in sorted((PROJECT.parent / "quality_features/quality_features").glob("*.parquet")):
        if path.stem not in train_ids:
            continue
        data = downsample(read_points(path), 0)
        for scheme in SOURCES:
            ids = assign_windows(data, 50, False) if scheme == "points50" else duration_windows(data, 120, False)
            member = data[["user_id", "input_feature_block_id", "timestamp", "mode"]].rename(
                columns={"input_feature_block_id": "feature_block_id"}).copy()
            member["window_index"] = lookups[scheme].reindex(
                pd.MultiIndex.from_arrays([data.user_id, ids])).to_numpy()
            if member.window_index.isna().any():
                raise ValueError("Missing training window membership")
            parts[scheme].append(member)
        print(f"[prepare] {path.stem}: {len(data):,} training points", flush=True)
    for scheme in SOURCES:
        members = pd.concat(parts[scheme], ignore_index=True)
        check_members(members, targets[scheme])
        if len(members) != 3486792 or members.user_id.nunique() != 51:
            raise ValueError("Expected all training points and users")
        members.to_parquet(ROOT / scheme / "train_members.parquet", index=False)
    for split in ["train", "test"]:
        a, b = (read_members(s, split) for s in SOURCES)
        if not a[["user_id", "timestamp", "mode", "feature_block_id"]].equals(
                b[["user_id", "timestamp", "mode", "feature_block_id"]]):
            raise ValueError("Window schemes cover different observations")
    pd.DataFrame(coverage).to_csv(ROOT / "coverage.csv", index=False)
    files = [Path(__file__), Path(p.__file__), PROJECT / "src/model_training/models.py",
             PROJECT / "src/model_training/prepare_sampling.py", ROOT / "folds_51.json"]
    files += [source / f for source in SOURCES.values() for f in ["windows.csv", "windows_blind.csv", "condition.json"]]
    files += [ROOT / scheme / f"{split}_members.parquet" for scheme in SOURCES for split in ["train", "test"]]
    protocol = {"date": "2026-11-20", "purpose": "Full-coverage window comparison, original GPS only",
                "schemes": ["50 points with original short-tail merge", "120 seconds, partial tails retained"],
                "min_training_points": 1, "train_users": 51, "train_points": 3486792,
                "test_users": 13, "test_points": 860972, "model": MODEL, "components": 1000,
                "C": 1, "class_weights": "balanced", "primary_metric": "unsmoothed point macro-F1",
                "secondary_smoothing": "Centered point-score average within +/-600 seconds inside original blocks",
                "span_seconds": SPAN_SECONDS, "parameter_selection": "None; all parameters fixed before new evaluation",
                "status": "Follow-up analysis designed after prior test results; not an independent confirmatory study",
                "fold_extension": "Original 49 user assignments preserved; missing users assigned by SHA256(42:group) mod 5",
                "python": platform.python_version(),
                "versions": {m: md.version(m) for m in ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy"]},
                "file_sha256": {str(f.resolve()): sha(f) for f in files}}
    (ROOT / "protocol.json").write_text(json.dumps(protocol, indent=2))
    print("[prepared] identical full train/test observation coverage for both schemes", flush=True)


def run(scheme, stage):
    protocol = json.loads((ROOT / "protocol.json").read_text())
    for path, digest in protocol["file_sha256"].items():
        if sha(path) != digest:
            raise ValueError(f"Experiment source/input changed: {path}")
    source, out = SOURCES[scheme], ROOT / scheme
    pure, blind = load_windows(source / "windows.csv"), load_windows(source / "windows_blind.csv")
    train = pure[pure.split == "train"].reset_index(drop=True)
    Xtr, ytr, utr, _ = prepare(train)
    spec = p.model_specs(Xtr.shape[1])[MODEL]
    members = read_members(scheme, "train" if stage == "cv" else "test")
    target = blind[blind.split == "train"].reset_index(drop=True) if stage == "cv" else pure[pure.split == "test"].reset_index(drop=True)
    Xev, _, uev, _ = prepare(target)
    check_members(members, target)
    scores, raw = np.empty((len(target), 5)), np.empty(len(target), dtype=object)
    t0 = time.perf_counter()
    with threadpool_limits(2):
        if stage == "cv":
            if (out / "cv_metrics.json").exists():
                print(f"[cv already complete] {scheme}")
                return
            folds = json.loads((ROOT / "folds_51.json").read_text())
            fit_fold, eval_fold = utr.map(folds).to_numpy(), uev.map(folds).to_numpy()
            for fold in range(5):
                fit, evaluate = fit_fold != fold, eval_fold == fold
                cache = out / f"oof_fold_{fold}.npz"
                if cache.exists():
                    saved = np.load(cache, allow_pickle=False)
                    if not np.array_equal(saved["indices"], np.flatnonzero(evaluate)):
                        raise ValueError("Checkpoint rows do not match")
                    raw[evaluate], scores[evaluate] = saved["raw"], saved["scores"]
                    print(f"[cached] {scheme} fold {fold + 1}/5", flush=True)
                else:
                    print(f"[fit] {scheme} fold {fold + 1}/5 train={fit.sum()} eval={evaluate.sum()}", flush=True)
                    a, b, _ = p.fit_predict(spec, True, {"clf__C": 1.0}, Xtr[fit], ytr[fit], utr[fit], Xev[evaluate])
                    raw[evaluate], scores[evaluate] = a, b
                    np.savez_compressed(cache, indices=np.flatnonzero(evaluate), raw=np.asarray(a, dtype="U5"), scores=b)
        else:
            if not all((ROOT / s / "cv_metrics.json").exists() for s in SOURCES):
                raise ValueError("Complete both CV runs before any new test evaluation")
            freeze = ROOT / "cv_frozen_before_test.json"
            if not freeze.exists():
                if any((ROOT / s / "test_metrics.json").exists() for s in SOURCES):
                    raise ValueError("Unexpected pre-existing test results")
                freeze.write_text(json.dumps({s: sha(ROOT / s / "cv_metrics.json") for s in SOURCES}, indent=2))
            if (out / "test_metrics.json").exists():
                print(f"[test already complete] {scheme}")
                return
            print(f"[final fit] {scheme} {len(train):,} windows, all 51 users", flush=True)
            raw, scores, estimator = p.fit_predict(spec, True, {"clf__C": 1.0}, Xtr, ytr, utr, Xev)
            joblib.dump(estimator, out / "model.joblib")
            np.savez_compressed(out / "window_test_scores.npz", raw=np.asarray(raw, dtype="U5"), scores=scores)
    truth = encode(members["mode"])
    raw_codes = encode(raw)[members.window_index.to_numpy(dtype=int)]
    smoothed = smooth_points(members, scores)
    result = {"scheme": scheme, "stage": stage, "users": members.user_id.nunique(), "points": len(members),
              "training_points": int(train.n_points.sum()), "training_users": train.user_id.nunique(),
              "training_windows": len(train), "elapsed_seconds": time.perf_counter() - t0}
    if not np.array_equal(cm(truth, raw_codes), point_confusion(target, raw, MODES)):
        raise ValueError("Point/window confusion matrices disagree")
    for name, predictions in [("raw", raw_codes), ("time20min", smoothed)]:
        matrix = cm(truth, predictions)
        result[name] = {**{k: (v.tolist() if isinstance(v, np.ndarray) else float(v))
                          for k, v in metrics_from_cm(matrix).items()}, "confusion_matrix": matrix.tolist()}
        if stage == "cv":
            member_folds = members.user_id.map(folds).to_numpy()
            f1 = [metrics_from_cm(cm(truth[member_folds == f], predictions[member_folds == f]))["macro_f1"] for f in range(5)]
            result[name]["fold_macro_f1"] = f1
            result[name]["fold_mean_macro_f1"] = float(np.mean(f1))
    if stage == "test":
        pred = members.copy()
        pred["raw_code"], pred["time20min_code"] = raw_codes, smoothed
        pred.to_parquet(out / "test_predictions.parquet", index=False)
    (out / f"{stage}_metrics.json").write_text(json.dumps(result, indent=2))
    print(f"[done] {scheme} {stage}: raw={result['raw']['macro_f1']:.4f}, physical20min={result['time20min']['macro_f1']:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["prepare", "cv", "test"], required=True)
    ap.add_argument("--scheme", choices=list(SOURCES))
    a = ap.parse_args()
    if a.stage == "prepare":
        prepare_data()
    elif a.scheme:
        run(a.scheme, a.stage)
    else:
        ap.error("--scheme is required for training/evaluation")


if __name__ == "__main__":
    main()
