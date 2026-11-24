"""Read-only, post-hoc diagnostics for the completed fixed GeoLife experiment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "matplotlib_cache"))
sys.path.insert(0, str(ROOT / "src" / "model_training"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from sklearn.metrics.pairwise import rbf_kernel
from threadpoolctl import threadpool_limits

from models import MODES, cv_groups, metrics_from_cm, point_confusion

GROUPS = ["G1", "G2", "G3"]
SIGMAS = [0, 5, 10, 20]
SEEDS = [20261123, 20261124, 20261125, 20261126, 20261127]
SMOOTHERS = ["S0_raw_k1", "S1_centered_soft_k11", "S2_trailing_soft_k11"]
DT_LABELS = ["<=1s", "1-2s", "2-5s", ">5s", "invalid_dt"]
LENGTH_LABELS = ["1", "2-5", "6-10", ">=11"]
META = ["user_id", "feature_block_id", "start_time", "end_time",
        "n_points", "label", "purity", *[f"cnt_{m}" for m in MODES]]
NEAR_INTERCEPT = 1e-6
SCORE_ATOL = 1e-7
COLORS = ["#007F73", "#BC4749", "#4666AD", "#AB7C18", "#666666"]
CAPTION = (
    "Post-hoc diagnostics; fixed clean-trained models and existing test perturbations.\n"
    "Independent Gaussian E/N noise is not calibrated real GPS noise; sigma=0 means no added noise.\n"
    "Score smoothing is not coordinate denoising. Project-defined noise; Endo noise study used KT.\n"
    "Error bars, when shown, are perturbation-seed SD, not user-generalization confidence intervals."
)


def log(message):
    stamp = pd.Timestamp.now(tz="Asia/Shanghai").isoformat()
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / "analysis.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    tmp.replace(path)


def csv(name, data):
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    (HERE / "tables").mkdir(exist_ok=True)
    frame.to_csv(HERE / "tables" / f"{name}.csv", index=False, encoding="utf-8-sig")
    return frame


def read_table(path):
    d = pd.read_csv(path, dtype={"user_id": str})
    if "user_id" in d:
        d["user_id"] = d.user_id.str.zfill(3)
    for col in ("start_time", "end_time"):
        if col in d:
            d[col] = pd.to_datetime(d[col])
    return d


def conditions():
    return [(sigma, seed, f"sigma{sigma:02d}_seed{seed}")
            for sigma in SIGMAS for seed in ([0] if sigma == 0 else SEEDS)]


def sampling_bins(values):
    x = np.asarray(values, float)
    return np.select(
        [~np.isfinite(x) | (x <= 0), x <= 1, x <= 2, x <= 5],
        ["invalid_dt", "<=1s", "1-2s", "2-5s"], default=">5s")


def length_bin(n):
    return "1" if n == 1 else "2-5" if n <= 5 else "6-10" if n <= 10 else ">=11"


def duration_bin(seconds):
    return "<=60s" if seconds <= 60 else "60-180s" if seconds <= 180 else (
        "180-600s" if seconds <= 600 else ">600s")


def quantile(values, q):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if len(x) else np.nan


def partitions(base):
    yield "all", "all", np.ones(len(base), bool)
    for name, col in (("majority_class", "label"), ("linked_group", "linked_group"),
                      ("sampling_bin", "sampling_bin")):
        for value in sorted(base[col].unique()):
            yield name, str(value), base[col].eq(value).to_numpy()


def class_metrics(meta, pred):
    cm = point_confusion(meta, pred, MODES)
    m = metrics_from_cm(cm)
    return cm, {"macro_f1": float(m["macro_f1"]), "accuracy": float(m["accuracy"]),
                **{f"f1_{mode}": float(v) for mode, v in zip(MODES, m["per_class_f1"])}}


def bootstrap_metrics(cms):
    tp = np.diagonal(cms, axis1=-2, axis2=-1)
    den = cms.sum(axis=-1) + cms.sum(axis=-2)
    f1 = np.divide(2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den > 0)
    return f1.mean(axis=-1), tp.sum(axis=-1) / cms.sum(axis=(-2, -1))


def boundary_matches(pred, truth, tol=1):
    pi = np.flatnonzero(np.asarray(pred)[1:] != np.asarray(pred)[:-1]) + 1
    ti = np.flatnonzero(np.asarray(truth)[1:] != np.asarray(truth)[:-1]) + 1
    edges = np.abs(pi[:, None] - ti[None, :]) <= tol
    matched = maximum_bipartite_matching(csr_matrix(edges), perm_type="column")
    return pi, ti, [(int(pi[i]), int(ti[j])) for i, j in enumerate(matched) if j >= 0]


def support_indices(size, smoother):
    i = np.arange(size)
    if smoother == SMOOTHERS[0]:
        return i, i
    if smoother == SMOOTHERS[1]:
        return np.maximum(0, i - 5), np.minimum(size - 1, i + 5)
    if smoother == SMOOTHERS[2]:
        return np.maximum(0, i - 10), i
    raise ValueError(smoother)


def block_positions(base):
    return [g.sort_values("start_time", kind="stable").index.to_numpy()
            for _, g in base.groupby(["user_id", "feature_block_id"], sort=False)]


def run_slices(values):
    v = np.asarray(values)
    starts = np.r_[0, np.flatnonzero(v[1:] != v[:-1]) + 1]
    return list(zip(starts, np.r_[starts[1:], len(v)]))


def protected_files():
    files = {p for p in EXP.rglob("*") if p.is_file()
             and HERE not in p.parents and "__pycache__" not in p.parts
             and "matplotlib_cache" not in p.parts}
    manifest = json.loads((EXP / "manifest.json").read_text(encoding="utf-8"))
    files.update(ROOT / p for p in manifest["files"] if (ROOT / p).is_file())
    return sorted(files)


def hash_inputs(files):
    return {str(p.relative_to(ROOT)): {"sha256": sha(p), "bytes": p.stat().st_size}
            for p in files}


def load_context():
    protocol = json.loads((EXP / "protocol.json").read_text())
    assert protocol["sigmas_m"] == SIGMAS and protocol["seeds"] == SEEDS
    assert protocol["g4"] == "skipped by protocol"
    assert protocol["smoothing"] == SMOOTHERS
    base = read_table(EXP / "feature_cache" / "sigma00_seed0.csv")
    assert len(base) == 17380 and base.n_points.sum() == 860972
    base["linked_group"] = cv_groups(base.user_id, ROOT / "data/splits/report.json")
    base["sampling_bin"] = sampling_bins(base.dt_mean)
    assert base.linked_group.nunique() == 9 and base.user_id.nunique() == 13
    assert base.groupby("feature_block_id").user_id.nunique().max() == 1
    boot = np.load(EXP / "metrics/paired_bootstrap_draws.npz", allow_pickle=True)
    draws, clusters = boot["draws"], boot["clusters"].astype(str)
    assert draws.shape == (5000, 9) and np.all(draws.sum(axis=1) == 9)
    np.testing.assert_array_equal(clusters, pd.unique(base.linked_group))
    models, cols = {}, {}
    for group, dim in zip(GROUPS, (10, 15, 27)):
        config = json.loads((EXP / "models" / f"{group}_config.json").read_text())
        cols[group] = config["feature_columns"]
        assert len(cols[group]) == dim and not {"dt_mean", "dt_max"} & set(cols[group])
        models[group] = joblib.load(EXP / "models" / (
            "full.joblib" if group == "G3" else f"{group}.joblib"))
        m = models[group]
        assert list(m.named_steps) == ["impute", "scale", "rbf", "clf"]
        assert m.named_steps["rbf"].gamma == 1 / dim
        assert m.named_steps["rbf"].components_.shape == (1000, dim)
        assert m.named_steps["clf"].C == 1 and m.named_steps["clf"].random_state == 42
    return base, models, cols, draws, clusters


def aligned(frame, base):
    pd.testing.assert_frame_equal(frame[META], base[META], check_dtype=False)


def feature_drift(base, noisy, model, columns, key):
    raw0 = base[columns].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    raw1 = noisy[columns].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    scale = model.named_steps["scale"].scale_
    records = []
    for kind, value, mask in partitions(base):
        for j, feature in enumerate(columns):
            a, b = raw0[mask, j], raw1[mask, j]
            valid = np.isfinite(a) & np.isfinite(b)
            delta = b[valid] - a[valid]
            records.append({
                **key, "stratum_type": kind, "stratum": value, "feature": feature,
                "windows": int(mask.sum()), "paired_finite_windows": int(valid.sum()),
                "clean_median": quantile(a, .5), "noisy_median": quantile(b, .5),
                "clean_missing_fraction": float((~np.isfinite(a)).mean()),
                "noisy_missing_fraction": float((~np.isfinite(b)).mean()),
                "paired_signed_shift_median": quantile(delta, .5),
                "paired_abs_shift_median": quantile(np.abs(delta), .5),
                "paired_abs_shift_train_sd_median": quantile(np.abs(delta) / scale[j], .5),
                "paired_abs_shift_train_sd_p90": quantile(np.abs(delta) / scale[j], .9),
                "fitted_training_scale": float(scale[j]),
            })
    if key["sigma"] == 0:
        assert all(r["paired_abs_shift_median"] == 0 for r in records
                   if r["paired_finite_windows"])
    return records


def kernel_diagnostics(model, columns, features, saved, base, key, batch=512):
    x = features[columns].replace([np.inf, -np.inf], np.nan)
    z = model.named_steps["scale"].transform(model.named_steps["impute"].transform(x))
    rbf, clf = model.named_steps["rbf"], model.named_steps["clf"]
    expected = saved[[f"score_{c}" for c in model.classes_]].to_numpy(float)
    pred = saved.raw.to_numpy()
    maximum, norm, residual = (np.empty(len(base), float) for _ in range(3))
    max_error = 0.0
    for lo in range(0, len(base), batch):
        hi = min(len(base), lo + batch)
        kernel = rbf_kernel(z[lo:hi], rbf.components_, gamma=rbf.gamma)
        phi = kernel @ rbf.normalization_.T
        scores = clf.decision_function(phi)
        max_error = max(max_error, float(np.max(np.abs(scores - expected[lo:hi]))))
        np.testing.assert_allclose(scores, expected[lo:hi], atol=SCORE_ATOL, rtol=1e-9)
        np.testing.assert_array_equal(model.classes_[scores.argmax(axis=1)], pred[lo:hi])
        maximum[lo:hi] = kernel.max(axis=1)
        norm[lo:hi] = np.linalg.norm(phi, axis=1)
        residual[lo:hi] = np.max(np.abs(scores - clf.intercept_), axis=1)
    winner = str(model.classes_[clf.intercept_.argmax()])
    d = base[["user_id", "feature_block_id", "start_time", "n_points",
              "label", "linked_group", "sampling_bin"]].copy()
    d.insert(0, "window_row", np.arange(len(base)))
    for col, value in key.items():
        d[col] = value
    d["max_basis_rbf_similarity"] = maximum
    d["nystroem_l2_norm"] = norm
    d["max_score_residual_from_intercept"] = residual
    d["near_intercept"] = residual <= NEAR_INTERCEPT
    d["raw_prediction"] = pred
    d["intercept_winner"] = winner
    d.to_parquet(HERE / "window_diagnostics" / f"{key['group']}_{key['condition']}.parquet",
                 index=False)
    records = []
    for kind, value, mask in partitions(base):
        weights = base.loc[mask, "n_points"].to_numpy()
        near = residual[mask] <= NEAR_INTERCEPT
        winner_pred = pred[mask] == winner
        records.append({
            **key, "stratum_type": kind, "stratum": value,
            "windows": int(mask.sum()), "points": int(weights.sum()),
            "max_basis_similarity_median": quantile(maximum[mask], .5),
            "max_basis_similarity_p10": quantile(maximum[mask], .1),
            "representation_norm_median": quantile(norm[mask], .5),
            "score_residual_median": quantile(residual[mask], .5),
            "near_intercept_window_fraction": float(near.mean()),
            "near_intercept_point_fraction": float(np.average(near, weights=weights)),
            "intercept_winner": winner,
            "intercept_winner_prediction_window_fraction": float(winner_pred.mean()),
            "intercept_winner_prediction_point_fraction": float(np.average(
                winner_pred, weights=weights)),
            "near_intercept_predictions_agree_with_intercept_winner": (
                float(winner_pred[near].mean()) if near.any() else np.nan),
            "max_score_reconstruction_error": max_error,
        })
    return records


def support_table(base, blocks):
    records = []
    for ix in blocks:
        d = base.loc[ix]
        starts = d.start_time.to_numpy(dtype="datetime64[ns]")
        ends = d.end_time.to_numpy(dtype="datetime64[ns]")
        for smoother in SMOOTHERS:
            lo, hi = support_indices(len(ix), smoother)
            for i, row in enumerate(ix):
                records.append({
                    "window_row": row, "user_id": base.user_id.iloc[row],
                    "feature_block_id": base.feature_block_id.iloc[row],
                    "linked_group": base.linked_group.iloc[row], "smoother": smoother,
                    "available_windows": int(hi[i] - lo[i] + 1),
                    "support_start": starts[lo[i]], "support_end": ends[hi[i]],
                    "support_span_s": float((ends[hi[i]] - starts[lo[i]]) / np.timedelta64(1, "s")),
                    "future_window_wait_s": float((ends[hi[i]] - ends[i]) / np.timedelta64(1, "s")),
                })
    return pd.DataFrame(records)


def sequence_diagnostics(saved, base, blocks, key):
    run_rows, boundary_rows, error_rows, summary_rows = [], [], [], []
    total_truth = total_pred = total_hit = 0
    for smoother in SMOOTHERS:
        true_n = pred_n = hit_n = 0
        signed_errors, availability = [], []
        for ix in blocks:
            d = base.loc[ix]
            truth = d.label.to_numpy()
            pred = saved.loc[ix, smoother].to_numpy()
            raw = saved.loc[ix, SMOOTHERS[0]].to_numpy()
            starts = d.start_time.to_numpy(dtype="datetime64[ns]")
            ends = d.end_time.to_numpy(dtype="datetime64[ns]")
            _, support_hi = support_indices(len(ix), smoother)
            identity = {"user_id": d.user_id.iloc[0],
                        "linked_group": d.linked_group.iloc[0],
                        "feature_block_id": d.feature_block_id.iloc[0]}
            slices = run_slices(truth)
            for run_i, (a, b) in enumerate(slices):
                correct = pred[a:b] == truth[a]
                raw_correct = raw[a:b] == truth[a]
                duration = float((ends[b - 1] - starts[a]) / np.timedelta64(1, "s"))
                flanked = (run_i > 0 and run_i + 1 < len(slices)
                           and truth[slices[run_i - 1][0]] == truth[slices[run_i + 1][0]])
                run_rows.append({
                    **key, **identity, "smoother": smoother, "run_start_row": int(ix[a]),
                    "run_end_row": int(ix[b - 1]), "mode": str(truth[a]),
                    "start_time": starts[a], "end_time": ends[b - 1],
                    "n_windows": b - a, "n_points": int(d.n_points.iloc[a:b].sum()),
                    "duration_s": duration, "length_bin": length_bin(b - a),
                    "duration_bin": duration_bin(duration), "flanked_same_mode": flanked,
                    "absent": not correct.any(), "raw_present": bool(raw_correct.any()),
                    "raw_all_correct": bool(raw_correct.all()),
                    "newly_absent_from_raw": bool(raw_correct.any() and not correct.any()),
                    "all_raw_correct_then_absent": bool(raw_correct.all() and not correct.any()),
                    "window_majority_agreement": float(correct.mean()),
                })
            pi, ti, pairs = boundary_matches(pred, truth)
            mapping = {t: p for p, t in pairs}
            true_n += len(ti)
            pred_n += len(pi)
            hit_n += len(pairs)
            for t in ti:
                p = mapping.get(int(t))
                if p is None:
                    boundary_rows.append({
                        **key, **identity, "smoother": smoother,
                        "true_boundary_row": int(ix[t]), "true_boundary_time": starts[t],
                        "matched": False, "true_from": truth[t - 1], "true_to": truth[t],
                    })
                else:
                    error = float((starts[p] - starts[t]) / np.timedelta64(1, "s"))
                    latency = float((ends[support_hi[p]] - starts[t]) / np.timedelta64(1, "s"))
                    signed_errors.append(error)
                    availability.append(latency)
                    boundary_rows.append({
                        **key, **identity, "smoother": smoother,
                        "true_boundary_row": int(ix[t]), "true_boundary_time": starts[t],
                        "matched": True, "pred_boundary_row": int(ix[p]),
                        "pred_boundary_time": starts[p], "offset_windows": int(p - t),
                        "signed_boundary_error_s": error,
                        "score_available_latency_s": latency,
                        "true_from": truth[t - 1], "true_to": truth[t],
                        "pred_from": pred[p - 1], "pred_to": pred[p],
                        "direction_correct": bool(
                            pred[p - 1] == truth[t - 1] and pred[p] == truth[t]),
                    })
            wrong = pred != truth
            for a, b in run_slices(wrong):
                if wrong[a]:
                    error_rows.append({
                        **key, **identity, "smoother": smoother,
                        "start_row": int(ix[a]), "end_row": int(ix[b - 1]),
                        "n_windows": b - a, "n_points": int(d.n_points.iloc[a:b].sum()),
                        "duration_s": float((ends[b - 1] - starts[a]) / np.timedelta64(1, "s")),
                    })
        assert true_n == 529
        precision = hit_n / pred_n if pred_n else 0.0
        recall = hit_n / true_n
        summary_rows.append({
            **key, "smoother": smoother, "true_transitions": true_n,
            "pred_transitions": pred_n, "matched_transitions": hit_n,
            "missed_transitions": true_n - hit_n,
            "unmatched_pred_transitions": pred_n - hit_n,
            "transition_precision": precision, "transition_recall": recall,
            "transition_f1": 2 * precision * recall / (precision + recall)
            if precision + recall else 0.0,
            "matched_signed_error_s_median": quantile(signed_errors, .5),
            "matched_abs_error_s_median": quantile(np.abs(signed_errors), .5),
            "matched_availability_latency_s_median": quantile(availability, .5),
            "matched_availability_latency_s_p90": quantile(availability, .9),
        })
        total_truth += true_n
        total_pred += pred_n
        total_hit += hit_n
    assert total_truth == 3 * 529 and total_hit <= total_pred
    return run_rows, boundary_rows, error_rows, summary_rows


def aggregate_runs(runs):
    rows = []
    keys = ["sigma", "seed", "condition", "smoother"]
    for key, d in runs.groupby(keys, sort=False):
        identity = dict(zip(keys, key))
        for dimension in ("all", "length_bin", "duration_bin", "mode", "linked_group"):
            groups = [("all", d)] if dimension == "all" else d.groupby(dimension, sort=False)
            for value, sub in groups:
                raw_present = int(sub.raw_present.sum())
                raw_all = int(sub.raw_all_correct.sum())
                rows.append({
                    **identity, "stratum_type": dimension, "stratum": value,
                    "runs": len(sub), "absent_runs": int(sub.absent.sum()),
                    "absent_run_fraction": float(sub.absent.mean()),
                    "raw_present_runs": raw_present,
                    "newly_absent_runs": int(sub.newly_absent_from_raw.sum()),
                    "newly_absent_given_raw_present_fraction": (
                        float(sub.newly_absent_from_raw.sum() / raw_present)
                        if raw_present else np.nan),
                    "raw_all_correct_runs": raw_all,
                    "all_raw_correct_then_absent_runs": int(sub.all_raw_correct_then_absent.sum()),
                    "absent_given_raw_all_correct_fraction": (
                        float(sub.all_raw_correct_then_absent.sum() / raw_all)
                        if raw_all else np.nan),
                    "window_majority_agreement_mean": float(sub.window_majority_agreement.mean()),
                })
    return pd.DataFrame(rows)


def subgroup_metrics(saved, base, key, clusters, draws, main_metrics):
    metric_rows, confusion_rows, share_rows, cm_by_smoother, recon = [], [], [], {}, []
    types = [("all", [("all", np.ones(len(base), bool))]),
             ("linked_group", [(c, base.linked_group.eq(c).to_numpy()) for c in clusters]),
             ("user_id", [(u, base.user_id.eq(u).to_numpy()) for u in sorted(base.user_id.unique())]),
             ("sampling_bin", [(c, base.sampling_bin.eq(c).to_numpy())
                               for c in DT_LABELS if base.sampling_bin.eq(c).any()])]
    for smoother in SMOOTHERS:
        pred = saved[smoother].to_numpy()
        overall_cm = point_confusion(base, pred, MODES)
        cluster_cms = np.stack([point_confusion(base.loc[base.linked_group.eq(c)],
                                               pred[base.linked_group.eq(c)], MODES)
                                for c in clusters])
        cm_by_smoother[smoother] = cluster_cms
        for kind, masks in types:
            for value, mask in masks:
                sub = base.loc[mask]
                cm, metrics = class_metrics(sub, pred[mask])
                row = {**key, "smoother": smoother, "stratum_type": kind,
                       "stratum": value, "windows": len(sub),
                       "points": int(sub.n_points.sum()), **metrics}
                metric_rows.append(row)
                for j, mode in enumerate(MODES):
                    support = int(cm[j].sum())
                    count = int(cm[:, j].sum())
                    share_rows.append({
                        **key, "smoother": smoother, "stratum_type": kind, "stratum": value,
                        "mode": mode, "true_points": support, "predicted_points": count,
                        "total_points": int(cm.sum()), "true_point_fraction": support / cm.sum(),
                        "prediction_point_fraction": count / cm.sum(),
                        "precision": float(cm[j, j] / count) if count else 0.0,
                        "recall": float(cm[j, j] / support) if support else 0.0,
                        "f1": metrics[f"f1_{mode}"],
                        "class_present_in_truth": bool(support),
                    })
                if kind == "all":
                    ref = main_metrics[
                        main_metrics.group.eq(key["group"]) &
                        main_metrics.condition.eq(key["condition"]) &
                        main_metrics.smoother.eq(smoother)].iloc[0]
                    np.testing.assert_allclose(
                        [metrics["macro_f1"], metrics["accuracy"]],
                        [ref.macro_f1, ref.accuracy], atol=1e-12)
                    recon.append({**key, "smoother": smoother,
                                  "macro_f1_error": metrics["macro_f1"] - ref.macro_f1,
                                  "accuracy_error": metrics["accuracy"] - ref.accuracy})
                    for i, true_mode in enumerate(MODES):
                        for j, mode in enumerate(MODES):
                            confusion_rows.append({**key, "smoother": smoother,
                                                   "true_mode": true_mode, "pred_mode": mode,
                                                   "points": int(cm[i, j]),
                                                   "true_class_fraction": (
                                                       float(cm[i, j] / cm[i].sum())
                                                       if cm[i].sum() else np.nan)})
        for j, c in enumerate(clusters):
            cm = overall_cm - cluster_cms[j]
            m = metrics_from_cm(cm)
            metric_rows.append({
                **key, "smoother": smoother, "stratum_type": "leave_one_linked_group_out",
                "stratum": c, "windows": int(base.linked_group.ne(c).sum()),
                "points": int(cm.sum()), "macro_f1": float(m["macro_f1"]),
                "accuracy": float(m["accuracy"]),
                **{f"f1_{mode}": float(v) for mode, v in zip(MODES, m["per_class_f1"])},
            })
    distributions = {s: bootstrap_metrics(np.einsum("bg,gij->bij", draws, cms))
                     for s, cms in cm_by_smoother.items()}
    return metric_rows, confusion_rows, share_rows, distributions, recon


def seed_summary(frame, keys, values):
    result = frame.groupby(keys, sort=False)[values].agg(["mean", "std", "count"])
    result.columns = [f"{a}_seed_{b}" for a, b in result.columns]
    result = result.reset_index()
    for col in result.columns:
        if col.endswith("_seed_std"):
            result[col] = result[col].fillna(0)
    return result


def paired_subgroup_changes(metrics):
    keys = ["group", "smoother", "stratum_type", "stratum"]
    baseline = metrics[metrics.sigma.eq(0)][keys + ["macro_f1", "accuracy"]].rename(
        columns={"macro_f1": "clean_macro_f1", "accuracy": "clean_accuracy"})
    result = metrics.merge(baseline, on=keys, how="left", validate="many_to_one")
    result["macro_f1_delta"] = result.macro_f1 - result.clean_macro_f1
    result["accuracy_delta"] = result.accuracy - result.clean_accuracy
    assert result.clean_macro_f1.notna().all()
    return result


def bootstrap_table(distributions):
    rows = []
    for group in GROUPS:
        for smoother in SMOOTHERS:
            clean = distributions[(group, 0, 0)][smoother]
            for sigma in SIGMAS:
                seeds = [0] if sigma == 0 else SEEDS
                row = {"group": group, "smoother": smoother, "sigma": sigma,
                       "seed_count": len(seeds), "bootstrap_draws": 5000}
                for j, metric in enumerate(("macro_f1", "accuracy")):
                    values = np.mean([distributions[(group, sigma, seed)][smoother][j]
                                      for seed in seeds], axis=0)
                    delta = values - clean[j]
                    row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = np.percentile(
                        values, [2.5, 97.5])
                    row[f"{metric}_delta_ci_low"], row[f"{metric}_delta_ci_high"] = np.percentile(
                        delta, [2.5, 97.5])
                rows.append(row)
    result = pd.DataFrame(rows)
    ref = pd.read_csv(EXP / "metrics/stage5_bootstrap_all.csv")
    keys = ["group", "smoother", "sigma"]
    joined = result.merge(ref, on=keys, suffixes=("_new", "_ref"), validate="one_to_one")
    for col in [c for c in result if c.endswith(("_ci_low", "_ci_high"))]:
        np.testing.assert_allclose(joined[col + "_new"], joined[col + "_ref"], atol=1e-12)
    return result


def save_figure(fig, name):
    fig.text(.02, .015, CAPTION, va="bottom", fontsize=7, color="#444444")
    fig.savefig(HERE / "figures" / f"{name}.png", dpi=180, facecolor="white")
    fig.savefig(HERE / "figures" / f"{name}.pdf", facecolor="white")
    plt.close(fig)


def draw_curves(ax, frame, x, y, category, order, log_y=False):
    for color, label in zip(COLORS, order):
        d = frame[frame[category].eq(label)]
        if d.empty:
            continue
        q = d.groupby(x)[y].agg(["mean", "std"]).fillna(0)
        ax.errorbar(q.index, q["mean"], yerr=q["std"], color=color, marker="o",
                    markersize=4, capsize=3, label=str(label))
    ax.set_xticks(SIGMAS)
    ax.set_xlabel("Added per-axis sigma (m)")
    ax.grid(alpha=.2)
    if log_y:
        ax.set_yscale("log")


def plots(drift, kernels, subgroups, run_summary, support, boundaries, shares, user_groups):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    q = drift[drift.stratum_type.eq("all") & drift.sigma.gt(0)]
    matrix = q.pivot_table(index="feature", columns="sigma",
                           values="paired_abs_shift_train_sd_median", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 10))
    fig.subplots_adjust(left=.32, right=.88, bottom=.16, top=.94)
    im = ax.imshow(np.log1p(matrix.to_numpy()), aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(matrix)), matrix.index)
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns)
    ax.set_xlabel("Added per-axis sigma (m)")
    ax.set_title("Feature shifts relative to fitted training scales")
    fig.colorbar(im, ax=ax, label="log(1 + mean seed median absolute shift / training SD)")
    save_figure(fig, "01_feature_drift")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.subplots_adjust(left=.08, right=.98, bottom=.25, top=.88, wspace=.32)
    q = subgroups[subgroups.group.eq("G3") & subgroups.smoother.eq(SMOOTHERS[0]) &
                  subgroups.stratum_type.eq("sampling_bin")]
    draw_curves(axes[0], q, "sigma", "macro_f1", "stratum", DT_LABELS)
    axes[0].set(title="G3 point Macro-F1 by original sampling interval", ylabel="Macro-F1")
    axes[0].legend(fontsize=8)
    q = drift[drift.stratum_type.eq("sampling_bin") & drift.feature.eq("speed_mean")]
    draw_curves(axes[1], q, "sigma", "paired_abs_shift_median", "stratum", DT_LABELS)
    axes[1].set(title="Paired speed-mean shifts (window-equal)", ylabel="Median absolute shift (m/s)")
    save_figure(fig, "02_sampling_interval")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    fig.subplots_adjust(left=.06, right=.98, bottom=.25, top=.86, wspace=.35)
    q = kernels[kernels.stratum_type.eq("all")]
    for ax, col, title in zip(
            axes, ["max_basis_similarity_median", "near_intercept_point_fraction",
                   "intercept_winner_prediction_point_fraction"],
            ["Maximum basis RBF similarity", "Scores nearly equal intercepts",
             "Prediction share of intercept-winning class"]):
        draw_curves(ax, q, "sigma", col, "group", GROUPS,
                    log_y=(col == "max_basis_similarity_median"))
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Median similarity" if col.endswith("median") else "Point-weighted fraction")
    axes[0].legend()
    save_figure(fig, "03_rbf_collapse")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(left=.08, right=.98, bottom=.25, top=.83, wspace=.3)
    for ax, sigma in zip(axes, (0, 20)):
        q = run_summary[run_summary.stratum_type.eq("length_bin") &
                        run_summary.sigma.eq(sigma)]
        for offset, smoother, color in zip((-.24, 0, .24), SMOOTHERS, COLORS):
            sub = q[q.smoother.eq(smoother)].groupby("stratum").absent_run_fraction.agg(
                ["mean", "std"]).reindex(LENGTH_LABELS)
            ax.bar(np.arange(4) + offset, sub["mean"], width=.23,
                   yerr=sub["std"].fillna(0), capsize=2, color=color,
                   label=smoother.split("_")[1])
        ax.set(xticks=np.arange(4), xticklabels=LENGTH_LABELS,
               xlabel="True run length (windows)", ylabel="Fraction with mode absent in run",
               title=f"G3: sigma={sigma}m")
        ax.set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    fig.suptitle("Run absence is a sequence diagnostic, not point-level F1", y=.95)
    save_figure(fig, "04_short_mode_loss")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.subplots_adjust(left=.12, right=.98, bottom=.24, top=.88, wspace=.5)
    q = subgroups[subgroups.group.eq("G3") & subgroups.smoother.eq(SMOOTHERS[0]) &
                  subgroups.stratum_type.eq("linked_group")]
    order = user_groups.sort_values("points", ascending=False).linked_group.tolist()
    for off, sigma, color in ((-.16, 0, COLORS[0]), (.16, 20, COLORS[1])):
        a = q[q.sigma.eq(sigma)].groupby("stratum").macro_f1.agg(["mean", "std"]).reindex(order)
        axes[0].barh(np.arange(len(order)) + off, a["mean"], height=.3,
                     xerr=a["std"].fillna(0), capsize=2, color=color, label=f"sigma={sigma}m")
    axes[0].set(yticks=np.arange(len(order)), yticklabels=order,
                xlabel="Fixed-five-class point Macro-F1", title="G3 raw by linked group")
    axes[0].legend(fontsize=8)
    g = user_groups.set_index("linked_group").loc[order]
    axes[1].barh(np.arange(len(order)), g.point_fraction, color=COLORS[2])
    axes[1].set(yticks=np.arange(len(order)), yticklabels=order,
                xlabel="Fraction of all test points", title="Linked-group sample weights")
    save_figure(fig, "05_user_group_heterogeneity")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.subplots_adjust(left=.08, right=.98, bottom=.25, top=.86, wspace=.35)
    draw_curves(axes[0], boundaries, "sigma", "transition_recall", "smoother", SMOOTHERS)
    axes[0].set(title="G3: all 529 true boundaries", ylabel="Boundary recall")
    draw_curves(axes[1], boundaries, "sigma", "matched_availability_latency_s_median",
                "smoother", SMOOTHERS)
    axes[1].set(title="Matched boundaries only (+/-1 window)", ylabel="Median score-availability latency (s)")
    axes[0].legend(["raw", "centered", "trailing"], fontsize=8)
    save_figure(fig, "06_boundary_latency_and_recall")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    assert args.batch_size > 0
    for folder in ("tables", "figures", "window_diagnostics"):
        (HERE / folder).mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    files = protected_files()
    log(f"Hashing {len(files)} protected primary/input files before analysis")
    before = hash_inputs(files)
    json_write(HERE / "input_manifest.json", {
        "created_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "analysis_spec_sha256": sha(HERE / "ANALYSIS_SPEC.md"),
        "analysis_code_sha256": sha(Path(__file__)), "inputs": before,
        "python": sys.version, "executable": sys.executable,
        "dependencies": {p: importlib.metadata.version(p) for p in
                         ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "pyarrow", "matplotlib")},
        "batch_size": args.batch_size, "near_intercept_tolerance": NEAR_INTERCEPT,
        "score_reconstruction_atol": SCORE_ATOL,
        "post_hoc": True, "model_fitting": False,
    })
    base, models, columns, draws, clusters = load_context()
    blocks = block_positions(base)
    main_metrics = pd.read_csv(EXP / "metrics/all_metrics.csv")
    support = csv("smoothing_window_support", support_table(base, blocks))
    groups = base.groupby("linked_group", sort=False).agg(
        points=("n_points", "sum"), windows=("n_points", "size"),
        users=("user_id", lambda x: ",".join(sorted(x.unique())))).reset_index()
    groups["point_fraction"] = groups.points / base.n_points.sum()
    csv("linked_group_sample_sizes", groups)
    csv("sampling_stratum_sizes", base.groupby("sampling_bin", sort=False).agg(
        windows=("n_points", "size"), points=("n_points", "sum"),
        dt_mean_median_s=("dt_mean", "median")).reset_index())
    csv("model_intercepts", [
        {"group": g, "mode": str(c), "intercept": float(b),
         "is_intercept_winner": bool(b == models[g].named_steps["clf"].intercept_.max())}
        for g in GROUPS for c, b in zip(models[g].classes_,
                                        models[g].named_steps["clf"].intercept_)])
    drift_rows, kernel_rows, subgroup_rows, confusion_rows, share_rows = [], [], [], [], []
    run_rows, boundary_rows, error_rows, boundary_summary, reconstruction = [], [], [], [], []
    distributions = {}
    with threadpool_limits(limits=4):
        for condition_i, (sigma, seed, name) in enumerate(conditions()):
            t = time.monotonic()
            key = {"sigma": sigma, "seed": seed, "condition": name}
            features = read_table(EXP / "feature_cache" / f"{name}.csv")
            aligned(features, base)
            np.testing.assert_allclose(features[["dt_mean", "dt_max", "duration_s"]],
                                       base[["dt_mean", "dt_max", "duration_s"]],
                                       equal_nan=True, rtol=0, atol=0)
            drift_rows.extend(feature_drift(base, features, models["G3"], columns["G3"], key))
            for group in GROUPS:
                model_key = {**key, "group": group}
                saved = read_table(EXP / "predictions" / f"{group}_{name}_smoothed.csv")
                aligned(saved, base)
                raw = read_table(EXP / "predictions" / f"{group}_{name}.csv")
                aligned(raw, base)
                pd.testing.assert_frame_equal(
                    saved[[f"score_{m}" for m in MODES]], raw[[f"score_{m}" for m in MODES]])
                np.testing.assert_array_equal(saved[SMOOTHERS[0]], raw.raw)
                kernel_rows.extend(kernel_diagnostics(
                    models[group], columns[group], features, raw, base, model_key, args.batch_size))
                metrics, confusion, shares, dist, recon = subgroup_metrics(
                    saved, base, model_key, clusters, draws, main_metrics)
                subgroup_rows.extend(metrics)
                confusion_rows.extend(confusion)
                share_rows.extend(shares)
                reconstruction.extend(recon)
                distributions[(group, sigma, seed)] = dist
                if group == "G3":
                    runs, boundaries, errors, summary = sequence_diagnostics(
                        saved, base, blocks, key)
                    run_rows.extend(runs)
                    boundary_rows.extend(boundaries)
                    error_rows.extend(errors)
                    boundary_summary.extend(summary)
                log(f"{name} {group}: frozen scores reconstructed and metrics reconciled")
            json_write(HERE / "checkpoint.json", {
                "status": "running", "conditions_complete": condition_i + 1,
                "total_conditions": 16, "last_condition": name,
                "elapsed_seconds": time.monotonic() - started})
            log(f"Condition {condition_i + 1}/16 completed in {time.monotonic() - t:.1f}s")
    drift = csv("feature_drift", drift_rows)
    kernels = csv("kernel_summary", kernel_rows)
    subgroup = csv("subgroup_metrics", paired_subgroup_changes(pd.DataFrame(subgroup_rows)))
    csv("subgroup_seed_summary", seed_summary(
        subgroup, ["group", "sigma", "smoother", "stratum_type", "stratum"],
        ["macro_f1", "accuracy", "macro_f1_delta", "accuracy_delta"]))
    csv("confusion_matrices", confusion_rows)
    shares = csv("class_metrics_and_prediction_shares", share_rows)
    csv("class_seed_summary", seed_summary(
        shares, ["group", "sigma", "smoother", "stratum_type", "stratum", "mode"],
        ["f1", "precision", "recall", "prediction_point_fraction"]))
    csv("kernel_seed_summary", seed_summary(
        kernels, ["group", "sigma", "stratum_type", "stratum"],
        ["max_basis_similarity_median", "representation_norm_median",
         "near_intercept_point_fraction", "intercept_winner_prediction_point_fraction"]))
    runs = csv("true_run_diagnostics", run_rows)
    run_summary = csv("run_length_summary", aggregate_runs(runs))
    csv("run_seed_summary", seed_summary(
        run_summary, ["sigma", "smoother", "stratum_type", "stratum"],
        ["absent_run_fraction", "newly_absent_given_raw_present_fraction",
         "absent_given_raw_all_correct_fraction", "window_majority_agreement_mean"]))
    csv("boundary_matches", boundary_rows)
    boundaries = csv("boundary_summary", boundary_summary)
    csv("boundary_seed_summary", seed_summary(
        boundaries, ["sigma", "smoother"],
        ["transition_recall", "transition_f1", "matched_signed_error_s_median",
         "matched_availability_latency_s_median"]))
    errors = csv("consecutive_error_runs", error_rows)
    csv("consecutive_error_summary", errors.groupby(
        ["sigma", "seed", "condition", "smoother"], sort=False).agg(
            error_runs=("n_windows", "size"), error_run_windows_median=("n_windows", "median"),
            error_run_windows_max=("n_windows", "max"),
            error_run_duration_s_median=("duration_s", "median"),
            error_run_duration_s_p90=("duration_s", lambda x: x.quantile(.9)),
            error_run_duration_s_max=("duration_s", "max")).reset_index())
    csv("metric_reconstruction_checks", reconstruction)
    bootstrap = csv("bootstrap_reproduction", bootstrap_table(distributions))
    expected_boundaries = main_metrics[main_metrics.group.eq("G3")].merge(
        boundaries, on=["sigma", "seed", "condition", "smoother"], suffixes=("_old", "_new"))
    assert len(expected_boundaries) == 48
    for col in ("true_transitions", "pred_transitions", "matched_transitions",
                "transition_precision", "transition_recall", "transition_f1"):
        np.testing.assert_allclose(expected_boundaries[col + "_old"],
                                   expected_boundaries[col + "_new"], atol=1e-12)
    plots(drift, kernels, subgroup, run_summary, support, boundaries, shares, groups)
    log("Rehashing protected artifacts and original inputs")
    after = hash_inputs(files)
    changed = [path for path in before if before[path] != after[path]]
    assert not changed, changed
    audit = {
        "passed": True, "completed_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "conditions": 16, "frozen_models": 3, "kernel_window_rows": 48 * len(base),
        "kernel_parquet_files": len(list((HERE / "window_diagnostics").glob("*.parquet"))),
        "reconciled_primary_metric_rows": len(reconstruction),
        "reconciled_boundary_rows": len(boundaries),
        "reproduced_bootstrap_rows": len(bootstrap), "bootstrap_draws": len(draws),
        "protected_files": len(before), "protected_files_changed": changed,
        "training_performed": False, "protocol_changed": False,
        "figures_png": len(list((HERE / "figures").glob("*.png"))),
        "figure_visual_review": "pending",
    }
    json_write(HERE / "audit.json", audit)
    json_write(HERE / "checkpoint.json", {"status": "complete", **audit})
    log(json.dumps(audit))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log(traceback.format_exc())
        json_write(HERE / "failure.json", {
            "failed_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
            "traceback": traceback.format_exc()})
        raise
