"""Exploratory extensions; never writes the frozen experiment or reference."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
import traceback
import urllib.request
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(ROOT / "src/model_training"))
from run_experiment import (
    EARTH_RADIUS_M, FEATURE_GROUPS, META_COLS, MODEL_NAME, MODES, SEEDS,
    SMOOTHERS, assign_windows, build_features, condition_name, cv_groups,
    df_hash, isolated_flips, load_tables, model_specs, one_to_one_transitions,
    prepare, read_points, score_aligned, smooth_soft, trailing_soft, window_features,
)

sys.path.insert(0, str(EXP / "paired_analysis"))
from build_analysis import boundary_metrics, confusion, point_metrics, safe_ratio

TAU = 30.0
SIGMAS = (0, 5, 10, 20)
SCORE_COLS = [f"score_{m}" for m in MODES]
PRED_META = [
    "user_id", "feature_block_id", "start_time", "end_time", "n_points",
    "label", "purity", *[f"cnt_{m}" for m in MODES],
]
METRICS = [
    "macro_f1", "accuracy", "transition_precision", "transition_recall",
    "transition_f1", "wrong_isolated_flips_per100_windows",
    "prediction_change_from_clean_point_fraction",
]
START = time.monotonic()
DEADLINE = START + 7 * 3600
STATE = {}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
    temp.replace(path)


def checkpoint(stage, **kwargs):
    STATE.update(stage=stage, updated_at=pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
                 elapsed_seconds=time.monotonic() - START, **kwargs)
    write_json(HERE / "checkpoint.json", STATE)
    print(f"[{STATE['updated_at']}] {stage}: {kwargs}", flush=True)


def check_deadline():
    if time.monotonic() >= DEADLINE:
        checkpoint("deadline", status="stopped")
        raise TimeoutError("Supplemental seven-hour deadline reached")


def protected_files():
    found = []
    for folder in ("data", "src", "tests", "docs", "experiments"):
        found.extend(p for p in (ROOT / folder).rglob("*") if p.is_file()
                     and HERE not in p.parents and "__pycache__" not in p.parts
                     and "matplotlib_cache" not in p.parts and p.suffix != ".pyc"
                     and p != ROOT / "docs/aigc_log_part2.md")
    found.extend(ROOT / p for p in ("README.md", "CLAUDE.md", "pyproject.toml", "uv.lock"))
    return sorted(found)


def initialize():
    global DEADLINE
    for folder in ("models", "feature_cache", "predictions", "tables", "figures",
                   "literature", "diagnostics"):
        (HERE / folder).mkdir(exist_ok=True)
    protocol_sha = sha(HERE / "PROTOCOL.md")
    code_sha = sha(Path(__file__))
    path = HERE / "input_manifest.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        DEADLINE = START + max(0, 7 * 3600 - (
            pd.Timestamp.now(tz="Asia/Shanghai") - pd.Timestamp(saved["created_at"])).total_seconds())
        assert saved["protocol_sha256"] == protocol_sha, "Protocol changed after initialization"
        assert saved["code_sha256"] == code_sha, "Runner changed; review/resume explicitly"
        for rel, digest in saved["files"].items():
            assert sha(ROOT / rel) == digest, f"Protected input changed: {rel}"
        return
    files = {p.relative_to(ROOT).as_posix(): sha(p) for p in protected_files()}
    write_json(path, {
        "created_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "exploratory_after_inspecting_original_test": True,
        "protocol_sha256": protocol_sha, "code_sha256": code_sha, "files": files,
        "python": sys.version, "dependencies": {m: importlib.metadata.version(m)
             for m in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "pyarrow")},
    })
    checkpoint("initialized", protected_files=len(files))


def retrieve_sources():
    sources = [
        ("gpml_chapter4.pdf", "https://gaussianprocess.org/gpml/chapters/RW4.pdf"),
        ("nystroem.html", "https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html"),
        ("scientific_skills.html", "https://arxiv.org/abs/2609.00065"),
    ]
    records = []
    for filename, url in sources:
        path = HERE / "literature" / filename
        try:
            if not path.exists():
                request = urllib.request.Request(url, headers={"User-Agent": "GeoLife-course-audit/1.0"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    body = response.read()
                    path.write_bytes(body)
            records.append({"url": url, "file": filename, "bytes": path.stat().st_size,
                            "sha256": sha(path), "retrieved": True})
        except Exception as exc:
            records.append({"url": url, "retrieved": False, "error": str(exc)})
    write_json(HERE / "literature/retrieval.json", records)
    checkpoint("sources_recorded", retrieved=sum(r["retrieved"] for r in records))


def ou_from_innovations(z, seconds, blocks, tau=TAU):
    """Exact stationary exponential process on irregular, ordered block times."""
    z, seconds, blocks = np.asarray(z, float), np.asarray(seconds, float), np.asarray(blocks)
    assert z.ndim == 2 and z.shape[1] == 2
    assert len(z) == len(seconds) == len(blocks) and tau > 0
    out = np.empty_like(z)
    for start, stop in zip(
        np.r_[0, np.flatnonzero(blocks[1:] != blocks[:-1]) + 1],
        np.r_[np.flatnonzero(blocks[1:] != blocks[:-1]) + 1, len(z)],
    ):
        dt = np.diff(seconds[start:stop])
        assert (dt >= 0).all(), "Point time decreases inside a block"
        a = np.exp(-dt / tau)
        innovation_sd = np.sqrt(-np.expm1(-2 * dt / tau))
        out[start] = z[start]
        # A direct recurrence avoids unstable division by exp(-cumulative time).
        for j in range(start + 1, stop):
            out[j] = a[j - start - 1] * out[j - 1] + innovation_sd[j - start - 1] * z[j]
    return out


def ou_field(points, seed):
    uid = int(points.user_id.iloc[0])
    rng = np.random.default_rng(np.random.SeedSequence([seed, uid]))
    z = rng.standard_normal((len(points), 2))
    seconds = (points.timestamp - points.timestamp.iloc[0]).dt.total_seconds().to_numpy()
    q = ou_from_innovations(z, seconds, points.feature_block_id.to_numpy())
    return q


def displace(points, standard_field, sigma):
    result = points.copy()
    if sigma == 0:
        return result
    east, north = (standard_field * sigma).T
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
    return result


def read_table(path):
    d = pd.read_csv(path, dtype={"user_id": str})
    d.user_id = d.user_id.str.zfill(3)
    for col in ("start_time", "end_time"):
        d[col] = pd.to_datetime(d[col])
    return d


def rebuild(points, field, sigma):
    noisy = displace(points, field, sigma)
    invariant = ["user_id", "trip_id", "feature_block_id", "timestamp", "mode"]
    pd.testing.assert_frame_equal(noisy[invariant], points[invariant])
    assert np.isfinite(noisy[["latitude", "longitude"]].to_numpy(float)).all()
    old_ids = assign_windows(points, 50, cut_at_label_change=False)
    noisy = build_features(noisy)
    noisy["window_id"] = assign_windows(noisy, 50, cut_at_label_change=False)
    np.testing.assert_array_equal(noisy.window_id, old_ids)
    np.testing.assert_allclose(noisy.delta_t_s, points.delta_t_s, equal_nan=True)
    return noisy, window_features(noisy)


def save_cached_frame(path, frame, extra):
    frame.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    write_json(path.with_suffix(".json"), {
        **extra, "file_sha256": sha(path),
        "code_sha256": sha(Path(__file__)), "protocol_sha256": sha(HERE / "PROTOCOL.md"),
        "input_manifest_sha256": sha(HERE / "input_manifest.json"),
        "created_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
    })


def valid_saved(path):
    manifest = path.with_suffix(".json")
    if not path.exists() or not manifest.exists():
        return False
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert saved["code_sha256"] == sha(Path(__file__))
    assert saved["protocol_sha256"] == sha(HERE / "PROTOCOL.md")
    assert saved["input_manifest_sha256"] == sha(HERE / "input_manifest.json")
    assert saved["file_sha256"] == sha(path), f"Corrupt artifact: {path}"
    return True


def zero_check(base, cols):
    uid = sorted(base.user_id.unique())[0]
    p = read_points(ROOT / f"data/processed/quality_features/{uid}.parquet")
    p = p.sort_values(["feature_block_id", "timestamp"], kind="stable").reset_index(drop=True)
    field = ou_field(p, SEEDS[0])
    pd.testing.assert_frame_equal(displace(p, field, 0), p)
    _, rebuilt = rebuild(p, field, 0)
    expected = base[base.user_id == uid].reset_index(drop=True)
    pd.testing.assert_frame_equal(rebuilt[META_COLS], expected[META_COLS], check_dtype=False)
    np.testing.assert_allclose(rebuilt[cols], expected[cols], atol=1e-7, rtol=1e-10, equal_nan=True)
    noisy, _ = rebuild(p, field, 20)
    poisoned = displace(p, field, 20)
    motion_cols = [
        "distance_m", "speed_mps", "acceleration_mps2", "jerk_mps3",
        "heading_change_rad", "heading_change_rate_rad_s",
    ]
    poisoned[motion_cols] = 1234567.0
    recovered = build_features(poisoned)
    for col in motion_cols:
        np.testing.assert_allclose(recovered[col], noisy[col], equal_nan=True)
    write_json(HERE / "diagnostics/zero_and_recompute_check.json", {
        "passed": True, "user": uid, "windows": len(expected), "points": len(p),
        "zero_coordinate_identity": True, "zero_features_match": True,
        "all_stale_motion_columns_overwritten": True,
    })
    checkpoint("zero_check_passed", user=uid)


def generate_ou_caches(base, cols):
    for seed in SEEDS:
        paths = {sigma: HERE / "feature_cache" / f"OU30_{condition_name(sigma, seed)}.csv"
                 for sigma in SIGMAS[1:]}
        missing = [sigma for sigma, path in paths.items() if not valid_saved(path)]
        if not missing:
            continue
        parts, diags = {sigma: [] for sigma in missing}, {sigma: [] for sigma in missing}
        for uid in sorted(base.user_id.unique()):
            check_deadline()
            p = read_points(ROOT / f"data/processed/quality_features/{uid}.parquet")
            p = p.sort_values(["feature_block_id", "timestamp"], kind="stable").reset_index(drop=True)
            field = ou_field(p, seed)
            field_sha = hashlib.sha256(field.tobytes()).hexdigest()
            q_delta = np.diff(field, axis=0)
            valid_edge = p.feature_block_id.iloc[1:].to_numpy() == p.feature_block_id.iloc[:-1].to_numpy()
            dt = p.timestamp.diff().dt.total_seconds().to_numpy()[1:][valid_edge]
            target_increment_variance = 2 * (-np.expm1(-dt / TAU))
            standardized_inc = q_delta[valid_edge] / np.sqrt(
                np.maximum(target_increment_variance[:, None], np.finfo(float).tiny))
            for sigma in missing:
                noisy, w = rebuild(p, field, sigma)
                parts[sigma].append(w)
                noise = field * sigma
                diags[sigma].append({
                    "user_id": uid, "points": len(p), "windows": len(w),
                    "standard_field_sha256": field_sha, "blocks": p.feature_block_id.nunique(),
                    "east_mean_m": float(noise[:, 0].mean()),
                    "north_mean_m": float(noise[:, 1].mean()),
                    "east_std_m": float(noise[:, 0].std()),
                    "north_std_m": float(noise[:, 1].std()),
                    "radial_rms_m": float(np.sqrt((noise ** 2).sum(axis=1).mean())),
                    "standardized_increment_mean_square":
                        float((standardized_inc[dt > 0] ** 2).mean()) if (dt > 0).any() else None,
                    "expected_increment_variance_mean": float(target_increment_variance.mean())
                        if len(dt) else None,
                    "speed_gt100": int(noisy.speed_mps.gt(100).sum()),
                    "missing_speed": int(noisy.speed_mps.isna().sum()),
                })
            print(f"OU seed={seed} user={uid} completed sigmas={missing}", flush=True)
        for sigma in missing:
            frame = pd.concat(parts[sigma], ignore_index=True)
            frame.user_id = frame.user_id.astype(str).str.zfill(3)
            pd.testing.assert_frame_equal(frame[META_COLS], base[META_COLS], check_dtype=False)
            assert len(frame) == 17380 and frame.n_points.sum() == 860972
            diag = diags[sigma]
            save_cached_frame(paths[sigma], frame, {
                "sigma": sigma, "seed": seed, "tau_seconds": TAU, "noise": "OU30",
                "points": int(frame.n_points.sum()), "windows": len(frame),
                "realized_radial_rms_m": float(np.sqrt(
                    sum(d["points"] * d["radial_rms_m"] ** 2 for d in diag) / 860972)),
                "speed_gt100": sum(d["speed_gt100"] for d in diag),
                "feature_missing_fraction": frame[cols].isna().mean().to_dict(),
                "feature_hash": df_hash(frame), "user_diagnostics": diag,
            })
        checkpoint("OU_cache_seed_complete", seed=seed,
                   caches=len(list((HERE / "feature_cache").glob("*.csv"))))
    assert len(list((HERE / "feature_cache").glob("*.csv"))) == 15


def train_gamma_controls(train):
    result = {}
    for group in ("G1", "G2"):
        check_deadline()
        cols = FEATURE_GROUPS[group]
        path = HERE / f"models/{group}_gamma1over27.joblib"
        cfg = path.with_suffix(".json")
        if valid_saved(path):
            est = joblib.load(path)
        else:
            checkpoint("fitting_gamma_control", group=group)
            est = model_specs(len(cols))[MODEL_NAME].make(True).set_params(
                clf__C=1.0, rbf__gamma=1.0 / 27)
            tick = time.monotonic()
            est.fit(train[cols].replace([np.inf, -np.inf], np.nan), train.label)
            joblib.dump(est, path)
            write_json(cfg, {
                "feature_columns": cols, "classes": MODES, "fitting_windows": len(train),
                "fitting_users": train.user_id.nunique(), "gamma": 1 / 27,
                "fit_seconds": time.monotonic() - tick,
                "file_sha256": sha(path), "code_sha256": sha(Path(__file__)),
                "protocol_sha256": sha(HERE / "PROTOCOL.md"),
                "input_manifest_sha256": sha(HERE / "input_manifest.json"),
                "pipeline_parameters": est.get_params(),
            })
        assert est.named_steps["rbf"].gamma == 1 / 27
        result[group] = (est, cols)
        checkpoint("gamma_model_saved", group=group, path=str(path))
    return result


def predict_table(est, cols, frame):
    raw, scores = score_aligned(est, frame[cols].replace([np.inf, -np.inf], np.nan))
    assert scores.shape == (len(frame), 5) and np.isfinite(scores).all()
    np.testing.assert_array_equal(raw, np.array(MODES)[scores.argmax(axis=1)])
    out = frame[PRED_META].copy()
    out["raw"] = raw
    out[SCORE_COLS] = scores
    out[SMOOTHERS[0]] = raw
    out[SMOOTHERS[1]] = smooth_soft(scores, frame.feature_block_id, frame.start_time, 11)
    out[SMOOTHERS[2]] = trailing_soft(scores, frame.feature_block_id, frame.start_time, 11)
    return out


def generate_predictions(models, cols):
    full = joblib.load(EXP / "models/full.joblib")
    for sigma in SIGMAS:
        for seed in ([0] if sigma == 0 else SEEDS):
            check_deadline()
            cond = condition_name(sigma, seed)
            original_cache = EXP / f"feature_cache/{cond}.csv"
            frame = read_table(original_cache)
            for group, (est, names) in models.items():
                path = HERE / f"predictions/{group}_fixedgamma_iid_{cond}.csv"
                if not valid_saved(path):
                    save_cached_frame(path, predict_table(est, names, frame), {
                        "family": "fixed_gamma", "group": group, "noise": "iid",
                        "sigma": sigma, "seed": seed,
                        "feature_source": str(original_cache.relative_to(ROOT)),
                        "feature_sha256": sha(original_cache),
                        "model_sha256": sha(HERE / f"models/{group}_gamma1over27.joblib"),
                    })
            if sigma:
                cache = HERE / f"feature_cache/OU30_{cond}.csv"
                path = HERE / f"predictions/G3_originalgamma_OU30_{cond}.csv"
                if not valid_saved(path):
                    ou = read_table(cache)
                    save_cached_frame(path, predict_table(full, cols, ou), {
                        "family": "correlated_noise", "group": "G3", "noise": "OU30",
                        "sigma": sigma, "seed": seed, "tau_seconds": TAU,
                        "feature_source": str(cache.relative_to(ROOT)), "feature_sha256": sha(cache),
                        "model_sha256": sha(EXP / "models/full.joblib"),
                    })
            checkpoint("predictions_condition_complete", condition=cond)
    assert len(list((HERE / "predictions").glob("*.csv"))) == 47


def predictions():
    """Yield all originals and extensions with explicit reused-source provenance."""
    for sigma in SIGMAS:
        for seed in ([0] if sigma == 0 else SEEDS):
            cond = condition_name(sigma, seed)
            for group in ("G1", "G2", "G3"):
                path = EXP / f"predictions/{group}_{cond}_smoothed.csv"
                yield ("original", group, "iid", sigma, seed), path
            for group in ("G1", "G2"):
                path = HERE / f"predictions/{group}_fixedgamma_iid_{cond}.csv"
                yield ("fixed_gamma", group, "iid", sigma, seed), path
            path = (HERE / f"predictions/G3_originalgamma_OU30_{cond}.csv" if sigma else
                    EXP / "predictions/G3_sigma00_seed0_smoothed.csv")
            yield ("correlated_noise", "G3", "OU30", sigma, seed), path


def summarize():
    base = read_table(EXP / "predictions/G3_sigma00_seed0_smoothed.csv")
    counts = base[[f"cnt_{m}" for m in MODES]].to_numpy(np.int64)
    gids = np.asarray(cv_groups(base.user_id, ROOT / "data/splits/report.json"))
    with np.load(EXP / "metrics/paired_bootstrap_draws.npz", allow_pickle=True) as saved:
        draws, clusters = saved["draws"], saved["clusters"].astype(str)
    np.testing.assert_array_equal(pd.unique(gids), clusters)
    assert draws.shape == (5000, 9) and (draws.sum(axis=1) == 9).all()
    masks = [gids == c for c in clusters]
    supports = np.array([counts[mask].sum(axis=0) for mask in masks])
    sizes = supports.sum(axis=1)
    n_windows = np.array([mask.sum() for mask in masks])
    rows, loo_rows, user_rows, values, boot = [], [], [], {}, {}
    cm_archive, summary_keys = [], []
    clean_predictions = {}
    baseline = pd.read_csv(EXP / "metrics/all_metrics.csv")
    for prefix, path in predictions():
        family, group, noise, sigma, seed = prefix
        d = read_table(path)
        pd.testing.assert_frame_equal(d[PRED_META], base[PRED_META])
        if sigma == 0:
            clean_predictions[(family, group)] = d[list(SMOOTHERS)].copy()
        for smoother in SMOOTHERS:
            pred = d[smoother].to_numpy()
            clean = clean_predictions[(family, group)][smoother].to_numpy()
            key = (*prefix, smoother)
            cms = np.array([confusion(counts[mask], pred[mask]) for mask in masks])
            total_cm = cms.sum(axis=0)
            metrics = {k: float(v) for k, v in point_metrics(total_cm).items()}
            sampled = point_metrics(np.einsum("bg,gij->bij", draws, cms))
            trans, flips, changed = [], [], []
            for mask in masks:
                sub = d.loc[mask]
                trans.append(one_to_one_transitions(sub, pred[mask], sub.label))
                flips.append(isolated_flips(pred[mask], sub.label,
                                           sub.feature_block_id, sub.start_time)[1])
                changed.append(int(base.n_points.to_numpy()[mask & (pred != clean)].sum()))
            true = np.array([x["true_transitions"] for x in trans])
            proposed = np.array([x["pred_transitions"] for x in trans])
            hit = np.array([x["matched_transitions"] for x in trans])
            assert true.sum() == 529
            flips, changed = np.array(flips), np.array(changed)
            metrics.update({k: float(v) for k, v in boundary_metrics(
                true.sum(), proposed.sum(), hit.sum()).items()})
            sampled.update(boundary_metrics(draws @ true, draws @ proposed, draws @ hit))
            metrics["wrong_isolated_flips_per100_windows"] = float(flips.sum() / len(d) * 100)
            sampled["wrong_isolated_flips_per100_windows"] = (draws @ flips) / (draws @ n_windows) * 100
            metrics["prediction_change_from_clean_point_fraction"] = float(changed.sum() / counts.sum())
            sampled["prediction_change_from_clean_point_fraction"] = (draws @ changed) / (draws @ sizes)
            for j, mode in enumerate(MODES):
                metrics[f"f1_{mode}"] = float(safe_ratio(
                    2 * total_cm[j, j], total_cm[j].sum() + total_cm[:, j].sum()))
            info = dict(family=family, group=group, noise=noise, sigma=sigma,
                        seed=seed, smoother=smoother)
            values[key], boot[key] = metrics, sampled
            rows.append({**info, **metrics, "true_transitions": int(true.sum()),
                         "pred_transitions": int(proposed.sum()),
                         "matched_transitions": int(hit.sum()),
                         "source": path.relative_to(ROOT).as_posix(), "source_sha256": sha(path)})
            cm_archive.append(cms)
            summary_keys.append("|".join(map(str, key)))
            if family == "original":
                ref = baseline[(baseline.group == group) & (baseline.sigma == sigma)
                               & (baseline.seed == seed) & (baseline.smoother == smoother)].iloc[0]
                for metric in METRICS + [f"f1_{m}" for m in MODES]:
                    np.testing.assert_allclose(metrics[metric], ref[metric], atol=1e-12, rtol=0)
            for omitted in range(-1, len(clusters)):
                keep_cm = total_cm if omitted == -1 else total_cm - cms[omitted]
                keep_support = counts.sum(axis=0) if omitted == -1 else counts.sum(axis=0) - supports[omitted]
                loo_rows.append({
                    **info, "omitted_group": "none" if omitted == -1 else clusters[omitted],
                    **{k: float(v) for k, v in point_metrics(keep_cm).items()},
                    "retained_points": int(keep_support.sum()),
                    "omitted_points": 0 if omitted == -1 else int(sizes[omitted]),
                    "true_classes_present": int((keep_support > 0).sum()),
                    **{f"support_{m}": int(v) for m, v in zip(MODES, keep_support)},
                })
            for uid, sub in d.groupby("user_id", sort=False):
                umask = d.user_id.eq(uid).to_numpy()
                ucm = confusion(counts[umask], pred[umask])
                user_rows.append({
                    **info, "user_id": uid, "linked_group": str(gids[np.flatnonzero(umask)[0]]),
                    "points": int(counts[umask].sum()), "windows": len(sub),
                    **{k: float(v) for k, v in point_metrics(ucm).items()},
                    **one_to_one_transitions(sub, pred[umask], sub.label),
                    **{f"support_{m}": int(v) for m, v in zip(MODES, counts[umask].sum(axis=0))},
                })
        print(f"Scored {prefix}", flush=True)
    all_rows = pd.DataFrame(rows)
    all_rows.to_csv(HERE / "tables/all_seed_metrics.csv", index=False)
    pd.DataFrame(user_rows).to_csv(HERE / "tables/per_user_metrics.csv", index=False)
    pd.DataFrame(rows).melt(
        id_vars=["family", "group", "noise", "sigma", "seed", "smoother"],
        value_vars=[f"f1_{m}" for m in MODES], var_name="class", value_name="f1",
    ).to_csv(HERE / "tables/per_class_f1.csv", index=False)
    np.savez_compressed(HERE / "diagnostics/cluster_confusions.npz",
                        keys=np.array(summary_keys), clusters=clusters,
                        cms=np.array(cm_archive), supports=supports)
    aggregated, contrasts = [], []
    families = [("original", g, "iid") for g in ("G1", "G2", "G3")]
    families += [("fixed_gamma", g, "iid") for g in ("G1", "G2")]
    families += [("correlated_noise", "G3", "OU30")]
    for family, group, noise in families:
        for sigma in SIGMAS:
            seeds = [0] if sigma == 0 else SEEDS
            for smoother in SMOOTHERS:
                keys = [(family, group, noise, sigma, s, smoother) for s in seeds]
                clean_key = (family, group, noise, 0, 0, smoother)
                row = dict(family=family, group=group, noise=noise, sigma=sigma,
                           smoother=smoother, seed_count=len(seeds))
                for metric in METRICS:
                    vals = np.array([values[k][metric] for k in keys])
                    dist = np.array([boot[k][metric] for k in keys]).mean(axis=0)
                    row.update({
                        f"{metric}_mean": float(vals.mean()),
                        f"{metric}_seed_sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.,
                        f"{metric}_ci_low": float(np.percentile(dist, 2.5)),
                        f"{metric}_ci_high": float(np.percentile(dist, 97.5)),
                    })
                vals = np.array([values[k]["macro_f1"] - values[clean_key]["macro_f1"] for k in keys])
                dist = np.array([boot[k]["macro_f1"] - boot[clean_key]["macro_f1"] for k in keys]).mean(axis=0)
                row.update(noisy_minus_clean_f1=float(vals.mean()),
                           noisy_minus_clean_ci_low=float(np.percentile(dist, 2.5)),
                           noisy_minus_clean_ci_high=float(np.percentile(dist, 97.5)))
                aggregated.append(row)
                if family != "original":
                    other = [("original", group, "iid", sigma, s, smoother) for s in seeds]
                    add_contrast(contrasts, f"{family}_minus_original", keys, other,
                                 values, boot, sigma, smoother, group)
    # With gamma fixed, original G3 is the shared unchanged control.
    for sigma in SIGMAS:
        for smoother in SMOOTHERS:
            seeds = [0] if sigma == 0 else SEEDS
            groups = {"G3": [("original", "G3", "iid", sigma, s, smoother) for s in seeds]}
            groups.update({g: [("fixed_gamma", g, "iid", sigma, s, smoother) for s in seeds]
                           for g in ("G1", "G2")})
            for a, b in (("G3", "G1"), ("G3", "G2"), ("G2", "G1")):
                add_contrast(contrasts, f"fixed_gamma_{a}_minus_{b}", groups[a], groups[b],
                             values, boot, sigma, smoother, a)
    pd.DataFrame(aggregated).to_csv(HERE / "tables/summary.csv", index=False)
    pd.DataFrame(contrasts).to_csv(HERE / "tables/paired_contrasts.csv", index=False)
    loo = pd.DataFrame(loo_rows)
    keys = ["family", "group", "noise", "smoother", "omitted_group"]
    clean = loo[loo.sigma == 0][keys + ["macro_f1"]].rename(columns={"macro_f1": "clean_macro_f1"})
    loo = loo.merge(clean, on=keys, validate="many_to_one")
    loo["noisy_minus_clean_f1"] = loo.macro_f1 - loo.clean_macro_f1
    loo.to_csv(HERE / "tables/leave_one_group_out_seed.csv", index=False)
    keys += ["sigma"]
    nums = ["macro_f1", "accuracy", "noisy_minus_clean_f1", "clean_macro_f1",
            "retained_points", "omitted_points", "true_classes_present", *[f"support_{m}" for m in MODES]]
    ls = loo.groupby(keys, sort=False)[nums].mean().reset_index()
    ls["macro_f1_seed_sd"] = loo.groupby(keys, sort=False).macro_f1.std().fillna(0).to_numpy()
    ls.to_csv(HERE / "tables/leave_one_group_out_summary.csv", index=False)
    # Equal weight per observed group, still fixed-five and descriptive.
    group_metrics = []
    for key, cms in zip(summary_keys, cm_archive):
        vals = point_metrics(cms)
        group_metrics.append({"key": key, "unweighted_group_macro_f1": float(vals["macro_f1"].mean()),
                              "unweighted_group_accuracy": float(vals["accuracy"].mean())})
    pd.DataFrame(group_metrics).to_csv(HERE / "tables/equal_group_descriptive.csv", index=False)
    write_json(HERE / "diagnostics/scoring_checks.json", {
        "original_metric_rows_reproduced": 144, "seed_metric_rows": len(rows),
        "user_metric_rows": len(user_rows), "summary_rows": len(aggregated),
        "paired_contrast_rows": len(contrasts), "loo_seed_rows": len(loo),
        "bootstrap_draws": 5000, "clusters": clusters.tolist(),
        "zero_boundary_bootstrap_draws": int(((draws @ true) == 0).sum()),
        "fixed_five_class": True, "bootstrap_seedwise_then_average": True,
    })
    checkpoint("analysis_complete", metrics=len(rows), summaries=len(aggregated))


def add_contrast(rows, label, a_keys, b_keys, values, boot, sigma, smoother, group):
    for metric in METRICS:
        vals = np.array([values[a][metric] - values[b][metric] for a, b in zip(a_keys, b_keys)])
        dist = np.array([boot[a][metric] - boot[b][metric] for a, b in zip(a_keys, b_keys)]).mean(axis=0)
        rows.append({
            "contrast": label, "group": group, "sigma": sigma, "smoother": smoother,
            "metric": metric, "seed_count": len(vals), "difference_mean": float(vals.mean()),
            "difference_seed_sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.,
            "ci_low": float(np.percentile(dist, 2.5)), "ci_high": float(np.percentile(dist, 97.5)),
        })


def figures_and_report():
    """Generated narrative uses all fixed conditions; no outcome-driven case selection."""
    s = pd.read_csv(HERE / "tables/summary.csv")
    raw = s[s.smoother == SMOOTHERS[0]]
    contrasts = pd.read_csv(HERE / "tables/paired_contrasts.csv")
    loo = pd.read_csv(HERE / "tables/leave_one_group_out_summary.csv")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for family, name, color in [
        ("original", "iid position noise", "#b04442"),
        ("correlated_noise", "OU tau=30s position noise", "#217962"),
    ]:
        sub = raw[(raw.family == family) & (raw.group == "G3")]
        axes[0].errorbar(sub.sigma, sub.macro_f1_mean, yerr=sub.macro_f1_seed_sd,
                         marker="o", label=name, color=color, capsize=3)
        axes[1].plot(sub.sigma, sub.noisy_minus_clean_f1, marker="o", label=name, color=color)
    axes[0].set_ylabel("Point Macro-F1")
    axes[1].set_ylabel("Noisy minus own clean Macro-F1")
    for ax in axes:
        ax.set_xlabel("Per-axis sigma (m)")
        ax.set_xticks(SIGMAS)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Frozen G3: temporal noise structure sensitivity (exploratory)")
    save_figure(fig, "figure1_temporal_noise")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for ax, group in zip(axes, ("G1", "G2", "G3")):
        for family, label, color in [("original", "Original gamma=1/d", "#555555"),
                                      ("fixed_gamma", "Fixed gamma=1/27", "#307fa8")]:
            sub = raw[(raw.family == family) & (raw.group == group)]
            if len(sub):
                ax.errorbar(sub.sigma, sub.macro_f1_mean, yerr=sub.macro_f1_seed_sd,
                            marker="o", label=label, color=color, capsize=3)
        ax.set_title(group + (" (shared unchanged control)" if group == "G3" else ""))
        ax.set_xlabel("Per-axis sigma (m)")
        ax.set_xticks(SIGMAS)
        ax.set_ylim(0, .9)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Point Macro-F1")
    fig.suptitle("Feature recipes with gamma held fixed: all conditions, no selection")
    save_figure(fig, "figure2_fixed_gamma")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax, sigma in zip(axes, (0, 20)):
        sub = loo[(loo.family == "original") & (loo.smoother == SMOOTHERS[0]) & (loo.sigma == sigma)]
        order = ["none", *pd.read_csv(EXP / "paired_analysis/tables/linked_group_sizes.csv",
                                     dtype={"linked_group": str}).linked_group.tolist()]
        for group, color in [("G1", "#555555"), ("G2", "#307fa8"), ("G3", "#217962")]:
            d = sub[sub.group == group].set_index("omitted_group").reindex(order)
            ax.plot(d.macro_f1, np.arange(len(order)), marker="o", label=group, color=color)
        ax.set_yticks(np.arange(len(order)), order)
        ax.invert_yaxis()
        ax.set_title(f"Original iid sigma={sigma}m")
        ax.set_xlabel("Pooled retained-point Macro-F1 (fixed five classes)")
        ax.set_ylabel("Omitted linked group (none = full test)")
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("Deterministic group influence: no refitting, unequal class supports")
    save_figure(fig, "figure3_group_influence")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for smoother, label, color in zip(SMOOTHERS, ("Raw", "Centered k=11", "Trailing k=11"),
                                     ("#555555", "#307fa8", "#b04442")):
        d = s[(s.family == "correlated_noise") & (s.smoother == smoother)]
        for ax, metric in zip(axes, ("macro_f1", "transition_recall")):
            ax.errorbar(d.sigma, d[f"{metric}_mean"], yerr=d[f"{metric}_seed_sd"],
                        marker="o", label=label, color=color, capsize=3)
    for ax, ylabel in zip(axes, ("Point Macro-F1", "Boundary recall (+/-1 window)")):
        ax.set_xlabel("Per-axis sigma (m)")
        ax.set_xticks(SIGMAS)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("Score smoothing on OU-noise predictions; no coordinate denoising")
    save_figure(fig, "figure4_OU_smoothing")
    fig_meta = []
    from PIL import Image
    for path in sorted((HERE / "figures").glob("*.png")):
        image = np.asarray(Image.open(path).convert("RGB"))
        assert image.shape[0] > 400 and image.shape[1] > 800 and image.std() > 5
        fig_meta.append({"file": path.name, "shape": list(image.shape),
                         "pixel_std": float(image.std()), "nonblank": True})
    write_json(HERE / "diagnostics/figure_checks.json", {
        "automated_nonblank": fig_meta, "manual_visual_review": "pending",
        "caption": "Error bars are perturbation-seed SD, not user generalisation intervals.",
    })
    write_report(s, contrasts, loo)
    checkpoint("report_and_figures_saved", figures=4)


def save_figure(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for ax in fig.axes:
        for text in [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]:
            if text.get_text():
                box = text.get_window_extent(renderer)
                if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
                    clipped.append(text.get_text())
    assert not clipped, f"Text outside figure canvas: {clipped}"
    fig.savefig(HERE / f"figures/{name}.png", dpi=170)
    fig.savefig(HERE / f"figures/{name}.pdf")
    plt.close(fig)


def markdown(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def write_report(summary, contrasts, loo):
    raw = summary[summary.smoother == SMOOTHERS[0]]
    noise_rows, gamma_rows = [], []
    for sigma in SIGMAS:
        iid = raw[(raw.family == "original") & (raw.group == "G3") & (raw.sigma == sigma)].iloc[0]
        ou = raw[(raw.family == "correlated_noise") & (raw.sigma == sigma)].iloc[0]
        delta = contrasts[(contrasts.contrast == "correlated_noise_minus_original")
                          & (contrasts.sigma == sigma) & (contrasts.smoother == SMOOTHERS[0])
                          & (contrasts.metric == "macro_f1")].iloc[0]
        noise_rows.append([
            sigma, f"{iid.macro_f1_mean:.4f}", f"{ou.macro_f1_mean:.4f} ± {ou.macro_f1_seed_sd:.4f}",
            f"{ou.noisy_minus_clean_f1:+.4f}", f"{delta.difference_mean:+.4f}",
            f"[{delta.ci_low:+.4f}, {delta.ci_high:+.4f}]",
        ])
        for group in ("G1", "G2"):
            old = raw[(raw.family == "original") & (raw.group == group) & (raw.sigma == sigma)].iloc[0]
            new = raw[(raw.family == "fixed_gamma") & (raw.group == group) & (raw.sigma == sigma)].iloc[0]
            gamma_rows.append([group, sigma, f"{old.macro_f1_mean:.4f}", f"{new.macro_f1_mean:.4f}",
                               f"{new.noisy_minus_clean_f1:+.4f}"])
    lr = loo[(loo.family == "original") & (loo.smoother == SMOOTHERS[0])
             & (loo.sigma.isin([0, 20])) & (loo.omitted_group.isin(["none", "linked0"]))]
    loo_rows = [[r.group, int(r.sigma), r.omitted_group, int(r.retained_points),
                 int(r.true_classes_present), f"{r.macro_f1:.4f}", f"{r.noisy_minus_clean_f1:+.4f}"]
                for r in lr.itertuples()]
    smooth_rows = [
        [int(r.sigma), r.smoother, f"{r.macro_f1_mean:.4f}", f"{r.transition_recall_mean:.4f}",
         f"{r.transition_f1_mean:.4f}"]
        for r in summary[(summary.family == "correlated_noise") & summary.sigma.isin([0, 20])].itertuples()
    ]
    text = f"""# 噪声主线：新增探索性实验

2026-11-23。AI辅助分析草稿，供团队核对、改写与引用。
本轮实际补跑相关噪声与固定gamma对照，并完成逐组排除分析。
原测试结果已提前看过：新增协议在本轮计算前固定，但不是预注册、
新独立测试验证或按测试结果选择模型。原主实验、原v2结果均保留。

## 1. 时间相关噪声：原结论对噪声结构有多敏感

沿原实际点间隔，在每个原block内生成平稳OU噪声；单轴sigma为
5/10/20m，相关时间tau=30s，五个原种子，E/N独立。
每个block重新取平稳初值；跨sigma共享标准场，且与原iid使用相同创新随机数。
噪声坐标后重新计算全部原运动量和窗口特征，不删除超速点。G3原冻结模型直接推理。

tau=30s只是一档预先固定的人工敏感性条件，不是GPS设备标定或“真实噪声模型”。
理论单轴边际SD相同，实际有限样本RMS原样记录、不归一化；相关过程的有效独立误差数更少。
更平滑的误差通常减小相邻误差增量，但本表给出的是实测模型结果，不由公式替代。

{markdown(["sigma/m", "iid raw F1", "OU raw F1±种子SD", "OU noisy−clean", "OU−iid", "差值用户组95%区间"], noise_rows)}

![相关结构对照](figures/figure1_temporal_noise.png)

正的OU−iid表示同一G3在这一合成相关结构下分数较高；
它不能证明某种现实GPS设备具有对应收益，更不能归结为模型执行了坐标去噪。
区间复用原5000次九关联组配对抽样，先逐种子计分再平均。
原clean只有一份，SD=0是CSV约定，未制造重复clean实验。

## 2. 固定gamma后再比较特征配方

在原70,086个clean拟合窗上新拟合G1/G2，gamma都设1/27。
中位数填补、标准化、1000维Nystroem、LinearSVC C=1 balanced、
seed42及其他配方不变。使用原16份iid特征缓存；G3复用原模型。

{markdown(["组", "sigma/m", "原gamma=1/d raw F1", "固定gamma raw F1", "固定gamma noisy−clean"], gamma_rows)}

![固定gamma对照](figures/figure2_fixed_gamma.png)

这减少了比较中数值gamma同步变化的混杂，但输入维度、标准化距离和Nystroem表示仍会变化；
相同gamma不等于相同有效核宽度，更不是单独jerk/转向特征的因果消融。
全部条件与配对差见[summary](tables/summary.csv)、
[paired contrasts](tables/paired_contrasts.csv)，不据本次测试挑选“最优”配方。

## 3. 最大用户组影响：逐一排除九组

最大linked0组原占679,162/860,972点，即78.8832%。
依次排除每一原关联组，重算剩余点的固定五类F1，模型不重训。
下表重点显示全量与排除最大组；完整九组都在CSV保留。

{markdown(["组", "sigma/m", "排除组", "保留点", "真值类别数", "raw F1", "noisy−clean"], loo_rows)}

![组影响分析](figures/figure3_group_influence.png)

不同排除集的用户和类别构成不同，分数变化不能全部归因于“某用户难”。
固定五类计分保留；每行的support_*、真值类别数、点数均输出。
九次排除共享绝大部分样本，不能当九次独立复现或新的置信区间。
另外输出各关联组固定五类指标等权平均，明确是描述性补充，不替代主点加权指标。

## 4. 分数平滑在相关噪声下的代价

{markdown(["sigma/m", "平滑", "点F1", "边界召回", "边界F1"], smooth_rows)}

![OU下的平滑与边界](figures/figure4_OU_smoothing.png)

只平滑decision分数，不修改坐标。居中k=11用未来窗口，trailing只用当前与历史。
边界仍用窗口多数标签、同block内±1窗class-blind一对一最大匹配；
不把这一指标叫精确秒级实时切换能力。
clean只有529个边界，原5000个bootstrap抽样中133次无真边界，仍沿原零分母记0约定。
分类改善、少跳变和真实切换恢复必须分别讨论。

## 5. 证据边界与不足

本轮补的是明确的对照：人工噪声时间相关性、gamma同步变化、最大组影响。
它增加了对原结论适用范围的了解，不自动提高现实泛化能力。
没有新设备数据、新独立用户或按实际设备估计的误差过程。
tau只有30s一档；不代表全部时间相关噪声，更不覆盖多路径、偏置漂移、丢点或城市峡谷。
G1/G2/G3的原特征定义、signed acc_std和path_length/duration支持不一致仍保留。
raw基线1/17,380窗复现差异、混合标签窗口与粗粒度边界容差问题也没有被新实验消除。
未校正多重比较，区间只用于探索描述，不能据是否跨零宣称普遍排名。

全部新预测保留五类decision分数；模型、缓存与manifest均在本目录。
原144行指标逐项重算核对。自动验收与独立检查范围见
[验收](acceptance.json)、[独立验证](independent_verification.json)。
四张图提供PNG/PDF；自动布局与非空检查不等于完整目视审阅。

## 6. 可复现与来源

- [计算前固定的补充协议](PROTOCOL.md)
- [每种子完整指标](tables/all_seed_metrics.csv)、[逐用户](tables/per_user_metrics.csv)、
  [逐类F1](tables/per_class_f1.csv)
- [逐组排除全记录](tables/leave_one_group_out_seed.csv)、
  [逐组排除汇总](tables/leave_one_group_out_summary.csv)
- [输入哈希与软件版本](input_manifest.json)、[运行checkpoint](checkpoint.json)
- [原始文献访问记录](literature/retrieval.json)；原GPS论文阅读边界沿用
  [既有来源清单](../literature_review/SOURCES.md)。

OU公式依据Rasmussen与Williams《Gaussian Processes for Machine Learning》(2006)
第4章的指数协方差/OU过程；gamma依据scikit-learn Nystroem官方API。
流程使用scientific-critical-thinking技能，其软件引用为
Kassis等(2026), Scientific Agent Skills: A Library of Procedural Knowledge for
Research Agents, arXiv:2609.00065, DOI10.48550/arXiv.2609.00065。
该技能不是实验有效性的外部认证。
"""
    (HERE / "SUPPLEMENT_REPORT.md").write_text(text, encoding="utf-8")


def final_check():
    saved = json.loads((HERE / "input_manifest.json").read_text(encoding="utf-8"))
    current = {p.relative_to(ROOT).as_posix(): sha(p) for p in protected_files()}
    assert current == saved["files"], "Protected original contents or membership changed"
    assert saved["protocol_sha256"] == sha(HERE / "PROTOCOL.md")
    assert saved["code_sha256"] == sha(Path(__file__))
    for folder, expected in (("feature_cache", 15), ("predictions", 47)):
        paths = list((HERE / folder).glob("*.csv"))
        assert len(paths) == expected and all(valid_saved(p) for p in paths)
    s = pd.read_csv(HERE / "tables/summary.csv")
    assert len(s) == 72 and not s.select_dtypes("number").isna().any().any()
    assert len(pd.read_csv(HERE / "tables/all_seed_metrics.csv")) == 288
    assert len(pd.read_csv(HERE / "tables/per_user_metrics.csv")) == 3744
    assert len(pd.read_csv(HERE / "tables/leave_one_group_out_seed.csv")) == 2880
    assert len(list((HERE / "figures").glob("*.png"))) == 4
    write_json(HERE / "acceptance.json", {
        "passed": True, "protected_files_unchanged": len(current),
        "protocol_unchanged": True, "new_models": 2, "OU_feature_caches": 15,
        "new_score_files": 47, "all_metric_rows_including_reused": 288,
        "summary_rows": 72, "manual_visual_review": "pending",
        "completed_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "elapsed_seconds": time.monotonic() - START,
    })
    checkpoint("complete", status="passed", protected_files=len(current))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "init", "sources", "features", "models",
                                            "predict", "analyze", "report", "verify"), default="all")
    args = parser.parse_args()
    initialize()
    check_deadline()
    stage = args.stage
    if stage in ("all", "sources"):
        retrieve_sources()
    if stage == "init" or stage == "sources":
        return
    train, base = load_tables()
    cols = list(prepare(train)[0].columns)
    if stage in ("all", "features"):
        zero_check(base, cols)
        generate_ou_caches(base, cols)
    if stage in ("all", "models", "predict"):
        models = train_gamma_controls(train)
    if stage in ("all", "predict"):
        generate_predictions(models, cols)
    if stage in ("all", "analyze"):
        summarize()
    if stage in ("all", "report"):
        figures_and_report()
    if stage in ("all", "verify"):
        final_check()


if __name__ == "__main__":
    try:
        # BLAS parallelism is an execution setting, not an estimator hyperparameter.
        with threadpool_limits(limits=4):
            main()
    except Exception as exc:
        text = traceback.format_exc()
        (HERE / "error.log").write_text(text, encoding="utf-8")
        checkpoint("error", status="failed", error=str(exc))
        raise
