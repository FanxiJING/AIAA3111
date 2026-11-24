"""Separate tally, rolling-average and interval-matching verification."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import Nystroem
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
MODES = ["walk", "bike", "bus", "car", "train"]
SMOOTHERS = ["S0_raw_k1", "S1_centered_soft_k11", "S2_trailing_soft_k11"]
FEATURES = [f"cnt_{m}" for m in MODES]
SCORES = [f"score_{m}" for m in MODES]
META = ["user_id", "feature_block_id", "start_time", "end_time", "n_points",
        "label", "purity", *FEATURES]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ratio(a, b):
    return np.divide(a, b, out=np.zeros(np.broadcast_shapes(np.shape(a), np.shape(b))),
                     where=np.asarray(b) != 0)


def tally(counts, labels):
    # Direct indexed accumulation, independent of production confusion().
    code = pd.Categorical(labels, categories=MODES).codes
    assert (code >= 0).all()
    cm = np.zeros((5, 5), np.int64)
    for truth in range(5):
        np.add.at(cm[truth], code, counts[:, truth])
    return cm


def metric(cm):
    diag = np.diagonal(cm, axis1=-2, axis2=-1)
    f1 = ratio(2 * diag, cm.sum(axis=-1) + cm.sum(axis=-2))
    return f1.mean(axis=-1), ratio(diag.sum(axis=-1), cm.sum(axis=(-2, -1))), f1


def ordered_matching(predicted, true, tol=1):
    """Two-pointer maximum matching for sorted equal-radius interval neighbours."""
    i = j = hits = 0
    while i < len(predicted) and j < len(true):
        if predicted[i] < true[j] - tol:
            i += 1
        elif true[j] < predicted[i] - tol:
            j += 1
        else:
            hits += 1
            i += 1
            j += 1
    return hits


def read(path):
    d = pd.read_csv(path, dtype={"user_id": str})
    d.user_id = d.user_id.str.zfill(3)
    return d


def main():
    started = time.monotonic()
    summary = pd.read_csv(HERE / "tables/summary.csv")
    rows = pd.read_csv(HERE / "tables/all_seed_metrics.csv")
    loo = pd.read_csv(HERE / "tables/leave_one_group_out_seed.csv",
                      dtype={"omitted_group": str})
    base = read(EXP / "predictions/G3_sigma00_seed0_smoothed.csv")
    counts = base[FEATURES].to_numpy(np.int64)
    group_table = pd.read_csv(EXP / "paired_analysis/tables/linked_group_sizes.csv",
                              dtype={"linked_group": str})
    group_of = {u: r.linked_group for r in group_table.itertuples() for u in r.users.split(",")}
    gids = base.user_id.map(group_of).to_numpy()
    with np.load(EXP / "metrics/paired_bootstrap_draws.npz", allow_pickle=True) as saved:
        draws, clusters = saved["draws"], saved["clusters"].astype(str)
    with np.load(HERE / "diagnostics/cluster_confusions.npz") as saved:
        cm_saved = dict(zip(saved["keys"], saved["cms"]))
    blocks = [np.asarray(i) for i in base.groupby(["user_id", "feature_block_id"], sort=False).indices.values()]
    values, distributions, clean_preds, tables = {}, {}, {}, {}
    boundary_recomputed = smoothing_recomputed = score_smokes = 0
    for row in rows.itertuples():
        key = (row.family, row.group, row.noise, row.sigma, row.seed, row.smoother)
        source = ROOT / row.source
        assert sha(source) == row.source_sha256
        if row.source not in tables:
            tables[row.source] = read(source)
        d = tables[row.source]
        pd.testing.assert_frame_equal(d[META], base[META])
        pred = d[row.smoother].to_numpy()
        clean_key = (row.family, row.group, row.smoother)
        if row.sigma == 0:
            clean_preds[clean_key] = pred
        cm = tally(counts, pred)
        f1, accuracy, per_class = metric(cm)
        np.testing.assert_allclose([f1, accuracy], [row.macro_f1, row.accuracy], atol=1e-12, rtol=0)
        for mode, score in zip(MODES, per_class):
            np.testing.assert_allclose(score, getattr(row, f"f1_{mode}"), atol=1e-12, rtol=0)
        cluster_cm = np.stack([tally(counts[gids == c], pred[gids == c]) for c in clusters])
        np.testing.assert_array_equal(cluster_cm, cm_saved["|".join(map(str, key))])
        sampled = np.tensordot(draws, cluster_cm, axes=(1, 0))
        df1, dacc, _ = metric(sampled)
        true_n = pred_n = hit_n = wrong_flips = 0
        cluster_boundary = np.zeros((9, 4), int)
        for idx in blocks:
            p, t = pred[idx], base.label.to_numpy()[idx]
            bp = np.flatnonzero(p[1:] != p[:-1]) + 1
            bt = np.flatnonzero(t[1:] != t[:-1]) + 1
            hits = ordered_matching(bp, bt)
            flip = (p[1:-1] != p[:-2]) & (p[:-2] == p[2:]) & (p[1:-1] != t[1:-1])
            t_n, p_n, h_n, wrong = len(bt), len(bp), hits, int(flip.sum())
            true_n += t_n
            pred_n += p_n
            hit_n += h_n
            wrong_flips += wrong
            cluster_boundary[list(clusters).index(gids[idx[0]])] += [t_n, p_n, h_n, wrong]
        np.testing.assert_array_equal([true_n, pred_n, hit_n],
                                     [row.true_transitions, row.pred_transitions, row.matched_transitions])
        np.testing.assert_allclose(
            [ratio(hit_n, pred_n), ratio(hit_n, true_n), ratio(2 * hit_n, true_n + pred_n),
             wrong_flips / len(d) * 100],
            [row.transition_precision, row.transition_recall, row.transition_f1,
             row.wrong_isolated_flips_per100_windows], atol=1e-12, rtol=0)
        changed = pred != clean_preds[clean_key]
        changed_counts = np.array([base.n_points.to_numpy()[changed & (gids == c)].sum() for c in clusters])
        sizes = np.array([base.n_points.to_numpy()[gids == c].sum() for c in clusters])
        windows = np.array([(gids == c).sum() for c in clusters])
        sampled_boundary = draws @ cluster_boundary
        dt, dp, dh, dw = sampled_boundary.T
        distributions[key] = {
            "macro_f1": df1, "accuracy": dacc,
            "transition_precision": ratio(dh, dp), "transition_recall": ratio(dh, dt),
            "transition_f1": ratio(2 * dh, dt + dp),
            "wrong_isolated_flips_per100_windows": (draws @ cluster_boundary[:, 3]) / (draws @ windows) * 100,
            "prediction_change_from_clean_point_fraction": (draws @ changed_counts) / (draws @ sizes),
        }
        values[key] = {name: getattr(row, name) for name in distributions[key]}
        np.testing.assert_allclose(changed_counts.sum() / sizes.sum(),
                                   row.prediction_change_from_clean_point_fraction, atol=1e-12)
        check_loo = loo[
            (loo.family == row.family) & (loo.group == row.group) & (loo.sigma == row.sigma)
            & (loo.seed == row.seed) & (loo.smoother == row.smoother)]
        assert len(check_loo) == 10
        for left in check_loo.itertuples():
            keep = np.ones(len(d), bool) if left.omitted_group == "none" else gids != left.omitted_group
            lf1, lacc, _ = metric(tally(counts[keep], pred[keep]))
            cf1, _, _ = metric(tally(counts[keep], clean_preds[clean_key][keep]))
            np.testing.assert_allclose([lf1, lacc, lf1 - cf1, cf1],
                                       [left.macro_f1, left.accuracy, left.noisy_minus_clean_f1,
                                        left.clean_macro_f1], atol=1e-12)
            assert counts[keep].sum() == left.retained_points
            for mode, support in zip(MODES, counts[keep].sum(axis=0)):
                assert support == getattr(left, f"support_{mode}")
        if row.family != "original" and row.smoother == SMOOTHERS[0] and row.sigma != 0:
            scores = d[SCORES].to_numpy()
            np.testing.assert_array_equal(np.array(MODES)[scores.argmax(axis=1)], pred)
            # Independent block-wise pandas rolling means, without a global prefix sum.
            for idx in blocks:
                block_scores = pd.DataFrame(scores[idx])
                for smoother, center in [(SMOOTHERS[1], True), (SMOOTHERS[2], False)]:
                    means = block_scores.rolling(11, center=center, min_periods=1).mean().to_numpy()
                    np.testing.assert_array_equal(np.array(MODES)[means.argmax(axis=1)],
                                                  d[smoother].to_numpy()[idx])
            smoothing_recomputed += 2
        boundary_recomputed += 1
        if boundary_recomputed % 24 == 0:
            print(f"Independently checked {boundary_recomputed}/288 metric rows", flush=True)
    all_metrics = list(next(iter(distributions.values())))
    for row in summary.itertuples():
        keys = [(row.family, row.group, row.noise, row.sigma, seed, row.smoother)
                for seed in ([0] if row.sigma == 0 else range(20261123, 20261128))]
        for name in all_metrics:
            vals = [values[k][name] for k in keys]
            dist = np.stack([distributions[k][name] for k in keys]).mean(axis=0)
            expected = [np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0,
                        *np.percentile(dist, [2.5, 97.5])]
            found = [getattr(row, f"{name}_{suffix}") for suffix in ("mean", "seed_sd", "ci_low", "ci_high")]
            np.testing.assert_allclose(expected, found, atol=1e-12)
        clean = (row.family, row.group, row.noise, 0, 0, row.smoother)
        delta = np.mean([values[k]["macro_f1"] - values[clean]["macro_f1"] for k in keys])
        dist = np.mean([distributions[k]["macro_f1"] - distributions[clean]["macro_f1"] for k in keys], axis=0)
        np.testing.assert_allclose([delta, *np.percentile(dist, [2.5, 97.5])],
                                   [row.noisy_minus_clean_f1, row.noisy_minus_clean_ci_low,
                                    row.noisy_minus_clean_ci_high], atol=1e-12)
    contrasts = pd.read_csv(HERE / "tables/paired_contrasts.csv")
    for row in contrasts.itertuples():
        seeds = [0] if row.sigma == 0 else range(20261123, 20261128)
        if row.contrast == "fixed_gamma_minus_original":
            a, b = ("fixed_gamma", row.group, "iid"), ("original", row.group, "iid")
        elif row.contrast == "correlated_noise_minus_original":
            a, b = ("correlated_noise", "G3", "OU30"), ("original", "G3", "iid")
        else:
            ga, gb = row.contrast.removeprefix("fixed_gamma_").split("_minus_")
            a = ("original" if ga == "G3" else "fixed_gamma", ga, "iid")
            b = ("original" if gb == "G3" else "fixed_gamma", gb, "iid")
        pairs = [(a + (row.sigma, seed, row.smoother), b + (row.sigma, seed, row.smoother))
                 for seed in seeds]
        vals = [values[ka][row.metric] - values[kb][row.metric] for ka, kb in pairs]
        dist = np.mean([distributions[ka][row.metric] - distributions[kb][row.metric] for ka, kb in pairs], axis=0)
        np.testing.assert_allclose(
            [np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0, *np.percentile(dist, [2.5, 97.5])],
            [row.difference_mean, row.difference_seed_sd, row.ci_low, row.ci_high], atol=1e-12)
    for path in sorted((HERE / "predictions").glob("*.csv")):
        manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        d = read(path)
        if manifest["family"] == "fixed_gamma":
            model_path = HERE / f"models/{manifest['group']}_gamma1over27.joblib"
            model_cfg = model_path.with_suffix(".json")
        else:
            model_path, model_cfg = EXP / "models/full.joblib", EXP / "models/G3_config.json"
        assert sha(model_path) == manifest["model_sha256"]
        source = ROOT / manifest["feature_source"]
        assert sha(source) == manifest["feature_sha256"]
        frame = read(source)
        cols = json.loads(model_cfg.read_text(encoding="utf-8"))["feature_columns"]
        est = joblib.load(model_path)
        indices = np.linspace(0, len(d) - 1, 19).astype(int)
        x = frame.iloc[indices][cols].replace([np.inf, -np.inf], np.nan)
        scores = est.decision_function(x)
        order = [list(est.classes_).index(m) for m in MODES]
        np.testing.assert_allclose(scores[:, order], d.iloc[indices][SCORES], atol=1e-10, rtol=1e-10)
        score_smokes += 1
    for path in (HERE / "feature_cache").glob("*.csv"):
        d = read(path)
        # New feature metadata also includes trip_id, checked in the producer.
        pd.testing.assert_frame_equal(d[META], base[META])
        manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        assert sha(path) == manifest["file_sha256"]
    for seed in range(20261123, 20261128):
        records = [json.loads((HERE / f"feature_cache/OU30_sigma{sigma:02d}_seed{seed}.json")
                             .read_text(encoding="utf-8")) for sigma in (5, 10, 20)]
        field = [{r["user_id"]: r["standard_field_sha256"] for r in x["user_diagnostics"]} for x in records]
        assert field[0] == field[1] == field[2]
        np.testing.assert_allclose(
            [r["realized_radial_rms_m"] / r["sigma"] for r in records],
            records[0]["realized_radial_rms_m"] / 5, rtol=1e-12)
    protected = json.loads((HERE / "input_manifest.json").read_text(encoding="utf-8"))
    for rel, digest in protected["files"].items():
        assert sha(ROOT / rel) == digest
    # Official distribution's installed API doc is available even if website SSL fails.
    (HERE / "literature/installed_nystroem_api.txt").write_text(
        "Installed official scikit-learn distribution, version 1.9.1\n\n" + Nystroem.__doc__,
        encoding="utf-8")
    result = {
        "passed": True, "metric_rows_reconstructed": len(rows),
        "per_class_f1_values": len(rows) * 5, "boundary_rows_greedy_verified": boundary_recomputed,
        "summary_rows_all_metrics_and_CIs": len(summary), "paired_contrast_rows": len(contrasts),
        "LOO_seed_rows_reconstructed": len(loo), "smoothing_sequences_independently_recomputed": smoothing_recomputed,
        "new_models_score_smoke_files": score_smokes, "sampled_windows_per_score_file": 19,
        "OU_cache_metadata_and_shared_field_hashes": 15,
        "protected_files_unchanged": len(protected["files"]),
        "elapsed_seconds": time.monotonic() - started,
        "verifier_sha256": sha(Path(__file__)),
        "manual_visual_review": "pending",
    }
    (HERE / "independent_verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
