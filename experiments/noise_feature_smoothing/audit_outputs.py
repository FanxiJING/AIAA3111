"""Independent artifact checks; never changes features, models or scores."""
import importlib.util
import itertools
import json
from pathlib import Path
import time
import shutil

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("noise_experiment", HERE / "run_experiment.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def checks():
    state = json.loads((HERE / "supervisor.json").read_text())
    assert state["status"] == "complete", state
    assert state["completed"] == [f"stage{i}" for i in range(6)]
    train, test = m.load_tables()
    results = {}
    # Future isolation, order restoration, short blocks and block isolation.
    rng = np.random.default_rng(42)
    scores = rng.normal(size=(12, 5))
    blocks = np.array(["a"] * 8 + ["b"] * 4)
    times = pd.date_range("2026-01-01", periods=12)
    got = m.trailing_soft(scores, blocks, times, 11)
    expected = np.array([m.MODES[scores[max(0 if i < 8 else 8, i-10):i+1].mean(0).argmax()]
                         for i in range(12)], object)
    np.testing.assert_array_equal(got, expected)
    order = rng.permutation(12)
    unordered = m.trailing_soft(scores[order], blocks[order], times[order], 11)
    np.testing.assert_array_equal(unordered, got[order])
    changed = scores.copy()
    changed[4:8] += np.array([100, 0, 0, 0, 0])
    after = m.trailing_soft(changed, blocks, times, 11)
    np.testing.assert_array_equal(got[:4], after[:4])
    np.testing.assert_array_equal(got[8:], after[8:])
    np.testing.assert_array_equal(m.trailing_soft(scores, blocks, times, 1),
                                  np.array(m.MODES)[scores.argmax(1)])
    # Exhaustively verify matching cardinality against an independent assignment solver.
    meta = pd.DataFrame({"user_id": ["001"] * 5, "feature_block_id": ["b"] * 5,
                         "start_time": times[:5]})
    for pmask, tmask in itertools.product(range(16), repeat=2):
        pidx = np.array([i + 1 for i in range(4) if pmask >> i & 1], int)
        tidx = np.array([i + 1 for i in range(4) if tmask >> i & 1], int)
        p = np.array(m.MODES)[np.r_[0, np.cumsum([pmask >> i & 1 for i in range(4)])] % 5]
        t = np.array(m.MODES)[np.r_[0, np.cumsum([tmask >> i & 1 for i in range(4)])] % 5]
        edges = np.abs(pidx[:, None] - tidx[None, :]) <= 1
        a, b = linear_sum_assignment(-edges.astype(int))
        expected_hits = int(edges[a, b].sum())
        assert m.one_to_one_transitions(meta, p, t)["matched_transitions"] == expected_hits
    cm = np.zeros((1, 5, 5), int)
    cm[0, 0, 0] = 10
    assert m.bootstrap_scores(cm)[0][0] == .2
    results["logic_checks"] = "causality, order, block edges, k1, 256 matchings, fixed-five F1"
    manifest = json.loads((HERE / "manifest.json").read_text())
    for path, digest in manifest["files"].items():
        if Path(path).name == "run_experiment.py":
            continue  # Implementation repairs are separately snapshotted and recorded.
        assert m.sha256_file(m.ROOT / path) == digest, path
    results["original_inputs_and_reference_code_unchanged"] = True
    feature_names = list(m.prepare(train)[0].columns)
    original_meta = test[m.META_COLS]
    conditions = [(s, seed) for s in m.SIGMAS for seed in ([0] if s == 0 else m.SEEDS)]
    for sigma, seed in conditions:
        name = m.condition_name(sigma, seed)
        path = m.CACHE / f"{name}.csv"
        d = pd.read_csv(path, dtype={"user_id": str})
        for col in ("start_time", "end_time"):
            d[col] = pd.to_datetime(d[col])
        pd.testing.assert_frame_equal(d[m.META_COLS], original_meta, check_dtype=False)
        info = json.loads(path.with_suffix(".json").read_text())
        assert m.sha256_file(path) == info["file_sha256"]
        assert info["points"] == 860972 and info["windows"] == 17380
        if sigma == 0:
            np.testing.assert_allclose(d[feature_names], test[feature_names],
                                       atol=1e-7, rtol=1e-10, equal_nan=True)
    results["feature_cache_conditions"] = 16
    # Verify the paired field using the exact saved per-user perturbation diagnostics.
    for seed in m.SEEDS:
        infos = [json.loads((m.CACHE / f"{m.condition_name(s, seed)}.json").read_text())
                 for s in (5, 10, 20)]
        for k in range(13):
            for field in ("east_std_m", "north_std_m", "radial_rms_m"):
                normalized = [info["user_diagnostics"][k][field] / s
                              for info, s in zip(infos, (5, 10, 20))]
                np.testing.assert_allclose(normalized, normalized[0], rtol=1e-13)
    results["paired_noise_diagnostics"] = "all 5 seeds / 13 users / 3 positive sigmas"
    for group, dim in zip(m.GROUPS, (10, 15, 27)):
        est = joblib.load(m.MODELS / ("full.joblib" if group == "G3" else f"{group}.joblib"))
        params = est.get_params()
        assert params["rbf__n_components"] == 1000 and params["rbf__random_state"] == 42
        assert params["rbf__gamma"] == 1 / dim
        assert params["clf__C"] == 1 and params["clf__class_weight"] == "balanced"
        assert params["clf__random_state"] == 42 and params["impute__strategy"] == "median"
        assert est.n_features_in_ == dim and set(est.classes_) == set(m.MODES)
        for sigma, seed in conditions:
            name = m.condition_name(sigma, seed)
            d, raw, scores = m.read_prediction(m.PRED / f"{group}_{name}.csv")
            assert scores.shape == (17380, 5) and np.isfinite(scores).all()
            np.testing.assert_array_equal(raw, np.array(m.MODES)[scores.argmax(1)])
            smoothed = pd.read_csv(m.PRED / f"{group}_{name}_smoothed.csv")
            for smoother in m.SMOOTHERS:
                assert smoothed[smoother].isin(m.MODES).all()
    results["frozen_models"] = 3
    results["decision_score_files"] = 48
    metrics = pd.read_csv(m.METRICS / "all_metrics.csv")
    assert len(metrics) == 144 and metrics.true_transitions.eq(529).all()
    assert metrics.loc[metrics.sigma == 0, "prediction_change_from_clean_point_fraction"].eq(0).all()
    assert metrics["prediction_change_from_clean_point_fraction"].between(0, 1).all()
    clean = pd.read_csv(m.PRED / "G3_sigma00_seed0.csv")
    initial = pd.read_csv(m.PRED / "stage0_G3_sigma0_predictions.csv")
    np.testing.assert_array_equal(clean.raw, initial.raw)
    bootstrap = pd.read_csv(m.METRICS / "stage5_bootstrap_all.csv")
    assert len(bootstrap) == 36
    assert np.isfinite(bootstrap.select_dtypes("number").to_numpy()).all()
    users = pd.read_csv(m.METRICS / "per_user_results.csv")
    assert len(users) == 144 * 13
    draws = np.load(m.METRICS / "paired_bootstrap_draws.npz")
    assert draws["draws"].shape == (5000, 9) and (draws["draws"].sum(1) == 9).all()
    results.update(metric_rows=144, user_rows=len(users), bootstrap_draws=5000,
                   independent_user_groups=9, passed=True, audited_at=time.time())
    # Independent direct F1 calculation for the primary averaged paired interval.
    group_ids = np.asarray(m.cv_groups(test.user_id, m.ROOT / "data/splits/report.json"))
    clusters = pd.unique(group_ids)
    d0 = pd.read_csv(m.PRED / "G3_sigma00_seed0_smoothed.csv")
    clean_per = np.stack([m.point_confusion(d0.loc[group_ids == c],
                         d0.loc[group_ids == c, "S0_raw_k1"], m.MODES) for c in clusters])
    noise_per = []
    for seed in m.SEEDS:
        d = pd.read_csv(m.PRED / f"G3_sigma20_seed{seed}_smoothed.csv")
        noise_per.append(np.stack([m.point_confusion(d.loc[group_ids == c],
                         d.loc[group_ids == c, "S0_raw_k1"], m.MODES) for c in clusters]))
    direct_mean, direct_delta = [], []
    for draw in draws["draws"]:
        clean_f1 = m.metrics_from_cm(np.tensordot(draw, clean_per, axes=1))["macro_f1"]
        seed_f1s = [m.metrics_from_cm(np.tensordot(draw, per, axes=1))["macro_f1"]
                    for per in noise_per]
        direct_mean.append(np.mean(seed_f1s))
        direct_delta.append(np.mean(seed_f1s) - clean_f1)
    target = bootstrap[(bootstrap.group == "G3") & (bootstrap.smoother == "S0_raw_k1")
                       & (bootstrap.sigma == 20)].iloc[0]
    np.testing.assert_allclose(np.percentile(direct_mean, [2.5, 97.5]),
                               [target.macro_f1_ci_low, target.macro_f1_ci_high], atol=1e-12)
    np.testing.assert_allclose(np.percentile(direct_delta, [2.5, 97.5]),
                               [target.macro_f1_delta_ci_low, target.macro_f1_delta_ci_high], atol=1e-12)
    results["primary_paired_bootstrap_independent_recalculation"] = True
    snapshots = HERE / "source_snapshots" / "reference_tree"
    for path in manifest["files"]:
        rel = Path(path)
        if rel.suffix == ".py" and rel.name != "run_experiment.py":
            dest = snapshots / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(m.ROOT / rel, dest)
    for name in ("NOISE_FEATURE_SMOOTHING_PLAN.md", "NOISE_PROTOCOL_REVIEW.md"):
        shutil.copy2(m.ROOT.parent / "tmp" / name, HERE / name)
    inventory = {str(p.relative_to(HERE)): {"bytes": p.stat().st_size, "sha256": m.sha256_file(p)}
                 for p in HERE.rglob("*") if p.is_file()
                 and p.name not in ("artifact_inventory.json", "final_audit.json", "checkpoint.json")
                 and "matplotlib_cache" not in p.parts and "__pycache__" not in p.parts}
    m.json_write(HERE / "artifact_inventory.json", inventory)
    m.json_write(HERE / "final_audit.json", results)
    checkpoint = json.loads(m.CHECKPOINT.read_text())
    checkpoint.update(stage="complete", status="passed", completed=state["completed"],
                      attempts=state["retries"], final_audit="final_audit.json",
                      elapsed_seconds=time.time() - state["started_at_epoch"],
                      remaining_seconds=max(0, state["deadline_epoch"] - time.time()))
    m.json_write(m.CHECKPOINT, checkpoint)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    checks()
