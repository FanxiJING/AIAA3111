"""Independent numerical verification; intentionally does not import the runner."""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics.pairwise import rbf_kernel
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
MODES = ["walk", "bike", "bus", "car", "train"]
sys.path.insert(0, str(ROOT / "src/model_training"))
from models import cv_groups


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, tol=2e-12):
    np.testing.assert_allclose(a, b, atol=tol, rtol=1e-11, equal_nan=True)


def main():
    base = pd.read_csv(EXP / "feature_cache/sigma00_seed0.csv", dtype={"user_id": str})
    counts = base[[f"cnt_{m}" for m in MODES]].to_numpy(np.int64)
    weight = base.n_points.to_numpy()
    membership = pd.read_parquet(HERE / "membership.parquet")
    linked = np.asarray(cv_groups(base.user_id, ROOT / "data/splits/report.json"), str)
    np.testing.assert_array_equal(membership.linked_group, linked)
    dtbins, speedbins = [], []
    for d, v in zip(base.dt_mean, base.speed_mean):
        dtbins.append("invalid_dt" if not np.isfinite(d) or d <= 0 else
                      "<=1s" if d <= 1 else "1-2s" if d <= 2 else "2-5s" if d <= 5 else ">5s")
        speedbins.append("invalid_speed" if not np.isfinite(v) or v < 0 else
                         "<=0.5mps" if v <= .5 else "0.5-2mps" if v <= 2 else ">2mps")
    np.testing.assert_array_equal(membership.dt_bin, dtbins)
    np.testing.assert_array_equal(membership.speed_bin, speedbins)
    joint = pd.read_csv(HERE / "tables/joint_metrics_by_seed.csv", dtype={"linked_group": str})
    with np.load(HERE / "joint_confusions.npz") as a:
        saved_cms = dict(zip(a["keys"], a["cms"]))
    g1 = joblib.load(EXP / "models/G1.joblib")
    landmark_responses = pd.read_csv(HERE / "tables/g1_landmark_responses.csv")
    rbf, scale = g1.named_steps["rbf"], g1.named_steps["scale"]
    g1cols = json.loads((EXP / "models/G1_config.json").read_text())["feature_columns"]
    predicted = g1.predict(pd.DataFrame(scale.inverse_transform(rbf.components_), columns=g1cols))
    np.testing.assert_array_equal(predicted, landmark_responses.frozen_model_response)
    kr = pd.read_csv(HERE / "tables/g1_kernel_bins_by_seed.csv")
    tr = pd.read_csv(HERE / "tables/g1_landmark_transitions_by_seed.csv")
    shifts = pd.read_csv(HERE / "tables/g1_feature_shifts_by_seed.csv")
    clean_idx = {}
    sample = np.unique(np.linspace(0, len(base)-1, 32, dtype=int))
    checked_cells = metric_checks = flow_checks = direct_checks = landmark_cells = 0
    with np.load(EXP / "followup_rounds/diagnostics/original_G1_sigma00_seed0.npz") as a:
        clean_nearest = a["nearest"].copy()
    old_lm = landmark_responses.frozen_model_response.to_numpy()[clean_nearest]
    for group in ("G1", "G2", "G3"):
        est = joblib.load(EXP / "models" / ("full.joblib" if group == "G3" else f"{group}.joblib"))
        cols = json.loads((EXP / f"models/{group}_config.json").read_text())["feature_columns"]
        clean = pd.read_csv(EXP / f"predictions/{group}_sigma00_seed0_smoothed.csv").raw.to_numpy()
        clean_idx[group] = pd.Categorical(clean, categories=MODES).codes
        for sigma in (0, 5, 10, 20):
            for seed in ([0] if sigma == 0 else [20261123, 20261124, 20261125, 20261126, 20261127]):
                p = pd.read_csv(EXP / f"predictions/{group}_sigma{sigma:02d}_seed{seed}_smoothed.csv")
                noisy = p.raw.to_numpy()
                sub = joint[(joint.group == group) & (joint.sigma == sigma) & (joint.seed == seed)]
                for row in sub.itertuples(index=False):
                    keep = (membership.dt_bin == row.dt_bin) & (membership.speed_bin == row.speed_bin)
                    if row.linked_group != "all":
                        keep &= membership.linked_group == row.linked_group
                    keep = np.asarray(keep)
                    c = counts[keep]
                    if keep.any():
                        actual_cm = confusion_matrix(
                            np.tile(MODES, keep.sum()), np.repeat(noisy[keep], 5),
                            labels=MODES, sample_weight=c.ravel()).astype(np.int64)
                    else:
                        actual_cm = np.zeros((5, 5), np.int64)
                    key = f"{group}|{sigma}|{seed}|{row.dt_bin}|{row.speed_bin}|{row.linked_group}"
                    np.testing.assert_array_equal(actual_cm, saved_cms[key])
                    assert c.sum() == row.points and keep.sum() == row.windows
                    for j, mode in enumerate(MODES):
                        assert actual_cm.sum(0)[j] == getattr(row, f"predicted_{mode}")
                        assert actual_cm.sum(1)[j] == getattr(row, f"support_{mode}")
                        assert actual_cm[j, j] == getattr(row, f"tp_{mode}")
                    if row.linked_group == "all" and row.points:
                        actual_f1 = f1_score(
                            np.tile(MODES, keep.sum()), np.repeat(noisy[keep], 5),
                            labels=MODES, sample_weight=c.ravel(), average="macro", zero_division=0)
                        close(actual_f1, row.macro_f1)
                        metric_checks += 1
                    damage = rescue = clean_correct = noisy_correct = 0
                    for j, mode in enumerate(MODES):
                        old_yes, new_yes = clean[keep] == mode, noisy[keep] == mode
                        damage += c[old_yes & ~new_yes, j].sum()
                        rescue += c[~old_yes & new_yes, j].sum()
                        clean_correct += c[old_yes, j].sum()
                        noisy_correct += c[new_yes, j].sum()
                    np.testing.assert_array_equal(
                        [damage, rescue, clean_correct, noisy_correct],
                        [row.damaged, row.rescued, row.clean_correct, row.noisy_correct])
                    assert weight[keep & (clean != noisy)].sum() == row.changed_prediction_points
                    checked_cells += 1
                    flow_checks += 1
                cache = pd.read_csv(EXP / f"feature_cache/sigma{sigma:02d}_seed{seed}.csv")
                actual_scores = est.decision_function(cache.loc[sample, cols])
                align = pd.Index(est.classes_).get_indexer(MODES)
                close(actual_scores[:, align], p.loc[sample, [f"score_{m}" for m in MODES]], tol=2e-7)
                x = est.named_steps["scale"].transform(
                    est.named_steps["impute"].transform(cache.loc[sample, cols]))
                components = est.named_steps["rbf"].components_
                squared = ((x[:, None, :]-components[None, :, :])**2).sum(2)
                direct_kernel = rbf_kernel(x, components, gamma=est.named_steps["rbf"].gamma)
                with np.load(EXP / f"followup_rounds/diagnostics/original_{group}_sigma{sigma:02d}_seed{seed}.npz") as a:
                    np.testing.assert_array_equal(squared.argmin(1), a["nearest"][sample])
                    close(direct_kernel.max(1), a["max_kernel"][sample], tol=2e-9)
                    all_nearest, all_maxk = a["nearest"], a["max_kernel"]
                direct_checks += len(sample)
                if group == "G1":
                    new_lm = landmark_responses.frozen_model_response.to_numpy()[all_nearest]
                    for row in tr[(tr.sigma == sigma) & (tr.seed == seed)].itertuples(index=False):
                        keep = ((old_lm == row.clean_nearest_model_response) &
                                (new_lm == row.noisy_nearest_model_response))
                        assert keep.sum() == row.windows
                        assert weight[keep].sum() == row.points
                        for j, mode in enumerate(MODES):
                            assert counts[keep, j].sum() == getattr(row, f"true_{mode}")
                            assert weight[keep & (noisy == mode)].sum() == getattr(row, f"model_predicted_{mode}")
                        landmark_cells += 1
                    for row in kr[(kr.sigma == sigma) & (kr.seed == seed)].itertuples(index=False):
                        label = row.max_kernel_bin
                        keep = (np.ones(len(base), bool) if label == "all" else
                                all_maxk < .1 if label == "[0,0.1)" else
                                (all_maxk >= .1) & (all_maxk < .5) if label == "[0.1,0.5)" else
                                (all_maxk >= .5) & (all_maxk < .9) if label == "[0.5,0.9)" else
                                all_maxk >= .9)
                        total = weight[keep].sum()
                        assert total == row.points and keep.sum() == row.windows
                        close(weight[keep & (noisy == new_lm)].sum()/total if total else np.nan,
                              row.model_nearest_response_agreement_point_fraction)
                    clean_z = scale.transform(g1.named_steps["impute"].transform(base[g1cols]))
                    noisy_z = scale.transform(g1.named_steps["impute"].transform(cache[g1cols]))
                    for row in shifts[(shifts.sigma == sigma) & (shifts.seed == seed)].itertuples(index=False):
                        keep = (np.ones(len(base), bool) if row.majority_class == "all" else
                                base.label.eq(row.majority_class).to_numpy())
                        j = g1cols.index(row.feature)
                        change = noisy_z[keep, j]-clean_z[keep, j]
                        close(np.median(change) if keep.any() else np.nan,
                              row.standardized_signed_shift_median)
                        close(np.median(np.abs(change)) if keep.any() else np.nan,
                              row.standardized_abs_shift_median)
                        for source, column, count_column in (
                            (base, "clean_median", "clean_finite_values"),
                            (cache, "noisy_median", "noisy_finite_values"),
                        ):
                            values = source.loc[keep, row.feature].to_numpy(float)
                            values = values[np.isfinite(values)]
                            assert len(values) == getattr(row, count_column)
                            close(np.median(values) if len(values) else np.nan,
                                  getattr(row, column))
                print(f"Verified {group} sigma={sigma} seed={seed}", flush=True)
    summary_checks = 0
    configs = [
        ("joint", ["group", "sigma", "dt_bin", "speed_bin", "linked_group"]),
        ("g1_kernel_bins", ["sigma", "max_kernel_bin"]),
        ("g1_landmark_transitions", ["sigma", "clean_nearest_model_response", "noisy_nearest_model_response"]),
        ("g1_feature_shifts", ["sigma", "majority_class", "feature"]),
    ]
    for prefix, keys in configs:
        rawfile = "joint_metrics_by_seed" if prefix == "joint" else f"{prefix}_by_seed"
        raw = pd.read_csv(HERE / f"tables/{rawfile}.csv", dtype={"linked_group": str})
        agg = pd.read_csv(HERE / f"tables/{prefix}_summary.csv", dtype={"linked_group": str})
        idx = raw.groupby(keys, sort=False, dropna=False)
        for row in agg.to_dict("records"):
            subset = idx.get_group(tuple(row[k] for k in keys))
            assert len(subset) == row["n_seeds"]
            for col, value in row.items():
                if col.endswith("_mean") or col.endswith("_std"):
                    stat, stem = col.rsplit("_", 1)[1], col.rsplit("_", 1)[0]
                    values = subset[stem].dropna().to_numpy(float)
                    expected = (values.mean() if stat == "mean" and len(values) else
                                values.std(ddof=1) if stat == "std" and len(values) > 1 else np.nan)
                    close(expected, value)
                    summary_checks += 1
    protected = json.loads((HERE / "input_manifest.json").read_text())["protected"]
    current = [p for folder in ("data", "src", "tests", "docs", "experiments")
               for p in (ROOT / folder).rglob("*") if p.is_file()
               and HERE not in p.parents and "__pycache__" not in p.parts
               and "matplotlib_cache" not in p.parts and p.suffix != ".pyc"
               and p != ROOT / "docs/aigc_log_part2.md"]
    current += [ROOT / n for n in ("README.md", "CLAUDE.md", "pyproject.toml", "uv.lock")]
    assert set(protected) == {p.relative_to(ROOT).as_posix() for p in current}
    for name, expected in protected.items():
        assert digest(ROOT / name) == expected, name
    outcome = {
        "status": "passed", "joint_confusion_and_flow_rows": checked_cells,
        "pooled_nonempty_weighted_sklearn_f1": metric_checks,
        "actual_pipeline_and_distance_samples": direct_checks,
        "landmark_response_transitions": landmark_cells, "summary_values": summary_checks,
        "landmark_model_responses": len(landmark_responses),
        "protected_files": len(protected), "protected_content_and_membership_unchanged": True,
        "verifier_sha256": digest(Path(__file__)),
        "verified_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
    }
    (HERE / "independent_verification.json").write_text(
        json.dumps(outcome, indent=2), encoding="utf-8")
    print(json.dumps(outcome), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
