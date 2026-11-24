"""Descriptive analysis of frozen predictions; no fitting or feature generation."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "src/model_training"))
from models import MODES, cv_groups

SEEDS = (20261123, 20261124, 20261125, 20261126, 20261127)
DT = ("<=1s", "1-2s", "2-5s", ">5s", "invalid_dt")
SPEED = ("<=0.5mps", "0.5-2mps", ">2mps", "invalid_speed")
META = ["user_id", "feature_block_id", "start_time", "end_time", "n_points",
        "label", "purity", *[f"cnt_{m}" for m in MODES]]
SCORES = [f"score_{m}" for m in MODES]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory():
    paths = [p for folder in ("data", "src", "tests", "docs", "experiments")
             for p in (ROOT / folder).rglob("*") if p.is_file()
             and HERE not in p.parents and "__pycache__" not in p.parts
             and "matplotlib_cache" not in p.parts and p.suffix != ".pyc"
             and p != ROOT / "docs/aigc_log_part2.md"]
    paths += [ROOT / n for n in ("README.md", "CLAUDE.md", "pyproject.toml", "uv.lock")]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def js(name, obj):
    path = HERE / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def csv(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(HERE / "tables" / f"{name}.csv", index=False)
    return frame


def read(path):
    return pd.read_csv(path, dtype={"user_id": str})


def conditions():
    return [(0, 0), *[(s, k) for s in (5, 10, 20) for k in SEEDS]]


def bins(base):
    dt = base.dt_mean.to_numpy(float)
    v = base.speed_mean.to_numpy(float)
    return (np.select([~np.isfinite(dt) | (dt <= 0), dt <= 1, dt <= 2, dt <= 5],
                      ["invalid_dt", "<=1s", "1-2s", "2-5s"], default=">5s"),
            np.select([~np.isfinite(v) | (v < 0), v <= .5, v <= 2],
                      ["invalid_speed", "<=0.5mps", "0.5-2mps"], default=">2mps"))


def ids(pred):
    idx = pd.Index(MODES).get_indexer(pred)
    assert (idx >= 0).all()
    return idx


def cm(counts, pred):
    matrix = np.zeros((5, 5), np.int64)
    for j in range(5):
        matrix[:, j] = counts[pred == j].sum(0)
    return matrix


def f1(matrix):
    den = matrix.sum(0) + matrix.sum(1)
    out = np.divide(2*np.diag(matrix), den, out=np.zeros(5), where=den > 0)
    return out.mean()


def ratio(a, b):
    return a/b if b else np.nan


def med(values):
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if len(finite) else np.nan


def flow(counts, clean, noisy):
    weight = counts.sum(1)
    old_correct = counts[np.arange(len(clean)), clean]
    new_correct = counts[np.arange(len(noisy)), noisy]
    changed = clean != noisy
    damaged = old_correct[changed].sum()
    rescued = new_correct[changed].sum()
    assert old_correct.sum()-new_correct.sum() == damaged-rescued
    return {
        "clean_correct": int(old_correct.sum()), "noisy_correct": int(new_correct.sum()),
        "damaged": int(damaged), "rescued": int(rescued),
        "changed_prediction_points": int(weight[changed].sum()),
        "damage_given_clean_correct": ratio(damaged, old_correct.sum()),
        "rescue_given_clean_wrong": ratio(rescued, weight.sum()-old_correct.sum()),
    }


def summary(name, frame, keys, values):
    agg = frame.groupby(keys, sort=False, dropna=False)[values].agg(["mean", "std"])
    agg.columns = [f"{x}_{a}" for x, a in agg.columns]
    agg = agg.reset_index()
    n = frame.groupby(keys, sort=False, dropna=False).size().reset_index(name="n_seeds")
    return csv(name, agg.merge(n, on=keys, validate="1:1"))


def main():
    start = time.monotonic()
    (HERE / "tables").mkdir(exist_ok=True)
    before = inventory()
    js("input_manifest.json", {
        "time": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "protected": before, "spec_sha256": sha(HERE / "ANALYSIS_SPEC.md"),
        "runner_sha256": sha(Path(__file__)),
        "versions": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scikit-learn", "joblib")},
    })
    base = read(EXP / "feature_cache/sigma00_seed0.csv")
    counts = base[[f"cnt_{m}" for m in MODES]].to_numpy(np.int64)
    weights = counts.sum(1)
    np.testing.assert_array_equal(weights, base.n_points)
    assert len(base) == 17380 and weights.sum() == 860972
    dt, speed = bins(base)
    linked = np.asarray(cv_groups(base.user_id, ROOT / "data/splits/report.json"), str)
    groups = pd.unique(linked)
    assert len(groups) == 9 and base.user_id.nunique() == 13
    cells = [(d, v, (dt == d) & (speed == v)) for d in DT for v in SPEED]
    assert np.stack([mask for _, _, mask in cells]).sum(0).min() == 1
    assert np.stack([mask for _, _, mask in cells]).sum(0).max() == 1
    members = base[META].copy()
    members["dt_bin"], members["speed_bin"], members["linked_group"] = dt, speed, linked
    members["window_row"] = np.arange(len(base))
    members.to_parquet(HERE / "membership.parquet", index=False)
    composition = []
    for d, v, mask in cells:
        for g in ["all", *groups]:
            keep = mask if g == "all" else mask & (linked == g)
            w = weights[keep].sum()
            composition.append({
                "dt_bin": d, "speed_bin": v, "linked_group": g,
                "windows": int(keep.sum()), "points": int(w),
                "test_point_fraction": w/weights.sum(),
                **{f"support_{m}": int(a) for m, a in zip(MODES, counts[keep].sum(0))},
            })
    csv("composition", composition)
    model = joblib.load(EXP / "models/G1.joblib")
    cols = json.loads((EXP / "models/G1_config.json").read_text())["feature_columns"]
    imp, scale, rbf, clf = [model.named_steps[n] for n in ("impute", "scale", "rbf", "clf")]
    landmarks = rbf.components_
    lm_scores = clf.decision_function(rbf.transform(landmarks))
    lm_classes = model.classes_[lm_scores.argmax(1)]
    csv("g1_landmark_responses", [
        {"landmark_index": i, "frozen_model_response": lm_classes[i],
         **{f"standardized_{c}": x for c, x in zip(cols, landmarks[i])},
         **{f"score_{m}": lm_scores[i, j] for j, m in enumerate(model.classes_)}}
        for i in range(len(landmarks))])
    np.testing.assert_array_equal(model.predict(pd.DataFrame(
        scale.inverse_transform(landmarks), columns=cols)), lm_classes)
    clean_z = scale.transform(imp.transform(base[cols]))
    clean_pred = {g: ids(read(EXP / f"predictions/{g}_sigma00_seed0_smoothed.csv").raw)
                  for g in ("G1", "G2", "G3")}
    with np.load(EXP / "followup_rounds/diagnostics/original_G1_sigma00_seed0.npz") as a:
        clean_nearest = a["nearest"].copy()
    clean_lm_response = ids(lm_classes[clean_nearest])
    clean_score = read(EXP / "predictions/G1_sigma00_seed0_smoothed.csv")[SCORES].to_numpy()
    clean_gap = np.sort(clean_score, axis=1)[:, -1] - np.sort(clean_score, axis=1)[:, -2]
    raw_reference = read(EXP / "followup_rounds/tables/raw_metrics_by_seed.csv")
    joint_rows, kernel_rows, transitions, feature_rows, checks = [], [], [], [], []
    g1_scores = []
    feature_masks = [("all", np.ones(len(base), bool)),
                     *[(m, base.label.eq(m).to_numpy()) for m in MODES]]
    all_cms, cm_keys = [], []
    for sigma, seed in conditions():
        cache = read(EXP / f"feature_cache/sigma{sigma:02d}_seed{seed}.csv")
        pd.testing.assert_frame_equal(cache[META], base[META], check_dtype=False)
        np.testing.assert_allclose(cache[["dt_mean", "dt_max"]], base[["dt_mean", "dt_max"]],
                                   rtol=0, atol=0, equal_nan=True)
        x = scale.transform(imp.transform(cache[cols]))
        change = x-clean_z
        for majority, mask in feature_masks:
            for j, c in enumerate(cols):
                feature_rows.append({
                    "sigma": sigma, "seed": seed, "majority_class": majority,
                    "feature": c, "windows": int(mask.sum()),
                    "clean_finite_values": int(np.isfinite(base.loc[mask, c]).sum()),
                    "noisy_finite_values": int(np.isfinite(cache.loc[mask, c]).sum()),
                    "clean_median": med(base.loc[mask, c]),
                    "noisy_median": med(cache.loc[mask, c]),
                    "standardized_signed_shift_median": med(change[mask, j]),
                    "standardized_abs_shift_median": med(np.abs(change[mask, j])),
                    "increase_window_fraction": np.mean(change[mask, j] > 0) if mask.any() else np.nan,
                    "decrease_window_fraction": np.mean(change[mask, j] < 0) if mask.any() else np.nan,
                })
        for group in ("G1", "G2", "G3"):
            p = read(EXP / f"predictions/{group}_sigma{sigma:02d}_seed{seed}_smoothed.csv")
            pd.testing.assert_frame_equal(p[META], base[META], check_dtype=False)
            pred = ids(p.raw)
            clean = clean_pred[group]
            global_cm = cm(counts, pred)
            ref = raw_reference.query(
                "family == 'original' and group == @group and sigma == @sigma and seed == @seed").iloc[0]
            np.testing.assert_allclose(f1(global_cm), ref.macro_f1, atol=1e-14)
            diag = np.load(EXP / f"followup_rounds/diagnostics/original_{group}_sigma{sigma:02d}_seed{seed}.npz")
            maxk, nearest = diag["max_kernel"], diag["nearest"]
            assert ((maxk >= 0) & (maxk <= 1+1e-12)).all()
            condition = dict(group=group, sigma=sigma, seed=seed)
            for d, v, cell in cells:
                for g in ["all", *groups]:
                    keep = cell if g == "all" else cell & (linked == g)
                    local = counts[keep]
                    mat = cm(local, pred[keep])
                    total = int(local.sum())
                    old_cm = cm(local, clean[keep])
                    flows = flow(local, clean[keep], pred[keep])
                    row = {
                        **condition, "dt_bin": d, "speed_bin": v, "linked_group": g,
                        "windows": int(keep.sum()), "points": total,
                        "macro_f1": f1(mat) if total else np.nan,
                        "clean_macro_f1": f1(old_cm) if total else np.nan,
                        "accuracy": ratio(np.trace(mat), total),
                        "clean_accuracy": ratio(np.trace(old_cm), total),
                        "max_kernel_median": med(maxk[keep]), **flows,
                        **{f"support_{m}": int(a) for m, a in zip(MODES, mat.sum(1))},
                        **{f"predicted_{m}": int(a) for m, a in zip(MODES, mat.sum(0))},
                        **{f"tp_{m}": int(a) for m, a in zip(MODES, mat.diagonal())},
                    }
                    row["macro_f1_drop"] = row["clean_macro_f1"]-row["macro_f1"]
                    row["accuracy_drop"] = row["clean_accuracy"]-row["accuracy"]
                    row["damaged_point_fraction"] = ratio(flows["damaged"], total)
                    row["rescued_point_fraction"] = ratio(flows["rescued"], total)
                    joint_rows.append(row)
                    cm_keys.append(f"{group}|{sigma}|{seed}|{d}|{v}|{g}")
                    all_cms.append(mat)
            if group == "G1":
                lm_resp = ids(lm_classes[nearest])
                score = p[SCORES].to_numpy()
                sorted_score = np.sort(score, axis=1)
                gap = sorted_score[:, -1]-sorted_score[:, -2]
                # All coordinate shifts are diagnostics, never altered model input.
                kernel_masks = [
                    ("all", np.ones(len(base), bool)),
                    ("[0,0.1)", maxk < .1),
                    ("[0.1,0.5)", (maxk >= .1) & (maxk < .5)),
                    ("[0.5,0.9)", (maxk >= .5) & (maxk < .9)),
                    ("[0.9,1]", maxk >= .9),
                ]
                for label, keep in kernel_masks:
                    total = weights[keep].sum()
                    local = cm(counts[keep], pred[keep])
                    kernel_rows.append({
                        "sigma": sigma, "seed": seed, "max_kernel_bin": label,
                        "windows": int(keep.sum()), "points": int(total),
                        "point_fraction": total/weights.sum(),
                        "accuracy": ratio(np.trace(local), total),
                        "max_kernel_median": med(maxk[keep]), "score_gap_median": med(gap[keep]),
                        "clean_score_gap_same_windows_median": med(clean_gap[keep]),
                        "nearest_identity_changed_point_fraction": ratio(
                            weights[keep & (nearest != clean_nearest)].sum(), total),
                        "nearest_response_changed_point_fraction": ratio(
                            weights[keep & (lm_resp != clean_lm_response)].sum(), total),
                        "model_nearest_response_agreement_point_fraction": ratio(
                            weights[keep & (pred == lm_resp)].sum(), total),
                        **flow(counts[keep], clean[keep], pred[keep]),
                    })
                for a, left in enumerate(MODES):
                    for b, right in enumerate(MODES):
                        keep = (clean_lm_response == a) & (lm_resp == b)
                        transitions.append({
                            "sigma": sigma, "seed": seed,
                            "clean_nearest_model_response": left,
                            "noisy_nearest_model_response": right,
                            "windows": int(keep.sum()), "points": int(weights[keep].sum()),
                            **flow(counts[keep], clean[keep], pred[keep]),
                            **{f"true_{m}": int(z) for m, z in zip(MODES, counts[keep].sum(0))},
                            **{f"model_predicted_{m}": int(weights[keep & (pred == j)].sum())
                               for j, m in enumerate(MODES)},
                        })
                g1_scores.append({
                    "sigma": sigma, "seed": seed, "score_gap_median": med(gap),
                    "clean_score_gap_median": med(clean_gap),
                    "correct_majority_score_gap_median": med(gap[p.raw.eq(base.label)]),
                    "wrong_majority_score_gap_median": med(gap[p.raw.ne(base.label)]),
                    "window_majority_accuracy": float(p.raw.eq(base.label).mean()),
                })
            checks.append({**condition, "windows": len(p), "points": int(weights.sum()),
                           "macro_f1": f1(global_cm), "score_argmax_matches": bool(
                               np.array_equal(np.asarray(MODES)[p[SCORES].to_numpy().argmax(1)], p.raw))})
            assert checks[-1]["score_argmax_matches"]
            diag.close()
        js("checkpoint.json", {
            "completed_conditions": len(checks), "last": {"sigma": sigma, "seed": seed},
            "elapsed_seconds": time.monotonic()-start,
            "updated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat()})
        print(f"Completed {len(checks)}/48 model-condition diagnostics", flush=True)
    joint = csv("joint_metrics_by_seed", joint_rows)
    keys = ["group", "sigma", "dt_bin", "speed_bin", "linked_group"]
    values = ["macro_f1", "clean_macro_f1", "macro_f1_drop", "accuracy", "clean_accuracy",
              "accuracy_drop", "damaged", "rescued", "damaged_point_fraction",
              "rescued_point_fraction", "damage_given_clean_correct", "rescue_given_clean_wrong",
              "changed_prediction_points", "max_kernel_median"]
    summary("joint_summary", joint, keys, values)
    kernel = csv("g1_kernel_bins_by_seed", kernel_rows)
    summary("g1_kernel_bins_summary", kernel, ["sigma", "max_kernel_bin"],
            [n for n in kernel if n not in ("sigma", "seed", "max_kernel_bin")])
    transition = csv("g1_landmark_transitions_by_seed", transitions)
    summary("g1_landmark_transitions_summary", transition,
            ["sigma", "clean_nearest_model_response", "noisy_nearest_model_response"],
            [n for n in transition if n not in (
                "sigma", "seed", "clean_nearest_model_response", "noisy_nearest_model_response")])
    features = csv("g1_feature_shifts_by_seed", feature_rows)
    summary("g1_feature_shifts_summary", features, ["sigma", "majority_class", "feature"],
            ["windows", "clean_finite_values", "noisy_finite_values",
             "clean_median", "noisy_median", "standardized_signed_shift_median",
             "standardized_abs_shift_median", "increase_window_fraction", "decrease_window_fraction"])
    csv("g1_score_gaps_by_seed", g1_scores)
    csv("condition_checks", checks)
    np.savez_compressed(HERE / "joint_confusions.npz", keys=np.asarray(cm_keys), cms=np.stack(all_cms))
    # All joint cells and linked groups must add up independently.
    for group in ("G1", "G2", "G3"):
        for sigma, seed in conditions():
            block = joint[(joint.group == group) & (joint.sigma == sigma) & (joint.seed == seed)]
            pooled = block[block.linked_group == "all"]
            specific = block[block.linked_group != "all"]
            for col in ["points", "windows", "clean_correct", "noisy_correct", "damaged", "rescued",
                        *[f"support_{m}" for m in MODES], *[f"predicted_{m}" for m in MODES]]:
                assert pooled[col].sum() == specific[col].sum()
            assert pooled.points.sum() == 860972 and pooled.windows.sum() == 17380
            if sigma == 0:
                assert pooled.damaged.sum() == pooled.rescued.sum() == 0
    after = inventory()
    assert before == after, "Protected content or membership changed"
    js("acceptance.json", {
        "status": "passed", "model_conditions": len(checks),
        "joint_rows": len(joint), "protected_files": len(before),
        "protected_content_and_membership_unchanged": True,
        "new_fitting_or_noise_or_bootstrap": False, "elapsed_seconds": time.monotonic()-start,
        "completed_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat()})
    js("checkpoint.json", {"status": "complete", "completed_model_conditions": len(checks),
                            "elapsed_seconds": time.monotonic()-start})


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
