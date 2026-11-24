"""Audit committed v2 artifacts without training or changing model choices.

Run from any directory: python <path>/audit_local_results.py
Outputs go to outputs/local_audit/, leaving the historical reference intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from models import MODES, cv_groups, metrics_from_cm

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=HERE / "results/basic_w50_no_dt_v2")
    ap.add_argument("--out", type=Path, default=PROJECT / "outputs/local_audit")
    a = ap.parse_args()
    root = a.results.resolve()
    checks = []

    def check(name, passed):
        checks.append({"check": name, "passed": bool(passed)})

    tests = pd.read_csv(root / "test_results.csv")
    preds = pd.read_csv(root / "window_predictions_test.csv", dtype={"user_id": str})
    users = pd.read_csv(root / "test_per_user.csv", dtype={"user_id": str})
    selected = json.loads((root / "selection.json").read_text())["selected"]
    row = tests.set_index("model").loc[selected]
    matrix = pd.read_csv(root / "confusion_matrix_final_points.csv", index_col=0).loc[MODES, MODES]
    metrics = metrics_from_cm(matrix.to_numpy())
    train_ids = set(json.loads((PROJECT / "data/splits/train_users.json").read_text()))
    test_ids = set(json.loads((PROJECT / "data/splits/test_users.json").read_text()))
    check("train_test_users_disjoint", not train_ids & test_ids)
    check("prediction_users_equal_fixed_test_users", set(preds.user_id) == test_ids)
    check("prediction_point_total_860972", preds.n_points.sum() == 860972)
    check("confusion_point_total_matches_predictions", matrix.to_numpy().sum() == preds.n_points.sum())
    check("confusion_reproduces_macro_f1", abs(metrics["macro_f1"] - row.point_macro_f1) < 0.000051)
    check("confusion_reproduces_accuracy", abs(metrics["accuracy"] - row.point_accuracy) < 0.000051)
    # build_windows merges tails shorter than W/2: W=50 allows up to 74 points.
    check("all_windows_within_w50_tail_merge_rule", preds.n_points.between(1, 74).all())
    check("each_block_has_one_user_and_trip",
          preds.groupby("feature_block_id")[["user_id", "trip_id"]].nunique().le(1).all().all())
    check("window_times_valid", (pd.to_datetime(preds.start_time) <= pd.to_datetime(preds.end_time)).all())
    ordered = preds.sort_values(["feature_block_id", "start_time"])
    previous_end = ordered.groupby("feature_block_id").end_time.shift()
    check("windows_do_not_overlap_inside_blocks",
          (previous_end.isna() | (ordered.start_time > previous_end)).all())
    groups = cv_groups(preds.user_id, PROJECT / "data/splits/report.json")
    check("nine_independent_test_groups", groups.nunique() == 9)
    point_users = preds.groupby("user_id").n_points.sum()
    for model, per_user in users.groupby("model"):
        per_user = per_user.set_index("user_id").sort_index()
        reported = tests.set_index("model").loc[model]
        check(f"{model}: per_user_point_counts",
              per_user.points.equals(point_users.reindex(per_user.index).rename("points")))
        check(f"{model}: weighted_accuracy",
              abs(np.average(per_user.accuracy, weights=per_user.points) - reported.point_accuracy) < 0.00011)
        check(f"{model}: user_average_macro_f1",
              abs(per_user.macro_f1.mean() - reported.user_avg_macro_f1) < 0.00011)
        check(f"{model}: prediction_labels", preds[model].isin(MODES).all())
    cv = pd.read_csv(root / "cv_summary.csv")
    check("selected_model_matches_cv_summary", cv.loc[cv.selected, "model"].tolist() == [selected])
    check("cv_score_matches_test_metadata",
          abs(cv.set_index("model").loc[selected].cv_macro_f1_mean - row.cv_macro_f1) < 0.000051)
    per_class = pd.read_csv(root / "test_per_class_f1.csv")
    reported_class = per_class[(per_class.model == selected) & (per_class.version == "final")]
    check("confusion_reproduces_class_f1",
          np.allclose(reported_class[MODES].to_numpy()[0], metrics["per_class_f1"], atol=0.000051, rtol=0))
    concentration = pd.DataFrame({"group": groups, "points": preds.n_points}).groupby("group").points.sum()
    concentration = concentration.sort_values(ascending=False).rename_axis("group").reset_index()
    concentration["fraction"] = concentration.points / concentration.points.sum()
    inputs = [HERE / "windows_w50.csv", HERE / "windows_w50_blind.csv",
              HERE / "windows_w50_pure_all.csv", PROJECT / "data/processed/quality_features"]
    inventory = [{"path": str(p), "exists": p.exists()} for p in inputs]
    manifest = [{"path": str(p.relative_to(PROJECT)), "bytes": p.stat().st_size,
                 "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(root.rglob("*")) if p.is_file()]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True).strip()
    report = {"scope": "Historical artifact audit; no training or new condition evaluated",
              "git_head": commit, "selected_model": selected,
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "pandas": pd.__version__, "sklearn": sklearn.__version__},
              "point_macro_f1_recomputed": metrics["macro_f1"],
              "point_accuracy_recomputed": metrics["accuracy"], "checks": checks,
              "training_input_inventory": inventory,
              "limitations": ["Predictions omit per-class counts for mixed windows; full point-level recomputation from predictions requires the window bundle.",
                              "Artifact consistency does not prove the historical training run is reproducible.",
                              "Local environment differs from the recorded historical environment."]}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (a.out / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    concentration.to_csv(a.out / "test_group_concentration.csv", index=False)
    pd.DataFrame(checks).to_csv(a.out / "checks.csv", index=False)
    print(f"Artifact checks: {sum(c['passed'] for c in checks)}/{len(checks)} passed")
    print(f"Historical selected model: {selected}")
    print(f"Recomputed from confusion matrix: F1={metrics['macro_f1']:.6f}, accuracy={metrics['accuracy']:.6f}")
    print(f"Missing training inputs: {sum(not x['exists'] for x in inventory)}/{len(inventory)}")
    print(f"Audit: {a.out.resolve()}")
    if not all(c["passed"] for c in checks):
        raise SystemExit("Artifact inconsistencies found; see checks.csv")


if __name__ == "__main__":
    main()
