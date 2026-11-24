"""Independently reconstruct final metrics from saved predictions and window class counts."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

from models import MODES, load_windows, metrics_from_cm, point_confusion

PROJECT = Path(__file__).resolve().parents[2]
MODEL = "SVM (RBF, Nystroem, all data)"


def main():
    checks = []
    frozen = json.loads((PROJECT / "outputs/sampling_results_reference/all_seven_choices_frozen_before_test.json").read_text())
    choices = {r["condition"]: r for r in frozen["choices"]}
    for prepared, results in [("sampling_main_v1", "sampling_results_reference"),
                              ("sampling_duration_v1", "sampling_duration_results_reference")]:
        for condition in sorted((PROJECT / "outputs" / prepared).glob("*/condition.json")):
            name = condition.parent.name
            root = PROJECT / "outputs" / results / name
            windows = load_windows(condition.parent / "windows.csv")
            if name == "interval_30s_time_120s":
                checks.append({"condition": name, "check": "infeasible_combination_has_no_test_results",
                               "passed": not (root / "test_results.csv").exists()
                               and not ((windows.split == "train") & (windows.n_points >= 10)).any()})
                continue
            test = windows[windows.split == "test"].reset_index(drop=True)
            pred = pd.read_csv(root / "window_predictions_test.csv", dtype={"user_id": str})
            row = pd.read_csv(root / "test_results.csv").set_index("model").loc[MODEL]
            def check(label, result):
                checks.append({"condition": name, "check": label, "passed": bool(result)})
            check("window_alignment", test[["user_id", "start_time", "end_time", "n_points"]].equals(
                  pred[["user_id", "start_time", "end_time", "n_points"]]))
            cm = point_confusion(test, pred[MODEL], MODES)
            saved = pd.read_csv(root / "confusion_matrix_final_points.csv", index_col=0).loc[MODES, MODES]
            check("confusion_matches_independent_reconstruction", np.array_equal(cm, saved))
            check("test_point_conservation", cm.sum() == test.n_points.sum())
            metrics = metrics_from_cm(cm)
            check("final_macro_f1", abs(metrics["macro_f1"] - row.point_macro_f1) <= 0.000051)
            check("final_accuracy", abs(metrics["accuracy"] - row.point_accuracy) <= 0.000051)
            raw = metrics_from_cm(point_confusion(test, pred[MODEL + " | raw"], MODES))
            check("raw_macro_f1", abs(raw["macro_f1"] - row.raw_macro_f1) <= 0.000051)
            check("smoothing_k_frozen_before_test", int(row.k) == choices[name]["k"])
            cv = root / "cv/SVM_RBF_Nystroem_all_data.json"
            check("cv_choice_file_unchanged_after_test", hashlib.sha256(cv.read_bytes()).hexdigest() == choices[name]["cv_result_sha256"])
            users = pd.read_csv(root / "test_per_user.csv", dtype={"user_id": str})
            check("per_user_point_counts", users.set_index("user_id").points.sort_index().equals(
                  test.groupby("user_id").n_points.sum().sort_index().rename("points")))
            check("user_average_macro_f1", abs(users.macro_f1.mean() - row.user_avg_macro_f1) <= 0.00011)
            check("weighted_user_accuracy", abs(np.average(users.accuracy, weights=users.points) - row.point_accuracy) <= 0.00011)
    report = {"checks": checks, "passed": sum(c["passed"] for c in checks), "total": len(checks)}
    path = PROJECT / "outputs/sampling_results_reference/final_validation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Final result checks: {report['passed']}/{report['total']} passed")
    if report["passed"] != report["total"]:
        raise SystemExit("Result consistency checks failed; see final_validation.json")


if __name__ == "__main__":
    main()
