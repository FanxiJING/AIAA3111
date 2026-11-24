"""Paired summary and independent checks for the full-coverage window follow-up."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.metadata as md
import json
import platform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import MODES, cv_groups, user_bootstrap, metrics_from_cm
from pipeline_v2 import per_user_scores
from window_controlled import ROOT, SOURCES, PROJECT, cm, encode, read_members


def main():
    checks, runs, cv_rows, user_rows, class_rows = [], [], [], [], []
    reference = None
    freeze = json.loads((ROOT / "cv_frozen_before_test.json").read_text())
    def check(name, passed):
        checks.append({"check": name, "passed": bool(passed)})
    protocol = json.loads((ROOT / "protocol.json").read_text())
    check("Python version matches frozen protocol", platform.python_version() == protocol["python"])
    check("Package versions match frozen protocol",
          all(md.version(k) == v for k, v in protocol["versions"].items()))
    check("All frozen source/input hashes unchanged",
          all(hashlib.sha256(Path(k).read_bytes()).hexdigest() == v
              for k, v in protocol["file_sha256"].items()))
    folds = json.loads((ROOT / "folds_51.json").read_text())
    fold_users = pd.Series(list(folds))
    fold_groups = cv_groups(fold_users, PROJECT / "data/splits/report.json")
    assignments = pd.DataFrame({"group": fold_groups, "fold": fold_users.map(folds)})
    check("Linked training users never cross CV folds",
          assignments.groupby("group").fold.nunique().max() == 1)
    for scheme in SOURCES:
        frame = pd.read_parquet(ROOT / scheme / "test_predictions.parquet")
        frame["user_id"] = frame.user_id.astype(str).str.zfill(3)
        keys = ["user_id", "feature_block_id", "timestamp", "mode"]
        if reference is None:
            reference = frame[keys].copy()
        check(f"{scheme}: identical test observations", reference.equals(frame[keys]))
        check(f"{scheme}: full test count", len(frame) == 860972)
        check(f"{scheme}: unique test points", not frame.duplicated(["user_id", "timestamp"]).any())
        report = json.loads((ROOT / scheme / "test_metrics.json").read_text())
        cv = ROOT / scheme / "cv_metrics.json"
        check(f"{scheme}: CV frozen before test", hashlib.sha256(cv.read_bytes()).hexdigest() == freeze[scheme])
        check(f"{scheme}: full training coverage", report["training_points"] == 3486792 and report["training_users"] == 51)
        train_members = read_members(scheme, "train")
        check(f"{scheme}: training members complete", len(train_members) == 3486792 and train_members.user_id.nunique() == 51)
        if scheme == "points50":
            original_train = train_members[keys].copy()
        else:
            check("identical training observations across schemes", original_train.equals(train_members[keys]))
        truth = encode(frame["mode"])
        saved_scores = np.load(ROOT / scheme / "window_test_scores.npz", allow_pickle=False)
        indices = frame.window_index.to_numpy(dtype=int)
        check(f"{scheme}: raw predictions reconstructed from window scores",
              np.array_equal(encode(saved_scores["raw"])[indices], frame.raw_code.to_numpy()))
        # Independently reconstruct physical smoothing with pandas' time rolling
        # rather than the experiment's prefix sums/searchsorted implementation.
        reconstructed = np.empty(len(frame), dtype=int)
        for positions in frame.groupby(["user_id", "feature_block_id"], sort=False).indices.values():
            values = pd.DataFrame(saved_scores["scores"][indices[positions]],
                                  index=pd.DatetimeIndex(frame.timestamp.iloc[positions]))
            reconstructed[positions] = values.rolling("1200s", center=True, closed="both").mean().to_numpy().argmax(axis=1)
        check(f"{scheme}: physical smoothing independently reconstructed",
              np.array_equal(reconstructed, frame.time20min_code.to_numpy()))
        for version in ["raw", "time20min"]:
            values = frame[f"{version}_code"].to_numpy(dtype=int)
            matrix = cm(truth, values)
            check(f"{scheme}/{version}: saved confusion matrix", np.array_equal(matrix, report[version]["confusion_matrix"]))
            metrics = metrics_from_cm(matrix)
            for mode, score, support in zip(MODES, metrics["per_class_f1"], matrix.sum(axis=1)):
                class_rows.append({"scheme": scheme, "version": version, "mode": mode,
                                   "f1": score, "test_points": int(support)})
            check(f"{scheme}/{version}: saved macro-F1", abs(metrics["macro_f1"] - report[version]["macro_f1"]) < 1e-12)
            runs.append(SimpleNamespace(name=f"{scheme}: {version}", y_pred=np.asarray(MODES)[values]))
        cv_report = json.loads(cv.read_text())
        for version in ["raw", "time20min"]:
            cv_rows.append({"scheme": scheme, "version": version,
                            "cv_fold_mean_macro_f1": cv_report[version]["fold_mean_macro_f1"],
                            "cv_pooled_macro_f1": cv_report[version]["macro_f1"],
                            **{f"cv_fold_{i}": value for i, value in enumerate(cv_report[version]["fold_macro_f1"])}})
    meta = pd.DataFrame({f"cnt_{m}": reference["mode"].eq(m).astype(int).to_numpy() for m in MODES})
    meta["n_points"] = 1
    users = reference.user_id.reset_index(drop=True)
    groups = cv_groups(users, PROJECT / "data/splits/report.json")
    check("13 test users and 9 independent groups", users.nunique() == 13 and groups.nunique() == 9)
    ci, pairs = user_bootstrap(runs, meta, groups, MODES, n_boot=5000, seed=42)
    for run in runs:
        per_user = per_user_scores(meta, users, groups, run.y_pred)
        user_rows.append(per_user.assign(model=run.name))
        ci.loc[ci.model == run.name, "user_avg_macro_f1"] = per_user.macro_f1.mean()
        ci.loc[ci.model == run.name, "user_avg_accuracy"] = per_user.accuracy.mean()
    ci.to_csv(ROOT / "results.csv", index=False)
    pairs.to_csv(ROOT / "paired_differences.csv", index=False)
    pd.concat(user_rows, ignore_index=True).to_csv(ROOT / "per_user_results.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(ROOT / "cv_results.csv", index=False)
    pd.DataFrame(class_rows).to_csv(ROOT / "per_class_results.csv", index=False)
    validation = {"passed": sum(c["passed"] for c in checks), "total": len(checks), "checks": checks}
    (ROOT / "validation.json").write_text(json.dumps(validation, indent=2))
    if not all(c["passed"] for c in checks):
        raise ValueError("Result reconstruction checks failed")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for i, (scheme, label, color) in enumerate([("points50", "50 points", "#2468A0"),
                                                ("time120", "120 seconds", "#C87E35")]):
        data = ci.set_index("model").loc[[f"{scheme}: raw", f"{scheme}: time20min"]]
        x = np.array([0, 1]) + (i - 0.5) * .12
        axes[0].errorbar(x, data.point_macro_f1,
                        yerr=[data.point_macro_f1 - data.macro_f1_ci_low,
                              data.macro_f1_ci_high - data.point_macro_f1], fmt="o-", capsize=4, label=label, color=color)
    axes[0].set_xticks([0, 1], ["No smoothing", "Centered 20 min"])
    axes[0].set(ylabel="Point macro-F1", ylim=(0, 1), title="Identical training/test observations")
    axes[0].legend(frameon=False)
    differences = pairs.set_index(["model_a", "model_b"]).loc[
        [("points50: raw", "time120: raw"), ("points50: time20min", "time120: time20min")]]
    axes[1].axhline(0, color="#929BA8", linestyle="--", linewidth=1)
    axes[1].errorbar([0, 1], differences.macro_f1_diff,
                    yerr=[differences.macro_f1_diff - differences.macro_f1_diff_ci_low,
                          differences.macro_f1_diff_ci_high - differences.macro_f1_diff], fmt="o", capsize=5, color="#2468A0")
    axes[1].set_xticks([0, 1], ["No smoothing", "Centered 20 min"])
    axes[1].set(ylabel="Macro-F1 difference: 50 points - 120 seconds", title="Paired 95% group-bootstrap intervals")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Original GPS | 51 training users, 13 test users | Same physical smoothing span", fontsize=13)
    fig.savefig(ROOT / "window_controlled_comparison.png", dpi=300)
    fig.savefig(ROOT / "window_controlled_comparison.svg")
    plt.close(fig)
    print(ci.to_string(index=False))
    print(differences[["macro_f1_diff", "macro_f1_diff_ci_low", "macro_f1_diff_ci_high"]].to_string())
    print(f"Validation: {validation['passed']}/{validation['total']} passed")


if __name__ == "__main__":
    main()
