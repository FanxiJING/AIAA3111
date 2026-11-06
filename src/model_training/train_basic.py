"""
Part 2, step 2 - train and compare the basic models on the window table.

    python train_basic.py                                   # windows_w50.csv, full grid search
    python train_basic.py --windows windows_w50.csv --no-tune   # fast run
    python train_basic.py --models "Random Forest" "Gradient Boosting (HGB)"

Protocol
    train / test users = Part 1's fixed lists (the 'split' column written by build_windows.py)
    tuning             = grid search, 3-fold CV grouped by user (linked users kept together),
                         on training users only, scoring macro-F1; test users never used for tuning
    main metric        = POINT-level macro-F1 on the test users (each point gets its window's
                         prediction); window-level scores are reported too
    model selection    = highest CV macro-F1 on training users; test scores are reported for
                         every model but never used to choose one

    features           = dt_mean and dt_max are excluded by default; --keep-dt includes them
    uncertainty        = bootstrap 95% CIs resampling test users, linked users drawn together

Outputs in results/basic_w<W>_no_dt/ (results/basic_w<W>/ with --keep-dt) :
    results.csv  per_class_f1_points.csv  window_predictions_test.csv  feature_importance.csv
    bootstrap_ci.csv  bootstrap_pairwise.csv  per_user_test.csv
    model_comparison.png  confusion_matrix_points.png  best_model.joblib  run_config.json

--no-tune has no CV scores, so it selects no model: it writes no best_model.joblib or confusion
matrix, and "_notune" is added to its default output folder.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from models import (EXCLUDED_FEATURES, MODES, cv_groups, get_models, load_windows, per_user_accuracy,
                    prepare, results_table, select_best, train_and_evaluate, user_bootstrap)

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DEFAULT_SPLITS_REPORT = next((p for p in (PROJECT / "data" / "splits" / "report.json",
                                          PROJECT / "splits" / "report.json") if p.exists()),
                             PROJECT / "splits" / "report.json")
INK, MUTED, GRID, BAR = "#1f2328", "#6b7280", "#e5e7eb", "#2f6fb3"


def plot_comparison(t: pd.DataFrame, path: Path):
    t = t.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(t) + 1.3))
    y = np.arange(len(t))
    ax.barh(y, t["point_macro_f1"], height=0.55, color=BAR, label="Macro-F1 (points)")
    ax.scatter(t["point_accuracy"], y, s=36, color=INK, zorder=3, label="Accuracy (points)")
    for yi, v, a in zip(y, t["point_macro_f1"], t["point_accuracy"]):
        ax.text(max(v, a) + 0.025, yi, f"F1 {v:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, t["model"], fontsize=9, color=INK)
    ax.set_xlim(0, 1.15)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Score on test users (point level)", color=MUTED)
    ax.set_title("Basic models: transport-mode classification", loc="left", color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confusion(cm, labels, title, path):
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(len(labels) + 2.2, len(labels) + 1.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]:,})", ha="center", va="center",
                    fontsize=7.5, color="white" if cmn[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, loc="left", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall (row-normalised)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", default=str(HERE / "windows_w50.csv"))
    ap.add_argument("--splits-report", default=str(DEFAULT_SPLITS_REPORT),
                    help="Part 1's splits/report.json (for duplicate-linked user groups)")
    ap.add_argument("--models", nargs="*", default=None, help="subset of model names")
    ap.add_argument("--drop-features", nargs="*", default=[],
                    help=f"features to drop in addition to the default exclusions {EXCLUDED_FEATURES}")
    ap.add_argument("--keep-dt", action="store_true",
                    help="include the sampling-interval features (to reproduce the ablation)")
    ap.add_argument("--n-boot", type=int, default=5000, help="bootstrap resamples of test users")
    ap.add_argument("--min-train-points", type=int, default=10,
                    help="drop training windows shorter than this (test windows are all kept)")
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--cv-splits", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    df = load_windows(a.windows)
    m = re.search(r"w(\d+)", Path(a.windows).stem)
    suffix = ("" if a.keep_dt else "_no_dt") + ("_notune" if a.no_tune else "")
    out = Path(a.out or HERE / "results" / f"basic_w{m.group(1) if m else 'x'}{suffix}")
    out.mkdir(parents=True, exist_ok=True)

    tr = df[(df.split == "train") & (df.n_points >= a.min_train_points)].reset_index(drop=True)
    te = df[df.split == "test"].reset_index(drop=True)
    print(f"[data] dropped {int(((df.split == 'train') & (df.n_points < a.min_train_points)).sum()):,} "
          f"training windows with < {a.min_train_points} points")
    exclude = () if a.keep_dt else EXCLUDED_FEATURES
    Xtr, ytr, utr, _ = prepare(tr, a.drop_features, exclude)
    Xte, yte, ute, mte = prepare(te, a.drop_features, exclude)
    gtr = cv_groups(utr, a.splits_report)
    gte = cv_groups(ute, a.splits_report)
    print(f"[data] train: {len(tr):,} windows / {tr.n_points.sum():,} points / {utr.nunique()} users "
          f"({gtr.nunique()} CV groups)")
    print(f"[data] test : {len(te):,} windows / {te.n_points.sum():,} points / {ute.nunique()} users")
    print(f"[data] {Xtr.shape[1]} features; train window labels {ytr.value_counts().to_dict()}")
    assert not set(utr) & set(ute), "train and test users overlap!"

    models = get_models(a.seed)
    if a.models:
        models = {k: v for k, v in models.items() if k in a.models}
    print("[train]")
    res = train_and_evaluate(Xtr, ytr, gtr, Xte, yte, mte, models=models,
                             tune=not a.no_tune, cv_splits=a.cv_splits, seed=a.seed)
    labels = [l for l in MODES if l in set(ytr) | set(yte)]

    table = results_table(res)
    table.to_csv(out / "results.csv", index=False)
    pd.DataFrame({r.name: dict(zip(labels, np.round(r.point["per_class_f1"], 4))) for r in res}).T \
        .to_csv(out / "per_class_f1_points.csv")
    pred = te[["user_id", "trip_id", "feature_block_id", "start_time", "end_time", "n_points",
               "label", "purity"]].copy()
    for r in res:
        pred[r.name] = r.y_pred
    pred.to_csv(out / "window_predictions_test.csv", index=False)

    per_user = per_user_accuracy(res, mte, ute, gte, labels)
    per_user.round(4).to_csv(out / "per_user_test.csv", index=False)
    ci, pairs = user_bootstrap(res, mte, gte, labels, n_boot=a.n_boot, seed=a.seed)
    ci["user_avg_accuracy"] = ci["model"].map(per_user[[r.name for r in res]].mean())
    ci.round(4).to_csv(out / "bootstrap_ci.csv", index=False)
    pairs.round(4).to_csv(out / "bootstrap_pairwise.csv", index=False)
    print(f"\n[bootstrap] {gte.nunique()} resampling groups from {ute.nunique()} test users, "
          f"{a.n_boot} resamples")
    print(ci.round(3).to_string(index=False))

    best = select_best(res)                    # by CV macro-F1 on training users, never test scores
    plot_comparison(table, out / "model_comparison.png")
    if best is not None:
        joblib.dump({"model": best.estimator, "features": list(Xtr.columns), "labels": labels,
                     "windows_file": Path(a.windows).name}, out / "best_model.joblib")
        plot_confusion(best.point_cm, labels, f"{best.name} – test users, point level",
                       out / "confusion_matrix_points.png")

    rf = next((r for r in res if r.name == "Random Forest"), None)
    if rf is not None:
        perm = permutation_importance(rf.estimator, Xte, yte, scoring="f1_macro", n_repeats=5,
                                      random_state=a.seed, n_jobs=-1)
        pd.DataFrame({"feature": Xtr.columns,
                      "rf_impurity": rf.estimator.named_steps["clf"].feature_importances_,
                      "perm_macro_f1_drop": perm.importances_mean, "perm_std": perm.importances_std}) \
            .sort_values("perm_macro_f1_drop", ascending=False).to_csv(out / "feature_importance.csv", index=False)

    (out / "run_config.json").write_text(json.dumps(
        {**vars(a), "n_train_windows": len(tr), "n_test_windows": len(te),
         "features": list(Xtr.columns), "best_model": best.name if best else None,
         "selection_rule": "highest CV macro-F1 on training users"}, indent=2, default=str))
    print("\n" + table[["model", "selected_by_cv", "point_accuracy", "point_macro_f1",
                        "window_macro_f1", "cv_macro_f1", "cv_macro_f1_std"]].to_string(index=False))
    chosen = f"selected by CV = {best.name}" if best else "no CV scores, no model selected"
    print(f"\n[done] {chosen}; outputs in {out}")


if __name__ == "__main__":
    main()
