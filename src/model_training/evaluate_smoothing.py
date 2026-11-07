"""
Part 2, step 3b - apply the smoothing k chosen on training users (smoothing_choice.json) once to
the test users, and build the segment output.

    python evaluate_smoothing.py

Test predictions come from the headline run (window_predictions_test.csv), so every model is exactly
as trained by train_basic.py. k is read from smoothing_choice.json and never changed here.

Outputs in results/basic_w50_no_dt/smoothing/ :
    smoothing_test.csv                one row per model: k, baseline vs smoothed point macro-F1 and
                                      accuracy with 95% CIs, mode changes, isolated flips
    smoothing_bootstrap_pairwise.csv  paired gain (smoothed - baseline) with 95% CI per model
    smoothing_per_class_f1.csv        per-class point F1, baseline and smoothed
    segment_metrics_test.csv          segment counts, segment recall/precision (IoU >= 0.5),
                                      transition precision/recall (+/- 1 window), baseline and smoothed
    segments_test.csv                 final output: segments of the selected model after smoothing
    segments_test_baseline.csv        the same without smoothing
    window_predictions_test_smoothed.csv
    confusion_matrix_final_points.csv point confusion matrix of the selected model after smoothing
    per_user_test_smoothed.csv        per-user point accuracy, baseline and smoothed (selected model)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from models import MODES, cv_groups, load_windows, per_user_accuracy, point_confusion, prepare, user_bootstrap
from segments import make_segments, segment_metrics
from smoothing import count_changes, isolated_flips, smooth_majority
from tune_smoothing import score

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", default=str(HERE / "windows_w50.csv"))
    ap.add_argument("--run", default=str(HERE / "results" / "basic_w50_no_dt"))
    ap.add_argument("--splits-report", default=str(PROJECT / "data" / "splits" / "report.json"))
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run = Path(a.run)
    out = run / "smoothing"
    choice = json.loads((out / "smoothing_choice.json").read_text())
    ks = {m: c["k"] for m, c in choice["models"].items()}
    selected = choice["selected_model"]

    df = load_windows(a.windows)
    te = df[df.split == "test"].reset_index(drop=True)
    wp = pd.read_csv(run / "window_predictions_test.csv", dtype={"user_id": str})
    assert (wp.start_time.to_numpy() == te.start_time.to_numpy()).all() and \
        (wp.n_points.to_numpy() == te.n_points.to_numpy()).all(), "prediction rows out of order"
    _, _, ute, mte = prepare(te)
    gte = cv_groups(ute, a.splits_report)
    blocks, times = te.feature_block_id, te.start_time
    true_changes = count_changes(te.label, blocks, times)

    smoothed = te[["user_id", "trip_id", "feature_block_id", "start_time", "end_time", "n_points",
                   "label", "purity"]].copy()
    test_rows, class_rows, pair_rows, seg_rows = [], [], [], []
    for name, k in ks.items():
        base = wp[name].to_numpy(dtype=object)
        smo = smooth_majority(base, blocks, times, k)
        smoothed[name] = smo
        ci, pairs = user_bootstrap([SimpleNamespace(name="baseline", y_pred=base),
                                    SimpleNamespace(name="smoothed", y_pred=smo)],
                                   mte, gte, MODES, n_boot=a.n_boot, seed=a.seed)
        ci = ci.set_index("model")
        row = {"model": name, "selected_by_cv": name == selected, "k": k}
        for v in ("baseline", "smoothed"):
            for col in ("point_macro_f1", "macro_f1_ci_low", "macro_f1_ci_high",
                        "point_accuracy", "accuracy_ci_low", "accuracy_ci_high"):
                row[f"{col}_{v}"] = ci.loc[v, col]
        for v, pred in (("baseline", base), ("smoothed", smo)):
            flips, wrong = isolated_flips(pred, te.label, blocks, times)
            row.update({f"pred_changes_{v}": count_changes(pred, blocks, times),
                        f"isolated_flips_{v}": flips, f"isolated_flips_wrong_{v}": wrong})
            class_rows.append({"model": name, "version": v, "k": k if v == "smoothed" else 1,
                               **{c: score(mte, pred)[f"f1_{c}"] for c in MODES}})
            seg_rows.append({"model": name, "version": v, "k": k if v == "smoothed" else 1,
                             **segment_metrics(te, pred, te.label)})
        row["true_changes"] = true_changes
        test_rows.append(row)
        p = pairs.iloc[0]          # baseline - smoothed -> flip the sign for the gain
        pair_rows.append({"model": name, "k": k,
                          "macro_f1_gain": -p.macro_f1_diff,
                          "macro_f1_gain_ci_low": -p.macro_f1_diff_ci_high,
                          "macro_f1_gain_ci_high": -p.macro_f1_diff_ci_low,
                          "macro_f1_gain_ci_includes_0": bool(p.macro_f1_ci_includes_0),
                          "accuracy_gain": -p.accuracy_diff,
                          "accuracy_gain_ci_low": -p.accuracy_diff_ci_high,
                          "accuracy_gain_ci_high": -p.accuracy_diff_ci_low,
                          "accuracy_gain_ci_includes_0": bool(p.accuracy_ci_includes_0)})

    pd.DataFrame(test_rows).round(4).to_csv(out / "smoothing_test.csv", index=False)
    pd.DataFrame(pair_rows).round(4).to_csv(out / "smoothing_bootstrap_pairwise.csv", index=False)
    pd.DataFrame(class_rows).round(4).to_csv(out / "smoothing_per_class_f1.csv", index=False)
    pd.DataFrame(seg_rows).round(4).to_csv(out / "segment_metrics_test.csv", index=False)
    smoothed.to_csv(out / "window_predictions_test_smoothed.csv", index=False)

    final, base_sel = smoothed[selected].to_numpy(dtype=object), wp[selected].to_numpy(dtype=object)
    make_segments(te, final).to_csv(out / "segments_test.csv", index=False)
    make_segments(te, base_sel).to_csv(out / "segments_test_baseline.csv", index=False)
    pd.DataFrame(point_confusion(mte, final, MODES), index=MODES, columns=MODES) \
        .rename_axis("true \\ predicted").to_csv(out / "confusion_matrix_final_points.csv")
    pu = per_user_accuracy([SimpleNamespace(name="baseline", y_pred=base_sel),
                            SimpleNamespace(name="smoothed", y_pred=final)], mte, ute, gte, MODES)
    pu.round(4).to_csv(out / "per_user_test_smoothed.csv", index=False)

    pd.set_option("display.width", 240)
    t = pd.DataFrame(test_rows).set_index("model")
    print(t[["k", "point_macro_f1_baseline", "point_macro_f1_smoothed", "point_accuracy_baseline",
             "point_accuracy_smoothed", "pred_changes_baseline", "pred_changes_smoothed", "true_changes"]]
          .round(4).to_string())
    print("\n" + pd.DataFrame(seg_rows).round(4).to_string(index=False))
    print(f"\n[done] {selected} with k = {ks[selected]}; outputs in {out}")


if __name__ == "__main__":
    main()
