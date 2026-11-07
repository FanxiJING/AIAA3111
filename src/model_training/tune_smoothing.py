"""
Part 2, step 3a - choose the smoothing window k on TRAINING users only.

    python build_windows.py --no-pure-train --out windows_w50_blind.csv    # once
    python tune_smoothing.py                # refits the models fold by fold (~10 min)
    python tune_smoothing.py --reuse-oof    # rescore saved out-of-fold predictions (seconds)
    python evaluate_smoothing.py            # step 3b: apply the chosen k to test users

Tuning
    Training windows used for fitting are cut at label changes, so they do not look like real input.
    For tuning, the training users' windows are cut BLIND (windows_w50_blind.csv), exactly like test.
    Folds = the same user-grouped StratifiedGroupKFold as train_basic.py (linked users together).
    Each fold: fit the model with its tuned parameters on the pure windows of the other folds' users
    and predict the blind windows of this fold's users -> out-of-fold (OOF) predictions.
    Each k is scored on the OOF predictions: point macro-F1 per fold, pooled point metrics, mode
    changes and segment metrics (to show over-smoothing).
Choice (paired one-standard-error rule)
    best = the k with the highest mean fold macro-F1 (ties: smaller k).
    For every k, d_f = F1_f(k) - F1_f(best) in each fold f; SE = std(d_f) / sqrt(n_folds).
    chosen = the smallest k with mean(d_f) >= -SE.
    The SE is of the paired difference, because all k are scored on the same folds. The unpaired SE
    of the fold scores mostly measures how much harder one fold's users are than another's (fold
    macro-F1 spread about 0.14 at k = 1), which is the same for every k; with it, even k = 1 falls
    within one SE of the best k although smoothing helps in every fold.
    The choice is written to smoothing_choice.json, which evaluate_smoothing.py reads; test users
    are not loaded here.

Outputs in results/basic_w50_no_dt/smoothing/ :
    smoothing_cv.csv  smoothing_choice.json  oof_predictions_train.csv
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from models import (MODES, _subsample, cv_groups, get_models, load_windows, metrics_from_cm,
                    point_confusion, prepare)
from segments import segment_metrics
from smoothing import count_changes, smooth_majority

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
KS = [1, 3, 5, 7, 9, 11, 13, 15, 21]


def score(meta: pd.DataFrame, pred) -> dict:
    m = metrics_from_cm(point_confusion(meta, pred, MODES))
    return {"point_macro_f1": m["macro_f1"], "point_accuracy": m["accuracy"],
            **{f"f1_{c}": f for c, f in zip(MODES, m["per_class_f1"])}}


def user_folds(tr: pd.DataFrame, splits_report: Path, n_splits: int, seed: int) -> dict:
    """user -> fold, from the same StratifiedGroupKFold call train_basic.py uses."""
    X, y, u, _ = prepare(tr)
    g = cv_groups(u, splits_report)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_of = {}
    for f, (_, val) in enumerate(cv.split(X, y, groups=g)):
        for user in u.iloc[val].unique():
            fold_of[user] = f
    groups = pd.DataFrame({"user": u, "group": g}).drop_duplicates()
    assert groups.groupby("group").user.apply(lambda s: len({fold_of[x] for x in s})).max() == 1, \
        "linked users split across folds"
    return fold_of


def paired_one_se(rows: pd.DataFrame, n_folds: int) -> tuple[dict, pd.DataFrame]:
    """rows: one model's rows, sorted by k, with fold<f>_macro_f1 columns.
    -> (choice, rows with the paired difference to the best k and its SE)"""
    rows = rows.sort_values("k").reset_index(drop=True)
    folds = rows[[f"fold{f}_macro_f1" for f in range(n_folds)]].to_numpy()
    best = int(folds.mean(axis=1).argmax())          # first maximum = smallest k on ties
    diff = folds - folds[best]
    rows["diff_to_best_mean"] = diff.mean(axis=1)
    rows["diff_to_best_se"] = diff.std(axis=1, ddof=1) / np.sqrt(n_folds)
    chosen = int(rows.loc[rows.diff_to_best_mean >= -rows.diff_to_best_se, "k"].min())
    return {"k": chosen, "best_k": int(rows.k[best]),
            "best_fold_mean": float(folds[best].mean())}, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--windows", default=str(HERE / "windows_w50.csv"))
    ap.add_argument("--blind-windows", default=str(HERE / "windows_w50_blind.csv"))
    ap.add_argument("--run", default=str(HERE / "results" / "basic_w50_no_dt"))
    ap.add_argument("--splits-report", default=str(PROJECT / "data" / "splits" / "report.json"))
    ap.add_argument("--min-train-points", type=int, default=10)
    ap.add_argument("--cv-splits", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse-oof", action="store_true", help="load saved OOF predictions instead of refitting")
    a = ap.parse_args()
    run = Path(a.run)
    out = run / "smoothing"
    out.mkdir(parents=True, exist_ok=True)

    pure, blind = load_windows(a.windows), load_windows(a.blind_windows)
    tr = pure[(pure.split == "train") & (pure.n_points >= a.min_train_points)].reset_index(drop=True)
    bl = blind[blind.split == "train"].reset_index(drop=True)
    fold_of = user_folds(tr, Path(a.splits_report), a.cv_splits, a.seed)
    bl = bl[bl.user_id.isin(fold_of)].reset_index(drop=True)   # users with no training window have no fold
    print(f"[tune] {len(fold_of)} training users in {a.cv_splits} folds; {len(bl):,} blind windows, "
          f"{bl.n_points.sum():,} points")

    results = pd.read_csv(run / "results.csv")
    params = dict(zip(results.model, results.best_params.map(ast.literal_eval)))
    selected = results.loc[results.selected_by_cv, "model"].iloc[0]
    models = get_models(a.seed)
    oof_path = out / "oof_predictions_train.csv"

    if a.reuse_oof:
        oof = pd.read_csv(oof_path, dtype={"user_id": str})
        assert (oof.start_time.to_numpy() == bl.start_time.to_numpy()).all() and \
            (oof.n_points.to_numpy() == bl.n_points.to_numpy()).all(), "saved OOF rows do not match"
        print(f"[tune] reusing {oof_path.name}")
    else:
        Xtr, ytr, utr, _ = prepare(tr)
        Xbl, _, ubl, _ = prepare(bl)
        oof = bl[["user_id", "trip_id", "feature_block_id", "start_time", "n_points", "label"]].copy()
        oof["fold"] = ubl.map(fold_of).to_numpy()
        for name, (pipe, _, max_n) in models.items():
            pred = np.empty(len(bl), dtype=object)
            for f in range(a.cv_splits):
                fit_rows = (utr.map(fold_of) != f).to_numpy()
                Xf, yf, _ = _subsample(Xtr[fit_rows], ytr[fit_rows], utr[fit_rows], max_n, a.seed)
                est = clone(pipe).set_params(**params[name]).fit(Xf, yf)
                rows = (oof.fold == f).to_numpy()
                pred[rows] = est.predict(Xbl[rows])
            oof[name] = pred
            print(f"  OOF done: {name}")
        oof.to_csv(oof_path, index=False)

    _, _, _, mbl = prepare(bl)
    fold = oof.fold.to_numpy()
    blocks, times = bl.feature_block_id, bl.start_time
    true_changes = count_changes(bl.label, blocks, times)
    rows = []
    for name in models:
        for k in KS:
            s = smooth_majority(oof[name], blocks, times, k)
            per_fold = [score(mbl[fold == f], s[fold == f])["point_macro_f1"] for f in range(a.cv_splits)]
            rows.append({"model": name, "k": k, **score(mbl, s),
                         **{f"fold{f}_macro_f1": v for f, v in enumerate(per_fold)},
                         "fold_macro_f1_mean": np.mean(per_fold),
                         "fold_macro_f1_se_unpaired": np.std(per_fold, ddof=1) / np.sqrt(a.cv_splits),
                         "pred_changes": count_changes(s, blocks, times), "true_changes": true_changes,
                         **segment_metrics(bl, s, bl.label)})
    cvt = pd.DataFrame(rows)
    choice, parts = {}, []
    for name in models:
        choice[name], part = paired_one_se(cvt[cvt.model == name], a.cv_splits)
        parts.append(part)
    cvt = pd.concat(parts, ignore_index=True)
    cvt["chosen"] = cvt.apply(lambda r: r.k == choice[r.model]["k"], axis=1)
    cvt["best_mean"] = cvt.apply(lambda r: r.k == choice[r.model]["best_k"], axis=1)
    cvt.round(4).to_csv(out / "smoothing_cv.csv", index=False)
    (out / "smoothing_choice.json").write_text(json.dumps(
        {"rule": "paired one-standard-error: best = highest mean fold macro-F1; chosen = smallest k "
                 "with mean(F1_f(k) - F1_f(best)) >= -SE of that paired difference across folds",
         "data": "out-of-fold predictions on blind-cut windows of training users only",
         "ks": KS, "selected_model": selected, "n_fold_users": len(fold_of),
         "n_blind_train_windows": len(bl), "models": choice}, indent=2))

    pd.set_option("display.width", 240)
    show = ["k", "fold_macro_f1_mean", "diff_to_best_mean", "diff_to_best_se", "pred_changes",
            "true_changes", "pred_segments", "true_segments", "segment_recall_iou50",
            "segment_precision_iou50", "transition_precision", "transition_recall"]
    print(f"\n[cv] {selected}\n" + cvt[cvt.model == selected][show].round(4).to_string(index=False))
    print("\n[cv] mean fold macro-F1 by k\n" +
          cvt.pivot(index="model", columns="k", values="fold_macro_f1_mean").round(4).to_string())
    print("\n[choice] " + json.dumps({m: c["k"] for m, c in choice.items()}))
    print(f"[done] selected model {selected}: k = {choice[selected]['k']} "
          f"(best k {choice[selected]['best_k']}); outputs in {out}")


if __name__ == "__main__":
    main()
