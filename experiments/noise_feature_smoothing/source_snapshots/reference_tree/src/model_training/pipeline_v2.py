"""
Part 2 v2 - hardened training pipeline. Every choice is made by grouped CV on the training users;
the test users are used once, in the `final` stage.

    python build_windows.py                                              # windows_w50.csv (once)
    python build_windows.py --no-pure-train --out windows_w50_blind.csv  # blind windows (once)
    python pipeline_v2.py --stage cv        # tune, choose variants and k, select the model (~1 h)
    python pipeline_v2.py --stage final     # fit on all training users, evaluate test once
    python pipeline_v2.py --stage allusers  # report-only grouped CV over all 64 users
    python make_figures_v2.py

What changed against v1 (train_basic.py + tune_smoothing.py)
* Validation is blind: every CV fold fits on the pure (label-cut) windows of the training-fold users
  and is scored on the BLIND windows of the held-out users, at point level. v1's grid search scored
  label-cut windows, which do not look like real input.
* 5 user-grouped folds (StratifiedGroupKFold, duplicate-linked users in one fold) instead of 3, so
  the one-SE rules have a usable standard error.
* Fair comparison: every model is fitted on the same stratified 20,000-window subsample of its
  fitting set (v1: only the SVM). One extra model, an RBF-SVM approximated with Nystroem features
  and a linear SVM, is fitted on all windows.
* Per model, on top of its tuned hyperparameters: its own smoothing k, soft voting (average class
  scores over k windows) and class weights on/off. A variant is adopted only if its mean paired
  per-fold gain in macro-F1 is larger than one SE of that gain.
* The model is selected on the full pipeline (hyperparameters + variant + k). The SVM is kept if it
  is within one paired SE of the best model.
* Test: all models with their own k; CIs from a bootstrap over the 9 independent test groups.

Outputs: results/basic_w50_no_dt_v2/ (see README).
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from models import (MODES, _macro_f1_accuracy, _subsample, cv_groups, load_windows, metrics_from_cm,
                    point_confusion, prepare, user_bootstrap)
from segments import make_segments, segment_metrics
from smoothing import count_changes, isolated_flips, smooth_majority, smooth_soft

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SPLITS_REPORT = PROJECT / "data" / "splits" / "report.json"
OUT = HERE / "results" / "basic_w50_no_dt_v2"
SEED, N_SUB, N_FOLDS, N_BOOT = 42, 20_000, 5, 5000
KS = [1, 3, 5, 7, 9, 11, 13, 15, 21]
SVM_NAMES = ("SVM (RBF)", "SVM (RBF, Nystroem, all data)")
NYSTROEM_COMPONENTS = 1000


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
def _scaled(clf):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("clf", clf)])


def _plain(clf):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", clf)])


def model_specs(n_features: int) -> dict:
    """name -> SimpleNamespace(make(cw) -> pipeline, grid, subsample, has_cw).
    cw=True means class weights balanced (v1 default), False means none."""
    bal = lambda cw: "balanced" if cw else None  # noqa: E731
    return {
        "Majority (baseline)": SimpleNamespace(
            make=lambda cw: _plain(DummyClassifier(strategy="most_frequent")), grid={}, subsample=True, has_cw=False),
        "Logistic Regression": SimpleNamespace(
            make=lambda cw: _scaled(LogisticRegression(max_iter=3000, class_weight=bal(cw))),
            grid={"clf__C": [0.1, 1.0, 10.0]}, subsample=True, has_cw=True),
        "Decision Tree": SimpleNamespace(
            make=lambda cw: _plain(DecisionTreeClassifier(class_weight=bal(cw), random_state=SEED)),
            grid={"clf__max_depth": [8, 16, None], "clf__min_samples_leaf": [1, 10]}, subsample=True, has_cw=True),
        "k-NN": SimpleNamespace(
            make=lambda cw: _scaled(KNeighborsClassifier()),
            grid={"clf__n_neighbors": [5, 15, 31], "clf__weights": ["uniform", "distance"]},
            subsample=True, has_cw=False),
        "SVM (RBF)": SimpleNamespace(
            make=lambda cw: _scaled(SVC(kernel="rbf", gamma="scale", class_weight=bal(cw), random_state=SEED)),
            grid={"clf__C": [1.0, 10.0]}, subsample=True, has_cw=True),
        "Random Forest": SimpleNamespace(
            make=lambda cw: _plain(RandomForestClassifier(
                n_estimators=200, class_weight="balanced_subsample" if cw else None, n_jobs=-1, random_state=SEED)),
            grid={"clf__min_samples_leaf": [1, 5], "clf__max_features": ["sqrt", 0.5]}, subsample=True, has_cw=True),
        "Gradient Boosting (HGB)": SimpleNamespace(
            make=lambda cw: _plain(HistGradientBoostingClassifier(
                class_weight=bal(cw), early_stopping=False, max_iter=300, random_state=SEED)),
            grid={"clf__learning_rate": [0.05, 0.1], "clf__max_leaf_nodes": [31, 63],
                  "clf__l2_regularization": [0.0, 1.0]}, subsample=True, has_cw=True),
        # RBF kernel approximated by Nystroem features; gamma = 1 / n_features equals SVC's
        # gamma="scale" on standardised features (variance 1)
        "SVM (RBF, Nystroem, all data)": SimpleNamespace(
            make=lambda cw: Pipeline([
                ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                ("rbf", Nystroem(kernel="rbf", gamma=1.0 / n_features, n_components=NYSTROEM_COMPONENTS,
                                 random_state=SEED)),
                ("clf", LinearSVC(class_weight=bal(cw), dual=False, max_iter=5000, random_state=SEED))]),
            grid={"clf__C": [0.1, 1.0, 10.0]}, subsample=False, has_cw=True),
    }


def class_scores(est, X) -> np.ndarray:
    """Per-class scores with columns in MODES order: probabilities if the model has them,
    otherwise one-vs-rest decision values (SVMs)."""
    s = est.predict_proba(X) if hasattr(est, "predict_proba") else est.decision_function(X)
    s = np.asarray(s, dtype=float)
    out = np.full((len(X), len(MODES)), s.min() - 1.0)
    for j, c in enumerate(est.classes_):
        out[:, MODES.index(c)] = s[:, j]
    return out


def fit_predict(spec, cw, params, X_fit, y_fit, u_fit, X_eval):
    if spec.subsample:
        X_fit, y_fit, _ = _subsample(X_fit, y_fit, u_fit, N_SUB, SEED)
    est = spec.make(cw).set_params(**params).fit(X_fit, y_fit)
    return est.predict(X_eval).astype(object), class_scores(est, X_eval), est


# --------------------------------------------------------------------------------------
# Folds, scoring, rules
# --------------------------------------------------------------------------------------
def user_folds(pure: pd.DataFrame, n_folds: int = N_FOLDS) -> dict:
    """user -> fold from StratifiedGroupKFold on pure windows; linked users always share a fold."""
    X, y, u, _ = prepare(pure)
    g = cv_groups(u, SPLITS_REPORT)
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    fold_of = {}
    for f, (_, val) in enumerate(cv.split(X, y, groups=g)):
        for user in u.iloc[val].unique():
            fold_of[user] = f
    pairs = pd.DataFrame({"user": u, "group": g}).drop_duplicates()
    assert pairs.groupby("group").user.apply(lambda s: len({fold_of[x] for x in s})).max() == 1
    return fold_of


def fold_f1(meta: pd.DataFrame, pred, fold: np.ndarray, n_folds: int = N_FOLDS) -> np.ndarray:
    pred = np.asarray(pred, dtype=object)
    return np.array([metrics_from_cm(point_confusion(meta[fold == f], pred[fold == f], MODES))["macro_f1"]
                     for f in range(n_folds)])


def smooth(voting: str, raw_pred, scores, blocks, times, k: int):
    if voting == "soft":
        return smooth_soft(scores, blocks, times, k)
    return smooth_majority(raw_pred, blocks, times, k)


def paired_one_se_k(per_k: dict) -> tuple[int, int]:
    """per_k: k -> per-fold scores. best = highest mean; chosen = smallest k whose mean paired
    difference to the best is >= -SE of that difference (same rule as v1, now with 5 folds)."""
    ks = sorted(per_k)
    folds = np.array([per_k[k] for k in ks])
    best = int(folds.mean(axis=1).argmax())
    diff = folds - folds[best]
    se = diff.std(axis=1, ddof=1) / np.sqrt(folds.shape[1])
    ok = diff.mean(axis=1) >= -se
    return int(np.array(ks)[ok].min()), int(ks[best])


def paired_gain(new: np.ndarray, old: np.ndarray) -> dict:
    d = np.asarray(new) - np.asarray(old)
    se = d.std(ddof=1) / np.sqrt(len(d))
    return {"gain_mean": float(d.mean()), "gain_se": float(se), "adopt": bool(d.mean() > se)}


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_cv_data():
    pure = load_windows(HERE / "windows_w50.csv")
    blind = load_windows(HERE / "windows_w50_blind.csv")
    tr = pure[(pure.split == "train") & (pure.n_points >= 10)].reset_index(drop=True)
    fold_of = user_folds(tr)
    bl = blind[(blind.split == "train") & blind.user_id.isin(fold_of)].reset_index(drop=True)
    return tr, bl, fold_of


# --------------------------------------------------------------------------------------
# Stage 1: CV on training users
# --------------------------------------------------------------------------------------
def stage_cv(models: list[str] | None, resume: bool):
    cv_dir = OUT / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)
    tr, bl, fold_of = load_cv_data()
    Xtr, ytr, utr, _ = prepare(tr)
    Xbl, _, ubl, mbl = prepare(bl)
    fold_tr = utr.map(fold_of).to_numpy()
    fold_bl = ubl.map(fold_of).to_numpy()
    blocks, times = bl.feature_block_id, bl.start_time
    specs = model_specs(Xtr.shape[1])
    print(f"[cv] {len(fold_of)} users in {N_FOLDS} folds; fit on {len(tr):,} pure windows, "
          f"score {len(bl):,} blind windows ({bl.n_points.sum():,} points); {Xtr.shape[1]} features")

    def oof(spec, cw, params):
        pred = np.empty(len(bl), dtype=object)
        scores = np.zeros((len(bl), len(MODES)))
        for f in range(N_FOLDS):
            fit, ev = fold_tr != f, fold_bl == f
            p, s, _ = fit_predict(spec, cw, params, Xtr[fit], ytr[fit], utr[fit], Xbl[ev])
            pred[ev], scores[ev] = p, s
        return pred, scores

    for name, spec in specs.items():
        if models and name not in models:
            continue
        path = cv_dir / f"{name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}.json"
        if resume and path.exists():
            print(f"[cv] {name}: cached")
            continue
        t0 = time.time()
        # 1) hyperparameters: default variant (class weights as v1, hard voting, k = 1)
        tuning, best = [], None
        for params in ParameterGrid(spec.grid):
            pred, scores = oof(spec, True, params)
            f = fold_f1(mbl, pred, fold_bl)
            tuning.append({"params": params, "fold_f1": f.tolist(), "mean": float(f.mean())})
            if best is None or f.mean() > best[0]:
                best = (f.mean(), params, pred, scores)
        params = best[1]
        runs = {True: (best[2], best[3])}
        if spec.has_cw:
            runs[False] = oof(spec, False, params)

        # 2) every variant with its own k
        variants = {}
        for cw, (pred, scores) in runs.items():
            for voting in ("hard", "soft"):
                per_k = {k: fold_f1(mbl, smooth(voting, pred, scores, blocks, times, k), fold_bl) for k in KS}
                k_sel, k_best = paired_one_se_k(per_k)
                variants[(cw, voting)] = {"per_k": per_k, "k": k_sel, "k_best_mean": k_best}

        # 3) adoption: start from v1's choice (class weights on, hard voting); try soft voting,
        #    then flipping class weights; adopt only if the paired gain is above one SE
        cur = (True, "hard")
        log = []
        steps = [("soft voting", lambda c: (c[0], "soft"))]
        if spec.has_cw:
            steps.append(("class weights off", lambda c: (False, c[1])))
        for label, move in steps:
            cand = move(cur)
            old = variants[cur]["per_k"][variants[cur]["k"]]
            new = variants[cand]["per_k"][variants[cand]["k"]]
            g = paired_gain(new, old)
            log.append({"step": label, "from": _vname(cur, variants), "to": _vname(cand, variants), **g})
            if g["adopt"]:
                cur = cand
        final = variants[cur]
        k = final["k"]
        pred, scores = runs[cur[0]]
        sm = smooth(cur[1], pred, scores, blocks, times, k)

        # 4) over-smoothing check for the final variant
        seg = []
        for kk in KS:
            s_kk = smooth(cur[1], pred, scores, blocks, times, kk)
            seg.append({"k": kk, **segment_metrics(bl, s_kk, bl.label),
                        "pred_changes": count_changes(s_kk, blocks, times)})
        pooled = metrics_from_cm(point_confusion(mbl, sm, MODES))
        result = {
            "model": name, "params": params, "subsample": spec.subsample,
            "class_weights": ("balanced" if cur[0] else "none") if spec.has_cw else "n/a",
            "voting": cur[1], "k": k,
            "fold_f1": final["per_k"][k].tolist(), "mean_f1": float(final["per_k"][k].mean()),
            "se_f1": float(final["per_k"][k].std(ddof=1) / np.sqrt(N_FOLDS)),
            "pooled_macro_f1": float(pooled["macro_f1"]), "pooled_accuracy": float(pooled["accuracy"]),
            "raw_fold_f1": variants[(True, "hard")]["per_k"][1].tolist(),
            "tuning": tuning, "adoption": log,
            "variants": [{"variant": _vname(v, variants), "k": d["k"], "k_best_mean": d["k_best_mean"],
                          **{f"k{kk}": float(d["per_k"][kk].mean()) for kk in KS}} for v, d in variants.items()],
            "segments_by_k": seg, "true_changes": count_changes(bl.label, blocks, times),
            "seconds": round(time.time() - t0, 1)}
        path.write_text(json.dumps(result, indent=2, default=str))
        print(f"[cv] {name}: {params} | {result['class_weights']}, {cur[1]}, k={k} | "
              f"CV macro-F1 {result['mean_f1']:.4f} +/- {result['se_f1']:.4f} ({result['seconds']} s)")
    select()


def _vname(v, variants=None) -> str:
    cw, voting = v
    return f"class weights {'on' if cw else 'off'}, {voting} voting"


def load_cv_results() -> dict:
    out = {}
    for p in sorted((OUT / "cv").glob("*.json")):
        r = json.loads(p.read_text())
        out[r["model"]] = r
    return out


def select():
    """Model selection on the full pipeline. best = highest mean fold macro-F1 (baseline excluded);
    an SVM is kept if the paired difference best - SVM is within one SE."""
    res = load_cv_results()
    cand = {m: np.array(r["fold_f1"]) for m, r in res.items() if m != "Majority (baseline)"}
    if not cand:
        return
    best = max(cand, key=lambda m: cand[m].mean())
    svms = [m for m in SVM_NAMES if m in cand]
    rule = {"best_by_cv": best, "best_mean": float(cand[best].mean())}
    selected = best
    if svms and best not in svms:
        svm = max(svms, key=lambda m: cand[m].mean())
        d = cand[best] - cand[svm]
        se = float(d.std(ddof=1) / np.sqrt(len(d)))
        rule.update({"svm_candidate": svm, "diff_mean": float(d.mean()), "diff_se": se,
                     "svm_within_one_se": bool(d.mean() <= se)})
        if d.mean() <= se:
            selected = svm
    rule["selected"] = selected
    (OUT / "selection.json").write_text(json.dumps(rule, indent=2))
    table = pd.DataFrame([{"model": m, "class_weights": r["class_weights"], "voting": r["voting"], "k": r["k"],
                           "params": r["params"], "subsample_20k": r["subsample"],
                           "cv_macro_f1_mean": r["mean_f1"], "cv_macro_f1_se": r["se_f1"],
                           "cv_macro_f1_raw_k1": float(np.mean(r["raw_fold_f1"])),
                           "pooled_macro_f1": r["pooled_macro_f1"], "pooled_accuracy": r["pooled_accuracy"],
                           "selected": m == selected} for m, r in res.items()])
    table.sort_values("cv_macro_f1_mean", ascending=False).round(4).to_csv(OUT / "cv_summary.csv", index=False)
    print("[select] " + json.dumps(rule))


# --------------------------------------------------------------------------------------
# Stage 2: final fit and the single test evaluation
# --------------------------------------------------------------------------------------
def per_user_scores(meta, users, groups, pred) -> pd.DataFrame:
    users, groups, pred = np.asarray(users), np.asarray(groups), np.asarray(pred, dtype=object)
    rows = []
    for u in pd.unique(users):
        m = users == u
        macro, acc = _macro_f1_accuracy(point_confusion(meta[m], pred[m], MODES)[None])
        rows.append({"user_id": u, "group": groups[m][0], "points": int(meta.n_points[m].sum()),
                     "accuracy": acc[0], "macro_f1": macro[0]})
    return pd.DataFrame(rows)


def stage_final():
    res = load_cv_results()
    sel = json.loads((OUT / "selection.json").read_text())["selected"]
    pure = load_windows(HERE / "windows_w50.csv")
    tr = pure[(pure.split == "train") & (pure.n_points >= 10)].reset_index(drop=True)
    te = pure[pure.split == "test"].reset_index(drop=True)
    Xtr, ytr, utr, _ = prepare(tr)
    Xte, _, ute, mte = prepare(te)
    gte = cv_groups(ute, SPLITS_REPORT)
    specs = model_specs(Xtr.shape[1])
    blocks, times = te.feature_block_id, te.start_time
    print(f"[final] fit on {len(tr):,} training windows; test {len(te):,} windows, {ute.nunique()} users, "
          f"{gte.nunique()} resampling groups; selected model {sel}")

    preds = te[["user_id", "trip_id", "feature_block_id", "start_time", "end_time", "n_points", "label",
                "purity"]].copy()
    finals, raws, rows, classes, segs, users = [], [], [], [], [], []
    true_changes = count_changes(te.label, blocks, times)
    for name, r in res.items():
        spec = specs[name]
        cw = r["class_weights"] == "balanced"
        raw, scores, _ = fit_predict(spec, cw, r["params"], Xtr, ytr, utr, Xte)
        fin = smooth(r["voting"], raw, scores, blocks, times, r["k"])
        preds[f"{name} | raw"], preds[name] = raw, fin
        finals.append(SimpleNamespace(name=name, y_pred=fin))
        raws.append(SimpleNamespace(name=name, y_pred=raw))
        for version, p in (("raw (k=1, hard)", raw), ("final", fin)):
            m = metrics_from_cm(point_confusion(mte, p, MODES))
            classes.append({"model": name, "version": version, **dict(zip(MODES, m["per_class_f1"]))})
            segs.append({"model": name, "version": version, **segment_metrics(te, p, te.label),
                         "pred_changes": count_changes(p, blocks, times), "true_changes": true_changes})
        pu = per_user_scores(mte, ute, gte, fin)
        users.append(pu.assign(model=name))
        flips = isolated_flips(fin, te.label, blocks, times)
        rows.append({"model": name, "selected": name == sel, "class_weights": r["class_weights"],
                     "voting": r["voting"], "k": r["k"], "cv_macro_f1": r["mean_f1"], "cv_se": r["se_f1"],
                     "user_avg_accuracy": pu.accuracy.mean(), "user_avg_macro_f1": pu.macro_f1.mean(),
                     "isolated_flips": flips[0], "isolated_flips_wrong": flips[1]})
        print(f"  {name}: done")

    ci, pairs = user_bootstrap(finals, mte, gte, MODES, n_boot=N_BOOT, seed=SEED)
    ci_raw, _ = user_bootstrap(raws, mte, gte, MODES, n_boot=N_BOOT, seed=SEED)
    table = pd.DataFrame(rows).merge(ci, on="model").merge(
        ci_raw[["model", "point_macro_f1", "point_accuracy"]].rename(
            columns={"point_macro_f1": "raw_macro_f1", "point_accuracy": "raw_accuracy"}), on="model")
    smooth_gain = []
    for f, r0 in zip(finals, raws):
        # user_bootstrap keys its scores by name, so the raw run needs its own name; with the
        # shared name the final scores overwrote the raw ones and every gain came out as 0
        raw_named = SimpleNamespace(name=f"{f.name} [raw]", y_pred=r0.y_pred)
        _, p = user_bootstrap([raw_named, f], mte, gte, MODES, n_boot=N_BOOT, seed=SEED)
        p = p.iloc[0]
        smooth_gain.append({"model": f.name, "gain": -p.macro_f1_diff, "gain_ci_low": -p.macro_f1_diff_ci_high,
                            "gain_ci_high": -p.macro_f1_diff_ci_low})
    table = table.merge(pd.DataFrame(smooth_gain), on="model")
    table.sort_values("cv_macro_f1", ascending=False).round(4).to_csv(OUT / "test_results.csv", index=False)
    pairs.round(4).to_csv(OUT / "test_bootstrap_pairwise.csv", index=False)
    pd.DataFrame(classes).round(4).to_csv(OUT / "test_per_class_f1.csv", index=False)
    pd.DataFrame(segs).round(4).to_csv(OUT / "test_segment_metrics.csv", index=False)
    pd.concat(users).round(4).to_csv(OUT / "test_per_user.csv", index=False)
    preds.to_csv(OUT / "window_predictions_test.csv", index=False)
    make_segments(te, preds[sel]).to_csv(OUT / "segments_test.csv", index=False)
    pd.DataFrame(point_confusion(mte, preds[sel], MODES), index=MODES, columns=MODES) \
        .rename_axis("true \\ predicted").to_csv(OUT / "confusion_matrix_final_points.csv")
    run_config(sel)
    pd.set_option("display.width", 220)
    print(table[["model", "selected", "k", "voting", "class_weights", "cv_macro_f1", "point_macro_f1",
                 "macro_f1_ci_low", "macro_f1_ci_high", "point_accuracy", "user_avg_macro_f1"]].round(4)
          .to_string(index=False))


def run_config(selected):
    import joblib
    import matplotlib
    import pyarrow
    (OUT / "run_config.json").write_text(json.dumps({
        "selected_model": selected, "seed": SEED, "n_subsample": N_SUB, "n_folds": N_FOLDS, "n_boot": N_BOOT,
        "ks": KS, "nystroem_components": NYSTROEM_COMPONENTS, "splits_report": str(SPLITS_REPORT),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                     "scikit-learn": sklearn.__version__, "pyarrow": pyarrow.__version__,
                     "matplotlib": matplotlib.__version__, "joblib": joblib.__version__}}, indent=2))


# --------------------------------------------------------------------------------------
# Stage 3: report-only grouped CV over all 64 users (configurations fixed, nothing chosen here)
# --------------------------------------------------------------------------------------
def stage_allusers():
    res = load_cv_results()
    pure = load_windows(HERE / "windows_w50_pure_all.csv")
    blind = load_windows(HERE / "windows_w50_blind.csv")
    tr = pure[pure.n_points >= 10].reset_index(drop=True)
    fold_of = user_folds(tr)
    bl = blind[blind.user_id.isin(fold_of)].reset_index(drop=True)
    Xtr, ytr, utr, _ = prepare(tr)
    Xbl, _, ubl, mbl = prepare(bl)
    fold_tr, fold_bl = utr.map(fold_of).to_numpy(), ubl.map(fold_of).to_numpy()
    specs = model_specs(Xtr.shape[1])
    print(f"[allusers] {len(fold_of)} users in {N_FOLDS} folds; {len(bl):,} blind windows")
    rows = []
    for name, r in res.items():
        spec, cw = specs[name], r["class_weights"] == "balanced"
        pred = np.empty(len(bl), dtype=object)
        scores = np.zeros((len(bl), len(MODES)))
        for f in range(N_FOLDS):
            fit, ev = fold_tr != f, fold_bl == f
            p, s, _ = fit_predict(spec, cw, r["params"], Xtr[fit], ytr[fit], utr[fit], Xbl[ev])
            pred[ev], scores[ev] = p, s
        raw = fold_f1(mbl, pred, fold_bl)
        fin = fold_f1(mbl, smooth(r["voting"], pred, scores, bl.feature_block_id, bl.start_time, r["k"]), fold_bl)
        rows.append({"model": name, "k": r["k"], "voting": r["voting"], "class_weights": r["class_weights"],
                     "raw_mean": raw.mean(), "raw_se": raw.std(ddof=1) / np.sqrt(N_FOLDS),
                     "final_mean": fin.mean(), "final_se": fin.std(ddof=1) / np.sqrt(N_FOLDS),
                     **{f"final_fold{f}": v for f, v in enumerate(fin)}})
        print(f"  {name}: final {fin.mean():.4f} +/- {fin.std(ddof=1) / np.sqrt(N_FOLDS):.4f}")
    pd.DataFrame(rows).round(4).to_csv(OUT / "allusers_cv_report_only.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["cv", "select", "final", "allusers"], required=True)
    ap.add_argument("--models", nargs="*", default=None, help="cv stage: only these models")
    ap.add_argument("--resume", action="store_true", help="cv stage: skip models whose result file exists")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"cv": lambda: stage_cv(a.models, a.resume), "select": select, "final": stage_final,
     "allusers": stage_allusers}[a.stage]()


if __name__ == "__main__":
    main()
