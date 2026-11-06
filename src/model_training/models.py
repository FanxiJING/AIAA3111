"""
Part 2 - Basic model training for GeoLife transportation-mode classification.

Input = the window table from build_windows.py (one row per window of W GPS points).
This module is importable so the Part 3/4 experiments reuse exactly the same models, grids,
validation scheme and metrics:

    from models import load_windows, prepare, get_models, train_and_evaluate, results_table
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

MODES = ["walk", "bike", "bus", "car", "train"]
LABEL_COL, USER_COL, SPLIT_COL = "label", "user_id", "split"
META_COLS = {"user_id", "split", "trip_id", "feature_block_id", "start_time", "end_time",
             "n_points", "label", "purity", "window_id", *[f"cnt_{m}" for m in MODES]}
# Kept in the window table for QA but not fed to the models: they encode the sampling rate, which
# shifts between users (see README, "Why the sampling-interval features are excluded").
EXCLUDED_FEATURES = ("dt_mean", "dt_max")


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_windows(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={USER_COL: str})
    df[USER_COL] = df[USER_COL].str.zfill(3)
    return df


def feature_columns(df: pd.DataFrame, drop: list[str] | None = None,
                    exclude=EXCLUDED_FEATURES) -> list[str]:
    drop = set(drop or []) | set(exclude)
    return [c for c in df.columns if c not in META_COLS and c not in drop
            and pd.api.types.is_numeric_dtype(df[c])]


def prepare(df: pd.DataFrame, drop: list[str] | None = None, exclude=EXCLUDED_FEATURES):
    """-> X (features), y (window majority label), users, meta (counts etc.)
    Pass exclude=() to include the sampling-interval features."""
    cols = feature_columns(df, drop, exclude)
    X = df[cols].replace([np.inf, -np.inf], np.nan)
    return X, df[LABEL_COL], df[USER_COL], df[[c for c in df.columns if c in META_COLS]]


def cv_groups(users: pd.Series, splits_report: str | Path | None) -> pd.Series:
    """Users that share duplicate records (Part 1's duplicate_linked_groups) must stay in the
    same CV fold, otherwise validation leaks. Returns one group id per row."""
    gid = {}
    if splits_report and Path(splits_report).exists():
        rep = json.loads(Path(splits_report).read_text(encoding="utf-8"))
        for i, grp in enumerate(rep.get("duplicate_linked_groups", [])):
            for u in grp:
                gid[str(u).zfill(3)] = f"linked{i}"
    return users.map(lambda u: gid.get(u, u))


# --------------------------------------------------------------------------------------
# Model zoo: name -> (pipeline, small grid, max training windows or None)
# --------------------------------------------------------------------------------------
def _scaled(clf):
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()), ("clf", clf)])


def _unscaled(clf):
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", clf)])


def get_models(seed: int = 42) -> dict:
    return {
        "Majority (baseline)": (_unscaled(DummyClassifier(strategy="most_frequent")), {}, None),
        "Logistic Regression": (
            _scaled(LogisticRegression(max_iter=3000, class_weight="balanced")),
            {"clf__C": [0.1, 1.0, 10.0]}, None),
        "Decision Tree": (
            _unscaled(DecisionTreeClassifier(class_weight="balanced", random_state=seed)),
            {"clf__max_depth": [8, 16, None], "clf__min_samples_leaf": [1, 10]}, None),
        "k-NN": (
            _scaled(KNeighborsClassifier()),
            {"clf__n_neighbors": [5, 15, 31], "clf__weights": ["uniform", "distance"]}, None),
        # SVC scales ~quadratically -> trained on a stratified subsample of the training windows
        "SVM (RBF)": (
            _scaled(SVC(kernel="rbf", class_weight="balanced", random_state=seed)),
            {"clf__C": [1.0, 10.0], "clf__gamma": ["scale"]}, 20_000),
        "Random Forest": (
            _unscaled(RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                             n_jobs=-1, random_state=seed)),
            {"clf__min_samples_leaf": [1, 5], "clf__max_features": ["sqrt", 0.5]}, None),
        "Gradient Boosting (HGB)": (
            # fixed max_iter, no early stopping: its internal validation split is random, not
            # grouped by user, so it would leak users between fitting and stopping
            _unscaled(HistGradientBoostingClassifier(class_weight="balanced", random_state=seed,
                                                     early_stopping=False, max_iter=300)),
            {"clf__learning_rate": [0.05, 0.1], "clf__max_leaf_nodes": [31, 63],
             "clf__l2_regularization": [0.0, 1.0]}, None),
    }


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
def window_metrics(y_true, y_pred, labels):
    return {"accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)}


def point_confusion(meta: pd.DataFrame, y_pred, labels) -> np.ndarray:
    """Every point inherits its window's prediction. cm[i, j] = #points of true class i
    predicted as j. Uses the per-window class counts, so mixed windows are scored fairly."""
    cnt = meta[[f"cnt_{m}" for m in labels]].to_numpy()
    pred_idx = pd.Categorical(y_pred, categories=labels).codes
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for j in range(len(labels)):
        cm[:, j] = cnt[pred_idx == j].sum(axis=0)
    return cm


def metrics_from_cm(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(float)
    recall = tp / cm.sum(1).clip(min=1)
    precision = tp / cm.sum(0).clip(min=1)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    return {"accuracy": tp.sum() / cm.sum(), "balanced_accuracy": recall.mean(),
            "macro_f1": f1.mean(), "per_class_f1": f1}


# --------------------------------------------------------------------------------------
# Train + evaluate
# --------------------------------------------------------------------------------------
@dataclass
class Result:
    name: str
    estimator: object
    best_params: dict
    cv_macro_f1: float
    window: dict
    point: dict
    y_pred: np.ndarray
    window_cm: np.ndarray | None = None
    point_cm: np.ndarray | None = None
    fit_seconds: float = 0.0
    report: dict = field(default_factory=dict)
    cv_macro_f1_std: float = float("nan")      # spread across CV folds for the chosen parameters


def _subsample(X, y, g, n, seed):
    if n is None or len(X) <= n:
        return X, y, g
    rng = np.random.default_rng(seed)
    pos = np.arange(len(X))
    idx = np.concatenate([rng.choice(pos[y.values == c], size=max(1, round(n * (y.values == c).mean())),
                                     replace=False) for c in pd.unique(y)])
    idx.sort()
    return X.iloc[idx], y.iloc[idx], g.iloc[idx]


def train_and_evaluate(X_train, y_train, g_train, X_test, y_test, meta_test,
                       models: dict | None = None, labels=MODES, tune=True,
                       cv_splits=3, seed=42, verbose=True) -> list[Result]:
    """Fit each model (grid search with user-grouped CV on training users only, scoring
    macro-F1), then score on the test users at window level and point level."""
    models = models or get_models(seed)
    labels = [l for l in labels if l in set(y_train) | set(y_test)]
    out = []
    for name, (pipe, grid, max_n) in models.items():
        t0 = time.time()
        Xt, yt, gt = _subsample(X_train, y_train, g_train, max_n, seed)
        if tune and grid:
            cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
            s = GridSearchCV(clone(pipe), grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)
            s.fit(Xt, yt, groups=gt)
            est, best, cvs = s.best_estimator_, s.best_params_, s.best_score_
            cv_std = float(s.cv_results_["std_test_score"][s.best_index_])
        else:
            est, best, cvs = clone(pipe).fit(Xt, yt), {}, float("nan")
            cv_std = float("nan")
        yp = est.predict(X_test)
        pcm = point_confusion(meta_test, yp, labels)
        r = Result(name=name, estimator=est, best_params=best, cv_macro_f1=cvs, cv_macro_f1_std=cv_std,
                   window=window_metrics(y_test, yp, labels), point=metrics_from_cm(pcm),
                   y_pred=yp, window_cm=confusion_matrix(y_test, yp, labels=labels),
                   point_cm=pcm, fit_seconds=round(time.time() - t0, 1),
                   report=classification_report(y_test, yp, labels=labels, output_dict=True,
                                                zero_division=0))
        out.append(r)
        if verbose:
            print(f"  {name:<25} point acc={r.point['accuracy']:.3f} macroF1={r.point['macro_f1']:.3f} | "
                  f"window macroF1={r.window['macro_f1']:.3f} | cvF1={cvs:.3f} +/- {cv_std:.3f} "
                  f"({r.fit_seconds}s{', n=' + str(len(Xt)) if max_n else ''}) {best}")
    return out


def select_best(results: list[Result]) -> Result | None:
    """Pick the model with the highest CV macro-F1 on training users. Test scores are never
    used. Returns None when no model has a CV score (e.g. --no-tune); the majority baseline
    has no grid, so it has no CV score and cannot be selected."""
    scored = [r for r in results if np.isfinite(r.cv_macro_f1)]
    return max(scored, key=lambda r: r.cv_macro_f1) if scored else None


def results_table(results: list[Result]) -> pd.DataFrame:
    """One row per model, sorted by test score for reading only. Use the selected_by_cv column
    (or select_best), not the first row, to find the selected model."""
    best = select_best(results)
    rows = [{"model": r.name, "selected_by_cv": r is best,
             "point_accuracy": r.point["accuracy"], "point_balanced_acc": r.point["balanced_accuracy"],
             "point_macro_f1": r.point["macro_f1"],
             "window_accuracy": r.window["accuracy"], "window_macro_f1": r.window["macro_f1"],
             "cv_macro_f1": r.cv_macro_f1, "cv_macro_f1_std": r.cv_macro_f1_std,
             "fit_seconds": r.fit_seconds, "best_params": r.best_params} for r in results]
    t = pd.DataFrame(rows).sort_values("point_macro_f1", ascending=False).reset_index(drop=True)
    num = t.select_dtypes("number").columns
    t[num] = t[num].round(4)
    return t


# --------------------------------------------------------------------------------------
# User-level uncertainty on the test set
# --------------------------------------------------------------------------------------
def _macro_f1_accuracy(cms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scores for a stack of confusion matrices (n, k, k). Macro-F1 averages over the classes
    with true points in that matrix, because a resample may not contain every class."""
    tp = np.diagonal(cms, axis1=1, axis2=2).astype(float)
    support, predicted = cms.sum(axis=2), cms.sum(axis=1)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    present = support > 0
    macro = (f1 * present).sum(axis=1) / present.sum(axis=1)
    return macro, tp.sum(axis=1) / cms.sum(axis=(1, 2))


def user_bootstrap(results: list[Result], meta_test: pd.DataFrame, groups_test, labels,
                   n_boot: int = 5000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """95% percentile CIs for test point macro-F1 and accuracy, resampling test users with
    replacement. groups_test holds one id per window (from cv_groups), so duplicate-linked users
    are drawn together. Every model sees the same resamples, so differences are paired.
    -> (one row per model, one row per model pair)"""
    groups = np.asarray(groups_test)
    clusters = pd.unique(groups)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)), size=n_boot)

    scores, ci_rows = {}, []
    for r in results:
        per = np.stack([point_confusion(meta_test[groups == c], r.y_pred[groups == c], labels)
                        for c in clusters])
        macro, acc = _macro_f1_accuracy(np.tensordot(draws, per, axes=1))
        est_macro, est_acc = _macro_f1_accuracy(per.sum(axis=0, keepdims=True))
        scores[r.name] = {"macro_f1": (macro, est_macro[0]), "accuracy": (acc, est_acc[0])}
        ci_rows.append({"model": r.name,
                        "point_macro_f1": est_macro[0],
                        "macro_f1_ci_low": np.percentile(macro, 2.5),
                        "macro_f1_ci_high": np.percentile(macro, 97.5),
                        "point_accuracy": est_acc[0],
                        "accuracy_ci_low": np.percentile(acc, 2.5),
                        "accuracy_ci_high": np.percentile(acc, 97.5)})

    pair_rows = []
    for i, a in enumerate(results):
        for b in results[i + 1:]:
            row = {"model_a": a.name, "model_b": b.name}
            for metric in ("macro_f1", "accuracy"):
                (boot_a, est_a), (boot_b, est_b) = scores[a.name][metric], scores[b.name][metric]
                low, high = np.percentile(boot_a - boot_b, [2.5, 97.5])
                row.update({f"{metric}_diff": est_a - est_b,
                            f"{metric}_diff_ci_low": low, f"{metric}_diff_ci_high": high,
                            f"{metric}_ci_includes_0": bool(low <= 0 <= high)})
            pair_rows.append(row)
    return pd.DataFrame(ci_rows), pd.DataFrame(pair_rows)


def per_user_accuracy(results: list[Result], meta_test: pd.DataFrame, users_test, groups_test,
                      labels) -> pd.DataFrame:
    """Point accuracy of every model for every test user, largest user first."""
    users, groups = np.asarray(users_test), np.asarray(groups_test)
    n_points = meta_test["n_points"].to_numpy()
    total = n_points.sum()
    rows = []
    for u in pd.unique(users):
        m = users == u
        row = {"user_id": u, "resample_group": groups[m][0], "points": int(n_points[m].sum()),
               "point_share": n_points[m].sum() / total}
        for r in results:
            cm = point_confusion(meta_test[m], r.y_pred[m], labels)
            row[r.name] = np.trace(cm) / cm.sum()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("points", ascending=False).reset_index(drop=True)
