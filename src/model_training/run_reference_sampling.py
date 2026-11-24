"""Controlled sampling experiments: freeze Part 2's C=1/balanced/soft reference model.

Only k is selected on frozen training-user folds. Each completed fold is checkpointed.
No new test evaluation is performed in the cv stage.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata as md
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import pipeline_v2 as p
from models import MODES, metrics_from_cm, point_confusion, prepare

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
MODEL = "SVM (RBF, Nystroem, all data)"
CV_FILE = "SVM_RBF_Nystroem_all_data.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cv(resume):
    directory = p.OUT / "cv"
    directory.mkdir(parents=True, exist_ok=True)
    if resume and (directory / CV_FILE).exists():
        print(f"[cv complete] {p.OUT.name}", flush=True)
        return
    started = time.perf_counter()
    train, blind, folds = p.load_cv_data()
    Xtr, ytr, utr, _ = prepare(train)
    Xbl, _, ubl, meta = prepare(blind)
    fit_fold, eval_fold = utr.map(folds).to_numpy(), ubl.map(folds).to_numpy()
    spec = p.model_specs(Xtr.shape[1])[MODEL]
    raw = np.empty(len(blind), dtype=object)
    scores = np.empty((len(blind), len(MODES)))
    for fold in range(p.N_FOLDS):
        fit, evaluate = fit_fold != fold, eval_fold == fold
        path = directory / f"oof_fold_{fold}.npz"
        if resume and path.exists():
            cache = np.load(path, allow_pickle=False)
            if not np.array_equal(cache["indices"], np.flatnonzero(evaluate)):
                raise ValueError("Checkpoint indices do not match frozen folds")
            raw[evaluate], scores[evaluate] = cache["predictions"], cache["scores"]
            print(f"[cached fold] {p.OUT.name} {fold + 1}/5", flush=True)
            continue
        print(f"[fit] {p.OUT.name} fold={fold + 1}/5 train={fit.sum()} eval={evaluate.sum()} C=1 balanced", flush=True)
        predictions, values, _ = p.fit_predict(spec, True, {"clf__C": 1.0}, Xtr[fit], ytr[fit], utr[fit], Xbl[evaluate])
        raw[evaluate], scores[evaluate] = predictions, values
        np.savez_compressed(path, indices=np.flatnonzero(evaluate), predictions=np.asarray(predictions, dtype="U5"), scores=values)
    per_k = {k: p.fold_f1(meta, p.smooth("soft", raw, scores, blind.feature_block_id, blind.start_time, k), eval_fold)
             for k in p.KS}
    k, best = p.paired_one_se_k(per_k)
    chosen = p.smooth("soft", raw, scores, blind.feature_block_id, blind.start_time, k)
    pooled = metrics_from_cm(point_confusion(meta, chosen, MODES))
    result = {"model": MODEL, "params": {"clf__C": 1.0}, "subsample": False,
              "class_weights": "balanced", "voting": "soft", "k": k, "k_best_mean": best,
              "fold_f1": per_k[k].tolist(), "mean_f1": float(per_k[k].mean()),
              "se_f1": float(per_k[k].std(ddof=1) / np.sqrt(p.N_FOLDS)),
              "raw_fold_f1": p.fold_f1(meta, raw, eval_fold).tolist(),
              "pooled_macro_f1": float(pooled["macro_f1"]), "pooled_accuracy": float(pooled["accuracy"]),
              "per_k_fold_f1": {str(key): value.tolist() for key, value in per_k.items()},
              "protocol": "Fixed reference family/C/weights/voting; training-only k selection",
              "seconds": time.perf_counter() - started}
    (directory / CV_FILE).write_text(json.dumps(result, indent=2), encoding="utf-8")
    p.select()
    print(f"[cv done] {p.OUT.name}: k={k}, mean={result['mean_f1']:.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["cv", "final"], required=True)
    ap.add_argument("--prepared-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--intervals", type=float, nargs="+", default=[0, 5, 10, 30])
    ap.add_argument("--fold-map", type=Path, default=PROJECT / "outputs/sampling_results_v1/frozen_training_folds.json")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    conditions = sorted([c for c in a.prepared_root.glob("*/condition.json")
                         if json.loads(c.read_text())["interval_seconds"] in a.intervals],
                        key=lambda c: json.loads(c.read_text())["interval_seconds"])
    if not conditions or a.threads < 1:
        ap.error("No matching conditions, or invalid thread count")
    p.FOLD_MAP = a.fold_map.resolve()
    for condition in conditions:
        p.OUT = a.out.resolve() / condition.parent.name
        p.WINDOWS = (condition.parent / "windows.csv").resolve()
        p.BLIND_WINDOWS = (condition.parent / "windows_blind.csv").resolve()
        p.OUT.mkdir(parents=True, exist_ok=True)
        protocol = {"reference_model": MODEL, "C": 1.0, "class_weights": "balanced", "voting": "soft",
                    "k_candidates": p.KS, "fold_map_sha256": sha(p.FOLD_MAP),
                    "input_sha256": {f: sha(condition.parent / f) for f in ["condition.json", "windows.csv", "windows_blind.csv"]},
                    "code_sha256": {f: sha(HERE / f) for f in ["pipeline_v2.py", "models.py", "smoothing.py", "run_reference_sampling.py"]},
                    "python": platform.python_version(), "threads": a.threads,
                    "versions": {m: md.version(m) for m in ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy"]}}
        path = p.OUT / "controlled_protocol.json"
        if path.exists() and json.loads(path.read_text()) != protocol:
            raise ValueError("Protocol or inputs changed; use a separate output directory")
        path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        windows = p.load_windows(p.WINDOWS)
        if not ((windows.split == "train") & (windows.n_points >= 10)).any():
            (p.OUT / "infeasible.json").write_text(json.dumps({"reason": "No training windows have at least 10 points"}))
            print(f"[infeasible] {condition.parent.name}", flush=True)
            continue
        with threadpool_limits(a.threads):
            if a.stage == "cv":
                cv(a.resume)
            elif not (a.resume and (p.OUT / "run_config.json").exists()):
                if not (p.OUT / "cv" / CV_FILE).exists():
                    raise ValueError("Complete training-only CV before final evaluation")
                p.stage_final()


if __name__ == "__main__":
    main()
