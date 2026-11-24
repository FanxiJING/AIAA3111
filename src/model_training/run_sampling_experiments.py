"""Optional adaptive parameter search; not used for the formal controlled experiment.

Formal experiment entrypoint: run_reference_sampling.py.
This earlier pilot searches parameters with frozen training-user CV folds.

All conditions finish training-only selection before any new test evaluation is run.
The fixed model family is the reference Nystroem SVM; its original C grid, variant
adoption rules and smoothing grid are reused. --resume requires identical inputs/code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

import pipeline_v2 as pipeline
from models import feature_columns, load_windows

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
MODEL = "SVM (RBF, Nystroem, all data)"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared-root", type=Path, default=PROJECT / "outputs/sampling_main_v1")
    ap.add_argument("--out", type=Path, default=PROJECT / "outputs/sampling_results_v1")
    ap.add_argument("--fold-map", type=Path, help="reuse folds from the primary fixed-point experiment")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    conditions = sorted(a.prepared_root.glob("*/condition.json"),
                        key=lambda p: json.loads(p.read_text())["interval_seconds"])
    if not conditions:
        ap.error("No prepared real conditions found")
    if a.threads < 1:
        ap.error("threads must be positive")
    baseline = load_windows(conditions[0].parent / "windows.csv")
    if json.loads(conditions[0].read_text())["interval_seconds"] != 0:
        ap.error("The first condition must be the original-sampling baseline")
    a.out.mkdir(parents=True, exist_ok=True)
    fold_path = a.fold_map.resolve() if a.fold_map else a.out.resolve() / "frozen_training_folds.json"
    if a.fold_map is None and not fold_path.exists():
        folds = pipeline.user_folds(baseline[(baseline.split == "train") & (baseline.n_points >= 10)])
        fold_path.write_text(json.dumps(folds, indent=2), encoding="utf-8")
    if not fold_path.is_file():
        ap.error("Frozen fold map does not exist")
    protocol = {"model_family": MODEL, "fold_map_sha256": sha(fold_path),
                "threads": a.threads, "seed": pipeline.SEED,
                "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                "pandas": pd.__version__, "sklearn": sklearn.__version__},
                "code_sha256": {p.name: sha(p) for p in
                                [HERE / "pipeline_v2.py", HERE / "models.py", HERE / "smoothing.py",
                                 HERE / "run_sampling_experiments.py"]},
                "split_report_sha256": sha(pipeline.SPLITS_REPORT),
                "conditions": [{"directory": str(p.parent.resolve()),
                                "sha256": {f: sha(p.parent / f) for f in
                                           ["condition.json", "windows.csv", "windows_blind.csv"]}}
                               for p in conditions],
                "selection": "Original v2 C/weight/voting/k search on training users only; same folds across conditions",
                "evaluation_support": "Retained observations, plus a separate all-condition common-support analysis"}
    protocol_path = a.out / "experiment_protocol.json"
    if protocol_path.exists():
        if not a.resume or json.loads(protocol_path.read_text()) != protocol:
            ap.error("Existing experiment: use --resume with unchanged inputs, environment and code, or a new --out")
    else:
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    features = feature_columns(baseline)
    counts = []
    for condition in conditions:
        d = load_windows(condition.parent / "windows.csv")
        if feature_columns(d) != features:
            raise ValueError("Conditions have different input feature sets")
        for split, group in d.groupby("split"):
            fit = group[group.n_points >= 10] if split == "train" else group
            counts.append({"condition": condition.parent.name, "split": split, "points": int(group.n_points.sum()),
                           "windows": len(group), "users": group.user_id.nunique(), "fit_windows": len(fit),
                           "fit_users": fit.user_id.nunique(), "excluded_training_points": int(group.n_points.sum() - fit.n_points.sum()),
                           "features": len(features)})
    pd.DataFrame(counts).to_csv(a.out / "training_data_report.csv", index=False)
    elapsed = {}
    with threadpool_limits(limits=a.threads):
        for condition in conditions:
            name = condition.parent.name
            pipeline.WINDOWS = (condition.parent / "windows.csv").resolve()
            pipeline.BLIND_WINDOWS = (condition.parent / "windows_blind.csv").resolve()
            pipeline.OUT = a.out.resolve() / name
            pipeline.FOLD_MAP = fold_path
            t0 = time.perf_counter()
            print(f"[condition CV] {name}", flush=True)
            pipeline.stage_cv([MODEL], a.resume)
            elapsed[name] = {"cv_this_invocation_seconds": time.perf_counter() - t0}
        choices = {p.parent.name: json.loads((a.out / p.parent.name / "cv/SVM_RBF_Nystroem_all_data.json").read_text())
                   for p in conditions}
        (a.out / "all_choices_frozen_before_test.json").write_text(json.dumps(choices, indent=2), encoding="utf-8")
        for condition in conditions:
            name = condition.parent.name
            pipeline.WINDOWS = (condition.parent / "windows.csv").resolve()
            pipeline.BLIND_WINDOWS = (condition.parent / "windows_blind.csv").resolve()
            pipeline.OUT = a.out.resolve() / name
            pipeline.FOLD_MAP = fold_path
            if a.resume and (pipeline.OUT / "test_results.csv").is_file() and (pipeline.OUT / "run_config.json").is_file():
                print(f"[condition final] {name}: already complete", flush=True)
                continue
            t0 = time.perf_counter()
            print(f"[condition final] {name}", flush=True)
            pipeline.stage_final()
            elapsed[name]["final_seconds"] = time.perf_counter() - t0
            (a.out / "timing.json").write_text(json.dumps(elapsed, indent=2), encoding="utf-8")
    print(f"[done] {a.out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
