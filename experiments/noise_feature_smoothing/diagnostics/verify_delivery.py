"""Independent output checks; does not modify the primary experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from PIL import Image
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    original = json.loads((HERE / "input_manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((HERE / "audit.json").read_text(encoding="utf-8"))
    assert audit["passed"] and audit["protected_files_changed"] == []
    layout = json.loads((HERE / "plot_layout_audit.json").read_text(encoding="utf-8"))
    assert layout["passed"] and len(layout["figures"]) == 6
    assert (HERE / "VISUAL_QA.md").is_file() and (HERE / "SOURCES.md").is_file()
    assert original["analysis_code_sha256"] == digest(HERE / "analyze.py")
    assert original["analysis_spec_sha256"] == digest(HERE / "ANALYSIS_SPEC.md")
    changed = [
        str(path) for path, info in original["inputs"].items()
        if digest(ROOT / path) != info["sha256"]
    ]
    assert not changed, changed
    parquet_files = sorted((HERE / "window_diagnostics").glob("*.parquet"))
    assert len(parquet_files) == 48
    parquet_rows = []
    for path in parquet_files:
        meta = pq.read_metadata(path)
        assert meta.num_rows == 17380, path
        d = pd.read_parquet(path)
        assert d.n_points.sum() == 860972
        assert d.window_row.is_unique and d.window_row.tolist() == list(range(17380))
        assert d.user_id.nunique() == 13 and d.linked_group.nunique() == 9
        assert np.isfinite(d.max_score_residual_from_intercept).all()
        near = d[d.near_intercept]
        assert near.raw_prediction.eq(near.intercept_winner).all()
        parquet_rows.append(meta.num_rows)
    checks = pd.read_csv(HERE / "tables/metric_reconstruction_checks.csv")
    assert len(checks) == 144
    assert checks.macro_f1_error.abs().max() < 1e-12
    assert checks.accuracy_error.abs().max() < 1e-12
    bounds = pd.read_csv(HERE / "tables/boundary_summary.csv")
    assert len(bounds) == 48 and bounds.true_transitions.eq(529).all()
    assert (bounds.missed_transitions + bounds.matched_transitions).eq(529).all()
    details = pd.read_csv(HERE / "tables/boundary_matches.csv")
    assert len(details) == 48 * 529
    matched = details[details.matched]
    assert matched.offset_windows.abs().le(1).all()
    assert not matched.duplicated(
        ["condition", "smoother", "feature_block_id", "pred_boundary_row"]).any()
    boot = pd.read_csv(HERE / "tables/bootstrap_reproduction.csv")
    assert len(boot) == 36 and boot.bootstrap_draws.eq(5000).all()
    support = pd.read_csv(HERE / "tables/smoothing_window_support.csv")
    assert len(support) == 3 * 17380
    assert support[support.smoother.eq("S2_trailing_soft_k11")].future_window_wait_s.eq(0).all()
    assert support.available_windows.between(1, 11).all()
    images = []
    for path in sorted((HERE / "figures").glob("*.png")):
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
            fraction = float((rgb.min(axis=2) < 240).mean())
            assert im.width >= 1400 and im.height >= 900
            assert .01 < fraction < .95
            images.append({"file": path.name, "width": im.width, "height": im.height,
                           "nonwhite_fraction": fraction, "sha256": digest(path)})
        assert path.with_suffix(".pdf").stat().st_size > 1000
    assert len(images) == 6
    suite = unittest.defaultTestLoader.discover(str(HERE), pattern="test_diagnostics.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful() and result.testsRun == 7
    output = {
        "passed": True, "checked_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "primary_files_rehashed": len(original["inputs"]),
        "changed_primary_files": changed, "parquet_files": len(parquet_files),
        "parquet_rows": int(sum(parquet_rows)), "reconciled_metric_rows": len(checks),
        "boundary_rows": len(details), "bootstrap_rows": len(boot),
        "tests_run": result.testsRun, "images": images,
        "rendered_text_boundary_check_passed": layout["passed"],
        "manual_visual_review_completed": False,
        "visual_review": "See VISUAL_QA.md; pixel checks alone do not verify layout.",
        "report_present": (HERE / "DIAGNOSTIC_REPORT.md").is_file(),
    }
    assert output["report_present"]
    (HERE / "delivery_audit.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    inventory = {
        str(path.relative_to(HERE)): {"sha256": digest(path), "bytes": path.stat().st_size}
        for path in sorted(HERE.rglob("*"))
        if path.is_file() and "matplotlib_cache" not in path.parts
        and "__pycache__" not in path.parts and path.name != "delivery_inventory.json"
    }
    (HERE / "delivery_inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
