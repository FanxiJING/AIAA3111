"""Independently check saved predictions, reporting joins and protected inputs."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
MODES = ["walk", "bike", "bus", "car", "train"]
SMOOTHERS = ["S0_raw_k1", "S1_centered_soft_k11", "S2_trailing_soft_k11"]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def point_scores(counts, predictions):
    counts = np.asarray(counts, dtype=np.int64)
    indices = pd.Index(MODES).get_indexer(predictions)
    if counts.ndim != 2 or counts.shape[1] != 5 or len(indices) != len(counts):
        raise ValueError("Unaligned point counts")
    if (counts < 0).any() or (indices < 0).any() or counts.sum() <= 0:
        raise ValueError("Invalid class or point counts")
    true = counts.sum(axis=0)
    predicted = np.bincount(indices, weights=counts.sum(axis=1), minlength=5)
    tp = np.array([counts[indices == i, i].sum() for i in range(5)])
    denominator = true + predicted
    per_class = np.divide(2 * tp, denominator, out=np.zeros(5), where=denominator != 0)
    return float(per_class.mean()), float(tp.sum() / counts.sum()), per_class


def main():
    target = HERE / "delivery_verification.json"
    target.unlink(missing_ok=True)
    acceptance = json.loads((HERE / "acceptance.json").read_text(encoding="utf-8"))
    assert acceptance["passed"]
    manifest = json.loads((HERE / "protected_input_manifest.json").read_text(encoding="utf-8"))
    for path, expected in manifest.items():
        assert (ROOT / path).is_file() and sha(ROOT / path) == expected, path
    # Check membership too: added/deleted files can otherwise escape content-only manifests.
    from build_revision import protected_files
    assert {str(p.relative_to(ROOT)) for p in protected_files()} == set(manifest)
    for path, expected in acceptance["output_sha256"].items():
        assert sha(HERE / path) == expected, path

    windows = pd.read_csv(EXP / "feature_cache/sigma00_seed0.csv", dtype={"user_id": str})
    counts = windows[[f"cnt_{m}" for m in MODES]].to_numpy(np.int64)
    primary = pd.read_csv(EXP / "metrics/all_metrics.csv")
    summary = pd.read_csv(HERE / "tables/all_conditions.csv")
    for column in ("point_metric_source", "user_ci_source"):
        for source in summary[column].unique():
            assert (HERE / "tables" / source).is_file(), source
    reconstructed = []
    invariant = ["user_id", "feature_block_id", "start_time", "end_time",
                 "n_points", "label", *[f"cnt_{m}" for m in MODES]]
    files = sorted((EXP / "predictions").glob("*_smoothed.csv"))
    assert len(files) == 48
    for path in files:
        group, condition = path.stem.removesuffix("_smoothed").split("_", 1)
        pred = pd.read_csv(path, dtype={"user_id": str})
        pd.testing.assert_frame_equal(pred[invariant], windows[invariant], check_dtype=False)
        for smoother in SMOOTHERS:
            f1, accuracy, classes = point_scores(counts, pred[smoother])
            source = primary[(primary.group == group) & (primary.condition == condition) &
                             (primary.smoother == smoother)]
            assert len(source) == 1
            row = source.iloc[0]
            np.testing.assert_allclose(
                [f1, accuracy, *classes],
                [row.macro_f1, row.accuracy, *[row[f"f1_{m}"] for m in MODES]],
                rtol=0, atol=1e-12)
            reconstructed.append({"group": group, "sigma": row.sigma,
                                  "smoother": smoother, "macro_f1": f1, "accuracy": accuracy})
    fresh = pd.DataFrame(reconstructed)
    comparisons = 0
    for key, rows in fresh.groupby(["group", "smoother", "sigma"]):
        group, smoother, sigma = key
        out = summary[(summary.group == group) & (summary.smoother == smoother) &
                      (summary.sigma == sigma)].iloc[0]
        for metric in ("macro_f1", "accuracy"):
            np.testing.assert_allclose(out[f"{metric}_seed_mean"], rows[metric].mean(),
                                       atol=1e-12, rtol=0)
            sd = rows[metric].std(ddof=1) if sigma else 0
            np.testing.assert_allclose(out[f"{metric}_seed_sd"], sd, atol=1e-12, rtol=0)
        assert out.seed_count == (5 if sigma else 1)
        comparisons += 1
    assert comparisons == 36 and len(fresh) == 144

    boundary = pd.read_csv(HERE / "tables/boundary_tolerance_seconds.csv", dtype={"user_id": str})
    same = windows.user_id.eq(windows.user_id.shift()) & windows.feature_block_id.eq(
        windows.feature_block_id.shift())
    expected = windows.index[same & windows.label.ne(windows.label.shift())]
    np.testing.assert_array_equal(sorted(boundary.boundary_window_row), sorted(expected))
    assert len(boundary) == 529
    for row in boundary.itertuples():
        part = windows[(windows.user_id == row.user_id) &
                       (windows.feature_block_id == row.feature_block_id)].sort_values("start_time")
        position = part.index.get_loc(row.boundary_window_row)
        eligible = [j for j in range(1, len(part)) if abs(j - position) <= 1]
        true_time = pd.Timestamp(windows.loc[row.boundary_window_row, "start_time"])
        seconds = [(pd.Timestamp(part.iloc[j].start_time) - true_time).total_seconds()
                   for j in eligible]
        np.testing.assert_allclose([row.early_tolerance_s, row.late_tolerance_s,
                                    row.full_candidate_span_s],
                                   [-min(seconds), max(seconds), max(seconds) - min(seconds)],
                                   rtol=0, atol=1e-10)
        assert row.eligible_boundary_positions == len(eligible)

    resolution = json.loads((HERE / "label_resolution.json").read_text(encoding="utf-8"))
    np.testing.assert_allclose(windows.purity, counts.max(axis=1) / counts.sum(axis=1),
                               atol=1e-12, rtol=0)
    assert resolution["mixed_label_windows"] == int(windows.purity.lt(1).sum())
    assert resolution["points_not_in_window_majority"] == int(
        (counts.sum(axis=1) - counts.max(axis=1)).sum())
    assert len(json.loads((ROOT / "data/splits/train_users.json").read_text(encoding="utf-8"))) == 51
    protocol = pd.read_csv(HERE / "tables/protocol_clarified.csv").set_index("item")
    assert "51 fixed" in protocol.loc["split", "details"]
    assert "49 contribute" in protocol.loc["split", "details"]

    pair = pd.read_csv(EXP / "paired_analysis/tables/paired_feature_contrasts.csv")
    pair = pair[(pair.sigma == 20) & (pair.metric == "macro_f1") &
                (pair.comparison == "G3 minus G2")].iloc[0]
    np.testing.assert_allclose(
        [pair.difference_seed_mean, pair.ci_low, pair.ci_high,
         pair.degradation_difference, pair.degradation_ci_low, pair.degradation_ci_high],
        [-.016913, -.031275, .227581, -.079430, -.098176, .139658], atol=5e-7, rtol=0)
    stage0 = json.loads((EXP / "metrics/stage0_baseline.json").read_text(encoding="utf-8"))
    assert stage0["raw_label_diff"] == 1 and stage0["final_label_diff"] == 0
    paired_acceptance = json.loads(
        (EXP / "paired_analysis/acceptance.json").read_text(encoding="utf-8"))
    assert paired_acceptance["draws"] == 5000
    assert paired_acceptance["zero_true_boundary_draws"] == 133
    plot_qa = json.loads((HERE / "plot_layout_audit.json").read_text(encoding="utf-8"))
    assert len(plot_qa) == 3
    assert all(not row["outside_canvas"] and not row["subplot_title_overlaps"] for row in plot_qa)
    raster = json.loads((HERE / "raster_audit.json").read_text(encoding="utf-8"))
    assert len(raster) == 12
    for row in raster:
        assert sha(ROOT / row["path"]) == row["sha256"]
    assert acceptance["manual_figure_review_completed"] is False

    links = 0
    for path in [HERE / name for name in
                 ("README.md", "REPORT_DRAFT.md", "REPORT_TEMPLATE.md")]:
        content = path.read_text(encoding="utf-8")
        for match in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", content):
            target_path = match.group(1).strip("<>").split("#")[0]
            if not target_path or re.match(r"https?://", target_path):
                continue
            assert (path.parent / target_path).exists(), (path.name, target_path)
            links += 1
    assert "{{" not in (HERE / "REPORT_DRAFT.md").read_text(encoding="utf-8")
    tests = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", str(HERE), "-p", "test_*.py", "-v"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    (HERE / "tests.log").write_text(tests.stdout + tests.stderr, encoding="utf-8")
    assert tests.returncode == 0, tests.stdout + tests.stderr
    # Recheck preservation after all verification, without rewriting historical audit snapshots.
    assert all(sha(ROOT / path) == expected for path, expected in manifest.items())
    result = {
        "passed": True, "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protected_file_count": len(manifest), "changed_protected_files": [],
        "independent_point_metric_rows": len(fresh), "per_class_metrics_checked": len(fresh) * 5,
        "independent_summary_rows": comparisons, "boundary_seconds_rows_checked": len(boundary),
        "local_markdown_links_checked": links, "unit_tests": "passed_see_tests.log",
        "manual_visual_review": "pending_image_tool_unavailable",
        "training_or_noise_or_bootstrap_generation": False,
    }
    target = HERE / "delivery_verification.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
