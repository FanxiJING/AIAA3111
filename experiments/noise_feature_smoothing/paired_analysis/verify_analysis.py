"""Independent verification of paired-analysis delivery; writes only here."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
import unittest

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "src/model_training"))
from models import MODES, cv_groups


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    status = json.loads((HERE / "acceptance.json").read_text(encoding="utf-8"))
    assert status["passed"]
    manifest = json.loads((HERE / "input_manifest.json").read_text(encoding="utf-8"))
    changed = [p for p, sha in manifest["files"].items() if digest(ROOT / p) != sha]
    assert not changed, changed
    assert manifest["analysis_sha256"] == digest(HERE / "build_analysis.py")
    assert manifest["readme_sha256"] == digest(HERE / "README.md")
    suite = unittest.defaultTestLoader.discover(str(HERE), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful() and result.testsRun == 12
    flows = pd.read_csv(HERE / "tables/smoothing_point_flows.csv")
    assert len(flows) == 96
    # Expand correctness by true class counts without calling the analysis flow helper.
    for (group, sigma, seed), frame in flows.groupby(["group", "sigma", "seed"]):
        path = EXP / "predictions" / f"{group}_sigma{sigma:02d}_seed{seed}_smoothed.csv"
        d = pd.read_csv(path, dtype={"user_id": str})
        for _, row in frame.iterrows():
            rescued = damaged = still_wrong = 0
            changed_mask = d.S0_raw_k1 != d[row.smoother]
            for mode in MODES:
                cnt = d[f"cnt_{mode}"]
                rescued += int(cnt[changed_mask & (d[row.smoother] == mode)].sum())
                damaged += int(cnt[changed_mask & (d.S0_raw_k1 == mode)].sum())
                still_wrong += int(cnt[changed_mask & (d.S0_raw_k1 != mode)
                                       & (d[row.smoother] != mode)].sum())
            assert [rescued, damaged, still_wrong] == [
                row.rescued_points, row.damaged_points, row.changed_still_wrong_points]
            assert row.net_correct_points == rescued - damaged
            assert row.changed_prediction_points == rescued + damaged + still_wrong
            np.testing.assert_allclose(row.accuracy_delta_from_raw, (rescued - damaged) / row.points,
                                       rtol=0, atol=1e-12)
    # Reconstruct per-group boundary counts for clean and one prespecified noisy seed.
    # Hungarian assignment is independent of the primary sparse-matching implementation.
    users = pd.read_csv(EXP / "metrics/per_user_results.csv", dtype={"user_id": str})
    checks = []
    for sigma, seed in ((0, 0), (20, 20261123)):
        d = pd.read_csv(EXP / "predictions" / f"G3_sigma{sigma:02d}_seed{seed}_smoothed.csv",
                        dtype={"user_id": str})
        d["linked_group"] = cv_groups(d.user_id, ROOT / "data/splits/report.json")
        for smoother in ("S0_raw_k1", "S1_centered_soft_k11", "S2_trailing_soft_k11"):
            for cluster, sub in d.groupby("linked_group", sort=False):
                true_n = pred_n = hits = 0
                for _, block in sub.groupby(["user_id", "feature_block_id"], sort=False):
                    block = block.sort_values("start_time", kind="stable")
                    pred, true = block[smoother].to_numpy(), block.label.to_numpy()
                    pi = np.flatnonzero(pred[1:] != pred[:-1]) + 1
                    ti = np.flatnonzero(true[1:] != true[:-1]) + 1
                    pred_n += len(pi)
                    true_n += len(ti)
                    edges = np.abs(pi[:, None] - ti[None, :]) <= 1
                    ri, ci = linear_sum_assignment(-edges.astype(int))
                    hits += int(edges[ri, ci].sum())
                selected = users[(users.group == "G3") & (users.sigma == sigma)
                                 & (users.seed == seed) & (users.smoother == smoother)
                                 & users.user_id.isin(sub.user_id.unique())]
                target = selected[["true_transitions", "pred_transitions", "matched_transitions"]].sum()
                assert [true_n, pred_n, hits] == target.tolist()
                checks.append({"sigma": sigma, "seed": seed, "smoother": smoother,
                               "linked_group": cluster, "true_transitions": true_n,
                               "pred_transitions": pred_n, "matched_transitions": hits})
    pd.DataFrame(checks).to_csv(HERE / "tables/independent_boundary_reconstruction.csv", index=False)
    reproduction = pd.read_csv(HERE / "tables/paired_summary_reproduction.csv")
    reference = pd.read_csv(EXP / "metrics/stage5_bootstrap_all.csv")
    pd.testing.assert_frame_equal(reproduction, reference[reproduction.columns],
                                  check_exact=False, rtol=0, atol=1e-12)
    feature = pd.read_csv(HERE / "tables/paired_feature_contrasts.csv")
    smoothing = pd.read_csv(HERE / "tables/paired_smoothing_contrasts.csv")
    interactions = pd.read_csv(HERE / "tables/paired_smoothing_interactions.csv")
    assert len(feature) == 24 and len(smoothing) == 56 and len(interactions) == 16
    np.testing.assert_allclose(feature.loc[feature.sigma == 0, [
        "degradation_difference", "degradation_ci_low", "degradation_ci_high"]], 0, atol=1e-12)
    np.testing.assert_allclose(interactions.loc[interactions.sigma == 0, [
        "difference_in_differences", "ci_low", "ci_high"]], 0, atol=1e-12)
    report = (HERE / "NOISE_HANDOVER.md").read_text(encoding="utf-8")
    links = re.findall(r"\]\(<([^>]+)>\)", report)
    assert all(Path(path).is_file() for path in links)
    assert "\ufffd" not in report and "133/5000" in report
    verification = {
        "passed": True, "verified_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "protected_files_rehashed": len(manifest["files"]), "changed_original_files": changed,
        "unit_tests_passed": result.testsRun, "independent_point_flows_checked": len(flows),
        "independent_boundary_group_rows_checked": len(checks),
        "independent_boundary_conditions": "G3 sigma0 and sigma20 seed20261123, all 3 smoothers",
        "bootstrap_summary_rows_checked": len(reproduction), "report_links_checked": len(links),
        "scope": "post-hoc delivery checks; not new experimental results",
        "manual_visual_review_completed": False}
    (HERE / "delivery_verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    inventory = {str(p.relative_to(HERE)): {"sha256": digest(p), "bytes": p.stat().st_size}
                 for p in sorted(HERE.rglob("*")) if p.is_file()
                 and "__pycache__" not in p.parts and p.name != "output_inventory.json"}
    (HERE / "output_inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
