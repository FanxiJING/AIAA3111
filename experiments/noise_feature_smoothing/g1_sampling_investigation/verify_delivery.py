"""Check derived class tables, report claims, links and final preservation."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODES = ["walk", "bike", "bus", "car", "train"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    verified = json.loads((HERE / "independent_verification.json").read_text())
    assert verified["status"] == "passed"
    assert verified["verifier_sha256"] == digest(HERE / "verify_analysis.py")
    manifest = json.loads((HERE / "input_manifest.json").read_text())
    assert manifest["runner_sha256"] == digest(HERE / "run_analysis.py")
    assert manifest["spec_sha256"] == digest(HERE / "ANALYSIS_SPEC.md")
    joint = pd.read_csv(HERE / "tables/joint_metrics_by_seed.csv", dtype={"linked_group": str})
    pooled = joint[joint.linked_group == "all"]
    clean = pooled[pooled.sigma == 0].set_index(["group", "dt_bin", "speed_bin"])
    original = pooled.set_index(["group", "sigma", "seed", "dt_bin", "speed_bin"])
    classes = pd.read_csv(HERE / "tables/joint_class_metrics_by_seed.csv")
    assert len(classes) == 4800
    assert not classes.duplicated(["group", "sigma", "seed", "dt_bin", "speed_bin", "true_class"]).any()
    for row in classes.itertuples(index=False):
        ref = original.loc[row.group, row.sigma, row.seed, row.dt_bin, row.speed_bin]
        old = clean.loc[row.group, row.dt_bin, row.speed_bin]
        assert row.support == ref[f"support_{row.true_class}"]
        assert row.true_positives == ref[f"tp_{row.true_class}"]
        assert row.predicted_points == ref[f"predicted_{row.true_class}"]
        assert row.clean_true_positives == old[f"tp_{row.true_class}"]
        expected = 2*row.true_positives/(row.support+row.predicted_points) if (
            row.support+row.predicted_points) else 0.
        np.testing.assert_allclose(row.class_f1, expected, atol=1e-14)
        for col, numerator in (("noisy_recall", row.true_positives),
                               ("clean_recall", row.clean_true_positives)):
            expected = numerator/row.support if row.support else np.nan
            np.testing.assert_allclose(getattr(row, col), expected, atol=1e-14, equal_nan=True)
    report = (HERE / "FINDINGS.md").read_text(encoding="utf-8")
    kernel = pd.read_csv(HERE / "tables/g1_kernel_bins_summary.csv")
    high = kernel[(kernel.sigma == 20) & (kernel.max_kernel_bin == "[0.9,1]")].iloc[0]
    assert f"{high.points_mean:,.1f}" in report
    assert f"{high.accuracy_mean:.2%}" in report
    near = kernel[(kernel.sigma == 20) & (kernel.max_kernel_bin == "all")].iloc[0]
    assert f"{near.nearest_response_changed_point_fraction_mean:.2%}" in report
    assert f"{near.model_nearest_response_agreement_point_fraction_mean:.2%}" in report
    g2 = pooled[(pooled.group == "G2") & (pooled.dt_bin == "<=1s") &
                (pooled.speed_bin == "0.5-2mps") & pooled.sigma.isin([0, 20])]
    assert len(g2) == 6 and (g2.predicted_walk == g2.points).all()
    assert (g2.damaged == 0).all() and (g2.rescued == 0).all()
    assert len(pooled[(pooled.group == "G1") & (pooled.sigma == 0) &
                      (pooled.points > 0)]) == 13
    comp = pd.read_csv(HERE / "tables/composition.csv", dtype={"linked_group": str})
    assert comp[comp.linked_group == "all"].points.sum() == 860972
    links = []
    for path in HERE.glob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            target = target.strip("<>").split("#", 1)[0]
            if not target or "://" in target:
                continue
            resolved = (path.parent / target).resolve()
            if resolved == HERE / "delivery_verification.json":
                continue
            assert resolved.exists(), (path.name, target)
            links.append([path.name, target])
    paths = [p for folder in ("data", "src", "tests", "docs", "experiments")
             for p in (ROOT / folder).rglob("*") if p.is_file()
             and HERE not in p.parents and "__pycache__" not in p.parts
             and "matplotlib_cache" not in p.parts and p.suffix != ".pyc"
             and p != ROOT / "docs/aigc_log_part2.md"]
    paths += [ROOT / n for n in ("README.md", "CLAUDE.md", "pyproject.toml", "uv.lock")]
    protected = manifest["protected"]
    assert set(protected) == {p.relative_to(ROOT).as_posix() for p in paths}
    for name, expected in protected.items():
        assert digest(ROOT / name) == expected, name
    package = {p.relative_to(HERE).as_posix(): digest(p) for p in HERE.rglob("*")
               if p.is_file() and p.name != "delivery_verification.json" and "__pycache__" not in p.parts}
    result = {
        "status": "passed", "derived_class_rows": len(classes), "local_links": len(links),
        "report_numeric_claim_checks": 7, "protected_files": len(protected),
        "protected_content_and_membership_unchanged": True,
        "package_sha256": package, "links": links,
        "verified_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
    }
    (HERE / "delivery_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("package_sha256", "links")}))


if __name__ == "__main__":
    main()
