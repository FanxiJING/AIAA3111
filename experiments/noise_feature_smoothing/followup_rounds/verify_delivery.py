"""Check final delivery claims, local links and protected inventory."""
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(2**20), b""):
            h.update(data)
    return h.hexdigest()


def main():
    manifest = json.loads((HERE / "input_manifest.json").read_text(encoding="utf-8"))
    files = [p for folder in ("data", "src", "tests", "docs", "experiments")
             for p in (ROOT / folder).rglob("*") if p.is_file()
             and HERE not in p.parents and "__pycache__" not in p.parts
             and "matplotlib_cache" not in p.parts and p.suffix != ".pyc"
             and p != ROOT / "docs/aigc_log_part2.md"]
    files += [ROOT / n for n in ("README.md", "CLAUDE.md", "pyproject.toml", "uv.lock")]
    actual = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(files)}
    assert actual == manifest["files"], "Protected content/membership changed"
    assert digest(HERE / "ANALYSIS_SPEC.md") == manifest["spec_sha256"]
    assert digest(HERE / "run_rounds.py") == manifest["runner_sha256"]
    for name in ("acceptance.json", "independent_verification.json"):
        assert json.loads((HERE / name).read_text())["passed"]

    links = []
    for path in HERE.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        for label, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            assert (path.parent / target.split("#")[0]).exists(), (path.name, label, target)
            links.append([path.name, target])

    findings = (HERE / "FINDINGS.md").read_text(encoding="utf-8")
    summary = pd.read_csv(HERE / "tables/raw_summary.csv")
    for group in ("G1", "G2", "G3"):
        rows = summary[(summary.family == "original") & (summary.group == group)]
        for r in rows.itertuples():
            assert f"{r.macro_f1_mean:.4f}" in findings
            if r.sigma:
                assert f"{r.macro_f1_seed_sd:.4f}" in findings
        r = rows[rows.sigma == 20].iloc[0]
        for col in ("clean_minus_noisy_loss", "loss_ci_low", "loss_ci_high"):
            assert f"{r[col]:.4f}" in findings

    kernel = pd.read_csv(HERE / "tables/kernel_summary.csv")
    kernel = kernel[(kernel.family == "original") & (kernel.sigma == 20)]
    for r in kernel.itertuples():
        assert f"{r.max_kernel_median:.6f}" in findings
        assert f"{r.score_minus_intercept_norm_median:.6f}" in findings
        assert f"{100*r.intercept_class_point_fraction:.2f}%" in findings

    buckets = pd.read_csv(HERE / "tables/distance_buckets_summary.csv")
    buckets = buckets[(buckets.family == "original") & (buckets.group == "G3") & (buckets.sigma == 20)]
    for r in buckets.itertuples():
        assert f"{r.fixed_clean_reference_distance_change_mean:.4f}" in findings
        assert f"{r.fixed_clean_reference_distance_change_median:.4f}" in findings
    share = (buckets[buckets.bucket.isin(["acceleration", "jerk_turn"])]
             .fixed_clean_reference_distance_change_mean.sum()
             / buckets.fixed_clean_reference_distance_change_mean.sum())
    assert f"{100*share:.2f}%" in findings

    flows = pd.read_csv(HERE / "tables/point_error_flows_summary.csv")
    flows = flows[(flows.family == "original") & (flows.group == "G3") & (flows.sigma == 20)]
    for r in flows.itertuples():
        assert f"{int(r.support):,}" in findings
        assert f"{100*r.damaged/r.support:.2f}%" in findings
        assert f"{100*r.rescued/r.support:.2f}%" in findings
    for col in ("damaged", "rescued"):
        assert f"{flows[col].sum():,.1f}" in findings
    raw = pd.read_csv(HERE / "tables/raw_metrics_by_seed.csv")
    clean_acc = raw[(raw.family == "original") & (raw.group == "G3") & (raw.sigma == 0)].accuracy.iloc[0]
    noisy_acc = raw[(raw.family == "original") & (raw.group == "G3") & (raw.sigma == 20)].accuracy.mean()
    np.testing.assert_allclose(
        (flows.rescued.sum()-flows.damaged.sum())/flows.support.sum(),
        noisy_acc-clean_acc, atol=1e-12)

    contrast = pd.read_csv(HERE / "tables/paired_feature_contrasts.csv")
    for recipe in ("original", "common_gamma"):
        r = contrast[(contrast.recipe == recipe) & (contrast.left == "G2")
                     & (contrast.right == "G1") & (contrast.sigma == 5)].iloc[0]
        for col in ("gap_mean", "gap_ci_low", "gap_ci_high"):
            assert f"{r[col]:.4f}" in findings
    r = contrast[(contrast.recipe == "original") & (contrast.left == "G3")
                 & (contrast.right == "G2") & (contrast.sigma == 20)].iloc[0]
    for col in ("gap_mean", "gap_ci_low", "gap_ci_high"):
        assert f"{r[col]:.4f}" in findings
    omission = pd.read_csv(HERE / "tables/omitted_group_contrasts.csv")
    for right, omitted in (("G2", "linked0"), ("G1", "052")):
        r = omission[(omission.recipe == "original") & (omission.left == "G3")
                     & (omission.right == right) & (omission.sigma == 20)
                     & (omission.omitted_group == omitted)].iloc[0]
        assert f"{r.gap_mean:+.4f}" in findings

    result = {
        "passed": True,
        "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protected_file_content_and_membership": len(actual),
        "local_markdown_links": len(links),
        "numeric_report_claims_checked_against_tables": True,
        "error_flows_equal_accuracy_change": True,
        "input_spec_and_runner_hashes_unchanged": True,
        "independent_verification_passed": True,
        "manual_visual_review": "not_completed_image_tool_unavailable",
        "artifact_hashes": {
            p.relative_to(HERE).as_posix(): digest(p)
            for p in sorted(HERE.rglob("*")) if p.is_file()
            and p.name != "delivery_verification.json"
            and "matplotlib_cache" not in p.parts and "__pycache__" not in p.parts
        },
        "disclosure_log_sha256": digest(ROOT / "docs/aigc_log_part2.md"),
    }
    (HERE / "delivery_verification.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "artifact_hashes"},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
