"""Check the delivered supplemental package and cited descriptive slices."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    results = {
        name: json.loads((HERE / name).read_text(encoding="utf-8"))
        for name in ("acceptance.json", "independent_verification.json",
                     "diagnostics/render_recovery.json", "input_manifest.json")
    }
    for name in ("acceptance.json", "independent_verification.json",
                 "diagnostics/render_recovery.json"):
        assert results[name]["passed"]
    protected = results["input_manifest.json"]["files"]
    for rel, digest in protected.items():
        assert sha(ROOT / rel) == digest, rel
    current = set()
    for folder in ("data", "src", "tests", "docs", "experiments"):
        for path in (ROOT / folder).rglob("*"):
            if (path.is_file() and HERE not in path.parents
                    and "__pycache__" not in path.parts
                    and "matplotlib_cache" not in path.parts
                    and path.suffix != ".pyc" and path != ROOT / "docs/aigc_log_part2.md"):
                current.add(path.relative_to(ROOT).as_posix())
    current.update(("README.md", "CLAUDE.md", "pyproject.toml", "uv.lock"))
    assert current == set(protected), "Original protected membership changed"
    for name, field in (("run_supplement.py", "code_sha256"), ("PROTOCOL.md", "protocol_sha256")):
        assert sha(HERE / name) == results["input_manifest.json"][field]
    assert sha(HERE / "verify_supplement.py") == results["independent_verification.json"]["verifier_sha256"]
    assert sha(HERE / "complete_delivery.py") == results["diagnostics/render_recovery.json"]["recovery_script_sha256"]

    expected = {
        "all_seed_metrics.csv": 288, "summary.csv": 72, "paired_contrasts.csv": 504,
        "per_user_metrics.csv": 3744, "per_class_f1.csv": 1440,
        "leave_one_group_out_seed.csv": 2880, "noise_structure_by_omitted_group.csv": 40,
        "noise_increment_summary.csv": 6, "G3_clean_and_20m_class_summary.csv": 20,
    }
    for name, count in expected.items():
        assert len(pd.read_csv(HERE / "tables" / name)) == count, name
    assert len(list((HERE / "models").glob("*.joblib"))) == 2
    assert len(list((HERE / "predictions").glob("*.csv"))) == 47
    assert len(list((HERE / "feature_cache").glob("*.csv"))) == 15

    group = pd.read_csv(HERE / "tables/noise_structure_by_omitted_group.csv")
    full = group[group.omitted_group.eq("none") & group.sigma.eq(20)].iloc[0]
    omit = group[group.omitted_group.eq("linked0") & group.sigma.eq(20)].iloc[0]
    np.testing.assert_allclose(full.OU30_minus_iid_f1, full.macro_f1_OU30 - full.macro_f1_iid)
    assert full.OU30_minus_iid_f1 > 0 and omit.OU30_minus_iid_f1 < 0
    assert omit.retained_points == 181810 and omit.support_train == 3
    notes = (HERE / "FINDINGS.md").read_text(encoding="utf-8")
    for val in (full.macro_f1_OU30, full.macro_f1_iid, omit.macro_f1_OU30,
                omit.macro_f1_iid, full.OU30_minus_iid_f1, omit.OU30_minus_iid_f1):
        assert f"{val:.4f}" in notes
    increments = pd.read_csv(HERE / "tables/noise_increment_summary.csv")
    for row in increments[increments.sigma.eq(20)].itertuples():
        for name in ("radial_rms_m_mean", "error_increment_vector_rms_m_mean",
                     "error_increment_div_dt_vector_rms_mps_mean"):
            assert f"{getattr(row, name):.4f}" in notes
    figures = []
    for path in sorted((HERE / "figures").glob("*.png")):
        pixels = np.asarray(Image.open(path).convert("RGB"))
        assert pixels.std() > 5 and path.with_suffix(".pdf").stat().st_size > 1000
        figures.append({"file": path.name, "shape": list(pixels.shape), "nonblank": True})
    assert len(figures) == 4

    link_count = 0
    for path in HERE.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            if "://" in target or target.startswith("#"):
                continue
            assert (path.parent / target.split("#", 1)[0]).exists(), (path.name, target)
            link_count += 1
    body = {
        "passed": True, "protected_original_files_content_and_membership": len(protected),
        "runner_and_protocol_unchanged": True, "independent_numeric_verification_passed": True,
        "table_row_counts": expected, "local_markdown_links_checked": link_count,
        "key_findings_and_noise_increment_text_checked": True,
        "figures_automated_nonblank": figures, "manual_visual_review": "not_completed",
        "delivery_verifier_sha256": sha(Path(__file__)),
    }
    (HERE / "delivery_verification.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
    print(json.dumps(body, indent=2), flush=True)


if __name__ == "__main__":
    main()
