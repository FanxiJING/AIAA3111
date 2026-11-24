"""Independent delivery checks; writes only within this review directory."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    target = HERE / "delivery_verification.json"
    target.unlink(missing_ok=True)
    before = json.loads((HERE / "protected_input_manifest.json").read_text(encoding="utf-8"))
    changed = [p for p, value in before.items() if sha(ROOT / p) != value]
    assert not changed, changed
    acceptance = json.loads((HERE / "acceptance.json").read_text(encoding="utf-8"))
    assert acceptance["passed"] and acceptance["metric_rows_reconstructed"] == 144
    assert acceptance["protected_files"] == len(before)
    for name, expected in acceptance["code_sha256"].items():
        assert sha(HERE / name) == expected, name

    reconstructed = pd.read_csv(HERE / "tables/reconstructed_metrics_and_concentration.csv")
    diagnostic = pd.read_csv(EXP / "diagnostics/tables/confusion_matrices.csv")
    keys = ["group", "condition", "smoother"]
    cm_count = 0
    for key, rows in diagnostic.groupby(keys):
        cm = rows.pivot(index="true_mode", columns="pred_mode", values="points").fillna(0)
        modes = ["walk", "bike", "bus", "car", "train"]
        cm = cm.reindex(index=modes, columns=modes, fill_value=0).to_numpy(float)
        tp = np.diag(cm)
        denominator = cm.sum(axis=0) + cm.sum(axis=1)
        f1 = np.mean(np.divide(2 * tp, denominator,
                              out=np.zeros(5), where=denominator != 0))
        accuracy = np.trace(cm) / cm.sum()
        mask = np.ones(len(reconstructed), bool)
        for name, value in zip(keys, key):
            mask &= reconstructed[name].eq(value).to_numpy()
        ref = reconstructed[mask]
        assert len(ref) == 1
        np.testing.assert_allclose([f1, accuracy], ref[["macro_f1", "accuracy"]].iloc[0],
                                   atol=1e-12, rtol=0)
        cm_count += 1
    assert cm_count == 144

    constant = pd.read_csv(HERE / "tables/constant_class_references.csv").set_index("constant_mode")
    np.testing.assert_allclose(constant.loc["walk", "accuracy"], 309175 / 860972, atol=1e-15)
    np.testing.assert_allclose(constant.loc["train", "macro_f1"],
                               2 * 87288 / (860972 + 87288) / 5, atol=1e-15)
    support = pd.read_csv(HERE / "tables/actual_interval_support.csv").set_index("interval_stratum")
    assert support.loc["all", "intervals"] == 859763
    assert support.loc[support.index != "all", "intervals"].sum() == 859763
    short = (support.loc["<=1s", "intervals"] + support.loc["1-2s", "intervals"]) / 859763
    np.testing.assert_allclose(short, .8286016029998965, atol=1e-12)
    theory = pd.read_csv(HERE / "tables/analytical_stationary_reference.csv")
    row = theory[(theory.sigma_axis_m == 5) & (theory.dt_s == 2)].iloc[0]
    np.testing.assert_allclose(row.stationary_mean_speed_mps, 5 * np.sqrt(np.pi) / 2)
    np.testing.assert_allclose(row.p_below_stop_threshold, 1 - np.exp(-.01))
    assert len(pd.read_csv(HERE / "tables/constant_reference_contrasts.csv")) == 720

    records = json.loads((HERE / "source_verification.json").read_text(encoding="utf-8"))
    assert len(records) == 8
    title_checks = 0
    for record in records:
        if record["key"] != "skill":
            assert record["title_verified"], record["key"]
            title_checks += 1
        for response in record["requests"]:
            if response["ok"]:
                assert sha(HERE / response["path"]) == response["sha256"]
    farha = next(r for r in records if r["key"] == "farha")
    assert farha["doi"].endswith("00369")
    lu = next(r for r in records if r["key"] == "lu")
    assert lu["metadata"]["published"]["date-parts"][0][:2] == [2026, 1]
    skill = (HERE / "sources/skill_primary.txt").read_text(encoding="utf-8")
    assert all(s in skill for s in ("Scientific Agent Skills", "Timothy Kassis", "2 Sep 2026"))
    for name in ("README.md", "REVIEW_REPORT.md", "SOURCES.md"):
        assert (HERE / name).stat().st_size > 0
    report = (HERE / "REVIEW_REPORT.md").read_text(encoding="utf-8")
    assert all(value in report for value in ("82.8602%", "4.4311m/s", "0.105688", "0.137013"))

    result = subprocess.run(
        [sys.executable, "-B", "-m", "unittest", "discover", "-s", str(HERE),
         "-p", "test_*.py", "-v"], text=True, capture_output=True, check=False)
    (HERE / "tests.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    assert "Ran 9 tests" in result.stderr
    inventory = {str(p.relative_to(HERE)): sha(p) for p in HERE.rglob("*")
                 if p.is_file() and p.name not in ("output_inventory.json",
                                                   "delivery_verification.json")
                 and "__pycache__" not in p.parts}
    (HERE / "output_inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    target.write_text(json.dumps({
        "passed": True, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protected_file_count": len(before), "changed_protected_files": changed,
        "independent_diagnostic_confusion_comparisons": cm_count,
        "known_answer_tests_passed": 9, "doi_title_pairs_verified": title_checks,
        "bibliographic_records": len(records),
        "source_scope": "selected_R1_sections_R2_R3_R5_R6_abstracts_R4_R7_metadata_skill_record",
        "full_text_review_claimed_for_all_sources": False,
        "manual_figure_review_completed": False,
        "new_training_or_noise_runs": False,
        "docs_aigc_log_append_is_outside_protected_numerical_inputs": True,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(target.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
