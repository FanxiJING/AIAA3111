"""Traceable descriptive slices for the supplemental findings; no new fitting."""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = "S0_raw_k1"


def main():
    tables = HERE / "tables"
    loo = pd.read_csv(tables / "leave_one_group_out_summary.csv")
    original = loo[loo.family.eq("original") & loo.group.eq("G3") & loo.smoother.eq(RAW)]
    correlated = loo[loo.family.eq("correlated_noise") & loo.group.eq("G3") & loo.smoother.eq(RAW)]
    keys = ["group", "smoother", "omitted_group", "sigma"]
    supports = ["retained_points", "true_classes_present", *[
        f"support_{mode}" for mode in ("walk", "bike", "bus", "car", "train")
    ]]
    compare = original[keys + ["macro_f1", *supports]].merge(
        correlated[keys + ["macro_f1", *supports]],
        on=keys, suffixes=("_iid", "_OU30"), validate="one_to_one",
    )
    for name in supports:
        np.testing.assert_array_equal(compare[f"{name}_iid"], compare[f"{name}_OU30"])
        compare[name] = compare.pop(f"{name}_iid").astype(int)
        compare.drop(columns=f"{name}_OU30", inplace=True)
    compare["OU30_minus_iid_f1"] = compare.macro_f1_OU30 - compare.macro_f1_iid
    compare.to_csv(tables / "noise_structure_by_omitted_group.csv", index=False)

    diagnostic = pd.read_csv(HERE / "diagnostics/noise_structure_summary.csv")
    measures = [
        "radial_rms_m", "error_increment_vector_rms_m",
        "expected_error_increment_vector_rms_m",
        "error_increment_div_dt_vector_rms_mps",
        "expected_error_increment_div_dt_vector_rms_mps",
    ]
    rows = []
    for (noise, sigma), d in diagnostic.groupby(["noise", "sigma"], sort=False):
        assert len(d) == 5 and d.points.nunique() == d.valid_derivative_edges.nunique() == 1
        row = {
            "noise": noise, "sigma": sigma, "seed_count": len(d),
            "points": int(d.points.iloc[0]),
            "valid_derivative_edges": int(d.valid_derivative_edges.iloc[0]),
            "meaning": "Added-error vector quantities, not measured scalar speed errors.",
        }
        for name in measures:
            row[f"{name}_mean"] = d[name].mean()
            row[f"{name}_seed_sd"] = d[name].std(ddof=1)
        rows.append(row)
    pd.DataFrame(rows).to_csv(tables / "noise_increment_summary.csv", index=False)

    class_scores = pd.read_csv(tables / "per_class_f1.csv")
    class_scores = class_scores[
        class_scores.group.eq("G3") & class_scores.smoother.eq(RAW) & class_scores.sigma.isin([0, 20])
    ]
    class_summary = class_scores.groupby(["family", "sigma", "class"], sort=False).agg(
        f1_mean=("f1", "mean"), f1_seed_sd=("f1", "std"), seed_count=("seed", "size"),
    ).reset_index()
    class_summary.f1_seed_sd = class_summary.f1_seed_sd.fillna(0)
    class_summary.to_csv(tables / "G3_clean_and_20m_class_summary.csv", index=False)
    print("Saved 40 group-sensitivity rows, 6 increment rows and 20 class-F1 rows.", flush=True)
    assert len(compare) == 40 and len(rows) == 6 and len(class_summary) == 20


if __name__ == "__main__":
    main()
