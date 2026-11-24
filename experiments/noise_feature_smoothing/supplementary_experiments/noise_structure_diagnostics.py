"""Describe error increments without confusing them with scalar speed error."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_supplement import ROOT, EXP, SEEDS, TAU, ou_field, read_points


def main():
    users = sorted(pd.read_csv(EXP / "predictions/G3_sigma00_seed0_smoothed.csv",
                               dtype={"user_id": str}).user_id.unique())
    totals = []
    for seed in SEEDS:
        for uid in users:
            p = read_points(ROOT / f"data/processed/quality_features/{uid}.parquet")
            p = p.sort_values(["feature_block_id", "timestamp"], kind="stable").reset_index(drop=True)
            q = ou_field(p, seed)
            manifest = json.loads((HERE / f"feature_cache/OU30_sigma05_seed{seed}.json")
                                  .read_text(encoding="utf-8"))
            record = next(d for d in manifest["user_diagnostics"] if d["user_id"] == uid)
            assert hashlib.sha256(q.tobytes()).hexdigest() == record["standard_field_sha256"]
            dt = p.timestamp.diff().dt.total_seconds().to_numpy()[1:]
            same = p.feature_block_id.to_numpy()[1:] == p.feature_block_id.to_numpy()[:-1]
            valid = same & (dt > 0) & (dt <= 1200)
            delta_t = dt[valid]
            iid = np.random.default_rng(np.random.SeedSequence([seed, int(uid)])).standard_normal((len(p), 2))
            for kind, field in (("iid", iid), ("OU30", q)):
                increments = np.diff(field, axis=0)[valid]
                theoretical_var = np.full(len(delta_t), 2.) if kind == "iid" else 2 * (-np.expm1(-delta_t / TAU))
                totals.append({
                    "seed": seed, "user_id": uid, "noise": kind, "points": len(p),
                    "valid_derivative_edges": int(valid.sum()),
                    "standard_field_radial_squared_sum": float((field * field).sum()),
                    "standard_field_E_sum": float(field[:, 0].sum()),
                    "standard_field_N_sum": float(field[:, 1].sum()),
                    "increment_vector_squared_sum": float((increments * increments).sum()),
                    "expected_increment_vector_squared_sum": float(2 * theoretical_var.sum()),
                    "increment_per_dt_vector_squared_sum": float(((increments / delta_t[:, None]) ** 2).sum()),
                    "expected_increment_per_dt_vector_squared_sum": float((2 * theoretical_var / delta_t ** 2).sum()),
                })
        print(f"Noise-increment diagnostics seed={seed} complete", flush=True)
    data = pd.DataFrame(totals)
    data.to_csv(HERE / "diagnostics/noise_structure_per_user.csv", index=False)
    rows = []
    for (seed, kind), g in data.groupby(["seed", "noise"]):
        n = g.points.sum()
        edges = g.valid_derivative_edges.sum()
        for sigma in (5, 10, 20):
            row = {
                "seed": seed, "noise": kind, "sigma": sigma,
                "points": int(n), "valid_derivative_edges": int(edges),
                "radial_rms_m": sigma * np.sqrt(g.standard_field_radial_squared_sum.sum() / n),
                "east_mean_m": sigma * g.standard_field_E_sum.sum() / n,
                "north_mean_m": sigma * g.standard_field_N_sum.sum() / n,
                "error_increment_vector_rms_m": sigma * np.sqrt(g.increment_vector_squared_sum.sum() / edges),
                "expected_error_increment_vector_rms_m": sigma * np.sqrt(g.expected_increment_vector_squared_sum.sum() / edges),
                "error_increment_div_dt_vector_rms_mps": sigma * np.sqrt(g.increment_per_dt_vector_squared_sum.sum() / edges),
                "expected_error_increment_div_dt_vector_rms_mps": sigma * np.sqrt(g.expected_increment_per_dt_vector_squared_sum.sum() / edges),
                "warning": "Added-error vector increment per dt; NOT scalar measured speed error.",
            }
            rows.append(row)
    pd.DataFrame(rows).to_csv(HERE / "diagnostics/noise_structure_summary.csv", index=False)
    writeup = """# Noise structure diagnostic definitions

Valid edges match motion-feature time validity: same original block, 0<dt<=1200s.
No point is removed; this mask is only for error-increment diagnostics.

All 65 user/seed OU standard fields are independently regenerated and compared
with saved cache manifest hashes. Per-point iid innovations are regenerated from
the same original SeedSequence recipe. Values are not renormalised.

For a valid edge, per-axis additional-position-error increment variance is
2*sigma^2 for iid, and 2*sigma^2*(1-exp(-dt/30)) for OU30.
Vector RMS is sqrt(mean(E_increment^2 + N_increment^2)).
The per-dt variant divides each vector increment by that edge's actual seconds.
It is NOT the difference between noisy and clean scalar speeds, and its variance
does not give this project's scalar acceleration or jerk variance.

The original observed positions need not be true positions. These diagnostics
describe only the additional artificial error fields. They support an explanation
of why equal marginal position variance can produce different motion perturbation;
they do not establish a universal classification effect or a real GPS error model.
"""
    (HERE / "diagnostics/NOISE_STRUCTURE_NOTES.md").write_text(writeup, encoding="utf-8")


if __name__ == "__main__":
    main()
