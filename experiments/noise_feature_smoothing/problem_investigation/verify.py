"""Independent checks against original prediction files and source code."""
from pathlib import Path
import hashlib
import itertools
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "src/model_training"))
from models import MODES, cv_groups


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(2**20):
            result.update(chunk)
    return result.hexdigest()


def greedy_count(t, p, tol=1):
    ti = np.where(t[1:] != t[:-1])[0] + 1
    pi = np.where(p[1:] != p[:-1])[0] + 1
    i = j = hit = 0
    while i < len(ti) and j < len(pi):
        if ti[i] < pi[j] - tol:
            i += 1
        elif pi[j] < ti[i] - tol:
            j += 1
        else:
            hit += 1
            i += 1
            j += 1
    return hit


def main():
    expected = pd.read_csv(HERE / "group_class_by_seed.csv")
    boundaries = pd.read_csv(HERE / "boundary_by_seed.csv")
    summary = pd.read_csv(HERE / "kernel_by_seed.csv")
    group_checks = boundary_checks = 0
    for r in summary.itertuples():
        cond = f"sigma{r.sigma:02d}_seed{r.seed}"
        path = (EXP / f"predictions/G3_{cond}_smoothed.csv" if r.noise != "OU30" else
                EXP / f"supplementary_experiments/predictions/G3_originalgamma_OU30_{cond}.csv")
        d = pd.read_csv(path, dtype={"user_id": str})
        gids = cv_groups(d.user_id, ROOT / "data/splits/report.json").to_numpy()
        actual = expected[(expected.noise == r.noise) & (expected.sigma == r.sigma) &
                          (expected.seed == r.seed)]
        for name in actual["slice"].unique():
            keep = np.ones(len(d), bool) if name == "all" else (
                gids != "linked0" if name == "without_linked0" else gids == name)
            sub = d.loc[keep]
            count = sub[[f"cnt_{m}" for m in MODES]].to_numpy()
            true = np.tile(np.array(MODES), len(sub))
            pred = np.repeat(sub.raw.to_numpy(), 5)
            cm = confusion_matrix(true, pred, labels=MODES, sample_weight=count.ravel())
            metric = f1_score(true, pred, labels=MODES, sample_weight=count.ravel(),
                              average=None, zero_division=0)
            ref = actual[actual["slice"] == name].set_index("class").loc[MODES]
            np.testing.assert_allclose(metric, ref.f1, rtol=0, atol=1e-14)
            np.testing.assert_array_equal(cm.sum(axis=1), ref.support)
            np.testing.assert_array_equal(cm.sum(axis=0), ref.predicted)
            np.testing.assert_array_equal(np.diag(cm), ref.tp)
            group_checks += 1
        for smoother in ("S0_raw_k1", "S1_centered_soft_k11", "S2_trailing_soft_k11"):
            hit = 0
            for _, sub in d.groupby("feature_block_id", sort=False):
                sub = sub.sort_values("start_time", kind="stable")
                hit += greedy_count(sub.label.to_numpy(), sub[smoother].to_numpy())
            ref = boundaries[(boundaries.noise == r.noise) & (boundaries.sigma == r.sigma) &
                             (boundaries.seed == r.seed) & (boundaries.smoother == smoother)].iloc[0]
            assert hit == ref.primary_hits
            boundary_checks += 1
    # Verify that assignment-based diagnostic matching attains exhaustive maximum
    # cardinality on every pair of binary sequences of length five.
    sys.path.insert(0, str(HERE))
    from investigate import match_boundaries
    exhaustive_checks = 0
    for a, b in itertools.product(itertools.product("ab", repeat=5), repeat=2):
        t, p = np.array(a), np.array(b)
        assert len(match_boundaries(t, p)[2]) == greedy_count(t, p)
        exhaustive_checks += 1
    files = json.loads((HERE / "input_manifest.json").read_text())["files"]
    for path, saved in files.items():
        assert digest(ROOT / path) == saved, path
    report = {
        "passed": True, "independent_weighted_sklearn_group_confusions": group_checks,
        "independent_primary_boundary_counts": boundary_checks,
        "binary_sequence_matching_checks": exhaustive_checks,
        "protected_input_hashes": len(files),
    }
    (HERE / "independent_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
