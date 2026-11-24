"""Independent verification of follow-up outputs, without importing their runner."""
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
os.environ["MPLCONFIGDIR"] = str(HERE / "matplotlib_cache")
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.metrics.pairwise import rbf_kernel
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(ROOT / "src/model_training"))
from models import cv_groups

CLASSES = ["walk", "bike", "bus", "car", "train"]
SEEDS = (20261123, 20261124, 20261125, 20261126, 20261127)


def path_for(family, group, sigma, seed):
    name = f"sigma{sigma:02d}_seed{seed}"
    if family == "original":
        return EXP / f"predictions/{group}_{name}_smoothed.csv"
    return EXP / f"supplementary_experiments/predictions/{group}_fixedgamma_iid_{name}.csv"


def bootstrap_f1(confusions, draw_counts):
    sampled = np.tensordot(draw_counts, confusions, axes=(1, 0))
    tp = np.diagonal(sampled, axis1=1, axis2=2)
    denom = sampled.sum(1)+sampled.sum(2)
    scores = np.zeros_like(tp, dtype=float)
    np.divide(2*tp, denom, out=scores, where=denom > 0)
    return scores.mean(1)


def close(a, b, tol=2e-12):
    np.testing.assert_allclose(a, b, atol=tol, rtol=1e-12)


def main():
    reference = pd.read_csv(EXP / "feature_cache/sigma00_seed0.csv", dtype={"user_id": str})
    true_counts = reference[[f"cnt_{m}" for m in CLASSES]].to_numpy()
    group_ids = np.asarray(cv_groups(reference.user_id, ROOT / "data/splits/report.json"), str)
    with np.load(EXP / "metrics/paired_bootstrap_draws.npz", allow_pickle=True) as a:
        draws, clusters = a["draws"], a["clusters"].astype(str)
    with np.load(HERE / "diagnostics/group_confusions.npz") as a:
        saved_cms = dict(zip(a["keys"], a["cms"]))
        assert np.array_equal(a["clusters"], clusters)
    with np.load(EXP / "supplementary_experiments/diagnostics/cluster_confusions.npz") as a:
        old_cms = dict(zip(a["keys"], a["cms"]))
        assert np.array_equal(a["clusters"], clusters)
    seed_metrics = pd.read_csv(HERE / "tables/raw_metrics_by_seed.csv")
    flows = pd.read_csv(HERE / "tables/point_error_flows_by_seed.csv")
    pairs = pd.read_csv(HERE / "tables/prediction_pair_flows_by_seed.csv")
    kernel_stats = pd.read_csv(HERE / "tables/kernel_by_seed.csv")
    bucket_rows = pd.read_csv(HERE / "tables/distance_buckets_by_seed.csv")
    configs = json.loads((HERE / "model_configs.json").read_text())
    condition_checks = group_checks = point_flow_checks = score_samples = 0
    bootstrap = {}
    omitted_metrics = {}
    expected_values = {}
    for config in configs:
        family, group = config["family"], config["group"]
        est = joblib.load(ROOT / config["model"])
        cols = config["feature_columns"]
        clean_x = est.named_steps["scale"].transform(
            est.named_steps["impute"].transform(reference[cols]))
        sample = np.unique(np.linspace(0, len(reference)-1, 64, dtype=int))
        landmarks = est.named_steps["rbf"].components_
        clean_ix = np.sum((clean_x[sample, None, :]-landmarks[None, :, :])**2, axis=2).argmin(1)
        clean = pd.read_csv(path_for(family, group, 0, 0)).raw.to_numpy()
        clean_idx = pd.Categorical(clean, categories=CLASSES).codes
        for sigma in (0, 5, 10, 20):
            for seed in ([0] if sigma == 0 else SEEDS):
                info = dict(family=family, group=group, sigma=sigma, seed=seed)
                key = f"{family}|{group}|{sigma}|{seed}"
                pred = pd.read_csv(path_for(family, group, sigma, seed), dtype={"user_id": str})
                raw = pred.raw.to_numpy()
                noisy_idx = pd.Categorical(raw, categories=CLASSES).codes
                expanded_true = np.tile(CLASSES, len(pred))
                expanded_pred = np.repeat(raw, 5)
                point_f1 = f1_score(expanded_true, expanded_pred, sample_weight=true_counts.ravel(),
                                    labels=CLASSES, average="macro", zero_division=0)
                point_acc = accuracy_score(expanded_true, expanded_pred, sample_weight=true_counts.ravel())
                mask = np.logical_and.reduce([seed_metrics[k].eq(v) for k, v in info.items()])
                row = seed_metrics.loc[mask].iloc[0]
                close([point_f1, point_acc], [row.macro_f1, row.accuracy])
                expected_values[family, group, sigma, seed] = point_f1
                by_group = []
                for name in clusters:
                    keep = group_ids == name
                    cm = confusion_matrix(
                        np.tile(CLASSES, keep.sum()), np.repeat(raw[keep], 5),
                        labels=CLASSES, sample_weight=true_counts[keep].ravel())
                    by_group.append(cm)
                    group_checks += 1
                by_group = np.stack(by_group)
                np.testing.assert_array_equal(by_group, saved_cms[key])
                oldkey = f"{family}|{group}|iid|{sigma}|{seed}|S0_raw_k1"
                np.testing.assert_array_equal(by_group, old_cms[oldkey])
                bootstrap[family, group, sigma, seed] = bootstrap_f1(by_group, draws)
                for omit in ["none", *clusters]:
                    keep = np.ones(len(raw), bool) if omit == "none" else group_ids != omit
                    omitted_metrics[family, group, sigma, seed, omit] = f1_score(
                        np.tile(CLASSES, keep.sum()), np.repeat(raw[keep], 5),
                        labels=CLASSES, sample_weight=true_counts[keep].ravel(),
                        average="macro", zero_division=0)
                tensor = np.zeros((5, 5, 5), dtype=np.int64)
                for t in range(5):
                    np.add.at(tensor[t], (clean_idx, noisy_idx), true_counts[:, t])
                fr = flows.loc[np.logical_and.reduce([flows[k].eq(v) for k, v in info.items()])]
                pr = pairs.loc[np.logical_and.reduce([pairs[k].eq(v) for k, v in info.items()])]
                for i, left in enumerate(CLASSES):
                    for j, right in enumerate(CLASSES):
                        r = pr[(pr.clean_prediction == left) & (pr.noisy_prediction == right)].iloc[0]
                        np.testing.assert_array_equal(tensor[:, i, j], r[[f"true_{m}" for m in CLASSES]].to_numpy(np.int64))
                for t, mode in enumerate(CLASSES):
                    block = tensor[t]
                    diag = block[t, t]
                    rescue = block[:, t].sum()-diag
                    damage = block[t, :].sum()-diag
                    unchanged_wrong = np.trace(block)-diag
                    changed_wrong = block.sum()-diag-rescue-damage-unchanged_wrong
                    r = fr[fr.true_class == mode].iloc[0]
                    np.testing.assert_array_equal(
                        [diag, unchanged_wrong, rescue, damage, changed_wrong, rescue-damage],
                        r[["unchanged_correct", "unchanged_wrong", "rescued", "damaged",
                           "changed_still_wrong", "net_correct_change"]].to_numpy(np.int64))
                    point_flow_checks += 1
                cache = pd.read_csv(EXP / f"feature_cache/sigma{sigma:02d}_seed{seed}.csv")
                x = est.named_steps["scale"].transform(est.named_steps["impute"].transform(cache[cols]))
                # Direct kernel evaluation and explicit per-coordinate distances
                # are independent of the runner's pairwise distance routine.
                sample_kernel = rbf_kernel(x[sample], landmarks, gamma=config["gamma"])
                fixed = config["gamma"]*((x[sample]-landmarks[clean_ix])**2-
                                         (clean_x[sample]-landmarks[clean_ix])**2)
                with np.load(HERE / f"diagnostics/{family}_{group}_sigma{sigma:02d}_seed{seed}.npz") as a:
                    close(sample_kernel.max(1), a["max_kernel"][sample], tol=1e-10)
                    close(fixed, a["fixed_reference_delta"][sample], tol=2e-8)
                    all_delta = a["fixed_reference_delta"].copy()
                    all_kernel = a["max_kernel"].copy()
                    all_norm = a["score_minus_intercept_norm"].copy()
                r = kernel_stats.loc[np.logical_and.reduce([kernel_stats[k].eq(v) for k, v in info.items()])].iloc[0]
                close(np.mean(all_delta.sum(1)), r.fixed_clean_reference_distance_change_mean, tol=2e-8)
                close(np.median(all_kernel), r.max_kernel_median)
                close(np.median(all_norm), r.score_minus_intercept_norm_median)
                br = bucket_rows.loc[np.logical_and.reduce([bucket_rows[k].eq(v) for k, v in info.items()])]
                close(br.fixed_clean_reference_distance_change_mean.sum(),
                      r.fixed_clean_reference_distance_change_mean, tol=2e-8)
                scores = est.decision_function(cache.loc[sample, cols])
                order = pd.Index(est.classes_).get_indexer(CLASSES)
                close(scores[:, order], pred.loc[sample, [f"score_{m}" for m in CLASSES]], tol=2e-7)
                score_samples += len(sample)
                condition_checks += 1
        print("independent", family, group, condition_checks, flush=True)
    summary = pd.read_csv(HERE / "tables/raw_summary.csv")
    for r in summary.itertuples():
        seeds = [0] if r.sigma == 0 else SEEDS
        samples = [expected_values[r.family, r.group, r.sigma, s] for s in seeds]
        b = np.mean([bootstrap[r.family, r.group, r.sigma, s] for s in seeds], axis=0)
        clean_b = bootstrap[r.family, r.group, 0, 0]
        close(np.mean(samples), r.macro_f1_mean)
        close(np.std(samples, ddof=1) if r.sigma else 0, r.macro_f1_seed_sd)
        close(np.quantile(b, [.025, .975]), [r.macro_f1_ci_low, r.macro_f1_ci_high])
        close(expected_values[r.family, r.group, 0, 0]-np.mean(samples), r.clean_minus_noisy_loss)
        close(np.quantile(clean_b-b, [.025, .975]), [r.loss_ci_low, r.loss_ci_high])
    contrasts = pd.read_csv(HERE / "tables/paired_feature_contrasts.csv")
    for r in contrasts.itertuples():
        seeds = [0] if r.sigma == 0 else SEEDS
        lf = "original" if r.recipe == "original" or r.left == "G3" else "fixed_gamma"
        rf = "original" if r.recipe == "original" or r.right == "G3" else "fixed_gamma"
        gap = np.mean([expected_values[lf, r.left, r.sigma, s]-
                       expected_values[rf, r.right, r.sigma, s] for s in seeds])
        distribution = np.mean([bootstrap[lf, r.left, r.sigma, s]-
                                bootstrap[rf, r.right, r.sigma, s] for s in seeds], axis=0)
        clean_gap = expected_values[lf, r.left, 0, 0]-expected_values[rf, r.right, 0, 0]
        clean_distribution = bootstrap[lf, r.left, 0, 0]-bootstrap[rf, r.right, 0, 0]
        close([gap, gap-clean_gap], [r.gap_mean, r.noisy_gap_minus_clean_gap])
        close(np.quantile(distribution, [.025, .975]), [r.gap_ci_low, r.gap_ci_high])
        close(np.quantile(distribution-clean_distribution, [.025, .975]),
              [r.gap_change_ci_low, r.gap_change_ci_high])
    omissions = pd.read_csv(HERE / "tables/omitted_group_contrasts.csv")
    for r in omissions.itertuples():
        seeds = [0] if r.sigma == 0 else SEEDS
        lf = "original" if r.recipe == "original" or r.left == "G3" else "fixed_gamma"
        rf = "original" if r.recipe == "original" or r.right == "G3" else "fixed_gamma"
        expected = np.mean([omitted_metrics[lf, r.left, r.sigma, s, r.omitted_group]-
                            omitted_metrics[rf, r.right, r.sigma, s, r.omitted_group] for s in seeds])
        close(expected, r.gap_mean)
    protected = json.loads((HERE / "input_manifest.json").read_text())["files"]
    for rel, expected in protected.items():
        h = hashlib.sha256()
        with (ROOT / rel).open("rb") as f:
            for block in iter(lambda: f.read(2**20), b""):
                h.update(block)
        assert h.hexdigest() == expected, rel
    result = {
        "passed": True, "conditions": condition_checks, "independent_group_confusions": group_checks,
        "per_class_error_flows": point_flow_checks, "pipeline_score_samples": score_samples,
        "bootstrap_summary_rows": len(summary), "paired_contrast_rows": len(contrasts),
        "group_omission_rows": len(omissions), "protected_file_hashes": len(protected),
        "distance_decomposition_samples": score_samples,
        "manual_visual_review": "not_completed_image_tool_unavailable",
    }
    (HERE / "independent_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
