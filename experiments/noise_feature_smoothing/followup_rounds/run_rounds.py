"""Three finite, read-only scientific follow-ups to frozen experiment artifacts."""
from __future__ import annotations

import hashlib
import argparse
import importlib.metadata
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
os.environ["MPLCONFIGDIR"] = str(HERE / "matplotlib_cache")
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(ROOT / "src/model_training"))
from models import MODES, cv_groups

SEEDS = (20261123, 20261124, 20261125, 20261126, 20261127)
SCORE = [f"score_{m}" for m in MODES]
META = ["user_id", "feature_block_id", "start_time", "end_time", "n_points",
        "label", "purity", *[f"cnt_{m}" for m in MODES]]
BUCKETS = {
    "speed": ["speed_mean", "speed_std", "speed_min", "speed_max", "speed_p10",
              "speed_p50", "speed_p90", "speed_p95", "stop_rate", "slow_rate"],
    "acceleration": ["acc_mean", "acc_std", "acc_p90", "acc_max", "big_acc_rate"],
    "jerk_turn": ["jerk_mean", "jerk_p90", "hcr_mean", "hcr_p50", "hcr_p90",
                  "heading_change_mean", "sharp_turn_rate"],
    "geometry_coverage": ["path_length_m", "valid_speed_frac", "duration_s",
                          "straightness", "avg_speed_overall"],
}
START = time.monotonic()
STATE = {"completed": [], "retries": {}}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(name, obj):
    path = HERE / name
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")
    temp.replace(path)


def save_csv(name, rows):
    d = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    d.to_csv(HERE / "tables" / f"{name}.csv", index=False)
    return d


def inventory():
    files = [p for folder in ("data", "src", "tests", "docs", "experiments")
             for p in (ROOT / folder).rglob("*") if p.is_file()
             and HERE not in p.parents and "__pycache__" not in p.parts
             and "matplotlib_cache" not in p.parts and p.suffix != ".pyc"
             and p != ROOT / "docs/aigc_log_part2.md"]
    files += [ROOT / n for n in ("README.md", "CLAUDE.md", "pyproject.toml", "uv.lock")]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}


def checkpoint(current, **extra):
    STATE.update(current=current, updated_at=pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
                 elapsed_seconds=time.monotonic()-START, **extra)
    write_json("checkpoint.json", STATE)
    print(json.dumps(STATE, ensure_ascii=False), flush=True)


def read_table(path):
    d = pd.read_csv(path, dtype={"user_id": str})
    for col in ("start_time", "end_time"):
        d[col] = pd.to_datetime(d[col])
    return d


def models():
    for family, groups in (("original", ("G1", "G2", "G3")),
                           ("fixed_gamma", ("G1", "G2"))):
        for group in groups:
            if family == "original":
                mp = EXP / "models" / ("full.joblib" if group == "G3" else f"{group}.joblib")
                cp = EXP / "models" / f"{group}_config.json"
            else:
                mp = EXP / "supplementary_experiments/models" / f"{group}_gamma1over27.joblib"
                cp = mp.with_suffix(".json")
            yield family, group, mp, json.loads(cp.read_text())["feature_columns"]


def conditions():
    yield 0, 0
    for sigma in (5, 10, 20):
        for seed in SEEDS:
            yield sigma, seed


def prediction_path(family, group, sigma, seed):
    name = f"sigma{sigma:02d}_seed{seed}"
    if family == "original":
        return EXP / f"predictions/{group}_{name}_smoothed.csv"
    return EXP / f"supplementary_experiments/predictions/{group}_fixedgamma_iid_{name}.csv"


def metrics(cms):
    cm = np.asarray(cms)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    den = cm.sum(-1) + cm.sum(-2)
    f1 = np.divide(2*tp, den, out=np.zeros(tp.shape, float), where=den != 0)
    return {"macro_f1": f1.mean(-1),
            "accuracy": tp.sum(-1) / cm.sum(axis=(-2, -1))}


def point_cm(counts, pred):
    idx = pd.Index(MODES).get_indexer(pred)
    assert (idx >= 0).all()
    cm = np.zeros((5, 5), np.int64)
    for j in range(5):
        cm[:, j] = counts[idx == j].sum(0)
    return cm


def round1():
    baseline = read_table(EXP / "feature_cache/sigma00_seed0.csv")
    caches = {(s, k): read_table(EXP / f"feature_cache/sigma{s:02d}_seed{k}.csv")
              for s, k in conditions()}
    rows, feature_rows, checks, configs = [], [], [], []
    for family, group, path, cols in models():
        est = joblib.load(path)
        imp, scale, rbf, clf = [est.named_steps[n] for n in ("impute", "scale", "rbf", "clf")]
        assert est.n_features_in_ == len(cols)
        assert rbf.gamma == (1/len(cols) if family == "original" else 1/27)
        assert rbf.n_components == 1000 and rbf.random_state == 42
        assert clf.C == 1 and clf.class_weight == "balanced" and clf.random_state == 42
        assert len(set(sum(BUCKETS.values(), []))) == 27
        assert set(cols) <= set(sum(BUCKETS.values(), []))
        clean_z = scale.transform(imp.transform(baseline[cols]))
        weight = rbf.normalization_.T @ clf.coef_.T
        align = pd.Index(est.classes_).get_indexer(MODES)
        clean_nearest = None
        configs.append({"family": family, "group": group, "model": str(path.relative_to(ROOT)),
                        "feature_columns": cols, "gamma": rbf.gamma,
                        "intercept_class": est.classes_[clf.intercept_.argmax()],
                        "intercept": dict(zip(est.classes_, clf.intercept_.tolist()))})
        for sigma, seed in conditions():
            d = caches[sigma, seed]
            p = read_table(prediction_path(family, group, sigma, seed))
            pd.testing.assert_frame_equal(d[META], baseline[META], check_dtype=False)
            pd.testing.assert_frame_equal(d[META], p[META], check_dtype=False)
            x = scale.transform(imp.transform(d[cols]))
            n = len(x)
            nearest, maxk, norm = np.empty(n, int), np.empty(n), np.empty(n)
            squared = np.empty(n)
            reconstructed = np.empty((n, 5))
            for start in range(0, n, 1024):
                stop = min(start+1024, n)
                dist = euclidean_distances(x[start:stop], rbf.components_, squared=True)
                nearest[start:stop] = dist.argmin(1)
                squared[start:stop] = dist.min(1)
                k = np.exp(-rbf.gamma * dist)
                maxk[start:stop] = k.max(1)
                part = k @ weight
                norm[start:stop] = np.linalg.norm(part, axis=1)
                reconstructed[start:stop] = (part + clf.intercept_)[:, align]
            if sigma == 0:
                clean_nearest = nearest.copy()
            pred = np.array(MODES)[reconstructed.argmax(1)]
            np.testing.assert_array_equal(pred, p.raw)
            np.testing.assert_allclose(reconstructed, p[SCORE], atol=2e-7, rtol=1e-7)
            ix = np.unique(np.linspace(0, n-1, 64, dtype=int))
            np.testing.assert_allclose(est.decision_function(d.loc[ix, cols])[:, align],
                                       reconstructed[ix], atol=2e-7, rtol=1e-7)
            assert n == 17380 and d.n_points.sum() == 860972
            fixed_ref = rbf.components_[clean_nearest]
            delta = rbf.gamma * ((x-fixed_ref)**2 - (clean_z-fixed_ref)**2)
            moving = rbf.gamma * (x-rbf.components_[nearest])**2
            np.testing.assert_allclose(moving.sum(1), rbf.gamma*squared, atol=2e-9, rtol=1e-9)
            ref_change = rbf.gamma * (
                np.sum((x-fixed_ref)**2, axis=1)-np.sum((clean_z-fixed_ref)**2, axis=1))
            np.testing.assert_allclose(delta.sum(1), ref_change, atol=2e-9, rtol=1e-9)
            block_sum = np.zeros(n)
            info = dict(family=family, group=group, sigma=sigma, seed=seed)
            for bucket, names in BUCKETS.items():
                js = [cols.index(c) for c in names if c in cols]
                if not js:
                    continue
                block_sum += delta[:, js].sum(1)
                feature_rows.append({
                    **info, "bucket": bucket, "dimensions": len(js),
                    "fixed_clean_reference_distance_change_mean": delta[:, js].sum(1).mean(),
                    "fixed_clean_reference_distance_change_median": np.median(delta[:, js].sum(1)),
                    "moving_nearest_distance_mean": moving[:, js].sum(1).mean(),
                })
            np.testing.assert_allclose(block_sum, ref_change, atol=2e-9, rtol=1e-9)
            total = int(d.n_points.sum())
            intercept_class = est.classes_[clf.intercept_.argmax()]
            rows.append({
                **info, "gamma": rbf.gamma, "windows": n, "points": total,
                "max_kernel_median": np.median(maxk),
                "tiny_kernel_point_fraction": d.loc[maxk < 1e-6, "n_points"].sum()/total,
                "score_minus_intercept_norm_median": np.median(norm),
                "intercept_class": intercept_class,
                "intercept_class_point_fraction": d.loc[pred == intercept_class, "n_points"].sum()/total,
                "predicted_train_point_fraction": d.loc[pred == "train", "n_points"].sum()/total,
                "fixed_clean_reference_distance_change_mean": ref_change.mean(),
                "fixed_clean_reference_distance_change_median": np.median(ref_change),
                "nearest_reference_changed_window_fraction": np.mean(nearest != clean_nearest),
            })
            checks.append({
                **info, "all_saved_scores_max_abs_error": np.abs(
                    reconstructed-p[SCORE].to_numpy()).max(),
                "pipeline_sample_windows": len(ix),
                "squared_distance_max_abs_error": np.abs(moving.sum(1)-rbf.gamma*squared).max(),
                "additive_decomposition_max_abs_error": np.abs(block_sum-ref_change).max(),
            })
            np.savez_compressed(
                HERE / "diagnostics" / f"{family}_{group}_sigma{sigma:02d}_seed{seed}.npz",
                nearest=nearest, max_kernel=maxk, fixed_reference_delta=delta,
                score_minus_intercept_norm=norm)
            print("round1", info, flush=True)
    save_csv("kernel_by_seed", rows)
    save_csv("distance_buckets_by_seed", feature_rows)
    save_csv("round1_numeric_checks", checks)
    write_json("model_configs.json", configs)
    assert len(rows) == 80 and len(checks) == 80
    write_json("round1_acceptance.json", {"passed": True, "model_conditions": 80,
               "actual_pipeline_sample_windows": sum(x["pipeline_sample_windows"] for x in checks),
               "maximum_saved_score_error": max(x["all_saved_scores_max_abs_error"] for x in checks)})


def round2():
    with np.load(EXP / "metrics/paired_bootstrap_draws.npz", allow_pickle=True) as a:
        draws, clusters = a["draws"], a["clusters"].astype(str)
    assert draws.shape == (5000, 9) and (draws.sum(1) == 9).all()
    b = read_table(EXP / "feature_cache/sigma00_seed0.csv")
    gids = np.asarray(cv_groups(b.user_id, ROOT / "data/splits/report.json"), str)
    counts = b[[f"cnt_{m}" for m in MODES]].to_numpy(np.int64)
    keys, cms, seed_rows, error_rows, flow_rows = [], [], [], [], []
    values, distributions = {}, {}
    for family, group, _, _ in models():
        clean = read_table(prediction_path(family, group, 0, 0)).raw.to_numpy()
        ci = pd.Index(MODES).get_indexer(clean)
        for sigma, seed in conditions():
            p = read_table(prediction_path(family, group, sigma, seed))
            pd.testing.assert_frame_equal(p[META], b[META], check_dtype=False)
            raw = p.raw.to_numpy()
            ni = pd.Index(MODES).get_indexer(raw)
            cm = np.stack([point_cm(counts[gids == g], raw[gids == g]) for g in clusters])
            keys.append(f"{family}|{group}|{sigma}|{seed}")
            cms.append(cm)
            info = dict(family=family, group=group, sigma=sigma, seed=seed)
            result = metrics(cm.sum(0))
            seed_rows.append({**info, **result})
            values[family, group, sigma, seed] = result
            distributions[family, group, sigma, seed] = metrics(np.einsum("bg,gij->bij", draws, cm))
            changed = ci != ni
            for j, mode in enumerate(MODES):
                support = counts[:, j]
                unchanged_correct = int(support[(~changed) & (ci == j)].sum())
                unchanged_wrong = int(support[(~changed) & (ci != j)].sum())
                rescued = int(support[changed & (ni == j)].sum())
                damaged = int(support[changed & (ci == j)].sum())
                still_wrong = int(support[changed & (ci != j) & (ni != j)].sum())
                assert sum((unchanged_correct, unchanged_wrong, rescued, damaged, still_wrong)) == support.sum()
                error_rows.append({
                    **info, "true_class": mode, "support": int(support.sum()),
                    "unchanged_correct": unchanged_correct, "unchanged_wrong": unchanged_wrong,
                    "rescued": rescued, "damaged": damaged, "changed_still_wrong": still_wrong,
                    "net_correct_change": rescued-damaged,
                })
            for i, left in enumerate(MODES):
                for j, right in enumerate(MODES):
                    flow_rows.append({**info, "clean_prediction": left, "noisy_prediction": right,
                        **dict(zip([f"true_{m}" for m in MODES], counts[(ci == i) & (ni == j)].sum(0)))})
    np.savez_compressed(HERE / "diagnostics/group_confusions.npz",
                        keys=np.array(keys), clusters=clusters, cms=np.stack(cms))
    save_csv("raw_metrics_by_seed", seed_rows)
    save_csv("point_error_flows_by_seed", error_rows)
    save_csv("prediction_pair_flows_by_seed", flow_rows)
    summary, contrasts, omissions = [], [], []
    for family, group, _, _ in models():
        cleanval = values[family, group, 0, 0]["macro_f1"]
        cleandist = distributions[family, group, 0, 0]["macro_f1"]
        for sigma in (0, 5, 10, 20):
            seeds = [0] if sigma == 0 else SEEDS
            f1s = [values[family, group, sigma, s]["macro_f1"] for s in seeds]
            accs = [values[family, group, sigma, s]["accuracy"] for s in seeds]
            dist = np.mean([distributions[family, group, sigma, s]["macro_f1"] for s in seeds], axis=0)
            lower, upper = np.quantile(dist, [.025, .975])
            loss_low, loss_high = np.quantile(cleandist-dist, [.025, .975])
            summary.append({
                "family": family, "group": group, "sigma": sigma, "seed_count": len(seeds),
                "macro_f1_mean": np.mean(f1s),
                "macro_f1_seed_sd": np.std(f1s, ddof=1) if sigma else 0,
                "accuracy_mean": np.mean(accs), "macro_f1_ci_low": lower, "macro_f1_ci_high": upper,
                "clean_minus_noisy_loss": cleanval-np.mean(f1s),
                "loss_ci_low": loss_low, "loss_ci_high": loss_high,
            })
    for family in ("original", "common_gamma"):
        def key(group, sigma, seed):
            return ("original" if group == "G3" or family == "original" else "fixed_gamma",
                    group, sigma, seed)
        for left, right in (("G3", "G2"), ("G3", "G1"), ("G2", "G1")):
            clean_gap = values[key(left, 0, 0)]["macro_f1"]-values[key(right, 0, 0)]["macro_f1"]
            clean_gap_dist = distributions[key(left, 0, 0)]["macro_f1"]-distributions[key(right, 0, 0)]["macro_f1"]
            for sigma in (0, 5, 10, 20):
                seeds = [0] if sigma == 0 else SEEDS
                gaps = [values[key(left, sigma, s)]["macro_f1"]-values[key(right, sigma, s)]["macro_f1"]
                        for s in seeds]
                dist = np.mean([distributions[key(left, sigma, s)]["macro_f1"]-
                                distributions[key(right, sigma, s)]["macro_f1"] for s in seeds], axis=0)
                lo, hi = np.quantile(dist, [.025, .975])
                dlo, dhi = np.quantile(dist-clean_gap_dist, [.025, .975])
                contrasts.append({
                    "recipe": family, "left": left, "right": right, "sigma": sigma,
                    "gap_mean": np.mean(gaps), "gap_ci_low": lo, "gap_ci_high": hi,
                    "noisy_gap_minus_clean_gap": np.mean(gaps)-clean_gap,
                    "gap_change_ci_low": dlo, "gap_change_ci_high": dhi,
                })
                for omitted in ["none", *clusters]:
                    mask = np.ones(9, bool) if omitted == "none" else clusters != omitted
                    left_cms = [cms[keys.index("|".join(map(str, key(left, sigma, s))))][mask].sum(0) for s in seeds]
                    right_cms = [cms[keys.index("|".join(map(str, key(right, sigma, s))))][mask].sum(0) for s in seeds]
                    gap = np.mean([metrics(a)["macro_f1"]-metrics(b)["macro_f1"]
                                   for a, b in zip(left_cms, right_cms)])
                    omissions.append({"recipe": family, "left": left, "right": right,
                                      "sigma": sigma, "omitted_group": omitted, "gap_mean": gap})
    save_csv("raw_summary", summary)
    save_csv("paired_feature_contrasts", contrasts)
    save_csv("omitted_group_contrasts", omissions)
    for filename, groupcols in [
        ("point_error_flows", ["family", "group", "sigma", "true_class"]),
        ("distance_buckets", ["family", "group", "sigma", "bucket"]),
        ("kernel", ["family", "group", "sigma"]),
    ]:
        d = pd.read_csv(HERE / f"tables/{filename}_by_seed.csv")
        result = d.drop(columns="seed").groupby(groupcols).mean(numeric_only=True).reset_index()
        save_csv(filename+"_summary", result)
    assert len(summary) == 20 and len(contrasts) == 24 and len(omissions) == 240
    write_json("round2_acceptance.json", {
        "passed": True, "confusion_conditions": len(cms), "error_flow_rows": len(error_rows),
        "prediction_pair_flow_rows": len(flow_rows), "summary_rows": len(summary),
        "paired_feature_contrasts": len(contrasts), "group_omission_contrasts": len(omissions),
        "bootstrap_draws_reused": 5000, "new_bootstrap_draws": 0,
    })


def figure_save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    labels = []
    for ax in fig.axes:
        labels.extend([ax.title, ax.xaxis.label, ax.yaxis.label])
        labels.extend(t for t in ax.get_xticklabels()
                      if min(ax.get_xlim()) <= t.get_position()[0] <= max(ax.get_xlim()))
        labels.extend(t for t in ax.get_yticklabels()
                      if min(ax.get_ylim()) <= t.get_position()[1] <= max(ax.get_ylim()))
        if ax.get_legend():
            labels.append(ax.get_legend())
    for artist in labels:
        if not artist.get_visible() or (hasattr(artist, "get_text") and not artist.get_text()):
            continue
        box = artist.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= canvas.x1+1 and box.y1 <= canvas.y1+1, (
            name, str(artist), box.bounds, canvas.bounds)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"figures/{name}.{ext}", dpi=160)
    arr = plt.imread(HERE / f"figures/{name}.png")
    assert np.std(arr[..., :3]) > .05
    plt.close(fig)
    return {"figure": name, "layout_and_nonblank": True, "shape": list(arr.shape),
            "manual_visual_review": "not_completed_image_tool_unavailable"}


def round3():
    summary = pd.read_csv(HERE / "tables/raw_summary.csv")
    kernel = pd.read_csv(HERE / "tables/kernel_summary.csv")
    bucket = pd.read_csv(HERE / "tables/distance_buckets_summary.csv")
    colors = {"G1": "#287D8E", "G2": "#C65F32", "G3": "#64703A"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figures = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, recipe in zip(axes, ("original", "common_gamma")):
        for group in ("G1", "G2", "G3"):
            family = "original" if recipe == "original" or group == "G3" else "fixed_gamma"
            d = summary[(summary.family == family) & (summary.group == group)].sort_values("sigma")
            ax.errorbar(d.sigma, d.macro_f1_mean, yerr=d.macro_f1_seed_sd, marker="o",
                        color=colors[group], label=group, capsize=4)
        ax.set(title="Original gamma = 1 / dimension" if recipe == "original" else "Common gamma = 1 / 27",
               xlabel="Per-axis iid position noise sigma (m)", ylabel="Point Macro-F1",
               xticks=[0, 5, 10, 20], ylim=(0, .9))
        ax.legend()
        ax.grid(axis="y", alpha=.2)
    figures.append(figure_save(fig, "figure1_feature_recipe"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for family, group, _, _ in models():
        d = kernel[(kernel.family == family) & (kernel.group == group)].sort_values("sigma")
        style = "-" if family == "original" else "--"
        label = group + (" common gamma" if family != "original" else "")
        axes[0].plot(d.sigma, d.max_kernel_median, style, marker="o", color=colors[group], label=label)
        axes[1].plot(d.sigma, d.predicted_train_point_fraction, style, marker="o", color=colors[group], label=label)
    for ax in axes:
        ax.set(xticks=[0, 5, 10, 20], xlabel="Per-axis iid position noise sigma (m)", ylim=(0, 1.02))
        ax.grid(axis="y", alpha=.2)
    axes[0].set(ylabel="Median nearest landmark RBF similarity")
    axes[1].set(ylabel="Fraction of points predicted as train")
    axes[0].legend(fontsize=8)
    figures.append(figure_save(fig, "figure2_kernel_and_predictions"))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), layout="constrained")
    d = bucket[(bucket.family == "original") & (bucket.group == "G3")]
    for name, color in zip(BUCKETS, ("#287D8E", "#C65F32", "#856BA6", "#64703A")):
        sub = d[d.bucket == name].sort_values("sigma")
        axes[0].plot(sub.sigma, sub.fixed_clean_reference_distance_change_mean,
                     marker="o", color=color, label=name.replace("_", " "))
    axes[0].set(xlabel="Per-axis iid position noise sigma (m)", xticks=[0, 5, 10, 20],
                ylabel="Mean change in gamma * squared distance", title="G3: fixed clean landmark reference")
    axes[0].legend(fontsize=8)
    flows = pd.read_csv(HERE / "tables/point_error_flows_summary.csv")
    d = flows[(flows.family == "original") & (flows.group == "G3") & (flows.sigma == 20)].set_index("true_class").loc[MODES]
    positions = np.arange(5)
    axes[1].bar(positions-.18, d.damaged/d.support, .36, color="#B84342", label="Correct to wrong")
    axes[1].bar(positions+.18, d.rescued/d.support, .36, color="#287D8E", label="Wrong to correct")
    axes[1].set(xticks=positions, xticklabels=MODES, ylabel="Fraction of each true class",
                title="G3: iid20m point error flows", ylim=(0, 1))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
    figures.append(figure_save(fig, "figure3_distance_and_error_flows"))
    write_json("figure_checks.json", figures)
    write_json("round3_acceptance.json", {"passed": True, "figures": 3, "formats": ["png", "pdf"],
               "manual_visual_review": "not_completed_image_tool_unavailable"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-round", type=int, choices=(1, 2, 3), default=1)
    args = parser.parse_args()
    for name in ("tables", "diagnostics", "figures"):
        (HERE / name).mkdir(exist_ok=True)
    before = inventory()
    if not (HERE / "input_manifest.json").exists():
        write_json("input_manifest.json", {
            "created_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(), "files": before,
            "spec_sha256": sha(HERE / "ANALYSIS_SPEC.md"), "runner_sha256": sha(__file__),
            "python": sys.version, "dependencies": {k: importlib.metadata.version(k)
                for k in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib")},
        })
    else:
        saved = json.loads((HERE / "input_manifest.json").read_text(encoding="utf-8"))
        assert saved["files"] == before
        assert saved["spec_sha256"] == sha(HERE / "ANALYSIS_SPEC.md")
    for index in range(1, args.start_round):
        assert json.loads((HERE / f"round{index}_acceptance.json").read_text())["passed"]
        STATE["completed"].append(f"round{index}")
    with threadpool_limits(limits=2):
        rounds = (("round1", round1), ("round2", round2), ("round3", round3))
        for name, run in rounds[args.start_round-1:]:
            for attempt in range(4):
                checkpoint(name, attempt=attempt, status="running")
                try:
                    run()
                    STATE["completed"].append(name)
                    STATE["retries"][name] = attempt
                    checkpoint(name, status="accepted")
                    break
                except Exception:
                    (HERE / f"{name}_attempt{attempt}_failure.log").write_text(traceback.format_exc(), encoding="utf-8")
                    checkpoint(name, status="failed")
                    # A failure requires diagnosis before any retry.
                    raise
    after = inventory()
    assert before == after, "Protected file content or membership changed"
    write_json("acceptance.json", {
        "passed": True, "rounds_completed": STATE["completed"], "protected_files": len(before),
        "protected_content_and_membership_unchanged": True,
        "elapsed_seconds": time.monotonic()-START,
        "completed_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "no_training_noise_or_new_bootstrap": True,
    })
    checkpoint("all_rounds", status="complete")


if __name__ == "__main__":
    main()
