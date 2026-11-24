"""Publish existing frozen results with explicit units and traceable comparisons."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(HERE / "matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from PIL import Image
import numpy as np
import pandas as pd

MODES = ["walk", "bike", "bus", "car", "train"]
SMOOTHERS = ["S0_raw_k1", "S1_centered_soft_k11", "S2_trailing_soft_k11"]
LABELS = ["Raw", "Centered k=11", "Trailing k=11"]
COLORS = ["#2676ad", "#cc573b", "#278267"]
MODE_COLORS = ["#2686b4", "#44a16b", "#d09c2d", "#be667b", "#8169a5"]
KEYS = ["group", "smoother", "sigma"]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False),
                   encoding="utf-8")
    tmp.replace(path)


def protected_files():
    files = []
    for folder in ("src", "tests", "data", "experiments"):
        files.extend(p for p in (ROOT / folder).rglob("*")
                     if p.is_file() and HERE not in p.parents
                     and "__pycache__" not in p.parts and p.suffix != ".pyc"
                     and "matplotlib_cache" not in p.parts)
    files.extend(ROOT / name for name in ("CLAUDE.md", "README.md", "pyproject.toml", "uv.lock"))
    return sorted(files)


def unique_join(left, right, keys):
    if left.duplicated(keys).any() or right.duplicated(keys).any():
        raise ValueError(f"Duplicate reporting keys: {keys}")
    result = left.merge(right, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not result["_merge"].eq("both").all():
        raise ValueError("Missing reporting condition")
    return result.drop(columns="_merge")


def summarize(rows):
    if rows.duplicated(["group", "condition", "smoother"]).any():
        raise ValueError("Duplicate experimental condition")
    metrics = ["macro_f1", "accuracy", "transition_precision", "transition_recall",
               "transition_f1", "wrong_isolated_flips_per100_windows",
               "prediction_change_from_clean_point_fraction",
               *[f"f1_{m}" for m in MODES]]
    grouped = rows.groupby(KEYS, sort=True)
    result = grouped[metrics].agg(["mean", "std"])
    result.columns = [f"{name}_seed_{'sd' if stat == 'std' else stat}"
                      for name, stat in result.columns]
    result = result.reset_index()
    counts = grouped.size().rename("seed_count").reset_index()
    result = unique_join(result, counts, KEYS)
    for name in metrics:
        result[f"{name}_seed_sd"] = result[f"{name}_seed_sd"].fillna(0)
    expected = np.where(result.sigma.eq(0), 1, 5)
    if not np.array_equal(result.seed_count, expected):
        raise ValueError("Expected one clean condition or five perturbation seeds")
    result["seed_sd_interpretation"] = np.where(
        result.sigma.eq(0), "not_applicable_one_clean_result_stored_as_0",
        "sample_sd_of_5_perturbations_not_user_confidence_interval")
    return result


def boundary_support(windows):
    """All eligible +/-1-window positions, not only successfully matched events."""
    rows = []
    for (user, block), part in windows.groupby(["user_id", "feature_block_id"], sort=False):
        part = part.sort_values("start_time", kind="stable")
        starts = pd.to_datetime(part.start_time)
        if starts.isna().any() or not starts.is_monotonic_increasing:
            raise ValueError("Invalid or unsorted window timestamps")
        labels = part.label.to_numpy()
        times = (starts - starts.iloc[0]).dt.total_seconds().to_numpy()
        for i in np.flatnonzero(labels[1:] != labels[:-1]) + 1:
            lo, hi = max(1, i - 1), min(len(part) - 1, i + 1)
            rows.append({
                "user_id": str(user), "feature_block_id": block,
                "boundary_window_row": int(part.index[i]),
                "boundary_time": str(starts.iloc[i]),
                "from_mode": labels[i - 1], "to_mode": labels[i],
                "eligible_boundary_positions": hi - lo + 1,
                "early_tolerance_s": float(times[i] - times[lo]),
                "late_tolerance_s": float(times[hi] - times[i]),
                "full_candidate_span_s": float(times[hi] - times[lo]),
                "interpretation": "timestamp_support_of_frozen_window_tolerance_not_a_new_metric",
            })
    return pd.DataFrame(rows)


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(map(str, r)) + " |" for r in rows]])


def visible_texts(fig):
    result = list(fig.texts)
    for ax in fig.axes:
        result.extend([ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts])
        legend = ax.get_legend()
        if legend:
            result.extend(legend.get_texts())
        for axis in (ax.xaxis, ax.yaxis):
            low, high = sorted(axis.get_view_interval())
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                if low <= tick.get_loc() <= high:
                    result.extend([tick.label1, tick.label2])
    for legend in fig.legends:
        result.extend(legend.get_texts())
    return [t for t in result if t.get_visible() and t.get_text().strip()]


def save_figure(fig, name, qa):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    texts = visible_texts(fig)
    clipped = []
    for text in texts:
        box = text.get_window_extent(renderer)
        if (box.x0 < canvas.x0 - 2 or box.y0 < canvas.y0 - 2
                or box.x1 > canvas.x1 + 2 or box.y1 > canvas.y1 + 2):
            clipped.append(text.get_text())
    titles = [ax.title.get_window_extent(renderer) for ax in fig.axes if ax.title.get_text()]
    overlaps = [(i, j) for i, a in enumerate(titles) for j, b in enumerate(titles)
                if i < j and a.overlaps(b)]
    if clipped or overlaps:
        raise AssertionError({"figure": name, "outside_canvas": clipped, "title_overlap": overlaps})
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"figures/{name}.{ext}", dpi=180, facecolor="white")
    qa.append({"figure": name, "visible_text_count": len(texts),
               "outside_canvas": clipped, "subplot_title_overlaps": overlaps,
               "scope": "canvas_bounds_and_subplot_titles_not_full_visual_review"})
    plt.close(fig)


def make_figures(table, windows):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlepad": 12, "axes.grid": True, "grid.alpha": .18})
    qa = []
    raw = table[table.smoother.eq(SMOOTHERS[0])]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    for color, group in zip(COLORS, ["G1", "G2", "G3"]):
        part = raw[raw.group.eq(group)].sort_values("sigma")
        x = part.sigma.to_numpy()
        axes[0].errorbar(x, part.macro_f1_seed_mean, yerr=part.macro_f1_seed_sd,
                         color=color, marker="o", capsize=4, label=group)
        axes[1].plot(x, part.macro_f1_delta_from_clean, color=color, marker="o")
        for offset, sigma in enumerate([0, 20]):
            r = part[part.sigma.eq(sigma)].iloc[0]
            pos = (int(group[1]) - 1) * 3 + offset
            axes[2].vlines(pos, r.macro_f1_ci_low, r.macro_f1_ci_high,
                           color=color, linewidth=2)
            axes[2].scatter(pos, r.macro_f1_seed_mean, color=color, marker="o", s=45)
    for ax in axes[:2]:
        ax.set_xlabel("Added per-axis sigma (m)")
        ax.set_xticks([0, 5, 10, 20])
    axes[0].set(title="A. Absolute performance", ylabel="Point Macro-F1", ylim=(0, .9))
    axes[0].legend(frameon=False)
    axes[1].set(title="B. Change from own clean", ylabel="Noisy minus clean Macro-F1")
    axes[1].axhline(0, color="#777777", linewidth=.8)
    axes[2].set(title="C. User-group uncertainty",
                ylabel="Point Macro-F1 and 95% interval", ylim=(0, .9))
    axes[2].set_xticks([0, 1, 3, 4, 6, 7],
                       ["G1\n0m", "G1\n20m", "G2\n0m", "G2\n20m", "G3\n0m", "G3\n20m"])
    fig.subplots_adjust(left=.06, right=.98, top=.87, bottom=.24, wspace=.35)
    fig.suptitle("Frozen clean-trained recipes under added iid position noise", fontsize=14, y=.97)
    fig.text(.06, .08, "A: error bars = perturbation seed SD (5 seeds for added noise; clean n=1). "
             "C: intervals = 5,000 paired draws of 9 linked user groups.\n"
             "Each draw scores seeds separately, then averages. Feature sets and gamma=1/d both vary. "
             "Intervals are not obtained from seed SD.", fontsize=9)
    save_figure(fig, "figure1_performance_and_uncertainty", qa)

    g3 = table[table.group.eq("G3")]
    panels = [("macro_f1", "Point Macro-F1", (0, .95)),
              ("transition_precision", "Boundary precision", (0, 1)),
              ("transition_recall", "Boundary recall", (0, 1)),
              ("transition_f1", "Boundary F1", (0, .45)),
              ("wrong_isolated_flips_per100_windows", "Wrong isolated flips / 100 windows", (0, 4.4)),
              ("most_predicted_point_fraction", "Largest predicted class point fraction", (0, 1))]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (metric, title, limits) in zip(axes.flat, panels):
        for i, (smoother, color, label) in enumerate(zip(SMOOTHERS, COLORS, LABELS)):
            part = g3[g3.smoother.eq(smoother)].set_index("sigma").loc[[0, 20]]
            ax.bar(np.arange(2) + (i - 1) * .24, part[f"{metric}_seed_mean"],
                   width=.22, color=color, label=label,
                   yerr=part[f"{metric}_seed_sd"], capsize=3)
        ax.set(title=title, ylim=limits)
        ax.set_xticks([0, 1], ["No added noise", "20m per axis"])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, .94))
    fig.suptitle("G3 score smoothing: classification and boundary tradeoffs", y=.985, fontsize=14)
    fig.subplots_adjust(left=.06, right=.98, top=.84, bottom=.19, hspace=.42, wspace=.25)
    fig.text(.06, .065, "Error bars = perturbation seed SD; clean is a single result. "
             "Boundary matching: same block, +/-1 window, one-to-one, class-blind.\n"
             "Lower flip counts or higher prediction concentration alone do not establish better recognition. "
             "Decision-score smoothing does not denoise coordinates.\n"
             "Centered uses up to 5 future windows; trailing waits for the current window. "
             "For paired user-group intervals use the accompanying table.", fontsize=9)
    save_figure(fig, "figure2_smoothing_tradeoffs", qa)

    cases = pd.read_csv(EXP / "metrics/figure3_case_data.csv", dtype={"user_id": str})
    clean = pd.read_csv(EXP / "predictions/G3_sigma00_seed0_smoothed.csv",
                        dtype={"user_id": str})
    identity = ["user_id", "feature_block_id", "start_time"]
    if clean.duplicated(identity).any():
        raise ValueError("Ambiguous clean prediction keys")
    cases = cases.merge(clean[identity + ["S0_raw_k1"]].rename(
        columns={"S0_raw_k1": "clean_raw"}), on=identity, how="left", validate="many_to_one")
    if cases.clean_raw.isna().any():
        raise ValueError("Unaligned saved case")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    selection = json.loads((EXP / "metrics/case_selection.json").read_text(encoding="utf-8"))
    for ax, case in zip(axes, [1, 2]):
        part = cases[cases.case.eq(case)].sort_values("start_time")
        start = pd.to_datetime(part.start_time)
        end = pd.to_datetime(part.end_time)
        origin = start.min()
        left = (start - origin).dt.total_seconds().to_numpy() / 60
        widths = (end - start).dt.total_seconds().to_numpy() / 60
        # Zero-duration windows remain visible as a mark, without changing source times.
        widths = np.maximum(widths, .02)
        cols = ["label", "clean_raw", *SMOOTHERS] if case == 1 else ["label", *SMOOTHERS]
        names = ["Truth majority", "Clean raw", "Noisy raw", "Noisy centered", "Noisy trailing"] \
            if case == 1 else ["Truth majority", *LABELS]
        for j, column in enumerate(cols):
            for k, mode in enumerate(MODES):
                mask = part[column].eq(mode).to_numpy()
                ax.broken_barh(list(zip(left[mask], widths[mask])), (j - .32, .64),
                               facecolors=MODE_COLORS[k], edgecolors="none")
        if case == 2:
            first, stop = selection["short_run_indices"]
            a = (pd.to_datetime(windows.iloc[first].start_time) - origin).total_seconds() / 60
            b = (pd.to_datetime(windows.iloc[stop - 1].end_time) - origin).total_seconds() / 60
            ax.add_patch(Rectangle((a, -.48), b - a, len(cols) - .04, fill=False,
                                   linewidth=1.8, edgecolor="black", linestyle="--"))
        ax.set_yticks(range(len(cols)), names)
        ax.set_ylim(len(cols) - .35, -.65)
        ax.set_xlabel("Minutes from first displayed window (actual timestamps)")
        ax.set_title("A. Saved noise-failure case: 20m, seed 20261123" if case == 1
                     else "B. Saved short-mode case: no added noise")
        ax.grid(axis="x")
    fig.legend(handles=[Patch(color=c, label=m) for c, m in zip(MODE_COLORS, MODES)],
               loc="lower center", bbox_to_anchor=(.5, .13), ncol=5, frameon=False)
    fig.subplots_adjust(left=.11, right=.98, top=.83, bottom=.31, wspace=.4)
    fig.text(.06, .055, "Original deterministic case selection is unchanged. Dashed outline: original "
             "short truth run. Truth bands show window majority, not every point label.\n"
             "Band width uses actual window duration; gaps are not filled. Examples explain failure, "
             "not its population frequency. Smoothing acts on decision scores.", fontsize=9)
    save_figure(fig, "figure3_saved_cases_actual_time", qa)
    return qa


def raster_audit():
    rows = []
    for folder in (EXP / "figures", EXP / "diagnostics/figures", HERE / "figures"):
        for path in sorted(folder.glob("*.png")):
            with Image.open(path) as image:
                pixels = np.asarray(image.convert("RGB"))
            nonwhite = float((pixels.min(axis=2) < 245).mean())
            assert pixels.std() > 1 and .005 < nonwhite < .95, path
            rows.append({"path": str(path.relative_to(ROOT)), "width": pixels.shape[1],
                         "height": pixels.shape[0], "nonwhite_fraction": nonwhite,
                         "sha256": sha(path), "manual_review": "pending_tool_cannot_view_images"})
    return rows


def report_tables(table, tolerance, windows):
    raw = table[table.smoother.eq(SMOOTHERS[0])]
    raw_md = md_table(["组", "单轴σ/m", "raw F1均值±种子SD", "noisy−clean", "用户组95%区间"],
        [[r.group, r.sigma, f"{r.macro_f1_seed_mean:.4f}" + (
            f" ± {r.macro_f1_seed_sd:.4f}" if r.sigma else "（单份）"),
          f"{r.macro_f1_delta_from_clean:+.4f}",
          f"[{r.macro_f1_ci_low:.4f}, {r.macro_f1_ci_high:.4f}]"]
         for r in raw.itertuples()])
    g3 = table[table.group.eq("G3") & table.sigma.isin([0, 20])]
    smooth_md = md_table(["单轴σ/m", "平滑", "点级F1", "Accuracy", "边界P", "边界R", "边界F1",
                          "错误孤立跳变/100窗", "最大预测类别占比"],
        [[r.sigma, LABELS[SMOOTHERS.index(r.smoother)], f"{r.macro_f1_seed_mean:.4f}",
          f"{r.accuracy_seed_mean:.4f}", f"{r.transition_precision_seed_mean:.4f}",
          f"{r.transition_recall_seed_mean:.4f}", f"{r.transition_f1_seed_mean:.4f}",
          f"{r.wrong_isolated_flips_per100_windows_seed_mean:.4f}",
          f"{r.most_predicted_point_fraction_seed_mean:.2%}"] for r in g3.itertuples()])
    contrasts = pd.read_csv(EXP / "paired_analysis/tables/paired_smoothing_contrasts.csv")
    subset = contrasts[contrasts.sigma.isin([0, 20]) &
                       contrasts.metric.isin(["macro_f1", "transition_recall"])]
    paired_md = md_table(["σ/m", "平滑−raw", "指标", "配对差均值", "用户组配对95%区间"],
        [[r.sigma, r.comparison, r.metric, f"{r.difference_seed_mean:+.4f}",
          f"[{r.ci_low:+.4f}, {r.ci_high:+.4f}]"] for r in subset.itertuples()])
    counts = windows[[f"cnt_{m}" for m in MODES]].to_numpy(np.int64)
    mixed = (counts > 0).sum(axis=1) > 1
    support = {
        "all_windows": int(len(windows)), "all_points": int(counts.sum()),
        "mixed_label_windows": int(mixed.sum()),
        "points_in_mixed_label_windows": int(counts[mixed].sum()),
        "points_not_in_window_majority": int((counts.sum(axis=1) - counts.max(axis=1)).sum()),
        "window_majority_boundaries": int(len(tolerance)),
        "definition": "counts_only_no_predictions_or_primary_metric_changed",
    }
    write_json(HERE / "label_resolution.json", support)
    tol_md = md_table(["±1窗候选位置的时间支持", "中位数/s", "P90/s", "最小/s", "最大/s"],
        [[name, f"{tolerance[col].median():.1f}", f"{tolerance[col].quantile(.9):.1f}",
          f"{tolerance[col].min():.1f}", f"{tolerance[col].max():.1f}"]
         for col, name in [("early_tolerance_s", "可提前"),
                           ("late_tolerance_s", "可滞后"),
                           ("full_candidate_span_s", "最早至最晚候选")]])
    return {"RAW_TABLE": raw_md, "SMOOTH_TABLE": smooth_md, "PAIRED_TABLE": paired_md,
            "TOLERANCE_TABLE": tol_md, "MIXED_WINDOWS": f"{support['mixed_label_windows']:,}",
            "MIXED_POINTS": f"{support['points_in_mixed_label_windows']:,}",
            "MINORITY_POINTS": f"{support['points_not_in_window_majority']:,}"}


def main():
    HERE.mkdir(exist_ok=True)
    for folder in ("tables", "figures"):
        (HERE / folder).mkdir(exist_ok=True)
    target = HERE / "acceptance.json"
    target.unlink(missing_ok=True)
    (HERE / "delivery_verification.json").unlink(missing_ok=True)
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected_files()}
    write_json(HERE / "protected_input_manifest.json", before)

    protocol = json.loads((EXP / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["sigmas_m"] == [0, 5, 10, 20]
    assert protocol["seeds"] == list(range(20261123, 20261128))
    original = pd.read_csv(EXP / "metrics/all_metrics.csv")
    assert len(original) == 144
    table = summarize(original)
    assert len(table) == 36
    bootstrap = pd.read_csv(EXP / "metrics/stage5_bootstrap_all.csv")
    overlapping = [c for c in bootstrap if c in table and c not in KEYS]
    check = unique_join(table[KEYS + overlapping], bootstrap[KEYS + overlapping], KEYS)
    for col in overlapping:
        np.testing.assert_allclose(check[f"{col}_x"], check[f"{col}_y"], rtol=0, atol=1e-12)
    table = unique_join(table, bootstrap.drop(columns=overlapping), KEYS)
    concentration = pd.read_csv(
        EXP / "literature_review/tables/reconstructed_metrics_and_concentration.csv")
    concentration = concentration.groupby(KEYS).most_predicted_point_fraction.agg(
        ["mean", "std"]).reset_index().rename(columns={
            "mean": "most_predicted_point_fraction_seed_mean",
            "std": "most_predicted_point_fraction_seed_sd"})
    concentration["most_predicted_point_fraction_seed_sd"] = (
        concentration.most_predicted_point_fraction_seed_sd.fillna(0))
    table = unique_join(table, concentration, KEYS).sort_values(KEYS)
    table["point_metric_source"] = "../../metrics/all_metrics.csv"
    table["user_ci_source"] = "../../metrics/stage5_bootstrap_all.csv"
    table.to_csv(HERE / "tables/all_conditions.csv", index=False)
    table[table.smoother.eq(SMOOTHERS[0])].to_csv(HERE / "tables/raw_comparison.csv", index=False)

    windows = pd.read_csv(EXP / "feature_cache/sigma00_seed0.csv", dtype={"user_id": str})
    assert len(windows) == 17380 and windows.n_points.sum() == 860972
    assert windows.user_id.nunique() == 13
    support = boundary_support(windows)
    assert len(support) == 529
    support.to_csv(HERE / "tables/boundary_tolerance_seconds.csv", index=False)
    summaries = []
    for col in ["early_tolerance_s", "late_tolerance_s", "full_candidate_span_s"]:
        summaries.append({"quantity": col, "boundaries": len(support),
                          "median_s": support[col].median(), "p90_s": support[col].quantile(.9),
                          "min_s": support[col].min(), "max_s": support[col].max()})
    pd.DataFrame(summaries).to_csv(HERE / "tables/boundary_tolerance_summary.csv", index=False)

    protocol_table = pd.read_csv(EXP / "metrics/table1_protocol.csv")
    protocol_table.loc[protocol_table.item.eq("split"), "details"] = (
        "51 fixed training users; 49 contribute eligible fitting windows; "
        "13 fixed test users; 9 linked test groups")
    protocol_table.to_csv(HERE / "tables/protocol_clarified.csv", index=False)
    texts = report_tables(table, support, windows)
    template = (HERE / "REPORT_TEMPLATE.md").read_text(encoding="utf-8")
    for key, value in texts.items():
        template = template.replace("{{" + key + "}}", value)
    if "{{" in template:
        raise ValueError("Unresolved report template field")
    (HERE / "REPORT_DRAFT.md").write_text(template, encoding="utf-8")
    layout = make_figures(table, windows)
    write_json(HERE / "plot_layout_audit.json", layout)
    pixels = raster_audit()
    write_json(HERE / "raster_audit.json", pixels)
    after = {str(p.relative_to(ROOT)): sha(p) for p in protected_files()}
    assert before == after, "Protected file set or contents changed"
    outputs = {str(p.relative_to(HERE)): sha(p) for p in HERE.rglob("*")
               if p.is_file() and p.name not in
               {"acceptance.json", "delivery_verification.json", "tests.log"}
               and "__pycache__" not in p.parts and "matplotlib_cache" not in p.parts}
    write_json(target, {
        "passed": True, "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protected_file_count": len(before), "changed_protected_files": [],
        "condition_summary_rows": len(table), "source_metric_rows": len(original),
        "boundary_support_rows": len(support), "new_figures": len(layout),
        "nonblank_pngs_checked": len(pixels), "manual_figure_review_completed": False,
        "no_training_noise_or_bootstrap_runs": True,
        "new_metric_status": "descriptive_time_support_and_label_counts_only",
        "original_experiment_protocol_changed": False,
        "docs_disclosure_outside_protected_set": True, "output_sha256": outputs,
    })
    print(json.dumps({"passed": True, "conditions": len(table), "boundaries": len(support),
                      "protected_files": len(before), "new_figures": len(layout)}))


if __name__ == "__main__":
    main()
