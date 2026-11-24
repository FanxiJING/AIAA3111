"""Collect completed sampling runs and compare predictions on identical retained GPS points."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import MODES, cv_groups, load_windows, user_bootstrap

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
MODEL = "SVM (RBF, Nystroem, all data)"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared-root", type=Path, default=PROJECT / "outputs/sampling_main_v1")
    ap.add_argument("--results", type=Path, default=PROJECT / "outputs/sampling_results_reference")
    a = ap.parse_args()
    output = a.results / "summary"
    output.mkdir(parents=True, exist_ok=True)
    conditions = sorted(a.prepared_root.glob("*/condition.json"),
                        key=lambda p: json.loads(p.read_text())["interval_seconds"])
    rows, point_predictions, support_checks, skipped = [], [], [], []
    for condition in conditions:
        cfg = json.loads(condition.read_text())
        name = condition.parent.name
        root = a.results / name
        if not (root / "test_results.csv").is_file():
            windows = load_windows(condition.parent / "windows.csv")
            if not ((windows.split == "train") & (windows.n_points >= 10)).any():
                skipped.append({"condition": name, "reason": "No training windows satisfy the fixed minimum of 10 points"})
                continue
            raise FileNotFoundError(f"Feasible condition has no completed result: {root}")
        results = pd.read_csv(root / "test_results.csv")
        row = results.set_index("model").loc[MODEL].to_dict()
        row.update({"condition": name, "interval_s": cfg["interval_seconds"]})
        stats = pd.read_csv(condition.parent / "sampling_by_user.csv")
        row["retained_train_points"] = int(stats.loc[stats.split == "train", "retained_points"].sum())
        row["retained_test_points"] = int(stats.loc[stats.split == "test", "retained_points"].sum())
        row["test_point_fraction"] = row["retained_test_points"] / int(stats.loc[stats.split == "test", "input_points"].sum())
        durations = pd.read_csv(condition.parent / "blind_window_durations.csv").set_index("split")
        row["test_median_window_seconds"] = durations.loc["test", "50%"]
        rows.append(row)
        pred = pd.read_csv(root / "window_predictions_test.csv", dtype={"user_id": str})
        windows = load_windows(condition.parent / "windows.csv")
        windows = windows[windows.split == "test"].reset_index(drop=True)
        keys = ["user_id", "feature_block_id", "start_time", "end_time", "n_points"]
        if not windows[keys].equals(pred[keys]):
            raise ValueError(f"Window prediction alignment failed: {name}")
        members = pd.read_csv(condition.parent / "test_point_membership.csv", dtype={"user_id": str})
        which = members.test_window_index.to_numpy(dtype=int)
        if (which < 0).any() or (which >= len(pred)).any():
            raise ValueError("Invalid point-to-window indices")
        reconstructed = pd.crosstab(members.test_window_index, members["mode"]).reindex(
            index=windows.index, columns=MODES, fill_value=0)
        if not np.array_equal(reconstructed.to_numpy(), windows[[f"cnt_{m}" for m in MODES]].to_numpy()):
            raise ValueError(f"Point membership does not reproduce window class counts: {name}")
        start, end = pred.start_time.to_numpy()[which], pred.end_time.to_numpy()[which]
        if not ((members.timestamp.to_numpy() >= start) & (members.timestamp.to_numpy() <= end)).all():
            raise ValueError("Point timestamp outside prediction window")
        support_checks.append({"condition": name, "mapped_points": len(members), "counts_match": True})
        frame = members[["user_id", "timestamp", "mode"]].copy()
        frame[f"{name}: raw"] = pred[f"{MODEL} | raw"].to_numpy()[which]
        frame[f"{name}: final"] = pred[MODEL].to_numpy()[which]
        frame = frame.set_index(["user_id", "timestamp"], verify_integrity=True)
        point_predictions.append(frame.rename(columns={"mode": name + ": truth"}))
    table = pd.DataFrame(rows).sort_values("interval_s")
    (output / "infeasible_conditions.json").write_text(json.dumps(skipped, indent=2))
    table.to_csv(output / "retained_support_results.csv", index=False)
    common = pd.concat(point_predictions, axis=1, join="inner").sort_index()
    truth_cols = [c for c in common if c.endswith(": truth")]
    if not common[truth_cols].eq(common[truth_cols[0]], axis=0).all().all():
        raise ValueError("Ground-truth labels differ across conditions on the same GPS observations")
    if common.empty:
        raise ValueError("No all-condition common support")
    truth = common[truth_cols[0]]
    meta = pd.DataFrame({f"cnt_{m}": truth.eq(m).astype(int).to_numpy() for m in MODES})
    meta["n_points"] = 1
    users = pd.Series(common.index.get_level_values("user_id"))
    groups = cv_groups(users, PROJECT / "data/splits/report.json")
    runs = [SimpleNamespace(name=c, y_pred=common[c].to_numpy()) for c in common if not c.endswith(": truth")]
    ci, pairs = user_bootstrap(runs, meta, groups, MODES, n_boot=5000, seed=42)
    ci.to_csv(output / "common_support_results.csv", index=False)
    pairs.to_csv(output / "common_support_paired_differences.csv", index=False)
    common.to_csv(output / "common_support_predictions.csv")
    by_user = pd.DataFrame({"user_id": users, "group": groups}).groupby(["user_id", "group"]).size().rename("points")
    by_user.to_csv(output / "common_support_by_user.csv")
    (output / "support_verification.json").write_text(json.dumps({"checks": support_checks,
        "common_points": len(common), "common_users": users.nunique(), "common_groups": groups.nunique()}, indent=2))

    # Main plot explicitly shows that the evaluated observations vary by condition.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    x = np.arange(len(table))
    labels = ["Original" if s == 0 else f">= {s:g} s" for s in table.interval_s]
    axes[0].plot(x, table.raw_macro_f1, "o--", color="#929BA8", label="No smoothing")
    axes[0].errorbar(x, table.point_macro_f1,
                     yerr=[table.point_macro_f1 - table.macro_f1_ci_low,
                           table.macro_f1_ci_high - table.point_macro_f1],
                     fmt="o-", color="#2468A0", capsize=4, label="Training-CV smoothing")
    axes[0].set(ylabel="Point macro-F1", ylim=(0, 1), title="Retained observations (support varies)")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].bar(x, table.test_point_fraction * 100, color="#4A8F92", width=0.6)
    axes[1].set(ylabel="Retained test observations (%)", ylim=(0, 110), title="GPS observation budget")
    for i, r in enumerate(table.itertuples()):
        axes[1].text(i, r.test_point_fraction * 100 + 2, f"{r.test_point_fraction:.1%}", ha="center")
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_xlabel("Minimum sampling interval")
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "sampling_retained_support.png", dpi=300)
    fig.savefig(output / "sampling_retained_support.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
    for suffix, color, offset in [("raw", "#929BA8", -0.06), ("final", "#2468A0", 0.06)]:
        subset = ci.set_index("model").loc[[f"{n}: {suffix}" for n in table.condition]]
        ax.errorbar(x + offset, subset.point_macro_f1,
                    yerr=[subset.point_macro_f1 - subset.macro_f1_ci_low,
                          subset.macro_f1_ci_high - subset.point_macro_f1], fmt="o-", capsize=4,
                    color=color, label="No smoothing" if suffix == "raw" else "Training-CV smoothing")
    ax.set_xticks(x, labels)
    ax.set(ylabel="Point macro-F1", ylim=(0, 1), xlabel="Minimum sampling interval",
           title=f"Identical GPS observations across all conditions (n={len(common):,})")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(output / "sampling_common_support.png", dpi=300)
    fig.savefig(output / "sampling_common_support.svg")
    plt.close(fig)
    cols = ["interval_s", "raw_macro_f1", "point_macro_f1", "user_avg_macro_f1", "test_point_fraction", "test_median_window_seconds", "k"]
    summary = table[cols].round(4).to_string(index=False)
    (output / "results_brief.txt").write_text(
        "GPS 降采样实验：真实数据结果\n\n" + summary +
        f"\n\n共同支持集：{len(common):,} 点，{users.nunique()} 用户，{groups.nunique()} 独立关联组。\n"
        "上表在各自保留点上评估；严格的同点比较见 common_support_results.csv 和配对差值表。\n"
        "固定点数窗口随采样稀疏而延长，平滑 k 也表示更长时间；结论不能归因于采样率单一因素。\n"
        "置信区间按关联用户整组 bootstrap，5000 次；测试组数很少，需同时报告用户平均分数。\n",
        encoding="utf-8")
    print(summary)
    print(f"Common support: {len(common):,} points, {users.nunique()} users, {groups.nunique()} groups")
    print(f"[summary] {output.resolve()}")


if __name__ == "__main__":
    main()
