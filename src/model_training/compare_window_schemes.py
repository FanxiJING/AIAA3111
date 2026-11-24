"""Paired fixed-point vs fixed-duration comparison on the same retained GPS observations."""
from pathlib import Path
from types import SimpleNamespace
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from models import MODES, cv_groups, user_bootstrap

PROJECT = Path(__file__).resolve().parents[2]
MODEL = "SVM (RBF, Nystroem, all data)"


def point_predictions(prepared, results):
    members = pd.read_csv(prepared / "test_point_membership.csv", dtype={"user_id": str})
    windows = pd.read_csv(results / "window_predictions_test.csv", dtype={"user_id": str})
    index = members.test_window_index.to_numpy(dtype=int)
    for suffix, column in [("raw", MODEL + " | raw"), ("final", MODEL)]:
        members[suffix] = windows[column].to_numpy()[index]
    return members.set_index(["user_id", "timestamp"], verify_integrity=True)[["mode", "raw", "final"]].sort_index()


def main():
    output = PROJECT / "outputs/window_scheme_comparison_v1"
    output.mkdir(parents=True, exist_ok=True)
    summaries, differences, skipped = [], [], []
    for interval in (0, 5, 10, 30):
        point_name, time_name = f"interval_{interval}s_points_50", f"interval_{interval}s_time_120s"
        point_root = PROJECT / "outputs/sampling_results_reference" / point_name
        time_root = PROJECT / "outputs/sampling_duration_results_reference" / time_name
        if interval == 30:
            skipped.append({"interval_s": interval, "reason": "120 s time windows contain at most 4 points; no training window passes min_points=10"})
            continue
        a = point_predictions(PROJECT / "outputs/sampling_main_v1" / point_name, point_root)
        b = point_predictions(PROJECT / "outputs/sampling_duration_v1" / time_name, time_root)
        if not a.index.equals(b.index) or not a["mode"].equals(b["mode"]):
            raise ValueError("The two window schemes do not cover identical GPS observations")
        meta = pd.DataFrame({f"cnt_{m}": a["mode"].eq(m).astype(int).to_numpy() for m in MODES})
        meta["n_points"] = 1
        users = pd.Series(a.index.get_level_values("user_id"))
        groups = cv_groups(users, PROJECT / "data/splits/report.json")
        runs = [SimpleNamespace(name=f"{scheme}: {version}", y_pred=frame[version].to_numpy())
                for scheme, frame in [("points50", a), ("time120", b)] for version in ("raw", "final")]
        ci, pairs = user_bootstrap(runs, meta, groups, MODES, n_boot=5000, seed=42)
        summaries.append(ci.assign(interval_s=interval, test_points=len(a)))
        differences.append(pairs.assign(interval_s=interval, test_points=len(a)))
    results = pd.concat(summaries, ignore_index=True)
    pairs = pd.concat(differences, ignore_index=True)
    results.to_csv(output / "window_scheme_results.csv", index=False)
    pairs.to_csv(output / "window_scheme_paired_differences.csv", index=False)
    (output / "infeasible_conditions.json").write_text(json.dumps(skipped, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, version, title in zip(axes, ["raw", "final"], ["No smoothing", "Training-CV smoothing"]):
        for scheme, color, offset in [("points50", "#2468A0", -0.08), ("time120", "#C87E35", 0.08)]:
            data = results[results.model == f"{scheme}: {version}"].sort_values("interval_s")
            x = pd.Series(range(len(data))).to_numpy() + offset
            ax.errorbar(x, data.point_macro_f1,
                        yerr=[data.point_macro_f1 - data.macro_f1_ci_low, data.macro_f1_ci_high - data.point_macro_f1],
                        fmt="o-", capsize=4, label="50 points" if scheme == "points50" else "120 seconds", color=color)
        ax.set_xticks(range(3), ["Original", ">= 5 s", ">= 10 s"])
        ax.set(xlabel="Minimum GPS sampling interval", ylabel="Point macro-F1", ylim=(0, 1), title=title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle("Window schemes: identical retained observations within each interval", fontsize=13)
    fig.savefig(output / "window_scheme_comparison.png", dpi=300)
    fig.savefig(output / "window_scheme_comparison.svg")
    plt.close(fig)
    (output / "interpretation_notes.txt").write_text(
        "每个采样条件内，两种窗口在完全相同的 GPS 点上配对评估；不同采样条件之间保留点仍然不同。\n"
        "两种窗口沿用相同的每窗至少10点训练规则，但固定时长方案排除更多训练窗与用户，须同时报告训练覆盖率。\n"
        "模型族、特征、用户折、C=1、balanced权重、soft投票均固定；只有k由每条件训练CV选择。\n"
        "30秒采样+120秒窗口不满足训练最低点数，记为不适用，不能填0分或用测试集挑选其他窗口。\n"
        "差值表的方向为 model_a - model_b；检查95%区间是否包含0后再写显著性结论。\n", encoding="utf-8")
    print(results[["interval_s", "model", "point_macro_f1", "test_points"]].to_string(index=False))


if __name__ == "__main__":
    main()
