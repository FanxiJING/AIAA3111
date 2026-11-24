"""Render the Chinese diagnostic report from completed, reconciled tables."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
S0, S1, S2 = ["S0_raw_k1", "S1_centered_soft_k11", "S2_trailing_soft_k11"]
NAMES = {S0: "原始分数", S1: "居中 soft k=11", S2: "因果 trailing soft k=11"}


def read(name):
    return pd.read_csv(HERE / "tables" / f"{name}.csv")


def md(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |",
         "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def link(label, path):
    return f"[{label}](<{path.as_posix()}>)"


def picture(name):
    return f"![{name}](<{(HERE / 'figures' / (name + '.png')).as_posix()}>)"


def mean_sd(values, percent=False):
    values = pd.Series(values)
    scale = 100 if percent else 1
    suffix = "%" if percent else ""
    if len(values) == 1:
        return f"{values.iloc[0] * scale:.4f}{suffix}"
    return f"{values.mean() * scale:.4f}{suffix} ± {values.std() * scale:.4f}{suffix}"


def main():
    audit = json.loads((HERE / "audit.json").read_text(encoding="utf-8"))
    assert audit["passed"] and audit["protected_files_changed"] == []
    metrics = pd.read_csv(EXP / "metrics/all_metrics.csv")
    drift, kernel = read("feature_drift"), read("kernel_summary")
    sub, runs = read("subgroup_metrics"), read("run_length_summary")
    boundaries, support = read("boundary_summary"), read("smoothing_window_support")
    sizes, group_sizes = read("sampling_stratum_sizes"), read("linked_group_sample_sizes")
    intercepts, errors = read("model_intercepts"), read("consecutive_error_summary")
    shares = read("class_metrics_and_prediction_shares")
    bootstrap = read("bootstrap_reproduction")
    contrasts = pd.read_csv(EXP / "metrics/paired_group_delta_contrasts.csv")
    score_rows = []
    for group in ("G1", "G2", "G3"):
        d = metrics[metrics.group.eq(group) & metrics.smoother.eq(S0)]
        clean = d.loc[d.sigma.eq(0), "macro_f1"].iloc[0]
        for sigma in (0, 5, 10, 20):
            v = d.loc[d.sigma.eq(sigma), "macro_f1"]
            score_rows.append([group, sigma, mean_sd(v), f"{clean - v.mean():.4f}"])
    drift_rows = []
    for feature in ("speed_mean", "speed_p90", "acc_mean", "acc_p90",
                    "jerk_mean", "stop_rate", "straightness"):
        d = drift[drift.stratum_type.eq("all") & drift.feature.eq(feature)]
        if d.empty:
            continue
        drift_rows.append([
            feature, f"{d.loc[d.sigma.eq(0), 'clean_median'].iloc[0]:.5f}",
            f"{d.loc[d.sigma.eq(5), 'noisy_median'].mean():.5f}",
            f"{d.loc[d.sigma.eq(20), 'noisy_median'].mean():.5f}",
            f"{d.loc[d.sigma.eq(20), 'paired_abs_shift_train_sd_median'].mean():.4f}",
        ])
    sampling_rows = []
    for _, size in sizes.iterrows():
        q = sub[sub.group.eq("G3") & sub.smoother.eq(S0) &
                sub.stratum_type.eq("sampling_bin") & sub.stratum.eq(size.sampling_bin)]
        sampling_rows.append([
            size.sampling_bin, int(size.windows), int(size.points),
            f"{q.loc[q.sigma.eq(0), 'macro_f1'].iloc[0]:.4f}",
            mean_sd(q.loc[q.sigma.eq(20), "macro_f1"]),
            f"{q.loc[q.sigma.eq(20), 'macro_f1_delta'].mean():+.4f}",
        ])
    krows = []
    k = kernel[kernel.group.eq("G3") & kernel.stratum_type.eq("all")]
    for sigma in (0, 5, 10, 20):
        d = k[k.sigma.eq(sigma)]
        krows.append([
            sigma, f"{d.max_basis_similarity_median.mean():.6g}",
            f"{d.representation_norm_median.mean():.6g}",
            mean_sd(d.near_intercept_point_fraction, True),
            mean_sd(d.intercept_winner_prediction_point_fraction, True)])
    ints = intercepts[intercepts.group.eq("G3")].sort_values("intercept", ascending=False)
    intercept_text = "、".join(f"{r.mode}={r.intercept:.6f}" for r in ints.itertuples())
    clean_metrics = metrics[metrics.group.eq("G3") & metrics.sigma.eq(0)]
    clean_bounds = boundaries[boundaries.sigma.eq(0)]
    smoothing_rows = []
    for smoother in (S0, S1, S2):
        m = clean_metrics[clean_metrics.smoother.eq(smoother)].iloc[0]
        b = clean_bounds[clean_bounds.smoother.eq(smoother)].iloc[0]
        smoothing_rows.append([
            NAMES[smoother], f"{m.macro_f1:.4f}",
            f"{m.wrong_isolated_flips_per100_windows:.4f}",
            f"{b.transition_recall:.4f}", f"{b.transition_f1:.4f}",
            f"{int(b.matched_transitions)}/529", int(b.missed_transitions),
            f"{b.matched_availability_latency_s_median:.1f}"])
    short_rows = []
    for length in ("1", "2-5", "6-10", ">=11"):
        q = runs[runs.sigma.eq(0) & runs.stratum_type.eq("length_bin") &
                 runs.stratum.eq(length)]
        for smoother in (S0, S1, S2):
            d = q[q.smoother.eq(smoother)].iloc[0]
            short_rows.append([
                length, NAMES[smoother], int(d.runs), int(d.absent_runs),
                f"{100 * d.absent_run_fraction:.2f}%",
                f"{int(d.newly_absent_runs)}/{int(d.raw_present_runs)}",
                f"{int(d.all_raw_correct_then_absent_runs)}/{int(d.raw_all_correct_runs)}"])
    support_rows = []
    for smoother in (S0, S1, S2):
        d = support[support.smoother.eq(smoother)]
        support_rows.append([
            NAMES[smoother], f"{d.support_span_s.median():.1f}",
            f"{d.support_span_s.quantile(.9):.1f}",
            f"{d.future_window_wait_s.median():.1f}",
            f"{d.future_window_wait_s.quantile(.9):.1f}"])
    grows = []
    for size in group_sizes.sort_values("points", ascending=False).itertuples():
        d = sub[sub.group.eq("G3") & sub.smoother.eq(S0) &
                sub.stratum_type.eq("linked_group") & sub.stratum.eq(size.linked_group)]
        grows.append([
            size.linked_group, size.users, int(size.points), f"{size.point_fraction:.2%}",
            f"{d.loc[d.sigma.eq(0), 'macro_f1'].iloc[0]:.4f}",
            mean_sd(d.loc[d.sigma.eq(20), "macro_f1"]),
            f"{d.loc[d.sigma.eq(20), 'macro_f1_delta'].mean():+.4f}"])
    biggest = group_sizes.loc[group_sizes.points.idxmax()]
    l = sub[sub.group.eq("G3") & sub.smoother.eq(S0) &
            sub.stratum_type.eq("leave_one_linked_group_out") &
            sub.stratum.eq(biggest.linked_group)]
    contrast = contrasts[contrasts.comparison.eq("G3 minus G2") &
                         contrasts.smoother.eq(S0)].iloc[0]
    class_rows = []
    for mode in ("walk", "bike", "bus", "car", "train"):
        d = shares[shares.group.eq("G3") & shares.smoother.eq(S0) &
                   shares.stratum_type.eq("all") & shares["mode"].eq(mode)]
        class_rows.append([
            mode, int(d.loc[d.sigma.eq(0), "true_points"].iloc[0]),
            f"{d.loc[d.sigma.eq(0), 'f1'].iloc[0]:.4f}",
            mean_sd(d.loc[d.sigma.eq(20), "f1"]),
            mean_sd(d.loc[d.sigma.eq(20), "prediction_point_fraction"], True)])
    brows = []
    for group in ("G1", "G2", "G3"):
        for sigma in (0, 20):
            d = bootstrap[bootstrap.group.eq(group) & bootstrap.smoother.eq(S0) &
                          bootstrap.sigma.eq(sigma)].iloc[0]
            brows.append([
                group, sigma, f"[{d.macro_f1_ci_low:.4f}, {d.macro_f1_ci_high:.4f}]",
                f"[{d.macro_f1_delta_ci_low:+.4f}, {d.macro_f1_delta_ci_high:+.4f}]"])
    error_rows = []
    for sigma in (0, 20):
        for smoother in (S0, S1, S2):
            d = errors[errors.sigma.eq(sigma) & errors.smoother.eq(smoother)]
            error_rows.append([
                sigma, NAMES[smoother], f"{d.error_runs.mean():.1f}",
                f"{d.error_run_windows_median.mean():.2f}",
                f"{d.error_run_duration_s_median.mean():.1f}",
                f"{d.error_run_duration_s_max.mean():.1f}"])
    report = f"""# GeoLife 噪声鲁棒性补充诊断

日期：2026-11-23。性质：固定实验完成后的事后描述性分析，不是新注册实验，也不是模型选择。

## 结论摘要

1. 已完成特征与采样间隔、冻结 RBF 表示、平滑的短模式代价、用户组与类别差异四条诊断。所有主实验文件保持不变，没有训练新模型。
2. 定位扰动伴随运动特征明显偏移，G3 的有限基点 RBF 相似度和映射范数下降，并出现接近截距的决策分数及类别集中。这支持一个部分失效机制，不证明全部错误由它造成。
3. 居中平滑能提高无附加噪声时的点级 Macro-F1，但真实边界召回明显降低。短模式是否被抹除必须单独计数，不能仅凭曲线更平滑宣称鲁棒性改善。
4. 最大关联用户组 {biggest.linked_group} 占 {biggest.point_fraction:.4%} 测试点。G3 与 G2 的退化差异的配对用户区间跨零，不能据此宣布谁普遍更抗噪。

## 1. 保持不变的实验与核验

- 测试集：13 用户、9 关联用户组、860,972 点、17,380 个固定窗口；50 点窗口及原尾窗合并、block 和标签规则均保留。
- 读取现有 16 份特征缓存、48 份原始预测及 48 份平滑预测；无坐标重扰动、无特征重生成、无删点、无 G4。
- 模型仍为原 G1/G2/G3，训练使用 clean，测试坐标使用原固定 E/N 独立高斯噪声。sigma 是单轴标准差，水平 RMS 为 sigma × sqrt(2)。
- 重建全部 834,240 个模型-条件-窗口的冻结 decision 分数，容差为绝对 1e-7；预测标签逐窗完全一致。
- 144 行点级指标、48 行 G3 边界指标与主结果核对一致；36 行 bootstrap 汇总使用原 5000×9 重采样矩阵复现。
- 分析前后对 {audit['protected_files']} 个原始输入及主产物文件做 SHA-256 核验，变化数为 0。交付时另做独立复核。
- 7 项单元测试覆盖采样分箱边界、256 种二分匹配例、因果支持区间、固定五类计分、混合标签窗口点计分和短模式分母。

所有“±”都是 5 个扰动种子的样本标准差，sigma=0 仅有一份，不是 5 次独立实验。bootstrap 区间在第 6 节单列。

### 主实验 raw 结果保留

{md(["特征组", "sigma/m", "Macro-F1：种子均值 ± SD", "相对自身 sigma=0 的下降"], score_rows)}

## 2. 特征偏移与采样间隔

下表每个条件先在窗口间取中位数，再对 5 个种子取均值。不是点加权均值，也不是先混合全部种子。最后一列是“逐窗配对绝对变化 / 训练集已拟合尺度”的中位数再取种子均值，不能与前两列中位数之差混淆。速度单位 m/s，加速度 m/s²，jerk m/s³，比例与 straightness 无量纲。

{md(["特征", "无附加噪声中位数", "sigma=5 中位数", "sigma=20 中位数", "sigma=20 配对绝对偏移/训练尺度"], drift_rows)}

{picture("01_feature_drift")}

按无附加噪声的原始 `dt_mean` 做事后分层，仅用于描述；没有把采样间隔加入模型输入，也没有按噪声重新划分窗口。每层仍按固定五类点级 Macro-F1 计分。

{md(["dt_mean", "窗口数", "点数", "G3 raw F1：0m", "G3 raw F1：20m", "20m 减 0m"], sampling_rows)}

{picture("02_sampling_interval")}

解释边界：相同位置扰动经过距离和时间差运算可能对高频采样产生较强影响，但各层同时具有不同用户、类别和运动状态构成，不能将层间差异当作采样间隔的因果效应。真实 GPS 误差还可能存在时序相关；本实验未进行真实误差标定。距离高估的理论条件及相应文献见 SOURCES.md。

## 3. 冻结 RBF 表示与预测集中

RBF 采用 `exp(-gamma * squared_distance)`。诊断使用训练时冻结的填补器、标准化器、1000 个 Nystroem 基点、归一化矩阵与分类器，不对测试集重新拟合。

G3 截距从大到小为：{intercept_text}。

{md(["sigma/m", "最大基点相似度中位数", "映射 L2 范数中位数", "近截距分数点占比", "截距赢家预测点占比"], krows)}

{picture("03_rbf_collapse")}

“近截距”仅是诊断标记：五类分数与各自截距的最大绝对差不超过 1e-6，不改变预测规则。相似度和范数是窗口中位数再做种子均值；后两列按真实窗口点数加权。该相似度只针对冻结的 1000 个基点，不代表与全部训练样本的最近相似度。decision 分数不是概率。

可支持的解释是：特征分布偏移与 RBF 表示衰减、截距主导决策同时出现。不能把全部 G3 失败归因于 jerk 或转向，也不能从这组观察中推出某个新 gamma、去噪方法或模型会更好；本轮没有做这些实验。

## 4. 平滑的分类收益与短模式代价

无附加噪声时 G3 的分类与边界表现：

{md(["处理", "点级 Macro-F1", "错误孤立跳变/100窗口", "切换 Recall", "切换 F1", "匹配真实边界", "漏掉边界", "已匹配边界分数可用延迟中位数/s"], smoothing_rows)}

边界仍按 block 内 ±1 窗口、一对一最大数量匹配，且不约束切换类别方向。方向是否正确单列在逐边界表中，不偷换主评价定义。

### 短模式是否被完全抹除

真实 run 是同一个原 block 内连续的窗口多数标签。“模式缺失”指该 run 内没有任何预测窗口给出这个真实模式，不是点级召回率。下表使用 sigma=0，避免把原始噪声错误误认为平滑新增损失：

{md(["真实长度/窗口", "处理", "真实 run 数", "模式缺失数", "模式缺失率", "新增缺失/原始至少命中一次", "原始全对后缺失/原始全对"], short_rows)}

{picture("04_short_mode_loss")}

sigma=20 的面板反映“噪声与固定平滑共同作用后的模式缺失”，不能全部算作平滑造成。其新增缺失和原始全对子集的严格分母见 `run_length_summary.csv`。另按真实持续秒数分层，避免把短窗口序列直接等同于短时间。

### 11 个窗口实际意味着多长时间

{md(["处理", "支持跨度中位数/s", "支持跨度 P90/s", "额外未来等待中位数/s", "额外未来等待 P90/s"], support_rows)}

支持跨度为首个纳入窗口起点到末个纳入窗口终点。额外未来等待相对于当前窗口完成时刻；trailing 为 0，但仍必须等待当前窗口的数据，不代表系统零延迟。居中是离线方案，最多读取 5 个未来窗口；trailing 只读当前及历史窗口。

{picture("06_boundary_latency_and_recall")}

时间误差和分数最早可用延迟仅对成功匹配的边界统计；最大数量匹配不保证最小时间误差。漏检可能很多，因此不能把图中的条件延迟推广到全部真实切换，更不能把少量匹配下较低延迟解释为整体更好。

### 连续错误补表

{md(["sigma/m", "处理", "错误 run 数", "每条件错误长度中位数再均值/窗口", "错误时长中位数再均值/s", "每条件最大错误时长再均值/s"], error_rows)}

此处的连续错误使用窗口多数标签，与点级 Macro-F1 的真值粒度不同，仅作序列诊断。平均最大时长不是全条件最大值。

## 5. 用户组与类别异质性

{md(["关联组", "用户", "点数", "测试点占比", "G3 raw F1：0m", "G3 raw F1：20m", "20m 减 0m"], grows)}

{picture("05_user_group_heterogeneity")}

去掉最大关联组后的描述性敏感性结果：G3 raw 无附加噪声 F1 为 {l.loc[l.sigma.eq(0), 'macro_f1'].iloc[0]:.4f}，20m 为 {mean_sd(l.loc[l.sigma.eq(20), 'macro_f1'])}。这是同一模型对现有子集的计分，不是新划分或重新训练，不能替代主结果。

各组类别覆盖不同。固定五类 Macro-F1 会对缺失类别记 0，组间低分未必完全来自识别更差，因此需要结合类别支持数和组内配对变化解释。逐用户和逐组值分别保留，不把 13 个用户当作 13 个独立关联组。

G3 raw 的全测试集类别结果：

{md(["类别", "真实点数", "F1：0m", "F1：20m", "20m 预测点占比"], class_rows)}

## 6. 用户不确定性与不可宣称的结论

下表为 9 个关联用户组的 5000 次配对 bootstrap 百分位区间。对每次用户抽样分别计算各种子的 F1，再对种子取均值，不先平均混淆矩阵。变化定义为 noisy 减 clean，负数代表下降：

{md(["特征组", "sigma/m", "Macro-F1 的 95% 区间", "相对自身 clean 变化的 95% 区间"], brows)}

G3 减 G2 的 raw 配对变化差为 {contrast.delta_difference:+.6f}，95% 区间 [{contrast.ci_low:+.6f}, {contrast.ci_high:+.6f}]，跨越 0。这里的差是 `(G3 noisy-clean) - (G2 noisy-clean)`，不是两个噪声分数的直接差。没有证据据此宣布总体抗噪排名，也不能把基线较低而下降较少当作更强。

独立用户组只有 9 个且权重高度不均，区间可能较宽；其可信度受这些数据和重采样假设限制。新增分层和诊断阈值均为事后描述，不提供未校正多重检验的“显著性”结论。

## 7. 可直接用于课程报告的表述

> 在固定 clean-training/noisy-test 协议下，额外独立定位扰动使运动特征分布偏移，并伴随冻结 RBF 表示衰减及类别预测集中。分数平滑在无附加噪声条件下改善点级分类，但同时减少真实切换的检出，并可能完全抹除短模式。点级 F1、真实边界保留与可用延迟因而需要共同报告。上述解释属于现有测试样本上的事后机制证据，不构成真实 GPS 场景或模型抗噪排名的普遍结论。

建议正文以“性能退化、特征偏移、表示变化、平滑代价”为四段主线；用户组与类别表以及条件延迟放入结果解释或附录。不要新增未经批准的噪声模型、坐标滤波、gamma 调优或训练增强。

## 8. 交付入口

- {link("文献与软件核查", HERE / "SOURCES.md")}，包含原文范围、可引用内容和不可外推的结论。
- {link("分析规范", HERE / "ANALYSIS_SPEC.md")}。
- {link("逐窗 RBF 诊断目录", HERE / "window_diagnostics")}，48 份 Parquet。
- {link("全部诊断 CSV", HERE / "tables")}，包含全部条件、类别、用户、关联组、短模式和逐边界结果。
- {link("六图的 PNG 与矢量 PDF", HERE / "figures")}。
- {link("运行核验", HERE / "audit.json")}、{link("交付独立复核", HERE / "delivery_audit.json")}、{link("图表视觉检查", HERE / "VISUAL_QA.md")}。

研究流程使用 Scientific Agent Skills 的 scientific-critical-thinking 辅助检查抽样单位、证据边界和事后解释，已在 SOURCES.md 的软件与流程部分注明。它不替代数值验收。
"""
    (HERE / "DIAGNOSTIC_REPORT.md").write_text(report, encoding="utf-8")
    print(f"Report written: {HERE / 'DIAGNOSTIC_REPORT.md'}")
    print(md(["sigma", "G3 basis similarity", "norm", "near intercept", "winner share"], krows))
    print(md(["method", "F1", "wrong flips", "recall", "switch F1", "matches",
              "misses", "availability s"], smoothing_rows))


if __name__ == "__main__":
    main()
