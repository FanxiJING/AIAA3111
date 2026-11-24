"""Derive report tables from accepted diagnostics; no model changes."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TABLES = HERE / "tables"
MODES = ["walk", "bike", "bus", "car", "train"]


def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(str(x) for x in row) + " |" for row in rows],
    ])


def main():
    accepted = json.loads((HERE / "acceptance.json").read_text())
    verified = json.loads((HERE / "independent_verification.json").read_text())
    assert accepted["status"] == verified["status"] == "passed"
    joint = pd.read_csv(TABLES / "joint_metrics_by_seed.csv", dtype={"linked_group": str})
    summary = pd.read_csv(TABLES / "joint_summary.csv", dtype={"linked_group": str})
    composition = pd.read_csv(TABLES / "composition.csv", dtype={"linked_group": str})
    kernel = pd.read_csv(TABLES / "g1_kernel_bins_summary.csv")
    features = pd.read_csv(TABLES / "g1_feature_shifts_summary.csv")
    transitions = pd.read_csv(TABLES / "g1_landmark_transitions_summary.csv")
    pooled = joint[joint.linked_group == "all"]
    clean = pooled[pooled.sigma == 0].set_index(["group", "dt_bin", "speed_bin"])
    class_rows = []
    for row in pooled.to_dict("records"):
        ref = clean.loc[row["group"], row["dt_bin"], row["speed_bin"]]
        for mode in MODES:
            support = row[f"support_{mode}"]
            tp, pred = row[f"tp_{mode}"], row[f"predicted_{mode}"]
            class_rows.append({
                **{k: row[k] for k in ("group", "sigma", "seed", "dt_bin", "speed_bin", "points")},
                "true_class": mode, "support": support, "predicted_points": pred,
                "true_positives": tp, "clean_true_positives": ref[f"tp_{mode}"],
                "clean_recall": ref[f"tp_{mode}"]/support if support else np.nan,
                "noisy_recall": tp/support if support else np.nan,
                "class_f1": 2*tp/(support+pred) if support+pred else 0.,
            })
    pd.DataFrame(class_rows).to_csv(TABLES / "joint_class_metrics_by_seed.csv", index=False)
    kernels = []
    for sigma in (0, 5, 10, 20):
        row = kernel[(kernel.sigma == sigma) & (kernel.max_kernel_bin == "all")].iloc[0]
        kernels.append([
            sigma, f"{row.nearest_identity_changed_point_fraction_mean:.2%}",
            f"{row.nearest_response_changed_point_fraction_mean:.2%}",
            f"{row.model_nearest_response_agreement_point_fraction_mean:.2%}",
            f"{row.accuracy_mean:.2%}", f"{row.score_gap_median_mean:.4f}",
        ])
    high = kernel[(kernel.sigma == 20) & (kernel.max_kernel_bin == "[0.9,1]")].iloc[0]
    medium = kernel[(kernel.sigma == 20) & (kernel.max_kernel_bin == "[0.5,0.9)")].iloc[0]
    tr20 = transitions[transitions.sigma == 20]
    fractions = {m: tr20[f"model_predicted_{m}_mean"].sum()/860972 for m in MODES}
    ft = features[(features.sigma == 20) & (features.majority_class == "all")]
    feature_table = table(
        ["G1特征", "无附加噪声中位数", "20米中位数的种子均值", "训练尺度下有符号偏移中位数"],
        [[r.feature, f"{r.clean_median_mean:.4f}", f"{r.noisy_median_mean:.4f}",
          f"{r.standardized_signed_shift_median_mean:+.4f}"] for r in ft.itertuples()])
    comp = composition[composition.linked_group == "all"].set_index(["dt_bin", "speed_bin"])
    select = [
        ("G1", "1-2s", "0.5-2mps"), ("G1", "2-5s", "0.5-2mps"),
        ("G1", ">5s", "0.5-2mps"), ("G1", "1-2s", ">2mps"),
        ("G1", ">5s", ">2mps"), ("G3", "1-2s", "0.5-2mps"),
        ("G3", "2-5s", "0.5-2mps"), ("G3", ">5s", "0.5-2mps"),
    ]
    sample_rows = []
    for group, dt, speed in select:
        r = summary[(summary.group == group) & (summary.sigma == 20) &
                    (summary.dt_bin == dt) & (summary.speed_bin == speed) &
                    (summary.linked_group == "all")].iloc[0]
        n = comp.loc[dt, speed].points
        sample_rows.append([
            group, dt, speed, f"{n:,}", f"{r.clean_macro_f1_mean:.4f}",
            f"{r.macro_f1_mean:.4f} ± {r.macro_f1_std:.4f}",
            f"{r.damage_given_clean_correct_mean:.2%}",
        ])
    shortcut = joint[(joint.group == "G2") & (joint.sigma == 20) &
                     (joint.dt_bin == "<=1s") & (joint.speed_bin == "0.5-2mps") &
                     (joint.linked_group == "all")]
    assert len(shortcut) == 5 and (shortcut.predicted_walk == shortcut.points).all()
    assert (shortcut.clean_correct == shortcut.noisy_correct).all()
    shortcut_row = shortcut.iloc[0]
    low = composition[(composition.dt_bin == "1-2s") &
                      (composition.speed_bin == "0.5-2mps")]
    low_all = low[low.linked_group == "all"].iloc[0]
    linked0 = low[low.linked_group == "linked0"].iloc[0]
    report = f"""# 噪声进一步探索：G1失效与采样间隔、速度的交叉影响

这是AI辅助的事后诊断草稿，需要小组核验、改写与引用检查。
本轮复用原G1/G2/G3、16份iid缓存及已有raw分数，完成48组条件诊断，
没有新增训练、噪声、用户划分、平滑方案或bootstrap。
原协议、模型、主结果与此前诊断包全部保留。

## 1. 新增证据

1. **G1不能统一解释为核响应消失。** 高相似度区域也大量判错，
   受扰动输入的最近参考点及对应模型响应发生变化。
2. **G1主要流向car和train。** 20米下预测点占比为
   car {fractions['car']:.2%}、train {fractions['train']:.2%}；
   应与G2的walk集中、G3的train集中分别解释。
3. **分层“准确率保住了”可能只是类别构成碰巧有利。**
   G2有一层达到92.11%准确率，但全部预测walk；
   不能据此宣称它保留了该层的多类别识别能力。

## 2. G1：有核响应并不意味着判对

这里只把冻结模型的1000个Nystroem基点称为参考点，
不把它们当成全部训练样本的代表或真实标签。
“参考点响应”是在该标准化特征向量上运行同一个冻结模型所得类别，
不是参考点所属训练样本的ground-truth。

{table(["sigma/米", "最近参考点身份改变/点", "最近参考点模型响应改变/点", "实际预测与最近参考点响应一致/点", "点Accuracy", "分数间隔中位数"], kernels)}

表中的点比例按原窗口所含点数加权。分数间隔是最大与次大decision分数之差，
先取窗口中位数再取种子均值；它不是校准概率，也不直接代表判对可能性。
未把不同模型间的分数尺度拿来比较。

20米下，maxK≥0.9的窗口覆盖平均{high.points_mean:,.1f}个点
（{high.point_fraction_mean:.2%}测试点），其点准确率仅
{high.accuracy_mean:.2%}；maxK在[0.5,0.9)的窗口覆盖
{medium.points_mean:,.1f}个点，准确率{medium.accuracy_mean:.2%}。
这些不是独立用户样本；小数是五种子均值。
因此最大基点相似度不能单独用作本研究的可靠性判据或已验证的OOD检测器。

20米下最近参考点的模型响应改变占{kernel[(kernel.sigma == 20) & (kernel.max_kernel_bin == 'all')].iloc[0].nearest_response_changed_point_fraction_mean:.2%}点，
而实际模型预测与最近参考点响应的一致率仅41.52%。
这也说明**实际分类不是最近参考点类别投票**；最终分数来自整个冻结表示和分类器。
本轮没有识别某一个参考点变化的因果效应。

逐参考点响应见[g1_landmark_responses.csv](tables/g1_landmark_responses.csv)，
全部25种参考响应转移及真实类别构成见
[g1_landmark_transitions_summary.csv](tables/g1_landmark_transitions_summary.csv)。

## 3. 速度、波动与stop/slow比例同时改变

{feature_table}

速度类单位m/s，stop/slow是窗口中满足原阈值的比例。
原始特征中位数只取有限观测值；有效值数列在完整CSV。
训练尺度下的配对偏移使用冻结填补器和标准化器，保留全部窗口。
“两个中位数之差”与“逐窗配对差的中位数”不同。
速度均值、波动及高分位上移，stop/slow比例下移，与实际类别流向相符，
但不能用它们线性分摊Macro-F1损失，也不能把它们写成已识别的因果贡献。

见[g1_feature_shifts_summary.csv](tables/g1_feature_shifts_summary.csv)。
全部真实多数类别的窗口诊断保留；多数类别只用于此处窗口描述，
从未用于替代点级真实类别计数。

## 4. 联合分层：低速和短采样间隔不能混为单一效应

分层始终使用无附加噪声缓存：dt沿用≤1、(1,2]、(2,5]、>5秒及invalid；
clean speed_mean分为≤0.5、(0.5,2]、>2 m/s及invalid。
速度阈值借用原点特征的stop/slow数值，但窗口均速分层不等同真实静止/慢行状态。
未加回dt模型特征、重新采样或修改窗口。
20个笛卡尔单元全部输出，其中13个非空；17个缺失运动特征的单点窗口也保留。

下表摘选常见单元说明现象；完整20单元和9个关联组结果均可复查。
“原对→错率”分母是该模型、该单元sigma=0判对的真实点数。

{table(["模型", "dt_mean", "clean speed_mean", "点数", "clean F1", "20米F1：种子均值±SD", "原对→错率"], sample_rows)}

这些F1均固定五类、按窗口内真实类别点数计算；缺失类别记零，
不同单元的类别覆盖会影响绝对分数。不要跨行只看F1大小判断谁更抗噪。
SD仅反映五个人工扰动种子；本轮没有新增跨用户置信区间。

例如dt=(1,2]秒、clean均速=(0.5,2]m/s这一层有
{low_all.points:,}点，walk占{low_all.support_walk/low_all.points:.2%}，
linked0占{linked0.points/low_all.points:.2%}。
即使控制了这两个分箱，用户、真实类别、block状态及箱内连续取值仍可不同，
所以不能把层间差异写成采样间隔或低速的因果效应。

完整点计分见[joint_summary.csv](tables/joint_summary.csv)，
分母和构成见[composition.csv](tables/composition.csv)，
逐类支持数、预测数、召回和F1见
[joint_class_metrics_by_seed.csv](tables/joint_class_metrics_by_seed.csv)。

## 5. 一个应主动展示的反例：高Accuracy不等于多类别抗噪

G2在dt≤1秒、clean均速=(0.5,2]m/s这一层共有
{shortcut_row.points:,}点，其中walk {shortcut_row.support_walk:,}点，
占{shortcut_row.support_walk/shortcut_row.points:.2%}。
五个20米种子均将该层所有点预测为walk；
其Accuracy={shortcut_row.accuracy:.4f}，固定五类Macro-F1={shortcut_row.macro_f1:.4f}。
clean下也已全部预测walk，因此原对→错和原错→对均为零。

结合既有G2核响应衰减和walk截距偏好的诊断，这表明该单元的高准确率
与多数类别构成、预测集中同时存在，**不能作为多类别识别能力保留的充分证据**。
原Macro-F1和Accuracy全部照实保留，没有因为结果难解释而删除样本。

## 6. 对课程主线的实际帮助

建议把已有结论写得更准确：
“定位扰动导致运动特征偏移；不同特征模型出现不同的表示与决策变化。
G2/G3存在核响应明显衰减，G1则即使对部分参考点仍有较强响应也可能分类失效。
分层指标受真实类别和用户组构成影响，必须同时报告类别支持数、绝对表现与配对变化。”

新增内容适合放入**失效机制与限制**，与原分类曲线、平滑/短模式代价并列。
不能据此在测试集重选G1、gamma、噪声方案、阈值或k。
当前证据不能区分某个新去噪方法是否有效；它需要另一份训练用户内验证的独立方案，
不属于本轮已经完成的结论。

## 7. 验收及限制

- 独立程序复核{verified['joint_confusion_and_flow_rows']:,}行联合单元及组内混淆/流向；
  {verified['pooled_nonempty_weighted_sklearn_f1']:,}个非空全组单元用sklearn加权F1核对。
- 实际Pipeline及直接距离/核计算核对{verified['actual_pipeline_and_distance_samples']:,}个取样窗口；
  全部1000个G1参考点的响应通过逆标准化后完整Pipeline再次核对。
- {verified['landmark_response_transitions']:,}行参考响应转移、
  {verified['summary_values']:,}个汇总值通过独立复算。
- {verified['protected_files']:,}个既有文件的SHA-256和清单保持一致。
- 本轮发现并修复的是新诊断表中缺失值导致中位数为空的汇总代码问题；
  增补有限值分母并完整重跑，未更改模型、原特征定义或实验协议。
- 全部分析是多轮查看同一测试集后的探索；没有新增独立用户或真实设备误差标定。
  本轮未制作新图，既有图表人工视觉核对的限制仍保留。

验收见[acceptance.json](acceptance.json)和
[independent_verification.json](independent_verification.json)。
范围及执行修正见[ANALYSIS_SPEC.md](ANALYSIS_SPEC.md)。
流程使用scientific-critical-thinking辅助区分描述、因果与独立单位，
软件引用及此前文献实际阅读范围沿用[来源记录](../problem_investigation/SOURCES.md)。
本轮官方网页读取没有返回可用内容，不声称新增论文阅读或最新文献核验。
"""
    (HERE / "FINDINGS.md").write_text(report, encoding="utf-8")
    print(json.dumps({"report_written": True, "class_rows": len(class_rows),
                      "report_lines": len(report.splitlines())}))


if __name__ == "__main__":
    main()
