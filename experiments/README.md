# GeoLife Part 3 实验交接

Part 3 从两个方向检验 Part 2 的 GPS 出行方式分类器。噪声实验固定窗口和用户划分，在测试坐标上加扰动，比较不同特征组与分数平滑的表现。采样实验改变 GPS 观测密度，同时比较窗口与平滑设置，分别报告回溯分类和在线回放结果。

## 报告与证据

| 内容 | 入口 |
|---|---|
| 定位噪声主实验 | [报告草稿](noise_feature_smoothing/report_revision/REPORT_DRAFT.md)、[完整运行报告](noise_feature_smoothing/FINAL_REPORT.md)、[冻结协议](noise_feature_smoothing/protocol.json)、[指标](noise_feature_smoothing/metrics/)、[图表](noise_feature_smoothing/figures/) |
| 特征与平滑的配对比较 | [配对分析交接](noise_feature_smoothing/paired_analysis/NOISE_HANDOVER.md)、[配对对照表](noise_feature_smoothing/paired_analysis/tables/) |
| 误差机制与边界诊断 | [特征偏移、类别流向与 RBF 响应](noise_feature_smoothing/diagnostics/DIAGNOSTIC_REPORT.md)、[后续问题排查](noise_feature_smoothing/problem_investigation/INVESTIGATION_REPORT.md)、[窗口时长、G1 和混合标签诊断](noise_feature_smoothing/g1_sampling_investigation/FINDINGS.md)、[分轮复核](noise_feature_smoothing/followup_rounds/FINDINGS.md) |
| 噪声结构和对照实验 | [OU 噪声、固定 gamma 与逐关联组排除](noise_feature_smoothing/supplementary_experiments/FINDINGS.md)、[完整补充报告](noise_feature_smoothing/supplementary_experiments/SUPPLEMENT_REPORT.md)、[汇总表](noise_feature_smoothing/supplementary_experiments/tables/summary.csv) |
| 文献身份与引用范围 | [核查报告](noise_feature_smoothing/literature_review/REVIEW_REPORT.md)、[来源登记](noise_feature_smoothing/literature_review/SOURCES.md) |
| 第 1、4、5 题及联合比较 | [采样、窗口与平滑协议](sampling_window_smoothing/README.md)、[结果报告](sampling_window_smoothing/results/REPORT.md)、[因子对照](sampling_window_smoothing/results/factor_contrasts.csv)、[在线回放指标](sampling_window_smoothing/results/replay_point_metrics.csv)、[配对区间](sampling_window_smoothing/results/paired_bootstrap.csv) |
| 实现修复与研究记录 | [噪声协议复核](noise_feature_smoothing/NOISE_PROTOCOL_REVIEW.md)、[实施修复](noise_feature_smoothing/IMPLEMENTATION_REPAIRS.md)、[AIGC 使用记录](../docs/aigc_log_part2.md) |

## 结果摘要

- 在固定划分和 clean-training/noisy-test 协议下，G3 未平滑的点级 Macro-F1 从无附加噪声时的 0.7788 降至 20 m 独立高斯扰动下的 0.1370。相同单向标准差的 OU 时间相关噪声结果为 0.3234。按关联用户组配对估计的差值区间跨过 0，剔除占比较大的关联组后，对照方向可能反转。
- 15 s 最小间隔下，匹配采样率训练的 120 s 窗候选相对于该档选中的 50 点配置，在线回放预测覆盖率由 51.0% 升至 94.2%，出分时点的切换 F1 由约 0.033 升至 0.124，回溯点级 Macro-F1 由约 0.799 降至 0.780。
- 噪声主实验使用窗口多数标签和 block 内 ±1 窗的一对一切换匹配。采样实验使用原始点标签的切换时刻和 ±120 s 匹配，并区分回溯对齐与出分回放。切换分数需按各自协议解读。

## 复现所需输入

1. 用 Part 1 共享的 `quality_features.zip` 恢复 `data/processed/quality_features/`，或从 GeoLife 原始轨迹按根目录 README 的 Part 1 流程生成。数据版本与用户划分可查 [主实验 manifest](noise_feature_smoothing/manifest.json)。
2. 参考 [Part 2 交接](../docs/part2_handover.md) 恢复 `src/model_training/windows_w50.csv`，优先使用配套的 `part2_results_93e50ac_full.zip`。Part 2 v2 参考预测和折划分沿用仓库中的结果与代码。
3. 使用 Python 3.12 和 [Part 2 的依赖清单](../src/model_training/requirements.txt)。主实验 manifest 记录核心依赖的实际版本，各分实验 README 给出入口和额外前置产物。

主噪声实验使用固定路径 `experiments/noise_feature_smoothing/`。从头复跑时，在单独的 checkout 中将历史实验目录移到存档位置，再用该目录中的 `run_experiment.py` 和 `supervise.py` 建立干净的同名目录。恢复上面列出的输入后，从仓库根目录调用 Windows 运行器 `python experiments/noise_feature_smoothing/supervise.py`，由它依次执行阶段 0 至 5。

补充噪声实验的入口为 `supplementary_experiments/run_supplement.py --stage all`，所需前置产物和复跑路径见 [补充实验说明](noise_feature_smoothing/supplementary_experiments/README.md)。采样实验的训练和报告脚本接收同一个独立 `--output` 路径。

```powershell
python experiments/sampling_window_smoothing/run.py --output experiments/sampling_window_smoothing/rerun
python experiments/sampling_window_smoothing/report.py --output experiments/sampling_window_smoothing/rerun
```

GitHub 收录报告、代码、汇总 CSV、小型复核产物、图表和运行记录。大体积窗口级缓存、模型、全量逐窗预测、逐事件明细及第三方资料快照保存在原实验目录或共享存档。历史验收、checkpoint 和哈希清单描述完整运行，部分记录包含原机器路径。需要继续计算时，先恢复脚本要求的产物，或使用全新的运行目录从头生成。
