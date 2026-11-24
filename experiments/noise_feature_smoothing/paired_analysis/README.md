# 噪声实验：配对证据与交接补全

本目录是 2026-11-23 对已完成实验的**事后补充分析**，不是新实验、调参或重新选择模型。只读原实验和诊断产物，所有新增文件均保存在本目录，不覆盖任何原结果。

## 范围

- 从保存的 48 份平滑预测表重建 144 组固定五类、点级混淆矩阵，与主结果逐项核对。
- 复用原 `paired_bootstrap_draws.npz` 的 5000×9 抽样次数矩阵及其用户组顺序，不创建新随机数或种子。
- 报告 G3 的 S1/S2 相对 S0 的配对差，包括 Macro-F1、Accuracy、边界 Precision/Recall/F1、错误孤立跳变率和相对各自 clean 的标签变化比例。
- 对 Macro-F1、Accuracy 补充 `(平滑-raw)_noisy - (平滑-raw)_clean`，不能用单独的区间是否重叠代替配对比较。
- 对 G1/G2/G3 raw 报告绝对分数差及相对自身 clean 变化的差，两者不可混为“抗噪排名”。
- 用窗口内真实类别点数核算平滑“纠正点数”“破坏点数”和净 Accuracy 变化；同一个混合窗口中可能同时发生纠正和破坏。

所有噪声条件、特征组、原划分、模型和 k 不变。G4 仍跳过。边界仍使用原先 class-blind、±1 窗口、block 内一对一最大匹配的计数，不在这里重新匹配。seed SD 是扰动波动，不是用户置信区间。新增比较均属描述性补充，无多重比较校正，不作显著性排名。

## 重现

在仓库根目录运行：

```powershell
.venv-audit/Scripts/python.exe -B -m unittest discover -s experiments/noise_feature_smoothing/paired_analysis -p "test_*.py" -v
.venv-audit/Scripts/python.exe -B experiments/noise_feature_smoothing/paired_analysis/build_analysis.py
.venv-audit/Scripts/python.exe -B experiments/noise_feature_smoothing/paired_analysis/verify_analysis.py
```

脚本先保存输入哈希，再分析，最后核对所有原始输入、代码、主实验与旧诊断文件未变。重新运行只覆盖本目录的派生产物。中途失败不写“通过”记录。

## 产物

- `NOISE_HANDOVER.md`：中文结果、证据边界、汇报建议和复现入口。
- `tables/paired_smoothing_contrasts.csv`：G3 平滑相对 raw 的配对变化与区间。
- `tables/paired_smoothing_interactions.csv`：平滑收益随噪声变化的配对差中差。
- `tables/paired_feature_contrasts.csv`：raw 特征组直接比较及退化差异。
- `tables/smoothing_point_flows.csv`：逐条件纠正、破坏、标签变化及预测 train 占比。
- `tables/smoothing_point_flows_summary.csv`：种子均值和样本 SD，点数均值可为非整数。
- `tables/paired_summary_reproduction.csv`：36 行主 bootstrap 汇总复核。
- `tables/linked_group_sizes.csv`：独立抽样组的用户与样本数。
- `input_manifest.json`、`acceptance.json`、`output_inventory.json`：来源、检查范围、输入不变性及新产物哈希。
- `delivery_verification.json`：独立复核 96 组纠正/破坏，以及 clean 和 20m 首个固定种子下全部平滑的 54 组边界计数；边界复核使用独立的 Hungarian assignment，只核对最大匹配数量，不改主匹配规则或产物。

不新增图；原六张诊断图仍存在未完成的人工视觉复核限制。本目录不把自动像素/布局检查冒充目视检查。
