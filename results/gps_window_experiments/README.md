# GPS 降采样与窗口对比：团队结果包

2026-11-24 整合至默认分支 `main`。本目录提供可直接用于讨论和海报的结果表、PNG/SVG 图、冻结折分配和验证记录。原始 GPS、窗口明细、大型模型和逐点预测不在本结果包中，完整运行输出保留在实验者本地 `outputs/`。

## 两项实验

| 实验 | 方法与结果说明 | 本目录文件 |
|---|---|---|
| GPS 降采样主实验 | [11 月 20 日结果](../../docs/sampling_findings_20261120.md) | `gps_downsampling/` |
| 第一轮跨采样条件的窗口比较 | 同上；训练覆盖不同，不能当作纯窗口时长效应 | `initial_window_comparison/` |
| 原始 GPS、完整覆盖的窗口对照 | [11 月 24 日结果](../../docs/window_comparison_findings_20261124.md) | `controlled_window_comparison/` |

GPS 主实验：各自保留点上的平滑 macro-F1 为原始 0.8442、5 秒 0.8146、10 秒 0.7991、30 秒 0.7424；比较的点集合不同。四条件共同保留 46,714 点，平滑后所有两两配对差值区间均包含 0。

完整覆盖窗口对照：50 点与 120 秒未平滑 macro-F1 为 0.7824 / 0.7815，统一 20 分钟平滑后为 0.8501 / 0.8518；窗口差值区间均包含 0。没有检测到明确优劣，不作等效性结论。

只有 9 个独立测试关联组，一个组占约 79% 的点。补充窗口对照复用已查看的测试用户，属于探索性分析。共同 20 分钟平滑使用未来观测，属于离线方法。

## 建议直接使用的海报图

- [GPS 观测保留比例与性能](gps_downsampling/sampling_retained_support.png)：各条件在各自保留观测上评估。
- [GPS 相同观测点比较](gps_downsampling/sampling_common_support.png)：四条件共同点上的比较。
- [完整覆盖窗口方案比较](controlled_window_comparison/window_controlled_comparison.png)：包括配对差值区间。

每张图旁均有同名 SVG；结果值以 CSV 为准。`artifact_manifest.json` 记录发布时复制的原始文件、大小和 SHA256。历史 protocol/environment 中的绝对路径是原运行机的审计记录，不是新机器的目录要求。

## 代码与复现

代码在 `src/model_training/`；[主实验协议](../../docs/local_sampling_experiments.md)和[窗口协议](../../docs/window_comparison_protocol_20261120.md)提供运行入口。`run_reference_sampling.py` 是正式降采样入口；`run_sampling_experiments.py` 是保留的全搜索预实验程序，其输出不属于正式成绩。

实验使用独立 Python 3.13.5 环境及 Part 2 的 `src/model_training/requirements.txt`。项目根目录的 pyproject/uv.lock 是原预处理环境，约束不同；本次未改动它们。实际实验环境快照见 `gps_downsampling/environment.json`。

在全新检出中，从队友处恢复逐点 Parquet，放到仓库同级 `quality_features/quality_features/`，或在准备脚本中指定 `--points`。窗口独立对照的准备入口沿用这一同级路径。需要先按协议重新生成 `outputs/sampling_main_v1/` 与 `outputs/sampling_duration_v1/`。

正式训练复用的 49 用户折分配在本目录 `frozen_training_folds.json`。从仓库根目录运行以下 PowerShell 命令恢复脚本默认位置；已有文件会先核对内容，不会覆盖不同折分配：

```powershell
$foldSource = 'results/gps_window_experiments/frozen_training_folds.json'
$foldTarget = 'outputs/sampling_results_v1/frozen_training_folds.json'
New-Item -ItemType Directory -Force 'outputs/sampling_results_v1' | Out-Null
if (Test-Path -LiteralPath $foldTarget) {
    if ((Get-FileHash -LiteralPath $foldSource).Hash -ne (Get-FileHash -LiteralPath $foldTarget).Hash) {
        throw '已有折分配与发布记录不同，请使用独立输出目录。'
    }
} else {
    Copy-Item -LiteralPath $foldSource -Destination $foldTarget
}
```

之后按协议先完成各组 CV、冻结选择，再进行最终测试。不要把本目录归档的 protocol.json 直接当作新机器的恢复检查点；新运行应生成自己的路径和指纹。摘要脚本需要完整运行输出，本目录的轻量表格不能代替逐点预测。

18 项测试运行方式（先激活独立实验环境）：

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m unittest discover -s tests -v
```

GPS 正式结果 78/78 项重建检查通过；完整覆盖窗口结果 30/30 项核验通过。检查的是实验产物一致性，不代表消除了数据代表性和研究设计局限。
