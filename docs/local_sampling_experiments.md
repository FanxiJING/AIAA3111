# 本地继续实验：GPS 降采样

更新：2026-11-20。用户确认的分工：**主实验为 GPS 降采样，之后比较固定点数与固定时长窗口**。Part 2 是复用的参考实现，不是本次负责的任务。

## 当前可验证的状态

- 已通过 GitHub 查询 main 和 part2-fixes。main 为 4e0f40e，part2-fixes 为 ab9f59a；本地跟踪后者。此前本地为浅克隆，仅跟踪 part2-fixes。
- GitHub 已提交 v2 结果，但未提交 quality_features 或窗口 CSV。仓库交接说明要求单独取得这些数据。
- 本地历史结果一致性审计：46 项通过。由已提交混淆矩阵重算 macro-F1=0.844269、accuracy=0.853080。该审计没有重新训练模型，不能当作新实验结果。
- 已建立项目独立环境 `.venv-experiments`，模型依赖按 Part 2 requirements.txt 安装（numpy 2.5.3、pandas 3.1.1、PyArrow 26.0.1、sklearn 1.9.1 等）；Python 为 3.13.5。所有本次条件使用同一环境，完整版本保存在 outputs/sampling_results_v1/environment.json。
- 预处理、降采样、窗口和冻结用户折共 16 项测试通过。
- **11 月 20 日逐点数据和参考窗口包已到位。** 实际路径为仓库同级 `quality_features/quality_features/`（69 个 Parquet，5 个空用户，64 个有效用户）。总计 4,347,764 点。
- 本机重建的原始采样窗口与队友窗口包完全一致：87,821 窗，训练 3,486,792 点、测试 860,972 点，27 个模型特征逐值相同。
- 四组固定点数和三组固定 120 秒窗口组合已完成真实训练及测试，16 项测试和最终 78 项一致性检查通过。30 秒采样加 120 秒窗口没有任何达到 10 点的训练窗，标为不适用。完整结果见 [实验结果](sampling_findings_20261120.md)。

## 主实验预设

| 项目 | 规则 |
|---|---|
| 输入 | Part 1 的 quality_features，64 个固定用户的逐点 Parquet |
| 条件 | 原始采样、目标最小间隔 5 / 10 / 30 秒；0 表示保留原始点 |
| 取样 | 每个原 block 保留首点，之后保留下一个距离上次保留点至少指定秒数的原始观测；不插值、不强行补末点 |
| 实际采样率 | 原始采样不规则，因此目标间隔不是精确频率；记录每用户实际间隔与保留率 |
| 特征 | 使用同一 motion_features.build_features，从保留坐标和时间重新计算；不直接抽取旧导数 |
| 边界 | 不跨原 feature_block；降采样后间隔超过 1200 秒时增加计算边界，避免跨长缺口求差分 |
| 主窗口 | W=50 点，沿用参考实现不足半窗尾部并入前窗的规则，实际可到 74 点 |
| 模型 | 固定参考 Nystroem SVM，1000 个组件、C=1、balanced 权重、soft voting；使用同一用户划分与环境 |
| 验证 | 训练窗口可按标签切纯段，验证和测试盲切；每个条件只用训练用户分组 CV 选择平滑 k |
| 报告 | 未平滑与平滑的点级 macro-F1、用户平均分数、类别 F1、实际窗口时长、保留点数、运行耗时 |

正式主实验在每个采样条件重新训练同一个参考模型，固定 C、类别权重与投票方式，只按同一训练 CV 规则选择 k。若研究“原始采样模型直接遇到稀疏 GPS”的部署退化，需要另列实验。

协议修订记录：11 月 20 日最初启动了沿用 Part 2 完整 C/权重/投票搜索的预实验。由于成本较高，且固定参考配置更直接地控制了模型因素，已在查看任何新的真实测试结果之前停止预实验，改用上述受控协议。预实验未产出正式测试指标，日志保留；正式输出目录使用 sampling_results_reference 与 sampling_duration_results_reference，不与预实验混合。

注意：不同条件的测试保留点不同。主表明确为**保留观测上的性能**，不能据此声称在相同全部原始 GPS 点上提升或下降。已冻结原始采样训练用户的五折映射，全部条件复用该映射。每条件额外保存测试点到窗口的映射，供训练结束后在共同保留点上计算配对 bootstrap；在这些步骤完成前，不写显著性结论。

## 本次执行入口

以下命令在仓库根目录运行。先核对真实数据的用户清单、来源和点数，再执行：

```powershell
./.venv-experiments/Scripts/python.exe src/model_training/prepare_sampling.py --points ../quality_features/quality_features --intervals 0 5 10 30 --out outputs/sampling_main_v1
```

每个条件独立保存 windows.csv（纯训练+盲测）、windows_blind.csv（训练 CV 盲切）、condition.json（输入 SHA256、规则及点数）、逐用户采样统计及盲切窗口时长分布。

受控实验入口使用已冻结的训练用户五折映射，逐折保存检查点：

数据准备目录已存在时直接复用；重新准备数据需指定新的 --out，避免覆盖已有条件。训练恢复使用 --resume。

```powershell
./.venv-experiments/Scripts/python.exe src/model_training/run_reference_sampling.py --stage cv --prepared-root outputs/sampling_main_v1 --out outputs/sampling_results_reference --threads 2 --resume
```

先完成全部训练 CV，冻结条件及比较规则后，再用相同参数运行 --stage final。不要根据测试表现改间隔、C、窗口或 k。--resume 会检查输入、版本和代码指纹是否一致。

## 后续：固定点数与固定时长

预设固定时长为 120 秒，配合同一组降采样条件；该值在新结果出现前确定。

```powershell
./.venv-experiments/Scripts/python.exe src/model_training/prepare_sampling.py --points ../quality_features/quality_features --intervals 0 5 10 30 --duration 120 --out outputs/sampling_duration_v1
```

固定时长窗从 block（纯训练时为标签 run）首个保留点起，使用左闭右开时间箱；不跨 block，不合并末尾不足时长的窗，空箱跳过。该规则与固定点数窗的尾部规则不同，需在方法部分说明。低采样率下，参考训练器的 n_points>=10 会排除更多训练窗，必须报告排除比例，不能把影响全归因于特征变差。

11 月 20 日训练前诊断：固定点数四组均有 49 个用户提供合格训练窗；固定时长的原始/5/10 秒组分别仅有 42/41/33 个用户提供合格训练窗。30 秒采样下 120 秒窗口最多 4 点，故无法沿用至少 10 点的训练协议，记为不适用，不填 0 分。完整统计在 outputs/sampling_results_v1/pretraining_sampling_diagnostics.csv。

## 面向 11 月 27 日海报的工作安排

- 11 月 19–20 日：恢复逐点数据、验证本机基线、冻结实验协议。
- 11 月 20–23 日：完成降采样主实验与统计，优先交付结果表。
- 11 月 23–25 日：固定点数/固定时长对比，完成必要诊断与共同支持集比较。
- 11 月 25–26 日：冻结结果、出图、检查海报中每个数字的来源。
- 11 月 27 日：完成海报设计。

原始输入与参考结果不覆盖。2026-11-24：新增代码、轻量结果及审计记录整合至默认分支，见 [团队结果包](../results/gps_window_experiments/README.md)。完整数据及大型运行输出仍保留在本地。
