# 原始 GPS：固定点数与固定时长的覆盖率匹配对照

2026-11-20。此项独立于 GPS 降采样条件，只使用原始采样数据，研究窗口方案在完整训练覆盖和共同后处理下的表现。作为前一轮结果之后设计的补充分析，复用既有测试用户，不作为新的独立确认性研究。

## 预先确定的设计

| 项目 | 两组共同设置 |
|---|---|
| 窗口 | 50 点；120 秒 |
| 训练数据 | 全部 51 用户、3,486,792 点；保留所有正长度训练窗，最低点数为 1 |
| 测试数据 | 相同的 13 用户、860,972 点、9 独立关联组 |
| 模型 | Nystroem RBF 1000 组件，LinearSVC C=1，balanced 类别权重规则，27 个参考特征 |
| 主要指标 | 未平滑的测试点级 macro-F1 |
| 次要指标 | 统一物理时间范围平滑后的 macro-F1；准确率、各类 F1、用户平均分数 |
| 平滑 | 每个原始 block 内，平均当前 GPS 点前后各 600 秒范围内所有观测点的类别分数；两组使用完全相同的点与时间范围 |
| 参数选择 | 不搜索参数，不根据本轮测试成绩改变窗口、最低点数或平滑时长 |
| 验证 | 用户分组五折，训练纯窗、验证盲窗；先完成两组 CV，再冻结记录并开始最终测试 |
| 不确定性 | 测试关联用户整组配对 bootstrap，5000 次，95% 逐项区间 |

原始 49 个有合格参考训练窗用户的折分配保持不变；原先因点数过少排除的 100/147 用户，按 `SHA256("42:" + group_id) mod 5` 补入固定折。如其关联组已有用户，则继承组内分配。该规则不查看测试成绩。

## 控制与保留的差异

- 两组训练点、测试点、用户、特征定义、模型配置和物理平滑范围相同。
- 逐点成员表与窗口类别计数、时间范围、唯一性均核对，验证每个点有且仅有一个窗口归属。
- 50 点方案沿用短尾并窗，120 秒方案保留部分尾窗；两种方案均不跨原 feature block。
- 训练窗口数仍不同：50 点为 70,441 窗，120 秒为 93,785 窗。窗口统计、短窗分布、每窗类别权重和训练样本单位是方案的一部分，不能据此宣称只改变了一个纯粹的“时长”因素。
- 最低训练点数从参考协议的 10 改为 1，以完整保留训练覆盖。短窗缺失特征由训练内中位数插补，结果应与前一轮参考协议分开报告。
- 平滑采用未来观测，是离线评估，不代表实时识别延迟。

## 复现入口

在仓库根目录、项目环境 `.venv-experiments` 下运行：

```powershell
./.venv-experiments/Scripts/python.exe src/model_training/window_controlled.py --stage prepare
./.venv-experiments/Scripts/python.exe src/model_training/window_controlled.py --stage cv --scheme points50
./.venv-experiments/Scripts/python.exe src/model_training/window_controlled.py --stage cv --scheme time120
./.venv-experiments/Scripts/python.exe src/model_training/window_controlled.py --stage test --scheme points50
./.venv-experiments/Scripts/python.exe src/model_training/window_controlled.py --stage test --scheme time120
./.venv-experiments/Scripts/python.exe src/model_training/summarize_window_controlled.py
```

prepare 只运行一次。训练折检查点自动恢复；源文件、输入和版本指纹保持不变时复用。输出独立保存在 `outputs/window_controlled_v1/`，不覆盖前一轮结果。主要记录为 protocol.json、coverage.csv、逐折 OOF 分数、CV/测试指标、模型、逐点预测和配对差值表。
