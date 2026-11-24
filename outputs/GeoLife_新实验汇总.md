# GeoLife 出行方式分类补充实验汇总

本报告汇总接续 Part 2 v2 的五组工作。前三组比较逐点与窗口、BiGRU 前后文及平滑，两组新增工作验证 robust_g3 并交叉比较逐点表示与分类器。

## 数据与评估

- 使用 Part 1 的原始采样逐点质量特征。任务包含步行、骑车、公交、汽车、火车五类。固定划分有 51 位训练用户和 13 位测试用户。所有最终结果均在同一批 **860,972 个测试点**上计分，共有 **658 次轨迹块内真实切换**。
- 一个预测切换是同一轨迹块内相邻预测类别发生变化，真实切换同样从逐点真实标签得到。在 ±30、±60、±120 秒内分别对预测与真实切换做一对一时间匹配。匹配只检验是否发生类别变化。令 M 为命中数，P 为预测切换数，T 为真实切换数，则精确率为 M/P，召回率为 M/T，切换 F1 为 2M/(P+T)。
- **离线边界时刻**取新类别的第一个点。固定窗的全窗标签对齐到窗首。**窗尾出分回放**把固定窗的边界预测记到该窗最后一个点，近似衡量何时能拿到预测。BiGRU 和居中平滑使用未来点，只报告离线边界结果。

## 14 种方案的同口径对比

切换指标统一用离线边界、±60 秒、一对一匹配。重复出现的参考行只列一次。

| 方案 | 点级准确率 | 点级 Macro-F1 | 命中 / 预测切换 | 切换精确率 | 切换召回率 | 切换 F1 |
|---|---:|---:|---:|---:|---:|---:|
| 逐点 SVM，原版抽样 | 64.41% | 0.5507 | 622 / 118,963 | 0.52% | 94.53% | 0.0104 |
| 25 点窗 SVM | 77.48% | 0.7373 | 402 / 6,807 | 5.91% | 61.09% | 0.1077 |
| 50 点窗 SVM | 80.89% | 0.7789 | 289 / 3,014 | 9.59% | 43.92% | 0.1574 |
| 50 点窗 SVM＋11 窗软投票 | 85.31% | 0.8443 | 52 / 473 | 10.99% | 7.90% | 0.0920 |
| robust_g3（50 点窗 HGB） | 79.79% | 0.7663 | 287 / 3,362 | 8.54% | 43.62% | 0.1428 |
| 逐点神经网络，每点重置状态 | 51.51% | 0.3542 | 580 / 94,612 | 0.61% | 88.15% | 0.0122 |
| 原 BiGRU＋线性头 | 76.60% | 0.6838 | 333 / 18,224 | 1.83% | 50.61% | 0.0353 |
| BiGRU＋合并 ≤5 秒 A→B→A 段 | 76.79% | 0.6854 | 312 / 13,346 | 2.34% | 47.42% | 0.0446 |
| BiGRU＋前后各 120 秒概率平均 | 78.97% | 0.7034 | 210 / 1,614 | 13.01% | 31.91% | 0.1849 |
| 匹配抽样逐点特征＋Nyström SVM | 63.78% | 0.5359 | 630 / 115,714 | 0.54% | 95.74% | 0.0108 |
| 匹配抽样逐点特征＋随机森林 | 61.86% | 0.5449 | 625 / 196,281 | 0.32% | 94.98% | 0.0063 |
| BiGRU 向量＋Nyström SVM | 78.87% | 0.7272 | 351 / 21,249 | 1.65% | 53.34% | 0.0320 |
| BiGRU 向量＋随机森林 | 78.51% | 0.7246 | 328 / 16,962 | 1.93% | 49.85% | 0.0372 |
| BiGRU 向量＋Nyström SVM＋前后各 60 秒平滑 | 80.10% | 0.7386 | 266 / 2,546 | 10.45% | 40.43% | 0.1660 |

点级 Macro-F1 最高的是 50 点窗 SVM＋软投票（0.8443）。切换 F1 最高的是原 BiGRU＋120 秒概率平均（0.1849）。新方案 BiGRU 向量＋SVM＋60 秒平滑多命中 56 次切换（266 对 210），点级 Macro-F1 也更高（0.7386 对 0.7034），切换 F1 为 0.1660。

原版逐点 SVM 和匹配抽样逐点 SVM 用的拟合点不同。实验五内部使用同一批拟合点。

[指标定义与窗尾回放](GeoLife_方案对比与指标说明.md) · [加入 robust_g3 后的九方案记录](GeoLife_九种方案对比.md)

## 实验一｜逐点、25 点窗、50 点窗 SVM

[原始报告](point_window_comparison/REPORT.md) · [完整结果](point_window_comparison/results.csv) · [逐用户结果](point_window_comparison/per_user.csv) · [复现脚本](point_window_comparison/run.py)

**方法与算法。** 三组使用 Nyström 近似 RBF SVM（1,000 个核分量，C=1，类别权重平衡，种子 42），全部关闭时序平滑。逐点组按标签分层抽取 70,000 个训练点，输入距离、速度、加速度、jerk、转向角和转向角速度，每点独立输出。特征本身由当前点和历史点计算。

窗口组使用 27 个统计特征，训练时在真实类别变化处断开，窗口至少保留 10 点。测试时按连续轨迹块盲切，尾窗不足半窗时并入前窗。每窗预测一个类别，窗内的点继承该类别。25 点组用 139,510 个训练窗口，50 点组复用以 70,086 个训练窗口拟合的 v2 预测，并核对窗口边界和点数。实验比较完整预测方案，逐点和窗口组的特征表示与训练样本构造也有区别。

**±60 秒，按窗尾出分回放。** 逐点预测用当前点时间计分。

| 模型 | 点级 Macro-F1 | 切换精确率 | 切换召回率 | 切换 F1 | 预测切换数 |
|---|---:|---:|---:|---:|---:|
| 逐点 SVM | 0.5507 | 0.52% | 94.53% | 0.0104 | 118,963 |
| 25 点窗 SVM | 0.7373 | 5.08% | 52.58% | 0.0927 | 6,807 |
| 50 点窗 SVM | 0.7789 | 5.77% | 26.44% | 0.0948 | 3,014 |

25 点组比 50 点组多匹配 172 次真实切换，点级 Macro-F1 低 0.0417，切换 F1 接近。切换 F1 的排序随容差变化。逐点 SVM 在三档容差下均产生大量误报。

## 实验二｜逐点 BiGRU 与同结构逐点对照

[原始报告](point_bigru/REPORT.md) · [完整结果](point_bigru/results.csv) · [逐用户结果](point_bigru/per_user.csv) · [分类别结果](point_bigru/per_class.csv) · [训练记录](point_bigru/training.json)

**方法与算法。** 模型输入一整段连续轨迹块，用两个单向 GRU 分别按正序和逆序读取同一组六维运动特征。每个方向隐藏维度为 24。每个点的正反向状态拼接后过线性分类层，输出五类分数。相邻预测标签变化即预测切换。对照采用同一结构，把两个方向都改为长度为 1 的序列，每点从零状态开始。

沿用原有测试用户，从训练用户中划出 10 位验证用户，把有重复轨迹关联的用户留在同组。两网络使用相同的 578 个完整轨迹块、385,563 个拟合点，来自其余 41 位训练用户，按每用户约八分之一的完整块抽取。特征做保留符号的 log1p 变换，标准化参数只从拟合子集估计，缺失值映射到标准化后的 0。训练使用 Adam（学习率 0.003）和加权逐点交叉熵，权重按类别频次的平方根倒数确定。最多训练 8 轮，按验证点级 Macro-F1 各自选最佳轮次，BiGRU 第 8 轮，逐点重置对照第 7 轮。

**±60 秒，离线边界时刻。**

| 模型 | 点级 Macro-F1 | 切换精确率 | 切换召回率 | 切换 F1 | 命中 / 预测切换 |
|---|---:|---:|---:|---:|---:|
| 每点重置状态 | 0.3542 | 0.61% | 88.15% | 0.0122 | 580 / 94,612 |
| BiGRU，整块前后文 | 0.6838 | 1.83% | 50.61% | 0.0353 | 333 / 18,224 |
| 25 点窗 SVM 原始输出 | 0.7373 | 5.91% | 61.09% | 0.1077 | 402 / 6,807 |
| 50 点窗 SVM 原始输出 | 0.7789 | 9.59% | 43.92% | 0.1574 | 289 / 3,014 |

前后文提高了点级 Macro-F1，并减少了预测切换数，召回也下降。相对每点重置对照，点级 Macro-F1 差值为 +0.3296，配对 95% 区间为 [+0.0734, +0.3433]。切换 F1 差值为 +0.0231，区间为 [+0.0023, +0.0241]。区间按 9 个测试用户组做 5,000 次配对重抽样，反映固定模型在该测试划分上的用户组差异。

**误差分析。** BiGRU 的 18,224 次预测切换里，有 17,059 次距同一轨迹块内最近的真实切换超过 60 秒，或该块没有真实切换。这里面 11,685 次是步行与骑车之间的预测切换。训练子集只有 179 次真实切换。逐点交叉熵对短暂错分仅按错点计损失，训练也只按点级分数选模型。验证 Macro-F1 最后一轮仍在上升，后续训练能否改善切换检测还需要验证。

## 实验三｜逐点 BiGRU 加时序平滑

[原始报告](point_bigru_smoothing/REPORT.md) · [验证扫描](point_bigru_smoothing/validation.csv) · [选参记录](point_bigru_smoothing/selection.json) · [测试结果](point_bigru_smoothing/test.csv) · [50 点窗参考结果](point_bigru_smoothing/w50_reference.csv)

**方法与算法。** 复用实验二的 BiGRU 检查点，逐点输出五类概率。对每个目标点，在同一轨迹块内求前后各 r 秒范围的类别概率均值，再对该点取最高分类别。用二分查找定位邻域端点，用累计和计算均值。每个点保留自己的预测时刻。另一个轻量对照只合并标签形如 A→B→A 的短段，要求中间 B 段时长不超过阈值，且两侧 A 段都超过阈值。

在实验二的 10 位验证用户上比较概率平均半径 5、10、20、30、60、120 秒，以及短段合并阈值 3、5、10、20、30、60 秒。每个方法选 ±60 秒切换 F1 最高、且点级 Macro-F1 较未经处理的版本下降不超过 0.01 的候选，要求切换 F1 高于未经处理的版本。选中了前后各 120 秒概率平均及 5 秒短段合并。确定参数后再评估固定测试集。120 秒是本次概率平均候选的上限。

**±60 秒，离线边界时刻。**

| 模型 | 点级 Macro-F1 | 切换精确率 | 切换召回率 | 切换 F1 | 命中 / 预测切换 |
|---|---:|---:|---:|---:|---:|
| BiGRU 原始输出 | 0.6838 | 1.83% | 50.61% | 0.0353 | 333 / 18,224 |
| BiGRU 合并 ≤5 秒 A→B→A 段 | 0.6854 | 2.34% | 47.42% | 0.0446 | 312 / 13,346 |
| BiGRU 前后各 120 秒概率平均 | 0.7034 | 13.01% | 31.91% | 0.1849 | 210 / 1,614 |
| 50 点窗 SVM 原始输出 | 0.7789 | 9.59% | 43.92% | 0.1574 | 289 / 3,014 |
| 50 点窗 SVM 加 11 窗软投票 | 0.8443 | 10.99% | 7.90% | 0.0920 | 52 / 473 |

在同一 BiGRU 上，概率平滑将未匹配的预测切换从 17,891 次降到 1,404 次，命中则从 333 次降到 210 次。固定测试划分上，它的切换 F1 高于表中的两个 50 点窗参考配置，点级 Macro-F1 则较低。各模型使用各自的训练方案，且测试划分此前已用于分析，这些结果按探索性对照解读。居中概率平均和 BiGRU 都使用未来数据，表中指标衡量离线边界定位。

## 实验四｜robust_g3 的固定测试集验证

[对比结果](robust_g3_validation/comparison_row.csv) · [分类指标](robust_g3_validation/metrics.json) · [混淆矩阵](robust_g3_validation/confusion_matrix.csv) · [逐用户结果](robust_g3_validation/per_user.csv) · [分类别结果](robust_g3_validation/per_class.csv)

**方法与算法。** 复用收到的 50 点窗 HGB 模型。推理按轨迹块盲切 50 点窗，从窗内提取不同步长的速度、加速度和转向统计、10/30 秒跨度的速度与质心位移速度。推理使用模型包内声明的特征列，最终窗口标签赋给窗内各点。在同一测试集上评价分类及 ±60 秒离线切换。

**结果。** 点级准确率 79.79%，Macro-F1 0.7663，切换精确率 8.54%，召回率 43.62%，F1 为 0.1428。命中 287 次，预测切换总数 3,362 次。相较同为 50 点窗的原始 Nyström SVM，点级 Macro-F1 低 0.0126，切换 F1 低 0.0146。训练用户、训练策略和降采样及噪声下的表现仍未核实。

**路径。** [特征提取](robust_g3_validation/features.py) · [推理入口](robust_g3_validation/predict.py) · [分类评估](robust_g3_validation/evaluate.py) · [切换评估](robust_g3_validation/score_transitions.py)。本地预测在 outputs/robust_g3_validation/test_predictions.parquet。重新推理需要提供方的 robust_g3.joblib。metrics.json 记录原模型的 SHA-256。

## 实验五｜逐点表示 × Nyström SVM / 随机森林

[原始报告](point_bigru_classifiers/REPORT.md) · [测试结果](point_bigru_classifiers/test.csv) · [逐用户结果](point_bigru_classifiers/test_per_user.csv) · [验证扫描](point_bigru_classifiers/validation.csv) · [选参记录](point_bigru_classifiers/selection.json)

**方法与算法。** 比较原始六维逐点运动特征和冻结 BiGRU 的 48 维逐点向量。后者拼接分类前的正反两个方向隐藏状态。四个新分类器用同一批 70,000 个拟合点，按类别分层抽样。SVM 使用 1,000 个 Nyström 分量、C=1、平衡类别权重。随机森林用 200 棵树、最小叶样本数 5、balanced_subsample 类别权重。同一分类器的参数跨表示固定。原线性头作为参照。

主对照关闭平滑。对两个 BiGRU 向量分类头，在 10 位验证用户上扫描 5、10、20、30、60、120 秒的居中平滑。SVM 平均决策分数，随机森林平均类别概率。在切换 F1 高于自身未平滑版本、点级 Macro-F1 降幅不超过 0.01 的配置中选最佳。验证选中 SVM＋前后各 60 秒平滑。

**结果。** 换用 BiGRU 向量后，SVM 点级 Macro-F1 从 0.5359 升到 0.7272，随机森林从 0.5449 升到 0.7246。两个新头未平滑切换 F1 分别为 0.0320 和 0.0372，与原线性头的 0.0353 接近。选中的平滑方案测试点级 Macro-F1 为 0.7386，切换召回 40.43%，切换 F1 为 0.1660。编码器此前在 385,563 个点上做过监督训练，因此表示间差异包含上下文和监督特征学习的共同作用。

**路径。** [向量提取](point_bigru_classifiers/encode.py) · [训练和评估](point_bigru_classifiers/run.py) · [用户划分、抽样和检查点哈希](point_bigru_classifiers/features.json)。四个分类头权重在 outputs/point_bigru_classifiers/models/{raw_svm,raw_rf,hidden_svm,hidden_rf}.joblib。本地复用权重时可使用 --reuse-models。

## 文件路径与复现

项目根目录为 **C:/Users/whliu/Desktop/AIAA3111**，记录对应工作区的 Part 2 代码版本为 **ab9f59a**。ZIP 从项目根目录解压后，文件会落在下列原路径。ZIP 可以独立查阅报告和 CSV，执行脚本需本地项目与数据。

| 项目根目录下的路径 | 内容 |
|---|---|
| outputs/GeoLife_新实验汇总.md | 本汇总 |
| outputs/point_window_comparison/ | 原始报告、run.py、results.csv、per_user.csv |
| outputs/point_bigru/ | prepare.py、train.py、score.py，报告、训练配置、两份检查点、三张结果表 |
| outputs/point_bigru/predictions/{uid}.npz | 13 位测试用户的逐点预测、真实类别索引与秒级时间戳 |
| outputs/point_bigru_smoothing/ | run.py、w50_reference.py、报告、验证与测试结果、选参记录、逐用户测试结果 |
| outputs/robust_g3_validation/ | 推理及评分代码、分类与切换指标、模型哈希 |
| outputs/point_bigru_classifiers/ | 2×2 报告、训练及评估代码、验证与测试结果、选参和抽样记录 |
| outputs/point_bigru_classifiers/{features,models}/ | 本地派生特征及已训练分类头权重 |
| data/processed/quality_features/{uid}.parquet | Part 1 逐点质量特征，复现所需的项目数据 |
| data/splits/{train_users.json,test_users.json,report.json,manifest.csv} | 固定用户划分、关联用户组和样本清单 |
| src/model_training/windows_w50.csv | W=50 训练窗口表，实验一读取此表计训练窗口数 |
| src/model_training/results/basic_w50_no_dt_v2/window_predictions_test.csv | v2 W=50 未平滑和已平滑的保存预测 |
| outputs/point_bigru/cache/{uid}.npz | 逐点特征缓存，由 prepare.py 生成 |

压缩包收录报告、代码、指标、配置、两份 BiGRU 检查点和 BiGRU 的逐点测试预测。原始轨迹、Part 1 Parquet、训练窗口表、派生特征缓存、robust_g3 的预测 Parquet 和 2×2 的四个 joblib 权重沿用项目本地文件。robust_g3 原模型由提供方交付。

从项目根目录运行。标准环境为 Python 3.12.6、NumPy 2.5.3、pandas 3.1.1、PyArrow 26.0.1、scikit-learn 1.9.1。GPU 环境为 Python 3.12.13、PyTorch 2.13.0+cu130、NumPy 2.5.1、pandas 2.3.3、scikit-learn 1.9.0。按需使用现有检查点和预测跳过重新训练。

```powershell
# 实验一
python outputs/point_window_comparison/run.py

# 实验二
python outputs/point_bigru/prepare.py
& 'C:/Users/whliu/miniconda3/envs/py_ai/python.exe' outputs/point_bigru/train.py
python outputs/point_bigru/score.py

# 实验三，可复用压缩包里的 BiGRU 检查点和测试预测
& 'C:/Users/whliu/miniconda3/envs/py_ai/python.exe' outputs/point_bigru_smoothing/run.py
python outputs/point_bigru_smoothing/w50_reference.py

# 实验四，已有预测时直接评分
python outputs/robust_g3_validation/score_transitions.py

# 实验五
& 'C:/Users/whliu/miniconda3/envs/py_ai/python.exe' outputs/point_bigru_classifiers/encode.py
python outputs/point_bigru_classifiers/run.py
```
