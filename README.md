# GPS 轨迹逐点出行方式识别

本项目使用 GeoLife GPS 轨迹识别每个样本点的出行方式，允许同一 trip 内包含多种类别。
当前已完成逐点标签匹配、基础清洗、trip 划分与第一批运动特征构造（Part 1），以及基础模型训练、时序平滑、分段输出与评估（Part 2）；
窗口大小、采样率、噪声和特征组实验留给 Part 3/4。

## 项目状态与交接

| 阶段 | 当前状态 | 交接说明 |
|---|---|---|
| 数据预处理 | 已完成当前版本 | 成功匹配标签后清洗，按时间差超过 20 分钟划分 trip；未做 Stay/Move 过滤 |
| 特征工程 | 第一批已完成 | 10 个数值运动特征与 10 个有效性标记；定义见 [FEATURES.md](FEATURES.md) |
| 异常运动处理 | 已实施连接阻断 | 来源文件变化或速度 >100 m/s 时重启 feature_block；重算特征，不删点、不补值；剩余高加速度和 jerk 仅标记 |
| train/test 划分 | 已完成 | 51 个训练用户、13 个测试用户；重复记录关联用户绑定；validation 由训练用户内部另分 |
| 类别筛选 | 已完成 | 当前交付 walk、bus、bike、train、car 五类；其他类别从交付样本中移除 |
| 窗口构造与窗口特征 | 已完成（Part 2） | 每个 feature_block 切成 50 点窗口，不跨 feature_block_id；测试窗口盲切；27 个窗口特征进入模型 |
| 模型训练与评估 | 已完成（Part 2 v2） | 8 个模型（含全量数据的 Nystroem SVM）；盲切验证窗口、5 折用户分组 CV；在完整流程（模型 + 平滑）上选模型；选中 SVM (RBF, Nystroem, all data)；未平滑测试点级 macro-F1 0.779（v1：SVM (RBF) 0.760） |
| 时序平滑 | 已完成（Part 2 v2） | block 内 k 窗口投票；每个模型的 k、软/硬投票与类别权重仅由训练用户 CV 选定；选中模型软投票 k = 11；测试点级 macro-F1 0.844，平滑增益 +0.065（95% CI 0.032–0.135）（v1：多数投票 k = 9，0.821） |
| 分段输出 | 已完成（Part 2） | 同一 block 内相邻同类窗口合并为单一出行方式子序列；输出 segments_test.csv |

**Part 2 当前结果与文档（v2）**：最新数字、结论与局限见 [docs/part2_v2_findings.md](docs/part2_v2_findings.md)；
给 Part 1 与 Part 3/4 的注意事项（采样率经窗口时长残留、疑似标签噪声、过小用户）见 [docs/part2_notes_for_teammates.md](docs/part2_notes_for_teammates.md)；
代码为 `src/model_training/pipeline_v2.py`，结果在 `src/model_training/results/basic_w50_no_dt_v2/`（不提交版本管理）。

接手数据入口为 `data/processed/quality_features/`。feature_points 和 five_class_features 作为处理前中间版本保留。
有效性标记表示数值可计算，不能保证所有噪声已消除。训练窗口不得跨 feature_block_id。
原始数据保存在 `data/original/`。

## 环境与复现

使用 Python 3.12 和 uv 管理环境。

```powershell
uv sync
uv run preprocess-labels
uv run preprocess-trips
uv run build-features
uv run filter-modes
uv run build-features --input data/processed/five_class_points --output data/processed/five_class_features
uv run quality-features
uv run split-dataset
uv run python -m unittest discover -s tests
```

## 当前目录

- `data/original/<user_id>/`：原始 Trajectory/*.plt 和 labels.txt，保持内容不变。
- `data/processed/labeled_points/`：每个用户一个 Parquet 文件及 report.json。
- `data/processed/cleaned_points/`：清洗及分配 trip_id 后的逐点数据与统计报告，供后续特征工程使用。
- `src/data_preprocessing/`：轨迹读取、标签匹配和后续清洗；现有代码位于 `label_matching.py`。
- `src/feature_enginneering/`：逐点运动特征工程，实现在 `motion_features.py`。
- `data/processed/feature_points/`：包含原字段、10 个运动特征及有效性标记的逐点数据。
- `data/processed/five_class_points/`：从 cleaned_points 筛出的五类样本与类别筛选报告。
- `data/processed/five_class_features/`：五类样本重新计算后的运动特征，异常连接处理前中间版本。
- `data/processed/quality_features/`：连接质量处理后重新计算的特征，当前交接入口。
- `data/splits/`：固定的 train/test 用户清单、manifest.csv 和划分报告；这些小文件应提交版本管理。
- `tests/`：标签匹配边界和重叠情况验证。
- `src/model_training/`：Part 2 窗口构造、模型训练、时序平滑、分段输出与作图代码；说明见该目录 README.md。
- `src/model_training/windows_w50*.csv`、`src/model_training/results/`：Part 2 生成的窗口表与结果，不提交版本管理。
- `docs/`：Part 2 v2 结果与结论（part2_v2_findings.md）、给队友的注意事项（part2_notes_for_teammates.md）、交接说明（part2_handover.md，按 v1 编写）、报告草稿（part2_report_draft.md，按 v1 数字编写）与 AIGC 使用记录（aigc_log_part2.md）。

`uv.lock` 提交版本管理；原始/生成数据、缓存和 `.venv` 不提交。

## 数据预处理

以下点数以当前本地数据和各阶段 `report.json` 为准。“样本点”指 GPS 记录行，不是 trip 数、窗口数或特征有效值数量。
原始输入包含 69 个用户、10,906 个轨迹文件、12,517,364 个点；成功匹配样本来自其中 64 个用户。
所有阶段均保留原始数据；处理后样本减少不等于原始文件被删除。

### 处理前后点数总账

| 顺序 | 处理步骤 | 处理前总点数 | 处理后总点数 | 减少点数 | 统计依据与说明 |
|---|---|---:|---:|---:|---|
| 0 | 原始用户目录迁移到 data/original | 12,517,364 | 12,517,364 | 0 | 仅移动目录；原始文件独立行数统计见 raw_point_counts.json |
| 1 | 读取轨迹、逐点匹配标签并保留 matched | 12,517,364 | 5,396,631 | 7,120,733 | labeled_points/report.json；未匹配 7,076,748 点、标签冲突 43,985 点；无无效时间点 |
| 2.1 | 删除无效时间、非法坐标或无有效标签的点 | 5,396,631 | 5,396,631 | 0 | cleaned_points/report.json：invalid_points_removed = 0 |
| 2.2 | 时间及来源稳定排序 | 5,396,631 | 5,396,631 | 0 | 只改变顺序 |
| 2.3 | 剔除同时间戳不同标签的记录 | 5,396,631 | 5,396,631 | 0 | conflicting_timestamp_points_removed = 0 |
| 2.4 | 删除重复观测 | 5,396,631 | 5,301,704 | 94,927 | duplicate_observations_removed = 94,927 |
| 2.5 | 同秒多条观测选取代表点 | 5,301,704 | 4,823,144 | 478,560 | same_timestamp_observations_merged = 478,560 |
| 2.6 | 海拔缺失哨兵值及非有限值转为空 | 4,823,144 | 4,823,144 | 0 | 修改海拔值，不删除点 |
| 2.7 | 按大于 1200 秒的间隔建立 trip | 4,823,144 | 4,823,144 | 0 | 得到 6,458 个 trip；保留单点 trip |
| 3 | 构造第一批逐点运动特征及有效性标记 | 4,823,144 | 4,823,144 | 0 | feature_points/report.json；缺失特征不导致整行删除 |
| 4 | 保留 walk、bus、bike、train、car 五类 | 4,823,144 | 4,347,764 | 475,380 | 从 cleaned_points 筛选，输出 five_class_points/report.json |
| 5 | 按类别筛选缺口重算五类运动特征 | 4,347,764 | 4,347,764 | 0 | five_class_features/report.json；不跨 feature_block_id 计算 |
| 6 | 阻断跨文件及速度 >100 m/s 连接并重算 | 4,347,764 | 4,347,764 | 0 | quality_features/report.json；减少有效连接与差分数量，不删除样本点 |
| 7 | 按用户分配 train/test | 4,347,764 | 4,347,764 | 0 | train 3,486,792 + test 860,972；不复制或删除数据点 |

步骤 2 整体从 5,396,631 点降至 4,823,144 点，共减少 573,487 点。
截至当前，相比原始输入共减少 8,169,600 点。步骤 2 的中间点数按代码执行顺序和报告中的移除数量逐步核算，
不代表每个子步骤都另存了一份数据。步骤 1 的未匹配和冲突是同一次筛选中的不同原因，不虚构先后处理顺序。

### 步骤 1：读取与逐点标签匹配

在项目根目录执行 `uv run preprocess-labels`。可用 `--input`、`--output` 指定其他路径。
程序逐文件读取，按用户写出 Zstandard 压缩的 Parquet；重新运行会替换同名用户输出及报告。
发生解析错误时中止，不静默跳过文件。每个用户完成后才将临时输出替换为最终文件。

当前匹配标签后，仅保存 `label_status == matched` 的点；原始数据不修改。
不筛选类别、不生成特征、不划分 train/test。没有成功匹配点的用户保留带完整字段的空 Parquet 文件。
保留原始文件中的顺序；不同文件之间不推定连续性。所有 11 种原始类别保持原名。
轨迹时间与标签时间按文件记录直接匹配，不擅自转换时区。

标签区间采用闭区间 `start <= timestamp <= end`。多个同类区间覆盖同一点时标签仍然有效，
不同类区间重叠时标记为 conflict，mode 为空；不任意选择某一类。
因此不同类别在同一端点相接时，该端点会标记冲突，留待后续处理。
未标注、标签冲突和时间无效点从处理结果中移除。原始 `point_id` 不重新编号。
后续特征工程应注意来源文件和点序号的缺口，避免无条件将过滤后的相邻点视为原始相邻点。
这些缺口只作标记，不额外划分 trip；trip 规则见下文。

### 步骤 2：清洗与 trip 划分

执行 `uv run preprocess-trips`，读取 labeled_points，写入独立的 cleaned_points 目录。
原始数据与上一步标签数据均保留。每个用户独立处理，规则如下：

1. 去掉无效时间、非法坐标和没有成功标签的记录。
2. 按时间、来源文件、原始点序号稳定排序；跨文件统一处理同一用户。
3. 同一时间戳若出现不同 mode，移除该时间戳全部记录并统计。
4. 按时间、经纬度、海拔及 mode 删除重复观测，保留一份来源记录。
5. 同秒剩余多条记录时，选择距离该组经纬度中位数最近的一条原始观测。
   使用经度余弦校正的局部角度距离；距离相同时按来源文件及点序号确定。
   这是一项处理秒级时间精度的确定性约定，不代表所选坐标一定最准确。
6. 海拔 -777 或非有限值转为空值，不因此删除定位点。
7. 每个用户首点开启 trip，后续点仅在时间差 **大于 1200 秒**时开启新 trip。
   恰好 1200 秒不切分；文件变化、mode 变化、原始点序号缺口均不单独触发切分。

不做 Stay/Move 过滤，不按速度、停留或最短长度删点/删 trip；保留单点 trip。
trip 是基于已匹配、清洗后的观测序列建立的，不保证内部点点连续采样。

新增字段：

- `trip_id`：用户 ID 加用户内六位 trip 序号。
- `trip_point_index`：trip 内从 0 开始的点序号。
- `time_gap_s`：与该用户上一个保留点的时间差，包含跨 trip 间隔；用户首点为空。
- `trip_start`：是否为 trip 首点。
- `source_file_changed`、`source_point_gap`：同一 trip 内来源变化和点序号缺口提示。
- `same_timestamp_input_count`：该时间戳合并前的观测数（含完全重复）。
- `same_timestamp_position_count`：该时间戳去掉重复观测后、选择代表点前的记录数。

cleaned_points/report.json 记录每个用户及总体各步骤移除数量、保留点数及 trip 数。
代表点保留原始 point_id/source_line/source_file，便于追溯；不重新编号原始点。

### 预处理输出字段

| 字段 | 含义 |
|---|---|
| user_id | 保留前导零的用户标识 |
| source_file | 相对 data/original 的轨迹路径 |
| point_id | 原始文件内从 0 开始的点序号 |
| source_line | 原始文件行号，从 7 开始 |
| timestamp | 原始日期和时间解析结果 |
| latitude, longitude | 纬度和经度 |
| altitude_ft | 英尺海拔；labeled_points 保留原始 -777，cleaned_points 及后续数据将其转为空 |
| valid_coordinates | 是否为合法范围内的有效坐标 |
| mode | 匹配到的原始交通方式；当前保存的样本均已成功匹配，非空 |
| label_status | 匹配程序支持 matched / unlabeled / conflict / invalid_timestamp；当前输出仅保留 matched |
| label_match_count | 覆盖该点的标签区间数，包括同类重叠 |

`(user_id, source_file, point_id)` 可唯一追溯每个点。报告包括逐用户点数、文件数、
标签区间数、类别数量、匹配状态、无效坐标及标签重叠数量。
报告中的 `points` / `total_points`、`status_counts` 和质量统计描述筛选前输入；
`output_points` 为实际保留行数，`removed_points` 为移除行数。

## 特征工程

### 连接质量处理与当前交付

执行 `uv run quality-features`，从 five_class_features 生成 quality_features。
保留原 user/trip、坐标、时间、标签及样本点数量，重新分配 feature_block_id。
在原 block 开始、来源文件变化、或处理前 speed_mps >100 时开启新 block。
阈值单位为 m/s，严格大于；可通过 `--speed-limit` 显式配置。
规则不读取 mode；已核查 train 的 22 个 >100 m/s 区间，均为孤立尖峰，未见连续平稳超阈值行驶。
100 m/s 是本项目五类数据的质量控制约定，不宣称是所有地面交通的物理上限。

每个新 block 首点的全部运动特征为空，高阶差分随历史长度逐步恢复；没有使用均值插补或速度截断。
高加速度与 jerk 在重算后只标记审查候选，不删点，不触发额外断开。
来源变化会保守断开部分本来连续的连接，这是为避免未经验证的跨文件连接采用的明确约定。
统计报告给出处理前后有效特征数、极值、各原因数量和原因重叠数。

本次新增 549 个计算边界（241 个高速连接 + 348 个来源边界 − 40 个重叠），block 数从 6,415 增至 6,964。

| 数值/特征 | 处理前 | 处理后 | 变化 |
|---|---:|---:|---:|
| 总点数 | 4,347,764 | 4,347,764 | 0 |
| 有效速度点数 | 4,341,349 | 4,340,800 | −549 |
| 有效加速度点数 | 4,334,963 | 4,333,882 | −1,081 |
| 有效 jerk 点数 | 4,328,606 | 4,327,004 | −1,602 |
| 最大有效速度（m/s） | 4,142.19 | 99.79 | 连接阻断后重算 |
| 绝对加速度 >20 m/s² 候选数 | 2,423 | 2,000 | −423 |
| 绝对 jerk >20 m/s³ 候选数 | 3,613 | 3,186 | −427 |

候选数量可能相互重叠，不能相加作为唯一异常点数；仍有速度不超阈值的噪声与大差分，当前没有进一步删除。

| 新增/更新字段 | 定义 |
|---|---|
| pre_quality_speed_mps | 阻断前的速度，含原始缺失，仅用于审计，不作为建议模型输入 |
| speed_outlier_candidate | 阻断前速度严格大于本次 speed-limit；非有限/缺失不标 True |
| source_boundary_break | 原 block 内来源文件发生变化，触发阻断 |
| input_feature_block_id | 处理前类别筛选阶段的 feature_block_id |
| feature_block_id / feature_block_start | 质量处理后的计算边界；训练和窗口构造使用此版本 |
| break_reason | input_block_start、source_file_change、speed_gt_limit；同时满足时以竖线连接，无原因时 none |
| acceleration_review_candidate | 重算后加速度绝对值 >20 m/s²，仅作审查候选 |
| jerk_review_candidate | 重算后 jerk 绝对值 >20 m/s³，仅作审查候选 |

### 五类筛选与当前交付

类别筛选从清洗后的 4,823,144 点保留 4,347,764 点，减少 475,380 点（约 9.86%）。

| 类别 | 筛选前点数 | 筛选后点数 | 移除点数 |
|---|---:|---:|---:|
| walk | 1,321,226 | 1,321,226 | 0 |
| bus | 1,192,765 | 1,192,765 | 0 |
| bike | 775,977 | 775,977 | 0 |
| train | 556,382 | 556,382 | 0 |
| car | 501,414 | 501,414 | 0 |
| subway | 250,418 | 0 | 250,418 |
| taxi | 209,895 | 0 | 209,895 |
| airplane | 9,191 | 0 | 9,191 |
| boat | 3,565 | 0 | 3,565 |
| run | 1,975 | 0 | 1,975 |
| motorcycle | 336 | 0 | 336 |
| 合计 | 4,823,144 | 4,347,764 | 475,380 |

不将 taxi 合并到 car，也不将 subway 合并到 train。所有原始数据及各阶段中间数据均保留。
原 trip_id、trip_point_index 与来源点编号保持不变；新增 feature_block_id 与 feature_block_start。
每个 trip 的第一段保留样本、以及被删类别后重新出现的保留样本，开启新的 feature block。
例如 walk→subway→walk 的两段 walk 不相连；walk→bus 的直接变化不会因此切断。
特征在 `(user_id, trip_id, feature_block_id)` 内重算，后续窗口也不得跨 feature_block_id。
此边界仅阻断删去类别造成的连接，尚未实施按异常速度或来源文件变化切断的方案。

### 第一批特征及中间版本

输入 4,823,144 点，输出 4,823,144 点。执行 `uv run build-features`，在原字段基础上新增：

| 特征组 | 字段 |
|---|---|
| 时间与位移 | delta_t_s、distance_m、speed_mps |
| 速度变化 | acceleration_mps2、jerk_mps3 |
| 方向 | bearing_rad、bearing_sin、bearing_cos |
| 转向 | heading_change_rad、heading_change_rate_rad_s |
| 有效性 | 上述每个字段各有一个同名加 _valid 的布尔字段 |

完整公式、单位、时间对齐、缺失值及质量规则见 [FEATURES.md](FEATURES.md)。
数值按物理单位保存，未归一化或插补。只在同一 trip 内作后向差分；当前仍可能跨来源文件或被筛除的原始点区间计算。
trip 首点及历史不足时的高阶差分为空，零位移时方向为空；这些行全部保留。

### 点数核算与后续更新

| 文件 | 需要读取的计数字段 | 含义 |
|---|---|---|
| data/processed/raw_point_counts.json | 各用户点数之和 | 原始轨迹的独立行数统计 |
| data/processed/labeled_points/report.json | total_points、output_points、removed_points | 标签匹配输入、保留、移除点数；status_counts 描述筛选前输入 |
| data/processed/cleaned_points/report.json | totals.input_points、totals.output_points、各移除字段 | 清洗前后总数及顺序处理的移除数量 |
| data/processed/feature_points/report.json | total_points、feature_statistics | 特征输出总点数；每个特征的 valid + missing 应等于总点数 |
| data/processed/five_class_points/report.json | input_points、output_points、removed_points、各类别计数 | 五类筛选前后数量及移除原因 |
| data/processed/five_class_features/report.json | total_points、feature_statistics | 五类重算特征的总点数及缺失统计 |
| data/processed/quality_features/report.json | totals、feature_statistics | 质量处理前后点数、block 数、原因计数、有效特征数量及绝对极值 |

后续新增类别筛选、异常处理、窗口构造或数据集划分时，应追加输入/输出数量及原因。
窗口数量与唯一 GPS 点数应分别统计；不要把重叠窗口中的重复出现次数记作新增唯一样本点。

## Train/test 划分与训练交接

输入为 quality_features，共 64 个非空用户、4,347,764 个点。5 个空用户 059、088、116、118、124 不进入任一集合。
执行 `uv run split-dataset` 可复现，种子 42，目标测试比例 20%。

| 集合 | 用户数 | 点数 | 总点数占比 |
|---|---:|---:|---:|
| train | 51 | 3,486,792 | 80.20% |
| test | 13 | 860,972 | 19.80% |
| 合计 | 64 | 4,347,764 | 100% |

| 类别 | train 点数 | test 点数 |
|---|---:|---:|
| walk | 1,012,051 | 309,175 |
| bus | 981,452 | 211,313 |
| bike | 583,792 | 192,185 |
| train | 469,094 | 87,288 |
| car | 440,403 | 61,011 |

用户独立且类别均覆盖两侧；用户轨迹长度与类别构成不同，不能保证每类严格 80/20。
划分前对所有点的 `(timestamp, latitude, longitude, mode)` 检查跨用户完全重复。
先用哈希筛候选，再用原字段精确相等验证；只要共享记录就保守绑定用户。
绑定组为 069/126/129/167、081/125、084/085、089/144、108/139、153/163。
绑定防止重复记录跨集合，但不删除集合内部的重复记录，也不等同于排除近似轨迹重叠。
从 10,000 次固定种子的分组随机候选中选择划分，目标是各类点数测试比例、总点数测试比例接近 20%，
并兼顾用户比例；只使用类别数量，不使用模型表现选择测试集。之后固定清单，不随训练结果调整。

| 交接文件 | 用法 |
|---|---|
| data/splits/train_users.json | 训练用户 ID 字符串列表，保留前导零 |
| data/splits/test_users.json | 测试用户 ID 字符串列表 |
| data/splits/manifest.csv | 每个非空用户一行：user_id、split、path、points；path 相对 manifest 所在目录；user_id 应按字符串读取 |
| data/splits/report.json | 种子、统计、重复关联组、用户清单和各类点数 |

数据文件仍保存在 quality_features，不额外生成 train/test 副本。每个用户全部 trip/block 属于同一集合。
队友先读取清单，再在各集合内部生成窗口；窗口不得跨 feature_block_id。
validation 应从 train 用户内另划，并继续将重复关联用户绑定；不得用 test 调参。
缩放和缺失值插补参数仅在实际训练部分拟合。manifest 和清单相对项目可迁移，报告中的 input_directory 仅记录本次执行位置。

## 队友拉取仓库后的数据准备

Git 管理代码、环境锁文件和 `data/splits/` 下的固定划分清单；原始数据及生成的 Parquet 通过压缩包单独交接。仅 clone/pull 仓库不会获得训练数据。

### 直接使用已处理数据（训练队友推荐）

将完整的 `data/processed/quality_features/` 目录（全部用户 Parquet 和 report.json）打包共享，并注明对应的代码提交版本。队友解压到项目的同名目录，不要多嵌套一层 quality_features。

在项目根目录运行 `uv sync` 安装环境。训练读取已有 `data/splits/manifest.csv`；路径相对 manifest 所在目录解析，user_id 按字符串读取以保留前导零。当前数据总计 4,347,764 点，train 为 3,486,792 点，test 为 860,972 点。

此方式不需要原始数据和其他中间目录，也无需重跑预处理或 split-dataset。validation 可在 train 内做分组交叉验证，重复关联用户必须绑定同组，具体关联组见 splits/report.json。

### 从原始数据复现

需要复现预处理的队友，另行取得本项目相同的原始数据压缩包，解压到 `data/original/<user_id>/Trajectory/*.plt` 及 `data/original/<user_id>/labels.txt`。在项目根目录依次执行：

```powershell
uv sync
uv run preprocess-labels
uv run preprocess-trips
uv run filter-modes
uv run build-features --input data/processed/five_class_points --output data/processed/five_class_features
uv run quality-features
```

上述命令生成当前训练入口，无需生成旧 feature_points 中间版本。复现后核对 quality_features/report.json 的数量与本文记录，并沿用仓库已有的 train/test 清单。仅在有意更新数据集版本时重新生成划分，并一起更新清单和统计。

团队交接时应记录数据包对应的版本和校验值。目前仓库未配置自动下载，队友需从数据提供者取得数据包。

## 模型训练与评估（Part 2）

> **注意：本节描述 v1 流程（`train_basic.py` + `tune_smoothing.py`），数字已被 v2 取代。**
> v2（`pipeline_v2.py`）改为盲切验证窗口、5 折、所有模型同一 20k 子样本、按模型选择平滑方式并在完整流程上选模型，
> 选中 SVM (RBF, Nystroem, all data) + 软投票 k = 11，测试点级 macro-F1 0.844。
> 当前数字以 [docs/part2_v2_findings.md](docs/part2_v2_findings.md) 为准，给队友的说明见 [docs/part2_notes_for_teammates.md](docs/part2_notes_for_teammates.md)。

代码位于 `src/model_training/`，细节见该目录 README.md；给 Part 3/4 的交接说明见 `docs/part2_handover.md`。
输入为 `data/processed/quality_features/` 与 `data/splits/`，不修改 Part 1 的任何数据或清单。
以下数字均来自 `src/model_training/results/` 下的结果文件；“点级”指每个 GPS 点继承其所在窗口的预测后计分。

### 环境与运行

使用 Python 3.12.6，依赖版本固定在 `src/model_training/requirements.txt`（pandas 3.1.1、scikit-learn 1.9.1 等）。
pandas 2.x 未测试。内存不足时先设 `$env:LOKY_MAX_CPU_COUNT = "8"`，结果与 20 线程完全一致。

```powershell
cd src\model_training
pip install -r requirements.txt
python build_windows.py                                      # 窗口表 windows_w50.csv
python build_windows.py --no-pure-train --out windows_w50_blind.csv   # 全部用户盲切窗口，用于选 k
python train_basic.py                                        # 7 个模型，调参、选模型、测试 -> results/basic_w50_no_dt/
python train_basic.py --keep-dt                              # 含 dt 特征的消融 -> results/basic_w50/
python tune_smoothing.py                                     # 仅用训练用户选平滑窗口 k
python evaluate_smoothing.py                                 # 把选定的 k 一次性用于测试用户，生成分段
python make_figures.py                                       # 报告/海报图 -> results/figures/
```

### 窗口构造

推理时输入是没有分隔的逐点序列（见 关于输入输出的一点说明.docx），因此先把序列切成固定长度窗口，
对窗口分类，再把窗口预测赋给其中每个点。

| 项目 | 规则 |
|---|---|
| 窗口长度 | W = 50 点；不足 W/2 的尾段并入前一窗口，每个点恰好属于一个窗口 |
| 边界 | 窗口不跨 feature_block_id |
| 测试用户 | 盲切：不使用标签决定窗口边界，与真实推理一致 |
| 训练用户（拟合用） | 另在真实类别变化处切开，使每个训练窗口只含一类；少于 10 点的训练窗口丢弃 |
| 训练用户（选 k 用） | 与测试相同的盲切（windows_w50_blind.csv），因为按标签切开的窗口不像真实输入 |
| 窗口标签 | 窗口内多数类别；同时保留各类点数，点级评估按点数计分，混合窗口不失真 |

| 集合 | 用户数 | 点数 | 窗口数 | 说明 |
|---|---:|---:|---:|---|
| train（拟合） | 51 → 49 | 3,486,792 → 3,485,114 | 70,441 → 70,086 | 丢弃 355 个少于 10 点的窗口（1,678 点）；用户 100、147 仅有 9、7 个点，因此不再有窗口 |
| train（盲切，选 k） | 49 | 3,486,776 | 70,203 | 用户 100、147 不属于任何 CV 折，不参与 |
| test（盲切） | 13 | 860,972 | 17,380 | 平均纯度 0.9905；96.36% 的窗口只含一类 |

窗口点数与 Part 1 逐类核对一致（train：walk 1,012,051、bus 981,452、bike 583,792、train 469,094、car 440,403；
test：walk 309,175、bus 211,313、bike 192,185、train 87,288、car 61,011）。
窗口时长中位数约 98 秒。

### 窗口特征

每个窗口计算 29 个统计量，其中 27 个作为模型输入：

| 特征组 | 内容 |
|---|---|
| 速度 | 均值、标准差、最小、最大、10/50/90/95 分位数 |
| 加速度绝对值 | 均值、标准差、90 分位数、最大值 |
| jerk 绝对值 | 均值、90 分位数 |
| 转向 | 转向率均值、中位数、90 分位数；转向角均值 |
| 事件比例 | 停止（<0.5 m/s）、低速（<2 m/s）、急转（>30°）、大加速度（>1 m/s²） |
| 整体 | 时长、路径长度、直线度（位移/路径长度）、整体平均速度、有效速度点比例 |
| 不进入模型 | dt_mean、dt_max（采样间隔，保留在窗口表中供核查） |

采样间隔描述的是设备记录频率而非运动方式，且在用户间漂移：car 窗口的 dt_mean 中位数在训练用户为 4.12 s，
在测试用户为 2.0 s。消融结果（不平滑）如下；去掉后没有任何模型的 CV macro-F1 下降，因此默认排除。

| 模型 | CV macro-F1 含 dt | CV 不含 dt | 测试 macro-F1 含 dt | 测试 不含 dt |
|---|---:|---:|---:|---:|
| SVM (RBF) | 0.648 | 0.653 | 0.760 | 0.760 |
| 逻辑回归 | 0.631 | 0.631 | 0.716 | 0.717 |
| 随机森林 | 0.607 | 0.618 | 0.769 | 0.770 |
| 梯度提升 (HGB) | 0.602 | 0.607 | 0.758 | 0.767 |
| k-NN | 0.582 | 0.585 | 0.751 | 0.752 |
| 决策树 | 0.575 | 0.581 | 0.723 | 0.703 |

HGB 与 k-NN 在两次运行中选到的超参数不同，其变化不完全来自特征。

### 模型、调参与选模型

| 模型 | 选定超参数 | 说明 |
|---|---|---|
| 多数类基线 | – | 永远预测 walk |
| 逻辑回归 | C = 0.1 | 标准化；类别加权 |
| 决策树 | max_depth = 8，min_samples_leaf = 10 | 类别加权 |
| k-NN | k = 15，按距离加权 | 标准化 |
| SVM (RBF) | C = 1，gamma = scale | 标准化；类别加权；在 20,000 个窗口的分层子样本上训练 |
| 随机森林 | 200 棵树，max_features = sqrt，min_samples_leaf = 5 | balanced_subsample 加权 |
| 梯度提升 (HGB) | learning_rate = 0.05，max_leaf_nodes = 31，l2 = 1.0 | 固定 300 轮，不用早停（其内部验证集是随机划分，不按用户分组） |

- 调参：网格搜索，3 折 StratifiedGroupKFold，**只用训练用户**，目标 macro-F1。
- 分组：按用户分组；Part 1 的重复记录关联组（report.json 中 duplicate_linked_groups）放在同一折，训练用户共 45 个分组。
- 选模型：**CV macro-F1 最高者**，即 SVM (RBF)。该规则在看到测试结果前确定；测试成绩只报告，不用于选模型、选特征或选 k。
- 缺失值用训练集中位数填补。

### 时序平滑

同一 block 内窗口时间相邻，但分类器逐窗口独立预测。未平滑时 SVM 在测试 block 内改变预测 3,163 次，
真实类别只变化 529 次；858 个窗口是孤立的单窗口翻转，其中 82% 是错的。因此在模型最后加一步：

| 项目 | 规则 |
|---|---|
| 方法 | 每个窗口的预测替换为以它为中心的 k 个连续窗口的多数类；只在同一 feature_block_id 内投票，靠近边界时只用 block 内窗口；平票保留原预测 |
| 候选 k | 1（不平滑）、3、5、7、9、11、13、15、21 |
| 选 k 的数据 | 只用训练用户：沿用上面的 3 折，每折用其余两折的纯窗口重新拟合（超参数不变），预测本折用户的盲切窗口，得到 out-of-fold 预测 |
| 选 k 的规则 | 配对 one-SE：先取各折平均 macro-F1 最高的 k；对每个 k 计算与它的逐折差值，取平均差值不低于“负一个配对标准误”的最小 k |
| 结果 | SVM 9、逻辑回归 13、随机森林 9、HGB 13、k-NN 7、决策树 9；写入 smoothing_choice.json 后才计算任何测试分段指标 |

不用普通（非配对）标准误的原因：三折难度差异很大（SVM 在 k = 1 时三折 macro-F1 为 0.639、0.697、0.556，相差 0.14），
这部分差异对所有 k 相同，会使标准误约达 0.04，连 k = 1 也落在“一个标准误以内”；而实际上 k = 9 在每一折都比 k = 1 高至少 0.033。

SVM 在训练用户 out-of-fold 预测上的选择过程：

| k | 各折平均 macro-F1 | 与最优 k 的差 | 配对标准误 | 分段召回 (IoU ≥ 0.5) | 变化点召回 (±1 窗口) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.631 | −0.039 | 0.003 | 0.670 | 0.914 |
| 3 | 0.647 | −0.023 | 0.003 | 0.728 | 0.864 |
| 5 | 0.660 | −0.010 | 0.004 | 0.717 | 0.677 |
| 7 | 0.666 | −0.004 | 0.004 | 0.704 | 0.607 |
| **9（选定）** | 0.668 | −0.002 | 0.003 | 0.687 | 0.476 |
| 13（最高） | 0.670 | 0 | 0 | 0.662 | 0.345 |
| 21 | 0.666 | −0.004 | 0.004 | 0.632 | 0.161 |

k 越大，点级 F1 越高但真实的短段被一起抹掉：变化点召回从 0.91 降到 0.48（k = 9），这是已知局限，后续可改用显式建模类别转换的方法（如 HMM 或变化点检测）。

### 测试结果

13 个测试用户；95% 置信区间由按用户重抽样的 bootstrap 得到（5,000 次，关联用户整组抽取，共 9 个独立组）。
关联组 069/126/129/167 占测试点数的 78.9%，因此区间较宽且偏斜。

| 模型 | CV macro-F1（3 折均值 ± 标准差） | 测试 macro-F1 不平滑（95% CI） | k | 测试 macro-F1 平滑后（95% CI） | 测试准确率 平滑后（95% CI） |
|---|---:|---:|---:|---:|---:|
| 多数类基线 | – | 0.106 (0.094–0.177) | – | – | 0.359 (0.289–0.612) |
| **SVM (RBF)，选定** | **0.653 ± 0.060** | 0.760 (0.404–0.792) | 9 | **0.821 (0.460–0.856)** | **0.845 (0.744–0.898)** |
| 逻辑回归 | 0.631 ± 0.052 | 0.717 (0.375–0.757) | 13 | 0.785 (0.462–0.829) | 0.822 (0.715–0.931) |
| 随机森林 | 0.618 ± 0.049 | 0.770 (0.415–0.791) | 9 | 0.830 (0.467–0.853) | 0.850 (0.779–0.864) |
| 梯度提升 (HGB) | 0.607 ± 0.051 | 0.767 (0.400–0.796) | 13 | 0.844 (0.474–0.875) | 0.854 (0.761–0.935) |
| k-NN | 0.585 ± 0.062 | 0.752 (0.386–0.761) | 7 | 0.804 (0.417–0.819) | 0.841 (0.585–0.862) |
| 决策树 | 0.581 ± 0.040 | 0.703 (0.369–0.725) | 9 | 0.787 (0.424–0.811) | 0.817 (0.728–0.837) |

- 所有模型都显著优于多数类基线（配对差值的置信区间均不含 0）。
- SVM、随机森林、HGB、k-NN 之间的配对差值置信区间都包含 0，无法区分；随机森林和 HGB 在测试上略高，但按既定规则仍选 SVM。
- 平滑对 6 个模型的 macro-F1 提升都显著；SVM 提升 +0.061（95% CI 0.018–0.108），准确率 +0.048（0.002–0.072）。
- 加入平滑后，逻辑回归在训练 out-of-fold 上略高于 SVM（0.752 对 0.746，k = 9），差距小于折间波动，仍保留 SVM。
- CV（0.58–0.65）低于测试（0.70–0.77，不平滑），SVM 的 CV 值落在其测试置信区间内；可能原因是测试集由一个关联组主导。

SVM 各类别测试 F1：

| 版本 | walk | bike | bus | car | train |
|---|---:|---:|---:|---:|---:|
| 不平滑 | 0.865 | 0.836 | 0.698 | 0.565 | 0.836 |
| 平滑（k = 9） | 0.877 | 0.902 | 0.768 | 0.669 | 0.891 |

最终模型（SVM + 平滑）主要混淆（占真实类别点数）：bus → walk 14.1%（29,809 点）、car → train 14.0%（8,546 点）、
car → bus 11.3%（6,905 点）、bus → car 9.1%（19,136 点）、bike → walk 7.3%（14,077 点）。car 的精确率最低（0.651）。

### 分段输出

同一 block 内相邻且平滑后预测相同的窗口合并为一段，即单一出行方式子序列；
字段为 user_id、trip_id、feature_block_id、start_time、end_time、n_points、predicted_mode、majority_true_mode、n_windows。
真实段为多数真实类别相同的相邻窗口，评估以窗口为粒度。

| SVM，测试用户 | 段数（真实 1,738） | 分段召回 (IoU ≥ 0.5) | 分段精确率 | 变化点精确率 (±1 窗口) | 变化点召回 | block 内预测变化次数（真实 529） |
|---|---:|---:|---:|---:|---:|---:|
| 不平滑 | 4,372 | 0.724 | 0.288 | 0.203 | 0.894 | 3,163 |
| 平滑（k = 9） | 1,946 | 0.727 | 0.649 | 0.388 | 0.482 | 737 |

1,946 段中 511 段类别错误，覆盖 12.3% 的测试点；多数很短（中位数 2 个窗口），但 56 段长度 ≥ 10 窗口的错误段
共 43,655 点，几乎全部来自用户 167、052、126，平滑无法纠正这类整段误判。

### 测试用户构成（SVM 点级准确率）

| 用户 | 点数 | 占测试点数 | 重抽样组 | 不平滑 | 平滑 |
|---|---:|---:|---|---:|---:|
| 167 | 441,886 | 51.3% | 关联组 069/126/129/167 | 0.816 | 0.872 |
| 126 | 229,055 | 26.6% | 关联组 069/126/129/167 | 0.807 | 0.844 |
| 052 | 133,174 | 15.5% | 052 | 0.706 | 0.741 |
| 082 | 18,077 | 2.1% | 082 | 0.877 | 0.879 |
| 089 | 17,676 | 2.1% | 关联组 089/144 | 0.848 | 0.941 |
| 144 | 9,622 | 1.1% | 关联组 089/144 | 0.779 | 0.884 |
| 069 | 4,141 | 0.5% | 关联组 069/126/129/167 | 0.675 | 0.801 |
| 129 | 4,080 | 0.5% | 关联组 069/126/129/167 | 0.660 | 0.787 |
| 097 | 1,511 | 0.2% | 097 | 0.896 | 0.896 |
| 058 | 911 | 0.1% | 058 | 0.573 | 0.573 |
| 091 | 703 | 0.1% | 091 | 0.350 | 0.279 |
| 087 | 119 | <0.1% | 087 | 1.000 | 1.000 |
| 060 | 17 | <0.1% | 060 | 1.000 | 1.000 |

按用户不加权平均准确率：不平滑 0.768，平滑 0.808。

### 输出文件

| 文件 | 内容 |
|---|---|
| results/basic_w50_no_dt/results.csv | 每个模型的点级/窗口级指标、CV 均值与标准差、选定超参数、是否被 CV 选中 |
| results/basic_w50_no_dt/bootstrap_ci.csv、bootstrap_pairwise.csv | 测试置信区间、模型两两配对差值 |
| results/basic_w50_no_dt/per_user_test.csv | 每个测试用户的点数、占比与各模型准确率 |
| results/basic_w50_no_dt/smoothing/smoothing_cv.csv、smoothing_choice.json | 训练 out-of-fold 上每个 k 的得分与选定的 k |
| results/basic_w50_no_dt/smoothing/smoothing_test.csv、segment_metrics_test.csv | 测试上平滑前后的点级与分段指标 |
| results/basic_w50_no_dt/smoothing/segments_test.csv | **最终输出**：SVM 平滑后的单一出行方式分段 |
| results/basic_w50/ | 含 dt 特征的消融运行 |
| results/figures/ | 报告和海报用图 fig1–fig5（200 dpi）及 captions.md |

窗口表和 `results/` 不提交版本管理，需与对应提交号一起另行打包共享（清单见 `docs/part2_handover.md`）。

### 局限

- 测试集由关联组 069/126/129/167 主导（78.9% 的点），只有 9 个独立组，测试成绩主要反映这几位用户，置信区间仅供参考。
- 51 个训练用户中只有 49 个提供训练窗口。
- SVM 在 20,000 个窗口的子样本上调参和训练（约占 70,000 个训练窗口），与其他模型的比较并非完全同等条件。
- 平滑窗口 k = 9 约合 15 分钟出行，会抹掉真实的短段（变化点召回 0.91 → 0.48）。
- 分段与变化点按窗口粒度评估，窗口内部的类别变化被归到最近的窗口边界。
- 只评估了一种配置（W = 50、27 个特征）；窗口大小、采样率、噪声和特征组实验由 Part 3/4 负责。

## 环境故障排查

在受限 Windows 环境下，如果 uv 的系统临时目录不可写，可先运行：

```powershell
New-Item -ItemType Directory -Path .uv-cache/tmp -Force | Out-Null
$env:TEMP = (Resolve-Path .uv-cache/tmp).Path
$env:TMP = $env:TEMP
uv sync --link-mode copy
```
