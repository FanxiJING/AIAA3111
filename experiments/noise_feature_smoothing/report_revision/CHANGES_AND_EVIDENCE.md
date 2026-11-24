# 修订与证据清单

日期：2026-11-23。本轮为固定实验的展示与事后描述补充，不新增训练、噪声条件、随机抽样、超参数或主指标。

| 原不足 | 已采取的修订 | 可核对的证据 | 保留的边界 |
| --- | --- | --- | --- |
| 主表将49称作训练划分人数，容易与51混淆 | 协议表区分51固定训练用户与49个贡献拟合窗的用户 | tables/protocol_clarified.csv；原CLAUDE.md与docs/part2_handover.md | 原始协议表作为历史产物保留 |
| 三轮结果分散，容易混用不同口径 | 合成36行全条件结果，连结144行原指标、固定五类bootstrap与预测集中度 | tables/all_conditions.csv；metrics/all_metrics.csv；metrics/stage5_bootstrap_all.csv | 不混用Part2按出现类别计算的区间 |
| SD与用户区间容易误读 | 图1将种子误差棒、配对退化、用户组区间分为三幅 | figures/figure1_performance_and_uncertainty.png | clean是一份，不称五次重复 |
| 只看平滑后F1和跳变，遗漏召回代价 | 图2同时呈现分类F1、边界P/R/F1、错误孤立跳变及最大预测类别占比 | figures/figure2_smoothing_tradeoffs.png；原配对平滑表 | 未以测试集选择新的k或平滑器 |
| ±1窗容易被当作固定秒数 | 为全部529真实窗口边界导出合法候选位置的秒级支持 | tables/boundary_tolerance_seconds.csv、boundary_tolerance_summary.csv | 仅解释既定容差，不重新按秒打分 |
| 窗口多数标签与逐点真值的差别不够具体 | 计数混合标签窗口、其中点数及非多数点数 | label_resolution.json；feature_cache/sigma00_seed0.csv中的cnt_* | 不将计数当事件频率或新识别指标 |
| 案例仅按窗口轴，易忽略时间不均匀 | 同一批原案例改用实际时间，增加同窗clean raw，框出原短run | figures/figure3_saved_cases_actual_time.png；metrics/case_selection.json | 未重新选好看案例，不估计总体发生率 |
| 机制解释可能超出数据 | 结论改为“伴随”“部分机制”，分别列绝对差与退化差 | REPORT_DRAFT.md第3、4、6节；paired_feature_contrasts.csv | 不归因于单个导数，不把预测集中等同下溢 |
| 引文与本项目方法可能混淆 | 连接已核查书目并逐项说明读取范围 | literature_review/SOURCES.md、source_verification.json | 没读到全文的研究不报告方法细节/效果 |
| 自动图表检查可能被写成目视通过 | 新验收单独保留manual=false及本次工具失败事实 | plot_layout_audit.json、raster_audit.json、acceptance.json | 非空像素与文字边界检查不替代完整目视检查 |

这里的跨目录路径，除tables/、figures/、本目录文件外，均以实验根目录或仓库根目录中的同名文件为准。

## 无法在冻结协议内消除的限制

- 九个关联组且最大组占78.8832%点数：报告分组、类别支持和现有配对区间，不能通过再切数据消除。
- 特征集合与gamma=1/d共同变化：只能比较预定配方，单因素归因需要另一份研究协议。
- 噪声是额外独立E/N高斯扰动：相关误差、坐标滤波或训练增强尚未运行，不能写成已验证。
- acc_std为有符号加速度标准差，path_length和duration支持不完全相同：保留并披露原定义。
- class-blind窗口边界指标不要求转换方向正确，成功匹配延迟也受到漏检选择影响。
- 原v2 raw复现差1窗、居中逐窗相同：承认版本复现差异，不能声称找回原拟合模型。

## 实际复核范围

build_revision.py验证36行与原bootstrap共同列一致、144行条件覆盖、529边界计数、全部受保护输入哈希与文件集合不变，生成三图并做画布/像素检查。

verify_revision.py另从48份既有平滑预测和cnt_*重算144个五类点级F1/Accuracy；检查报告的常用数值、原案例选择不变、输出哈希、Markdown本地链接和重点计数；执行针对混合真值计分、合法边界支持、重复/缺失连接键和种子SD的已知答案测试。验收以实际运行日志为准。
