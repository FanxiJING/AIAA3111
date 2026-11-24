# 文献对照与确定性补充检查

本目录只读现有实验及原测试数据，保存事后描述性分析和文献身份记录。
入口为 [中文复盘](REVIEW_REPORT.md) 和 [来源范围](SOURCES.md)。
原固定协议、模型、缓存、预测、诊断与配对分析不修改。

在仓库根目录运行；联网检索步骤与数值分析可单独执行：

```powershell
.venv-audit/Scripts/python.exe -B experiments/noise_feature_smoothing/literature_review/collect_sources.py
.venv-audit/Scripts/python.exe -B -m unittest discover -s experiments/noise_feature_smoothing/literature_review -p "test_*.py" -v
.venv-audit/Scripts/python.exe -B experiments/noise_feature_smoothing/literature_review/build_checks.py
.venv-audit/Scripts/python.exe -B experiments/noise_feature_smoothing/literature_review/verify_review.py
```

collect_sources.py归档HTTP响应、题目与DOI核对及失败信息，不自动认定全文已审阅。
数值检查不依赖网络、不生成随机数、不重算bootstrap，也不运行原主实验或旧诊断的写入脚本。
build_checks.py仅覆盖本目录的派生检查结果；输出前后校验506个原有文件。
docs/aigc_log_part2.md另有本轮披露追加，不在该只读数值保护集合中。

解析表仅为局部平面、无原始位移时的额外独立误差参照。
坐标线性二阶/三阶差分不等于项目标量speed的acc/jerk计算。
常量类别参照不是新拟合基线或按测试结果选出的模型。
所有新增比较均为事后描述，不能用于模型选择、因果归因或显著性排名。
