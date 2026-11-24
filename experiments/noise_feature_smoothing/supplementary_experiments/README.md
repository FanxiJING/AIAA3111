# 噪声补充实验索引

2026-11-23。原主实验保持不变，本目录是独立保存的探索性扩展。

- 先读[结果解释与不足](FINDINGS.md)：关键数值、反向结果、剩余限制和报告组织。
- [完整结果报告](SUPPLEMENT_REPORT.md)：全部噪声档位、固定gamma、组影响、平滑表。
- [补充协议](PROTOCOL.md)：在本轮结果产生前固定的设置。
- [验收](acceptance.json)、[独立验算](independent_verification.json)、
  [测试日志](tests.log)、[checkpoint](checkpoint.json)。完整运行日志 `run.log` 保存在本地存档。
- [绘图恢复说明](REPAIR_NOTES.md)及[修复后的布局检查](diagnostics/render_recovery.json)。
- [四张PNG/PDF图](figures/)、[完整指标表](tables/summary.csv)、
  [配对比较](tables/paired_contrasts.csv)、[原始种子指标](tables/all_seed_metrics.csv)。
- [逐用户](tables/per_user_metrics.csv)、[逐类](tables/per_class_f1.csv)、
  [逐组排除](tables/leave_one_group_out_summary.csv)、
  [组影响与OU减独立噪声](tables/noise_structure_by_omitted_group.csv)。

## 本轮命令

在仓库根目录，使用`.venv-audit/Scripts/python.exe -B`。

```powershell
# 原始完整运行；模型、特征、预测与指标已完成，出图检查首次报错。
& '.venv-audit/Scripts/python.exe' -B 'experiments/noise_feature_smoothing/supplementary_experiments/run_supplement.py' --stage all

# 仅恢复绘图与最终验收，不重算实验；保留原运行器与协议哈希。
& '.venv-audit/Scripts/python.exe' -B 'experiments/noise_feature_smoothing/supplementary_experiments/complete_delivery.py'

# 独立数值验算，无需重训。
& '.venv-audit/Scripts/python.exe' -B 'experiments/noise_feature_smoothing/supplementary_experiments/verify_supplement.py'

# 重新生成已保存噪声场的解释性诊断及结论用表。
& '.venv-audit/Scripts/python.exe' -B 'experiments/noise_feature_smoothing/supplementary_experiments/noise_structure_diagnostics.py'
& '.venv-audit/Scripts/python.exe' -B 'experiments/noise_feature_smoothing/supplementary_experiments/build_interpretation_tables.py'
```

`run_supplement.py`的七小时限制从`input_manifest.json`中的首次初始化起算，
恢复不会重置计时。`complete_delivery.py`保留这一限制。
日后独立复跑应将源码与协议放入本目录同层的全新独立输出目录，
从新manifest开始，保留现有结果。不要删除现有manifest以绕过恢复检查。
独立验算和解释性诊断不会重新拟合模型。

原数据、原源码、原结果、原报告共572份文件受本轮哈希清单保护；
唯一单列更新的是`docs/aigc_log_part2.md`中的流程披露。
手工图表目视审阅仍待完成。
