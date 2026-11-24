# 噪声实验本轮修订入口

先阅读[综合分析草稿](REPORT_DRAFT.md)，再核对[修订与证据清单](CHANGES_AND_EVIDENCE.md)。
三张修订图的PNG和矢量PDF位于figures/；完整数值位于tables/。
原FINAL_REPORT、诊断、文献复盘与配对分析保留为原快照，本目录是补充入口。

在仓库根目录复现本轮派生输出：

```powershell
.venv-audit/Scripts/python.exe -B experiments/noise_feature_smoothing/report_revision/build_revision.py
.venv-audit/Scripts/python.exe -B experiments/noise_feature_smoothing/report_revision/verify_revision.py
```

只重算现有结果的统计展示与时间/标签支持，不训练模型、不重新扰动坐标、不抽新bootstrap。
本轮所有新产物均写入本目录，另在docs/aigc_log_part2.md追加披露。
acceptance.json记录生成验收；delivery_verification.json记录独立复核；tests.log记录测试。

图表已经执行的自动检查范围见plot_layout_audit.json与raster_audit.json。
图像查看工具不支持图像输入，完整目视核对仍待团队完成。正文是AI辅助草稿，需要团队核对、改写和完成引用。
