# 本轮来源与实际阅读范围

日期：2026-11-23。按三个已见异常选择来源，不声称系统综述或全球最新论文检索。
网页检索工具本轮未返回可用内容；以下官方页面经直接HTTP获取，
读取保存的正文或摘要。请求、内容检查、时间、哈希见
[source_retrieval.json](source_retrieval.json)。HTTP 200不自动代表拿到论文。

## 1. GPS误差与自相关

Ranacher, P., Brunauer, R., Trutschnig, W., Van der Spek, S., & Reich, S.
Why GPS makes distances bigger than they are.
International Journal of Geographical Information Science, 30(2), 316-333.
2015在线，2016卷期。DOI:10.1080/13658816.2015.1086924。

本轮PMC请求返回浏览器验证页，已判为**未取得本文内容**，没有把200状态当作重读全文。
本轮实际复核此前已保存的期刊全文提取 `../literature_review/sources/ranacher_primary.txt`（完整本地运行存档）
中摘要及第3、4、6节相关内容。旧文件哈希已加入本轮读取记录。

原文支持：在插值误差可忽略等条件下，位置测量误差引起距离高估，
其影响与位置误差的自相关有关。该文没有认证我们的OU30参数、GeoLife分数、
RBF退化机制，亦不证明所有真实GPS均服从同一种高斯误差模型。
本项目的相邻误差差分公式属于固定人工过程下的数学推导。

## 2. RBF核与Nystroem

scikit-learn官方 `rbf_kernel` API：
本轮读取核公式、gamma 参数及返回值。页面文本存档为 `sources/rbf.txt`。
官方给出 `K(x,y)=exp(-gamma*||x-y||^2)`。

scikit-learn官方 `Nystroem` API：
本轮读取 components、normalization 及 gamma 定义。页面文本存档为 `sources/nystroem.txt`。
文档与本地模型实际步骤共同用于解释表示。
本轮另以已保存模型实际Pipeline核对31条件、每条件64窗，
并把代数重建分数与所有31份保存分数逐元素核对。
本地计算支持具体退化机制，文档本身不证明本数据存在分布外偏移。

## 3. 分类与序列过分割

Abu Farha, Y., & Gall, J. (2019).
MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation.
CVPR, pp.3575-3584. DOI:10.1109/CVPR.2019.00369。

本轮读取 CVF 官方摘要与书目信息，本地存档为 `sources/farha.txt`。
摘要说明其分类损失与针对过分割的平滑损失；研究对象为视频动作分割。
支持把逐时刻分类与序列分段分开讨论这一相关背景。
**没有**把它写成GPS论文、我们11窗后处理的验证，亦未声称通读论文PDF。
短段消失与滞后的项目结论来自本轮本地序列诊断及已知答案试验。

## 4. OU公式的沿用来源

Rasmussen, C. E., & Williams, C. K. I. (2006).
Gaussian Processes for Machine Learning, chapter4, pp.85-86.
沿用上一轮已读指数协方差与OU段落及
[已保存获取记录](../supplementary_experiments/literature/retrieval.json)；
本轮不声称再次通读PDF。30秒是本项目探索性设定，没有设备标定依据。

## 5. 流程软件

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents.
arXiv:2609.00065. DOI:10.48550/arXiv.2609.00065。

本轮重新获取并读取当时的 arXiv 记录，核对作者、题名、年份与修订记录。本地存档为 `sources/skill.txt`。
该技能用于区分描述、机制证据、独立样本和待验证解释，不构成外部有效性认证。
