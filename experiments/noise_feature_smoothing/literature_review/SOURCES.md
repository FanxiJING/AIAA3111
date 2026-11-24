# 文献身份、读取范围与可支持结论

核查日期：2026-11-23。针对当前问题选择七篇研究与一项流程软件引用，不声称覆盖所有论文或找到全球最新研究。
下载记录、实际请求地址、失败原因、时间、正文或元数据哈希均在 source_verification.json，
完整本地运行将原始返回保存在 `sources/`，获取入口为 [collect_sources.py](collect_sources.py)。HTTP 成功只记录获取状态，实际阅读范围见下文。

Crossref 用于核实出版者登记的书目信息和摘要；领域结论来自下列注明的原文或摘要，
不用搜索片段推测参数、结果或数据集。来源里的版权内容不复制到报告正文。

## R1：误差、距离与自相关

Ranacher, P., Brunauer, R., Trutschnig, W., Van der Spek, S., & Reich, S.
*Why GPS makes distances bigger than they are*. International Journal of Geographical Information Science, 30(2), 316–333.
DOI: 10.1080/13658816.2015.1086924。2015年在线，2016年卷期；不同数据库的在线日期记录略有不同，不混作两个研究。

读取范围：PMC期刊全文的摘要及第3、4、6节相关段落，另核对Crossref。
支持：忽略插值误差等条件下的距离高估，以及位置误差时空自相关的重要性。
限制：不能证明本项目所有统计量必然单调，也不意味着真实GPS为逐点独立高斯。
本轮Rayleigh参照与差分公式是针对已固定人工噪声的自行推导，不称为论文的GeoLife实测结论。

本地存档为 `sources/ranacher_primary.txt`（全文页面提取）与 `sources/ranacher_crossref.json`（元数据）。

## R2：轨迹坐标降噪与方式分类

Zeng, J., Zhang, G., Hu, Y., & Wang, D.
*Addressing robust travel mode identification with individual trip-chain trajectory noise reduction*.
IET Intelligent Transport Systems. DOI: 10.1049/itr2.12243。
2022-07-24在线，2023年1月卷期。

读取范围：出版者在Crossref登记的摘要及元数据；未获得并通读全文。
摘要描述六种轨迹降噪方法、运动特征恢复与方式分类的比较。
这使“降噪后特征与分类共同评价”具有直接相关性，但不能把其方法或30秒参数建议直接用于本研究。
坐标降噪与本项目decision分数平滑不同；不宣称本项目已验证Gaussian kernel坐标平滑。

本地存档 `sources/zeng_crossref.json` 包含出版者摘要和元数据。

## R3：运动导数的领域相关性

Dabiri, S., & Heaslip, K. (2018).
*Inferring transportation modes from GPS trajectories using a convolutional neural network*.
Transportation Research Part C, 86, 360–371. DOI: 10.1016/j.trc.2017.11.021。

读取范围：作者arXiv摘要、期刊映射和Crossref元数据。
摘要包含speed、acceleration、jerk和bearing rate。
支持这些运动量作为识别输入的领域背景，不证明某一导数更鲁棒，
也不能将其Accuracy与我们的固定五类点级Macro-F1直接比较。

本地存档为 `sources/dabiri_primary.txt`（作者摘要）与 `sources/dabiri_crossref.json`（元数据）。

## R4：Endo引用的适用范围

Endo, Y., Toda, H., Nishida, K., & Kawanobe, A. (2016).
*Deep Feature Extraction from Trajectories for Transportation Mode Estimation*.
PAKDD. DOI: 10.1007/978-3-319-31750-2_5。

本轮新增读取范围：Crossref身份核对，未重新读取PDF全文。
KT噪声实验的限定沿用前轮已经记录的作者预印本检查，见旧 diagnostics/SOURCES.md；
不写成“本轮重新核实全文”，也不写成GeoLife上复现该文。
本项目E/N噪声、σ档位、配对种子与窗口方案是自定协议。

本地元数据存档为 `sources/endo_crossref.json`。

## R5：依赖数据的验证单位

Roberts, D. R., et al. (2017).
*Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure*.
Ecography, 40, 913–929. DOI: 10.1111/ecog.02881。
完整作者名单保存在出版者元数据中。

读取范围：Crossref出版者摘要及身份；Wiley全文请求失败，未声称读完全文。
支持验证划分需要尊重数据依赖结构，以及预测目标会影响阻断方式。
该文研究背景主要是生态学，不是9组bootstrap覆盖率的认证，
不能用它证明本项目关联组已经消除了所有潜在依赖。

本地存档 `sources/roberts_crossref.json` 包含摘要和元数据。

## R6：序列过分割需要独立评价

Abu Farha, Y., & Gall, J. (2019).
*MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation*.
CVPR, 3575–3584. DOI: 10.1109/CVPR.2019.00369。

读取范围：CVF官方论文摘要页面与书目信息，未全文通读。
论文用专门的smoothing loss处理视频动作过分割；可作“分类与序列问题应区别评价”的相关背景。
不能当作GPS实验、k=11后处理或±1窗class-blind匹配的验证。
检索中一个候选DOI末尾00368对应另一篇论文，已按题目核对拒绝；
最终使用的00369已通过Crossref题目和CVF页面交叉核查。

本地存档为 `sources/farha_primary.txt`（CVF 摘要）与 `sources/farha_crossref.json`（核实的元数据）。

## R7：较新研究线索，结论暂不外推

Lu, S., Su, Y., Chai, J., & Yu, L.
*Enhancing applicable travel mode identification under real-world noise: A transformer-based framework with hybrid data and behavior-indication masks*.
Transportation Research Part C. DOI: 10.1016/j.trc.2025.105388。
出版者登记卷期日期为2026年1月；DOI中的2025不能单独当作卷期年份。

读取范围：Crossref完整身份核对；原站搜索摘要片段仅作检索定位。
本地直接请求ScienceDirect页面失败，未通读全文或存到完整摘要。
因此只列为真实存在的后续研究线索；不摘取其指标、噪声强度、数据融合细节或优越性结论，
不据此改成Transformer或训练增强。本轮不再把此前尚未核实的“2026论文”说法当作已读全文证据。

本地元数据存档为 `sources/lu_crossref.json`。HTTP 失败记录见 [source_verification.json](source_verification.json)。

## S1：流程软件引用

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*.
arXiv:2609.00065. DOI: 10.48550/arXiv.2609.00065。

直接获取arXiv记录核对了作者、标题、2026年份及当前修订记录（2026-09-02），引用不加版本后缀。
该arXiv DOI不在Crossref接口中，该请求404已保留；不因此声称论文不存在或元数据验证成功。
本轮使用其中scientific-critical-thinking流程区分独立单位、事后描述、机制证据与因果解释。
它不是领域实证论文，也不是代码正确性或模型性能的外部认证。

本地 arXiv 记录存档为 `sources/skill_primary.txt`。
