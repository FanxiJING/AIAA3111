# 文献、软件与证据边界

核查日期：2026-11-23。以下文献用于解释诊断方向，不用于更改固定实验协议或选取模型。网络检索接口本轮未返回可用正文，改用原始站点 HTTP 获取。Europe PMC XML 接口返回 500 后，使用 PMC 期刊全文页面核对。

## 1. 定位误差、距离与采样

Ranacher, P., Brunauer, R., Trutschnig, W., Van der Spek, S., & Reich, S.
*Why GPS makes distances bigger than they are*. International Journal of Geographical Information Science.
DOI: 10.1080/13658816.2015.1086924。2015 年在线发表，2016 年卷期。

原始全文：https://pmc.ncbi.nlm.nih.gov/articles/PMC4786863/

本轮核实了作者、DOI、摘要及第 3、4、6 节相关段落。论文讨论在插值误差可忽略等条件下的距离高估，以及位置误差时空自相关的作用。支持分析速度特征偏移与采样间隔，不意味着所有 GPS 噪声都独立同分布，也不保证所有轨迹统计量都单调增加。

## 2. 出行方式识别中的轨迹降噪

Zeng, J., Zhang, G., Hu, Y., & Wang, D.
*Addressing robust travel mode identification with individual trip-chain trajectory noise reduction*.
IET Intelligent Transport Systems. DOI: 10.1049/itr2.12243。2022-07-24 在线发表，2023 年卷期。

出版者登记摘要：https://api.crossref.org/works/10.1049/itr2.12243

本轮核实范围为 Crossref 中的出版者摘要及元数据，不是论文全文。摘要描述了六种轨迹降噪方法、运动特征恢复与分类表现的比较。该文的轨迹平滑不同于本项目的 decision 分数域平滑，不借用其最优半径或宣称本项目已经验证坐标去噪。

## 3. 运动导数作为分类输入

Dabiri, S., & Heaslip, K. (2018).
*Inferring transportation modes from GPS trajectories using a convolutional neural network*.
Transportation Research Part C. DOI: 10.1016/j.trc.2017.11.021。

作者预印本摘要：https://arxiv.org/abs/1804.02386

本轮核实摘要与作者元数据。摘要明确包含速度、加速度、jerk 和 bearing rate。本项目只借此说明运动导数分析具有领域相关性，不直接比较该文 Accuracy 和本项目 Macro-F1，也不以此证明 CNN 的抗噪优越性。

## 4. RBF 与 Nystroem 实现

scikit-learn 官方文档：

- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html
- https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html

本轮直接获取并核实了 RBF 文档中的 `exp(-gamma ||x-y||^2)` 定义。Nystroem 文档列为实现参考，不声称本轮重新逐段审阅。诊断使用本地已保存模型的 `components_`、`normalization_`、`coef_` 和 `intercept_`，逐个条件核对重建 decision 分数，而不是依靠文档推测模型行为。

“特征偏移后有限基点相似度下降、表示接近零、分数接近截距”是本项目的事后机制证据。相似度只针对 1000 个冻结基点，不代表到全部训练数据的最近距离，也不能解释全部错误。

## 5. Endo 引用范围

Endo et al. (2016).
*Deep Feature Extraction from Trajectories for Transportation Mode Estimation*.
PAKDD. DOI: 10.1007/978-3-319-31750-2_5。

作者预印本：https://www.cgg.cs.tsukuba.ac.jp/~endo/pdf/pakdd2016_endo_preprint.pdf

沿用前轮已核查的范围：所引用噪声实验使用 KT 数据集，不能写成在 GeoLife 上直接复现该文。本轮未重新提取该 PDF 全文。项目的 E/N 独立高斯噪声与 sigma 档位为自定压力测试，不是从这篇论文复制的标定方案。

## 6. 研究流程辅助

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*.
arXiv:2609.00065. DOI: 10.48550/arXiv.2609.00065。

原始记录：https://arxiv.org/abs/2609.00065

本轮核实了标题、作者及 2026 年元数据。使用其 `scientific-critical-thinking` 流程辅助区分事后描述、机制解释与因果结论，检查独立抽样单位和不确定性表达。它不是 GPS 实验的实证文献，也不是本项目结果正确性的外部认证。
