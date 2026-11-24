"""Read-only paired post-hoc analysis of the frozen GeoLife experiment."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "src/model_training"))
from models import MODES, cv_groups

GROUPS = ("G1", "G2", "G3")
SIGMAS = (0, 5, 10, 20)
SEEDS = (20261123, 20261124, 20261125, 20261126, 20261127)
SMOOTHERS = ("S0_raw_k1", "S1_centered_soft_k11", "S2_trailing_soft_k11")
METRICS = ("macro_f1", "accuracy", "transition_precision", "transition_recall",
           "transition_f1", "wrong_isolated_flips_per100_windows",
           "prediction_change_from_clean_point_fraction")
META = ["user_id", "feature_block_id", "start_time", "end_time", "n_points",
        "label", "purity", *[f"cnt_{m}" for m in MODES]]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def write_csv(name, rows):
    data = pd.DataFrame(rows)
    data.to_csv(HERE / "tables" / f"{name}.csv", index=False, encoding="utf-8-sig")
    return data


def protected_files():
    files = []
    for root in ("src", "tests", "data", "docs", "experiments"):
        files.extend(p for p in (ROOT / root).rglob("*") if p.is_file()
                     and HERE not in p.parents and "__pycache__" not in p.parts
                     and "matplotlib_cache" not in p.parts and p.suffix != ".pyc")
    files.extend(ROOT / name for name in ("README.md", "CLAUDE.md", "pyproject.toml", "uv.lock"))
    return sorted(files)


def safe_ratio(num, den):
    num, den = np.broadcast_arrays(np.asarray(num), np.asarray(den))
    return np.divide(num, den, out=np.zeros(num.shape, float), where=den != 0)


def confusion(counts, predictions):
    """Mixed windows contribute their true point counts, not majority weights."""
    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.shape[1] != len(MODES):
        raise ValueError("Expected N x 5 true point counts")
    pred = pd.Index(MODES).get_indexer(predictions)
    if len(pred) != len(counts) or np.any(pred < 0):
        raise ValueError("Unaligned or unknown predictions")
    return np.stack([counts[pred == j].sum(axis=0) for j in range(len(MODES))], axis=1)


def point_metrics(cms):
    cms = np.asarray(cms)
    tp = np.diagonal(cms, axis1=-2, axis2=-1)
    f1 = safe_ratio(2 * tp, cms.sum(axis=-1) + cms.sum(axis=-2))
    return {"macro_f1": f1.mean(axis=-1),
            "accuracy": safe_ratio(tp.sum(axis=-1), cms.sum(axis=(-2, -1)))}


def boundary_metrics(true, pred, hit):
    true, pred, hit = np.asarray(true), np.asarray(pred), np.asarray(hit)
    return {"transition_precision": safe_ratio(hit, pred),
            "transition_recall": safe_ratio(hit, true),
            "transition_f1": safe_ratio(2 * hit, np.asarray(true) + np.asarray(pred))}


def point_flow(counts, raw, smoothed):
    counts = np.asarray(counts)
    ri = pd.Index(MODES).get_indexer(raw)
    si = pd.Index(MODES).get_indexer(smoothed)
    if len(ri) != len(counts) or len(si) != len(counts) or (ri < 0).any() or (si < 0).any():
        raise ValueError("Unaligned or unknown flow predictions")
    changed = ri != si
    rows = np.arange(len(counts))
    rescued = int(counts[rows, si][changed].sum())
    damaged = int(counts[rows, ri][changed].sum())
    changed_points = int(counts.sum(axis=1)[changed].sum())
    return {"points": int(counts.sum()), "rescued_points": rescued,
            "damaged_points": damaged, "changed_prediction_points": changed_points,
            "changed_still_wrong_points": changed_points - rescued - damaged,
            "net_correct_points": rescued - damaged,
            "accuracy_delta_from_raw": (rescued - damaged) / counts.sum()}


def seed_mean(arrays):
    return np.stack(arrays).mean(axis=0)


def paired_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("Paired statistics must have identical draw/seed support")
    return a - b


def interval(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return {"ci_low": float(lo), "ci_high": float(hi)}


def seeds(sigma):
    return (0,) if sigma == 0 else SEEDS


def name(sigma, seed):
    return f"sigma{sigma:02d}_seed{seed}"


def read_prediction(group, sigma, seed):
    return pd.read_csv(EXP / "predictions" / f"{group}_{name(sigma, seed)}_smoothed.csv",
                       dtype={"user_id": str})


def markdown(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def evidence_row(a, b, sigma, metric, values, distributions):
    diffs = np.asarray([values[a(s)][metric] - values[b(s)][metric] for s in seeds(sigma)])
    dist = seed_mean([paired_delta(distributions[a(s)][metric],
                                   distributions[b(s)][metric]) for s in seeds(sigma)])
    return {"sigma": sigma, "metric": metric, "seed_count": len(diffs),
            "difference_seed_mean": float(diffs.mean()),
            "difference_seed_sd": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
            **interval(dist)}


def build_contrasts(values, distributions):
    smoothing, interactions, features = [], [], []
    for smoother in SMOOTHERS[1:]:
        for sigma in SIGMAS:
            a = lambda s: ("G3", smoother, sigma, s)
            b = lambda s: ("G3", SMOOTHERS[0], sigma, s)
            for metric in METRICS:
                row = evidence_row(a, b, sigma, metric, values, distributions)
                smoothing.append({"comparison": f"{smoother} minus raw", **row})
                if metric not in ("macro_f1", "accuracy"):
                    continue
                clean_a, clean_b = ("G3", smoother, 0, 0), ("G3", SMOOTHERS[0], 0, 0)
                clean_gain = values[clean_a][metric] - values[clean_b][metric]
                clean_dist = paired_delta(distributions[clean_a][metric],
                                          distributions[clean_b][metric])
                dist = seed_mean([paired_delta(distributions[a(s)][metric],
                                                distributions[b(s)][metric])
                                  for s in seeds(sigma)]) - clean_dist
                interactions.append({"comparison": f"{smoother} minus raw", "sigma": sigma,
                                     "metric": metric,
                                     "difference_in_differences":
                                     row["difference_seed_mean"] - clean_gain,
                                     **interval(dist)})
    for ga, gb in itertools.combinations(reversed(GROUPS), 2):
        for sigma in SIGMAS:
            a = lambda s: (ga, SMOOTHERS[0], sigma, s)
            b = lambda s: (gb, SMOOTHERS[0], sigma, s)
            for metric in ("macro_f1", "accuracy"):
                row = evidence_row(a, b, sigma, metric, values, distributions)
                ca, cb = (ga, SMOOTHERS[0], 0, 0), (gb, SMOOTHERS[0], 0, 0)
                clean_gap = values[ca][metric] - values[cb][metric]
                clean_dist = distributions[ca][metric] - distributions[cb][metric]
                dist = seed_mean([distributions[a(s)][metric] - distributions[b(s)][metric]
                                  for s in seeds(sigma)]) - clean_dist
                features.append({"comparison": f"{ga} minus {gb}", **row,
                                 "degradation_difference": row["difference_seed_mean"] - clean_gap,
                                 "degradation_ci_low": float(np.percentile(dist, 2.5)),
                                 "degradation_ci_high": float(np.percentile(dist, 97.5))})
    return smoothing, interactions, features


def reproduce_bootstrap(values, distributions):
    rows = []
    reference = pd.read_csv(EXP / "metrics/stage5_bootstrap_all.csv")
    for group, smoother, sigma in itertools.product(GROUPS, SMOOTHERS, SIGMAS):
        row = {"group": group, "smoother": smoother, "sigma": sigma}
        original = reference[(reference.group == group) & (reference.smoother == smoother)
                             & (reference.sigma == sigma)].iloc[0]
        for metric in ("macro_f1", "accuracy"):
            per_seed = np.array([values[(group, smoother, sigma, s)][metric] for s in seeds(sigma)])
            dist = seed_mean([distributions[(group, smoother, sigma, s)][metric]
                              for s in seeds(sigma)])
            clean = distributions[(group, smoother, 0, 0)][metric]
            entries = {f"{metric}_seed_mean": per_seed.mean(),
                       f"{metric}_seed_sd": per_seed.std(ddof=1) if sigma else 0.0,
                       f"{metric}_ci_low": np.percentile(dist, 2.5),
                       f"{metric}_ci_high": np.percentile(dist, 97.5),
                       f"{metric}_delta_from_clean":
                       per_seed.mean() - values[(group, smoother, 0, 0)][metric],
                       f"{metric}_delta_ci_low": np.percentile(dist - clean, 2.5),
                       f"{metric}_delta_ci_high": np.percentile(dist - clean, 97.5)}
            for field, val in entries.items():
                np.testing.assert_allclose(val, original[field], atol=1e-12, rtol=0)
            row.update(entries)
        rows.append(row)
    return write_csv("paired_summary_reproduction", rows)


def report(summary, contrasts, features, flows, sizes, zero_boundary_draws):
    def link(label, path):
        return f"[{label}](<{path.as_posix()}>)"

    raw_rows = []
    for _, r in summary[summary.smoother == SMOOTHERS[0]].iterrows():
        raw_rows.append([r["group"], int(r.sigma),
                         f"{r.macro_f1_seed_mean:.4f} ± {r.macro_f1_seed_sd:.4f}",
                         f"{r.macro_f1_delta_from_clean:+.4f}",
                         f"[{r.macro_f1_ci_low:.4f}, {r.macro_f1_ci_high:.4f}]"])
    comparison_rows, flow_rows = [], []
    for sigma, smoother in itertools.product((0, 20), SMOOTHERS[1:]):
        q = contrasts[(contrasts.sigma == sigma) &
                      (contrasts.comparison == f"{smoother} minus raw")]
        entries = [sigma, "S1 居中" if smoother == SMOOTHERS[1] else "S2 因果"]
        for metric in ("macro_f1", "accuracy", "transition_recall"):
            r = q[q.metric == metric].iloc[0]
            entries.append(f"{r.difference_seed_mean:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]")
        comparison_rows.append(entries)
        f = flows[(flows.group == "G3") & (flows.sigma == sigma) &
                  (flows.smoother == smoother)].iloc[0]
        flow_rows.append([sigma, entries[1], f"{f.rescued_points_mean:,.1f}",
                          f"{f.damaged_points_mean:,.1f}", f"{f.net_correct_points_mean:+,.1f}",
                          f"{f.accuracy_delta_from_raw_mean * 100:+.3f}",
                          f"{f.predicted_train_point_fraction_mean:.2%}"])
    g = features[(features.comparison == "G3 minus G2") & (features.sigma == 20)
                 & (features.metric == "macro_f1")].iloc[0]
    largest = sizes.sort_values("points", ascending=False).iloc[0]
    text = f"""# 噪声实验交接：主线、配对证据与汇报口径

日期：2026-11-23（Asia/Shanghai）。本次仅补全事后分析与交付，未训练模型、改协议或重跑坐标扰动。

## 1. 当前做到哪里

主线“定位噪声 → 特征组合抗噪比较 → 分数平滑与切换代价”已经跑完：
16 份共享特征缓存、3 个冻结模型、48 份 decision 分数和 144 组平滑指标；
原有六张诊断图及逐窗、逐边界数据均已存在。本轮进一步补齐配对比较与平滑纠正/破坏分解。
原始主实验和旧诊断文件均做运行前后哈希核对，检查范围见 acceptance.json。

本轮不更改 G1/G2/G3、clean-training/noisy-test、用户划分、50 点窗口、
噪声档位/配对种子、模型超参数、k=11 或既定评价。G4 仍跳过。

## 2. 原主结果：绝对分数与自身退化一起看

{markdown(["特征组", "单轴 sigma/m", "raw Macro-F1：均值 ± 种子SD", "noisy-clean", "用户组bootstrap 95%区间"], raw_rows)}

sigma=0 仅一份无附加噪声结果，不是五次重复，更不是真实无误差坐标。
5 个种子只是扰动波动；区间是 9 个关联用户组的 5000 次配对重采样，
每次先算各种子的固定五类 F1，再取种子均值，不先混合种子混淆矩阵。
最大组 {largest.linked_group}（{largest.users}）占测试点 {largest.point_share:.4%}；
这些区间不是对任意人群泛化的保证。

G3 在无附加噪声下最高，20m 下 raw 则为 G2 的点估计最高。
但 G3−G2 的 20m **直接分数差**为 {g.difference_seed_mean:+.6f}
（配对区间 [{g.ci_low:+.6f}, {g.ci_high:+.6f}]）；
两者**相对自身 clean 的变化差**为 {g.degradation_difference:+.6f}
（配对区间 [{g.degradation_ci_low:+.6f}, {g.degradation_ci_high:+.6f}]）。
这两个量回答不同问题，不能用后一个代替前一个，也不能仅据点估计宣布普遍抗噪排名。
G3−G2 同时新增 12 个特征，不能把差异单独归因于 jerk 或转向。

## 3. 平滑效果要做配对比较

下表为 G3 各平滑相对同条件 raw 的差，方括号是用户组配对 bootstrap 95% 区间，
不是种子 SD，也不是两组各自区间相减。完整 0/5/10/20m 结果在 CSV。
这里的 clean 收益区间采用本噪声研究的固定五类计分，因此不直接沿用旧交接稿的
按出现类别计算的 bootstrap 区间；这不是换了模型或改了本次评价协议。

{markdown(["sigma/m", "处理−raw", "Macro-F1差 [区间]", "Accuracy差 [区间]", "切换Recall差 [区间]"], comparison_rows)}

无附加噪声时，S1 的点级分类收益伴随边界召回损失；
20m 时两种固定平滑的全测试集分类点估计均低于 raw。
不能把错误孤立跳变更少解释成已经恢复了运动特征或提升了抗噪性。
`paired_smoothing_interactions.csv` 另列“平滑收益随噪声的变化”，避免将 clean 的收益外推到 noisy。

边界仍是窗口多数标签真值、同 block 内 ±1 窗口、class-blind 的一对一最大匹配；
这里复用已保存的逐用户匹配计数，不引入按类别方向匹配的新指标。
边界指标以抽样后匹配/预测/真实计数重算，而不是平均各用户比率。
{zero_boundary_draws}/5000 个抽样没有真实边界；分母为零时沿用主实现的 0 记分，
这一约定可能影响边界区间，不能把区间作为普遍实时切换性能的保证。

## 4. 平滑具体纠正/破坏了多少点

对每个窗口，若平滑改了预测，真实类别属于新预测的点为“纠正”，
属于原预测的点为“破坏”，其余点仍然错。使用 cnt_* 点数，不按多数标签给整窗判对错。
`纠正−破坏 = 新增正确点数 = Accuracy差 × 总测试点数`，全 96 组逐条件校验。
下表每行先逐种子计算再平均，20m 点数因此可能不是整数；它们不是独立样本量。

{markdown(["sigma/m", "处理", "纠正点数均值", "破坏点数均值", "净正确点数", "Accuracy变化/百分点", "预测train点占比"], flow_rows)}

此分解是 Accuracy 的恒等式，不是 Macro-F1 的加法分解。
混合标签窗口内可能同时纠正一些点、破坏另一些点；不能把整窗简单叫作“修复成功”。
全三模型、全部条件分解见 smoothing_point_flows.csv；占比高也不等于预测正确。

## 5. 可以用于正文的解释链

1. 定位压力测试：额外 E/N 独立高斯扰动下，全测试集的冻结模型性能点估计明显退化；不把人工噪声当真实 GPS 误差标定。
2. 特征偏移：旧诊断已量化速度、加速度、jerk 与 straightness 的偏移，采样间隔分层只能作关联性解释。
3. 表示变化：冻结 1000 基点的 RBF 相似度及映射范数衰减，伴随接近截距决策及预测集中；这支持部分失效机制而非唯一因果解释。
4. 平滑代价：clean 下分类改善不意味着边界保留或 noisy 下改善；结合本轮配对差、纠正/破坏，以及原短模式丢失表讨论。

推荐报告正文保留原三张主图和协议/汇总表，将六张机制诊断、配对差与用户组敏感性放附录。
不要只选曲线好看的档位，不把 5×13 次扰动计分说成 65 个独立用户，不据测试结果重新选择模型。
新增比较属于事后描述、未做多重比较校正，不用它们宣布显著性排名。

## 6. 已知限制与未完成事项

- 加的是逐点独立 E/N 高斯噪声，sigma 是单方向标准差，水平 RMS=sqrt(2)×sigma；不覆盖真实时序相关误差。
- 平滑只在 decision 分数域，分数不是概率；S1 用未来最多 5 个窗口，S2 只用历史但仍需等当前窗口完成。
- 继承 v2 特征定义：acc_std 为有符号加速度标准差；path_length 的入窗边与 duration 的时间支持不完全一致。本轮保持冻结定义，未声称已修正。
- 新噪声统计固定五类计分，原 Part 2 部分用户/重采样指标按出现类别计分；不可直接混用其 CI 或用户均分。
- 窗口真值共有 529 个边界，不等同于点级 658 个切换。误差案例不是事件频率估计。
- 原 v2 raw 复现仅有 1/17380 个窗口不同，居中结果逐窗相同；仍需保留这一复现偏差说明。
- 原图自动布局/像素核验已做，但人工视觉复核仍未完成；本轮查看工具同样不支持图像输入，未伪改为通过。
- 不新增 G4、相关噪声、GPS 坐标滤波、训练增强或 gamma/k 调优；这些只能在另行批准的新协议中做。

## 7. 文件与复现入口

- {link("原主实验报告", EXP / "FINAL_REPORT.md")}
- {link("原中文机制诊断与六图", EXP / "diagnostics/DIAGNOSTIC_REPORT.md")}
- {link("诊断来源与文献范围", EXP / "diagnostics/SOURCES.md")}
- {link("冻结实验协议", EXP / "protocol.json")}
- {link("本轮说明及运行命令", HERE / "README.md")}
- {link("配对平滑差", HERE / "tables/paired_smoothing_contrasts.csv")}
- {link("特征组直接差与退化差", HERE / "tables/paired_feature_contrasts.csv")}
- {link("纠正与破坏逐条件表", HERE / "tables/smoothing_point_flows.csv")}
- {link("本轮验收与输入不变性", HERE / "acceptance.json")}

流程辅助：Scientific Agent Skills 的 scientific-critical-thinking，仅用于检查抽样单位、
配对推断和结论边界；它不是数值正确性的认证。软件引用：
Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents.
arXiv:2609.00065, DOI: 10.48550/arXiv.2609.00065。
本轮直接 HTTP 核对 arXiv 作者、标题、年份与当前修订记录（2026-09-02）；
网络检索工具未返回正文，未据此新增 GPS 文献结论。
"""
    (HERE / "NOISE_HANDOVER.md").write_text(text, encoding="utf-8")


def main():
    started = time.perf_counter()
    (HERE / "tables").mkdir(exist_ok=True)
    acceptance = HERE / "acceptance.json"
    if acceptance.exists():
        write_json(acceptance, {"passed": False, "status": "rebuilding"})
    files = protected_files()
    before = {str(p.relative_to(ROOT)): sha(p) for p in files}
    write_json(HERE / "input_manifest.json", {
        "created_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "scope": "post-hoc paired analysis; no fitting or new experimental conditions",
        "python": sys.version, "analysis_sha256": sha(Path(__file__)),
        "readme_sha256": sha(HERE / "README.md"), "files": before})
    protocol = json.loads((EXP / "protocol.json").read_text())
    assert tuple(protocol["sigmas_m"]) == SIGMAS and tuple(protocol["seeds"]) == SEEDS
    assert tuple(protocol["smoothing"]) == SMOOTHERS and protocol["g4"] == "skipped by protocol"
    base = read_prediction("G3", 0, 0)
    assert len(base) == 17380 and base.n_points.sum() == 860972
    counts = base[[f"cnt_{m}" for m in MODES]].to_numpy(np.int64)
    np.testing.assert_array_equal(counts.sum(axis=1), base.n_points)
    group_ids = np.asarray(cv_groups(base.user_id, ROOT / "data/splits/report.json"))
    with np.load(EXP / "metrics/paired_bootstrap_draws.npz", allow_pickle=True) as saved:
        draws, clusters = saved["draws"], saved["clusters"].astype(str)
    assert draws.shape == (5000, 9) and (draws.sum(axis=1) == 9).all()
    np.testing.assert_array_equal(clusters, pd.unique(group_ids))
    sizes = write_csv("linked_group_sizes", [
        {"linked_group": c, "users": ",".join(pd.unique(base.loc[group_ids == c, "user_id"])),
         "windows": int((group_ids == c).sum()),
         "points": int(base.loc[group_ids == c, "n_points"].sum()),
         "point_share": base.loc[group_ids == c, "n_points"].sum() / base.n_points.sum()}
        for c in clusters])
    original = pd.read_csv(EXP / "metrics/all_metrics.csv")
    users = pd.read_csv(EXP / "metrics/per_user_results.csv", dtype={"user_id": str})
    users["linked_group"] = cv_groups(users.user_id, ROOT / "data/splits/report.json")
    values, distributions, flows = {}, {}, []
    zero_boundary_draws = 0
    for group in GROUPS:
        clean = read_prediction(group, 0, 0)
        for sigma in SIGMAS:
            for seed in seeds(sigma):
                d = read_prediction(group, sigma, seed)
                pd.testing.assert_frame_equal(d[META], base[META])
                for smoother in SMOOTHERS:
                    key = (group, smoother, sigma, seed)
                    pred = d[smoother].to_numpy()
                    cms = np.stack([confusion(counts[group_ids == c], pred[group_ids == c])
                                    for c in clusters])
                    sampled_cms = np.einsum("bg,gij->bij", draws, cms)
                    v = {k: float(x) for k, x in point_metrics(cms.sum(axis=0)).items()}
                    b = point_metrics(sampled_cms)
                    u = users[(users.group == group) & (users.smoother == smoother)
                              & (users.sigma == sigma) & (users.seed == seed)]
                    assert len(u) == 13
                    totals = u.groupby("linked_group")[[
                        "true_transitions", "pred_transitions", "matched_transitions",
                        "isolated_flips_wrong", "windows", "points"]].sum().reindex(clusters)
                    assert totals.notna().all().all()
                    t, p, h, f, w, n = totals.to_numpy().T
                    v.update({k: float(x) for k, x in boundary_metrics(t.sum(), p.sum(), h.sum()).items()})
                    b.update(boundary_metrics(draws @ t, draws @ p, draws @ h))
                    assert t.sum() == 529
                    zero_boundary_draws = int(((draws @ t) == 0).sum())
                    v["wrong_isolated_flips_per100_windows"] = f.sum() / w.sum() * 100
                    b["wrong_isolated_flips_per100_windows"] = (draws @ f) / (draws @ w) * 100
                    change = pred != clean[smoother].to_numpy()
                    change_counts = np.array([base.n_points.to_numpy()[change & (group_ids == c)].sum()
                                              for c in clusters])
                    v["prediction_change_from_clean_point_fraction"] = change_counts.sum() / n.sum()
                    b["prediction_change_from_clean_point_fraction"] = (draws @ change_counts) / (draws @ n)
                    ref = original[(original.group == group) & (original.smoother == smoother)
                                   & (original.sigma == sigma) & (original.seed == seed)].iloc[0]
                    for metric in METRICS:
                        np.testing.assert_allclose(v[metric], ref[metric], atol=1e-12, rtol=0)
                        assert np.isfinite(b[metric]).all()
                    values[key], distributions[key] = v, b
                    if smoother != SMOOTHERS[0]:
                        flow = point_flow(counts, d[SMOOTHERS[0]], pred)
                        raw_acc = float(point_metrics(confusion(counts, d[SMOOTHERS[0]]))["accuracy"])
                        np.testing.assert_allclose(flow["accuracy_delta_from_raw"],
                                                   v["accuracy"] - raw_acc, atol=1e-12, rtol=0)
                        assert flow["changed_still_wrong_points"] >= 0
                        flows.append({"group": group, "sigma": sigma, "seed": seed,
                                      "smoother": smoother, **flow,
                                      "predicted_train_point_fraction":
                                      float(base.n_points[pred == "train"].sum() / base.n_points.sum())})
                print(f"Reconciled {group} {name(sigma, seed)}", flush=True)
    summary = reproduce_bootstrap(values, distributions)
    smoothing, interactions, features = build_contrasts(values, distributions)
    smooth_df = write_csv("paired_smoothing_contrasts", smoothing)
    feature_df = write_csv("paired_feature_contrasts", features)
    write_csv("paired_smoothing_interactions", interactions)
    flow_df = write_csv("smoothing_point_flows", flows)
    cols = [c for c in flow_df.select_dtypes("number").columns if c not in ("sigma", "seed")]
    grouped = flow_df.groupby(["group", "sigma", "smoother"], sort=False)
    means = grouped[cols].mean().add_suffix("_mean")
    sds = grouped[cols].std(ddof=1).fillna(0).add_suffix("_seed_sd")
    flow_summary = write_csv("smoothing_point_flows_summary", means.join(sds).reset_index())
    # The prespecified raw degradation comparison must match the primary table exactly.
    ref = pd.read_csv(EXP / "metrics/paired_group_delta_contrasts.csv")
    for _, row in feature_df[(feature_df.sigma == 20) & (feature_df.metric == "macro_f1")].iterrows():
        target = ref[(ref.comparison == row.comparison) & (ref.smoother == SMOOTHERS[0])].iloc[0]
        np.testing.assert_allclose([row.degradation_difference, row.degradation_ci_low, row.degradation_ci_high],
                                   [target.delta_difference, target.ci_low, target.ci_high],
                                   atol=1e-12, rtol=0)
    assert (len(smooth_df), len(feature_df), len(flows), len(interactions)) == (56, 24, 96, 16)
    report(summary, smooth_df, feature_df, flow_summary, sizes, zero_boundary_draws)
    after = {str(p.relative_to(ROOT)): sha(p) for p in files}
    assert before == after, "A protected original file changed"
    assert set(files) == set(protected_files()), "Protected file set changed during analysis"
    write_json(acceptance, {
        "passed": True, "completed_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "protected_original_files": len(files), "changed_original_files": [],
        "point_metric_rows_reconciled": len(values), "metrics_per_row": len(METRICS),
        "bootstrap_rows_reproduced": len(summary), "draws": 5000, "linked_groups": 9,
        "smoothing_comparisons": len(smooth_df), "feature_comparisons": len(feature_df),
        "point_flow_identities_checked": len(flows), "smoothing_interactions": len(interactions),
        "raw_primary_degradation_contrasts_reproduced": 3,
        "zero_true_boundary_draws": zero_boundary_draws,
        "training_performed": False, "experimental_protocol_changed": False,
        "new_random_numbers_generated": False, "manual_figure_review_completed": False})
    write_json(HERE / "output_inventory.json", {
        str(p.relative_to(HERE)): {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(HERE.rglob("*")) if p.is_file()
        and "__pycache__" not in p.parts and p.name != "output_inventory.json"})
    print(acceptance.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
