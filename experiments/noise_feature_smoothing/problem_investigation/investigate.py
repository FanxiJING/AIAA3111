"""Read frozen results and explain failure mechanisms without modifying inputs."""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import euclidean_distances
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
SUP = EXP / "supplementary_experiments"
sys.path.insert(0, str(SUP))
from run_supplement import (
    MODES, SEEDS, SMOOTHERS, build_features, displace, ou_from_innovations,
    read_table, smooth_soft, trailing_soft,
)
from run_experiment import one_to_one_transitions, perturb
from build_analysis import point_metrics

SCORE = [f"score_{m}" for m in MODES]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(name, data):
    (HERE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                      default=str), encoding="utf-8")


def protected():
    paths = [p for folder in ("src", "tests", "data", "docs", "experiments")
             for p in (ROOT / folder).rglob("*") if p.is_file()
             and HERE not in p.parents and "__pycache__" not in p.parts
             and "matplotlib_cache" not in p.parts and p.suffix != ".pyc"
             and p != ROOT / "docs/aigc_log_part2.md"]
    paths += [ROOT / name for name in ("README.md", "CLAUDE.md", "pyproject.toml", "uv.lock")]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def conditions():
    yield "clean", 0, 0, EXP / "feature_cache/sigma00_seed0.csv", (
        EXP / "predictions/G3_sigma00_seed0_smoothed.csv")
    for noise in ("iid", "OU30"):
        for sigma in (5, 10, 20):
            for seed in SEEDS:
                cond = f"sigma{sigma:02d}_seed{seed}"
                if noise == "iid":
                    cache = EXP / f"feature_cache/{cond}.csv"
                    pred = EXP / f"predictions/G3_{cond}_smoothed.csv"
                else:
                    cache = SUP / f"feature_cache/OU30_{cond}.csv"
                    pred = SUP / f"predictions/G3_originalgamma_OU30_{cond}.csv"
                yield noise, sigma, seed, cache, pred


def match_boundaries(truth, pred, tol=1, typed=False):
    """Maximum count, then minimum absolute offset; dummy columns allow misses."""
    ti = np.flatnonzero(truth[1:] != truth[:-1]) + 1
    pi = np.flatnonzero(pred[1:] != pred[:-1]) + 1
    if not len(ti) or not len(pi):
        return ti, pi, []
    delta = pi[None, :] - ti[:, None]
    valid = np.abs(delta) <= tol
    if typed:
        valid &= ((truth[ti - 1, None] == pred[None, pi - 1]) &
                  (truth[ti, None] == pred[None, pi]))
    penalty = (min(len(ti), len(pi)) + 1) * (tol + 1)
    costs = np.full((len(ti), len(pi) + len(ti)), float(penalty))
    costs[:, :len(pi)] = np.where(valid, np.abs(delta), penalty * 3)
    row, col = linear_sum_assignment(costs)
    pairs = [(int(ti[i]), int(pi[j])) for i, j in zip(row, col)
             if j < len(pi) and valid[i, j]]
    return ti, pi, pairs


def known_answers():
    tests = []
    truth = np.array(["walk", "walk", "bike", "bike", "bus"], object)
    pred = np.array(["walk", "bike", "bike", "bus", "bus"], object)
    assert len(match_boundaries(truth, pred)[2]) == 2
    assert len(match_boundaries(truth, pred, typed=True)[2]) == 2
    false_types = np.array(["car", "car", "train", "train", "walk"], object)
    assert len(match_boundaries(truth, false_types)[2]) == 2
    assert len(match_boundaries(truth, false_types, typed=True)[2]) == 0
    tests.append("one-to-one count and class-pair separation")
    b = np.array(["a"] * 40 + ["b"] * 40)
    ts = pd.date_range("2020-01-01", periods=80, freq="s")
    scores = np.zeros((80, 5))
    scores[:, 0] = 1
    scores[15:20, 0], scores[15:20, 1] = 0, 1
    center = smooth_soft(scores, b, ts, 11)
    assert not (center[15:20] == MODES[1]).any()
    scores[20, 0], scores[20, 1] = 0, 1
    assert (smooth_soft(scores, b, ts, 11)[15:21] == MODES[1]).all()
    tests.append("equal-score five-window erasure versus six-window retention")
    scores = np.zeros((80, 5))
    scores[:20, 0] = 1
    scores[20:, 1] = 1
    causal = trailing_soft(scores, b, ts, 11)
    assert np.flatnonzero(causal[:40] == MODES[1])[0] == 25
    changed = scores.copy()
    changed[31:] = -10
    np.testing.assert_array_equal(causal[:31], trailing_soft(changed, b, ts, 11)[:31])
    tests.append("causal five-window lag and no future read")
    p = pd.DataFrame({
        "user_id": ["001"] * 5, "trip_id": ["t"] * 5,
        "feature_block_id": ["a"] * 4 + ["b"],
        "timestamp": pd.to_datetime(["2020-01-01 00:00:00", "2020-01-01 00:00:01",
                                     "2020-01-01 00:00:03", "2020-01-01 00:00:04",
                                     "2020-01-01 00:00:05"]),
        "latitude": np.zeros(5),
        "longitude": np.array([0, 10, 30, 50, 60]) / 6371008.8 * 180 / np.pi,
    })
    motion = build_features(p)
    np.testing.assert_allclose(motion.speed_mps.iloc[1:4], [10, 10, 20], atol=1e-7)
    np.testing.assert_allclose(motion.acceleration_mps2.iloc[2:4], [0, 10], atol=1e-7)
    np.testing.assert_allclose(motion.jerk_mps3.iloc[3], 10, atol=1e-7)
    assert np.isnan(motion.speed_mps.iloc[4])
    tests.append("metres, irregular dt, derivatives and block reset")
    field = np.tile([3.0, 4.0], (5, 1))
    shifted = displace(p, field, 1)
    lat, lon = np.radians(shifted.latitude), np.radians(shifted.longitude - p.longitude)
    distance = 2 * 6371008.8 * np.arcsin(np.sqrt(
        np.sin(lat / 2) ** 2 + np.cos(lat) * np.sin(lon / 2) ** 2))
    np.testing.assert_allclose(distance, 5, atol=1e-7)
    perturbed, _ = perturb(p, 20, SEEDS[0])
    z = np.random.default_rng(np.random.SeedSequence([SEEDS[0], 1])).standard_normal((5, 2))
    pd.testing.assert_frame_equal(perturbed, displace(p, z, 20))
    tests.append("spherical E/N displacement and identical iid innovations")
    z = np.array([[1., 2.], [3., 4.], [5., 6.]])
    q = ou_from_innovations(z, [0, 2, 7], ["a", "a", "b"])
    a = np.exp(-2 / 30)
    np.testing.assert_allclose(q[1], a * z[0] + np.sqrt(1 - a*a) * z[1])
    np.testing.assert_array_equal(q[2], z[2])
    tests.append("OU irregular transition and stationary block reset")
    write_json("known_answer_tests.json", {"passed": True, "checks": tests})


def kernel_diagnostics():
    est = joblib.load(EXP / "models/full.joblib")
    cols = json.loads((EXP / "models/G3_config.json").read_text())["feature_columns"]
    train = pd.read_csv(ROOT / "src/model_training/windows_w50.csv")
    train = train.loc[(train.split == "train") & (train.n_points >= 10), cols]
    assert len(train) == 70086
    imp, scale, rbf, clf = (est.named_steps[n] for n in ("impute", "scale", "rbf", "clf"))
    transformed_train = scale.transform(imp.transform(train))
    lower, upper = np.quantile(transformed_train, [.005, .995], axis=0)
    weights = rbf.normalization_.T @ clf.coef_.T
    align = pd.Index(est.classes_).get_indexer(MODES)
    intercept = clf.intercept_
    intercept_class = est.classes_[intercept.argmax()]
    rows, feature_rows, checks = [], [], []
    for noise, sigma, seed, cache, pred_path in conditions():
        d, pred = read_table(cache), read_table(pred_path)
        z = scale.transform(imp.transform(d[cols]))
        if not sigma:
            clean_z = z.copy()
        delta = z - clean_z
        maxk, minsq, contribution, reconstructed = [], [], [], []
        for start in range(0, len(z), 1024):
            dist = euclidean_distances(z[start:start+1024], rbf.components_, squared=True)
            kernel = np.exp(-rbf.gamma * dist)
            part = kernel @ weights
            maxk.extend(kernel.max(axis=1))
            minsq.extend(dist.min(axis=1))
            contribution.extend(np.linalg.norm(part, axis=1))
            reconstructed.extend(part + intercept)
        maxk, minsq = np.array(maxk), np.array(minsq)
        contribution, reconstructed = np.array(contribution), np.array(reconstructed)
        np.testing.assert_allclose(reconstructed[:, align], pred[SCORE], atol=2e-7, rtol=1e-7)
        ix = np.unique(np.linspace(0, len(z)-1, 64, dtype=int))
        direct = est.decision_function(d.loc[ix, cols])
        np.testing.assert_allclose(reconstructed[ix], direct, atol=2e-7, rtol=1e-7)
        votes = np.array(est.classes_)[reconstructed.argmax(axis=1)]
        np.testing.assert_array_equal(votes, pred.raw)
        counts = d.n_points.to_numpy()
        nearzero = maxk < 1e-6
        norm = np.linalg.norm(z, axis=1)
        info = dict(noise=noise, sigma=sigma, seed=seed)
        rows.append({
            **info, "windows": len(d), "points": int(counts.sum()),
            "scaled_norm_p50": np.median(norm), "scaled_norm_p95": np.quantile(norm, .95),
            "nearest_landmark_sqdist_p50": np.median(minsq),
            "max_kernel_p50": np.median(maxk), "max_kernel_p05": np.quantile(maxk, .05),
            "max_kernel_below_1em6_window_fraction": nearzero.mean(),
            "max_kernel_below_1em6_point_fraction": counts[nearzero].sum() / counts.sum(),
            "decision_minus_intercept_norm_p50": np.median(contribution),
            "intercept_class": intercept_class,
            "intercept_class_point_fraction": counts[votes == intercept_class].sum() / counts.sum(),
            "nearzero_intercept_class_fraction": (
                (votes[nearzero] == intercept_class).mean() if nearzero.any() else np.nan),
            "any_training_tail_exceedance": ((z < lower) | (z > upper)).any(axis=1).mean(),
        })
        for j, col in enumerate(cols):
            feature_rows.append({
                **info, "feature": col,
                "train_raw_p005": np.quantile(train[col].dropna(), .005),
                "train_raw_p995": np.quantile(train[col].dropna(), .995),
                "test_raw_p50": d[col].median(), "test_raw_p95": d[col].quantile(.95),
                "shift_scaled_abs_p50": np.median(np.abs(delta[:, j])),
                "shift_scaled_abs_p95": np.quantile(np.abs(delta[:, j]), .95),
                "outside_train_central99_fraction": ((z[:, j] < lower[j]) | (z[:, j] > upper[j])).mean(),
            })
        checks.append({**info, "max_saved_score_error": float(np.max(np.abs(
            reconstructed[:, align] - pred[SCORE].to_numpy()))), "pipeline_samples": len(ix)})
        print("kernel", info, flush=True)
    pd.DataFrame(rows).to_csv(HERE / "kernel_by_seed.csv", index=False)
    pd.DataFrame(rows).drop(columns="seed").groupby(["noise", "sigma"]).mean(numeric_only=True).reset_index().to_csv(
        HERE / "kernel_summary.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(HERE / "feature_shift_by_seed.csv", index=False)
    pd.DataFrame(feature_rows).drop(columns="seed").groupby(["noise", "sigma", "feature"]).mean(
        numeric_only=True).reset_index().to_csv(HERE / "feature_shift_summary.csv", index=False)
    write_json("score_reconstruction_checks.json", checks)
    write_json("model_algebra.json", {
        "intercept_class": intercept_class, "intercept_estimator_order": est.classes_.tolist(),
        "intercept": intercept.tolist(), "formula": "K(X,landmarks) @ normalization.T @ coef.T + intercept",
        "gamma": rbf.gamma, "landmarks": len(rbf.components_),
    })


def group_decomposition():
    with np.load(SUP / "diagnostics/cluster_confusions.npz") as a:
        keys, cms, clusters, supports = a["keys"], a["cms"], a["clusters"], a["supports"]
    rows, cells = [], []
    for key, cm in zip(keys, cms):
        family, group, noise, sigma, seed, smoother = key.split("|")
        if group != "G3" or smoother != SMOOTHERS[0] or family not in ("original", "correlated_noise"):
            continue
        if sigma == "0" and family != "original":
            continue
        info = dict(noise="clean" if sigma == "0" else noise, sigma=int(sigma), seed=int(seed))
        for slice_name, mask in [
            ("all", np.ones(len(clusters), bool)), ("without_linked0", clusters != "linked0"),
            *[(str(c), clusters == c) for c in clusters],
        ]:
            matrix = cm[mask].sum(axis=0)
            support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
            for j, mode in enumerate(MODES):
                den = support[j] + predicted[j]
                rows.append({
                    **info, "slice": slice_name, "class": mode,
                    "support": int(support[j]), "predicted": int(predicted[j]),
                    "tp": int(matrix[j, j]),
                    "f1": 2 * matrix[j, j] / den if den else 0,
                    "recall": matrix[j, j] / support[j] if support[j] else 0,
                    "precision": matrix[j, j] / predicted[j] if predicted[j] else 0,
                    "slice_macro_f1": float(point_metrics(matrix)["macro_f1"]),
                })
                for k, guessed in enumerate(MODES):
                    cells.append({**info, "slice": slice_name, "true": mode,
                                  "predicted": guessed, "points": int(matrix[j, k])})
    d = pd.DataFrame(rows)
    d.to_csv(HERE / "group_class_by_seed.csv", index=False)
    means = d.drop(columns="seed").groupby(["noise", "sigma", "slice", "class"]).mean(numeric_only=True).reset_index()
    means.to_csv(HERE / "group_class_summary.csv", index=False)
    pd.DataFrame(cells).to_csv(HERE / "confusion_cells_by_seed.csv", index=False)
    contrasts = []
    for sigma in (5, 10, 20):
        for group_slice in means.slice.unique():
            sub = means[(means.sigma == sigma) & (means.slice == group_slice)]
            ou, iid = sub[sub.noise == "OU30"].set_index("class"), sub[sub.noise == "iid"].set_index("class")
            for mode in MODES:
                contrasts.append({
                    "sigma": sigma, "slice": group_slice, "class": mode,
                    "support": int(ou.loc[mode, "support"]),
                    "OU_f1": ou.loc[mode, "f1"], "iid_f1": iid.loc[mode, "f1"],
                    "contribution_to_macro_gap": (ou.loc[mode, "f1"] - iid.loc[mode, "f1"]) / 5,
                })
    pd.DataFrame(contrasts).to_csv(HERE / "class_gap_decomposition.csv", index=False)
    weights = supports.sum(axis=1) / supports.sum()
    write_json("group_composition.json", {
        "clusters": clusters.tolist(), "point_weights": weights.tolist(),
        "inverse_squared_weight_sum": float(1 / np.sum(weights ** 2)),
        "weight_concentration_note": "Kish-style descriptive concentration, NOT number of independent groups",
    })


def boundary_diagnostics():
    summary, events, segments, spans = [], [], [], []
    base = read_table(EXP / "predictions/G3_sigma00_seed0_smoothed.csv")
    for block, sub in base.groupby("feature_block_id", sort=False):
        sub = sub.sort_values("start_time", kind="stable").reset_index(drop=True)
        n = len(sub)
        for i in range(n):
            for smoother in SMOOTHERS[1:]:
                lo = max(0, i - (5 if smoother == SMOOTHERS[1] else 10))
                hi = min(n-1, i + (5 if smoother == SMOOTHERS[1] else 0))
                spans.append({
                    "block": block, "smoother": smoother, "support_windows": hi-lo+1,
                    "support_seconds": (sub.end_time.iloc[hi]-sub.start_time.iloc[lo]).total_seconds(),
                    "future_start_seconds": (sub.start_time.iloc[hi]-sub.start_time.iloc[i]).total_seconds(),
                    "history_start_seconds": (sub.start_time.iloc[i]-sub.start_time.iloc[lo]).total_seconds(),
                })
    pd.DataFrame(spans).to_csv(HERE / "smoothing_support_spans.csv", index=False)
    for noise, sigma, seed, _, predpath in conditions():
        d = read_table(predpath)
        for smoother in SMOOTHERS:
            info = dict(noise=noise, sigma=sigma, seed=seed, smoother=smoother)
            nt = npred = hits = compatible = extended = 0
            for block, sub in d.groupby("feature_block_id", sort=False):
                sub = sub.sort_values("start_time", kind="stable").reset_index(drop=True)
                truth, raw, pred = (sub[c].to_numpy() for c in ("label", SMOOTHERS[0], smoother))
                ti, pi, pairs = match_boundaries(truth, pred, tol=1)
                _, _, typed = match_boundaries(truth, pred, tol=10, typed=True)
                nt += len(ti)
                npred += len(pi)
                hits += len(pairs)
                extended += len(typed)
                for target, detected in pairs:
                    pair_ok = truth[target-1] == pred[detected-1] and truth[target] == pred[detected]
                    compatible += int(pair_ok)
                    events.append({
                        **info, "block": block, "matching": "primary_classblind_tol1",
                        "true_index": target, "pred_index": detected, "pair_compatible": bool(pair_ok),
                        "lag_windows": detected-target, "lag_seconds": (
                            sub.start_time.iloc[detected]-sub.start_time.iloc[target]).total_seconds(),
                    })
                for target, detected in typed:
                    events.append({
                        **info, "block": block, "matching": "diagnostic_typed_tol10",
                        "true_index": target, "pred_index": detected, "pair_compatible": True,
                        "lag_windows": detected-target, "lag_seconds": (
                            sub.start_time.iloc[detected]-sub.start_time.iloc[target]).total_seconds(),
                    })
                starts = np.r_[0, ti]
                stops = np.r_[ti, len(sub)]
                for start, stop in zip(starts, stops):
                    label = truth[start]
                    has_raw = bool((raw[start:stop] == label).any())
                    has_smooth = bool((pred[start:stop] == label).any())
                    length = int(stop-start)
                    segments.append({
                        **info, "block": block, "start": int(start), "label": label,
                        "windows": length, "band": "<=5" if length <= 5 else ("6-10" if length <= 10 else ">=11"),
                        "interior": bool(start > 0 and stop < len(sub)),
                        "duration_seconds": (sub.end_time.iloc[stop-1]-sub.start_time.iloc[start]).total_seconds(),
                        "raw_label_present": has_raw, "label_present": has_smooth,
                        "lost_from_raw": has_raw and not has_smooth,
                        "gained_from_raw": has_smooth and not has_raw,
                        "window_correct_fraction": float((pred[start:stop] == label).mean()),
                    })
            ref = one_to_one_transitions(d, d[smoother], d.label)
            assert nt == 529 and hits == ref["matched_transitions"] and npred == ref["pred_transitions"]
            summary.append({**info, "true_boundaries": nt, "pred_boundaries": npred,
                            "primary_hits": hits, "primary_pair_compatible_hits": compatible,
                            "diagnostic_typed_tol10_hits": extended})
        print("boundary", noise, sigma, seed, flush=True)
    pd.DataFrame(summary).to_csv(HERE / "boundary_by_seed.csv", index=False)
    pd.DataFrame(summary).drop(columns="seed").groupby(["noise", "sigma", "smoother"]).mean(
        numeric_only=True).reset_index().to_csv(HERE / "boundary_summary.csv", index=False)
    pd.DataFrame(events).to_csv(HERE / "boundary_events.csv", index=False)
    seg = pd.DataFrame(segments)
    seg.to_csv(HERE / "true_segment_by_seed.csv", index=False)
    per_seed = seg.groupby(["noise", "sigma", "seed", "smoother", "band", "interior"]).agg(
        segments=("label_present", "size"), label_present_fraction=("label_present", "mean"),
        lost_from_raw=("lost_from_raw", "sum"), gained_from_raw=("gained_from_raw", "sum"),
        window_correct_fraction=("window_correct_fraction", "mean"),
    ).reset_index()
    per_seed.to_csv(HERE / "segment_summary_by_seed.csv", index=False)
    per_seed.drop(columns="seed").groupby(
        ["noise", "sigma", "smoother", "band", "interior"]).mean().reset_index().to_csv(
            HERE / "segment_summary.csv", index=False)


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip-1)

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.parts.append(data.strip())


def retrieve():
    urls = {
        "ranacher": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4786863/",
        "rbf": "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.rbf_kernel.html",
        "nystroem": "https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html",
        "farha": "https://openaccess.thecvf.com/content_CVPR_2019/html/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.html",
        "skill": "https://arxiv.org/abs/2609.00065",
    }
    records = []
    (HERE / "sources").mkdir(exist_ok=True)
    for name, url in urls.items():
        row = dict(name=name, requested_url=url,
                   retrieved_at=pd.Timestamp.now(tz="Asia/Shanghai").isoformat())
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()
                row.update(resolved_url=response.url, status=response.status)
            path = HERE / f"sources/{name}.html"
            path.write_bytes(data)
            parser = TextParser()
            parser.feed(data.decode("utf-8", errors="replace"))
            (HERE / f"sources/{name}.txt").write_text("\n".join(parser.parts), encoding="utf-8")
            row.update(ok=True, sha256=sha(path), bytes=len(data))
            text = "\n".join(parser.parts)
            expected = {
                "ranacher": "measurement error causes a systematic",
                "rbf": "K(x, y) = exp", "nystroem": "normalization_",
                "farha": "over-segmentation", "skill": "Timothy Kassis",
            }[name]
            row["expected_content_present"] = expected in text
            if name == "ranacher" and not row["expected_content_present"]:
                fallback = EXP / "literature_review/sources/ranacher_primary.txt"
                row.update(reading_fallback=str(fallback.relative_to(ROOT)),
                           fallback_sha256=sha(fallback),
                           note="HTTP 200 browser challenge is not article retrieval")
        except Exception as exc:
            row.update(ok=False, error=str(exc))
        records.append(row)
    write_json("source_retrieval.json", records)


def main():
    start = time.monotonic()
    before = protected()
    write_json("input_manifest.json", {
        "created_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(), "files": before,
        "code_sha256": sha(Path(__file__)), "spec_sha256": sha(HERE / "ANALYSIS_SPEC.md"),
    })
    known_answers()
    with threadpool_limits(limits=2):
        kernel_diagnostics()
        group_decomposition()
        boundary_diagnostics()
    retrieve()
    after = protected()
    assert before == after, "Protected file content or membership changed"
    write_json("acceptance.json", {
        "passed": True, "protected_files": len(before), "content_and_membership_unchanged": True,
        "kernel_conditions": 31, "boundary_conditions": 93,
        "elapsed_seconds": time.monotonic()-start,
        "completed_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
    })
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
