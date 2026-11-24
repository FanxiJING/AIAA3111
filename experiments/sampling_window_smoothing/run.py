"""Cross sampling, window design and score smoothing against the Part 2 reference."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODEL = ROOT / "src/model_training"
sys.path[:0] = [str(MODEL), str(ROOT / "src/feature_enginneering")]

from build_windows import assign_windows, load_split, read_points, window_features
from models import MODES, cv_groups, load_windows, prepare
from motion_features import build_features
from pipeline_v2 import SPLITS_REPORT, class_scores, model_specs, user_folds
from smoothing import smooth_soft

POINTS = ROOT / "data/processed/quality_features"
SPLITS = ROOT / "data/splits"
RATES = (0, 15, 30)
DESIGNS = {0: ("p25", "p50", "p100", "t60", "t120"), 15: ("p50", "t120"), 30: ("p50", "t120")}
POLICIES = ("raw", "center11", "past3", "past5", "past11", "past120s", "past300s", "past600s")
TOLERANCE_S = 120
MODEL_NAME = "SVM (RBF, Nystroem, all data)"
GROUP_KEYS = ["user_id", "feature_block_id"]
SCORE_COLS = [f"score_{m}" for m in MODES]
COUNT_COLS = [f"cnt_{m}" for m in MODES]
KEEP_COLS = [f"kept_{m}" for m in MODES]


def seconds(times):
    return np.asarray(times, dtype="datetime64[s]").astype("int64")


def block_runs(frame):
    return frame.groupby(GROUP_KEYS, sort=False, observed=True).indices.values()


def thin_indices(times, interval):
    """Keep the first available point, then require interval seconds since the last keep."""
    if not interval:
        return np.arange(len(times))
    keep = [0]
    while True:
        nxt = int(times.searchsorted(times[keep[-1]] + interval))
        if nxt == len(times):
            return np.array(keep)
        keep.append(nxt)


def time_windows(points, width, pure):
    new_run = points.feature_block_id.ne(points.feature_block_id.shift())
    if pure:
        new_run |= points["mode"].ne(points["mode"].shift())
    run = new_run.cumsum()
    start = points.timestamp.groupby(run).transform("first")
    chunk = (points.timestamp - start).dt.total_seconds().astype("int64") // width
    return pd.factorize(pd.MultiIndex.from_arrays([run, chunk]), sort=False)[0]


def build_windows(sample, design, pure):
    width = int(design[1:])
    sample = sample.copy()
    sample["window_id"] = (
        assign_windows(sample, width, pure) if design.startswith("p")
        else time_windows(sample, width, pure)
    )
    return window_features(sample), sample.window_id.to_numpy()


def dense_evaluation(original, sample, windows, ids, design):
    """Assign every original point to the last retained window start in its block."""
    window_start = seconds(windows.start_time)
    original_times = seconds(original.timestamp)
    sampled_times = seconds(sample.timestamp)
    dense_ids = np.empty(len(original), dtype=int)
    release = np.empty(len(windows), dtype=np.int64)
    width = int(design[1:])
    for pos, kept in zip(block_runs(original), block_runs(sample), strict=True):
        first = np.r_[True, np.diff(ids[kept]) != 0]
        wids = ids[kept][first]
        starts = np.flatnonzero(first)
        dense_ids[pos] = wids[np.searchsorted(window_start[wids], original_times[pos], side="right") - 1]
        if design.startswith("p"):
            # Wait until the next window is large enough to avoid a retrospective tail merge.
            release[wids] = sampled_times[kept[np.minimum(starts + width + (width + 1) // 2 - 1,
                                                          len(kept) - 1)]]
        else:
            anchor = sampled_times[kept[0]]
            deadline = anchor + ((window_start[wids] - anchor) // width + 1) * width
            release[wids] = np.minimum(deadline, original_times[pos[-1]])
    codes = pd.Categorical(original["mode"], categories=MODES).codes
    assert (codes >= 0).all()
    values = np.bincount(dense_ids * len(MODES) + codes,
                         minlength=len(windows) * len(MODES)).reshape(-1, len(MODES))
    assert values.sum() == len(original)
    windows = windows.rename(columns=dict(zip(COUNT_COLS, KEEP_COLS)))
    windows[COUNT_COLS] = values
    windows["release_s"] = release
    assert np.all(release >= seconds(windows.end_time))
    return windows


def source_points(path):
    points = read_points(path)
    return points[points["mode"].isin(MODES)].sort_values(
        ["feature_block_id", "timestamp"], kind="stable").reset_index(drop=True)


def recalculate_motion(sample):
    keys = ["user_id", "trip_id", "feature_block_id"]
    dt = sample.groupby(keys, sort=False).timestamp.diff().dt.total_seconds()
    long_gap = dt.gt(1200)
    motion_run = long_gap.groupby([sample[k] for k in keys], sort=False).cumsum()
    temporary_blocks = pd.factorize(pd.MultiIndex.from_frame(sample[keys].assign(run=motion_run)))[0]
    derived = build_features(sample.assign(feature_block_id=temporary_blocks))
    return derived.assign(feature_block_id=sample.feature_block_id), int(long_gap.sum())


def true_events(points):
    change = points.feature_block_id.eq(points.feature_block_id.shift()) & points["mode"].ne(points["mode"].shift())
    return points.loc[change, GROUP_KEYS + ["timestamp"]]


def build_rate(rate, splits, out):
    pure = {d: [] for d in DESIGNS[rate]}
    blind = {d: [] for d in DESIGNS[rate]}
    events, sizes = [], []
    for uid, split in sorted(splits.items()):
        original = source_points(POINTS / f"{uid}.parquet")
        if rate == 0:
            events.append(true_events(original))
        times = seconds(original.timestamp)
        keep = np.concatenate([i[thin_indices(times[i], rate)] for i in block_runs(original)])
        sample = original.iloc[keep].reset_index(drop=True)
        resets = 0
        if rate:
            sample, resets = recalculate_motion(sample)
        sizes.append((uid, split, rate, len(original), len(sample),
                      sample.delta_t_s.median(), sample.delta_t_s.quantile(.90), resets))
        for design in DESIGNS[rate]:
            win, ids = build_windows(sample, design, pure=False)
            blind[design].append(dense_evaluation(original, sample, win, ids, design).assign(split=split))
            if split == "train":
                win, _ = build_windows(sample, design, pure=True)
                pure[design].append(win.assign(split=split))
    for design in DESIGNS[rate]:
        pd.concat(pure[design], ignore_index=True).to_parquet(out / f"{rate}_{design}_fit.parquet", index=False)
        pd.concat(blind[design], ignore_index=True).to_parquet(out / f"{rate}_{design}_eval.parquet", index=False)
    if rate == 0:
        pd.concat(events, ignore_index=True).to_parquet(out / "events.parquet", index=False)
    pd.DataFrame(sizes, columns=["user_id", "split", "interval_s", "original_points", "kept_points",
                                 "median_dt_s", "p90_dt_s", "motion_resets"]).to_csv(out / f"{rate}_sampling.csv", index=False)
    print(f"Built sampling interval {rate}, {len(sizes)} users", flush=True)


def prefix_soft(scores, ready, policy):
    if policy == "raw":
        return scores.argmax(axis=1)
    previous = np.arange(len(scores))
    if policy.endswith("s"):
        span = int(policy[4:-1])
        # The clock uses window availability, which remains monotone inside a block.
        left = np.minimum(np.searchsorted(ready, ready - span), previous)
    else:
        left = np.maximum(0, previous - int(policy[4:]) + 1)
    cumulative = np.vstack([np.zeros(scores.shape[1]), np.cumsum(scores, axis=0)])
    return (cumulative[previous + 1] - cumulative[left]).argmax(axis=1)


def match_events(truth, pred, tolerance=TOLERANCE_S):
    """Maximum cardinality matching for ordered times under a fixed absolute tolerance."""
    i = j = 0
    errors = []
    while i < len(truth) and j < len(pred):
        error = int(pred[j] - truth[i])
        if error < -tolerance:
            j += 1
        elif error > tolerance:
            i += 1
        else:
            errors.append(error)
            i += 1
            j += 1
    return errors


def boundary_metrics(tp, predicted, truth):
    return {
        "transition_precision": tp / predicted if predicted else 0.0,
        "transition_recall": tp / truth if truth else 0.0,
        "transition_f1": 2 * tp / (predicted + truth) if predicted + truth else 0.0,
    }


def point_metrics(cms):
    tp = np.diagonal(cms, axis1=-2, axis2=-1)
    denom = cms.sum(axis=-1) + cms.sum(axis=-2)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    return f1.mean(axis=-1), tp.sum(axis=-1) / cms.sum(axis=(-2, -1)), f1


def boundaries(events):
    return {key: seconds(g.timestamp) for key, g in events.groupby(GROUP_KEYS, sort=False)}


def evaluate(windows, scores, truth_by_block, policies=POLICIES):
    users = windows.user_id.to_numpy()
    blocks = windows.feature_block_id.to_numpy()
    starts = seconds(windows.start_time)
    ready = windows.release_s.to_numpy()
    counts = windows[COUNT_COLS].to_numpy()
    grouped = {uid: [] for uid in windows.user_id.unique()}
    for pos in block_runs(windows):
        grouped[users[pos[0]]].append(pos)
    center = None
    if "center11" in policies:
        ids = pd.factorize(pd.MultiIndex.from_arrays([users, blocks]))[0]
        labels = smooth_soft(scores, ids, starts, 11)
        center = pd.Categorical(labels, categories=MODES).codes
    data = {}
    for policy in policies:
        predictions = np.empty(len(windows), dtype=np.int8)
        stats = {}
        for uid, parts in grouped.items():
            tp = total_pred = total_true = live_tp = live_pred = 0
            live_errors = []
            for pos in parts:
                ix = pos[0]
                truth = truth_by_block.get((uid, blocks[ix]), ())
                pred = center[pos] if policy == "center11" else prefix_soft(scores[pos], ready[pos], policy)
                predictions[pos] = pred
                changes = pos[np.flatnonzero(pred[1:] != pred[:-1]) + 1]
                tp += len(match_events(truth, starts[changes]))
                total_pred += len(changes)
                total_true += len(truth)
                if policy != "center11":
                    publish = np.r_[ready[pos][1:] != ready[pos][:-1], True]
                    labels = pred[publish]
                    emissions = ready[pos][publish][np.r_[False, labels[1:] != labels[:-1]]]
                    errors = match_events(truth, emissions)
                    live_tp += len(errors)
                    live_pred += len(emissions)
                    live_errors.extend(errors)
            pos = np.concatenate(parts)
            cm = np.zeros((len(MODES), len(MODES)), dtype=np.int64)
            np.add.at(cm.T, predictions[pos], counts[pos])
            stats[uid] = (cm, np.array([tp, total_pred, total_true, live_tp, live_pred]), live_errors)
        data[policy] = stats, predictions
    return data


def summarize(stats, policy):
    cm = sum(item[0] for item in stats.values())
    tp, pred, truth, live_tp, live_pred = sum(item[1] for item in stats.values())
    f1, acc, classes = point_metrics(cm)
    result = {"points": int(cm.sum()), "macro_f1": f1, "accuracy": acc,
              "transition_matches": tp, "transition_predicted": pred, "transition_true": truth,
              **dict(zip((f"f1_{m}" for m in MODES), classes)), **boundary_metrics(tp, pred, truth)}
    if policy != "center11":
        errors = [error for item in stats.values() for error in item[2]]
        result.update({f"release_{metric}": value for metric, value in boundary_metrics(live_tp, live_pred, truth).items()})
        result["release_transition_predicted"] = live_pred
        result["release_transition_matches"] = live_tp
        result["release_median_signed_error_s"] = np.median(errors) if errors else np.nan
        result["release_median_absolute_error_s"] = np.median(np.abs(errors)) if errors else np.nan
    return result


def fit_model(fit, features, design):
    minimum = 10 if design.startswith("p") else 3
    fit = fit.loc[fit.n_points >= minimum]
    assert set(fit.label) == set(MODES) and len(features) == 27
    return model_specs(len(features))[MODEL_NAME].make(True).fit(fit[features], fit.label)


def score(est, table, features):
    result = class_scores(est, table[features])
    assert np.isfinite(result).all()
    return result


def candidates(rate):
    for design in DESIGNS[rate]:
        yield design, rate
        if rate:
            yield design, 0


def choose(cv):
    """Choose within 0.02 CV macro-F1 of the best causal row, then maximize release-time F1."""
    options = cv[cv.policy.ne("center11")].groupby(
        ["interval_s", "window", "fit_interval_s", "policy"], as_index=False)[
        ["macro_f1", "release_transition_f1"]].mean()
    selected = []
    for _, group in options.groupby("interval_s"):
        feasible = group[group.macro_f1 >= group.macro_f1.max() - 0.02]
        selected.append(feasible.sort_values(["release_transition_f1", "macro_f1", "window", "fit_interval_s", "policy"],
                                             ascending=[False, False, True, True, True]).iloc[0])
    return pd.DataFrame(selected)


def paired_intervals(stats, selected, test_users):
    groups = cv_groups(pd.Series(sorted(test_users)), SPLITS_REPORT).to_numpy()
    unique, group_idx = np.unique(groups, return_inverse=True)
    assert len(unique) == 9
    draws = np.random.default_rng(42).multinomial(len(unique), np.full(len(unique), 1 / len(unique)), 5000)

    def distribution(key):
        grouped = np.zeros((len(unique), len(MODES), len(MODES)), dtype=np.int64)
        events = np.zeros((len(unique), 5), dtype=np.int64)
        for uid, i in zip(sorted(test_users), group_idx, strict=True):
            cm, counts, _ = stats[key][uid]
            grouped[i] += cm
            events[i] += counts
        f1 = point_metrics(np.einsum("bg,gij->bij", draws, grouped))[0]
        tp, pred, truth, live_tp, live_pred = (draws @ events).T
        denom = (pred + truth).clip(min=1)
        return {"macro_f1": f1, "transition_f1": 2 * tp / denom,
                "release_transition_f1": 2 * live_tp / (live_pred + truth).clip(min=1)}

    rows = []
    for row in selected.itertuples(index=False):
        rate = row.interval_s
        key = (rate, row.window, row.fit_interval_s, row.policy)
        for reference in [(rate, "p50", 0, "raw"), (rate, "p50", 0, "center11")]:
            for metric, values in distribution(key).items():
                if reference[-1] == "center11" and metric == "release_transition_f1":
                    continue
                lo, hi = np.quantile(values - distribution(reference)[metric], [.025, .975])
                rows.append((rate, row.window, row.fit_interval_s, row.policy, reference[-1], metric, lo, hi))
    return pd.DataFrame(rows, columns=["interval_s", "window", "fit_interval_s", "policy", "reference_policy",
                                       "metric", "paired_low", "paired_high"])


def assert_reference(cache):
    expected = load_windows(MODEL / "windows_w50.csv")
    original = pd.read_parquet(cache / "0_p50_eval.parquet")
    actual = original[original.split.eq("test")].reset_index(drop=True)
    ref = expected[expected.split.eq("test")].reset_index(drop=True)
    assert len(ref) == len(actual) == 17380
    for col in GROUP_KEYS + COUNT_COLS:
        np.testing.assert_array_equal(actual[col], ref[col])
    np.testing.assert_array_equal(seconds(actual.start_time), seconds(ref.start_time))
    np.testing.assert_allclose(prepare(actual.drop(columns=KEEP_COLS + ["release_s"]))[0], prepare(ref)[0],
                               rtol=1e-10, atol=1e-7, equal_nan=True)
    assert actual[COUNT_COLS].to_numpy().sum() == 860972
    train = pd.read_parquet(cache / "0_p50_fit.parquet").query("n_points >= 10").reset_index(drop=True)
    expected = expected.query("split == 'train' and n_points >= 10").reset_index(drop=True)
    np.testing.assert_array_equal(train.label, expected.label)
    np.testing.assert_array_equal(train[GROUP_KEYS + COUNT_COLS], expected[GROUP_KEYS + COUNT_COLS])
    np.testing.assert_allclose(prepare(train)[0], prepare(expected)[0], rtol=1e-10, atol=1e-7, equal_nan=True)
    return train


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def validate_fold(design, fit_rate, fit, tables, fold, folds, features, truth_by_block):
    held_out = {u for u, n in folds.items() if n == fold}
    training = {u for u, n in folds.items() if n != fold}
    est = fit_model(fit[fit.user_id.isin(training)], features, design)
    rows = []
    for rate in tables if fit_rate == 0 else [fit_rate]:
        held = tables[rate][tables[rate].user_id.isin(held_out)]
        for policy, (stats, _) in evaluate(held, score(est, held, features), truth_by_block).items():
            rows.append({"interval_s": rate, "window": design, "fit_interval_s": fit_rate,
                         "fold": fold, "policy": policy, **summarize(stats, policy)})
    print(f"CV {design} fit interval {fit_rate} fold {fold + 1}/5", flush=True)
    return rows


def analyze(cache, out, splits, jobs, resume):
    reference_train = assert_reference(cache)
    folds = user_folds(reference_train)
    save_json(out / "training_user_folds.json", folds)
    features = list(prepare(reference_train)[0].columns)
    truth_by_block = boundaries(pd.read_parquet(cache / "events.parquet"))
    train_ids = {uid for uid, split in splits.items() if split == "train"}
    test_ids = set(splits) - train_ids
    assert len(train_ids) == 51 and len(test_ids) == 13
    assert set(folds) <= train_ids
    assert not set(cv_groups(pd.Series(sorted(train_ids)), SPLITS_REPORT)) & set(
        cv_groups(pd.Series(sorted(test_ids)), SPLITS_REPORT))
    progress = out / "cv_progress.csv"
    cv_rows = pd.read_csv(progress).to_dict("records") if resume and progress.is_file() else []
    completed = {}
    for row in cv_rows:
        key = row["window"], row["fit_interval_s"]
        completed.setdefault(key, set()).add((row["interval_s"], row["fold"], row["policy"]))
    assert sum(map(len, completed.values())) == len(cv_rows)
    for design in ("p50", "t120", "p25", "p100", "t60"):
        rates = [rate for rate in RATES if design in DESIGNS[rate]]
        tables = {r: pd.read_parquet(cache / f"{r}_{design}_eval.parquet").query("split == 'train'") for r in rates}
        for fit_rate in rates:
            key = design, fit_rate
            if key in completed:
                expected_rates = rates if fit_rate == 0 else [fit_rate]
                assert completed[key] == {(r, f, p) for r in expected_rates for f in range(5) for p in POLICIES}
                print(f"Reusing validation for {design}, fit interval {fit_rate}", flush=True)
                continue
            fit = pd.read_parquet(cache / f"{fit_rate}_{design}_fit.parquet")
            print(f"Validating {design}, fit interval {fit_rate} ({len(fit):,} candidate windows)", flush=True)
            fold_rows = joblib.Parallel(n_jobs=jobs, prefer="threads")(
                joblib.delayed(validate_fold)(design, fit_rate, fit, tables, fold, folds, features, truth_by_block)
                for fold in range(5)
            )
            cv_rows.extend(row for rows in fold_rows for row in rows)
            temporary = progress.with_suffix(".tmp")
            pd.DataFrame(cv_rows).to_csv(temporary, index=False)
            temporary.replace(progress)
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(out / "cv_metrics.csv", index=False)
    selected = choose(cv)
    selected.to_csv(out / "selected_on_training.csv", index=False)

    out.joinpath("models").mkdir(exist_ok=True)
    out.joinpath("predictions").mkdir(exist_ok=True)
    models, test_rows, user_rows, all_stats, fit_rows = {}, [], [], {}, []
    for rate in RATES:
        for design, fit_rate in candidates(rate):
            key = (fit_rate, design)
            if key not in models:
                fit = pd.read_parquet(cache / f"{fit_rate}_{design}_fit.parquet")
                minimum = 10 if design.startswith("p") else 3
                eligibility = fit.n_points.ge(minimum)
                fit_rows.append({"fit_interval_s": fit_rate, "window": design, "minimum_points": minimum,
                                 "eligible_windows": int(eligibility.sum()), "all_windows": len(fit),
                                 "eligible_users": fit.loc[eligibility, "user_id"].nunique()})
                model_path = out / "models" / f"{fit_rate}_{design}.joblib"
                if resume and model_path.is_file():
                    try:
                        est = joblib.load(model_path)
                        assert list(est.feature_names_in_) == features and set(est.classes_) == set(MODES)
                    except (EOFError, OSError, ValueError, AttributeError, AssertionError):
                        est = None
                else:
                    est = None
                if est is None:
                    print(f"Fitting final {design}, sampling interval {fit_rate}", flush=True)
                    est = fit_model(fit, features, design)
                    temporary = model_path.with_suffix(".tmp")
                    joblib.dump(est, temporary)
                    temporary.replace(model_path)
                else:
                    print(f"Reusing fitted model {design}, sampling interval {fit_rate}", flush=True)
                models[key] = est
            test = pd.read_parquet(cache / f"{rate}_{design}_eval.parquet").query("split == 'test'").copy()
            assert set(test.user_id) == test_ids and test[COUNT_COLS].to_numpy().sum() == 860972
            scores = score(models[key], test, features)
            predictions = test[GROUP_KEYS + ["start_time", "end_time", "release_s", "n_points"] + COUNT_COLS].copy()
            predictions[SCORE_COLS] = scores
            for policy, (stats, pred) in evaluate(test, scores, truth_by_block).items():
                lookup = (rate, design, fit_rate, policy)
                all_stats[lookup] = stats
                predictions[policy] = np.asarray(MODES)[pred]
                context = {"interval_s": rate, "window": design, "fit_interval_s": fit_rate, "policy": policy}
                test_rows.append({**context, **summarize(stats, policy),
                                  "retained_points": int(test.n_points.sum()), "windows": len(test),
                                  "median_window_duration_s": float(test.duration_s.median()),
                                  "user_mean_macro_f1": np.mean([point_metrics(s[0])[0] for s in stats.values()])})
                user_rows.extend({**context, "user_id": uid, **summarize({uid: item}, policy)}
                                 for uid, item in stats.items())
            predictions.to_parquet(out / "predictions" / f"{rate}_{design}_fit{fit_rate}.parquet", index=False)
            print(f"Test interval {rate}, {design}, fit interval {fit_rate}", flush=True)
    test = pd.DataFrame(test_rows)
    native = test.query("interval_s == 0 and window == 'p50'").set_index("policy")
    for policy, f1 in [("raw", .7789), ("center11", .8443)]:
        assert abs(native.loc[policy, "macro_f1"] - f1) < .001
    test.to_csv(out / "test_metrics.csv", index=False)
    pd.DataFrame(fit_rows).to_csv(out / "training_support.csv", index=False)
    pd.DataFrame(user_rows).to_csv(out / "per_user_metrics.csv", index=False)
    paired_intervals(all_stats, selected, test_ids).to_csv(out / "paired_bootstrap.csv", index=False)
    chosen = selected.iloc[:, :4].merge(test, validate="one_to_one")
    chosen.to_csv(out / "selected_test_metrics.csv", index=False)
    return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--resume", action="store_true", help="Reuse prior caches, completed CV blocks, and models")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    cache = out / "windows"
    cache.mkdir(exist_ok=args.resume)
    splits = load_split(SPLITS)
    with threadpool_limits(limits=args.threads):
        for rate in RATES:
            if args.resume and (cache / f"{rate}_sampling.csv").is_file():
                if rate == 0:
                    assert (cache / "events.parquet").is_file()
                for design in DESIGNS[rate]:
                    assert all((cache / f"{rate}_{design}_{kind}.parquet").is_file() for kind in ("fit", "eval"))
                print(f"Reusing sampling interval {rate}", flush=True)
            else:
                build_rate(rate, splits, cache)
        features = analyze(cache, out, splits, args.jobs, args.resume)
    inputs = [POINTS / f"{uid}.parquet" for uid in sorted(splits)]
    inputs += [SPLITS / name for name in ("train_users.json", "test_users.json", "report.json")]
    inputs += [MODEL / "windows_w50.csv"]
    code = [Path(__file__), *[MODEL / name for name in ("build_windows.py", "models.py", "pipeline_v2.py", "smoothing.py")],
            ROOT / "src/feature_enginneering/motion_features.py"]
    save_json(out / "manifest.json", {"features": features, "classes": MODES, "sampling_intervals_s": RATES,
                                     "designs": DESIGNS, "policies": POLICIES, "transition_tolerance_s": TOLERANCE_S,
                                     "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                                     "sklearn": sklearn.__version__, "threads": args.threads, "cv_jobs": args.jobs,
                                     "sha256": {str(p.relative_to(ROOT)): hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
                                                for p in inputs + code}})


if __name__ == "__main__":
    main()
