"""Build the comparison figure and paired analysis from saved experiment outputs."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run import (
    COUNT_COLS, GROUP_KEYS, HERE, MODES, POINTS, POLICIES, RATES, SPLITS, SPLITS_REPORT,
    choose, cv_groups, load_split, point_metrics, seconds, source_points,
)

KEYS = ["interval_s", "window", "fit_interval_s", "policy"]


def factor_contrasts(out, metrics, selected, short_windows):
    cached = {}
    rng = np.random.default_rng(42)
    draws = rng.multinomial(9, np.full(9, 1 / 9), 5000)
    user = pd.read_csv(out / "per_user_metrics.csv", dtype={"user_id": str})
    user["group"] = cv_groups(user.user_id, SPLITS_REPORT)

    def distributions(key):
        if key not in cached:
            rate, design, fit, policy = key
            pred = pd.read_parquet(out / "predictions" / f"{rate}_{design}_fit{fit}.parquet",
                                   columns=["user_id", *COUNT_COLS, policy])
            pred["group"] = cv_groups(pred.user_id, SPLITS_REPORT)
            cms = []
            for _, windows in pred.groupby("group", sort=True):
                codes = pd.Categorical(windows[policy], categories=MODES).codes
                cm = np.zeros((len(MODES), len(MODES)), dtype=np.int64)
                np.add.at(cm.T, codes, windows[COUNT_COLS].to_numpy())
                cms.append(cm)
            assert len(cms) == 9
            macro = point_metrics(np.einsum("rg,gij->rij", draws, cms))[0]
            mask = user[KEYS].eq(pd.Series(key, index=KEYS)).all(axis=1)
            row = user[mask].groupby("group", sort=True).sum(numeric_only=True)
            tp = draws @ row.transition_matches.to_numpy()
            denom = draws @ (row.transition_predicted + row.transition_true).to_numpy()
            ret = {"macro_f1": macro, "transition_f1": 2 * tp / denom.clip(min=1)}
            if policy != "center11":
                tp = draws @ row.release_transition_matches.to_numpy()
                denom = draws @ (row.release_transition_predicted + row.transition_true).to_numpy()
                ret["release_transition_f1"] = 2 * tp / denom.clip(min=1)
            cached[key] = ret
        return cached[key]

    rows = []

    def contrast(label, key, reference, subtract_pair=None):
        current, base = distributions(key), distributions(reference)
        for metric in current.keys() & base.keys():
            value = metrics.loc[key, metric] - metrics.loc[reference, metric]
            delta = current[metric] - base[metric]
            if subtract_pair:
                alt, alt_ref = subtract_pair
                delta -= distributions(alt)[metric] - distributions(alt_ref)[metric]
                value -= metrics.loc[alt, metric] - metrics.loc[alt_ref, metric]
            lo, hi = np.quantile(delta, [.025, .975])
            rows.append({"comparison": label, "interval_s": key[0], "window": key[1],
                         "metric": metric, "difference": value, "paired_low": lo, "paired_high": hi})

    for rate in RATES:
        p50 = (rate, "p50", 0, "raw")
        t120 = (rate, "t120", 0, "raw")
        contrast("120 s window vs 50 points, native-trained", t120, p50)
        if rate:
            contrast("Window-by-sampling interaction", t120, p50,
                     ((0, "t120", 0, "raw"), (0, "p50", 0, "raw")))
            for design in ("p50", "t120"):
                contrast("Thinned vs native input, native-trained", (rate, design, 0, "raw"), (0, design, 0, "raw"))
                contrast("Matched-rate retraining vs native-trained", (rate, design, rate, "raw"), (rate, design, 0, "raw"))
        contrast("Centered smoothing vs raw, native-trained 50 points", (rate, "p50", 0, "center11"), p50)
    selected_keys = dict((key[0], key) for key in selected[KEYS].itertuples(index=False, name=None))
    for key in selected_keys.values():
        contrast("Selected smoother vs raw, same fitted model", key, (*key[:3], "raw"))
    for key in short_windows[KEYS].itertuples(index=False, name=None):
        contrast("Short-window compromise vs selected 50-point policy", key, selected_keys[key[0]])
    return pd.DataFrame(rows).sort_values(["comparison", "interval_s", "window", "metric"])


def audit(out, test, selected):
    cv = pd.read_csv(out / "cv_metrics.csv")
    assert len(cv) == 520
    assert cv.groupby(KEYS).fold.nunique().eq(5).all()
    assert set(cv.policy) == set(POLICIES)
    pd.testing.assert_frame_equal(choose(cv)[KEYS].reset_index(drop=True), selected[KEYS].reset_index(drop=True))
    assert test.transition_true.nunique() == 1
    assert test.transition_matches.le(test.transition_predicted).all()
    assert test.transition_matches.le(test.transition_true).all()
    causal = test[test.policy.ne("center11")]
    assert causal.release_transition_matches.le(causal.release_transition_predicted).all()
    assert causal.release_transition_matches.le(causal.transition_true).all()
    assert test.loc[test.policy.eq("center11"), "release_transition_f1"].isna().all()
    users = pd.read_csv(out / "per_user_metrics.csv", dtype={"user_id": str})
    assert users.groupby(KEYS).user_id.nunique().eq(13).all()
    assert users.groupby(KEYS).points.sum().eq(860972).all()
    return cv


def replay_confusion(times, truth, ready, labels):
    """Treat points before the first score as abstentions and use the last score at a tie."""
    available = np.searchsorted(ready, times, side="right") - 1
    covered = available >= 0
    matrix = np.zeros((len(MODES), len(MODES)), dtype=np.int64)
    np.add.at(matrix, (truth[covered], labels[available[covered]]), 1)
    return matrix


def replay_points(out, selected):
    original = {}
    for user, split in load_split(SPLITS).items():
        if split == "test":
            points = source_points(POINTS / f"{user}.parquet")
            for key, block in points.groupby(GROUP_KEYS, sort=False):
                truth = pd.Categorical(block["mode"], categories=MODES).codes
                assert (truth >= 0).all()
                original[key] = seconds(block.timestamp), truth
    total = sum(len(times) for times, _ in original.values())
    assert total == 860972
    keys = [(rate, "p50", 0, "raw") for rate in RATES]
    keys.extend(selected[KEYS].itertuples(index=False, name=None))
    rows = []
    for rate, design, fit, policy in dict.fromkeys(keys):
        pred = pd.read_parquet(out / "predictions" / f"{rate}_{design}_fit{fit}.parquet",
                               columns=[*GROUP_KEYS, "start_time", "release_s", policy])
        confusion = np.zeros((len(MODES), len(MODES)), dtype=np.int64)
        seen = set()
        for key, block in pred.groupby(GROUP_KEYS, sort=False):
            ready = block.release_s.to_numpy()
            labels = pd.Categorical(block[policy], categories=MODES).codes
            assert np.all(ready[1:] >= ready[:-1]) and (labels >= 0).all()
            seen.add(key)
            confusion += replay_confusion(*original[key], ready, labels)
        assert seen == original.keys()
        covered = int(confusion.sum())
        assert 0 < covered <= total
        macro, accuracy, _ = point_metrics(confusion)
        rows.append(dict(zip(KEYS, (rate, design, fit, policy))) | {
            "points": total, "covered_points": covered, "coverage": covered / total,
            "covered_macro_f1": macro, "covered_accuracy": accuracy,
            "accuracy_including_abstentions": float(confusion.trace()) / total,
            "median_start_to_release_s": np.median(pred.release_s.to_numpy() - seconds(pred.start_time)),
        })
    return pd.DataFrame(rows)


def validation_frontier(cv, test):
    merged = cv[cv.policy.ne("center11")].groupby(KEYS, as_index=False)[
        ["macro_f1", "release_transition_f1"]].mean()
    columns = ["macro_f1", "release_transition_f1"]
    rows = []
    for _, candidates in merged.groupby("interval_s"):
        values = candidates[columns].to_numpy()
        better = (values[:, None] >= values).all(axis=2) & (values[:, None] > values).any(axis=2)
        rows.append(candidates.loc[~better.any(axis=0)])
    frontier = pd.concat(rows).rename(columns={c: f"cv_{c}" for c in columns})
    return frontier.merge(test, on=KEYS, validate="one_to_one").sort_values(
        ["interval_s", "cv_macro_f1"], ascending=[True, False])


def plots(out, raw, selected, frontier):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), layout="constrained")
    for idx, (design, color, desc) in enumerate(
        [("p50", "#17617d", "50 points"), ("t120", "#ac441f", "120 seconds")]
    ):
        for fit, style, label in [(False, "--", "native fit"), (True, "-", "matched fit")]:
            settings = [(r, design, r if fit else 0, "raw") for r in RATES]
            for ax, metric in zip(axes[:2], ("macro_f1", "release_transition_f1"), strict=True):
                ax.plot(np.arange(3), raw.loc[settings, metric], style, color=color,
                        marker=["o", "s"][idx], label=f"{desc}, {label}")
    for ax, title in zip(axes[:2], ("Point classification", "Switches at prediction release"), strict=True):
        ax.set_xticks(np.arange(3), ["Native", "15 s", "30 s"])
        ax.set_xlabel("Minimum GPS sampling interval")
        ax.set_ylabel("Macro-F1" if ax is axes[0] else "Switch F1")
        ax.set_title(title)
        ax.grid(axis="y", alpha=.22)
    axes[0].legend(fontsize=8, loc="best")
    ax = axes[2]
    for row, color in zip(selected.itertuples(index=False), ("#17617d", "#ac441f", "#5d6e38"), strict=True):
        points = frontier[frontier.interval_s.eq(row.interval_s)]
        ax.scatter(points.release_transition_f1, points.macro_f1, facecolors="none", edgecolors=color, alpha=.7)
        ax.scatter(row.release_transition_f1, row.macro_f1, color=color,
                   label="Selected at native rate" if row.interval_s == 0 else f"Selected at {row.interval_s} s", zorder=3)
    row = raw.loc[(0, "p50", 0, "raw")]
    ax.scatter(row.release_transition_f1, row.macro_f1, color="#333333", marker="x", label="Native 50 points, raw")
    ax.set_title("Training-validated trade-offs")
    ax.set_xlabel("Switch F1 at prediction release")
    ax.set_ylabel("Macro-F1")
    ax.grid(alpha=.22)
    ax.legend(fontsize=8, loc="best")
    fig.savefig(out / "joint_comparison.png", dpi=170)
    plt.close(fig)


def write_report(out, metrics, selected, contrasts, replay, switch_first, short_windows):
    baseline = metrics.loc[(0, "p50", 0, "raw")]
    center = metrics.loc[(0, "p50", 0, "center11")]
    paired = pd.read_csv(out / "paired_bootstrap.csv")
    lines = [
        "# GeoLife Part 3 joint experiment results",
        "",
        "## Settings chosen by training-only validation",
        "",
        "| Input | Window | Fitting samples | Smoother | Point macro-F1 | Release switch recall | Release switch F1 |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in selected.itertuples(index=False):
        rate = "Native" if row.interval_s == 0 else f"{row.interval_s} s"
        fit = "Native" if row.fit_interval_s == 0 else f"{row.fit_interval_s} s"
        lines.append(f"| {rate} | {row.window} | {fit} | {row.policy} | {row.macro_f1:.4f} | "
                     f"{row.release_transition_recall:.4f} | {row.release_transition_f1:.4f} |")
    lines += [
        "",
        "| Input | Macro-F1 change vs same-rate native-trained 50-point raw | Release switch F1 change |",
        "|---|---:|---:|",
    ]
    intervals = paired[paired.reference_policy.eq("raw")].set_index(["interval_s", "metric"])
    for row in selected.itertuples(index=False):
        ref = metrics.loc[(row.interval_s, "p50", 0, "raw")]
        values = []
        for metric in ("macro_f1", "release_transition_f1"):
            limits = intervals.loc[(row.interval_s, metric)]
            delta = getattr(row, metric) - ref[metric]
            values.append(f"{delta:+.4f} [{limits.paired_low:+.4f}, {limits.paired_high:+.4f}]")
        rate = "Native" if row.interval_s == 0 else f"{row.interval_s} s"
        lines.append(f"| {rate} | {' | '.join(values)} |")
    lines += [
        "",
        "## Configurations for switch detection",
        "",
        "This exploratory comparison selects the highest mean release switch F1 on the training "
        "folds without a classification constraint, using mean macro-F1 to break ties. "
        "The table gives test results for these training-fold choices.",
        "",
        "| Input | Window | Fitting interval (s) | Smoother | Point macro-F1 | Release switch F1 |",
        "|---|---|---:|---|---:|---:|",
    ]
    for row in switch_first.itertuples(index=False):
        rate = "Native" if row.interval_s == 0 else f"{row.interval_s} s"
        lines.append(f"| {rate} | {row.window} | {row.fit_interval_s} | {row.policy} | "
                     f"{row.macro_f1:.4f} | {row.release_transition_f1:.4f} |")
    lines += [
        "",
        "## Short-window compromise for thinned input",
        "",
        "This supplementary analysis restricts the candidate set to 120-second windows. It "
        "applies the same 0.02 classification margin and switch-F1 rule to training folds. "
        "The window restriction was motivated by the release delays observed in the main analysis.",
        "",
        "| Input | Fitting interval (s) | Smoother | Point macro-F1 | Release switch F1 |",
        "|---|---:|---|---:|---:|",
    ]
    for row in short_windows.itertuples(index=False):
        lines.append(f"| {row.interval_s} s | {row.fit_interval_s} | {row.policy} | "
                     f"{row.macro_f1:.4f} | {row.release_transition_f1:.4f} |")
    lines += [
        "",
        "## Main comparisons",
        "",
        f"- The native 50-point reference gives macro-F1 {baseline.macro_f1:.4f} raw and {center.macro_f1:.4f} "
        f"with centered 11-window smoothing. Switch recall under original-label timing changes from "
        f"{baseline.transition_recall:.4f} to {center.transition_recall:.4f}. The centered policy uses future scores.",
        "- Point-level classification uses the same original 860,972 test points in each condition.",
        "- A release switch is a change in the prediction as new window scores become available during a replay. "
        "The replay assumes known block termination. Switch recall and classification on the retrospective timeline "
        "use window starts. Switch matching has a 120-second tolerance and does not require class agreement.",
        "",
        "| Input | Test fixes kept | Native-trained 50-point raw F1 | Native-trained 120-second raw F1 | "
        "Difference with paired 95% interval |",
        "|---|---:|---:|---:|---:|",
    ]
    effect = contrasts.query("comparison == '120 s window vs 50 points, native-trained' and metric == 'macro_f1'")
    for row in effect.itertuples(index=False):
        rate = row.interval_s
        p50 = metrics.loc[(rate, "p50", 0, "raw")]
        timer = metrics.loc[(rate, "t120", 0, "raw")]
        name = "Native" if rate == 0 else f"{rate} s"
        lines.append(f"| {name} | {p50.retained_points / p50.points:.1%} | {p50.macro_f1:.4f} | "
                     f"{timer.macro_f1:.4f} | {row.difference:+.4f} [{row.paired_low:+.4f}, {row.paired_high:+.4f}] |")
    lines += [
        "",
        "Paired intervals resample nine linked user groups 5,000 times. The comparison table contains "
        "descriptive estimates for this split. Fixed-duration windows use a different minimum fitting-point rule.",
        "",
        "## Point classification at prediction release",
        "",
        "Each point uses the most recently released prediction in its block. Coverage is the fraction of "
        "points with an available prediction. The all-point accuracy counts abstentions as incorrect. "
        "Scores on covered points use a subset that can differ across settings.",
        "",
        "| Input | Window | Fitting interval (s) | Smoother | Median window start to release (s) | "
        "Coverage | Covered macro-F1 | All-point accuracy |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in replay.itertuples(index=False):
        rate = "Native" if row.interval_s == 0 else f"{row.interval_s} s"
        lines.append(f"| {rate} | {row.window} | {row.fit_interval_s} | {row.policy} | "
                     f"{row.median_start_to_release_s:.0f} | {row.coverage:.1%} | "
                     f"{row.covered_macro_f1:.4f} | {row.accuracy_including_abstentions:.4f} |")
    lines += [
        "",
        "See `factor_contrasts.csv` for paired sampling, window, and retraining comparisons, "
        "`paired_bootstrap.csv` for the selected policies against the references, and `test_metrics.csv` "
        "for the full exploratory grid. `validation_frontier.csv` lists settings that no alternative "
        "outscored on both validation objectives for their input rate. Open markers in the third plot "
        "show their test results. `training_support.csv` documents training support by setting.",
        "",
        "![Joint comparison](joint_comparison.png)",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    out = args.output.resolve()
    test = pd.read_csv(out / "test_metrics.csv")
    selected = pd.read_csv(out / "selected_test_metrics.csv")
    assert len(test) == 104 and len(selected) == 3 and test.points.eq(860972).all()
    cv = audit(out, test, selected)
    metrics = test.set_index(KEYS)
    assert metrics.index.is_unique
    frontier = validation_frontier(cv, test)
    frontier.to_csv(out / "validation_frontier.csv", index=False)
    switch_first = frontier.sort_values(
        ["interval_s", "cv_release_transition_f1", "cv_macro_f1", "window", "fit_interval_s", "policy"],
        ascending=[True, False, False, True, True, True],
    ).groupby("interval_s", sort=False).head(1)
    short_windows = choose(cv[cv.interval_s.gt(0) & cv.window.eq("t120")])[KEYS].merge(
        test, on=KEYS, validate="one_to_one",
    )
    contrasts = factor_contrasts(out, metrics, selected, short_windows)
    contrasts.to_csv(out / "factor_contrasts.csv", index=False)
    replay = replay_points(out, pd.concat([selected[KEYS], switch_first[KEYS], short_windows[KEYS]],
                                        ignore_index=True))
    replay.to_csv(out / "replay_point_metrics.csv", index=False)
    plots(out, metrics, selected, frontier)
    write_report(out, metrics, selected, contrasts, replay, switch_first, short_windows)


if __name__ == "__main__":
    main()
