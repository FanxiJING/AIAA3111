"""
Part 2 figures for the report and poster (PNG, 200 dpi) -> results/figures/

    python make_figures.py

Needs: results/basic_w50_no_dt/ (train_basic.py), results/basic_w50/ (train_basic.py --keep-dt),
results/basic_w50_no_dt/smoothing/ (tune_smoothing.py, evaluate_smoothing.py).

    fig1_model_comparison.png   test point macro-F1 with 95% CIs, before and after smoothing
    fig2_dt_ablation.png        CV and test macro-F1 with and without dt_mean, dt_max
    fig3_smoothing_per_class.png per-class F1 of the selected model, before and after smoothing
    fig4_confusion_final.png    point-level confusion matrix of the final model
    fig5_example_trip.png       one test trip: true mode vs predictions, chosen by a fixed rule
    captions.md                 the caption of every figure
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from models import MODES

HERE = Path(__file__).resolve().parent
RUN = HERE / "results" / "basic_w50_no_dt"
SMO = RUN / "smoothing"
DT_RUN = HERE / "results" / "basic_w50"
OUT = HERE / "results" / "figures"

# reference palette (dataviz skill, light mode); slots 1-5 for the five modes, in fixed order
MODE_COLOR = dict(zip(MODES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
FINAL, BEFORE = "#2a78d6", "#eb6834"          # slot 1 = final/smoothed/no-dt, slot 2 = comparison
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 13, "axes.titlesize": 15,
    "axes.titleweight": "bold", "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 12, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK, "axes.edgecolor": AXIS})


def _clean(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _save(fig, name, captions, caption):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    captions.append(f"**{name}** — {caption}\n")


def fig_model_comparison(captions, selected, ks):
    t = pd.read_csv(SMO / "smoothing_test.csv")
    cv = pd.read_csv(RUN / "results.csv").set_index("model").cv_macro_f1
    t = t[t.model != "Majority (baseline)"].assign(cv=lambda d: d.model.map(cv))
    t = t.sort_values("cv", ascending=True).reset_index(drop=True)      # top = highest CV
    majority = pd.read_csv(RUN / "results.csv").set_index("model").loc["Majority (baseline)", "point_macro_f1"]
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(t))
    for off, v, color, label in ((-0.14, "baseline", BEFORE, "No smoothing"),
                                 (0.14, "smoothed", FINAL, "Smoothed (k chosen on training users)")):
        est = t[f"point_macro_f1_{v}"]
        lo, hi = t[f"macro_f1_ci_low_{v}"], t[f"macro_f1_ci_high_{v}"]
        ax.errorbar(est, y + off, xerr=[est - lo, hi - est], fmt="o", ms=9, color=color, ecolor=color,
                    elinewidth=2, capsize=0, label=label, zorder=3)
    sel = int(t.index[t.model == selected][0])
    ax.axhspan(sel - 0.45, sel + 0.45, color="#cde2fb", alpha=0.6, zorder=0)
    ax.axvline(majority, color=MUTED, linestyle="--", linewidth=1.5)
    ax.text(majority + 0.01, len(t) - 0.55, f"Majority\nbaseline {majority:.2f}", color=INK2, fontsize=11, va="top")
    labels = [f"{m}  (k={ks[m]})" + ("\nselected by CV" if m == selected else "") for m in t.model]
    ax.set_yticks(y, labels)
    for tick, m in zip(ax.get_yticklabels(), t.model):
        tick.set_fontweight("bold" if m == selected else "normal")
    s = t.loc[sel]
    ax.annotate(f"{s.point_macro_f1_smoothed:.3f}", (s.point_macro_f1_smoothed, sel + 0.14),
                xytext=(0, 12), textcoords="offset points", ha="center", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Test point macro-F1 (95% CI, test users resampled)")
    ax.set_title("Model comparison on unseen test users", loc="left", pad=40)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    _clean(ax)
    _save(fig, "fig1_model_comparison.png", captions,
          "Point-level macro-F1 on the 13 test users for every model, before (orange) and after (blue) "
          "temporal smoothing. Whiskers are 95% bootstrap CIs from resampling test users, with "
          "duplicate-linked users drawn together (9 groups). Models are ordered by CV macro-F1 on "
          "training users, the selection rule; the shaded row is the selected model. The CIs are wide "
          "and skewed because one linked group holds 78.9% of test points. k per model was chosen on "
          "training users only (paired one-SE rule).")


def fig_dt_ablation(captions):
    a = pd.read_csv(DT_RUN / "results.csv").set_index("model")
    b = pd.read_csv(RUN / "results.csv").set_index("model")
    order = b.drop("Majority (baseline)").sort_values("cv_macro_f1").index
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, col, title in ((axes[0], "cv_macro_f1", "CV on training users"),
                           (axes[1], "point_macro_f1", "Test users (point level)")):
        y = np.arange(len(order))
        ax.hlines(y, a.loc[order, col], b.loc[order, col], color=AXIS, linewidth=2, zorder=1)
        ax.scatter(a.loc[order, col], y, s=80, color=BEFORE, label="With dt_mean, dt_max (29 features)", zorder=3)
        ax.scatter(b.loc[order, col], y, s=80, color=FINAL, label="Without (27 features, default)", zorder=3)
        ax.set_yticks(y, order)
        ax.set_title(title, loc="left", fontsize=14)
        ax.set_xlabel("Macro-F1 (no smoothing)")
        vals = pd.concat([a.loc[order, col], b.loc[order, col]])
        pad = 0.15 * (vals.max() - vals.min())
        ax.set_xlim(vals.min() - pad, vals.max() + pad)
        _clean(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Dropping the sampling-interval features: CV macro-F1 does not fall for any model",
                 x=0.01, ha="left", fontsize=15, fontweight="bold", y=1.0)
    fig.legend(handles, labels, frameon=False, loc="upper left", bbox_to_anchor=(0.01, 0.95), ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _save(fig, "fig2_dt_ablation.png", captions,
          "Ablation of the sampling-interval features dt_mean and dt_max, without smoothing. Removing "
          "them did not lower CV macro-F1 for any model; on test the selected SVM changed by -0.0002. "
          "These features shift between users (median car-window dt_mean 4.12 s in train vs 2.0 s in "
          "test), so they are excluded by default. Test values here are single estimates; this run "
          "predates the bootstrap.")


def fig_per_class(captions, selected, k):
    p = pd.read_csv(SMO / "smoothing_per_class_f1.csv")
    p = p[p.model == selected].set_index("version")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = np.arange(len(MODES))
    w = 0.36
    for off, v, color, label in ((-w / 2 - 0.01, "baseline", BEFORE, "No smoothing"),
                                 (w / 2 + 0.01, "smoothed", FINAL, f"Smoothed (k = {k})")):
        vals = p.loc[v, MODES].to_numpy(dtype=float)
        ax.bar(x + off, vals, width=w, color=color, label=label, zorder=2)
        for xi, val in zip(x + off, vals):
            ax.text(xi, val + 0.015, f"{val:.2f}", ha="center", fontsize=11, color=INK2)
    ax.set_xticks(x, MODES)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Test point F1")
    ax.set_title(f"{selected}: per-class F1 before and after smoothing", loc="left", pad=36)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    _clean(ax, "y")
    _save(fig, "fig3_smoothing_per_class.png", captions,
          f"Per-class point F1 of the selected model ({selected}) on test users, without smoothing and "
          f"with the sliding majority vote over k = {k} windows inside each feature block. The largest "
          "gains are for car and bus, the two classes most often confused with each other.")


def fig_confusion(captions, selected, k):
    cm = pd.read_csv(SMO / "confusion_matrix_final_points.csv", index_col=0).loc[MODES, MODES].to_numpy()
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.2, 7))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ)
    im = ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(MODES)):
        for j in range(len(MODES)):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n{cm[i, j]:,}", ha="center", va="center", fontsize=11,
                    color="#ffffff" if cmn[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(MODES)), MODES)
    ax.set_yticks(range(len(MODES)), MODES)
    ax.set_xlabel("Predicted mode")
    ax.set_ylabel("True mode")
    ax.set_title(f"Final model: {selected}, smoothed (k = {k})", loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Share of the true class (row-normalised)")
    cb.outline.set_visible(False)
    _save(fig, "fig4_confusion_final.png", captions,
          f"Point-level confusion matrix of the final model ({selected} with smoothing, k = {k}) on the "
          "13 test users. Each cell shows the share of the true class's points and the point count; "
          "rows sum to 1. Every point inherits its window's prediction.")


def pick_trip(te: pd.DataFrame) -> str:
    """Rule: among test trips with at least one true mode change between consecutive windows of the
    same block, take the trip with the median number of windows (lower median; ties by trip_id)."""
    te = te.sort_values(["feature_block_id", "start_time"], kind="stable")
    same_block = te.feature_block_id.eq(te.feature_block_id.shift())
    change = same_block & te.label.ne(te.label.shift())
    trips = te.assign(change=change).groupby("trip_id").agg(n=("label", "size"), changes=("change", "sum"))
    cand = trips[trips.changes > 0].reset_index().sort_values(["n", "trip_id"]).reset_index(drop=True)
    return cand.loc[(len(cand) - 1) // 2, "trip_id"], len(cand)


def fig_example_trip(captions, selected, k):
    sm = pd.read_csv(SMO / "window_predictions_test_smoothed.csv", dtype={"user_id": str})
    base = pd.read_csv(RUN / "window_predictions_test.csv", dtype={"user_id": str})
    assert (sm.start_time == base.start_time).all()
    sm["baseline"] = base[selected]
    trip, n_cand = pick_trip(sm)
    d = sm[sm.trip_id == trip].copy()
    d["t0"], d["t1"] = pd.to_datetime(d.start_time), pd.to_datetime(d.end_time)
    origin = d.t0.min()
    d["x0"] = (d.t0 - origin).dt.total_seconds() / 60
    d["x1"] = (d.t1 - origin).dt.total_seconds() / 60
    rows = [("True mode\n(window majority)", "label"), ("No smoothing", "baseline"), (f"Smoothed, k = {k}", selected)]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    for r, (_, col) in enumerate(rows):
        yy = len(rows) - 1 - r
        for _, w in d.iterrows():
            ax.barh(yy, max(w.x1 - w.x0, 0.05), left=w.x0, height=0.7, color=MODE_COLOR[w[col]],
                    edgecolor=SURFACE, linewidth=0.6)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_xlabel("Minutes from trip start")
    used = [m for m in MODES if m in set(d.label) | set(d.baseline) | set(d[selected])]
    ax.legend(handles=[Patch(color=MODE_COLOR[m], label=m) for m in used], frameon=False, ncol=len(used),
              loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    ax.set_title(f"Example test trip {trip} (user {d.user_id.iloc[0]}): {len(d)} windows, "
                 f"{d.n_points.sum():,} points", loc="left", pad=34)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    _save(fig, "fig5_example_trip.png", captions,
          f"One test trip, chosen by a fixed rule before looking at its predictions: among the {n_cand} "
          "test trips with at least one true mode change between consecutive windows, the trip with the "
          f"median number of windows (trip {trip}). Rows show the true window mode, the {selected} "
          f"prediction without smoothing, and with smoothing (k = {k}). Each bar is one 50-point "
          "window; gaps are periods with no retained GPS points.")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    choice = json.loads((SMO / "smoothing_choice.json").read_text())
    selected = choice["selected_model"]
    ks = {m: c["k"] for m, c in choice["models"].items()}
    captions = ["# Figure captions (Part 2)\n"]
    fig_model_comparison(captions, selected, ks)
    fig_dt_ablation(captions)
    fig_per_class(captions, selected, ks[selected])
    fig_confusion(captions, selected, ks[selected])
    fig_example_trip(captions, selected, ks[selected])
    (OUT / "captions.md").write_text("\n".join(captions), encoding="utf-8")
    print(f"[done] figures in {OUT}")


if __name__ == "__main__":
    main()
