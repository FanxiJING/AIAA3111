"""
Figures for the v2 pipeline (PNG, 200 dpi) -> results/basic_w50_no_dt_v2/figures/

    python make_figures_v2.py      # after pipeline_v2.py --stage cv and --stage final

    fig1_v2_model_comparison.png  test macro-F1 with 95% group-bootstrap CIs, v1 vs v2, selected marked
    fig2_v2_per_class.png         per-class test F1 of the selected model, raw vs final
    fig3_v2_confusion.png         point-level confusion matrix of the selected model (final)
    fig4_v2_example_trip.png      example test trip (same fixed rule as v1): true vs raw vs final
    fig5_v2_k_tradeoff.png        training CV: macro-F1 and transition recall vs k (selected model)
    captions.md
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from make_figures import (AXIS, BEFORE, FINAL, INK, INK2, MODE_COLOR, MUTED, SEQ, SURFACE, _clean, _save,
                          matplotlib, pick_trip)
from models import MODES

HERE = Path(__file__).resolve().parent
V2 = HERE / "results" / "basic_w50_no_dt_v2"
V1 = HERE / "results" / "basic_w50_no_dt" / "smoothing" / "smoothing_test.csv"
OUT = V2 / "figures"


def fig_comparison(captions, sel):
    t = pd.read_csv(V2 / "test_results.csv")
    t = t[t.model != "Majority (baseline)"].sort_values("cv_macro_f1").reset_index(drop=True)
    v1 = pd.read_csv(V1).set_index("model")
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    y = np.arange(len(t))
    for i, r in t.iterrows():
        if r.model in v1.index:
            o = v1.loc[r.model]
            ax.errorbar(o.point_macro_f1_smoothed, i - 0.16,
                        xerr=[[o.point_macro_f1_smoothed - o.macro_f1_ci_low_smoothed],
                              [o.macro_f1_ci_high_smoothed - o.point_macro_f1_smoothed]],
                        fmt="o", ms=8, color=BEFORE, elinewidth=2, label="v1 (final)" if i == 0 else None)
        ax.errorbar(r.point_macro_f1, i + 0.16,
                    xerr=[[r.point_macro_f1 - r.macro_f1_ci_low], [r.macro_f1_ci_high - r.point_macro_f1]],
                    fmt="o", ms=8, color=FINAL, elinewidth=2, label="v2 (final)" if i == 0 else None)
    s = int(t.index[t.model == sel][0])
    ax.axhspan(s - 0.45, s + 0.45, color="#cde2fb", alpha=0.6, zorder=0)
    ax.set_yticks(y, [f"{m}  (k={k}, {v})" + ("\nselected by CV" if m == sel else "")
                      for m, k, v in zip(t.model, t.k, t.voting)])
    for tick, m in zip(ax.get_yticklabels(), t.model):
        tick.set_fontweight("bold" if m == sel else "normal")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Test point macro-F1 (95% CI, 9 test groups resampled)")
    ax.set_title("v1 vs v2 on unseen test users", loc="left", pad=40)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    _clean(ax)
    _save(fig, "fig1_v2_model_comparison.png", captions,
          "Test point macro-F1 of every model after its own smoothing step, v1 (orange) and v2 (blue), "
          "with 95% bootstrap CIs over the 9 independent test groups. Models are ordered by v2 CV macro-F1 "
          "on blind-cut training windows (5 user-grouped folds); the shaded row is the selected model. v2 "
          "fits all models on the same 20,000-window subsample except the Nystroem SVM (all windows). The "
          "Nystroem SVM has no v1 counterpart.")


def fig_per_class(captions, sel, k):
    p = pd.read_csv(V2 / "test_per_class_f1.csv")
    p = p[p.model == sel].set_index("version")
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x, w = np.arange(len(MODES)), 0.36
    for off, v, color, label in ((-w / 2 - 0.01, "raw (k=1, hard)", BEFORE, "Raw window predictions"),
                                 (w / 2 + 0.01, "final", FINAL, f"Final (k = {k})")):
        vals = p.loc[v, MODES].to_numpy(dtype=float)
        ax.bar(x + off, vals, width=w, color=color, label=label, zorder=2)
        for xi, val in zip(x + off, vals):
            ax.text(xi, val + 0.015, f"{val:.2f}", ha="center", fontsize=11, color=INK2)
    ax.set_xticks(x, MODES)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Test point F1")
    ax.set_title(f"{sel}: per-class F1 (v2)", loc="left", pad=36)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    _clean(ax, "y")
    _save(fig, "fig2_v2_per_class.png", captions,
          f"Per-class point F1 of the selected v2 model ({sel}) on the 13 test users: raw window "
          f"predictions and the final output after its smoothing step (k = {k}).")


def fig_confusion(captions, sel, k):
    cm = pd.read_csv(V2 / "confusion_matrix_final_points.csv", index_col=0).loc[MODES, MODES].to_numpy()
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.2, 7))
    im = ax.imshow(cmn, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", SEQ), vmin=0, vmax=1)
    for i in range(len(MODES)):
        for j in range(len(MODES)):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n{cm[i, j]:,}", ha="center", va="center", fontsize=11,
                    color="#ffffff" if cmn[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(MODES)), MODES)
    ax.set_yticks(range(len(MODES)), MODES)
    ax.set_xlabel("Predicted mode")
    ax.set_ylabel("True mode")
    ax.set_title(f"v2 final model: {sel} (k = {k})", loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Share of the true class (row-normalised)")
    cb.outline.set_visible(False)
    _save(fig, "fig3_v2_confusion.png", captions,
          f"Point-level confusion matrix of the selected v2 model ({sel}, final output) on the 13 test "
          "users; each cell shows the share of the true class and the point count.")


def fig_trip(captions, sel, k):
    w = pd.read_csv(V2 / "window_predictions_test.csv", dtype={"user_id": str})
    trip, n_cand = pick_trip(w)
    d = w[w.trip_id == trip].copy()
    t0 = pd.to_datetime(d.start_time)
    d["x0"] = (t0 - t0.min()).dt.total_seconds() / 60
    d["x1"] = (pd.to_datetime(d.end_time) - t0.min()).dt.total_seconds() / 60
    rows = [("True mode\n(window majority)", "label"), ("Raw", f"{sel} | raw"), (f"Final, k = {k}", sel)]
    fig, ax = plt.subplots(figsize=(12, 3.8))
    for r, (_, col) in enumerate(rows):
        for _, x in d.iterrows():
            ax.barh(len(rows) - 1 - r, max(x.x1 - x.x0, 0.05), left=x.x0, height=0.7,
                    color=MODE_COLOR[x[col]], edgecolor=SURFACE, linewidth=0.6)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    ax.set_xlabel("Minutes from trip start")
    used = [m for m in MODES if m in set(d.label) | set(d[f"{sel} | raw"]) | set(d[sel])]
    ax.legend(handles=[Patch(color=MODE_COLOR[m], label=m) for m in used], frameon=False, ncol=len(used),
              loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    ax.set_title(f"Example test trip {trip} (user {d.user_id.iloc[0]}): {len(d)} windows, "
                 f"{d.n_points.sum():,} points", loc="left", pad=34)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    _save(fig, "fig4_v2_example_trip.png", captions,
          f"The same fixed rule as v1: among the {n_cand} test trips with at least one true mode change "
          f"between consecutive windows, the trip with the median number of windows ({trip}). Rows: true "
          f"window mode, raw prediction and final output of {sel}.")


def fig_k(captions, sel):
    r = json.loads(next(p for p in (V2 / "cv").glob("*.json")
                        if json.loads(p.read_text())["model"] == sel).read_text())
    v = next(x for x in r["variants"] if x["variant"].endswith(f"{r['voting']} voting")
             and ("on" if r["class_weights"] != "none" else "off") in x["variant"])
    seg = pd.DataFrame(r["segments_by_k"])
    ks = seg.k.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(ks, [v[f"k{k}"] for k in ks], marker="o", color=FINAL, linewidth=2)
    axes[0].axvline(r["k"], color=MUTED, linestyle="--")
    axes[0].set_title("CV point macro-F1", loc="left", fontsize=14)
    axes[1].plot(ks, seg.transition_recall, marker="o", color=BEFORE, linewidth=2, label="recall")
    axes[1].plot(ks, seg.transition_precision, marker="s", color=INK2, linewidth=2, label="precision")
    axes[1].axvline(r["k"], color=MUTED, linestyle="--")
    axes[1].set_title("CV mode-change detection (±1 window)", loc="left", fontsize=14)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xlabel("Smoothing window k (windows)")
        ax.set_xticks(ks)
        _clean(ax, "y")
    fig.suptitle(f"{sel}: choosing k on training users (chosen k = {r['k']})", x=0.01, ha="left",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig5_v2_k_tradeoff.png", captions,
          f"Training-user CV only (blind windows, 5 folds) for the selected model ({sel}, {r['voting']} voting). "
          "Left: mean fold point macro-F1 for each smoothing window k. Right: precision and recall of mode "
          "changes within ±1 window. The dashed line is the k chosen by the paired one-SE rule; larger k "
          "raises point F1 a little but misses more real mode changes.")


def main():
    import make_figures
    make_figures.OUT = OUT          # _save writes into make_figures.OUT
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.loads((V2 / "selection.json").read_text())["selected"]
    k = int(pd.read_csv(V2 / "test_results.csv").set_index("model").loc[sel, "k"])
    captions = ["# Figure captions (Part 2 v2)\n"]
    fig_comparison(captions, sel)
    fig_per_class(captions, sel, k)
    fig_confusion(captions, sel, k)
    fig_trip(captions, sel, k)
    fig_k(captions, sel)
    (OUT / "captions.md").write_text("\n".join(captions), encoding="utf-8")
    print(f"[done] figures in {OUT}")


if __name__ == "__main__":
    main()
