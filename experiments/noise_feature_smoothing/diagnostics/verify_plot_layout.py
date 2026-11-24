"""Check rendered figure text extents; not a substitute for visual review."""
import json

import matplotlib.text
import pandas as pd

import analyze as a

records = []
save = a.save_figure


def checked_save(fig, name):
    # Add the exact caption used by the production renderer before measuring.
    caption = fig.text(.02, .015, a.CAPTION, va="bottom", fontsize=7, color="#444444")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    ignored_tick_labels = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            low, high = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                # Axis.draw omits generated ticks outside the view interval.
                if tick.get_loc() < low - 1e-10 or tick.get_loc() > high + 1e-10:
                    ignored_tick_labels.update((tick.label1, tick.label2))
    clipped = []
    text_count = 0
    for item in fig.findobj(matplotlib.text.Text):
        if (item in ignored_tick_labels or not item.get_visible()
                or not item.get_text().strip()):
            continue
        bbox = item.get_window_extent(renderer)
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        text_count += 1
        if bbox.x0 < -1 or bbox.y0 < -1 or bbox.x1 > width + 1 or bbox.y1 > height + 1:
            clipped.append({"text": item.get_text(), "bbox": list(bbox.extents)})
    title_overlaps = []
    titles = [ax.title for ax in fig.axes if ax.title.get_text()]
    for i, left in enumerate(titles):
        for right in titles[i + 1:]:
            if left.get_window_extent(renderer).overlaps(right.get_window_extent(renderer)):
                title_overlaps.append([left.get_text(), right.get_text()])
    records.append({"figure": name, "text_items_checked": text_count,
                    "canvas_width": width, "canvas_height": height,
                    "clipped_text": clipped, "subplot_title_overlaps": title_overlaps})
    caption.remove()
    save(fig, name)


def table(name):
    return pd.read_csv(a.HERE / "tables" / f"{name}.csv")


if __name__ == "__main__":
    a.save_figure = checked_save
    a.plots(table("feature_drift"), table("kernel_summary"), table("subgroup_metrics"),
            table("run_length_summary"), table("smoothing_window_support"),
            table("boundary_summary"), table("class_metrics_and_prediction_shares"),
            table("linked_group_sample_sizes"))
    passed = all(not r["clipped_text"] and not r["subplot_title_overlaps"] for r in records)
    output = {"passed": passed, "figures": records, "manual_visual_review": False}
    (a.HERE / "plot_layout_audit.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    assert passed, "See plot_layout_audit.json"
