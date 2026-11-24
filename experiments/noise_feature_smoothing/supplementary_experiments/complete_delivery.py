"""Resume reporting after excluding non-rendered ticks from canvas checks."""
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import run_supplement as experiment


def visible_tick_labels(axis):
    low, high = sorted(axis.get_view_interval())
    texts, excluded = [], []
    for locations, labels in (
        (axis.get_majorticklocs(), axis.get_majorticklabels()),
        (axis.get_minorticklocs(), axis.get_minorticklabels()),
    ):
        for location, label in zip(locations, labels):
            if low <= location <= high:
                texts.append(label)
            else:
                excluded.append({"text": label.get_text(), "position": float(location)})
    return texts, excluded


def main():
    experiment.initialize()
    experiment.check_deadline()
    original_code = experiment.sha(Path(experiment.__file__))
    records = []

    def save_figure(fig, name):
        experiment.check_deadline()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        clipped, skipped = [], []
        texts = list(fig.texts)
        for ax in fig.axes:
            texts.extend([ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts])
            for axis in (ax.xaxis, ax.yaxis):
                labels, excluded = visible_tick_labels(axis)
                texts.extend(labels)
                skipped.extend(excluded)
            legend = ax.get_legend()
            if legend is not None:
                texts.append(legend)
        for item in texts:
            if not item.get_visible():
                continue
            box = item.get_window_extent(renderer)
            if box.width == 0 or box.height == 0:
                continue
            if (box.x0 < -1 or box.y0 < -1
                    or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1):
                clipped.append(str(item))
        assert not clipped, f"Rendered text outside figure canvas: {clipped}"
        fig.savefig(experiment.HERE / f"figures/{name}.png", dpi=170)
        fig.savefig(experiment.HERE / f"figures/{name}.pdf")
        records.append({
            "figure": name, "rendered_text_and_legend_items_checked": len(texts),
            "out_of_view_ticks_excluded_from_QA": skipped, "rendered_clipping": [],
        })
        experiment.plt.close(fig)

    experiment.save_figure = save_figure
    experiment.figures_and_report()
    experiment.final_check()
    assert experiment.sha(Path(experiment.__file__)) == original_code
    experiment.write_json(experiment.HERE / "diagnostics/render_recovery.json", {
        "passed": True,
        "repair": "Ignore tick labels outside the axis view interval in canvas QA.",
        "data_axis_limits_ticks_and_experimental_results_modified": False,
        "runner_file_modified": False,
        "runner_sha256": original_code,
        "recovery_script_sha256": experiment.sha(Path(__file__)),
        "figure_layout_checks": records,
        "manual_visual_review": "pending",
        "elapsed_seconds": time.monotonic() - experiment.START,
    })
    print(json.dumps({"report_recovery": "passed", "figures": len(records)}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
