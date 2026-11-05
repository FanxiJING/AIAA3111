"""Read GeoLife points and retain only successfully matched interval labels."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def read_labels(path):
    labels = pd.read_csv(path, sep="\t", dtype=str)
    labels.columns = ["start", "end", "mode"]
    for name in ("start", "end"):
        labels[name] = pd.to_datetime(labels[name], format="%Y/%m/%d %H:%M:%S", errors="raise")
    if labels[["start", "end", "mode"]].isna().any().any() or (labels.start > labels.end).any():
        raise ValueError(f"Invalid label intervals: {path}")
    labels["mode"] = labels["mode"].str.strip()
    if labels["mode"].eq("").any():
        raise ValueError(f"Empty label: {path}")
    return labels


def match_labels(timestamps, labels):
    """Inclusive intervals; multiple distinct active modes are a conflict.

    Counting interval starts/ends handles overlaps, nested intervals and unsorted
    points without choosing an arbitrary latest label. Same-mode overlaps agree.
    """
    times = timestamps.to_numpy(dtype="datetime64[ns]")
    valid = timestamps.notna().to_numpy()
    modes = np.full(len(times), None, dtype=object)
    interval_count = np.zeros(len(times), dtype=np.int32)
    distinct_count = np.zeros(len(times), dtype=np.int16)
    for mode, group in labels.groupby("mode"):
        starts = np.sort(group.start.to_numpy(dtype="datetime64[ns]"))
        ends = np.sort(group.end.to_numpy(dtype="datetime64[ns]"))
        count = np.searchsorted(starts, times, side="right") - np.searchsorted(ends, times, side="left")
        count[~valid] = 0
        active = count > 0
        interval_count += count.astype(np.int32)
        distinct_count += active
        modes[active] = mode
    status = np.full(len(times), "unlabeled", dtype=object)
    status[distinct_count == 1] = "matched"
    status[distinct_count > 1] = "conflict"
    status[~valid] = "invalid_timestamp"
    modes[distinct_count != 1] = None
    return pd.DataFrame({"mode": pd.array(modes, dtype="string"), "label_status": pd.array(status, dtype="string"), "label_match_count": interval_count})


def read_points(path, user_id):
    raw = pd.read_csv(path, skiprows=6, header=None, dtype=str, keep_default_na=False)
    if raw.shape[1] != 7:
        raise ValueError(f"Expected 7 columns in {path}, got {raw.shape[1]}")
    raw.columns = ["latitude", "longitude", "reserved", "altitude_ft", "serial_day", "date", "time"]
    points = pd.DataFrame({
        "user_id": pd.Series(user_id, index=raw.index, dtype="string"),
        "source_file": pd.Series(f"{user_id}/Trajectory/{path.name}", index=raw.index, dtype="string"),
        "point_id": np.arange(len(raw), dtype=np.int64),
        "source_line": np.arange(7, len(raw) + 7, dtype=np.int64),
        "timestamp": pd.to_datetime(raw.date + " " + raw.time, format="%Y-%m-%d %H:%M:%S", errors="coerce"),
    })
    for column in ("latitude", "longitude", "altitude_ft"):
        points[column] = pd.to_numeric(raw[column], errors="coerce").astype(float)
    points["valid_coordinates"] = points.latitude.between(-90, 90) & points.longitude.between(-180, 180)
    return points


def run(input_dir, output_dir):
    users = sorted(p for p in input_dir.iterdir() if p.is_dir() and p.name.isdigit() and (p / "Trajectory").is_dir())
    if not users:
        raise ValueError(f"No user trajectories found in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"interval_convention": "start <= timestamp <= end; distinct-mode overlap is conflict", "users": [], "mode_counts": {}, "status_counts": {}}
    all_modes, all_status = Counter(), Counter()
    for user in users:
        labels = read_labels(user / "labels.txt")
        files = sorted((user / "Trajectory").glob("*.plt"))
        stats = {"user_id": user.name, "files": len(files), "label_intervals": len(labels), "points": 0, "invalid_coordinates": 0, "overlapping_label_points": 0}
        modes, statuses = Counter(), Counter()
        destination = output_dir / f"{user.name}.parquet"
        temporary = destination.with_suffix(".parquet.tmp")
        writer = None
        try:
            for path in files:
                points = read_points(path, user.name)
                points = pd.concat([points, match_labels(points.timestamp, labels)], axis=1)
                stats["points"] += len(points)
                stats["invalid_coordinates"] += int((~points.valid_coordinates).sum())
                stats["overlapping_label_points"] += int((points.label_match_count > 1).sum())
                modes.update(points["mode"].dropna().tolist())
                statuses.update(points.label_status.tolist())
                table = pa.Table.from_pandas(points.loc[points.label_status.eq("matched")], preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            raise ValueError(f"No trajectory files for {user.name}")
        temporary.replace(destination)
        stats["mode_counts"], stats["status_counts"] = dict(modes), dict(statuses)
        stats["output_points"] = statuses["matched"]
        stats["removed_points"] = stats["points"] - stats["output_points"]
        report["users"].append(stats)
        all_modes.update(modes)
        all_status.update(statuses)
        print(f"{user.name}: {stats['points']:,} points, {statuses['matched']:,} matched", flush=True)
    report["mode_counts"], report["status_counts"] = dict(all_modes), dict(all_status)
    report["total_points"] = sum(all_status.values())
    report["total_files"] = sum(u["files"] for u in report["users"])
    report["output_points"] = all_status["matched"]
    report["removed_points"] = report["total_points"] - report["output_points"]
    report["output_filter"] = "label_status == matched"
    report["counts_scope"] = "points, total_points, status_counts and quality counts describe input; output_points describes saved rows"
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "users"}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/original"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/labeled_points"))
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve() or args.input.resolve() in args.output.resolve().parents:
        parser.error("Output must be outside original input data")
    run(args.input, args.output)


if __name__ == "__main__":
    main()
