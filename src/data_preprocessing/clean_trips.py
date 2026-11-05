"""Clean labeled observations and split user timelines at gaps > 1200 seconds."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd


def clean_user(points):
    if points.user_id.nunique() > 1:
        raise ValueError("Expected one user per input file")
    stats = {"input_points": len(points)}
    valid = points.timestamp.notna() & points.latitude.between(-90, 90) & points.longitude.between(-180, 180) & points['mode'].notna() & points.label_status.eq('matched')
    stats['invalid_points_removed'] = int((~valid).sum())
    d = points.loc[valid].copy()
    # Source keys make ties reproducible without overwriting original identifiers.
    d = d.sort_values(['timestamp', 'source_file', 'point_id'], kind='stable')
    conflict = d.groupby('timestamp')['mode'].transform('nunique').gt(1)
    stats['conflicting_timestamp_points_removed'] = int(conflict.sum())
    d = d.loc[~conflict].copy()
    keys = ['timestamp', 'latitude', 'longitude', 'altitude_ft', 'mode']
    duplicates = d.duplicated(keys)
    stats['duplicate_observations_removed'] = int(duplicates.sum())
    counts = d.groupby('timestamp').size()
    d = d.loc[~duplicates].copy()
    groups = d.groupby('timestamp')
    lat = groups.latitude.transform('median')
    lon = groups.longitude.transform('median')
    d['_representative_distance'] = (d.latitude - lat)**2 + ((d.longitude - lon) * np.cos(np.deg2rad(lat)))**2
    d['same_timestamp_position_count'] = groups.latitude.transform('size').astype('int64')
    d = d.sort_values(['timestamp', '_representative_distance', 'source_file', 'point_id'], kind='stable')
    removed = d.duplicated('timestamp')
    stats['same_timestamp_observations_merged'] = int(removed.sum())
    d = d.loc[~removed].drop(columns='_representative_distance').reset_index(drop=True)
    d['same_timestamp_input_count'] = d.timestamp.map(counts).astype('int64')
    d['valid_coordinates'] = True
    # Missing altitude is not a reason to discard a valid location.
    d['altitude_ft'] = d.altitude_ft.mask(d.altitude_ft.eq(-777) | ~np.isfinite(d.altitude_ft))
    delta = d.timestamp.diff().dt.total_seconds()
    starts = delta.isna() | delta.gt(1200)
    trip_number = starts.cumsum().astype('int64')
    d['trip_id'] = pd.array([f'{u}_{n:06d}' for u, n in zip(d.user_id, trip_number)], dtype='string')
    d['trip_point_index'] = d.groupby('trip_id').cumcount().astype('int64')
    d['time_gap_s'] = delta
    d['trip_start'] = starts
    # Annotation only: these do NOT create additional trips.
    d['source_file_changed'] = d.source_file.ne(d.source_file.shift()) & ~starts
    d['source_point_gap'] = (d.source_file.eq(d.source_file.shift()) & d.point_id.diff().ne(1)) & ~starts
    stats['output_points'] = len(d)
    stats['trips'] = int(starts.sum())
    stats['removed_points'] = stats['input_points'] - len(d)
    return d, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/processed/labeled_points'))
    parser.add_argument('--output', type=Path, default=Path('data/processed/cleaned_points'))
    args = parser.parse_args()
    source, destination = args.input.resolve(), args.output.resolve()
    original = (Path(__file__).resolve().parents[2] / 'data/original').resolve()
    if source == destination or original == destination or original in destination.parents:
        parser.error('Output must differ from input and must not be inside data/original')
    files = sorted(source.glob('*.parquet'))
    if not files:
        parser.error('No input Parquet files found')
    destination.mkdir(parents=True, exist_ok=True)
    report = {'trip_gap_seconds': 1200, 'trip_rule': 'per user; new trip iff gap > 1200 seconds; no stay/move filtering', 'users': []}
    totals = Counter()
    for path in files:
        cleaned, stats = clean_user(pd.read_parquet(path))
        target = destination / path.name
        temp = target.with_suffix('.parquet.tmp')
        cleaned.to_parquet(temp, index=False, compression='zstd')
        temp.replace(target)
        totals.update(stats)
        report['users'].append({'user_id': path.stem, **stats})
        print(f'{path.stem}: {stats["output_points"]:,} points, {stats["trips"]:,} trips', flush=True)
    report['totals'] = dict(totals)
    (destination / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report['totals'], indent=2))


if __name__ == '__main__':
    main()
