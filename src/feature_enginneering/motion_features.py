"""Backward-looking, endpoint-aligned GPS motion features; see FEATURES.md."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6371008.8
FEATURES = ['delta_t_s', 'distance_m', 'speed_mps', 'acceleration_mps2',
            'jerk_mps3', 'bearing_rad', 'bearing_sin', 'bearing_cos',
            'heading_change_rad', 'heading_change_rate_rad_s']


def build_features(points):
    d = points.copy()
    keys = ['user_id', 'trip_id']
    if 'feature_block_id' in d.columns:
        keys.append('feature_block_id')
    g = d.groupby(keys, sort=False)
    dt = g.timestamp.diff().dt.total_seconds()
    if (dt.dropna() <= 0).any() or (dt.dropna() > 1200).any():
        raise ValueError('Expected strictly increasing timestamps with within-trip gaps <= 1200 s')
    lat = np.deg2rad(d.latitude.astype(float))
    lon = np.deg2rad(d.longitude.astype(float))
    prev_lat = np.deg2rad(g.latitude.shift())
    prev_lon = np.deg2rad(g.longitude.shift())
    dl = (lon - prev_lon + np.pi) % (2 * np.pi) - np.pi
    dp = lat - prev_lat
    h = (np.sin(dp / 2)**2 + np.cos(prev_lat) * np.cos(lat) * np.sin(dl / 2)**2).clip(0, 1)
    valid_coords = d.latitude.between(-90, 90) & d.longitude.between(-180, 180)
    prev_valid = valid_coords.groupby([d[k] for k in keys]).shift().eq(True)
    valid_interval = dt.gt(0) & valid_coords & prev_valid
    d['delta_t_s'] = dt
    d['distance_m'] = (2 * EARTH_RADIUS_M * np.arctan2(np.sqrt(h), np.sqrt(1 - h))).where(valid_interval)
    d['speed_mps'] = d.distance_m / dt
    # Interval-average speed is assigned to its ending timestamp. Derivatives
    # are backward differences of these endpoint-aligned discrete sequences.
    d['acceleration_mps2'] = d.groupby(keys, sort=False).speed_mps.diff() / dt
    d['jerk_mps3'] = d.groupby(keys, sort=False).acceleration_mps2.diff() / dt
    y = np.sin(dl) * np.cos(lat)
    x = np.cos(prev_lat) * np.sin(lat) - np.sin(prev_lat) * np.cos(lat) * np.cos(dl)
    direction_valid = valid_interval & d.distance_m.gt(0) & (x*x + y*y).gt(1e-24) & h.lt(1 - 1e-12)
    d['bearing_rad'] = (np.arctan2(y, x) % (2*np.pi)).where(direction_valid)
    d['bearing_sin'] = np.sin(d.bearing_rad)
    d['bearing_cos'] = np.cos(d.bearing_rad)
    turn = d.groupby(keys, sort=False).bearing_rad.diff()
    d['heading_change_rad'] = np.abs(np.arctan2(np.sin(turn), np.cos(turn)))
    d['heading_change_rate_rad_s'] = d.heading_change_rad / dt
    for name in FEATURES:
        d[name] = d[name].astype('float64').where(np.isfinite(d[name]))
        d[name + '_valid'] = d[name].notna()
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/processed/cleaned_points'))
    parser.add_argument('--output', type=Path, default=Path('data/processed/feature_points'))
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    original = Path(__file__).resolve().parents[2] / 'data/original'
    if source == output or output == original or original in output.parents:
        parser.error('Output must be separate from input and original data')
    files = sorted(source.glob('*.parquet'))
    if not files:
        parser.error('No input Parquet files')
    output.mkdir(parents=True, exist_ok=True)
    report = {'earth_radius_m': EARTH_RADIUS_M, 'alignment': 'backward endpoint-aligned',
              'features': FEATURES, 'total_points': 0, 'users': [], 'feature_statistics': {}}
    for path in files:
        data = pd.read_parquet(path)
        result = build_features(data)
        temp = output / (path.name + '.tmp')
        result.to_parquet(temp, index=False, compression='zstd')
        temp.replace(output / path.name)
        report['total_points'] += len(result)
        report['users'].append({'user_id': path.stem, 'points': len(result)})
        for name in FEATURES:
            values = result[name].dropna()
            stats = report['feature_statistics'].setdefault(name, {'valid': 0, 'missing': 0, 'min': None, 'max': None})
            stats['valid'] += len(values)
            stats['missing'] += int(result[name].isna().sum())
            if len(values):
                lo, hi = float(values.min()), float(values.max())
                stats['min'] = lo if stats['min'] is None else min(stats['min'], lo)
                stats['max'] = hi if stats['max'] is None else max(stats['max'], hi)
        print(f'{path.stem}: {len(result):,} points', flush=True)
    for stats in report['feature_statistics'].values():
        stats['missing_fraction'] = stats['missing'] / report['total_points'] if report['total_points'] else None
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report['feature_statistics'], indent=2))


if __name__ == '__main__':
    main()
