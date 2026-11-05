"""Keep five target modes and mark gaps introduced by class filtering."""

import argparse
from collections import Counter
import json
from pathlib import Path

import pandas as pd

MODES = ['walk', 'bus', 'bike', 'train', 'car']


def filter_points(points):
    d = points.copy()
    keep = d['mode'].isin(MODES)
    # Define blocks BEFORE removing rows. Included-mode changes do not split.
    start = (~keep.shift(fill_value=False)) | d.trip_id.ne(d.trip_id.shift()).fillna(True)
    numbers = (keep & start).cumsum()
    d = d.loc[keep].copy()
    d['feature_block_id'] = pd.array([f'{u}_f{n:06d}' for u, n in zip(d.user_id, numbers.loc[keep])], dtype='string')
    d['feature_block_start'] = start.loc[keep].to_numpy(dtype=bool)
    return d.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/processed/cleaned_points'))
    parser.add_argument('--output', type=Path, default=Path('data/processed/five_class_points'))
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    original = Path(__file__).resolve().parents[2] / 'data/original'
    if source == output or output == original or original in output.parents:
        parser.error('Output must be separate from input and original')
    files = sorted(source.glob('*.parquet'))
    if not files:
        parser.error('No input Parquet files')
    output.mkdir(parents=True, exist_ok=True)
    before, after = Counter(), Counter()
    report = {'retained_modes': MODES, 'users': []}
    for path in files:
        data = pd.read_parquet(path)
        filtered = filter_points(data)
        before.update(data['mode'].tolist())
        after.update(filtered['mode'].tolist())
        temp = output / (path.name + '.tmp')
        filtered.to_parquet(temp, index=False, compression='zstd')
        temp.replace(output / path.name)
        report['users'].append({'user_id': path.stem, 'input_points': len(data), 'output_points': len(filtered), 'removed_points': len(data)-len(filtered), 'feature_blocks': filtered.feature_block_id.nunique()})
    report.update(input_points=sum(before.values()), output_points=sum(after.values()), removed_points=sum(before.values())-sum(after.values()), input_mode_counts=dict(before), output_mode_counts=dict(after), removed_mode_counts={k:v for k,v in before.items() if k not in MODES})
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='users'}, indent=2))


if __name__ == '__main__':
    main()
