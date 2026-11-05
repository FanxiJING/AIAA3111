"""Break unreliable motion connections and recompute features without dropping rows."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .motion_features import FEATURES, build_features


def repair_connections(data, speed_limit=100.0):
    if not np.isfinite(speed_limit) or speed_limit <= 0:
        raise ValueError('Speed limit must be finite and positive')
    d = data.copy()
    base_start = d.feature_block_start.astype(bool)
    cross_file = d.source_file.ne(d.source_file.shift()) & ~base_start
    high_speed = d.speed_mps.gt(speed_limit)
    d['pre_quality_speed_mps'] = d.speed_mps
    d['speed_outlier_candidate'] = high_speed
    d['source_boundary_break'] = cross_file
    d['input_feature_block_id'] = d.feature_block_id
    starts = base_start | cross_file | high_speed
    d['break_reason'] = pd.array([
        '|'.join(reason for condition, reason in [(a, 'input_block_start'), (b, 'source_file_change'), (c, 'speed_gt_limit')] if condition) or 'none'
        for a,b,c in zip(base_start, cross_file, high_speed)], dtype='string')
    number = starts.cumsum()
    d['feature_block_id'] = pd.array([f'{u}_q{n:06d}' for u,n in zip(d.user_id,number)],dtype='string')
    d['feature_block_start'] = starts
    d = build_features(d)
    d['acceleration_review_candidate'] = d.acceleration_mps2.abs().gt(20)
    d['jerk_review_candidate'] = d.jerk_mps3.abs().gt(20)
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('data/processed/five_class_features'))
    parser.add_argument('--output',type=Path,default=Path('data/processed/quality_features'))
    parser.add_argument('--speed-limit',type=float,default=100.0)
    args=parser.parse_args()
    source,output=args.input.resolve(),args.output.resolve()
    original=Path(__file__).resolve().parents[2]/'data/original'
    if source==output or output==original or original in output.parents:
        parser.error('Output must be separate from input and original')
    files=sorted(source.glob('*.parquet'))
    if not files: parser.error('No input files')
    output.mkdir(parents=True,exist_ok=True)
    totals=Counter()
    features={f:{'valid_before':0,'valid_after':0,'max_abs_before':None,'max_abs_after':None} for f in FEATURES}
    report={'speed_limit_mps':args.speed_limit,'acceleration_review_limit_mps2':20,'jerk_review_limit_mps3':20,'rules':'break at input block starts, source file changes, or pre-quality speed > limit; no label-dependent rule; no rows removed or imputed','users':[]}
    for path in files:
        before=pd.read_parquet(path)
        after=repair_connections(before,args.speed_limit)
        stats={'input_points':len(before),'output_points':len(after),'removed_points':0,
               'blocks_before':int(before.feature_block_start.sum()),'blocks_after':int(after.feature_block_start.sum()),
               'speed_candidates':int(after.speed_outlier_candidate.sum()),'source_boundaries':int(after.source_boundary_break.sum()),
               'speed_and_source_overlap':int((after.speed_outlier_candidate&after.source_boundary_break).sum()),
               'acceleration_review_after':int(after.acceleration_review_candidate.sum()),'jerk_review_after':int(after.jerk_review_candidate.sum())}
        totals.update(stats)
        report['users'].append({'user_id':path.stem,**stats})
        for f in FEATURES:
            for stage,d in [('before',before),('after',after)]:
                values=d[f].dropna()
                features[f]['valid_'+stage]+=len(values)
                if len(values):
                    value=float(values.abs().max())
                    old=features[f]['max_abs_'+stage]
                    features[f]['max_abs_'+stage]=value if old is None else max(value,old)
        temp=output/(path.name+'.tmp')
        after.to_parquet(temp,index=False,compression='zstd')
        temp.replace(output/path.name)
    for stats in features.values(): stats['valid_reduction']=stats['valid_before']-stats['valid_after']
    report['totals']=dict(totals)
    report['feature_statistics']=features
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'totals':report['totals'],'feature_statistics':features},indent=2))


if __name__=='__main__': main()
