"""Create reproducible user-disjoint train/test manifests with duplicate guards."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODES = ['walk', 'bus', 'bike', 'train', 'car']
IDENTITY = ['timestamp', 'latitude', 'longitude', 'mode']


def choose_split(counts, groups, seed=42, fraction=.2, attempts=10000):
    rng=np.random.default_rng(seed)
    totals=counts.sum(axis=0).to_numpy(dtype=float)
    best=None
    for _ in range(attempts):
        order=rng.permutation(len(groups))
        # Prefix near target number of users; duplicate-linked users stay together.
        sizes=np.array([len(groups[i]) for i in order])
        k=int(np.argmin(abs(sizes.cumsum()-len(counts)*fraction)))+1
        test=sorted(u for i in order[:k] for u in groups[i])
        if len(test)==len(counts): continue
        test_counts=counts.loc[test].sum().to_numpy(dtype=float)
        if ((test_counts<=0)|(test_counts>=totals)).any(): continue
        ratios=test_counts/totals
        score=float(np.mean(abs(ratios-fraction)) + abs(test_counts.sum()/totals.sum()-fraction) + .25*abs(len(test)/len(counts)-fraction))
        if best is None or score<best[0]: best=(score,test)
    if best is None: raise ValueError('Unable to split while retaining every class in both sets')
    return sorted(set(counts.index)-set(best[1])),best[1],best[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('data/processed/quality_features'))
    parser.add_argument('--output',type=Path,default=Path('data/splits'))
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    source,output=args.input.resolve(),args.output.resolve()
    original=Path(__file__).resolve().parents[2]/'data/original'
    if output==source or output==original or original in output.parents: parser.error('Unsafe output path')
    files=sorted(source.glob('*.parquet'))
    if not files: parser.error('No input Parquet files')
    counts={}; empty=[]; fingerprints=[]
    for path in files:
        d=pd.read_parquet(path,columns=IDENTITY)
        if d.empty: empty.append(path.stem); continue
        counts[path.stem]=d['mode'].value_counts().reindex(MODES,fill_value=0).to_dict()
        # Only candidates are grouped by hash; collisions are resolved below
        # through exact identity-column equality.
        fingerprints.append(pd.DataFrame({'hash':pd.util.hash_pandas_object(d[IDENTITY],index=False).to_numpy(),'user':path.stem}))
    count_table=pd.DataFrame.from_dict(counts,orient='index')[MODES]
    parent={u:u for u in counts}
    def find(u):
        while parent[u]!=u:
            parent[u]=parent[parent[u]]; u=parent[u]
        return u
    def union(a,b): parent[find(b)]=find(a)
    hashes=pd.concat(fingerprints,ignore_index=True).drop_duplicates(['hash','user'])
    shared=hashes[hashes.duplicated('hash',keep=False)]
    candidate_users=sorted(shared.user.unique())
    # Load just cross-user hash candidates to verify actual duplicate records.
    exact=[]
    for u in candidate_users:
        d=pd.read_parquet(source/(u+'.parquet'),columns=IDENTITY)
        mask=pd.util.hash_pandas_object(d,index=False).isin(shared.loc[shared.user.eq(u),'hash'])
        q=d.loc[mask].copy(); q['user']=u; exact.append(q)
    links=Counter()
    if exact:
        same=pd.concat(exact,ignore_index=True).drop_duplicates(IDENTITY+['user'])
        for _,part in same.groupby(IDENTITY,sort=False):
            users=sorted(part.user.unique())
            for u in users[1:]: union(users[0],u); links[(users[0],u)]+=1
    grouped={}
    for u in counts: grouped.setdefault(find(u),[]).append(u)
    groups=sorted([sorted(g) for g in grouped.values()])
    train,test,score=choose_split(count_table,groups,args.seed)
    output.mkdir(parents=True,exist_ok=True)
    report={'seed':args.seed,'target_test_fraction':.2,'candidate_attempts':10000,'selection_score':score,
            'split_unit':'users; users sharing exact timestamp/latitude/longitude/mode records are grouped',
            'input_directory':str(source),'excluded_empty_users':empty,'duplicate_linked_groups':[g for g in groups if len(g)>1],
            'duplicate_links':[{'users':list(pair),'shared_unique_records':n} for pair,n in sorted(links.items())],
            'total_points':int(count_table.to_numpy().sum())}
    for name,users in [('train',train),('test',test)]:
        sums=count_table.loc[users].sum()
        report[name]={'users':users,'user_count':len(users),'points':int(sums.sum()),'mode_counts':{k:int(v) for k,v in sums.items()},'point_fraction':float(sums.sum()/report['total_points'])}
        (output/(name+'_users.json')).write_text(json.dumps(users,indent=2),encoding='utf-8')
    manifest=[{'user_id':u,'split':'train' if u in train else 'test','path':(Path('..')/'processed'/'quality_features'/(u+'.parquet')).as_posix(),'points':int(count_table.loc[u].sum())} for u in sorted(counts)]
    pd.DataFrame(manifest).to_csv(output/'manifest.csv',index=False)
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
