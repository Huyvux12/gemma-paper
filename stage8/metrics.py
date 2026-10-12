import argparse,ast,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau,spearmanr,pearsonr
from sklearn.metrics import roc_auc_score,average_precision_score
from common import read,write

def number(v):
    v=float(v)
    return v if np.isfinite(v) else None

def tied_dcg(gains,scores,k=5):
    order=np.argsort(-np.asarray(scores),kind='stable');g=np.asarray(gains)[order];s=np.asarray(scores)[order]
    total=0.;i=0
    while i<len(order):
        j=i+1
        while j<len(order) and s[j]==s[i]:j+=1
        end=min(j,k)
        if i<k:total+=float(g[i:j].mean()*np.sum(1/np.log2(np.arange(i,end)+2)))
        i=j
    return total

def evaluate_human(records,scores):
    y=np.array([v for r in records for v in r['ratings']],float)
    counts=np.array([len(r['ratings']) for r in records]);means=np.array([np.mean(r['ratings']) for r in records])
    assert len(records)==len(scores) and np.isfinite(scores).all()
    expanded=np.repeat(scores,counts);binary=(means>=3).astype(int)
    groups=defaultdict(list)
    for i,r in enumerate(records):groups[r['filename']].append(i)
    pair_wins=0;pair_count=0;macros=[];ndcg=[];top=[];ties=0
    for ids in groups.values():
        local_wins=0;local_pairs=0
        for a,i in enumerate(ids):
            for j in ids[a+1:]:
                if means[i]==means[j]:continue
                better,worse=(i,j) if means[i]>means[j] else (j,i)
                local_wins+=int(scores[better]>scores[worse]);local_pairs+=1;ties+=int(scores[better]==scores[worse])
        if local_pairs:macros.append(local_wins/local_pairs)
        pair_wins+=local_wins;pair_count+=local_pairs
        if len(ids)>=2:
            gains=2**(means[ids]-1)-1
            ideal=tied_dcg(gains,gains,5)
            if ideal>0:ndcg.append(tied_dcg(gains,scores[ids],5)/ideal)
            best=np.asarray(ids)[scores[ids]==max(scores[ids])]
            top.append(float(np.mean(means[best]==max(means[ids]))))
    return {'n_images':len(groups),'n_captions':len(records),'n_individual_ratings':len(y),
        'kendall_tau_c_individual':number(kendalltau(expanded,y,variant='c').statistic),
        'kendall_tau_b_individual':number(kendalltau(expanded,y,variant='b').statistic),
        'spearman_individual':number(spearmanr(expanded,y).statistic),
        'pearson_individual':number(pearsonr(expanded,y).statistic),
        'spearman_mean_rating':number(spearmanr(scores,means).statistic),
        'within_image_micro_pair_accuracy':pair_wins/pair_count if pair_count else None,
        'within_image_macro_pair_accuracy':number(np.mean(macros)) if macros else None,
        'n_unequal_human_pairs':pair_count,'n_pair_groups':len(macros),'pair_score_ties':ties,
        'ndcg_at5_tie_averaged':number(np.mean(ndcg)) if ndcg else None,'n_ndcg_groups':len(ndcg),
        'top1_human_best_tie_averaged':number(np.mean(top)) if top else None,'n_top1_groups':len(top),
        'auroc_mean_rating_ge3':number(roc_auc_score(binary,scores)) if len(set(binary))==2 else None,
        'average_precision_mean_rating_ge3':number(average_precision_score(binary,scores)) if len(set(binary))==2 else None,
        'n_binary_positive':int(binary.sum()),'n_binary_negative':int((1-binary).sum())}

def baseline_human(baseline):
    rows=list(csv.DictReader((Path(baseline)/'stage7/human_scores.csv').open()))
    mf=read(Path(baseline)/'stage7/human_manifest.json')['records']
    assert [r['id'] for r in rows]==[r['id'] for r in mf]
    ignore={'id','filename','caption','ratings','previously_seen'}
    return mf,{k:np.array([float(r[k]) for r in rows]) for k in rows[0] if k not in ignore}

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--output',required=True);p.add_argument('--extra-scores');args=p.parse_args()
    records,scores=baseline_human(args.baseline)
    if args.extra_scores:
        extra=read(args.extra_scores);assert extra['record_ids']==[r['id'] for r in records]
        for k,v in extra['scores'].items():scores[k]=np.array(v,float)
    cohorts={}
    for name,mask in [('full',np.ones(len(records),bool)),('unseen900',np.array([not r['previously_seen'] for r in records]))]:
        rows=[r for r,m in zip(records,mask) if m]
        cohorts[name]={k:evaluate_human(rows,v[mask]) for k,v in scores.items()}
    result={'scope':'Exploratory point estimates; no extra significance claims or probability calibration. Mean-rating AUROC/AP binarization>=3.','cohorts':cohorts}
    write(args.output,result)
    target=Path(args.output).with_suffix('.csv')
    flattened=[{'cohort':c,'method':k,**v} for c,methods in cohorts.items() for k,v in methods.items()]
    with target.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flattened[0]));w.writeheader();w.writerows(flattened)
    # The extra evaluator must reproduce the old endpoint precisely.
    old=read(Path(args.baseline)/'stage7/human_results.json')
    for k in baseline_human(args.baseline)[1]:
        assert abs(cohorts['unseen900'][k]['kendall_tau_c_individual']-old['cohorts']['unseen']['metrics'][k]['kendall_tau_c'])<1e-12
    print(json.dumps({'human_methods':len(scores),'old_tau_c_reproduced':True,'output':str(args.output)}))

if __name__=='__main__':main()
