"""Paired image bootstrap for exploratory adapter diagnostics."""
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau
from common import load_gemma,read,write
from metrics import baseline_human

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    b=Path(a.baseline);ev=Path(a.evaluation);mf,v=load_gemma(b)
    original=np.einsum('ij,ij->i',v[0],v[1])>np.einsum('ij,ij->i',v[0],v[2])
    with np.load(ev/'adapter_sugar_scores.npz',allow_pickle=False) as z:
        assert z['record_ids'].tolist()==[r['id'] for r in mf['records']]
        adapted=z['ensemble_positive']>z['ensemble_negative']
    frozen={r['id'] for r in read(b/'stage5/frozen_manifest.json')['records']};result={};rng=np.random.default_rng(81009)
    masks={'historical560':np.array([r['id'] in frozen for r in mf['records']]),'historical_unseen3949':np.array([not r['previously_seen'] for r in mf['records']])}
    for cohort,mask in masks.items():
        names=sorted({r['filename'] for r,m in zip(mf['records'],mask) if m});mapping={n:i for i,n in enumerate(names)}
        ids=np.array([mapping[r['filename']] for r,m in zip(mf['records'],mask) if m]);counts=np.bincount(ids);delta=np.bincount(ids,weights=adapted[mask].astype(float)-original[mask])
        draws=rng.integers(0,len(names),(5000,len(names)));boot=delta[draws].sum(1)/counts[draws].sum(1)
        result[cohort]={'n_images':len(names),'n_pairs':int(mask.sum()),'original_accuracy':float(original[mask].mean()),'adapter_accuracy':float(adapted[mask].mean()),'delta':float((adapted[mask].astype(float)-original[mask]).mean()),'paired_image_bootstrap_95_percentile_delta_ci':np.quantile(boot,[.025,.975]).tolist(),'bootstrap_replicates':5000,'baseline_only_correct':int((original[mask]&~adapted[mask]).sum()),'adapter_only_correct':int((~original[mask]&adapted[mask]).sum())}
    rows,base=baseline_human(b);extra=read(ev/'adapter_human_scores.json');assert extra['record_ids']==[r['id'] for r in rows]
    groups={}
    for i,r in enumerate(rows):
        if not r['previously_seen']:groups.setdefault(r['filename'],[]).append(i)
    group_indices=[np.array(g) for g in groups.values()];labels=np.array([r['ratings'] for r in rows]).ravel();old=np.repeat(base['gemma-CLIPLiteral-768-float32'],3);new=np.repeat(extra['scores']['gemma-adapter-ensemble5'],3)
    expanded=[np.concatenate([np.arange(i*3,i*3+3) for i in g]) for g in group_indices]
    values=[]
    for _ in range(1000):
        ix=np.concatenate([expanded[i] for i in rng.integers(0,len(expanded),len(expanded))]);y=labels[ix]
        values.append(kendalltau(new[ix],y,variant='c').statistic-kendalltau(old[ix],y,variant='c').statistic)
    result['human_unseen900']={'paired_image_bootstrap_95_percentile_tau_c_delta_ci':np.quantile(values,[.025,.975]).tolist(),'bootstrap_replicates':1000,'n_images':len(groups)}
    write(a.output,{'scope':'Exploratory paired image-bootstrap intervals on already observed historical benchmarks; no multiplicity adjustment or new confirmatory significance claim. Five-seed ensemble is fixed by validation selection.','seed':81009,'results':result});print(result)

if __name__=='__main__':main()
