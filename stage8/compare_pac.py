import argparse
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau
from common import read,write
from metrics import baseline_human

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--pac-scores',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    rows,scores=baseline_human(a.baseline);pac=read(a.pac_scores);assert pac['record_ids']==[r['id'] for r in rows]
    old=np.repeat(scores['gemma-CLIPLiteral-768-float32'],3);new=np.repeat(pac['scores']['PAC-S-ViTB32'],3);y=np.array([r['ratings'] for r in rows]).ravel();groups={}
    for i,r in enumerate(rows):
        if not r['previously_seen']:groups.setdefault(r['filename'],[]).extend(range(3*i,3*i+3))
    ids=[np.array(v) for v in groups.values()];rng=np.random.default_rng(81010);deltas=[]
    allids=np.concatenate(ids);point=kendalltau(new[allids],y[allids],variant='c').statistic-kendalltau(old[allids],y[allids],variant='c').statistic
    for _ in range(1000):
        ix=np.concatenate([ids[i] for i in rng.integers(0,len(ids),len(ids))]);deltas.append(kendalltau(new[ix],y[ix],variant='c').statistic-kendalltau(old[ix],y[ix],variant='c').statistic)
    write(a.output,{'direction':'PAC-S minus original Gemma','tau_c_delta':float(point),'paired_image_bootstrap_95_percentile_tau_c_delta_ci':np.quantile(deltas,[.025,.975]).tolist(),'bootstrap_replicates':1000,'n_images':len(ids),'seed':81010,'score_record_ids_aligned':True,'scope':'Exploratory historical benchmark; no fresh confirmatory significance claim.'});print(point,np.quantile(deltas,[.025,.975]))

if __name__=='__main__':main()
