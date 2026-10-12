"""Paired exploratory comparison of the newly run SigLIP2 baseline."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau
from diagnostics import SOURCE,ROOT,load,unit,paired

def new_scores(human=False):
    records=json.loads((SOURCE/('human_manifest.json' if human else 'manifest.json')).read_text())['records']
    folder=ROOT/('siglip2-human' if human else 'siglip2')
    m=json.loads((folder/'metadata.json').read_text());e=np.load(folder/'embeddings.npz')
    assert m['record_ids']==[r['id'] for r in records]
    ix={n:i for i,n in enumerate(m['media_order'])};im=unit(e['media'][[ix[r['filename']] for r in records]])
    return records,{p:{lab:2.5*np.maximum(np.sum(im*unit(e[p+'_'+lab]),1),0) for lab in (['candidate'] if human else ['positive','negative'])} for p in ['raw','CLIPLiteral']},m

def old_scores(human=False):
    scores={}
    for kind,prompt in [('gemma','CLIPLiteral'),('gemma','FactChecking'),('clip','CLIPScore'),('clip_large','CLIPScore')]:
        records,m,e,im=load(kind,human)
        scores[kind+'-'+prompt]={lab:2.5*np.maximum(np.sum(unit(im)*unit(e[prompt+'_'+lab]),1),0) for lab in (['candidate'] if human else ['positive','negative'])}
    return scores

def main():
    records,new,meta=new_scores();scores={**old_scores(),**{'siglip2-'+p:s for p,s in new.items()}}
    ok={k:v['positive']>v['negative'] for k,v in scores.items()}
    metrics={k:{'accuracy':float(v.mean()),'correct':int(v.sum()),'n':len(v),'category_accuracy':{c:float(v[[r['category']==c for r in records]].mean()) for c in sorted({r['category'] for r in records})}} for k,v in ok.items()}
    comparisons={k+'_vs_siglip2-raw':paired(ok[k],ok['siglip2-raw']) for k in ['gemma-CLIPLiteral','gemma-FactChecking']}
    prev=0
    for i,k in enumerate(sorted(comparisons,key=lambda k:comparisons[k]['p_exact'])):
        prev=max(prev,min(1,comparisons[k]['p_exact']*(len(comparisons)-i)));comparisons[k]['p_holm']=prev
    matched_prompt_comparison=paired(ok['gemma-CLIPLiteral'],ok['siglip2-CLIPLiteral'])
    rows=[]
    for i,r in enumerate(records):
        for method in scores:rows.append({'id':r['id'],'filename':r['filename'],'category':r['category'],'method':method,'positive_score':float(scores[method]['positive'][i]),'negative_score':float(scores[method]['negative'][i]),'correct':int(ok[method][i])})
    with (ROOT/'baseline_pair_scores.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    archive=np.load(ROOT/'siglip2/embeddings.npz');ix={n:i for i,n in enumerate(meta['media_order'])}
    media=unit(archive['media'][[ix[r['filename']] for r in records]])
    grounding={}
    for p in ['raw','CLIPLiteral']:
        a=unit(archive[p+'_positive'])@media.T;b=unit(archive[p+'_negative'])@media.T
        mat=2.5*np.maximum(a,0)-2.5*np.maximum(b,0)
        mask=mat>0;np.fill_diagonal(mask,False)
        raw=a>b;np.fill_diagonal(raw,False);ties=mat==0;np.fill_diagonal(ties,False)
        grounding[p]={'all_wrong_image_accuracy':float(mask.sum()/(len(records)*(len(records)-1))),'raw_cosine_wrong_image_accuracy':float(raw.sum()/(len(records)*(len(records)-1))),'clipped_wrong_image_tie_fraction':float(ties.sum()/(len(records)*(len(records)-1))),'matched_accuracy':metrics['siglip2-'+p]['accuracy']}
    result={'scope':'New baseline inference on observed samples; exploratory comparisons, not full benchmark or SOTA.', 'sugar':{'metrics':metrics,'comparisons':comparisons,'matched_prompt_secondary':matched_prompt_comparison,'grounding':grounding},'metadata':{'sugar':meta}}
    (ROOT/'baseline_results.json').write_text(json.dumps(result,indent=2))
    if not (ROOT/'siglip2-human/metadata.json').exists():
        print(json.dumps(result['sugar'],indent=2));return
    records,new,hmeta=new_scores(True);scores={**old_scores(True),**{'siglip2-'+p:s for p,s in new.items()}}
    groups=sorted({r['filename'] for r in records});indices=[];ys=[];expanded={k:[] for k in scores};pairs={g:[] for g in groups}
    for i,r in enumerate(records):
        pairs[r['filename']].append(i)
        for rating in r['ratings']:
            ys.append(rating)
            for k in scores:expanded[k].append(scores[k]['candidate'][i])
    yy=np.asarray(ys);expanded={k:np.asarray(v) for k,v in expanded.items()}
    rating_groups=np.array([r['filename'] for r in records for _ in r['ratings']])
    indices=[np.flatnonzero(rating_groups==g) for g in groups]
    tau={k:float(kendalltau(v,yy,variant='c').statistic) for k,v in expanded.items()}
    win={k:[] for k in scores};counts=[]
    for g in groups:
        valid=[];ix=pairs[g]
        for jj,i in enumerate(ix):
            for j in ix[jj+1:]:
                a=np.mean(records[i]['ratings']);b=np.mean(records[j]['ratings'])
                if a!=b:valid.append((i,j) if a>b else (j,i))
        counts.append(len(valid))
        for k in scores:win[k].append(sum(scores[k]['candidate'][i]>scores[k]['candidate'][j] for i,j in valid))
    win={k:np.asarray(v) for k,v in win.items()};counts=np.asarray(counts)
    within={k:float(v.sum()/counts.sum()) for k,v in win.items()}
    compare={k+'_vs_'+base:{'delta_tau_c':tau[k]-tau[base],'delta_within_pp':100*(within[k]-within[base])} for k in ['gemma-CLIPLiteral','gemma-FactChecking'] for base in ['siglip2-raw','siglip2-CLIPLiteral']}
    rng=np.random.default_rng(20261008);samples=rng.integers(0,len(groups),(2000,len(groups)))
    boot={k:[] for k in compare}
    for draw in samples:
        ix=np.concatenate([indices[i] for i in draw]);bt={base:kendalltau(expanded[base][ix],yy[ix],variant='c').statistic for base in ['siglip2-raw','siglip2-CLIPLiteral']}
        for k in ['gemma-CLIPLiteral','gemma-FactChecking']:
            value=kendalltau(expanded[k][ix],yy[ix],variant='c').statistic
            for base in bt:boot[k+'_vs_'+base].append(value-bt[base])
    for k in ['gemma-CLIPLiteral','gemma-FactChecking']:
        for base in ['siglip2-raw','siglip2-CLIPLiteral']:
            comp=compare[k+'_vs_'+base];comp['ci95_tau_c']=np.quantile(boot[k+'_vs_'+base],[.025,.975]).tolist()
            delta=(win[k]-win[base])[samples].sum(1)/counts[samples].sum(1)
            comp['ci95_within_pp']=(100*np.quantile(delta,[.025,.975])).tolist()
    result['human']={'n_images':len(groups),'n_captions':len(records),'n_ratings':len(yy),'tau_c':tau,'within_image_accuracy':within,'n_within_pairs':int(counts.sum()),'comparisons':compare,'bootstrap':'2000 paired resamples of images'}
    result['metadata']['human']=hmeta
    (ROOT/'baseline_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'sugar':result['sugar'],'human':result['human']},indent=2))

if __name__=='__main__':main()
