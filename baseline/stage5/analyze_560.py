"""Score the exact frozen 560 records from recovered completed vector evidence."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parent

def unit(x):
    x=np.asarray(x,dtype=np.float64);return x/np.linalg.norm(x,axis=1,keepdims=True)

def storage(x,d,fmt):
    x=unit(x[:,:d])
    if fmt=='float16':x=x.astype(np.float16).astype(float)
    elif fmt=='int8':
        scale=(np.max(abs(x),axis=1,keepdims=True)/127).astype(np.float32)
        x=np.rint(x/scale).clip(-127,127).astype(np.int8).astype(float)*scale
    return unit(x)

def compare(a,b):
    a=np.asarray(a,bool);b=np.asarray(b,bool);d=a.astype(int)-b.astype(int)
    rng=np.random.default_rng(20261009);boot=np.concatenate([d[rng.integers(0,len(d),(500,len(d)))].mean(1) for _ in range(20)])
    wins=int(np.sum(a&~b));losses=int(np.sum(~a&b))
    return {'delta_pp':float(d.mean()*100),'ci95_pp':(100*np.quantile(boot,[.025,.975])).tolist(),'wins':wins,'losses':losses,'p_exact':float(binomtest(wins,wins+losses).pvalue) if wins+losses else 1.}

def main():
    records=json.loads((ROOT/'frozen_manifest.json').read_text())['records'];scores={};meta={};details=[]
    for kind in ['gemma','clip','clip_large','siglip','specs']:
        folder=ROOT/'runs'/kind;m=json.loads((folder/'metadata.json').read_text());z=np.load(folder/'embeddings.npz');index={n:i for i,n in enumerate(m['media_order'])};im=z['media'][[index[r['filename']] for r in records]]
        assert m['record_ids']==[r['id'] for r in records];meta[kind]={'model':m.get('model'),'revision':m['revision'],'origin':m['origin']}
        for prompt in m['prompts']:
            for d in ([128,256,512,768] if kind=='gemma' else [im.shape[1]]):
                for fmt in (['float32','float16','int8'] if kind=='gemma' else ['float32']):
                    a,b,c=[storage(x,d,fmt) for x in [im,z[prompt+'_positive'],z[prompt+'_negative']]]
                    sp=np.einsum('ij,ij->i',a,b);sn=np.einsum('ij,ij->i',a,c)
                    if kind=='specs':sp=np.maximum((sp+1)/2,0);sn=np.maximum((sn+1)/2,0)
                    elif kind!='siglip':sp=2.5*np.maximum(sp,0);sn=2.5*np.maximum(sn,0)
                    name=f'{kind}-{prompt}-{d}-{fmt}';scores[name]={'positive':sp,'negative':sn}
                    for i,r in enumerate(records):details.append({'id':r['id'],'filename':r['filename'],'category':r['category'],'method':name,'positive_score':float(sp[i]),'negative_score':float(sn[i]),'correct':int(sp[i]>sn[i])})
    correct={k:v['positive']>v['negative'] for k,v in scores.items()};base='gemma-CLIPLiteral-768-float32'
    metrics={k:{'accuracy':float(v.mean()),'correct':int(v.sum()),'n':len(v),'ties':int(np.sum(scores[k]['positive']==scores[k]['negative'])),'category_accuracy':{c:float(v[[r['category']==c for r in records]].mean()) for c in sorted({r['category'] for r in records})}} for k,v in correct.items()}
    comparisons={k:compare(correct[base],correct[k]) for k in ['clip-CLIPScore-512-float32','clip_large-CLIPScore-768-float32','siglip-SigLIP2-768-float32','specs-NativeSPECS-512-float32']}
    # Original secondary family had THREE contrasts; one SigLIP224 contrast remains missing.
    keys=sorted(['clip-CLIPScore-512-float32','clip_large-CLIPScore-768-float32'],key=lambda k:comparisons[k]['p_exact']);prev=0
    for i,k in enumerate(keys):prev=max(prev,min(1,comparisons[k]['p_exact']*(3-i)));comparisons[k]['p_holm_reserved_family3']=prev
    comp={k:compare(v,correct[base]) for k,v in correct.items() if k.startswith('gemma-CLIPLiteral')}
    controls={}
    for kind,prompt in [('gemma','CLIPLiteral'),('gemma','FactChecking'),('clip','CLIPScore'),('clip_large','CLIPScore'),('siglip','SigLIP2'),('specs','NativeSPECS')]:
        folder=ROOT/'runs'/kind;m=json.loads((folder/'metadata.json').read_text());z=np.load(folder/'embeddings.npz');index={n:i for i,n in enumerate(m['media_order'])};im=unit(z['media'][[index[r['filename']] for r in records]]);p=unit(z[prompt+'_positive']);n=unit(z[prompt+'_negative']);a=p@im.T;b=n@im.T
        raw=a>b;np.fill_diagonal(raw,False);clips=2.5*np.maximum(a,0)-2.5*np.maximum(b,0);ties=clips==0;np.fill_diagonal(ties,False)
        controls[kind+'-'+prompt]={'wrong_image_raw_cosine_positive_preference':float(raw.sum()/(560*559)),'clipped_wrong_image_tie_fraction':float(ties.sum()/(560*559)),'n_wrong_pairings':560*559,'interpretation':'Positive-caption preference on unrelated images, not correctness of those captions for shuffled images.'}
    result={'n_images':560,'n_pairs':560,'metrics':metrics,'comparisons':comparisons,'compression_descriptive':comp,'grounding':controls,'source_models':meta,'primary_comparison_status':'PENDING: SigLIP2-224 unavailable; 256 checkpoint is explicitly different and exploratory.','scope':'Computed in this turn from previously completed encoder inference vectors. No new encoder inference. Exact stage-4 frozen sample; all 560 marked unseen in larger source study.'}
    (ROOT/'results_560.json').write_text(json.dumps(result,indent=2))
    with (ROOT/'pair_scores_560.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    print(json.dumps({'metrics':{k:v for k,v in metrics.items() if k in [base,'clip-CLIPScore-512-float32','clip_large-CLIPScore-768-float32','siglip-SigLIP2-768-float32','specs-NativeSPECS-512-float32','gemma-CLIPLiteral-256-int8']},'comparisons':comparisons},indent=2))

if __name__=='__main__':main()
