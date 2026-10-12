"""Exploratory grounding and storage diagnostics from archived float32 embeddings."""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest, kendalltau

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'research/embeddinggemma2_clipscore_replacement'

def unit(a):
    a = np.asarray(a, dtype=np.float64)
    n = np.linalg.norm(a, axis=1, keepdims=True)
    if np.any(n == 0):
        raise ValueError('Zero vector')
    return a / n

def compact(a, dim, fmt):
    a = unit(a[:, :dim])
    if fmt == 'float16':
        a = a.astype(np.float16).astype(np.float64)
    elif fmt == 'int8':
        scale = np.max(np.abs(a), axis=1, keepdims=True) / 127
        a = np.rint(a / scale).astype(np.int8).astype(np.float64) * scale
    return unit(a)

def paired(a, b, reps=10000):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(20261008)
    means = np.array([d[rng.integers(0, len(d), (min(500, reps-i), len(d)))].mean(1) for i in range(0, reps, 500)]).ravel()
    wins = int(np.sum(np.asarray(a) & ~np.asarray(b)))
    losses = int(np.sum(~np.asarray(a) & np.asarray(b)))
    return {'delta_pp':float(d.mean()*100), 'ci95_pp':(100*np.quantile(means,[.025,.975])).tolist(), 'a_only':wins, 'b_only':losses, 'p_exact':float(binomtest(wins,wins+losses).pvalue) if wins+losses else 1.}

def load(kind, human=False):
    records = json.loads((SOURCE/('human_manifest.json' if human else 'manifest.json')).read_text())['records']
    folder = SOURCE/'runs'/(kind+('-human' if human else ''))
    meta = json.loads((folder/'metadata.json').read_text())
    assert [r['id'] for r in records] == meta['record_ids']
    e = np.load(folder/'embeddings.npz')
    index = {n:i for i,n in enumerate(meta['media_order'])}
    im = e['media'][[index[r['filename']] for r in records]]
    return records, meta, e, im

def main():
    n = 350
    rng = np.random.default_rng(20261008)
    perms=[]
    while len(perms)<2000:
        ix=rng.permutation(n)
        if np.all(ix != np.arange(n)): perms.append(ix)
    perms=np.asarray(perms)
    out={'scope':'Exploratory diagnostics on previously observed samples; no additional encoder inference.', 'grounding':{}, 'compression':{}, 'human_compression':{}}
    details=[]
    for kind,prompt in [('gemma','CLIPLiteral'),('gemma','FactChecking'),('clip','CLIPScore'),('clip_large','CLIPScore')]:
        records,meta,e,im=load(kind)
        im=unit(im); pos=unit(e[prompt+'_positive']); neg=unit(e[prompt+'_negative'])
        # rows=current caption pair, columns=possible image. Strict clipped CLIPScore.
        posmat=pos@im.T;negmat=neg@im.T
        mat=2.5*np.maximum(posmat,0)-2.5*np.maximum(negmat,0)
        c=mat>0; real=np.diag(c)
        wrong=c.sum(1)-real.astype(int)
        wrong_per_item=wrong/(n-1)
        vals=c[np.arange(n)[None,:],perms].mean(1)
        prototype=unit((im.sum(0)[None,:]-im)/(n-1))
        proto=(np.maximum(np.sum(prototype*pos,1),0)>np.maximum(np.sum(prototype*neg,1),0))
        name=kind+'-'+prompt
        rawcorrect=posmat>negmat;raw_wrong=(rawcorrect.sum()-np.diag(rawcorrect).sum())/(n*(n-1))
        ties=mat==0;wrong_ties=(ties.sum()-np.diag(ties).sum())/(n*(n-1))
        out['grounding'][name]={'matched_accuracy':float(real.mean()),'all_wrong_image_accuracy':float(wrong_per_item.mean()),'raw_cosine_wrong_image_accuracy':float(raw_wrong),'clipped_wrong_image_tie_fraction':float(wrong_ties),'matched_minus_wrong_pp':float(100*(real.mean()-wrong_per_item.mean())), 'derangement_mean':float(vals.mean()),'derangement_95percent_range':np.quantile(vals,[.025,.975]).tolist(),'permutation_p_upper':float((1+np.sum(vals>=real.mean()))/(len(vals)+1)),'leave_one_out_prototype_accuracy':float(proto.mean()),'n_pairs':n,'n_wrong_comparisons':n*(n-1),'n_permutations':len(vals),'category_matched_and_wrong':{cat:{'matched':float(real[[r['category']==cat for r in records]].mean()),'wrong':float(wrong_per_item[[r['category']==cat for r in records]].mean())} for cat in sorted({r['category'] for r in records})}}
        if kind!='gemma': continue
        base=real
        for dim in [128,256,512,768]:
            for fmt in ['float32','float16','int8']:
                a,b,z=[compact(x,dim,fmt) for x in [im,pos,neg]]
                sp=2.5*np.maximum(np.sum(a*b,1),0); sn=2.5*np.maximum(np.sum(a*z,1),0)
                ok=sp>sn; key=f'{prompt}-{dim}-{fmt}'
                out['compression'][key]={'accuracy':float(ok.mean()),'correct':int(ok.sum()),'flips_from_768f32':int(np.sum(ok!=base)), 'raw_bytes_per_vector':dim*({'float32':4,'float16':2,'int8':1}[fmt])+(4 if fmt=='int8' else 0),'vs_768f32':paired(ok,base),'category_accuracy':{cat:float(ok[[r['category']==cat for r in records]].mean()) for cat in sorted({r['category'] for r in records})}}
                for i,r in enumerate(records): details.append({'id':r['id'],'filename':r['filename'],'category':r['category'],'method':key,'positive_score':float(sp[i]),'negative_score':float(sn[i]),'correct':int(ok[i])})
    # Human metrics retain the original individual ratings and cluster by image.
    records,meta,e,im=load('gemma',True)
    yy=np.array([rating for r in records for rating in r['ratings']]); repeats=[len(r['ratings']) for r in records]
    for prompt in ['CLIPLiteral','FactChecking']:
        for dim in [128,256,512,768]:
            for fmt in ['float32','float16','int8']:
                a,b=[compact(x,dim,fmt) for x in [im,e[prompt+'_candidate']]]
                scores=2.5*np.maximum(np.sum(a*b,1),0)
                tau=kendalltau(np.repeat(scores,repeats),yy,variant='c').statistic
                wins=total=0
                for image in sorted({r['filename'] for r in records}):
                    ix=[i for i,r in enumerate(records) if r['filename']==image]
                    for jj,i in enumerate(ix):
                        for j in ix[jj+1:]:
                            d=np.mean(records[i]['ratings'])-np.mean(records[j]['ratings'])
                            if d==0: continue
                            total+=1;wins+=int((scores[i]-scores[j])*d>0)
                out['human_compression'][f'{prompt}-{dim}-{fmt}']={'tau_c':float(tau),'within_image_accuracy':wins/total,'within_image_pairs':total}
    (ROOT/'diagnostics_results.json').write_text(json.dumps(out,indent=2))
    with (ROOT/'compression_scores.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    print(json.dumps({'grounding':out['grounding'],'main_compression':{k:v for k,v in out['compression'].items() if k.startswith('CLIPLiteral') and (k.endswith('float32') or k=='CLIPLiteral-256-int8')}},indent=2))

if __name__=='__main__': main()
