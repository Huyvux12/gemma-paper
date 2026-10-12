"""Image-cluster inference, independent continuation cohorts, fixed contrasts."""
import csv,json,sys
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau,spearmanr
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'replacement'))
from score_adapter import clipscore_from_embeddings,unit
G='gemma-CLIPLiteral-768';G256='gemma-CLIPLiteral-256';FC='gemma-FactChecking-768'
B='clip-CLIPScore-512';L='clip_large-CLIPScore-768';S='specs-NativeSPECS-512';D='siglip-SigLIP2-768'

def load(dataset):
    mf=json.loads((ROOT/('manifest.json' if dataset=='sugar' else 'human_manifest.json')).read_text());r=mf['records'];scores={};metadata={}
    for kind in ['gemma','clip','clip_large','specs','siglip']:
        path=ROOT/'runs'/(kind if dataset=='sugar' else kind+'-human');m=json.loads((path/'metadata.json').read_text());metadata[kind]=m
        assert m['record_ids']==[x['id'] for x in r]
        with np.load(path/'embeddings.npz') as ar:e={k:ar[k] for k in ar.files}
        ix={n:i for i,n in enumerate(m['media_order'])};a=e['media'][[ix[x['filename']] for x in r]]
        for prompt in m['prompts']:
            for dim in ([128,256,512,768] if kind=='gemma' else [a.shape[1]]):
                name=f'{kind}-{prompt}-{dim}';scores[name]={}
                for label in (['positive','negative'] if dataset=='sugar' else ['candidate']):
                    if kind in ['specs','siglip']:
                        cosine=np.einsum('ij,ij->i',unit(a),unit(e[prompt+'_'+label]))
                        scores[name][label]=np.maximum((cosine+1)/2,0) if kind=='specs' else cosine
                    else:scores[name][label]=clipscore_from_embeddings(a,e[prompt+'_'+label],dim)
    return r,scores,metadata

def cluster_sugar(records,correct,mask,reps=10000):
    indices=np.flatnonzero(mask);rows=[records[i] for i in indices];names=sorted({r['filename'] for r in rows});ix={n:i for i,n in enumerate(names)}
    groups=np.array([ix[r['filename']] for r in rows]);counts=np.bincount(groups,minlength=len(names));keys=list(correct)
    hits=np.zeros((len(names),len(keys)),int)
    for j,k in enumerate(keys):np.add.at(hits[:,j],groups,correct[k][indices].astype(int))
    rng=np.random.default_rng(202610082);sample=rng.integers(0,len(names),size=(reps,len(names)))
    denom=counts[sample].sum(1);boots=hits[sample].sum(1)/denom[:,None];point=hits.sum(0)/counts.sum();stats={}
    for j,k in enumerate(keys):
        cat={c:float(correct[k][indices[[r['category']==c for r in rows]]].mean()) for c in sorted({r['category'] for r in rows})}
        stats[k]={'accuracy':float(point[j]),'correct':int(hits[:,j].sum()),'n_pairs':len(rows),'n_images':len(names),
          'ci95':np.quantile(boots[:,j],[.025,.975]).tolist(),'macro_accuracy':float(np.mean(list(cat.values()))),'category_accuracy':cat}
    contrasts={}
    for aa,bb in [(G,B),(G,L),(G,S),(G,D),(FC,L),(G256,L),(G256,G)]:
        a=keys.index(aa);b=keys.index(bb);delta=boots[:,a]-boots[:,b];diff=hits[:,a]-hits[:,b]
        signs=np.random.default_rng(202610083).integers(0,2,size=(19999,len(names)),dtype=np.int8)*2-1
        perm=signs.astype(np.int32)@diff
        p=(1+int((np.abs(perm)>=abs(int(diff.sum()))).sum()))/20000
        contrasts[aa+'_vs_'+bb]={'delta_pp':100*float(point[a]-point[b]),'ci95_pp':(100*np.quantile(delta,[.025,.975])).tolist(),
          'one_sided_95_lower_pp':100*float(np.quantile(delta,.05)),'p_cluster_permutation':p,
          'n_images':len(names),'n_pairs':len(rows),'a_only_correct':int((correct[aa][indices]&~correct[bb][indices]).sum()),
          'b_only_correct':int((~correct[aa][indices]&correct[bb][indices]).sum())}
    primary=[G+'_vs_'+B,G+'_vs_'+L];order=sorted(primary,key=lambda k:contrasts[k]['p_cluster_permutation']);prev=0
    for j,k in enumerate(order):prev=max(prev,min(1,(2-j)*contrasts[k]['p_cluster_permutation']));contrasts[k]['p_holm_primary']=prev
    comp=contrasts[G256+'_vs_'+G];comp['noninferiority_margin_pp']=2;comp['noninferiority_passed']=comp['one_sided_95_lower_pp']>-2
    return {'metrics':stats,'comparisons':contrasts,'n_pairs':len(rows),'n_images':len(names)}

def sugar():
    r,scores,meta=load('sugar');correct={k:v['positive']>v['negative'] for k,v in scores.items()}
    cohorts={name:cluster_sugar(r,correct,mask) for name,mask in [('full',np.ones(len(r),bool)),('unseen',np.array([not x['previously_seen'] for x in r]))]}
    result={'cohorts':cohorts,'runtime':meta,'scope':'Only unseen cohort confirms continuation; full includes development media; image-cluster bootstrap and image-level sign permutations'}
    result['ties']={k:int((v['positive']==v['negative']).sum()) for k,v in scores.items()}
    (ROOT/'results.json').write_text(json.dumps(result,indent=2))
    with (ROOT/'pair_scores.csv').open('w') as f:
        fields=['id','filename','category','previously_seen','method','positive_score','negative_score','correct'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i,row in enumerate(r):
            for k,v in scores.items():w.writerow({**{x:row[x] for x in fields[:4]},'method':k,'positive_score':v['positive'][i],'negative_score':v['negative'][i],'correct':int(correct[k][i])})
    print(json.dumps({c:{'metrics':{k:v['metrics'][k] for k in [B,L,G,G256,S]},'comparisons':v['comparisons']} for c,v in cohorts.items()},indent=2),flush=True)

def human_cohort(records,scores,mask,reps=2000):
    indices=np.flatnonzero(mask);rows=[records[i] for i in indices];x={k:v['candidate'][indices] for k,v in scores.items()};expanded={k:np.repeat(v,[len(r['ratings']) for r in rows]) for k,v in x.items()}
    ratings=np.array([y for r in rows for y in r['ratings']]);names=sorted({r['filename'] for r in rows});groups={n:[] for n in names};expanded_ix={n:[] for n in names};offset=0
    for i,r in enumerate(rows):groups[r['filename']].append(i);expanded_ix[r['filename']].extend(range(offset,offset+len(r['ratings'])));offset+=len(r['ratings'])
    ix=[np.array(expanded_ix[n]) for n in names]
    metrics={k:{'kendall_tau_c':float(kendalltau(v,ratings,variant='c').statistic),'spearman':float(spearmanr(v,ratings).statistic)} for k,v in expanded.items()}
    tested=[G,B,L,G256,FC,S,D];boots={k:[] for k in tested};rng=np.random.default_rng(202610084)
    for _ in range(reps):
        sample=np.concatenate([ix[i] for i in rng.integers(0,len(names),len(names))]);yy=ratings[sample]
        for k in tested:boots[k].append(kendalltau(expanded[k][sample],yy,variant='c').statistic)
    boots={k:np.array(v) for k,v in boots.items()}
    for k in tested:metrics[k]['ci95']=np.quantile(boots[k],[.025,.975]).tolist()
    comparisons={}
    for a,b in [(G,B),(G,L),(G,S),(G,D),(FC,L),(G256,G)]:
        d=boots[a]-boots[b];comparisons[a+'_vs_'+b]={'delta_tau_c':metrics[a]['kendall_tau_c']-metrics[b]['kendall_tau_c'],
          'ci95':np.quantile(d,[.025,.975]).tolist(),'one_sided_95_lower':float(np.quantile(d,.05))}
    counts=[];wins={k:[] for k in x}
    for n in names:
        ids=groups[n];pairs=[]
        for ii,i in enumerate(ids):
            for j in ids[ii+1:]:
                ri=np.mean(rows[i]['ratings']);rj=np.mean(rows[j]['ratings'])
                if ri==rj:continue
                pairs.append((i,j) if ri>rj else (j,i))
        if not pairs:continue
        counts.append(len(pairs))
        for k,v in x.items():wins[k].append(sum(v[i]>v[j] for i,j in pairs))
    counts=np.array(counts);wins={k:np.array(v) for k,v in wins.items()};rng=np.random.default_rng(202610085)
    sample=rng.integers(0,len(counts),size=(5000,len(counts)));denom=counts[sample].sum(1)
    wb={k:v[sample].sum(1)/denom for k,v in wins.items()};wp={k:float(v.sum()/counts.sum()) for k,v in wins.items()}
    within={'n_pairs':int(counts.sum()),'n_images':len(counts),'accuracy':wp,'comparisons':{a+'_vs_'+b:{'delta_pp':100*(wp[a]-wp[b]),
      'ci95_pp':(100*np.quantile(wb[a]-wb[b],[.025,.975])).tolist()} for a,b in [(G,B),(G,L),(G,S),(G,D),(FC,L),(G256,G)]}}
    return {'n_images':len(names),'n_captions':len(rows),'n_individual_ratings':len(ratings),'metrics':metrics,'comparisons':comparisons,'within_image_ranking':within}

def human():
    r,scores,meta=load('human');cohorts={name:human_cohort(r,scores,mask) for name,mask in [('full',np.ones(len(r),bool)),('unseen',np.array([not x['previously_seen'] for x in r]))]}
    (ROOT/'human_results.json').write_text(json.dumps({'cohorts':cohorts,'runtime':meta,'scope':'Secondary human evaluation with individual ratings; bootstrap clustered by image'},indent=2))
    with (ROOT/'human_scores.csv').open('w') as f:
        fields=['id','filename','caption','ratings','previously_seen']+list(scores);w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for i,row in enumerate(r):w.writerow({**{k:row[k] for k in fields[:5]},**{k:v['candidate'][i] for k,v in scores.items()}})
    print(json.dumps({c:{'n_images':v['n_images'],'metrics':{k:v['metrics'][k] for k in [B,L,G,G256,S]},'comparisons':v['comparisons']} for c,v in cohorts.items()},indent=2),flush=True)

if __name__=='__main__':human() if '--human' in sys.argv else sugar()
