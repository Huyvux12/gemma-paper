"""Recompute scores from vectors; comparisons frozen in protocol.json."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest,kendalltau,spearmanr
from score_adapter import clipscore_from_embeddings
ROOT=Path(__file__).resolve().parent

def paired(a,b,seed=42,reps=10000):
    a=np.asarray(a,dtype=int);b=np.asarray(b,dtype=int);d=a-b
    rng=np.random.default_rng(seed);samples=rng.integers(0,len(d),size=(reps,len(d)))
    ci=np.quantile(d[samples].mean(1),[.025,.975])
    wins=int(((a==1)&(b==0)).sum());losses=int(((a==0)&(b==1)).sum())
    p=binomtest(wins,wins+losses,.5).pvalue if wins+losses else 1.
    return {'delta_pp':100*float(d.mean()),'ci95_pp':(100*ci).tolist(),'p_raw':float(p),'a_only_correct':wins,'b_only_correct':losses,'n':len(d)}

def holm(results):
    keys=sorted(results,key=lambda k:results[k]['p_raw']);prev=0
    for i,k in enumerate(keys):
        prev=max(prev,min(1,results[k]['p_raw']*(len(keys)-i)))
        results[k]['p_holm']=prev

def load_scores(dataset='sugar'):
    manifest=json.loads((ROOT/('manifest.json' if dataset=='sugar' else 'human_manifest.json')).read_text())
    records=manifest['records'];scores={};metadata={}
    for kind in ['gemma','clip','clip_large']:
        folder=ROOT/'runs'/(kind if dataset=='sugar' else kind+'-human')
        m=json.loads((folder/'metadata.json').read_text());e=np.load(folder/'embeddings.npz')
        assert m['record_ids']==[r['id'] for r in records]
        index={n:i for i,n in enumerate(m['media_order'])};a=e['media'][[index[r['filename']] for r in records]]
        for prompt in m['prompts']:
            for dim in ([128,256,512,768] if kind=='gemma' else [a.shape[1]]):
                name=f'{kind}-{prompt}-{dim}'
                labels=['positive','negative'] if dataset=='sugar' else ['candidate']
                scores[name]={label:clipscore_from_embeddings(a,e[prompt+'_'+label],dim) for label in labels}
        metadata[kind]=m
    return records,scores,metadata

def sugar():
    records,scores,metadata=load_scores();correct={k:v['positive']>v['negative'] for k,v in scores.items()}
    metrics={k:{'accuracy':float(v.mean()),'correct':int(v.sum()),'n':len(v),'ties':int((scores[k]['positive']==scores[k]['negative']).sum()),
        'category_accuracy':{c:float(v[[r['category']==c for r in records]].mean()) for c in sorted({r['category'] for r in records})}} for k,v in correct.items()}
    g='gemma-FactChecking-768';b='clip-CLIPScore-512';l='clip_large-CLIPScore-768'
    comparisons={'FactChecking768_vs_B32':paired(correct[g],correct[b]),'FactChecking768_vs_L14':paired(correct[g],correct[l])};holm(comparisons)
    exploratory={f'{name}_vs_L14':paired(correct[name],correct[l]) for name in ['gemma-CLIPLiteral-768','gemma-None-768','gemma-SearchQuery-768','gemma-FactChecking-256','gemma-FactChecking-128']}
    secondary={k:dict(v) for k,v in exploratory.items()};holm(secondary)
    compression={str(dim):paired(correct[f'gemma-FactChecking-{dim}'],correct[g]) for dim in [128,256,512]}
    result={'metrics':metrics,'primary':comparisons,'exploratory_unadjusted':exploratory,'secondary_holm':secondary,'compression_descriptive':compression,
      'scope':'350 globally unique images, balanced random subset, excluded all prior images. Not full SugarCrepe or SOTA.', 'runtime':{k:{'parameters':m['parameters'],'peak_rss_mib':m['peak_rss_mib'],'timings':m['timings']} for k,m in metadata.items()}}
    (ROOT/'results.json').write_text(json.dumps(result,indent=2))
    rows=[]
    for i,r in enumerate(records):
        for name,v in scores.items():rows.append({'id':r['id'],'filename':r['filename'],'category':r['category'],'method':name,'positive_score':v['positive'][i],'negative_score':v['negative'][i],'correct':int(correct[name][i])})
    with (ROOT/'pair_scores.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'main':{k:metrics[k] for k in [b,l,g]},'primary':comparisons},indent=2))

def human():
    records,scores,meta=load_scores('human');y=[];groups=[]
    expanded={k:[] for k in scores}
    for i,r in enumerate(records):
        for rating in r['ratings']:
            y.append(rating);groups.append(r['filename'])
            for k in scores:expanded[k].append(scores[k]['candidate'][i])
    y=np.array(y);groups=np.array(groups);expanded={k:np.array(v) for k,v in expanded.items()}
    metrics={k:{'kendall_tau_c':float(kendalltau(v,y,variant='c').statistic),'spearman':float(spearmanr(v,y).statistic)} for k,v in expanded.items()}
    names=sorted(set(groups));indices=[np.flatnonzero(groups==g) for g in names];rng=np.random.default_rng(20261008)
    g='gemma-FactChecking-768';b='clip-CLIPScore-512';l='clip_large-CLIPScore-768';literal='gemma-CLIPLiteral-768'
    pairs=[(g,b),(g,l),(literal,l)];vals={a+'_vs_'+b:[] for a,b in pairs}
    for _ in range(2000):
        ix=np.concatenate([indices[i] for i in rng.integers(0,len(names),len(names))]);yy=y[ix]
        for a,b in pairs:vals[a+'_vs_'+b].append(kendalltau(expanded[a][ix],yy,variant='c').statistic-kendalltau(expanded[b][ix],yy,variant='c').statistic)
    comparisons={a+'_vs_'+b:{'delta_tau_c':metrics[a]['kendall_tau_c']-metrics[b]['kendall_tau_c'],'ci95':np.quantile(vals[a+'_vs_'+b],[.025,.975]).tolist()} for a,b in pairs}
    # A secondary diagnostic that removes between-image score-scale effects:
    # rank captions of the SAME image by their mean individual expert rating.
    grouped={n:[] for n in names}
    for i,r in enumerate(records):grouped[r['filename']].append(i)
    counts=[];wins={k:[] for k in scores};pair_ids=[]
    for image in names:
        candidates=grouped[image];valid=[]
        for ii,i in enumerate(candidates):
            for j in candidates[ii+1:]:
                ri=float(np.mean(records[i]['ratings']));rj=float(np.mean(records[j]['ratings']))
                if ri==rj:continue
                valid.append((i,j) if ri>rj else (j,i))
        if not valid:continue
        pair_ids.extend([(image,records[i]['id'],records[j]['id']) for i,j in valid]);counts.append(len(valid))
        for k in scores:wins[k].append(sum(scores[k]['candidate'][i]>scores[k]['candidate'][j] for i,j in valid))
    counts=np.array(counts);wins={k:np.array(v) for k,v in wins.items()}
    rng=np.random.default_rng(20261009);samples=rng.integers(0,len(counts),size=(5000,len(counts)))
    within_metrics={k:float(v.sum()/counts.sum()) for k,v in wins.items()}
    within_comparisons={}
    for a,b in pairs:
        delta=(wins[a]-wins[b])[samples].sum(1)/counts[samples].sum(1)
        within_comparisons[a+'_vs_'+b]={'delta_pp':100*(within_metrics[a]-within_metrics[b]),'ci95_pp':(100*np.quantile(delta,[.025,.975])).tolist()}
    result={'n_images':len(names),'n_caption_pairs':len(records),'n_individual_ratings':len(y),'metrics':metrics,'comparisons_exploratory':comparisons,
      'protocol':'random 100-image sample, all non-reference candidates, original individual expert ratings, Kendall tau-c; paired image-cluster bootstrap 2000; secondary benchmark, not a full benchmark replication'}
    result['within_image_ranking_exploratory']={'n_images_with_unequal_rating_pairs':len(counts),'n_pairs':int(counts.sum()),'accuracy':within_metrics,'comparisons':within_comparisons,'protocol':'all within-image caption pairs with unequal mean expert ratings, strict score greater wins; bootstrap by image 5000; diagnostic only'}
    (ROOT/'human_results.json').write_text(json.dumps(result,indent=2))
    with (ROOT/'human_scores.csv').open('w') as f:
        keys=['id','filename','caption','ratings']+list(scores);w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for i,r in enumerate(records):w.writerow({**{k:r[k] for k in keys[:4]},**{k:scores[k]['candidate'][i] for k in scores}})
    print(json.dumps({'n_images':len(names),'metrics':{k:metrics[k] for k in [b,l,g,literal]},'comparisons':comparisons},indent=2))

if __name__=='__main__':
    import sys
    human() if '--human' in sys.argv else sugar()
