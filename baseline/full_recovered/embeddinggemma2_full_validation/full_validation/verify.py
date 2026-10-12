"""Independent scalar metric checks, full media and author-label integrity."""
import collections,hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau
from analyze import load,ROOT

def scalar(a,b,dim,native=None):
    aa=[float(v) for v in a[:dim]];bb=[float(v) for v in b[:dim]]
    c=math.fsum(x*y for x,y in zip(aa,bb))/(math.sqrt(math.fsum(x*x for x in aa))*math.sqrt(math.fsum(x*x for x in bb)))
    if native=='specs':return max((c+1)/2,0)
    if native=='siglip':return c
    return 2.5*max(c,0)

def main():
    checks={};previous=ROOT.parent/'replacement'
    old={r['filename'] for file in [previous/'manifest.json',previous/'data/prior_image_manifest.json'] for r in json.loads(file.read_text())['records']}
    authors=json.loads((previous/'provenance/expert_author_annotations.json').read_text())
    for ds in ['sugar','human']:
        mf=json.loads((ROOT/('manifest.json' if ds=='sugar' else 'human_manifest.json')).read_text())
        rows,scores,meta=load(ds);res=json.loads((ROOT/('results.json' if ds=='sugar' else 'human_results.json')).read_text())
        folder=ROOT/('images' if ds=='sugar' else 'human_images')
        for n,d in mf['media_sha256'].items():assert hashlib.sha256((folder/n).read_bytes()).hexdigest()==d
        if ds=='sugar':
            expected=[]
            for cat in ['add_att','add_obj','replace_att','replace_obj','replace_rel','swap_att','swap_obj']:
                source=previous/'data'/(cat+'.json')
                assert hashlib.sha256(source.read_bytes()).hexdigest()==mf['source_sha256'][source.name]
                expected.extend(dict(row,id=f'{cat}:{key}',category=cat,previously_seen=row['filename'] in old) for key,row in json.loads(source.read_text()).items())
            assert rows==expected,'SugarCrepe labels/order/cohort differ from frozen original source'
            assert not {r['filename'] for r in rows if not r['previously_seen']} & old
        else:
            published={Path(line.split(None,1)[1]).name:line.split()[0] for line in (ROOT/'original_checksum_flickr8k_checksum.txt').read_text().splitlines() if line.strip()}
            for n in mf['media_sha256']:
                assert hashlib.md5((folder/n).read_bytes()).hexdigest()==published[n],n
            old_h={r['filename'] for r in json.loads((previous/'human_manifest.json').read_text())['records']}
            assert not {r['filename'] for r in rows if not r['previously_seen']} & old_h
            assert all(r['previously_seen']==(r['filename'] in old_h) for r in rows)
            for r in rows:
                a=[v['rating'] for v in authors[Path(r['filename']).stem]['human_judgement'] if v['caption']==r['caption']]
                assert sorted(a)==sorted(r['ratings']),(r['id'],a,r['ratings'])
        count=0;error=0;models={}
        for kind,m in meta.items():
            path=ROOT/'runs'/(kind if ds=='sugar' else kind+'-human')
            with np.load(path/'embeddings.npz') as ar:e={k:ar[k] for k in ar.files}
            assert m['dtype']=='float32';ix={n:i for i,n in enumerate(m['media_order'])}
            for prompt in m['prompts']:
                for dim in ([128,256,512,768] if kind=='gemma' else [e['media'].shape[1]]):
                    name=f'{kind}-{prompt}-{dim}';values={label:[] for label in (['positive','negative'] if ds=='sugar' else ['candidate'])}
                    for label in values:
                        for i,r in enumerate(rows):
                            v=scalar(e['media'][ix[r['filename']]],e[prompt+'_'+label][i],dim,kind)
                            error=max(error,abs(v-scores[name][label][i]));values[label].append(v);count+=1
                    values={k:np.array(v) for k,v in values.items()}
                    for cohort,mask in [('full',np.ones(len(rows),bool)),('unseen',np.array([not r['previously_seen'] for r in rows]))]:
                        metrics=res['cohorts'][cohort]['metrics'][name]
                        if ds=='sugar':
                            hit=values['positive']>values['negative'];assert int(hit[mask].sum())==metrics['correct']
                            assert abs(float(hit[mask].mean())-metrics['accuracy'])<1e-12
                        else:
                            x=[float(values['candidate'][i]) for i,r in enumerate(rows) if mask[i] for _ in r['ratings']]
                            y=[v for i,r in enumerate(rows) if mask[i] for v in r['ratings']]
                            assert abs(kendalltau(x,y,variant='c').statistic-metrics['kendall_tau_c'])<1e-12
                            groups={}
                            for i,r in enumerate(rows):
                                if mask[i]:groups.setdefault(r['filename'],[]).append(i)
                            n=hits=0
                            for indices in groups.values():
                                for ii,i in enumerate(indices):
                                    for j in indices[ii+1:]:
                                        ri=math.fsum(rows[i]['ratings'])/len(rows[i]['ratings']);rj=math.fsum(rows[j]['ratings'])/len(rows[j]['ratings'])
                                        if ri==rj:continue
                                        n+=1;hits+=int((values['candidate'][i]-values['candidate'][j])*(ri-rj)>0)
                            w=res['cohorts'][cohort]['within_image_ranking'];assert n==w['n_pairs'];assert abs(hits/n-w['accuracy'][name])<1e-12
                    models[name]=True
            print('verified',ds,kind,flush=True)
        assert error<1e-12
        checks[ds]={'media_hashes_verified':len(mf['media_sha256']),'scalar_scores_recomputed':count,'maximum_abs_error':error,
          'all_cohort_metrics_match':True,'methods_checked':len(models),'author_labels_match':True}
        if ds=='human':checks[ds]['original_clipscore_published_image_checksums_matched']=len(mf['media_sha256'])
    checks['passed']=True;(ROOT/'independent_verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__':main()
