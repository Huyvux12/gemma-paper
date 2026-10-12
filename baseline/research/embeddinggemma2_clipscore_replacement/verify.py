"""Independent scalar recomputation, data integrity and report consistency."""
import hashlib,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def scalar_score(a,b,dim):
    # Independent scalar arithmetic, not score_adapter's vectorized function.
    aa=[float(v) for v in a[:dim]];bb=[float(v) for v in b[:dim]]
    cos=math.fsum(x*y for x,y in zip(aa,bb))/(math.sqrt(math.fsum(x*x for x in aa))*math.sqrt(math.fsum(y*y for y in bb)))
    return 2.5*max(cos,0)

def main():
    checks={};old=json.loads((ROOT/'data/prior_image_manifest.json').read_text())
    for dataset,manifest_name,result_name in [('sugar','manifest.json','results.json'),('human','human_manifest.json','human_results.json')]:
        m=json.loads((ROOT/manifest_name).read_text());r=m['records'];result=json.loads((ROOT/result_name).read_text())
        folder=ROOT/('images' if dataset=='sugar' else 'human_images')
        for n,digest in m['media_sha256'].items():assert hashlib.sha256((folder/n).read_bytes()).hexdigest()==digest
        if dataset=='sugar':
            assert len(r)==350==len({x['filename'] for x in r})
            assert not {x['filename'] for x in r} & {x['filename'] for x in old['records']}
        maxerr=0;score_count=0;correctness={}
        for kind in ['gemma','clip','clip_large']:
            path=ROOT/'runs'/(kind if dataset=='sugar' else kind+'-human')
            meta=json.loads((path/'metadata.json').read_text())
            with np.load(path/'embeddings.npz') as archive:
                e={key:archive[key] for key in archive.files}
            assert meta['record_ids']==[x['id'] for x in r]
            ix={n:i for i,n in enumerate(meta['media_order'])}
            for prompt in meta['prompts']:
                for dim in ([128,256,512,768] if kind=='gemma' else [e['media'].shape[1]]):
                    name=f'{kind}-{prompt}-{dim}';win=[];vals=[]
                    for i,row in enumerate(r):
                        a=e['media'][ix[row['filename']]]
                        labels=['positive','negative'] if dataset=='sugar' else ['candidate']
                        values=[]
                        for label in labels:
                            b=e[prompt+'_'+label][i];v=scalar_score(a,b,dim);values.append(v);score_count+=1
                            # Cross-check float64 einsum score separately from the scalar path.
                            aa=a[:dim].astype('float64');bb=b[:dim].astype('float64')
                            vector=2.5*max(float(np.dot(aa,bb)/np.linalg.norm(aa)/np.linalg.norm(bb)),0)
                            maxerr=max(maxerr,abs(vector-v))
                        if dataset=='sugar':win.append(int(values[0]>values[1]))
                        else:vals.append(values[0])
                    if dataset=='sugar':
                        assert sum(win)==result['metrics'][name]['correct'];correctness[name]=win
                    else:
                        from scipy.stats import kendalltau
                        scores=[v for v,row in zip(vals,r) for _ in row['ratings']];ratings=[v for row in r for v in row['ratings']]
                        tau=float(kendalltau(scores,ratings,variant='c').statistic)
                        assert abs(tau-result['metrics'][name]['kendall_tau_c'])<1e-12
                        byimage={}
                        for i,row in enumerate(r):byimage.setdefault(row['filename'],[]).append(i)
                        n=hits=0
                        for indices in byimage.values():
                            for ii,i in enumerate(indices):
                                for j in indices[ii+1:]:
                                    ri=math.fsum(r[i]['ratings'])/len(r[i]['ratings']);rj=math.fsum(r[j]['ratings'])/len(r[j]['ratings'])
                                    if ri==rj:continue
                                    n+=1;hits+=int((vals[i]-vals[j])*(ri-rj)>0)
                        assert n==result['within_image_ranking_exploratory']['n_pairs']
                        assert abs(hits/n-result['within_image_ranking_exploratory']['accuracy'][name])<1e-12
        if dataset=='sugar':
            a=correctness['gemma-FactChecking-768']
            for comparison,base in [('FactChecking768_vs_B32','clip-CLIPScore-512'),('FactChecking768_vs_L14','clip_large-CLIPScore-768')]:
                b=correctness[base];wins=sum(x==1 and y==0 for x,y in zip(a,b));losses=sum(x==0 and y==1 for x,y in zip(a,b));n=wins+losses
                exact=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/(2**n)) if n else 1.
                assert abs(exact-result['primary'][comparison]['p_raw'])<1e-12
                assert abs(100*(sum(a)-sum(b))/len(a)-result['primary'][comparison]['delta_pp'])<1e-12
        assert maxerr<1e-12
        checks[dataset]={'media_hashes_verified':len(m['media_sha256']),'scalar_scores_recomputed':score_count,'maximum_abs_error':maxerr,'all_metrics_match':True}
    checks['official_clip']=json.loads((ROOT/'official_clip_verification.json').read_text())
    checks['passed']=True;(ROOT/'independent_verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
