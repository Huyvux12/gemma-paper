"""Independent scalar arithmetic for all frozen-sample scores and archive bridges."""
import csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;FULL=ROOT.parent/'full_recovered/embeddinggemma2_full_validation/full_validation';OLD=ROOT.parent/'research/embeddinggemma2_clipscore_replacement'

def scalar_cos(a,b):
    aa=list(map(float,a));bb=list(map(float,b))
    return math.fsum(x*y for x,y in zip(aa,bb))/math.sqrt(math.fsum(x*x for x in aa)*math.fsum(y*y for y in bb))

def transform(v,d,fmt):
    x=list(map(float,v[:d]));norm=math.sqrt(math.fsum(t*t for t in x));x=[t/norm for t in x]
    if fmt=='float16':return list(map(float,np.asarray(x,dtype=np.float16)))
    if fmt=='int8':
        scale=float(np.float32(max(map(abs,x))/127));return [max(-127,min(127,round(t/scale)))*scale for t in x]
    return x

def main():
    r=json.loads((ROOT/'frozen_manifest.json').read_text())['records'];index={row['id']:i for i,row in enumerate(r)};archives={};scores=list(csv.DictReader((ROOT/'pair_scores_560.csv').open()));maximum=0;count=0;vectorcache={};decisions={}
    for kind in ['gemma','clip','clip_large','siglip','specs']:
        folder=ROOT/'runs'/kind;m=json.loads((folder/'metadata.json').read_text());assert m['record_ids']==[x['id'] for x in r];loaded=np.load(folder/'embeddings.npz');archives[kind]=({k:loaded[k] for k in loaded.files},{n:i for i,n in enumerate(m['media_order'])})
    for row in scores:
        kind,prompt,d,fmt=row['method'].split('-');d=int(d);z,ix=archives[kind];i=index[row['id']];key=(kind,row['filename'],d,fmt)
        if key not in vectorcache:vectorcache[key]=transform(z['media'][ix[row['filename']]],d,fmt)
        a=vectorcache[key];values=[]
        for label in ['positive','negative']:
            value=scalar_cos(a,transform(z[prompt+'_'+label][i],d,fmt))
            if kind=='specs':value=max((value+1)/2,0)
            elif kind!='siglip':value=2.5*max(value,0)
            maximum=max(maximum,abs(value-float(row[label+'_score'])));values.append(value);count+=1
        ok=int(values[0]>values[1]);assert ok==int(row['correct']);decisions.setdefault(row['method'],[]).append(ok)
    assert maximum<1e-10,maximum
    results=json.loads((ROOT/'results_560.json').read_text())
    for k,v in decisions.items():assert sum(v)==results['metrics'][k]['correct']
    # Reconstruct each paired discordance from CSV rather than from analyzer's bool arrays.
    base='gemma-CLIPLiteral-768-float32'
    for k,comp in results['comparisons'].items():
        a=decisions[base];b=decisions[k];wins=sum(x==1 and y==0 for x,y in zip(a,b));losses=sum(x==0 and y==1 for x,y in zip(a,b));assert wins==comp['wins'] and losses==comp['losses']
        total=wins+losses;p=min(1.,2*sum(math.comb(total,j) for j in range(min(wins,losses)+1))/(2**total)) if total else 1.
        assert abs(p-comp['p_exact'])<1e-12
    if '--scores-only' in sys.argv:
        out={'scalar_scores_checked':count,'maximum_abs_score_error':maximum,'all_560_method_counts_match':True,'paired_exact_pvalues_match':True,'archive_bridge_checked':False,'passed':True}
        (ROOT/'independent_scores_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2));return
    bridge={}
    # Verify the recovery contains the same archived old vectors/caption order where available.
    old_records=json.loads((OLD/'manifest.json').read_text())['records'];full_records=json.loads((FULL/'manifest.json').read_text())['records'];fi={row['id']:i for i,row in enumerate(full_records)}
    for kind,prompt in [('gemma','CLIPLiteral'),('clip','CLIPScore'),('clip_large','CLIPScore')]:
        oldm=json.loads((OLD/'runs'/kind/'metadata.json').read_text());newm=json.loads((FULL/'runs'/kind/'metadata.json').read_text());oz=np.load(OLD/'runs'/kind/'embeddings.npz');nz=np.load(FULL/'runs'/kind/'embeddings.npz');oldz={k:oz[k] for k in oz.files};newz={k:nz[k] for k in nz.files};oi={n:i for i,n in enumerate(oldm['media_order'])};ni={n:i for i,n in enumerate(newm['media_order'])};err=0;agree=0
        for i,row in enumerate(old_records):
            j=fi[row['id']];assert all(row[t]==full_records[j][t] for t in ['filename','caption','negative_caption'])
            a=oldz['media'][oi[row['filename']]];b=newz['media'][ni[row['filename']]];err=max(err,float(np.max(abs(a-b))))
            vals=[]
            for lab in ['positive','negative']:
                x=oldz[prompt+'_'+lab][i];y=newz[prompt+'_'+lab][j];err=max(err,float(np.max(abs(x-y))));vals.append((scalar_cos(a,x),scalar_cos(b,y)))
            agree+=int((vals[0][0]>vals[1][0])==(vals[0][1]>vals[1][1]))
        assert agree==350;bridge[kind]={'n_old_records':350,'maximum_embedding_element_difference':err,'all_old_decisions_match':True}
    # Confirm disjointness once more using IDs and media, not source's flags alone.
    prior={row['filename'] for file in ['manifest.json','data/prior_image_manifest.json'] for row in json.loads((OLD/file).read_text())['records']}
    assert not prior&{row['filename'] for row in r}
    out={'scalar_scores_checked':count,'maximum_abs_score_error':maximum,'all_560_method_counts_match':True,'paired_exact_pvalues_match':True,'560_prior_media_overlap':0,'old_vector_bridge':bridge,'source':'Recovered inference vectors, not new model inference.','requested_siglip224_inference':'blocked by network; pending','passed':True}
    (ROOT/'independent_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
