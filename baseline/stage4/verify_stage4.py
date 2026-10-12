"""Independent scalar score checks and decoded int8 integration verification."""
import csv,hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau

ROOT=Path(__file__).resolve().parent;SOURCE=ROOT.parent/'research/embeddinggemma2_clipscore_replacement'

def scalar(a,b):
    aa=list(map(float,a));bb=list(map(float,b))
    return 2.5*max(math.fsum(x*y for x,y in zip(aa,bb))/math.sqrt(math.fsum(x*x for x in aa))/math.sqrt(math.fsum(y*y for y in bb)),0)

def main():
    records=json.loads((SOURCE/'manifest.json').read_text())['records'];meta=json.loads((SOURCE/'runs/gemma/metadata.json').read_text());e=np.load(SOURCE/'runs/gemma/embeddings.npz');ix={n:i for i,n in enumerate(meta['media_order'])}
    out={'scope':'Independent scalar recomputation of exported scores; decode actual int8 file; dataset and frozen protocol hashes.'}
    protocol=json.loads((ROOT/'protocol.json').read_text())
    for name,digest in protocol['source_hashes'].items():assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==digest
    rows=list(csv.DictReader((ROOT/'compression_scores.csv').open()));max_error=0;count=0
    # Reconstruct each quantization with scalar arithmetic independently of diagnostics.
    def encode(v,d,fmt):
        x=list(map(float,v[:d]));norm=math.sqrt(math.fsum(t*t for t in x));x=[t/norm for t in x]
        if fmt=='float16':x=list(map(float,np.asarray(x,dtype=np.float16)))
        elif fmt=='int8':
            scale=max(map(abs,x))/127;x=[float(round(t/scale))*scale for t in x]
        return x
    cache={}
    for row in rows:
        prompt,d,fmt=row['method'].split('-');d=int(d);r_index=next(i for i,r in enumerate(records) if r['id']==row['id'])
        a_key=(row['filename'],d,fmt)
        if a_key not in cache:cache[a_key]=encode(e['media'][ix[row['filename']]],d,fmt)
        values=[]
        for label in ['positive','negative']:
            val=scalar(cache[a_key],encode(e[prompt+'_'+label][r_index],d,fmt));values.append(val)
            max_error=max(max_error,abs(val-float(row[label+'_score'])));count+=1
        assert int(values[0]>values[1])==int(row['correct'])
    assert max_error<1e-10,max_error
    out['compression_scores']={'n':count,'max_abs_error':max_error,'all_decisions_match':True}
    actual=np.load(ROOT/'gemma_256_int8.npz');hits={};payload=0
    for key in actual.files:payload+=actual[key].nbytes
    for prompt in ['CLIPLiteral','FactChecking']:
        wins=0
        for i,r in enumerate(records):
            row=ix[r['filename']];a=actual['media_values'][row].astype(float)*float(actual['media_scale'][row,0])
            values=[]
            for label in ['positive','negative']:
                k=prompt+'_'+label;b=actual[k+'_values'][i].astype(float)*float(actual[k+'_scale'][i,0]);values.append(scalar(a,b))
            wins+=int(values[0]>values[1])
        expected=json.loads((ROOT/'diagnostics_results.json').read_text())['compression'][prompt+'-256-int8']['correct']
        assert wins==expected,(wins,expected);hits[prompt]=wins
    out['actual_int8_file']={'payload_bytes':payload,'correct':hits,'float32_scale_roundtrip_decisions_match':True,'bytes_per_vector':260}
    # Verify all baseline CSV decisions and scores without vectorized scoring code.
    rows=list(csv.DictReader((ROOT/'baseline_pair_scores.csv').open()));archives={};baseline_error=0
    for k in ['gemma','clip','clip_large','siglip2']:
        folder=ROOT/'siglip2' if k=='siglip2' else SOURCE/'runs'/k
        m=json.loads((folder/'metadata.json').read_text());z=np.load(folder/'embeddings.npz');archives[k]=(z,{n:i for i,n in enumerate(m['media_order'])})
    r_index={r['id']:i for i,r in enumerate(records)}
    for row in rows:
        kind,prompt=row['method'].split('-',1);z,ix=archives[kind];a=z['media'][ix[row['filename']]];values=[]
        for label in ['positive','negative']:
            value=scalar(a,z[prompt+'_'+label][r_index[row['id']]]);values.append(value);baseline_error=max(baseline_error,abs(value-float(row[label+'_score'])))
        assert int(values[0]>values[1])==int(row['correct'])
    assert baseline_error<1e-10
    out['baseline_scores']={'n':len(rows)*2,'max_abs_error':baseline_error,'all_decisions_match':True}
    # Scalar wrong-image audit for a reproducible subset of rows, all 349 alternatives.
    dg=json.loads((ROOT/'diagnostics_results.json').read_text());wrong=0
    for i in range(12):
        correct_image=meta['media_order'].index(records[i]['filename'])
        for j in range(350):
            if j==correct_image:continue
            a=e['media'][j];wrong+=int(scalar(a,e['CLIPLiteral_positive'][i])>scalar(a,e['CLIPLiteral_negative'][i]))
    # Recompute matching matrix using separate einsum construction, only verification.
    media=e['media'].astype(float);media/=np.linalg.norm(media,axis=1,keepdims=True)
    pos=e['CLIPLiteral_positive'][:12].astype(float);pos/=np.linalg.norm(pos,axis=1,keepdims=True)
    neg=e['CLIPLiteral_negative'][:12].astype(float);neg/=np.linalg.norm(neg,axis=1,keepdims=True)
    m=2.5*np.maximum(np.einsum('id,jd->ij',pos,media),0)-2.5*np.maximum(np.einsum('id,jd->ij',neg,media),0)
    mask=m>0
    for i in range(12):mask[i,meta['media_order'].index(records[i]['filename'])]=False
    assert int(mask.sum())==wrong
    out['wrong_image_scalar_audit']={'pairs':12*349,'decisions_match':True}
    # Recompute SigLIP human tau with scalar scores and original individual ratings.
    hrecords=json.loads((SOURCE/'human_manifest.json').read_text())['records'];hm=json.loads((ROOT/'siglip2-human/metadata.json').read_text());hz=np.load(ROOT/'siglip2-human/embeddings.npz');hx={n:i for i,n in enumerate(hm['media_order'])}
    human=json.loads((ROOT/'baseline_results.json').read_text())['human'];taus={}
    for prompt in ['raw','CLIPLiteral']:
        scores=[];ratings=[]
        for i,r in enumerate(hrecords):
            v=scalar(hz['media'][hx[r['filename']]],hz[prompt+'_candidate'][i]);scores.extend([v]*len(r['ratings']));ratings.extend(r['ratings'])
        tau=float(kendalltau(scores,ratings,variant='c').statistic);assert abs(tau-human['tau_c']['siglip2-'+prompt])<1e-12;taus[prompt]=tau
    out['human_scalar_tau']=taus
    out['passed']=True;(ROOT/'independent_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
