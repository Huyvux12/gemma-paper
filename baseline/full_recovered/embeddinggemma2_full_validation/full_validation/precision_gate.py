"""Validate BF16 using previous-study outputs; no new-media scores inspected."""
import argparse,gc,json,random,time
from pathlib import Path
import numpy as np
import torch
from encoders import Encoder,PREV,ROOT
from score_adapter import clipscore_from_embeddings

def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['gemma','clip','clip_large']);a=p.parse_args()
    torch.set_num_threads(6);torch.manual_seed(42)
    sr=json.loads((PREV/'manifest.json').read_text())['records'];hr=json.loads((PREV/'human_manifest.json').read_text())['records']
    rng=random.Random(202610081);si=sorted(rng.sample(range(len(sr)),32));hi=sorted(rng.sample(range(len(hr)),32))
    model=Encoder(a.kind,'bfloat16');out={};start=time.perf_counter()
    for ds,rows,indices,folder in [('sugar',sr,si,'images'),('human',hr,hi,'human_images')]:
        path=PREV/'runs'/(a.kind if ds=='sugar' else a.kind+'-human')
        meta=json.loads((path/'metadata.json').read_text());names=meta['media_order'];ix={n:i for i,n in enumerate(names)}
        with np.load(path/'embeddings.npz') as arch:old={k:arch[k] for k in arch.files}
        selected=[rows[i] for i in indices];media=model.images([PREV/folder/r['filename'] for r in selected])
        om=old['media'][[ix[r['filename']] for r in selected]]
        cosine=np.einsum('ij,ij->i',media.astype(float),om.astype(float));checks={}
        for prompt in model.prompts:
            labels=[('positive','caption'),('negative','negative_caption')] if ds=='sugar' else [('candidate','caption')]
            scores={};reference={};textcos=[]
            for label,field in labels:
                text=model.texts([r[field] for r in selected],prompt)
                ref=old[prompt+'_'+label][indices]
                textcos.extend(np.einsum('ij,ij->i',text.astype(float),ref.astype(float)))
                scores[label]=clipscore_from_embeddings(media,text)
                reference[label]=clipscore_from_embeddings(om,ref)
            error=max(float(np.max(np.abs(scores[k]-reference[k]))) for k in scores)
            agreement=float(np.mean((scores['positive']>scores['negative'])==(reference['positive']>reference['negative']))) if ds=='sugar' else None
            checks[prompt]={'maximum_score_abs_error':error,'minimum_text_embedding_cosine':float(min(textcos)),
              'ranking_agreement':agreement,'passed':bool(error<=.02 and min(textcos)>=.99 and (agreement is None or agreement>=.95))}
        out[ds]={'n_records':len(selected),'indices':indices,'minimum_image_embedding_cosine':float(min(cosine)),
          'checks':checks,'passed':bool(min(cosine)>=.99 and all(c['passed'] for c in checks.values()))}
        print(a.kind,ds,json.dumps(out[ds]),flush=True)
    out['kind']=a.kind;out['dtype']='bfloat16';out['elapsed_seconds']=time.perf_counter()-start
    out['passed']=all(out[k]['passed'] for k in ['sugar','human'])
    (ROOT/f'precision_{a.kind}.json').write_text(json.dumps(out,indent=2))

if __name__=='__main__':main()
