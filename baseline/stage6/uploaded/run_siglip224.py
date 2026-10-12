"""Complete the original primary contrast on an unrestricted CPU/GPU runtime.

Gemma reference scores come from verified fixed vectors. Only SigLIP2-224 is
new encoder inference here. This does not regenerate or modify Gemma scores.
"""
import concurrent.futures,hashlib,importlib.metadata,json,time,urllib.request
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import SiglipModel,SiglipProcessor
from scipy.stats import binomtest
ROOT=Path(__file__).resolve().parent
MODEL='google/siglip2-base-patch16-224';REVISION='75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2'
WEIGHT_SHA='612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b'

def unit(x):
    x=np.asarray(x,dtype=np.float32);return x/np.linalg.norm(x,axis=1,keepdims=True)

def feature(x):return x if isinstance(x,torch.Tensor) else x.pooler_output

def main():
    manifest=json.loads((ROOT/'frozen_manifest.json').read_text());records=manifest['records'];reference=json.loads((ROOT/'primary_reference.json').read_text());assert reference['record_ids']==[r['id'] for r in records]
    assert hashlib.sha256((ROOT/'frozen_manifest.json').read_bytes()).hexdigest()==reference['frozen_manifest_sha256']
    folder=ROOT/'images';folder.mkdir(exist_ok=True)
    def download(r):
        path=folder/r['filename']
        if not path.exists():
            for attempt in range(4):
                try:
                    with urllib.request.urlopen('https://s3.us-east-1.amazonaws.com/images.cocodataset.org/val2017/'+r['filename'],timeout=60) as resp:data=resp.read()
                    path.with_suffix('.part').write_bytes(data);path.with_suffix('.part').replace(path);break
                except Exception:
                    if attempt==3:raise
        sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha==reference['image_sha256'][r['filename']],r['filename']
        with Image.open(path) as im:im.verify()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for i,_ in enumerate(pool.map(download,records),1):
            if i%80==0:print('Verified images',i,'/560',flush=True)
    path=Path(snapshot_download(MODEL,revision=REVISION,allow_patterns=['*.json','*.safetensors','*.model']))
    digest=hashlib.sha256()
    with (path/'model.safetensors').open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    assert digest.hexdigest()==WEIGHT_SHA
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(6);torch.manual_seed(42)
    if device=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=SiglipModel.from_pretrained(path,dtype=torch.float32,local_files_only=True).to(device).eval();processor=SiglipProcessor.from_pretrained(path,local_files_only=True)
    out=ROOT/'runs/siglip224';out.mkdir(parents=True,exist_ok=True);cached={}
    if (out/'partial.npz').exists():
        ar=np.load(out/'partial.npz');cached=dict(zip(ar['names'].tolist(),ar['media']))
    names=sorted({r['filename'] for r in records});missing=[n for n in names if n not in cached];start=time.perf_counter()
    with torch.inference_mode():
        for i in range(0,len(missing),4):
            batch=missing[i:i+4];images=[]
            for name in batch:
                with Image.open(folder/name) as im:images.append(im.convert('RGB'))
            inputs=processor(images=images,return_tensors='pt').to(device)
            vectors=unit(feature(model.get_image_features(**inputs)).float().cpu().numpy());cached.update(zip(batch,vectors))
            np.savez_compressed(out/'partial.npz',names=np.array(sorted(cached)),media=np.stack([cached[n] for n in sorted(cached)]))
            print('SigLIP224 images',min(i+4,len(missing)),'/',len(missing),flush=True)
        arrays={'media':np.stack([cached[n] for n in names])};audit={}
        for prompt in ['raw','CLIPLiteral']:
            for label,key in [('positive','caption'),('negative','negative_caption')]:
                texts=[('A photo depicts ' if prompt=='CLIPLiteral' else '')+r[key] for r in records]
                lengths=[len(ids) for ids in processor.tokenizer(texts,add_special_tokens=True)['input_ids']]
                limit=model.config.text_config.max_position_embeddings;audit[prompt+'_'+label]={'max_tokens':max(lengths),'context':limit,'over_context':sum(x>limit for x in lengths)}
                assert max(lengths)<=limit,'Truncation would invalidate declared comparison'
                chunks=[]
                for i in range(0,len(texts),32):
                    inputs=processor(text=texts[i:i+32],padding='max_length',truncation=True,max_length=limit,return_tensors='pt').to(device)
                    chunks.append(feature(model.get_text_features(**inputs)).float().cpu().numpy())
                arrays[prompt+'_'+label]=unit(np.concatenate(chunks))
        np.savez_compressed(out/'embeddings.npz',**arrays)
    idx={n:i for i,n in enumerate(names)};im=arrays['media'][[idx[r['filename']] for r in records]]
    def score(prompt,label):return 2.5*np.maximum(np.sum(im*arrays[prompt+'_'+label],1),0)
    g=np.asarray(reference['gemma_positive'])>np.asarray(reference['gemma_negative']);s=score('raw','positive')>score('raw','negative');d=g.astype(int)-s.astype(int);rng=np.random.default_rng(20261009)
    boot=np.concatenate([d[rng.integers(0,560,(500,560))].mean(1) for _ in range(20)]);wins=int(np.sum(g&~s));losses=int(np.sum(~g&s))
    result={'primary_completed':True,'n_images':560,'gemma_correct':int(g.sum()),'siglip224_correct':int(s.sum()),'gemma_accuracy':float(g.mean()),'siglip224_accuracy':float(s.mean()),'delta_pp':float(d.mean()*100),'ci95_pp':(100*np.quantile(boot,[.025,.975])).tolist(),'mcnemar_p':float(binomtest(wins,wins+losses).pvalue) if wins+losses else 1.,'wins':wins,'losses':losses,'model':MODEL,'revision':REVISION,'weight_sha256':WEIGHT_SHA,'dtype':'float32','device':device,'token_audit':audit,'elapsed_seconds_including_text':time.perf_counter()-start,'manifest_sha256':reference['frozen_manifest_sha256'],'gemma_origin':'Verified archived float32 vectors; SigLIP224 new inference in this run.','versions':{k:importlib.metadata.version(k) for k in ['torch','transformers','numpy']}}
    (ROOT/'primary_siglip224_results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
