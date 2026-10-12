"""Sequential, resumable CPU inference on the frozen manifest."""
import argparse, importlib.metadata, json, resource, time
from pathlib import Path
import numpy as np
import torch
from score_adapter import unit
ROOT=Path(__file__).resolve().parent

def feature(o):return o if isinstance(o,torch.Tensor) else o.pooler_output

def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['gemma','clip','clip_large'])
    p.add_argument('--threads',type=int,default=6);p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--dataset',choices=['sugar','human'],default='sugar')
    args=p.parse_args(); torch.set_num_threads(args.threads);torch.manual_seed(42)
    records=json.loads((ROOT/('manifest.json' if args.dataset=='sugar' else 'human_manifest.json')).read_text())['records']; names=sorted({r['filename'] for r in records})
    image_folder=ROOT/('images' if args.dataset=='sugar' else 'human_images')
    out=ROOT/'runs'/(args.kind if args.dataset=='sugar' else args.kind+'-human');out.mkdir(parents=True,exist_ok=True)
    partial=out/'partial.npz';cached={}
    if partial.exists():
        d=np.load(partial);cached=dict(zip(d['names'].tolist(),d['media']))
    start=time.perf_counter();path=ROOT/'models'/args.kind
    if args.kind=='gemma':
        from sentence_transformers import SentenceTransformer
        model=SentenceTransformer(str(path),device='cpu',local_files_only=True,config_kwargs={'audio_config':None},model_kwargs={'dtype':torch.float32,'attn_implementation':'sdpa'})
        processor=None
    else:
        from transformers import CLIPModel,CLIPProcessor
        model=CLIPModel.from_pretrained(path,local_files_only=True,dtype=torch.float32)
        processor=CLIPProcessor.from_pretrained(path,local_files_only=True)
    model.eval();timings={'load_seconds':time.perf_counter()-start};print(args.kind,'loaded',flush=True)
    missing=[n for n in names if n not in cached];start=time.perf_counter()
    with torch.inference_mode():
        for i in range(0,len(missing),args.batch_size):
            batch=missing[i:i+args.batch_size]
            if processor is None:
                e=model.encode([{'image':str(image_folder/n)} for n in batch],batch_size=args.batch_size,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
            else:
                from PIL import Image
                ims=[]
                for n in batch:
                    with Image.open(image_folder/n) as im:ims.append(im.convert('RGB'))
                e=feature(model.get_image_features(**processor(images=ims,return_tensors='pt'))).cpu().numpy()
            cached.update(zip(batch,unit(e).astype(np.float32)))
            np.savez_compressed(partial,names=np.array(sorted(cached)),media=np.stack([cached[n] for n in sorted(cached)]))
            print(args.kind,'images',min(i+args.batch_size,len(missing)),'/',len(missing),'elapsed',round(time.perf_counter()-start,1),flush=True)
        timings['media_seconds']=time.perf_counter()-start
        arrays={'media':np.stack([cached[n] for n in names])}
        prompts=['FactChecking','CLIPLiteral','None','SearchQuery'] if processor is None else ['CLIPScore']
        for prompt in prompts:
            t=time.perf_counter()
            labels=[('positive','caption'),('negative','negative_caption')] if args.dataset=='sugar' else [('candidate','caption')]
            for label,key in labels:
                texts=[r[key] for r in records]
                if prompt in ['CLIPLiteral','CLIPScore']:texts=['A photo depicts '+t for t in texts]
                if processor is None:
                    kwargs={'prompt_name':prompt} if prompt in ['FactChecking','SearchQuery'] else {'prompt':''}
                    e=model.encode(texts,batch_size=16,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False,**kwargs)
                else:
                    values=[]
                    for j in range(0,len(texts),32):
                        inputs=processor(text=texts[j:j+32],padding=True,truncation=True,max_length=77,return_tensors='pt')
                        values.append(feature(model.get_text_features(**inputs)).cpu().numpy())
                    e=np.concatenate(values)
                arrays[prompt+'_'+label]=unit(e).astype(np.float32)
            timings[prompt+'_text_seconds']=time.perf_counter()-t
            np.savez_compressed(out/'embeddings.npz',**arrays)
            print(args.kind,'text',prompt,'done',flush=True)
    receipt=json.loads((path/'download_receipt.json').read_text())
    meta={'model':receipt['model'],'revision':receipt['revision'],'media_order':names,'record_ids':[r['id'] for r in records],
      'dtype':'float32','device':'cpu','threads':args.threads,'batch_size':args.batch_size,'prompts':prompts,
      'parameters':sum(p.numel() for p in model.parameters()),'timings':timings,
      'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
      'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','sentence-transformers','numpy']},
      'finite':all(np.isfinite(v).all() for v in arrays.values())}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2));print(args.kind,'COMPLETE',flush=True)

if __name__=='__main__':main()
