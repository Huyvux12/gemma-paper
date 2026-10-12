"""Sequential full-set inference with atomic resumable checkpoints."""
import argparse,hashlib,json,os,resource,time
import numpy as np
import torch
from encoders import Encoder,ROOT,PREV

def atomic_json(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)

def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['gemma','clip','clip_large','specs','siglip']);p.add_argument('--dataset',choices=['sugar','human'],default='sugar')
    p.add_argument('--dtype',choices=['float32','bfloat16'],required=True);p.add_argument('--threads',type=int,default=6);p.add_argument('--batch-size',type=int,default=4)
    args=p.parse_args();torch.set_num_threads(args.threads);torch.manual_seed(42)
    manifest_file=ROOT/('manifest.json' if args.dataset=='sugar' else 'human_manifest.json')
    mf=json.loads(manifest_file.read_text());records=mf['records'];names=sorted(mf['media_sha256'])
    image_folder=ROOT/('images' if args.dataset=='sugar' else 'human_images')
    out=ROOT/'runs'/(args.kind if args.dataset=='sugar' else args.kind+'-human');out.mkdir(parents=True,exist_ok=True)
    config={'kind':args.kind,'dataset':args.dataset,'dtype':args.dtype,'batch_size':args.batch_size,'threads':args.threads,
      'manifest_sha256':hashlib.sha256(manifest_file.read_bytes()).hexdigest()}
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config
    else:atomic_json(out/'config.json',config)
    if (out/'metadata.json').exists():print('already complete',args.kind,args.dataset,flush=True);return
    start=time.perf_counter();encoder=Encoder(args.kind,args.dtype);timings={'load_seconds':time.perf_counter()-start}
    seed_media={};seed_text={};reused_images=reused_texts=encoded_images=encoded_texts=0
    prior=PREV/'runs'/(args.kind if args.dataset=='sugar' else args.kind+'-human')
    if args.dtype=='float32' and (prior/'metadata.json').exists():
        pm=json.loads((prior/'metadata.json').read_text());assert pm['dtype']=='float32'
        current=json.loads((PREV/'models'/args.kind/'download_receipt.json').read_text())
        assert pm['revision']==current['revision']
        oldmf=json.loads((PREV/manifest_file.name).read_text())
        with np.load(prior/'embeddings.npz') as ar:old={k:ar[k] for k in ar.files}
        for i,n in enumerate(pm['media_order']):
            if n in mf['media_sha256']:
                assert oldmf['media_sha256'][n]==mf['media_sha256'][n]
                seed_media[n]=old['media'][i]
        labels=[('positive','caption'),('negative','negative_caption')] if args.dataset=='sugar' else [('candidate','caption')]
        for prompt in encoder.prompts:
            seed_text[prompt]={row[field]:old[prompt+'_'+label][i] for label,field in labels for i,row in enumerate(oldmf['records'])}
    completed=0
    if (out/'image_state.json').exists():completed=json.loads((out/'image_state.json').read_text())['completed']
    # Probe shape only when starting; existing partial data is never relabeled.
    start=time.perf_counter()
    if completed:
        media=np.load(out/'media.npy',mmap_mode='r+')
    else:
        first_names=names[:args.batch_size];missing=[n for n in first_names if n not in seed_media]
        values=encoder.images([image_folder/n for n in missing],args.batch_size) if missing else np.empty((0,len(next(iter(seed_media.values())))))
        new=dict(zip(missing,values));first=np.stack([seed_media[n] if n in seed_media else new[n] for n in first_names])
        reused_images+=len(first_names)-len(missing);encoded_images+=len(missing)
        media=np.lib.format.open_memmap(out/'media.npy',mode='w+',dtype='float32',shape=(len(names),first.shape[1]))
        media[:len(first)]=first;completed=len(first);media.flush();atomic_json(out/'image_state.json',{'completed':completed})
    for i in range(completed,len(names),args.batch_size):
        batch_names=names[i:i+args.batch_size];missing=[n for n in batch_names if n not in seed_media]
        values=encoder.images([image_folder/n for n in missing],args.batch_size) if missing else np.empty((0,media.shape[1]))
        new=dict(zip(missing,values));emb=np.stack([seed_media[n] if n in seed_media else new[n] for n in batch_names])
        reused_images+=len(batch_names)-len(missing);encoded_images+=len(missing)
        media[i:i+len(emb)]=emb;media.flush();atomic_json(out/'image_state.json',{'completed':i+len(emb)})
        if i%64<args.batch_size:print(args.kind,args.dataset,'images',i+len(emb),'/',len(names),'elapsed',round(time.perf_counter()-start,1),flush=True)
    timings['media_seconds_this_invocation']=time.perf_counter()-start
    arrays={'media':np.array(media)}
    for prompt in encoder.prompts:
        labels=[('positive','caption'),('negative','negative_caption')] if args.dataset=='sugar' else [('candidate','caption')]
        texts=list(dict.fromkeys(r[field] for _,field in labels for r in records));index={t:i for i,t in enumerate(texts)}
        state_file=out/(prompt+'_text_state.json');array_file=out/(prompt+'_text.npy')
        completed=json.loads(state_file.read_text())['completed'] if state_file.exists() else 0
        text=np.load(array_file,mmap_mode='r+') if array_file.exists() else np.lib.format.open_memmap(array_file,mode='w+',dtype='float32',shape=(len(texts),media.shape[1]))
        start=time.perf_counter()
        for i in range(completed,len(texts),128):
            batch=texts[i:i+128];cached=seed_text.get(prompt,{});missing=[t for t in batch if t not in cached]
            values=encoder.texts(missing,prompt,batch_size=32) if missing else np.empty((0,media.shape[1]))
            new=dict(zip(missing,values));values=np.stack([cached[t] if t in cached else new[t] for t in batch]);text[i:i+len(values)]=values
            reused_texts+=len(batch)-len(missing);encoded_texts+=len(missing)
            text.flush();atomic_json(state_file,{'completed':i+len(values),'unique_texts':len(texts)})
            if i%1024==0:print(args.kind,args.dataset,prompt,'text',i+len(values),'/',len(texts),'elapsed',round(time.perf_counter()-start,1),flush=True)
        timings[prompt+'_text_seconds_this_invocation']=time.perf_counter()-start
        for label,field in labels:arrays[prompt+'_'+label]=np.array(text[[index[r[field]] for r in records]])
    assert all(np.isfinite(v).all() for v in arrays.values())
    np.savez_compressed(out/'embeddings.npz',**arrays)
    receipt_path=ROOT/'specs_checkpoint.json' if args.kind=='specs' else (ROOT/'models'/args.kind/'download_receipt.json' if args.kind=='siglip' else PREV/'models'/args.kind/'download_receipt.json')
    receipt=json.loads(receipt_path.read_text())
    meta={**config,'model':receipt['model'],'revision':receipt['revision'],'media_order':names,'record_ids':[r['id'] for r in records],
      'prompts':encoder.prompts,'native_score':('max((cosine+1)/2,0)' if args.kind=='specs' else ('cosine' if args.kind=='siglip' else '2.5*max(cosine,0)')),
      'parameters':sum(p.numel() for p in encoder.model.parameters()),'timings':timings,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
      'embedding_storage_dtype':'float32','normalization':'float64 L2 after model output converted to float32','completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    meta['reuse']={'source':'preceding float32 study, identical image SHA and text string, same pinned checkpoint',
      'images_reused_this_invocation':reused_images,'images_encoded_this_invocation':encoded_images,
      'unique_texts_reused_this_invocation':reused_texts,'unique_texts_encoded_this_invocation':encoded_texts,
      'warning':'On a resumed invocation, counts/timings describe only newly completed work.'}
    atomic_json(out/'metadata.json',meta);print('COMPLETE',args.kind,args.dataset,flush=True)

if __name__=='__main__':main()
