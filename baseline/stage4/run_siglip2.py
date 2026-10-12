"""New baseline inference on the archived, observed evaluation samples."""
import argparse,importlib.metadata,json,resource,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import SiglipModel,SiglipProcessor
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'research/embeddinggemma2_clipscore_replacement'

def unit(x):
    x=np.asarray(x,dtype=np.float32)
    return x/np.linalg.norm(x,axis=-1,keepdims=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['sugar','human'],default='sugar');args=ap.parse_args()
    torch.set_num_threads(6);torch.manual_seed(42)
    receipt=json.loads((ROOT/'download_receipt.json').read_text())
    start=time.perf_counter()
    model=SiglipModel.from_pretrained(ROOT/'models/siglip2',local_files_only=True,torch_dtype=torch.float32).eval()
    processor=SiglipProcessor.from_pretrained(ROOT/'models/siglip2',local_files_only=True)
    timings={'load_seconds':time.perf_counter()-start}
    manifest=SOURCE/('manifest.json' if args.dataset=='sugar' else 'human_manifest.json')
    records=json.loads(manifest.read_text())['records'];names=sorted({r['filename'] for r in records})
    folder=SOURCE/('images' if args.dataset=='sugar' else 'human_images')
    out=ROOT/('siglip2' if args.dataset=='sugar' else 'siglip2-human');out.mkdir(exist_ok=True)
    arrays={};start=time.perf_counter();cached={}
    if (out/'partial.npz').exists():
        e=np.load(out/'partial.npz');cached=dict(zip(e['names'].tolist(),e['media']))
    missing=[n for n in names if n not in cached]
    with torch.inference_mode():
        for i in range(0,len(missing),4):
            batch=missing[i:i+4];images=[]
            for n in batch:
                with Image.open(folder/n) as im:images.append(im.convert('RGB'))
            inputs=processor(images=images,return_tensors='pt')
            v=model.get_image_features(**inputs).cpu().numpy()
            cached.update(zip(batch,unit(v)))
            np.savez_compressed(out/'partial.npz',names=np.array(sorted(cached)),media=np.stack([cached[n] for n in sorted(cached)]))
            print('images',min(i+4,len(missing)),'/',len(missing),'seconds',round(time.perf_counter()-start,1),flush=True)
        timings['media_seconds']=time.perf_counter()-start
        arrays['media']=np.stack([cached[n] for n in names])
        audit={}
        for prompt in ['raw','CLIPLiteral']:
            start=time.perf_counter()
            labels=[('positive','caption'),('negative','negative_caption')] if args.dataset=='sugar' else [('candidate','caption')]
            for label,key in labels:
                texts=[('A photo depicts ' if prompt=='CLIPLiteral' else '')+r[key] for r in records]
                lengths=[len(x) for x in processor.tokenizer(texts,add_special_tokens=True)['input_ids']]
                audit[prompt+'_'+label]={'max_tokens':max(lengths),'n_over_context':sum(x>model.config.text_config.max_position_embeddings for x in lengths),'context':model.config.text_config.max_position_embeddings}
                if any(x>model.config.text_config.max_position_embeddings for x in lengths):raise ValueError('Caption truncation')
                chunks=[]
                for i in range(0,len(texts),32):
                    inputs=processor(text=texts[i:i+32],padding='max_length',truncation=True,max_length=model.config.text_config.max_position_embeddings,return_tensors='pt')
                    chunks.append(model.get_text_features(**inputs).cpu().numpy())
                arrays[prompt+'_'+label]=unit(np.concatenate(chunks))
            timings[prompt+'_text_seconds']=time.perf_counter()-start
            np.savez_compressed(out/'embeddings.npz',**arrays)
            print('text',prompt,'complete',flush=True)
        # Check direct joint-forward calibrated logits against separately encoded cosine.
        r=records[0]
        with Image.open(folder/r['filename']) as im: image=im.convert('RGB')
        texts=[r['caption'],r.get('negative_caption',r['caption'])]
        inputs=processor(text=texts,images=image,padding='max_length',max_length=model.config.text_config.max_position_embeddings,return_tensors='pt')
        direct=model(**inputs).logits_per_image.cpu().numpy()[0]
        ix=names.index(r['filename']);pos=arrays['raw_positive' if args.dataset=='sugar' else 'raw_candidate'][0]
        neg=arrays['raw_negative'][0] if args.dataset=='sugar' else pos
        expected=np.array([arrays['media'][ix]@pos,arrays['media'][ix]@neg])*float(model.logit_scale.exp())+float(model.logit_bias)
        error=float(np.max(np.abs(direct-expected)))
        assert error<1e-3,(direct,expected)
    meta={'model':receipt['model'],'revision':receipt['revision'],'media_order':names,'record_ids':[r['id'] for r in records],'parameters':sum(p.numel() for p in model.parameters()),'dtype':'float32','threads':6,'batch_size':4,'timings':timings,'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'prompts':['raw','CLIPLiteral'],'token_audit':audit,'joint_forward_max_abs_logit_error':error,'versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','numpy']}}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2))
    print('COMPLETE',args.dataset,flush=True)

if __name__=='__main__':main()
