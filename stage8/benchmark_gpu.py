"""Fresh same-session encoder benchmark. No cache timings or CPU/GPU mixing."""
import argparse,gc,json,sys,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from common import read,write,digest
from train_adapter import load_model

def unwrap(x):return x.pooler_output if hasattr(x,'pooler_output') else x

def measure(fn,batch_size,repeats=15,warmup=3):
    with torch.inference_mode():
        for _ in range(warmup):fn()
        torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();samples=[]
        for _ in range(repeats):
            torch.cuda.synchronize();start=time.perf_counter();score=fn();torch.cuda.synchronize()
            samples.append((time.perf_counter()-start)*1000);assert score.shape==(batch_size,) and torch.isfinite(score).all()
    return {'batch_size':batch_size,'warmup_iterations':warmup,'measured_iterations':repeats,
        'p50_wall_ms_per_batch':float(np.median(samples)),'p95_wall_ms_per_batch':float(np.quantile(samples,.95)),
        'pairs_per_second_from_median':batch_size*1000/float(np.median(samples)),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'samples_wall_ms':samples}

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--trained',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    assert torch.cuda.is_available(),'Fresh GPU benchmark requires CUDA; no CPU fallback'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    b=Path(a.baseline);sys.path.insert(0,str(b/'stage7'))
    from download_required import data
    from materialize_expert import main as materialize
    data();materialize()
    mf=read(b/'stage7/human_manifest.json');one={}
    for r in mf['records']:one.setdefault(r['filename'],r['caption'])
    names=sorted(one)[:8];captions=[one[n] for n in names];paths=[b/'stage7/images'/n for n in names]
    from sentence_transformers import SentenceTransformer
    from transformers import CLIPModel,CLIPProcessor,SiglipModel,SiglipProcessor
    from huggingface_hub import hf_hub_download
    defs=[('gemma','google/embeddinggemma-2','914f7f89142e33e77833254d9c9b90c3cef7303b'),
        ('siglip224','google/siglip2-base-patch16-224','75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2'),
        ('clip-B32','openai/clip-vit-base-patch32','3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268')]
    expected={'gemma':'197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79','siglip224':'612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b'}
    selection=read(Path(a.trained)/'selection.json');results={};receipts={}
    for kind,repo,revision in defs:
        if kind=='gemma':
            model=SentenceTransformer(repo,revision=revision,device='cuda',config_kwargs={'audio_config':None},model_kwargs={'dtype':torch.float32,'attn_implementation':'sdpa'}).eval();processor=None
        elif kind=='siglip224':
            model=SiglipModel.from_pretrained(repo,revision=revision,dtype=torch.float32).to('cuda').eval();processor=SiglipProcessor.from_pretrained(repo,revision=revision)
        else:
            model=CLIPModel.from_pretrained(repo,revision=revision,dtype=torch.float32).to('cuda').eval();processor=CLIPProcessor.from_pretrained(repo,revision=revision)
        assert next(model.parameters()).dtype==torch.float32
        weight_file='pytorch_model.bin' if kind=='clip-B32' else 'model.safetensors'
        wp=Path(hf_hub_download(repo,weight_file,revision=revision));weight_sha=digest(wp)
        if kind in expected:assert weight_sha==expected[kind]
        receipts[kind]={'model':repo,'revision':revision,'weights_sha256':weight_sha,'loaded_parameters':sum(v.numel() for v in model.parameters())}
        for batch in [1,8]:
            heads=[]
            chosen=paths[:batch];text=captions[:batch];rendered=[('A photo depicts ' if kind in ['gemma','clip-B32'] else '')+s for s in text]
            def encode_pair():
                if processor is None:
                    image=model.encode([{'image':str(n)} for n in chosen],batch_size=batch,convert_to_tensor=True,normalize_embeddings=True,show_progress_bar=False)
                    sentence=model.encode(rendered,batch_size=batch,prompt='',convert_to_tensor=True,normalize_embeddings=True,show_progress_bar=False)
                else:
                    ims=[]
                    for path in chosen:
                        with Image.open(path) as im:ims.append(im.convert('RGB'))
                    inputs=processor(images=ims,return_tensors='pt').to('cuda');image=unwrap(model.get_image_features(**inputs))
                    tt=processor(text=rendered,padding='max_length',truncation=True,max_length=64 if kind=='siglip224' else 77,return_tensors='pt').to('cuda')
                    sentence=unwrap(model.get_text_features(**tt));image=torch.nn.functional.normalize(image,dim=-1);sentence=torch.nn.functional.normalize(sentence,dim=-1)
                return image,sentence
            def original():
                image,sentence=encode_pair();return (image*sentence).sum(-1)
            results.setdefault(kind,[]).append(measure(original,batch))
            if kind=='gemma':
                heads=[load_model(Path(a.trained)/r['checkpoint'],r['rank'],'cuda') for r in selection['finalists']]
                def adapted():
                    image,sentence=encode_pair();return torch.stack([h.scores(image,sentence) for h in heads]).mean(0)
                results.setdefault('gemma-adapter-ensemble5',[]).append(measure(adapted,batch))
                del heads
                heads=[]
        del model,processor,heads
        gc.collect();torch.cuda.empty_cache()
        write(a.output,{'completed_models':list(receipts),'measurements':results,'receipts':receipts,
            'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'precision':'float32, TF32 disabled',
            'scope':'Fresh full image+text encoding including image decode/native preprocessing and head scoring; model download/load excluded. Same eight real expert inputs. GPU0 only. Native image budgets differ.',
            'input_image_sha256':{n:digest(path) for n,path in zip(names,paths)},'completed':len(receipts)==3})
    print('Fresh GPU benchmark complete',flush=True)

if __name__=='__main__':main()
