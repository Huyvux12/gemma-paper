"""Float32 resumable completion with fixed cache provenance and bridge gates."""
import argparse,gc,hashlib,importlib.metadata,json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parent

def unit(x):
 x=np.asarray(x,dtype=np.float64);return (x/np.linalg.norm(x,axis=1,keepdims=True)).astype(np.float32)

def arr(x):
 if hasattr(x,'pooler_output'):x=x.pooler_output
 return unit(x.detach().float().cpu().numpy())

def save(path,obj):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2));temp.replace(path)

def features(model,processor,paths,device):
 ims=[]
 for p in paths:
  with Image.open(p) as im:ims.append(im.convert('RGB'))
 inp=processor(images=ims,return_tensors='pt').to(device)
 return arr(model.get_image_features(**inp))

def bridge(old,new):
 old=np.asarray(old,dtype=float);new=np.asarray(new,dtype=float)
 cos=np.sum(old*new,axis=1)/np.linalg.norm(old,axis=1)/np.linalg.norm(new,axis=1)
 error=float(np.max(abs(old-new)));minimum=float(cos.min())
 return {'minimum_cosine':minimum,'maximum_element_error':error,'passed':minimum>=.9999 and error<=.002}

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('kind',choices=['gemma','siglip224']);p.add_argument('--device',default='auto');p.add_argument('--threads',type=int,default=6);args=p.parse_args()
 torch.set_num_threads(args.threads);torch.manual_seed(42)
 device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
 if device=='cuda':torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 mf=json.loads((ROOT/'human_manifest.json').read_text());records=mf['records'];names=sorted(mf['media_sha256']);sha=hashlib.sha256((ROOT/'human_manifest.json').read_bytes()).hexdigest()
 for n in names:assert hashlib.sha256((ROOT/'images'/n).read_bytes()).hexdigest()==mf['media_sha256'][n]
 out=ROOT/'runs'/(args.kind+'-human');out.mkdir(parents=True,exist_ok=True)
 modeldir=ROOT/'models'/args.kind
 if args.kind=='gemma':
  receipt=json.loads((ROOT/'gemma_checkpoint.json').read_text());prompts=['CLIPLiteral','FactChecking']
 else:
  receipt={'model':'google/siglip2-base-patch16-224','revision':'75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2','files':[{'filename':'model.safetensors','sha256':'612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b'}]};prompts=['raw','CLIPLiteral']
 if (out/'metadata.json').exists():
  m=json.loads((out/'metadata.json').read_text());assert m['manifest_sha256']==sha and m['revision']==receipt['revision'];print('Already complete',args.kind);return
 for f in receipt['files']:
  h=hashlib.sha256()
  with (modeldir/f['filename']).open('rb') as inp:
   for b in iter(lambda:inp.read(8*1024*1024),b''):h.update(b)
  assert h.hexdigest()==f['sha256']
 start=time.perf_counter()
 if args.kind=='gemma':
  from sentence_transformers import SentenceTransformer
  model=SentenceTransformer(str(modeldir),device=device,local_files_only=True,config_kwargs={'audio_config':None},model_kwargs={'dtype':torch.float32,'attn_implementation':'sdpa'}).eval();processor=None
 else:
  from transformers import SiglipModel,SiglipProcessor
  model=SiglipModel.from_pretrained(modeldir,dtype=torch.float32,local_files_only=True).to(device).eval();processor=SiglipProcessor.from_pretrained(modeldir,local_files_only=True)
 load_seconds=time.perf_counter()-start
 assert next(model.parameters()).dtype==torch.float32
 def encode_images(batch):
  if processor is None:return arr(model.encode([{'image':str(ROOT/'images'/n)} for n in batch],batch_size=4,normalize_embeddings=False,convert_to_tensor=True,show_progress_bar=False))
  return features(model,processor,[ROOT/'images'/n for n in batch],device)
 def encode_texts(batch,prompt):
  text=[('A photo depicts ' if prompt=='CLIPLiteral' else '')+t for t in batch]
  if processor is None:
   kw={'prompt':''} if prompt=='CLIPLiteral' else {'prompt_name':prompt}
   return arr(model.encode(text,batch_size=32,normalize_embeddings=False,convert_to_tensor=True,show_progress_bar=False,**kw))
  inp=processor(text=text,padding='max_length',max_length=64,truncation=True,return_tensors='pt').to(device)
  return arr(model.get_text_features(**inp))
 cached={};seed_text={};bridges={};seeded_images=0
 if args.kind=='gemma':
  seed=np.load(ROOT/'cache/gemma_media_936.npz');cached=dict(zip(seed['names'].tolist(),seed['media']))
  pm=json.loads((ROOT/'cache/gemma_prior100_metadata.json').read_text());pz=np.load(ROOT/'cache/gemma_prior100.npz')
  assert pm['revision']==receipt['revision']
  cached.update({n:pz['media'][i] for i,n in enumerate(pm['media_order']) if n not in cached})
  seed_text={prompt:dict(zip(pm['captions'],pz[prompt+'_candidate'])) for prompt in prompts}
  bridge_names=[names[i] for i in np.random.default_rng(202610087).choice(len(names[:936]),16,replace=False)]
  new=np.concatenate([encode_images(bridge_names[i:i+4]) for i in range(0,16,4)])
  bridges['images']=bridge(np.stack([cached[n] for n in bridge_names]),new)
  bridge_captions=pm['captions'][:32]
  for prompt in prompts:bridges[prompt]=bridge(np.stack([seed_text[prompt][t] for t in bridge_captions]),encode_texts(bridge_captions,prompt))
  if not all(b['passed'] for b in bridges.values()):cached={};seed_text={};print('Cache bridge failed; re-encoding full data.',flush=True)
  else:print('All cache bridges passed',json.dumps(bridges),flush=True)
  seeded_images=len(cached)
 state=out/'partial.npz'
 if state.exists():
  ar=np.load(state);assert str(ar['manifest_sha256'].item())==sha
  cached.update(dict(zip(ar['names'].tolist(),ar['media'])))
 missing=[n for n in names if n not in cached];start=time.perf_counter()
 for i in range(0,len(missing),4):
  batch=missing[i:i+4];new=encode_images(batch);cached.update(zip(batch,new))
  np.savez_compressed(state,names=np.array(sorted(cached)),media=np.stack([cached[n] for n in sorted(cached)]),manifest_sha256=np.array(sha))
  print(args.kind,'images',min(i+4,len(missing)),'/',len(missing),'seconds',round(time.perf_counter()-start,1),flush=True)
 media_seconds=time.perf_counter()-start;arrays={'media':np.stack([cached[n] for n in names])};token_audit={};unique=list(dict.fromkeys(r['caption'] for r in records));ix={t:i for i,t in enumerate(unique)};text_times={};text_reused={}
 for prompt in prompts:
  rendered=[('A photo depicts ' if prompt=='CLIPLiteral' else '')+t for t in unique]
  if processor is not None:
   lengths=[len(x) for x in processor.tokenizer(rendered,add_special_tokens=True)['input_ids']];limit=64
  else:
   pre=model.prompts[prompt] if prompt=='FactChecking' else ''
   lengths=[len(x) for x in model.tokenizer([pre+t for t in rendered],add_special_tokens=True)['input_ids']];limit=min(int(model.max_seq_length),int(json.loads((modeldir/'config.json').read_text())['text_config']['max_position_embeddings']))
  token_audit[prompt]={'max_tokens':max(lengths),'context':limit,'over_context':sum(x>limit for x in lengths)};assert max(lengths)<=limit
  dest=out/(prompt+'_partial_text.npz');vectors=dict(seed_text.get(prompt,{}));text_reused[prompt]=len(set(unique)&set(vectors))
  if dest.exists():
   ar=np.load(dest);assert str(ar['manifest_sha256'].item())==sha;vectors.update(dict(zip(ar['texts'].tolist(),ar['text'])))
  todo=[t for t in unique if t not in vectors];start=time.perf_counter()
  for i in range(0,len(todo),32):
   batch=todo[i:i+32];new=encode_texts(batch,prompt);vectors.update(zip(batch,new))
   if i%256==0 or i+32>=len(todo):
    np.savez_compressed(dest,texts=np.array(list(vectors)),text=np.stack(list(vectors.values())),manifest_sha256=np.array(sha))
    print(args.kind,prompt,'text',min(i+32,len(todo)),'/',len(todo),'seconds',round(time.perf_counter()-start,1),flush=True)
  text_times[prompt]=time.perf_counter()-start;arrays[prompt+'_candidate']=np.stack([vectors[r['caption']] for r in records])
 for v in arrays.values():assert v.shape==(1000,768) or v.shape==(len(records),768)
 np.savez_compressed(out/'embeddings.npz',**arrays)
 meta={'kind':args.kind,'dataset':'human','model':receipt['model'],'revision':receipt['revision'],'weight_hashes':receipt['files'],'dtype':'float32','device':device,'threads':args.threads,'batch_size':4,'manifest_sha256':sha,'record_ids':[r['id'] for r in records],'media_order':names,'prompts':prompts,'native_score':'2.5*max(cosine,0)','token_audit':token_audit,'cache_bridges':bridges,'seeded_image_count':seeded_images,'images_new_this_invocation':len(missing),'unique_texts_reused':text_reused,'timings':{'load_seconds':load_seconds,'media_seconds_this_invocation':media_seconds,**text_times},'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'versions':{k:importlib.metadata.version(k) for k in ['torch','transformers','numpy','sentence-transformers']},'normalization':'float64 L2 converted to float32 after encoder output','timing_scope':'Continuation/resume timing; no encoder latency comparison.'}
 save(out/'metadata.json',meta);print('COMPLETE',args.kind,json.dumps({k:v for k,v in meta.items() if k not in ['record_ids','media_order']}),flush=True)

if __name__=='__main__':main()
