import concurrent.futures,hashlib,json,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for data in iter(lambda:f.read(8*1024*1024),b''):h.update(data)
 return h.hexdigest()

def get(url,path,expected=None):
 path.parent.mkdir(parents=True,exist_ok=True)
 if not path.exists():
  for attempt in range(3):
   try:
    with urllib.request.urlopen(url,timeout=45) as r,path.with_suffix(path.suffix+'.part').open('wb') as f:
     for block in iter(lambda:r.read(8*1024*1024),b''):f.write(block)
    path.with_suffix(path.suffix+'.part').replace(path);break
   except Exception:
    if attempt==2:raise
 if expected:assert sha(path)==expected,str(path)
 print('Verified',path.relative_to(ROOT),path.stat().st_size,flush=True)

def data():
 manifest=json.loads((ROOT/'human_manifest.json').read_text());revision='c30e1e67e2b37a42ecffaf9a88a05458ca05b973'
 for i in [0,1]:
  get(f'https://huggingface.co/datasets/yuwd/Flickr8k-HumanEval/resolve/{revision}/expert/test-{i:05d}-of-00002.parquet',ROOT/'data'/f'flickr_expert_{i}.parquet',manifest['source_sha256'][f'flickr_expert_{i}.parquet'])

def gemma():
 receipt=json.loads((ROOT/'gemma_checkpoint.json').read_text());repo=receipt['model'];rev=receipt['revision']
 with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true',timeout=30) as r:metadata=json.load(r)
 assert metadata['sha']==rev
 known={x['filename']:x['sha256'] for x in receipt['files']}
 for f in metadata['siblings']:
  n=f['rfilename']
  if n=='model.safetensors' or n.endswith(('.json','.jinja','.model')) or n=='merges.txt':
   if n in known and f.get('lfs'):assert known[n]==f['lfs']['sha256']
   get(f'https://huggingface.co/{repo}/resolve/{rev}/{n}',ROOT/'models/gemma'/n,known.get(n,f.get('lfs',{}).get('sha256')))
 (ROOT/'models/gemma/download_receipt.json').write_text(json.dumps(receipt,indent=2))

if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for future in [pool.submit(data),pool.submit(gemma)]:future.result()
