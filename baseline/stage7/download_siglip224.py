import json,urllib.request
from pathlib import Path
from download_required import get
ROOT=Path(__file__).resolve().parent
repo='google/siglip2-base-patch16-224';revision='75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2'
with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true',timeout=30) as r:metadata=json.load(r)
assert metadata['sha']==revision
for f in metadata['siblings']:
 name=f['rfilename']
 if name.endswith(('.json','.model','.safetensors')):
  if name=='model.safetensors':assert f['lfs']['sha256']=='612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b'
  get(f'https://huggingface.co/{repo}/resolve/{revision}/{name}',ROOT/'models/siglip224'/name,f.get('lfs',{}).get('sha256'))
