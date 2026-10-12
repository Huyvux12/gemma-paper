import hashlib,json,time,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent
source=json.loads((ROOT/'siglip2_source.json').read_text())
folder=ROOT/'models/siglip2';folder.mkdir(parents=True,exist_ok=True)
files=[f for f in source['siblings'] if f['rfilename']!='.gitattributes']
def download(f):
    name=f['rfilename'];path=folder/name
    if not path.exists() or path.stat().st_size!=f['size']:
        url='https://huggingface.co/'+source['id']+'/resolve/'+source['sha']+'/'+name
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url,timeout=90) as response,path.with_suffix(path.suffix+'.part').open('wb') as output:
                    while chunk:=response.read(4*1024*1024):output.write(chunk)
                path.with_suffix(path.suffix+'.part').rename(path);break
            except Exception:
                if attempt==2:raise
    assert path.stat().st_size==f['size']
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    expected=f.get('lfs',{}).get('sha256')
    if expected:assert digest==expected,(name,digest,expected)
    print(name,'verified',path.stat().st_size,flush=True)
    return {'file':name,'size':path.stat().st_size,'sha256':digest,'official_sha256':expected}
with ThreadPoolExecutor(max_workers=3) as ex: receipts=list(ex.map(download,files))
(ROOT/'download_receipt.json').write_text(json.dumps({'model':source['id'],'revision':source['sha'],'files':receipts},indent=2))
