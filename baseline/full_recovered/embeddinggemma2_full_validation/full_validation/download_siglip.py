"""Pinned modern dual-encoder baseline, verifying weight SHA-256."""
import hashlib,json,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    meta=json.loads((ROOT/'siglip_hf_metadata.json').read_text());folder=ROOT/'models/siglip';folder.mkdir(parents=True,exist_ok=True);verified=[]
    for f in meta['siblings']:
        n=f['rfilename']
        if not n.endswith(('.json','.model','.safetensors')):continue
        p=folder/n
        for attempt in range(8):
            if p.exists() and p.stat().st_size==f['size']:break
            offset=p.stat().st_size if p.exists() else 0
            u=f'https://huggingface.co/google/siglip2-base-patch16-256/resolve/{meta["sha"]}/{n}?download=true&retry={attempt}'
            try:
                req=urllib.request.Request(u,headers={'Range':f'bytes={offset}-','Accept-Encoding':'identity'})
                with urllib.request.urlopen(req,timeout=90) as r:
                    append=r.status==206 and r.headers.get('Content-Range','').startswith(f'bytes {offset}-')
                    with p.open('ab' if append else 'wb') as out:
                        for b in iter(lambda:r.read(8*1024*1024),b''):out.write(b)
                print(n,p.stat().st_size,'/',f['size'],flush=True)
            except Exception:
                if attempt==7:raise
                time.sleep(min(2**attempt,8))
        assert p.stat().st_size==f['size']
        if f.get('lfs'):
            h=hashlib.sha256()
            with p.open('rb') as inp:
                for b in iter(lambda:inp.read(8*1024*1024),b''):h.update(b)
            assert h.hexdigest()==f['lfs']['sha256'];verified.append({'file':n,'bytes':f['size'],'sha256':h.hexdigest()})
    receipt={'model':'google/siglip2-base-patch16-256','revision':meta['sha'],'verified':True,'files':verified}
    (folder/'download_receipt.json').write_text(json.dumps(receipt,indent=2));(ROOT/'siglip_checkpoint.json').write_text(json.dumps(receipt,indent=2));print('VERIFIED',flush=True)

if __name__=='__main__':main()
