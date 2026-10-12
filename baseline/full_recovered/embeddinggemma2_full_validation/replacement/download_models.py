"""Download official CLIP and LAION CLAP checkpoints and verify LFS digests."""
import concurrent.futures
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent
MODELS={'gemma':'google/embeddinggemma-2','clip':'openai/clip-vit-base-patch32','clip_large':'openai/clip-vit-large-patch14'}
REVISIONS={'clip':'3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268',
           'clip_large':'32bd64288804d66eefd0ccbe215aa642df71cc41',
           'clap':'8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a',
           'gemma':'914f7f89142e33e77833254d9c9b90c3cef7303b'}


def download(item):
    alias,repo=item
    with urllib.request.urlopen('https://huggingface.co/api/models/'+repo+('/revision/'+REVISIONS[alias] if alias in REVISIONS else '')+'?blobs=true',timeout=60) as r:info=json.load(r)
    assert alias not in REVISIONS or info['sha']==REVISIONS[alias]
    target=ROOT/'models'/alias;target.mkdir(parents=True,exist_ok=True)
    files=info['siblings'];names={f['rfilename'] for f in files}
    weight='model.safetensors' if 'model.safetensors' in names else 'pytorch_model.bin'
    selected=[f for f in files if f['rfilename']==weight or f['rfilename'].endswith(('.json','.jinja')) or f['rfilename'] in ('merges.txt','tokenizer.model')]
    verified=[]
    for entry in sorted(selected,key=lambda f:f['rfilename']==weight):
        name=entry['rfilename'];p=target/name;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():
            print(alias,'download',name,flush=True)
            url='https://huggingface.co/'+repo+'/resolve/'+info['sha']+'/'+name
            with urllib.request.urlopen(url,timeout=90) as r,p.with_suffix(p.suffix+'.part').open('wb') as out:
                for block in iter(lambda:r.read(8*1024*1024),b''):out.write(block)
            p.with_suffix(p.suffix+'.part').replace(p)
        assert p.stat().st_size==entry['size'],(name,p.stat().st_size,entry['size'])
        if entry.get('lfs'):
            h=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
            assert h.hexdigest()==entry['lfs']['sha256']
            verified.append({'filename':name,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
    result={'model':repo,'revision':info['sha'],'files':verified,'verified':True}
    (target/'download_receipt.json').write_text(json.dumps(result,indent=2));print(alias,'verified',flush=True)
    return alias,result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--include-gemma',action='store_true');args=parser.parse_args()
    if args.include_gemma:pass
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:receipts=dict(pool.map(download,MODELS.items()))
    (ROOT/'baseline_checkpoints.json').write_text(json.dumps(receipts,indent=2))
