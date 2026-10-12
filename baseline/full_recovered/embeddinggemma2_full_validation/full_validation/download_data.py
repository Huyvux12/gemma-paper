"""Restore the pinned Expert source shards; verify the manifest digests."""
import hashlib,json,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent;PREV=ROOT.parent/'replacement'
REVISION='c30e1e67e2b37a42ecffaf9a88a05458ca05b973'

def main():
    expected=json.loads((ROOT/'human_manifest.json').read_text())['source_sha256'];folder=PREV/'data';folder.mkdir(exist_ok=True)
    for i in [0,1]:
        p=folder/f'flickr_expert_{i}.parquet';u=f'https://huggingface.co/datasets/yuwd/Flickr8k-HumanEval/resolve/{REVISION}/expert/test-{i:05d}-of-00002.parquet'
        if not p.exists():
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(u,timeout=90) as r,p.with_suffix('.part').open('wb') as out:
                        for b in iter(lambda:r.read(8*1024*1024),b''):out.write(b)
                    p.with_suffix('.part').replace(p);break
                except Exception:
                    if attempt==4:raise
                    time.sleep(min(2**attempt,8))
        assert hashlib.sha256(p.read_bytes()).hexdigest()==expected[p.name];print('verified',p.name,flush=True)

if __name__=='__main__':main()
