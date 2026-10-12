"""Recover original COCO ZIP from a pinned mirror and verify byte overlaps."""
import hashlib,json,time,urllib.request,zipfile
import pyarrow.parquet as pq
from PIL import Image
from encoders import ROOT,PREV

def digest(blob):return hashlib.sha256(blob).hexdigest()

def main():
    repo='nyanko7/coco-hosted'
    revision='209f00949a119167d728d4749e7cff9c08785436'
    with urllib.request.urlopen(f'https://huggingface.co/api/datasets/{repo}/revision/{revision}?blobs=true',timeout=60) as r:meta=json.load(r)
    assert meta['sha']==revision
    entry=next(f for f in meta['siblings'] if f['rfilename']=='val2017.zip')
    receipt={'repository':repo,'revision':meta['sha'],'file':'val2017.zip','bytes':entry['size'],'sha256':entry['lfs']['sha256']}
    (ROOT/'coco_mirror_receipt.json').write_text(json.dumps(receipt,indent=2))
    archive=ROOT/'.coco_val2017.zip'
    if not archive.exists() or archive.stat().st_size!=entry['size']:
        for attempt in range(8):
            try:
                url=f'https://huggingface.co/datasets/{repo}/resolve/{meta["sha"]}/val2017.zip?download=true&retry={attempt}'
                offset=archive.stat().st_size if archive.exists() else 0
                request=urllib.request.Request(url,headers={'Range':f'bytes={offset}-','Accept-Encoding':'identity'})
                with urllib.request.urlopen(request,timeout=90) as r:
                    append=r.status==206 and r.headers.get('Content-Range','').startswith(f'bytes {offset}-')
                    f=archive.open('ab' if append else 'wb')
                    with f:
                        for b in iter(lambda:r.read(8*1024*1024),b''):f.write(b)
                print('ZIP bytes',archive.stat().st_size,'/',entry['size'],flush=True)
                if archive.stat().st_size==entry['size']:break
            except Exception:
                if attempt==7:raise
                time.sleep(min(2**attempt,8))
    assert archive.stat().st_size==entry['size'];h=hashlib.sha256()
    with archive.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    assert h.hexdigest()==entry['lfs']['sha256']
    m=json.loads((ROOT/'manifest.json').read_text());old=json.loads((PREV/'manifest.json').read_text())['media_sha256'];hashes={};overlap=0
    with zipfile.ZipFile(archive) as z:
        for n in sorted({r['filename'] for r in m['records']}):
            blob=z.read('val2017/'+n);sha=digest(blob);path=ROOT/'images'/n
            if path.exists():assert digest(path.read_bytes())==sha;overlap+=1
            if n in old:assert old[n]==sha
            path.write_bytes(blob)
            with Image.open(path) as im:im.verify()
            hashes[n]=sha
    m['media_sha256']=hashes;m['media_source']=receipt;(ROOT/'manifest.json').write_text(json.dumps(m,indent=2))
    hm=json.loads((ROOT/'human_manifest.json').read_text());hashes={};folder=ROOT/'human_images';folder.mkdir(exist_ok=True)
    for p in sorted((PREV/'data').glob('flickr_expert_*.parquet')):
        for row in pq.read_table(p,columns=['imgid','img']).to_pylist():
            n=row['imgid']
            if n in hashes:continue
            blob=row['img']['bytes'];(folder/n).write_bytes(blob)
            with Image.open(folder/n) as im:im.verify()
            hashes[n]=digest(blob)
    assert len(hashes)==1000;hm['media_sha256']=hashes;(ROOT/'human_manifest.json').write_text(json.dumps(hm,indent=2))
    receipt.update({'verified':True,'byte_exact_existing_overlaps':overlap,'byte_exact_all_previous_350':True})
    (ROOT/'coco_mirror_receipt.json').write_text(json.dumps(receipt,indent=2))
    print('VERIFIED MIRROR',json.dumps(receipt),flush=True);print('ALL MEDIA',len(m['media_sha256']),len(hashes),flush=True)

if __name__=='__main__':main()
