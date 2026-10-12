"""Materialize exactly the stage-4 frozen confirmation set; never replace failures."""
import concurrent.futures,datetime,hashlib,json,urllib.request
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'research/embeddinggemma2_clipscore_replacement'
frozen=ROOT.parent/'stage4/next_560_manifest.json';protocol=json.loads((ROOT.parent/'stage4/next_protocol.json').read_text())
assert hashlib.sha256(frozen.read_bytes()).hexdigest()==protocol['manifest_sha256']
manifest=json.loads(frozen.read_text());records=manifest['records']
prior={r['filename'] for name in ['manifest.json','data/prior_image_manifest.json'] for r in json.loads((OLD/name).read_text())['records']}
assert len(records)==560==len({r['filename'] for r in records})
assert not prior&{r['filename'] for r in records}
assert all(sum(r['category']==c for r in records)==80 for c in {r['category'] for r in records})
(ROOT/'frozen_manifest.json').write_bytes(frozen.read_bytes());(ROOT/'frozen_protocol.json').write_bytes((ROOT.parent/'stage4/next_protocol.json').read_bytes())
addendum={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'primary':'Gemma CLIPLiteral 768 float32 vs SigLIP2 raw','secondary_holm_family':['Gemma CLIPLiteral768 vs SigLIP2 CLIPLiteral','Gemma CLIPLiteral768 vs CLIP B/32','Gemma CLIPLiteral768 vs CLIP L/14'],'gemma_extra_prompts':'FactChecking, None, SearchQuery, SentenceSimilarity exploratory ablations, no selection on confirmation results','latency':'Full sequential runs on one machine/session, CPU6 threads batch4; wall time includes preprocessing and saving batches. Not optimized deployment latency.','compression':'128/256/512/768; float32/float16/per-vector int8; descriptive tradeoffs, no noninferiority claim','bridge':'Before new inference, compare newly encoded old samples with archived vectors to audit reconstruction.'}
(ROOT/'execution_addendum.json').write_text(json.dumps(addendum,indent=2))

def download(r):
    name=r['filename'];path=ROOT/'images'/name
    if not path.exists():
        url='https://s3.us-east-1.amazonaws.com/images.cocodataset.org/val2017/'+name
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url,timeout=60) as response:data=response.read()
                path.with_suffix('.part').write_bytes(data);path.with_suffix('.part').replace(path);break
            except Exception:
                if attempt==3:raise
    with Image.open(path) as im:im.verify()
    return name,hashlib.sha256(path.read_bytes()).hexdigest()

hashes={}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    for i,(name,digest) in enumerate(pool.map(download,records),1):
        hashes[name]=digest
        if i%40==0:print('images downloaded',i,'/560',flush=True)
(ROOT/'manifest.json').write_text(json.dumps({**manifest,'not_run':False,'status':'Images downloaded; inference pending','media_sha256':hashes},indent=2))
print('560 images ready, disjoint from all prior experiment images',flush=True)
