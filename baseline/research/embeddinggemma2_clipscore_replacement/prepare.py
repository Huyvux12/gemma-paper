"""Freeze a media-disjoint test BEFORE looking at any new model outputs."""
import concurrent.futures, hashlib, json, random, time, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def fetch(url,path):
    if path.exists(): return
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url,timeout=60) as r: data=r.read()
            path.write_bytes(data); return
        except Exception:
            if attempt==4: raise
            time.sleep(2**attempt)

def main():
    prior=json.loads((ROOT/'data/prior_image_manifest.json').read_text())['records']
    excluded={r['filename'] for r in prior}; used=set(excluded)
    rng=random.Random(20261008); records=[]
    for cat in ['add_att','add_obj','replace_att','replace_obj','replace_rel','swap_att','swap_obj']:
        data=json.loads((ROOT/'data'/f'{cat}.json').read_text()); candidates=list(data.items())
        rng.shuffle(candidates); selected=[]
        for ident,r in candidates:
            if r['filename'] in used: continue
            selected.append(dict(r,id=f'{cat}:{ident}',category=cat,split='test'))
            used.add(r['filename'])
            if len(selected)==50: break
        assert len(selected)==50
        records.extend(selected)
    protocol={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
      'seed':20261008,'test_pairs':350,'per_category':50,'selection':'fixed random sample; globally unique media; exclude ALL prior media',
      'primary_comparisons':['gemma-FactChecking-768 minus clip-B32','gemma-FactChecking-768 minus clip-L14'],
      'secondary':['literal CLIPScore encoder replacement with A photo depicts prefix', 'FactChecking 256 and 128 dimensionality'],
      'score':'2.5*max(cosine,0) for EVERY encoder; strict greater wins, ties fail',
      'statistics':'95% paired bootstrap CI by image, 10000 resamples; exact paired McNemar binomial; Holm over two primary comparisons',
      'prefix_selection':'FactChecking chosen from PRIOR runs; no new-test tuning',
      'no_finetuning':True,'dtype':'float32','image_tokens_gemma':280,
      'scope':'caption pair ranking on held-out media within SugarCrepe, not free-form hallucination accuracy or human correlation'}
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    manifest={'dataset':'SugarCrepe','records':records,'excluded_prior_images':sorted(excluded),'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').glob('*.json')}}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Frozen',len(records),'pairs; no prior media overlap',flush=True)
    def download(r):
        p=ROOT/'images'/r['filename']
        fetch('https://s3.us-east-1.amazonaws.com/images.cocodataset.org/val2017/'+r['filename'],p)
        from PIL import Image
        with Image.open(p) as im: im.verify()
        return r['filename'],hashlib.sha256(p.read_bytes()).hexdigest()
    hashes={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for i,(name,digest) in enumerate(pool.map(download,records),1):
            hashes[name]=digest
            if i%25==0:print('media downloaded',i,flush=True)
    manifest['media_sha256']=hashes
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
