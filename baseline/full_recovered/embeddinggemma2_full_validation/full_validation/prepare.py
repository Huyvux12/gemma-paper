"""Freeze full benchmarks and genuinely unseen media before new inference."""
import concurrent.futures,hashlib,json,shutil,time,urllib.request
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image

ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'replacement'
CATS=['add_att','add_obj','replace_att','replace_obj','replace_rel','swap_att','swap_obj']

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ROOT.mkdir(exist_ok=True)
    excluded={r['filename'] for r in json.loads((PREV/'data/prior_image_manifest.json').read_text())['records']}
    excluded.update(r['filename'] for r in json.loads((PREV/'manifest.json').read_text())['records'])
    records=[dict(row,id=f'{cat}:{key}',category=cat,previously_seen=row['filename'] in excluded)
      for cat in CATS for key,row in json.loads((PREV/'data'/f'{cat}.json').read_text()).items()]
    manifest={'dataset':'SugarCrepe','records':records,'previously_seen_media':sorted(excluded),
      'source_sha256':{f'{c}.json':digest(PREV/'data'/f'{c}.json') for c in CATS}}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    old_h={r['filename'] for r in json.loads((PREV/'human_manifest.json').read_text())['records']}
    parquets=sorted((PREV/'data').glob('flickr_expert_*.parquet'));groups={};excluded_refs=0
    for p in parquets:
        for row in pq.read_table(p,columns=['id','imgid','cand','human_score','refs','source_row']).to_pylist():
            if row['cand'] in row['refs']:excluded_refs+=1;continue
            k=(row['imgid'],row['cand'])
            if k not in groups:groups[k]={'id':row['id'],'filename':row['imgid'],'caption':row['cand'],
              'ratings':[],'source_rows':[],'previously_seen':row['imgid'] in old_h}
            groups[k]['ratings'].append(row['human_score']);groups[k]['source_rows'].append(row['source_row'])
    hm={'dataset':'Flickr8k-Expert','records':list(groups.values()),'previously_seen_media':sorted(old_h),
      'reference_candidate_ratings_excluded':excluded_refs,'source_sha256':{p.name:digest(p) for p in parquets}}
    (ROOT/'human_manifest.json').write_text(json.dumps(hm,indent=2))
    protocol={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
      'full_sugar_pairs':len(records),'full_sugar_images':len({r['filename'] for r in records}),
      'novel_sugar_pairs':sum(not r['previously_seen'] for r in records),
      'novel_sugar_images':len({r['filename'] for r in records if not r['previously_seen']}),
      'full_human_images':len({r['filename'] for r in groups.values()}),'full_human_captions':len(groups),
      'novel_human_images':len({r['filename'] for r in groups.values() if not r['previously_seen']}),
      'primary':['Gemma CLIPLiteral 768 vs CLIP B32 on unseen SugarCrepe images',
                 'Gemma CLIPLiteral 768 vs CLIP L14 on unseen SugarCrepe images'],
      'prefixes_gemma':['CLIPLiteral','FactChecking'], 'dimensions_gemma':[128,256,512,768],
      'prefix_selection':'Fixed from preceding 350-image study; no retuning on new media',
      'statistics':'Image-cluster paired bootstrap 10000; image-level sign permutation 19999; Holm over 2 primary comparisons; macro accuracy descriptive',
      'human':'All 1000 images, all non-reference captions and individual expert ratings; Kendall tau-c, image-cluster bootstrap 2000; unseen 900 images separately; secondary',
      'compression':'256d vs 768d literal prefix; predeclared noninferiority margin 2 percentage points; one-sided 95% lower cluster-bootstrap bound on unseen SugarCrepe; exploratory human tau tolerance 0.01',
      'precision':'Uniform bfloat16 only if pilot validation passes for all 3 encoders; otherwise float32. No reused float32 embeddings in bfloat16 full runs.',
      'precision_gate':{'sample':'32 fixed random prior SugarCrepe images, their 64 captions; 32 prior expert captions',
        'max_abs_score_error':0.02,'minimum_ranking_agreement':0.95,'minimum_embedding_cosine':0.99},
      'additional_baselines':'Attempt official SPECS or PAC-S trained metric; if inaccessible record failure, do not substitute an untrained checkpoint as that metric',
      'no_training':True,'gemma_image_tokens':280,'score':'2.5*max(cosine,0) for encoder substitutions; native trained-metric scoring if available',
      'scope':'Full results include old development media and are descriptive; only media-disjoint continuation cohort supports new confirmation; no pretraining contamination guarantee'}
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print('FROZEN',json.dumps({k:v for k,v in protocol.items() if isinstance(v,int)}),flush=True)
    folder=ROOT/'images';folder.mkdir(exist_ok=True)
    def download(n):
        p=folder/n
        if not p.exists():
            old=PREV/'images'/n
            if old.exists():shutil.copyfile(old,p)
            else:
                for attempt in range(5):
                    try:
                        with urllib.request.urlopen('https://s3.us-east-1.amazonaws.com/images.cocodataset.org/val2017/'+n,timeout=60) as r:blob=r.read()
                        p.write_bytes(blob);break
                    except Exception:
                        if attempt==4:raise
                        time.sleep(min(2**attempt,8))
        with Image.open(p) as im:im.verify()
        return n,digest(p)
    hashes={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        for i,(n,h) in enumerate(pool.map(download,sorted({r['filename'] for r in records})),1):
            hashes[n]=h
            if i%100==0:print('Sugar media',i,flush=True)
    manifest['media_sha256']=hashes;(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    folder=ROOT/'human_images';folder.mkdir(exist_ok=True);hashes={}
    for p in parquets:
        for row in pq.read_table(p,columns=['imgid','img']).to_pylist():
            n=row['imgid']
            if n in hashes:continue
            blob=row['img']['bytes'];(folder/n).write_bytes(blob)
            with Image.open(folder/n) as im:im.verify()
            hashes[n]=hashlib.sha256(blob).hexdigest()
    assert len(hashes)==1000
    hm['media_sha256']=hashes;(ROOT/'human_manifest.json').write_text(json.dumps(hm,indent=2))
    print('ALL MEDIA VERIFIED',len(manifest['media_sha256']),len(hashes),flush=True)

if __name__=='__main__':main()
