"""Freeze secondary human-correlation sample independently of model scores."""
import hashlib,io,json,random,time
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image
ROOT=Path(__file__).resolve().parent

def main():
    source=[ROOT/'data/flickr_expert_0.parquet',ROOT/'data/flickr_expert_1.parquet']
    # Read metadata first; decode only the selected images.
    columns=['id','imgid','cand','human_score','refs','source_row']
    allrows=[r for p in source for r in pq.read_table(p,columns=columns).to_pylist()]
    names=sorted({r['imgid'] for r in allrows});selected=set(random.Random(20261008).sample(names,100))
    groups={}
    excluded=0
    for r in allrows:
        if r['imgid'] not in selected:continue
        if r['cand'] in r['refs']:excluded+=1;continue
        key=(r['imgid'],r['cand'])
        if key not in groups:groups[key]={'id':r['id'],'filename':r['imgid'],'caption':r['cand'],'ratings':[],'source_rows':[]}
        groups[key]['ratings'].append(r['human_score']);groups[key]['source_rows'].append(r['source_row'])
    records=list(groups.values());assert len({r['filename'] for r in records})==100
    manifest={'dataset':'Flickr8k-Expert','seed':20261008,'records':records,'reference_candidates_excluded':excluded,
      'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
      'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'selection':'random 100 images, ALL non-reference candidates; no score-based selection',
      'scoring':'repeat candidate score for original individual expert ratings, Kendall tau-c following CLIPScore evaluator; paired cluster bootstrap 2000',
      'scope':'secondary exploratory human-correlation benchmark, no prefix tuning on this sample'}
    (ROOT/'human_manifest.json').write_text(json.dumps(manifest,indent=2))
    folder=ROOT/'human_images';folder.mkdir(exist_ok=True);hashes={}
    for p in source:
        t=pq.read_table(p,columns=['imgid','img'])
        for r in t.to_pylist():
            n=r['imgid']
            if n not in selected or n in hashes:continue
            blob=r['img']['bytes'];path=folder/n;path.write_bytes(blob)
            with Image.open(path) as im:im.verify()
            hashes[n]=hashlib.sha256(blob).hexdigest()
    assert len(hashes)==100;manifest['media_sha256']=hashes
    (ROOT/'human_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('FROZEN',len(records),'caption pairs;',sum(len(r['ratings']) for r in records),'expert ratings;',len(hashes),'images',flush=True)

if __name__=='__main__':main()
