import hashlib,json
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image
ROOT=Path(__file__).resolve().parent

def main():
 mf=json.loads((ROOT/'human_manifest.json').read_text());old=set(mf['previously_seen_media']);groups={};hashes={};excluded=0
 folder=ROOT/'images';folder.mkdir(exist_ok=True)
 for p in sorted((ROOT/'data').glob('flickr_expert_*.parquet')):
  assert hashlib.sha256(p.read_bytes()).hexdigest()==mf['source_sha256'][p.name]
  pf=pq.ParquetFile(p)
  for batch in pf.iter_batches(batch_size=128,columns=['id','imgid','img','cand','human_score','refs','source_row']):
   for row in batch.to_pylist():
    name=row['imgid']
    if name not in hashes:
     blob=row['img']['bytes'];sha=hashlib.sha256(blob).hexdigest();assert sha==mf['media_sha256'][name],name
     (folder/name).write_bytes(blob)
     with Image.open(folder/name) as im:im.verify()
     hashes[name]=sha
    if row['cand'] in row['refs']:excluded+=1;continue
    key=(name,row['cand'])
    if key not in groups:groups[key]={'id':row['id'],'filename':name,'caption':row['cand'],'ratings':[],'source_rows':[],'previously_seen':name in old}
    groups[key]['ratings'].append(row['human_score']);groups[key]['source_rows'].append(row['source_row'])
 assert list(groups.values())==mf['records'];assert excluded==mf['reference_candidate_ratings_excluded'];assert hashes==mf['media_sha256']
 out={'all_1000_image_sha256_match':True,'all_5664_caption_rating_records_match_pinned_source':True,'n_ratings':sum(len(r['ratings']) for r in mf['records']),'reference_ratings_excluded':excluded,'passed':True}
 (ROOT/'data_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)

if __name__=='__main__':main()
