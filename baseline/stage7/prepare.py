import hashlib,json,shutil,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'full_recovered/embeddinggemma2_full_validation/full_validation'
manifest=(SOURCE/'human_manifest.json').read_bytes();(ROOT/'human_manifest.json').write_bytes(manifest)
m=json.loads(manifest);assert len(m['records'])==5664 and len(m['media_sha256'])==1000
assert len({r['filename'] for r in m['records'] if not r['previously_seen']})==900
(ROOT/'source_protocol.json').write_bytes((SOURCE/'protocol.json').read_bytes())
protocol={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'status_at_lock':'No expanded Gemma caption embeddings available; no new human correlation scores observed.','dataset':'Flickr8k-Expert','manifest_sha256':hashlib.sha256(manifest).hexdigest(),'n_images':1000,'n_captions':5664,'unseen_images':900,'unseen_captions':5112,'source_protocol_inherited':'source_protocol.json','metric':'Kendall tau-c on individual ratings, repeated scores for each expert rating; image-cluster paired bootstrap2000, no caption-independent resampling.','scope':'Human evaluation is secondary; full and unseen900 cohorts separately. Full includes prior100 development media.','comparisons':['Gemma literal768 vs CLIP B32','Gemma literal768 vs CLIP L14','Gemma literal768 vs SPECS native','Gemma literal768 vs SigLIP2-256 native raw','Gemma literal768 vs SigLIP2-224 raw','Gemma literal768 vs SigLIP2-224 matched literal','Gemma FactChecking768 vs CLIP L14','Gemma literal256 vs literal768'],'confidence_intervals':'Individual95% paired cluster-bootstrap intervals; report exploratory human conclusions, no familywise superiority claim. For six literal768-versus-baseline contrasts also show descriptive Bonferroni family6 intervals at 1-0.05/6.','compression':'128/256/512/768, float32/f16/per-vector symmetric int8(+f32scale) descriptive; source tau tolerance0.01 only original256f32 comparison, no int8 equivalence claim.','prefix_selection':'Frozen literal prefix A photo depicts and existing FactChecking; no tuning to expert labels.','precision':'float32 encoders; CPU cache and CPU/GPU continuation separately disclosed. No encoder speed comparison unless matched fresh timings.','gemma_revision':'914f7f89142e33e77833254d9c9b90c3cef7303b','siglip224_revision':'75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2','image_cache_gate':'16 deterministic cached-image bridge checks, cosine>=0.9999 and max element error<=0.002; full re-encode if failed, with explicit flag.','statistical_seed_tau':202610084,'statistical_seed_within':202610085}
(ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
for kind in ['clip','clip_large','specs','siglip']:
 source=SOURCE/'runs'/(kind+'-human');dest=ROOT/'runs'/(kind+'-human');dest.mkdir(parents=True,exist_ok=True)
 for name in ['metadata.json','embeddings.npz']:shutil.copyfile(source/name,dest/name)
 metadata=json.loads((dest/'metadata.json').read_text());assert metadata['manifest_sha256']==protocol['manifest_sha256']
 assert metadata['record_ids']==[r['id'] for r in m['records']]
media=np.load(SOURCE/'runs/gemma-human/media.npy');completed=json.loads((SOURCE/'runs/gemma-human/image_state.json').read_text())['completed'];assert completed==936
cache=ROOT/'cache';cache.mkdir(exist_ok=True);np.savez_compressed(cache/'gemma_media_936.npz',names=np.array(sorted(m['media_sha256'])[:completed]),media=media[:completed])
old=ROOT.parent/'research/embeddinggemma2_clipscore_replacement';oldm=json.loads((old/'human_manifest.json').read_text());oldmeta=json.loads((old/'runs/gemma-human/metadata.json').read_text());oldz=np.load(old/'runs/gemma-human/embeddings.npz')
arrays={k:oldz[k] for k in ['media','CLIPLiteral_candidate','FactChecking_candidate']};np.savez_compressed(cache/'gemma_prior100.npz',**arrays)
(cache/'gemma_prior100_metadata.json').write_text(json.dumps({'record_ids':oldmeta['record_ids'],'media_order':oldmeta['media_order'],'captions':[r['caption'] for r in oldm['records']],'source_manifest_sha256':hashlib.sha256((old/'human_manifest.json').read_bytes()).hexdigest(),'revision':oldmeta['revision'],'origin':'Prior100 development image/text float32 vectors, exact image hashes verified against full manifest'},indent=2))
assert all(oldm['media_sha256'][n]==m['media_sha256'][n] for n in oldmeta['media_order'])
checkpoints=json.loads((old/'baseline_checkpoints.json').read_text());(ROOT/'gemma_checkpoint.json').write_text(json.dumps(checkpoints['gemma'],indent=2))
print(json.dumps({'human_manifest_sha256':protocol['manifest_sha256'],'baselines_complete':4,'gemma_cached_images':completed,'gemma_prior_captions':len(oldm['records']),'plan_frozen':True}))
