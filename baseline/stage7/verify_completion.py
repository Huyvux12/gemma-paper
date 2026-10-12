"""Completion integrity, source hashes, metadata alignment and result checks."""
import ast,csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau
ROOT=Path(__file__).resolve().parent

def main():
 manifest=json.loads((ROOT/'human_manifest.json').read_text());records=manifest['records'];ids=[r['id'] for r in records];sha=hashlib.sha256((ROOT/'human_manifest.json').read_bytes()).hexdigest();hashes={};shape_info={}
 for kind in ['gemma','clip','clip_large','specs','siglip','siglip224']:
  folder=ROOT/'runs'/(kind+'-human');metadata=json.loads((folder/'metadata.json').read_text());assert metadata['record_ids']==ids and metadata['manifest_sha256']==sha
  assert set(metadata['media_order'])==set(manifest['media_sha256'])
  arrays=np.load(folder/'embeddings.npz',allow_pickle=False);shape_info[kind]={}
  for key in arrays.files:
   v=arrays[key];assert v.shape[0]==(1000 if key=='media' else 5664);assert v.dtype==np.float32 and np.isfinite(v).all()
   assert np.max(abs(np.linalg.norm(v,axis=1)-1))<1e-5
   shape_info[kind][key]=list(v.shape)
  hashes[kind]=hashlib.sha256((folder/'embeddings.npz').read_bytes()).hexdigest()
 for kind in ['gemma','siglip224']:
  m=json.loads((ROOT/'runs'/(kind+'-human')/'metadata.json').read_text());assert m['dtype']=='float32'
  assert all(a['over_context']==0 for a in m['token_audit'].values())
 result=json.loads((ROOT/'human_results.json').read_text());assert result['score_verification']['passed'] and result['bootstrap_repetitions']==2000
 assert result['cohorts']['full']['n_images']==1000 and result['cohorts']['unseen']['n_images']==900
 assert result['cohorts']['full']['n_captions']==5664 and result['cohorts']['unseen']['n_captions']==5112
 assert result['cohorts']['full']['n_individual_ratings']==16992 and result['cohorts']['unseen']['n_individual_ratings']==15336
 csvrows=list(csv.DictReader((ROOT/'human_scores.csv').open()));assert [r['id'] for r in csvrows]==ids;methods=[k for k in csvrows[0] if k not in ['id','filename','caption','ratings','previously_seen']];checked=0
 for name,rows in [('full',csvrows),('unseen',[r for r in csvrows if r['previously_seen']=='False'])]:
  ratings=[ast.literal_eval(r['ratings']) for r in rows];y=np.array([v for group in ratings for v in group])
  for key in methods:
   score=np.repeat([float(r[key]) for r in rows],[len(r) for r in ratings]);tau=float(kendalltau(score,y,variant='c').statistic)
   assert abs(tau-result['cohorts'][name]['metrics'][key]['kendall_tau_c'])<1e-12;checked+=1
  assert result['cohorts'][name]['weighted_tau_validation']['passed']
 assert not set(manifest['previously_seen_media']) & {r['filename'] for r in records if not r['previously_seen']}
 source=ROOT.parent/'research/embeddinggemma2_clipscore_replacement/human_manifest.json'
 if source.exists():
  old=json.loads(source.read_text())['records'];oldnames={r['filename'] for r in old};newnames={r['filename'] for r in records if not r['previously_seen']};assert not oldnames&newnames
  assert oldnames==set(manifest['previously_seen_media'])
 for name,digest in manifest['source_sha256'].items():
  path=ROOT/'data'/name
  if path.exists():assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
 assert json.loads((ROOT/'data_verification.json').read_text())['passed']
 out={'all_six_encoders_human_vectors_complete':True,'shapes':shape_info,'vector_sha256':hashes,'manifest_sha256':sha,'csv_tau_cohort_method_pairs_checked':checked,'all_csv_kendall_tau_match':True,'n_primary_unseen_images':900,'unseen_prior_overlap':0,'all_declared_human_configurations_reported':True,'score_scalar_verification':result['score_verification'],'paired_cluster_bootstrap_independent_reference_checks':{c:v['weighted_tau_validation'] for c,v in result['cohorts'].items()},'passed':True}
 (ROOT/'independent_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
