"""Save resumable checkpoints or verified final evidence; weights excluded."""
import argparse,hashlib,json,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'replacement'

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',action='store_true');args=p.parse_args()
    if not args.checkpoint:assert json.loads((ROOT/'independent_verification.json').read_text())['passed']
    files=[]
    for f in sorted(ROOT.rglob('*')):
        if not f.is_file():continue
        rel=f.relative_to(ROOT)
        if rel.parts[0] in ['models','images','human_images'] or any(x.startswith('.') or x=='__pycache__' for x in rel.parts):continue
        if f.suffix in ['.zip','.parquet','.part'] or f.name in ['package_manifest.json','checkpoint_status.json']:continue
        if rel.parts[0]=='runs' and f.suffix=='.npy' and (f.parent/'metadata.json').exists():continue
        if rel.parts[0]=='runs' and f.name=='embeddings.npz' and not (f.parent/'metadata.json').exists():continue
        if rel.parts[0]=='sources' and not (len(rel.parts)>2 and rel.parts[1]=='SPECS' and (rel.parts[2]=='model' or rel.name in ['LICENSE','README.md'])):continue
        files.append(('full_validation/'+rel.as_posix(),f))
    dependencies=['score_adapter.py','gemma_clipscore.py','download_models.py','requirements.txt','manifest.json','human_manifest.json','source_revisions.json','environment_freeze.txt',
      'data/prior_image_manifest.json','provenance/expert_author_annotations.json','provenance/clipscore_LICENSE','provenance/sugarcrepe_LICENSE']
    dependencies.extend('data/'+c+'.json' for c in ['add_att','add_obj','replace_att','replace_obj','replace_rel','swap_att','swap_obj'])
    for rel in dependencies:files.append(('replacement/'+rel,PREV/rel))
    # Read progress markers before raw partial arrays: every marked row is immutable.
    files.sort(key=lambda item:(not item[1].name.endswith('_state.json'),item[0]))
    manifest=[];snapshots={};archive=ROOT.parent/'embeddinggemma2_full_validation.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,f in files:
            data=f.read_bytes();z.writestr('embeddinggemma2_full_validation/'+name,data)
            manifest.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
            if f.name.endswith('_state.json'):snapshots[name]=json.loads(data)
        status={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'complete':not args.checkpoint,
          'weights_excluded':True,'full_media_excluded':True,'resumable_progress':snapshots,
          'warning':'Incomplete checkpoint; do not report unfinished methods as results.' if args.checkpoint else 'Final measured and independently verified evidence.'}
        z.writestr('embeddinggemma2_full_validation/full_validation/package_manifest.json',json.dumps({'files':manifest,'status':status},indent=2))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(archive,archive.stat().st_size,'bytes; checkpoint',args.checkpoint,flush=True)

if __name__=='__main__':main()
