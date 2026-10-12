"""Recover completed vector evidence without pretending to perform new inference."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;SOURCE=ROOT.parent/'full_recovered/embeddinggemma2_full_validation';FULL=SOURCE/'full_validation'
manifest=json.loads((FULL/'package_manifest.json').read_text());bad=[]
for f in manifest['files']:
    p=SOURCE/f['path']
    if not p.exists() or p.stat().st_size!=f['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=f['sha256']:bad.append(f['path'])
assert not bad,bad
frozen=json.loads((ROOT/'frozen_manifest.json').read_text());r=frozen['records'];full=json.loads((FULL/'manifest.json').read_text());original=full['records'];index={row['id']:i for i,row in enumerate(original)}
assert len(index)==len(original)
selected=[]
for row in r:
    i=index[row['id']];old=original[i]
    assert all(old[k]==row[k] for k in ['filename','caption','negative_caption','category'])
    assert not old['previously_seen'];selected.append(i)
names=sorted({row['filename'] for row in r});outcomes={}
for kind in ['gemma','clip','clip_large','siglip','specs']:
    folder=FULL/'runs'/kind;m=json.loads((folder/'metadata.json').read_text());assert m['record_ids']==[x['id'] for x in original]
    z=np.load(folder/'embeddings.npz');idx={n:i for i,n in enumerate(m['media_order'])}
    arrays={}
    for key in z.files:
        assert np.isfinite(z[key]).all()
        if key=='media':arrays[key]=z[key][[idx[n] for n in names]]
        else:
            assert z[key].shape[0]==len(original);arrays[key]=z[key][selected]
    target=ROOT/'runs'/kind;target.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target/'embeddings.npz',**arrays)
    receipt={**m,'media_order':names,'record_ids':[row['id'] for row in r],'origin':'Recovered from completed full SugarCrepe inference checkpoint; no new encoder inference in this turn.','source_archive_version':31,'source_embeddings_sha256':hashlib.sha256((folder/'embeddings.npz').read_bytes()).hexdigest(),'source_manifest_sha256':hashlib.sha256((FULL/'manifest.json').read_bytes()).hexdigest(),'source_record_indices':selected,'source_runtime_is_not_560_runtime':True}
    (target/'metadata.json').write_text(json.dumps(receipt,indent=2));outcomes[kind]={'shape':{k:list(v.shape) for k,v in arrays.items()},'model':m.get('model'),'revision':m.get('revision')}
result={'record_texts_identical':True,'all_560_covered':True,'all_560_unseen_in_original_cohort':True,'package_hash_files_verified':len(manifest['files']),'frozen_560_sha256':hashlib.sha256((ROOT/'frozen_manifest.json').read_bytes()).hexdigest(),'original_package_complete':manifest['status']['complete'],'sugar_vectors_complete':True,'gemma_human_status':'incomplete, 936/1000 images at source checkpoint, no final caption vectors','missing_requested_model':'SigLIP2 Base patch16-224; available saved model is patch16-256','models':outcomes,'passed':True}
(ROOT/'recovery_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
