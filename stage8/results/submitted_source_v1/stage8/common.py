import hashlib,json
from pathlib import Path
import numpy as np

FULL_REL='full_recovered/embeddinggemma2_full_validation/full_validation'

def read(p): return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def unit(x):
    x=np.asarray(x,dtype=np.float64)
    norms=np.linalg.norm(x,axis=1,keepdims=True)
    assert np.isfinite(x).all() and np.all(norms>0)
    return x/norms
def load_gemma(baseline,human=False):
    b=Path(baseline)
    root=b/'stage7' if human else b/FULL_REL
    mf=read(root/('human_manifest.json' if human else 'manifest.json'))
    folder=root/'runs'/('gemma-human' if human else 'gemma');meta=read(folder/'metadata.json')
    assert meta['record_ids']==[r['id'] for r in mf['records']]
    with np.load(folder/'embeddings.npz',allow_pickle=False) as z:
        ix={n:i for i,n in enumerate(meta['media_order'])}
        image=z['media'][[ix[r['filename']] for r in mf['records']]]
        labels=['candidate'] if human else ['positive','negative']
        arrays=[image]+[z['CLIPLiteral_'+label] for label in labels]
    return mf,[np.asarray(unit(a),dtype=np.float32) for a in arrays]
