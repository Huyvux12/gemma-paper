"""Official PAC-S checkpoint and implementation; never substitute PAC-S++."""
import argparse,importlib.util,sys,urllib.request,zipfile
from pathlib import Path
import numpy as np
import torch
from common import read,write,digest
from metrics import evaluate_human

REV='bfbc488e9f5cccd818e75c7cdfaa37e1c87f00eb'
CHECKPOINT_ID='1F-0Pma-vfJPAiDzeyl-iEdSXZIO1cDae'

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    assert torch.cuda.is_available()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);b=Path(a.baseline)
    sys.path.insert(0,str(b/'stage7'))
    from download_required import data
    from materialize_expert import main as materialize
    data();materialize()
    archive=out/'official_source.zip'
    urllib.request.urlretrieve(f'https://github.com/aimagelab/pacscore/archive/{REV}.zip',archive)
    with zipfile.ZipFile(archive) as z:
        assert all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in z.namelist())
        z.extractall(out/'source')
    root=out/'source'/('pacscore-'+REV);sys.path.insert(0,str(root))
    from models import clip
    source=root/'evaluation/pac_score/pac_score.py'
    spec=importlib.util.spec_from_file_location('official_pac',source);pac=importlib.util.module_from_spec(spec);spec.loader.exec_module(pac)
    import gdown
    weights=out/'clip_ViT-B-32.pth'
    assert gdown.download(id=CHECKPOINT_ID,output=str(weights),quiet=False) and weights.exists()
    model,preprocess=clip.load('ViT-B/32',device='cuda');model.float()
    checkpoint=torch.load(weights,map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    rows=read(b/'stage7/human_manifest.json')['records'];names=sorted({r['filename'] for r in rows})
    features=pac.extract_all_images([str(b/'stage7/images'/n) for n in names],model,preprocess,'cuda',batch_size=64,num_workers=2)
    mapping={n:i for i,n in enumerate(names)}
    repeated=features[[mapping[r['filename']] for r in rows]]
    _,scores,_,_=pac.PACScore(model,preprocess,repeated,[r['caption'] for r in rows],'cuda',w=2.0)
    assert len(scores)==len(rows) and np.isfinite(scores).all()
    write(out/'pac_scores.json',{'record_ids':[r['id'] for r in rows],'scores':{'PAC-S-ViTB32':scores.tolist()}})
    mask=np.array([not r['previously_seen'] for r in rows])
    write(out/'pac_metrics.json',{'full':evaluate_human(rows,scores),'unseen900':evaluate_human([r for r in rows if not r['previously_seen']],scores[mask]),
        'receipt':{'source_revision':REV,'source_archive_sha256':digest(archive),'implementation_sha256':digest(source),'checkpoint_google_drive_id':CHECKPOINT_ID,'checkpoint_sha256':digest(weights),'checkpoint_hash_preverified_by_author':False,'weight':2.0,'precision':'float32'},
        'scope':'Historical expert benchmark. Official trained PAC-S; author training-data overlap with Flickr8k not independently audited.'})
    print('Official PAC-S complete',flush=True)

if __name__=='__main__':main()
