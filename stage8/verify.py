"""Checks mathematical behavior and data separation, not fitted test outcomes."""
import argparse,ast,tempfile,json,zipfile,shutil
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import ndcg_score
from train_adapter import GemmaAdapter,save_model,load_model,make_split
from metrics import tied_dcg,evaluate_human
from common import read,FULL_REL

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);a=p.parse_args()
    torch.manual_seed(1);m=GemmaAdapter(rank=16);x=torch.nn.functional.normalize(torch.randn(12,768),dim=-1);y=torch.nn.functional.normalize(torch.randn(12,768),dim=-1)
    assert torch.allclose(m.image(x),x,atol=1e-6)
    assert torch.allclose(m.scores(x,y),(x*y).sum(-1),atol=1e-6)
    loss=m.scores(x,y).sum();loss.backward();assert m.image.up.weight.grad.abs().sum()>0
    with torch.no_grad():m.image.up.weight.add_(.01*torch.randn_like(m.image.up.weight))
    with tempfile.TemporaryDirectory() as t:
        f=Path(t)/'head.npz';save_model(f,m);loaded=load_model(f,16)
        assert torch.equal(m.scores(x,y),loaded.scores(x,y))
    gains=np.array([7.,0.,1.,3.,2.,4.]);scores=np.array([.3,.3,.3,.1,.2,.2])
    expected=ndcg_score(gains[None],scores[None],k=5,ignore_ties=False)
    actual=tied_dcg(gains,scores)/tied_dcg(gains,gains)
    assert abs(expected-actual)<1e-12
    rows=[{'filename':'a','ratings':[4,4,4]},{'filename':'a','ratings':[1,1,1]}]
    v=evaluate_human(rows,np.array([0.,0.]));assert v['top1_human_best_tie_averaged']==.5 and v['within_image_micro_pair_accuracy']==0
    b=Path(a.baseline);s=make_split(read(b/FULL_REL/'manifest.json'),read(b/'stage7/human_manifest.json'))
    assert len(s['train_images'])==444 and len(s['validation_images'])==111
    notebook=Path(__file__).parent/'kaggle_kernel/run_stage8.ipynb'
    if notebook.exists():
        code=''.join(json.loads(notebook.read_text())['cells'][1]['source']);tree=ast.parse(code)
        selection=[n for n in tree.body if (isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='archives') or (isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='archives')]
        assert len(selection)==2
        for archived in [True,False]:
            with tempfile.TemporaryDirectory() as t:
                base=Path(t);inputs=base/'input';inputs.mkdir();work=base/'working';work.mkdir()
                if archived:
                    with zipfile.ZipFile(inputs/'baseline.zip','w') as z:z.writestr('stage7/human_manifest.json','{}')
                else:
                    folder=inputs/'extracted/stage7';folder.mkdir(parents=True);(folder/'human_manifest.json').write_text('{}')
                fixture=ast.parse(ast.unparse(ast.Module(body=selection,type_ignores=[])).replace("'/kaggle/input'",repr(str(inputs))))
                exec(compile(fixture,'notebook_fixture','exec'),{'Path':Path,'work':work,'zipfile':zipfile,'shutil':shutil})
                assert (work/'baseline/stage7/human_manifest.json').read_text()=='{}'
    for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
    print('PASS: identity, learned gradients, safe checkpoint roundtrip, tie-aware NDCG, ranking ties, image/hash separation, notebook ZIP/extracted input layouts, source syntax')

if __name__=='__main__':main()
