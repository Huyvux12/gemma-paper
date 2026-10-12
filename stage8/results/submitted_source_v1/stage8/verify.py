"""Checks mathematical behavior and data separation, not fitted test outcomes."""
import argparse,ast,tempfile
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
    for f in Path(__file__).parent.glob('*.py'):ast.parse(f.read_text())
    print('PASS: identity, learned gradients, safe checkpoint roundtrip, tie-aware NDCG, ranking ties, image/hash separation, source syntax')

if __name__=='__main__':main()
