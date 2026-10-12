"""Learn a residual scoring head on historical development images only."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import load_gemma,read,write,digest

class ResidualHead(nn.Module):
    def __init__(self,dim=768,rank=16):
        super().__init__();self.down=nn.Linear(dim,rank,bias=False);self.up=nn.Linear(rank,dim,bias=False)
        nn.init.normal_(self.down.weight,std=.02);nn.init.zeros_(self.up.weight)
    def forward(self,x):return F.normalize(x+self.up(self.down(x)),dim=-1)

class GemmaAdapter(nn.Module):
    def __init__(self,dim=768,rank=16):
        super().__init__();self.image=ResidualHead(dim,rank);self.text=ResidualHead(dim,rank)
    def scores(self,image,text):return (self.image(image)*self.text(text)).sum(-1)

def save_model(path,model):
    np.savez_compressed(path,**{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()})

def load_model(path,rank,device='cpu'):
    m=GemmaAdapter(rank=rank)
    with np.load(path,allow_pickle=False) as z:
        assert set(z.files)==set(m.state_dict())
        m.load_state_dict({k:torch.from_numpy(z[k]) for k in z.files})
    return m.to(device).eval()

def make_split(manifest,human_manifest):
    rows=manifest['records'];names=sorted({r['filename'] for r in rows if r['previously_seen']})
    assert len(names)==555
    shuffled=np.array(names);np.random.default_rng(20261009).shuffle(shuffled)
    cut=int(.8*len(shuffled));train=set(shuffled[:cut]);valid=set(shuffled[cut:])
    tests={r['filename'] for r in rows if not r['previously_seen']}
    assert not(train&valid or train&tests or valid&tests)
    hashes=manifest['media_sha256'];hh=set(human_manifest['media_sha256'].values())
    sets=[{hashes[n] for n in ss} for ss in [train,valid,tests]]
    assert not(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    assert not((sets[0]|sets[1])&hh)
    return {'train_images':sorted(train),'validation_images':sorted(valid),'historical_unseen_images':sorted(tests),
        'train_validation_test_image_and_hash_disjoint':True,'human_image_hash_overlap_with_training':0,
        'scope':'Historical development split; historical test results were already observed.'}

def macro_accuracy(records,correct,indices):
    groups={}
    for i in indices:groups.setdefault(records[int(i)]['filename'],[]).append(bool(correct[int(i)]))
    return float(np.mean([np.mean(v) for v in groups.values()]))

@torch.inference_mode()
def all_scores(model,arrays,device):
    values=[]
    for start in range(0,len(arrays[0]),1024):
        a,p,n=[torch.from_numpy(v[start:start+1024]).to(device) for v in arrays]
        values.append(torch.stack([model.scores(a,p),model.scores(a,n)],1).cpu().numpy())
    return np.concatenate(values)

def train_one(rows,arrays,train_indices,val_indices,config,rank,anchor,seed,device,out):
    torch.manual_seed(seed);np.random.seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    model=GemmaAdapter(rank=rank).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    tensors=[torch.from_numpy(a).to(device) for a in arrays]
    rng=np.random.default_rng(seed);best=-1.;best_epoch=-1;state=None;history=[];started=time.perf_counter()
    for epoch in range(config['epochs']):
        model.train();total=0.;batches=0
        order=rng.permutation(train_indices)
        for start in range(0,len(order),config['batch_size']):
            ix=torch.as_tensor(order[start:start+config['batch_size']],device=device)
            image,pos,neg=[t[ix] for t in tensors];a=model.image(image);p=model.text(pos);n=model.text(neg)
            pair=F.softplus(((a*n).sum(-1)-(a*p).sum(-1)+config['pair_margin'])/config['pair_temperature']).mean()
            preservation=torch.stack([1-(a*image).sum(-1),1-(p*pos).sum(-1),1-(n*neg).sum(-1)]).mean()
            loss=pair+anchor*preservation
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
            total+=float(loss.detach());batches+=1
        model.eval();sc=all_scores(model,arrays,device);good=sc[:,0]>sc[:,1]
        va=macro_accuracy(rows,good,val_indices)
        history.append({'epoch':epoch+1,'loss':total/batches,'validation_image_macro_accuracy':va})
        if va>best+1e-12:best=va;best_epoch=epoch+1;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state);name=f'rank{rank}_anchor{anchor:g}_seed{seed}'
    path=out/'checkpoints'/(name+'.npz');path.parent.mkdir(parents=True,exist_ok=True);save_model(path,model)
    result={'name':name,'rank':rank,'anchor_weight':anchor,'seed':seed,'best_epoch':best_epoch,'validation_image_macro_accuracy':best,
        'history':history,'checkpoint':str(path.relative_to(out)),'checkpoint_sha256':digest(path),
        'trainable_parameters':sum(p.numel() for p in model.parameters()),'seconds':time.perf_counter()-started}
    write(out/'runs'/(name+'.json'),result);print(name,'validation',round(best,5),flush=True)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--baseline',required=True);parser.add_argument('--output',required=True);parser.add_argument('--device',default='cpu');parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True);config=read(Path(__file__).parent/'protocol.json')
    if args.device=='cuda':assert torch.cuda.is_available(),'GPU requested but CUDA unavailable'
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True)
    if args.device=='cuda':torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    mf,arrays=load_gemma(args.baseline);hm=read(Path(args.baseline)/'stage7/human_manifest.json');split=make_split(mf,hm);write(out/'split.json',split)
    rows=mf['records'];trainset=set(split['train_images']);valset=set(split['validation_images'])
    ti=np.array([i for i,r in enumerate(rows) if r['filename'] in trainset]);vi=np.array([i for i,r in enumerate(rows) if r['filename'] in valset])
    # Never feed historical test vectors into training or checkpoint-selection forward passes.
    dev_indices=np.concatenate([ti,vi]);dev_rows=[rows[i] for i in dev_indices];dev_arrays=[a[dev_indices] for a in arrays]
    local_train=np.arange(len(ti));local_val=np.arange(len(ti),len(dev_indices));runs=[]
    for rank in config['ranks']:
        for anchor in config['anchor_weights']:
            for seed in config['training_seeds']:runs.append(train_one(dev_rows,dev_arrays,local_train,local_val,config,rank,anchor,seed,args.device,out))
    groups=[]
    for rank in config['ranks']:
        for anchor in config['anchor_weights']:
            group=[r for r in runs if r['rank']==rank and r['anchor_weight']==anchor]
            groups.append({'rank':rank,'anchor_weight':anchor,'mean_validation_accuracy':float(np.mean([r['validation_image_macro_accuracy'] for r in group]))})
    selected=sorted(groups,key=lambda g:(-g['mean_validation_accuracy'],g['rank'],-g['anchor_weight']))[0]
    finalists=[r for r in runs if r['rank']==selected['rank'] and r['anchor_weight']==selected['anchor_weight']]
    result={'scope':config['scope'],'backbone_frozen':True,'new_backbone_inference':False,'device':args.device,'torch_version':torch.__version__,
        'train_images':len(trainset),'validation_images':len(valset),'train_pairs':len(ti),'validation_pairs':len(vi),
        'protocol_sha256':digest(Path(__file__).parent/'protocol.json'),'configurations':groups,'selected':selected,'finalists':finalists,
        'selection_uses_historical_benchmark_or_human_scores':False,'independent_new_test_completed':False}
    write(out/'selection.json',result)

if __name__=='__main__':main()
