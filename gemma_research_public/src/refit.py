"""Paired CPU refit, using the original mean config/epochs and frozen cache."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from bundle import load_data, read, sha, verify_assets, write
import learning


def traced_fit(cache, rows, config, hyper, epochs, phase, seed):
    original_init, original_randint = learning.BoundedAdapter, torch.randint
    initialization, batches = [], hashlib.sha256()
    def init(*args, **kwargs):
        model=original_init(*args,**kwargs);digest=hashlib.sha256()
        for name,value in sorted(model.state_dict().items()):
            digest.update(name.encode());digest.update(value.detach().cpu().numpy().tobytes())
        initialization.append(digest.hexdigest());return model
    def randint(*args,**kwargs):
        value=original_randint(*args,**kwargs);batches.update(value.detach().cpu().numpy().tobytes());return value
    learning.BoundedAdapter,torch.randint=init,randint
    try:
        head,history,_=learning.fit(cache,rows,[],config,hyper,seed,phase,'cpu',epochs=epochs,scale=2.5,clamp=True)
    finally:
        learning.BoundedAdapter,torch.randint=original_init,original_randint
    return head,history,{'initialization_sha256':initialization,'sampled_batch_ids_sha256':batches.hexdigest()}


def refit(root,output,seeds=(11,)):
    root=verify_assets(root);output=Path(output)
    if output.exists():raise FileExistsError('Choose a new output directory; never overwrite a previous run')
    output.mkdir(parents=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    core,records,cache=load_data(root)
    train=[r for r in records if r['split']=='train']+[r for r in records if r['split']=='validation']
    test=[r for r in records if r['split']=='test']
    protocol=read(core/'protocol.json');choice=read(core/'gemma/selection.json')['mean']['selection']
    if choice['fallback_base']:raise ValueError('Original mean config is unavailable')
    results,checks={},{}
    write(output/'status.json',{'complete':False})
    for seed in seeds:
        phases,traces,scores={},{},{}
        for phase in ['mean','uncertainty']:
            print('CPU_REFIT',seed,phase,'epochs',choice['epochs'],flush=True)
            head,history,traces[phase]=traced_fit(cache,train,choice['config'],protocol['learning'],choice['epochs'],phase,seed)
            candidate,negative=learning.prediction_pack(cache,test,head,'cpu',2.5,True)
            phases[phase]=learning.partition_metrics(test,candidate,negative)
            scores[phase+'_candidate'],scores[phase+'_negative']=candidate,negative
            torch.save({'state_dict':head.state_dict(),'dimension':768,'config':choice['config'],'seed':seed,'phase':phase},output/(phase+f'_seed{seed}.pt'))
            write(output/(phase+f'_seed{seed}_history.json'),history)
        if traces['mean']!=traces['uncertainty']:raise ValueError('Paired initialization or sampled batches differ')
        np.savez_compressed(output/f'predictions_seed{seed}.npz',**scores)
        reference=root/'assets/controls/gemma'/f'seed{seed}'/'predictions.npz'
        if reference.exists():
            with np.load(reference,allow_pickle=False) as z:
                delta=max(float(np.abs(scores[name]-z[name]).max()) for name in scores)
            same_runtime=read(root/'assets/controls/execution_plan.json')['torch']==torch.__version__
            checks[str(seed)]={'max_abs_difference_to_saved_cpu_control':delta,'same_torch_runtime':same_runtime}
            if same_runtime and delta>2e-5:raise ValueError('Refit differs from original paired CPU control beyond tolerance')
        results[str(seed)]=phases
        write(output/f'trace_seed{seed}.json',traces)
    write(output/'metrics.json',results)
    receipt={'complete':True,'seeds':list(seeds),'config':choice['config'],'epochs':choice['epochs'],
        'same_initialization_and_sampled_batches':True,'reference_controls':checks,
        'runtime_torch':torch.__version__,'device':'cpu','new_native_encoding':False,'primary_changed':False,
        'scope':'Post-test fixed-config paired CPU controls; compare CPU phases to each other, not to original CUDA fitting as an isolated loss effect'}
    write(output/'status.json',receipt)
    return results,receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle');p.add_argument('--output',required=True);p.add_argument('--seeds',default='11')
    a=p.parse_args();_,receipt=refit(a.bundle,a.output,tuple(int(s) for s in a.seeds.split(',')));print(receipt)
