import argparse,json
from pathlib import Path
import numpy as np
import torch
from common import read,write,load_gemma
from train_adapter import load_model,all_scores,macro_accuracy

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--trained',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);selection=read(Path(a.trained)/'selection.json')
    sugar,sv=load_gemma(a.baseline);human,hv=load_gemma(a.baseline,True)
    frozen_ids={r['id'] for r in read(Path(a.baseline)/'stage5/frozen_manifest.json')['records']}
    masks={'historical560':np.array([r['id'] in frozen_ids for r in sugar['records']]),'historical_unseen3949':np.array([not r['previously_seen'] for r in sugar['records']])}
    assert masks['historical560'].sum()==560
    original=2.5*np.maximum(np.einsum('ij,ij->i',hv[0],hv[1]),0)
    scores={'gemma-original-literal768':original.tolist()};results={};sugar_predictions=[];human_predictions=[]
    for r in selection['finalists']:
        model=load_model(Path(a.trained)/r['checkpoint'],r['rank'],a.device)
        pairs=2.5*np.maximum(all_scores(model,sv,a.device),0);sugar_predictions.append(pairs)
        human_s=[]
        with torch.inference_mode():
            for start in range(0,len(hv[0]),1024):
                image,text=[torch.from_numpy(v[start:start+1024]).to(a.device) for v in hv]
                human_s.append(model.scores(image,text).cpu().numpy())
        h=2.5*np.maximum(np.concatenate(human_s),0);human_predictions.append(h);name='gemma-adapter-seed'+str(r['seed']);scores[name]=h.tolist()
        results[name]={cohort:{'accuracy':float((pairs[mask,0]>pairs[mask,1]).mean()),'n_pairs':int(mask.sum()),'n_images':len({row['filename'] for row,m in zip(sugar['records'],mask) if m})} for cohort,mask in masks.items()}
    meanpairs=np.mean(sugar_predictions,axis=0);scores['gemma-adapter-ensemble5']=np.mean(human_predictions,axis=0).tolist()
    results['gemma-adapter-ensemble5']={cohort:{'accuracy':float((meanpairs[mask,0]>meanpairs[mask,1]).mean()),'n_pairs':int(mask.sum())} for cohort,mask in masks.items()}
    write(out/'adapter_human_scores.json',{'record_ids':[r['id'] for r in human['records']],'scores':scores,'scope':'Post-selection diagnostics on previously observed historical benchmarks; no fresh confirmation.'})
    write(out/'adapter_sugar_diagnostics.json',{'scope':'Historical observed test diagnostics; no model selection from these scores.','models':results,'independent_test_completed':False})
    np.savez_compressed(out/'adapter_sugar_scores.npz',record_ids=np.array([r['id'] for r in sugar['records']]),ensemble_positive=meanpairs[:,0],ensemble_negative=meanpairs[:,1],all_seed_pairs=np.stack(sugar_predictions))
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
