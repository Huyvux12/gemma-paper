import argparse,csv,json,shutil
from pathlib import Path
import numpy as np
from common import read,write,digest
from metrics import evaluate_human

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--output',required=True);p.add_argument('--spice',action='store_true');p.add_argument('--no-meteor',action='store_true');a=p.parse_args()
    assert shutil.which('java'),'COCO PTBTokenizer and METEOR require Java; no silent tokenizer substitution'
    b=Path(a.baseline);records=read(b/'stage7/human_manifest.json')['records']
    source=b/'research/embeddinggemma2_clipscore_replacement/provenance/expert_author_annotations.json';original=read(source)
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    tokenizer=PTBTokenizer()
    refs={r['id']:[{'caption':t} for t in original[Path(r['filename']).stem]['ground_truth']] for r in records}
    candidates={r['id']:[{'caption':r['caption']}] for r in records}
    assert all(len(v)==5 for v in refs.values())
    # Ground-truth corpus comes from preserved original-author annotations, not model generations.
    gts=tokenizer.tokenize(refs);res=tokenizer.tokenize(candidates);ids=list(res)
    assert ids==[r['id'] for r in records]
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    scorers=[(Bleu(4),['BLEU1','BLEU2','BLEU3','BLEU4']),(Rouge(),['ROUGE_L']),(Cider(),['CIDEr'])]
    if not a.no_meteor:
        from pycocoevalcap.meteor.meteor import Meteor
        scorers.append((Meteor(),['METEOR']))
    if a.spice:
        from pycocoevalcap.spice.spice import Spice
        scorers.append((Spice(),['SPICE']))
    scores={};corpus={}
    for scorer,names in scorers:
        overall,values=scorer.compute_score(gts,res)
        if len(names)==1:overall,values=[overall],[values]
        for name,ov,v in zip(names,overall,values):
            if name=='SPICE':v=[x['All']['f'] for x in v]
            scores[name]=np.array(v,float);assert scores[name].shape==(len(records),) and np.isfinite(scores[name]).all();corpus[name]=float(ov)
        print('Computed',','.join(names),flush=True)
    cohorts={}
    for cohort,mask in [('full',np.ones(len(records),bool)),('unseen900',np.array([not r['previously_seen'] for r in records]))]:
        rows=[r for r,m in zip(records,mask) if m]
        cohorts[cohort]={k:evaluate_human(rows,v[mask]) for k,v in scores.items()}
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    write(out/'reference_baseline_scores.json',{'record_ids':ids,'scores':{k:v.tolist() for k,v in scores.items()},'reference_source_sha256':digest(source)})
    write(out/'reference_metrics.json',{'scope':'Reference-based baselines; separate from reference-free methods. Exploratory new point estimates on observed datasets.',
        'tokenizer':'COCO PTBTokenizer','pycocoevalcap':'1.2','cider_idf':'Standard candidate-record corpus; references repeated for multiple rated candidates of the same image.',
        'cohorts':cohorts,'corpus_scores_full':corpus,'source_sha256':digest(source)})
    with (out/'reference_scores.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['id','filename',*scores]);w.writeheader()
        for i,r in enumerate(records):w.writerow({'id':r['id'],'filename':r['filename'],**{k:float(v[i]) for k,v in scores.items()}})

if __name__=='__main__':main()
