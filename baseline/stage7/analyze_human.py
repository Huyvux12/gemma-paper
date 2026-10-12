"""Frozen expert evaluation with independent weighted tau-c verification."""
import argparse,csv,json,math,time
from pathlib import Path
import numpy as np
from scipy.stats import kendalltau,spearmanr
ROOT=Path(__file__).resolve().parent
G='gemma-CLIPLiteral-768-float32';G256='gemma-CLIPLiteral-256-float32';FC='gemma-FactChecking-768-float32';B='clip-CLIPScore';L='clip_large-CLIPScore';S='specs-NativeSPECS';D='siglip-SigLIP2';R='siglip224-raw';M='siglip224-CLIPLiteral'

def unit(x):
 x=np.asarray(x,dtype=np.float64);return x/np.linalg.norm(x,axis=1,keepdims=True)

def storage(x,d,fmt):
 x=unit(x[:,:d])
 if fmt=='float16':x=x.astype(np.float16).astype(float)
 elif fmt=='int8':
  scale=(np.max(abs(x),axis=1,keepdims=True)/127).astype(np.float32);x=np.rint(x/scale).clip(-127,127).astype(np.int8).astype(float)*scale
 return unit(x)

class WeightedTau:
 def __init__(self,scores,rows):
  self.labels=sorted({v for r in rows for v in r['ratings']});self.names=sorted({r['filename'] for r in rows});ni={n:i for i,n in enumerate(self.names)}
  self.images=np.array([ni[r['filename']] for r in rows]);li={v:i for i,v in enumerate(self.labels)};hist=np.zeros((len(rows),len(li)),dtype=float)
  for i,r in enumerate(rows):
   for v in r['ratings']:hist[i,li[v]]+=1
  self.hist=hist;self.groups={};self.verification_max_error=0
  for k,v in scores.items():
   levels,inv=np.unique(v,return_inverse=True);self.groups[k]=(inv,len(levels))
 def evaluate(self,key,weights):
  inv,nlevels=self.groups[key];h=np.zeros((nlevels,len(self.labels)),float);np.add.at(h,inv,self.hist*weights[self.images,None])
  earlier=np.cumsum(h,axis=0)-h;low=np.cumsum(earlier,axis=1)-earlier;high=earlier.sum(1,keepdims=True)-np.cumsum(earlier,axis=1)
  net=float(np.sum(h*(low-high)));n=float(h.sum());m=min(int(np.sum(h.sum(0)>0)),int(np.sum(h.sum(1)>0)))
  return 2*net/(n*n*(m-1)/m) if m>1 else float('nan')

def bootstrap_cohort(records,scores,mask,reps=2000,baselines_only=False):
 selected=np.flatnonzero(mask);rows=[records[i] for i in selected];x={k:v[selected] for k,v in scores.items()};counts=np.array([len(r['ratings']) for r in rows]);y=np.array([v for r in rows for v in r['ratings']]);expanded={k:np.repeat(v,counts) for k,v in x.items()}
 tau=WeightedTau(x,rows);weights=np.ones(len(tau.names));metrics={}
 for k,v in expanded.items():
  scalar=tau.evaluate(k,weights);scipy=float(kendalltau(v,y,variant='c').statistic);assert abs(scalar-scipy)<1e-12,(k,scalar,scipy)
  metrics[k]={'kendall_tau_c':scipy,'spearman':float(spearmanr(v,y).statistic),'n_unique_scores':len(np.unique(x[k]))}
 tested=[k for k in [G,G256,FC,B,L,S,D,R,M,R+'-clipped',M+'-clipped','gemma-CLIPLiteral-256-int8'] if k in x]
 rng=np.random.default_rng(202610084);boots={k:np.empty(reps) for k in tested};expanded_by_image=[np.flatnonzero(np.repeat(tau.images,counts)==i) for i in range(len(tau.names))];max_error=0.
 start=time.perf_counter()
 for b in range(reps):
  draws=rng.integers(0,len(tau.names),len(tau.names));w=np.bincount(draws,minlength=len(tau.names))
  for k in tested:boots[k][b]=tau.evaluate(k,w)
  if b<4:
   ix=np.concatenate([expanded_by_image[i] for i in draws])
   for k in tested:
    reference=float(kendalltau(expanded[k][ix],y[ix],variant='c').statistic);max_error=max(max_error,abs(reference-boots[k][b]));assert abs(reference-boots[k][b])<1e-12
  if b and b%500==0:print('Bootstrap',len(tau.names),'images',b,'/',reps,'seconds',round(time.perf_counter()-start,1),flush=True)
 for k,values in boots.items():metrics[k]['ci95']=np.quantile(values,[.025,.975]).tolist()
 contrasts=[(G,k) for k in [B,L,S,D,R,M]]+[(FC,L),(G256,G),('gemma-CLIPLiteral-256-int8',G)]
 comparisons={}
 for a,b in contrasts:
  if a not in boots or b not in boots:continue
  diff=boots[a]-boots[b];v={'delta_tau_c':metrics[a]['kendall_tau_c']-metrics[b]['kendall_tau_c'],'ci95':np.quantile(diff,[.025,.975]).tolist(),'one_sided_95_lower':float(np.quantile(diff,.05))}
  if a==G:v['bonferroni_family6_ci']=np.quantile(diff,[.05/12,1-.05/12]).tolist()
  if a==G256 and b==G:v['source_exploratory_tau_tolerance']=.01;v['source_exploratory_tolerance_passed']=v['one_sided_95_lower']>-.01
  comparisons[a+'_vs_'+b]=v
 # Rank caption pairs within each image, unequal average human ratings only.
 names=tau.names;groups={n:[] for n in names}
 for i,row in enumerate(rows):groups[row['filename']].append(i)
 pair_counts=[];wins={k:[] for k in tested}
 for name in names:
  ids=groups[name];pairs=[]
  for ii,i in enumerate(ids):
   for j in ids[ii+1:]:
    ri=np.mean(rows[i]['ratings']);rj=np.mean(rows[j]['ratings'])
    if ri!=rj:pairs.append((i,j) if ri>rj else (j,i))
  if not pairs:continue
  pair_counts.append(len(pairs))
  for k in tested:wins[k].append(sum(x[k][i]>x[k][j] for i,j in pairs))
 pc=np.asarray(pair_counts);wp={k:float(np.sum(v)/np.sum(pc)) for k,v in wins.items()};rng=np.random.default_rng(202610085);draws=rng.integers(0,len(pc),(5000,len(pc)));denom=pc[draws].sum(1);wb={k:np.asarray(v)[draws].sum(1)/denom for k,v in wins.items()}
 within_comps={}
 for a,b in contrasts:
  if a in wb and b in wb:within_comps[a+'_vs_'+b]={'delta_pp':100*(wp[a]-wp[b]),'ci95_pp':(100*np.quantile(wb[a]-wb[b],[.025,.975])).tolist()}
 return {'n_images':len(names),'n_captions':len(rows),'n_individual_ratings':len(y),'metrics':metrics,'comparisons':comparisons,'within_image_ranking':{'n_unequal_human_caption_pairs':int(np.sum(pc)),'n_images_with_unequal_pairs':len(pc),'accuracy':wp,'comparisons':within_comps},'weighted_tau_validation':{'all_point_tau_match_scipy':True,'n_bootstrap_draws_checked_against_scipy':4,'max_abs_bootstrap_tau_error':max_error,'passed':True}}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--baselines-only',action='store_true');parser.add_argument('--bootstrap-reps',type=int,default=2000);args=parser.parse_args()
 mf=json.loads((ROOT/'human_manifest.json').read_text());records=mf['records'];scores={};metadata={};max_scalar_error=0;scalar_checked=0
 kinds=['clip','clip_large','specs','siglip'] if args.baselines_only else ['gemma','clip','clip_large','specs','siglip','siglip224']
 for kind in kinds:
  path=ROOT/'runs'/(kind+'-human');meta=json.loads((path/'metadata.json').read_text());metadata[kind]=meta;assert meta['record_ids']==[r['id'] for r in records]
  loaded=np.load(path/'embeddings.npz',allow_pickle=False);a={k:loaded[k] for k in loaded.files};ix={n:i for i,n in enumerate(meta['media_order'])};image=a['media'][[ix[r['filename']] for r in records]]
  for prompt in meta['prompts']:
   text=a[prompt+'_candidate']
   for dim in ([128,256,512,768] if kind=='gemma' else [image.shape[1]]):
    for fmt in (['float32','float16','int8'] if kind=='gemma' else ['float32']):
     im,tx=storage(image,dim,fmt),storage(text,dim,fmt);cos=np.einsum('ij,ij->i',im,tx)
     if kind=='gemma':key=f'{kind}-{prompt}-{dim}-{fmt}'
     else:key=f'{kind}-{prompt}'
     value=np.maximum((cos+1)/2,0) if kind=='specs' else cos if kind in ['siglip','siglip224'] else 2.5*np.maximum(cos,0)
     scores[key]=value
     if kind=='siglip224':scores[key+'-clipped']=2.5*np.maximum(cos,0)
     # Independent scalar check for every native-f32 image-caption score.
     if fmt=='float32' and (kind!='gemma' or dim==768):
      for i in range(len(records)):
       vecim=im[i];vectx=tx[i];scalar=math.fsum(float(x)*float(y) for x,y in zip(vecim,vectx));v=(scalar+1)/2 if kind=='specs' else scalar if kind in ['siglip','siglip224'] else 2.5*max(scalar,0)
       max_scalar_error=max(max_scalar_error,abs(v-value[i]));scalar_checked+=1
 assert max_scalar_error<1e-12
 cohorts={}
 for name,mask in [('full',np.ones(len(records),bool)),('unseen',np.array([not r['previously_seen'] for r in records]))]:
  cohorts[name]=bootstrap_cohort(records,scores,mask,reps=args.bootstrap_reps,baselines_only=args.baselines_only)
 out={'cohorts':cohorts,'metadata':metadata,'scope':'Human evaluation secondary. Full includes100 prior development images; unseen900 is the separate continuation cohort. SigLIP224 native cosine and clipped sensitivity both disclosed.','bootstrap_repetitions':args.bootstrap_reps,'score_verification':{'scalar_scores_checked':scalar_checked,'max_abs_score_error':max_scalar_error,'passed':True},'ties_and_clipping':{k:{'n_zero_scores':int(np.sum(v==0)),'n_unique_scores':len(np.unique(v))} for k,v in scores.items()}}
 basename='baseline_human' if args.baselines_only else 'human'
 (ROOT/(basename+'_results.json')).write_text(json.dumps(out,indent=2))
 with (ROOT/(basename+'_scores.csv')).open('w') as f:
  writer=csv.DictWriter(f,fieldnames=['id','filename','caption','ratings','previously_seen']+list(scores));writer.writeheader()
  for i,r in enumerate(records):writer.writerow({**{k:r[k] for k in ['id','filename','caption','ratings','previously_seen']},**{k:float(v[i]) for k,v in scores.items()}})
 print(json.dumps({'cohorts':{c:{'n_images':v['n_images'],'metrics':{k:v['metrics'][k] for k in [G,G256,B,L,S,D,R,M] if k in v['metrics']},'comparisons':v['comparisons']} for c,v in cohorts.items()},'verification':out['score_verification']},indent=2))

if __name__=='__main__':main()
