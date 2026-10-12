"""Fixed wrong-image control using cached vectors; not shuffled-image accuracy."""
import hashlib,json
import numpy as np
from analyze import load,ROOT,G,FC,B,L,S,D,unit,clipscore_from_embeddings

def main():
    rows,scores,metadata=load('sugar');selected=np.array([i for i,r in enumerate(rows) if not r['previously_seen']]);names=sorted({rows[i]['filename'] for i in selected});ix={n:i for i,n in enumerate(names)}
    groups=np.array([ix[rows[i]['filename']] for i in selected]);perms=[]
    for i in range(20):
        order=np.random.default_rng(202610086+i).permutation(len(names));mapping=np.empty(len(names),int);mapping[order]=np.roll(order,-1)
        assert np.all(mapping!=np.arange(len(names)));perms.append(mapping)
    out={};tests=[('gemma','CLIPLiteral',G),('gemma','FactChecking',FC),('clip','CLIPScore',B),('clip_large','CLIPScore',L),('specs','NativeSPECS',S),('siglip','SigLIP2',D)]
    for kind,prompt,name in tests:
        with np.load(ROOT/'runs'/kind/'embeddings.npz') as ar:e={k:ar[k] for k in ar.files}
        global_ix={n:i for i,n in enumerate(metadata[kind]['media_order'])};media=e['media'][[global_ix[n] for n in names]]
        text={k:e[prompt+'_'+k][selected] for k in ['positive','negative']};rates=[];ties=[]
        for perm in perms:
            a=media[perm[groups]];v={}
            for k,b in text.items():
                if kind in ['specs','siglip']:
                    c=np.einsum('ij,ij->i',unit(a),unit(b));v[k]=np.maximum((c+1)/2,0) if kind=='specs' else c
                else:v[k]=clipscore_from_embeddings(a,b)
            rates.append(float((v['positive']>v['negative']).mean()));ties.append(int((v['positive']==v['negative']).sum()))
        actual=float((scores[name]['positive'][selected]>scores[name]['negative'][selected]).mean())
        out[name]={'actual_accuracy':actual,'wrong_image_positive_preference_mean':float(np.mean(rates)),
          'wrong_image_positive_preference_min_max':[min(rates),max(rates)],'rates_each_permutation':rates,'ties_each_permutation':ties,
          'actual_minus_wrong_image_mean_pp':100*(actual-np.mean(rates))}
    maps=[{'seed':202610086+i,'mapping':{n:names[mapping[j]] for j,n in enumerate(names)}} for i,mapping in enumerate(perms)]
    result={'n_images':len(names),'n_pairs':len(selected),'n_permutations':20,'methods':out,
      'warning':'Wrong-image positive preference is a diagnostic of language priors and image sensitivity, not accuracy or human validation for the shuffled pairs.'}
    (ROOT/'negative_control.json').write_text(json.dumps(result,indent=2));(ROOT/'negative_control_maps.json').write_text(json.dumps(maps,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
