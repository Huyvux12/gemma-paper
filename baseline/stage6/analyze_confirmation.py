"""Recompute the returned SigLIP2-224 results; do not execute uploaded Python."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / 'stage5'
UP = ROOT / 'uploaded'
SEED = 20261009
N = 560


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    assert np.isfinite(x).all()
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    assert (norms > 0).all()
    return x / norms


def compare(a, b):
    d = a.astype(int) - b.astype(int)
    rng = np.random.default_rng(SEED)
    draws = np.concatenate([d[rng.integers(0, N, (500, N))].mean(1) for _ in range(20)])
    wins = int(np.sum(a & ~b))
    losses = int(np.sum(~a & b))
    return {'delta_pp': float(d.mean() * 100), 'ci95_pp': (100 * np.quantile(draws, [.025, .975])).tolist(),
            'wins': wins, 'losses': losses,
            'p_exact': float(binomtest(wins, wins + losses).pvalue) if wins + losses else 1.,
            'both_correct': int(np.sum(a & b)), 'both_wrong': int(np.sum(~a & ~b))}


def scalar_cos(a, b):
    a, b = list(map(float, a)), list(map(float, b))
    return math.fsum(x*y for x, y in zip(a, b)) / math.sqrt(math.fsum(x*x for x in a) * math.fsum(y*y for y in b))


def main():
    inputs = {}
    for name in ['frozen_manifest.json', 'frozen_protocol.json', 'primary_reference.json', 'run_siglip224.py']:
        assert (UP/name).read_bytes() == (PRIOR/name).read_bytes(), name
        inputs[name] = digest(UP/name)
    assert inputs['frozen_manifest.json'] == 'a2e9fa5fb7e9866631c0546e5a09a2f9d11044335267712591476572b6104efe'
    records = json.loads((UP/'frozen_manifest.json').read_text())['records']
    ref = json.loads((UP/'primary_reference.json').read_text())
    returned = json.loads((UP/'primary_siglip224_results.json').read_text())
    assert len(records) == N and len({r['filename'] for r in records}) == N
    assert ref['record_ids'] == [r['id'] for r in records]
    assert returned['revision'] == '75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2'
    assert returned['weight_sha256'] == '612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b'
    assert returned['model'] == 'google/siglip2-base-patch16-224' and returned['dtype'] == 'float32'
    assert returned['manifest_sha256'] == inputs['frozen_manifest.json']
    assert all(x['over_context'] == 0 and x['max_tokens'] <= x['context'] for x in returned['token_audit'].values())
    z = np.load(UP/'runs/siglip224/embeddings.npz', allow_pickle=False)
    arrays = {k: z[k] for k in z.files}
    assert set(arrays) == {'media', 'raw_positive', 'raw_negative', 'CLIPLiteral_positive', 'CLIPLiteral_negative'}
    for k, v in arrays.items():
        assert v.shape == (N, 768) and v.dtype == np.float32 and np.isfinite(v).all()
        assert np.max(abs(np.linalg.norm(v, axis=1)-1)) < 1e-5
    names = sorted(r['filename'] for r in records)
    positions = {name: i for i, name in enumerate(names)}
    images = arrays['media'][[positions[r['filename']] for r in records]]
    scores, vectors = {}, {}
    for prompt in ['raw', 'CLIPLiteral']:
        key = 'siglip224-' + prompt
        vectors[key] = (images, arrays[prompt+'_positive'], arrays[prompt+'_negative'])
        scores[key] = [2.5*np.maximum(np.einsum('ij,ij->i',unit(images),unit(arrays[prompt+'_'+label])),0) for label in ['positive','negative']]
    for kind, prompt, name in [('gemma','CLIPLiteral','gemma-CLIPLiteral'),('clip','CLIPScore','clip-B32'),('clip_large','CLIPScore','clip-L14')]:
        folder = PRIOR/'runs'/kind
        meta = json.loads((folder/'metadata.json').read_text())
        assert meta['record_ids'] == ref['record_ids']
        archive = np.load(folder/'embeddings.npz',allow_pickle=False)
        order = {x:i for i,x in enumerate(meta['media_order'])}
        image = archive['media'][[order[r['filename']] for r in records]]
        vectors[name] = (image, archive[prompt+'_positive'], archive[prompt+'_negative'])
        scores[name] = [2.5*np.maximum(np.einsum('ij,ij->i',unit(image),unit(archive[prompt+'_'+label])),0) for label in ['positive','negative']]
    assert np.max(abs(scores['gemma-CLIPLiteral'][0]-ref['gemma_positive'])) < 1e-12
    assert np.max(abs(scores['gemma-CLIPLiteral'][1]-ref['gemma_negative'])) < 1e-12
    correct = {k:sp>sn for k,(sp,sn) in scores.items()}
    metrics = {k:{'correct':int(ok.sum()),'n':N,'accuracy':float(ok.mean()),'ties':int(np.sum(scores[k][0]==scores[k][1])),
                  'category_accuracy':{c:float(ok[[r['category']==c for r in records]].mean()) for c in sorted({r['category'] for r in records})}} for k,ok in correct.items()}
    primary = compare(correct['gemma-CLIPLiteral'],correct['siglip224-raw'])
    secondary = {k:compare(correct['gemma-CLIPLiteral'],correct[k]) for k in ['siglip224-CLIPLiteral','clip-B32','clip-L14']}
    previous = 0.
    for i,k in enumerate(sorted(secondary,key=lambda x:secondary[x]['p_exact'])):
        previous = max(previous,min(1.,secondary[k]['p_exact']*(3-i)))
        secondary[k]['p_holm_family3'] = previous
    # Independently evaluate every cosine with Python float and fsum, plus exact p via integer arithmetic.
    maximum_error = 0.
    scalar_count = 0
    for key,(image,positive,negative) in vectors.items():
        for i in range(N):
            values = [2.5*max(scalar_cos(image[i],text[i]),0) for text in [positive,negative]]
            maximum_error = max(maximum_error,*[abs(values[j]-scores[key][j][i]) for j in [0,1]])
            assert (values[0]>values[1]) == correct[key][i]
            scalar_count += 2
    assert maximum_error < 1e-12
    for v in [primary,*secondary.values()]:
        total = v['wins']+v['losses']
        exact = min(1.,2*sum(math.comb(total,i) for i in range(min(v['wins'],v['losses'])+1))/2**total) if total else 1.
        assert abs(exact-v['p_exact']) < 1e-12
    for key, value in [('gemma_correct',metrics['gemma-CLIPLiteral']['correct']),('siglip224_correct',metrics['siglip224-raw']['correct']),('wins',primary['wins']),('losses',primary['losses'])]:
        assert returned[key] == value, key
    for key, value in [('delta_pp',primary['delta_pp']),('mcnemar_p',primary['p_exact'])]:
        assert abs(returned[key]-value) < 1e-12, key
    assert np.max(abs(np.asarray(returned['ci95_pp'])-primary['ci95_pp'])) < 1e-12
    native = {}
    controls = {}
    for key in ['siglip224-raw','siglip224-CLIPLiteral']:
        image,p,n = [unit(a) for a in vectors[key]]
        matchedp,matchedn = np.einsum('ij,ij->i',image,p),np.einsum('ij,ij->i',image,n)
        native[key] = {'raw_cosine_accuracy':float((matchedp>matchedn).mean()),'clipped_accuracy':metrics[key]['accuracy'],'clipped_ties':metrics[key]['ties'],'decisions_differ_from_raw_cosine':int(np.sum((matchedp>matchedn)!=correct[key]))}
        pos,neg = p@image.T,n@image.T
        pref = pos>neg;np.fill_diagonal(pref,False)
        ties = np.maximum(pos,0)==np.maximum(neg,0);np.fill_diagonal(ties,False)
        controls[key] = {'n_wrong_image_pairings':N*(N-1),'wrong_image_raw_cosine_positive_preference':float(pref.sum()/(N*(N-1))),'clipped_wrong_image_tie_fraction':float(ties.sum()/(N*(N-1))),'interpretation':'Preference for the original positive caption under unrelated images; not correctness for those images. Repeated pairings are not independent samples.'}
    result = {'primary_completed':True,'n_images':N,'n_pairs':N,'metrics':metrics,'primary':primary,'secondary':secondary,'native_cosine_sensitivity':native,'wrong_image_control':controls,'bootstrap_repetitions':10000,'bootstrap_seed':SEED,'secondary_holm_family_size':3,'input_sha256':inputs,'uploaded_npz_sha256':digest(UP/'runs/siglip224/embeddings.npz'),'colab_reported_metadata':returned,
              'verification':{'all_input_files_byte_identical':True,'all_uploaded_claimed_metrics_recomputed_and_match':True,'gemma_reference_recomputed_from_vectors':True,'independent_scalar_scores_checked':scalar_count,'maximum_score_error':maximum_error,'exact_binomial_pvalues_independently_match':True,'passed':True},
              'scope':'Gemma/CLIP vectors are archived CPU float32 runs; SigLIP224 vectors are the user-returned CUDA float32 inference. This turn only computes and verifies scores. Weights/images are not present in the upload, so their provenance hashes are runner checks and reported metadata, not new independent checks on their original bytes.'}
    with (ROOT/'pair_scores_confirmation.csv').open('w') as f:
        writer = csv.DictWriter(f,fieldnames=['id','filename','category','method','positive_score','negative_score','correct']);writer.writeheader()
        for k,(positive,negative) in scores.items():
            for i,r in enumerate(records):writer.writerow({'id':r['id'],'filename':r['filename'],'category':r['category'],'method':k,'positive_score':float(positive[i]),'negative_score':float(negative[i]),'correct':int(correct[k][i])})
    (ROOT/'verified_confirmation_results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
