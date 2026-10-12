"""Check the default API, including repeated images, against saved inference."""
import json
from pathlib import Path
import numpy as np
import torch
from analyze import load_scores
from gemma_clipscore import GemmaCLIPScore

ROOT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(6)
    records,scores,_=load_scores()
    paths=[];captions=[];expected=[]
    for i,row in enumerate(records[:3]):
        for label,field in [('positive','caption'),('negative','negative_caption')]:
            paths.append(ROOT/'images'/row['filename'])
            captions.append(row[field])
            expected.append(scores['gemma-CLIPLiteral-768'][label][i])
    scorer=GemmaCLIPScore(str(ROOT/'models/gemma'))
    actual=scorer.score(paths,captions)
    error=float(np.max(np.abs(actual-expected)))
    print('scores',actual.tolist(),'reference',expected,'maximum_abs_error',error,flush=True)
    assert error<1e-5,error
    result={'passed':True,'prompt':scorer.prompt,'dimension':scorer.dimension,
      'n_images':3,'n_captions':6,'maximum_abs_error':error,
      'repeated_image_cache_tested':True}
    (ROOT/'adapter_default_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
