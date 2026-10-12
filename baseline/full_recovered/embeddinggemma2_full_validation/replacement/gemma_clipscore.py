"""Drop-in-style reference-free caption scorer using EmbeddingGemma 2.

CLI mirrors the two required CLIPScore inputs: captions JSON + image directory.
It returns a score for every caption. Scores are uncalibrated compatibility,
not correctness probabilities. Fixed task prefix; no training required.
"""
import argparse,json
from pathlib import Path
import numpy as np
from score_adapter import clipscore_from_embeddings

class GemmaCLIPScore:
    def __init__(self,model_path='google/embeddinggemma-2',device='cpu',dimension=768,prompt='CLIPLiteral'):
        import torch
        from sentence_transformers import SentenceTransformer
        assert dimension in [128,256,512,768]
        self.dimension=dimension;self.prompt=prompt
        self.model=SentenceTransformer(model_path,device=device,config_kwargs={'audio_config':None},
          model_kwargs={'dtype':torch.float32,'attn_implementation':'sdpa'})

    def score(self,image_paths,captions,batch_size=4):
        assert len(image_paths)==len(captions)
        # Cache image representations when several candidate captions share an image.
        unique=list(dict.fromkeys(map(str,image_paths)));index={p:i for i,p in enumerate(unique)}
        image=self.model.encode([{'image':p} for p in unique],batch_size=batch_size,normalize_embeddings=True,show_progress_bar=False)
        text=list(captions)
        if self.prompt=='CLIPLiteral':text=['A photo depicts '+t for t in text];kwargs={'prompt':''}
        elif self.prompt=='None':kwargs={'prompt':''}
        else:kwargs={'prompt_name':self.prompt}
        caption=self.model.encode(text,batch_size=16,normalize_embeddings=True,show_progress_bar=False,**kwargs)
        return clipscore_from_embeddings(image[[index[str(p)] for p in image_paths]],caption,self.dimension)

def main():
    p=argparse.ArgumentParser();p.add_argument('candidates_json');p.add_argument('image_dir')
    p.add_argument('--model-path',default='google/embeddinggemma-2');p.add_argument('--dimension',type=int,default=768)
    p.add_argument('--prompt',choices=['FactChecking','CLIPLiteral','None','SearchQuery','SentenceSimilarity'],default='CLIPLiteral')
    p.add_argument('--device',default='cpu');p.add_argument('--save-per-instance',required=True)
    a=p.parse_args();captions=json.loads(Path(a.candidates_json).read_text());root=Path(a.image_dir)
    paths=[]
    for name in captions:
        matches=[p for p in [root/name,root/(name+'.jpg'),root/(name+'.png')] if p.is_file()]
        if not matches:raise FileNotFoundError(name)
        paths.append(str(matches[0]))
    scorer=GemmaCLIPScore(a.model_path,a.device,a.dimension,a.prompt);scores=scorer.score(paths,list(captions.values()))
    result={'mean_score':float(np.mean(scores)),'dimension':a.dimension,'prompt':a.prompt,
      'scores':{key:float(value) for key,value in zip(captions,scores)},'warning':'Uncalibrated compatibility score; not probability of factual correctness.'}
    Path(a.save_per_instance).write_text(json.dumps(result,indent=2));print('mean score',result['mean_score'])

if __name__=='__main__':main()
