"""Predeclared secondary prefix from the strongest PRIOR image result.

Uses the same media vectors; only computes a further text prefix. No test
selection or fitting. Run after all six inference runs, then re-analyze.
"""
import importlib.metadata,json,time
from pathlib import Path
import numpy as np,torch
from sentence_transformers import SentenceTransformer
ROOT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(6)
    model=SentenceTransformer(str(ROOT/'models/gemma'),device='cpu',local_files_only=True,
        config_kwargs={'vision_config':None,'audio_config':None},model_kwargs={'dtype':torch.float32,'attn_implementation':'sdpa'})
    for dataset in ['sugar','human']:
        manifest=json.loads((ROOT/('manifest.json' if dataset=='sugar' else 'human_manifest.json')).read_text())
        records=manifest['records'];folder=ROOT/'runs'/('gemma' if dataset=='sugar' else 'gemma-human')
        arrays=dict(np.load(folder/'embeddings.npz'));meta=json.loads((folder/'metadata.json').read_text())
        labels=[('positive','caption'),('negative','negative_caption')] if dataset=='sugar' else [('candidate','caption')]
        start=time.perf_counter()
        for label,key in labels:
            arrays['SentenceSimilarity_'+label]=model.encode([r[key] for r in records],prompt_name='SentenceSimilarity',batch_size=16,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
        np.savez_compressed(folder/'embeddings.npz',**arrays)
        if 'SentenceSimilarity' not in meta['prompts']:meta['prompts'].append('SentenceSimilarity')
        meta['secondary_similarity_addendum']={'parameters':sum(p.numel() for p in model.parameters()),'text_seconds':time.perf_counter()-start,'configuration':'text-only encoder; same verified text weights; frozen extra prefix from PRIOR results'}
        (folder/'metadata.json').write_text(json.dumps(meta,indent=2));print(dataset,'SentenceSimilarity COMPLETE',flush=True)

if __name__=='__main__':main()
