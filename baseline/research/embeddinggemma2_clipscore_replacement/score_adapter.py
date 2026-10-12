"""CLIPScore's scoring rule, with a replaceable embedding encoder.

This is a CLIPScore-style adaptation, not a calibrated probability and not
an exact reproduction of the original OpenAI/PyTorch implementation.
"""
import numpy as np

def unit(v,dim=None):
    v=np.asarray(v,dtype=np.float64)
    if dim is not None:v=v[...,:dim]
    assert np.isfinite(v).all()
    n=np.linalg.norm(v,axis=-1,keepdims=True)
    assert (n>1e-12).all()
    return v/n

def clipscore_from_embeddings(images,captions,dim=None):
    a,b=unit(images,dim),unit(captions,dim)
    assert a.shape==b.shape
    return 2.5*np.maximum(np.einsum('ij,ij->i',a,b),0)
