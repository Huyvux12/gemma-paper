"""Symmetric per-vector int8 storage. Encoder weights are not quantized."""
import argparse,json
from pathlib import Path
import numpy as np

def pack(vectors,dimension):
    v=np.asarray(vectors,dtype=np.float64)[:,:dimension]
    n=np.linalg.norm(v,axis=1,keepdims=True)
    if np.any(n==0) or not np.isfinite(v).all():raise ValueError('Invalid embedding')
    v=v/n
    scales=(np.max(np.abs(v),axis=1,keepdims=True)/127).astype(np.float32)
    values=np.rint(v/scales).clip(-127,127).astype(np.int8)
    return values,scales

def unpack(values,scales):
    v=np.asarray(values,dtype=np.float64)*np.asarray(scales,dtype=np.float64)
    return v/np.linalg.norm(v,axis=1,keepdims=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);p.add_argument('--dimension',type=int,default=256);a=p.parse_args()
    source=np.load(a.input);packed={};raw=0
    for key in source.files:
        if source[key].ndim!=2 or not np.issubdtype(source[key].dtype,np.floating):raise ValueError('Only 2D embeddings supported')
        if a.dimension>source[key].shape[1]:raise ValueError('Dimension exceeds embedding size')
        q,s=pack(source[key],a.dimension);packed[key+'_values']=q;packed[key+'_scale']=s;raw+=q.nbytes+s.nbytes
    np.savez(a.output,**packed)
    print(json.dumps({'arrays':len(source.files),'dimension':a.dimension,'payload_bytes':raw,'file_bytes':a.output.stat().st_size,'bytes_per_vector':a.dimension+4}))

if __name__=='__main__':main()
