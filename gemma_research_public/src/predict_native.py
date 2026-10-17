"""Score one new image/caption pair using the frozen Gemma primary head."""
import argparse
from pathlib import Path
from bundle import find_bundle,load_head,read,verify_assets,verify_code


def score(image,caption,root=None,variant='mean'):
    import torch
    from PIL import Image
    from native import Encoder
    if variant not in ['base','mean','uncertainty']:raise ValueError('Unknown variant')
    if not torch.cuda.is_available():raise RuntimeError('Use this inference route on a CUDA GPU')
    root=verify_assets(root);pins=verify_code();core=root/'assets/core'
    encoder=Encoder('gemma',pins['gemma_model'],Path('runs/inference_receipt'),dtype_override=torch.float32)
    if encoder.receipt['weights']!=pins['gemma_weights']:raise ValueError('Wrong author checkpoint')
    head=load_head(core,variant,'cuda') if variant!='base' else None
    with Image.open(image) as im:picture=im.convert('RGB')
    with torch.inference_mode():
        u,v=encoder.images([picture]),encoder.texts([caption])
        if head is not None:u,v=head(u),head(v)
        value=2.5*(u*v).sum(-1).clamp_min(0)
    result=float(value.item());encoder.unload()
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--caption',required=True);p.add_argument('--bundle');p.add_argument('--variant',choices=['base','mean','uncertainty'],default='mean')
    a=p.parse_args();print({'variant':a.variant,'score':score(a.image,a.caption,a.bundle,a.variant),'probability':False})
