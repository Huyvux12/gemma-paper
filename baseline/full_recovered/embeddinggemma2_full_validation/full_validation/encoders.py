"""Pinned native preprocessing, explicit dtype, float32 output vectors."""
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
PREV=ROOT.parent/'replacement'
sys.path.insert(0,str(PREV))
from score_adapter import unit

def array(v):
    if hasattr(v,'pooler_output'):v=v.pooler_output
    return unit(v.detach().float().cpu().numpy()).astype(np.float32)

class Encoder:
    def __init__(self,kind,dtype='bfloat16'):
        self.kind=kind;self.dtype=getattr(torch,dtype)
        if kind=='gemma':
            from sentence_transformers import SentenceTransformer
            self.model=SentenceTransformer(str(PREV/'models/gemma'),device='cpu',local_files_only=True,
              config_kwargs={'audio_config':None},model_kwargs={'dtype':self.dtype,'attn_implementation':'sdpa'})
            self.processor=None;self.prompts=['CLIPLiteral','FactChecking']
        elif kind=='specs':
            sys.path.insert(0,str(ROOT/'sources/SPECS'))
            from model import longclip
            self.native=longclip
            self.model,self.processor=longclip.load(str(ROOT/'models/spec.pt'),device='cpu')
            self.model.to(dtype=self.dtype);self.prompts=['NativeSPECS']
        elif kind=='siglip':
            from transformers import AutoModel,AutoProcessor
            path=ROOT/'models/siglip'
            self.model=AutoModel.from_pretrained(path,local_files_only=True,dtype=self.dtype)
            self.processor=AutoProcessor.from_pretrained(path,local_files_only=True);self.prompts=['SigLIP2']
        else:
            from transformers import CLIPModel,CLIPProcessor
            path=PREV/'models'/kind
            self.model=CLIPModel.from_pretrained(path,local_files_only=True,dtype=self.dtype)
            self.processor=CLIPProcessor.from_pretrained(path,local_files_only=True)
            self.prompts=['CLIPScore']
        self.model.eval()

    @torch.inference_mode()
    def images(self,paths,batch_size=8):
        if self.processor is None:
            return array(self.model.encode([{'image':str(p)} for p in paths],batch_size=batch_size,
              normalize_embeddings=False,convert_to_tensor=True,show_progress_bar=False))
        from PIL import Image
        ims=[]
        for p in paths:
            with Image.open(p) as im:ims.append(im.convert('RGB'))
        if self.kind=='specs':
            return array(self.model.encode_image(torch.stack([self.processor(im) for im in ims]).to(self.dtype)))
        inp=self.processor(images=ims,return_tensors='pt');inp['pixel_values']=inp['pixel_values'].to(self.dtype)
        return array(self.model.get_image_features(**inp))

    @torch.inference_mode()
    def texts(self,texts,prompt,batch_size=32):
        texts=list(texts)
        if prompt in ['CLIPLiteral','CLIPScore']:texts=['A photo depicts '+t for t in texts]
        if self.processor is None:
            kw={'prompt':''} if prompt=='CLIPLiteral' else {'prompt_name':prompt}
            return array(self.model.encode(texts,batch_size=batch_size,normalize_embeddings=False,
              convert_to_tensor=True,show_progress_bar=False,**kw))
        values=[]
        for i in range(0,len(texts),batch_size):
            if self.kind=='specs':
                values.append(array(self.model.encode_text(self.native.tokenize(texts[i:i+batch_size]))))
                continue
            kw={'padding':'max_length','max_length':64} if self.kind=='siglip' else {'padding':True,'max_length':77}
            inp=self.processor(text=texts[i:i+batch_size],truncation=True,return_tensors='pt',**kw)
            values.append(array(self.model.get_text_features(**inp)))
        return np.concatenate(values)
