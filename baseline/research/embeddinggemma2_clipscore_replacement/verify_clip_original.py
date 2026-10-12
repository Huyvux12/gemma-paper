"""Compare the HF baseline to official OpenAI CLIP code on identical images.

Map the same verified checkpoint into the original architecture, rather than
introducing another downloaded checkpoint with uncertain provenance.
"""
import json
from pathlib import Path
import clip,numpy as np,torch
from PIL import Image
from transformers import CLIPModel
ROOT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(4)
    hf=CLIPModel.from_pretrained(ROOT/'models/clip',local_files_only=True,dtype=torch.float32)
    hs=hf.state_dict();model=clip.model.CLIP(embed_dim=512,image_resolution=224,vision_layers=12,vision_width=768,vision_patch_size=32,
        context_length=77,vocab_size=49408,transformer_width=512,transformer_heads=8,transformer_layers=12).float().eval()
    s={};simple={'token_embedding.weight':'text_model.embeddings.token_embedding.weight','positional_embedding':'text_model.embeddings.position_embedding.weight',
        'ln_final.weight':'text_model.final_layer_norm.weight','ln_final.bias':'text_model.final_layer_norm.bias',
        'visual.conv1.weight':'vision_model.embeddings.patch_embedding.weight','visual.class_embedding':'vision_model.embeddings.class_embedding',
        'visual.positional_embedding':'vision_model.embeddings.position_embedding.weight',
        'visual.ln_pre.weight':'vision_model.pre_layrnorm.weight','visual.ln_pre.bias':'vision_model.pre_layrnorm.bias',
        'visual.ln_post.weight':'vision_model.post_layernorm.weight','visual.ln_post.bias':'vision_model.post_layernorm.bias','logit_scale':'logit_scale'}
    for a,b in simple.items():s[a]=hs[b]
    s['text_projection']=hs['text_projection.weight'].T;s['visual.proj']=hs['visual_projection.weight'].T
    for original,hprefix in [('transformer','text_model'),('visual.transformer','vision_model')]:
        for i in range(12):
            a=f'{original}.resblocks.{i}.';b=f'{hprefix}.encoder.layers.{i}.'
            for param in ['weight','bias']:
                s[a+'attn.in_proj_'+param]=torch.cat([hs[b+'self_attn.'+part+'_proj.'+param] for part in ['q','k','v']],dim=0)
                for x,y in [('attn.out_proj','self_attn.out_proj'),('ln_1','layer_norm1'),('ln_2','layer_norm2'),('mlp.c_fc','mlp.fc1'),('mlp.c_proj','mlp.fc2')]:s[a+x+'.'+param]=hs[b+y+'.'+param]
    model.load_state_dict(s,strict=True)
    records=json.loads((ROOT/'human_manifest.json').read_text())['records'][:30];meta=json.loads((ROOT/'runs/clip-human/metadata.json').read_text());saved=np.load(ROOT/'runs/clip-human/embeddings.npz')
    ix={n:i for i,n in enumerate(meta['media_order'])};preprocess=clip.clip._transform(224)
    images=torch.stack([preprocess(Image.open(ROOT/'human_images'/r['filename'])) for r in records])
    texts=clip.tokenize(['A photo depicts '+r['caption'] for r in records],truncate=True)
    with torch.inference_mode():
        a=model.encode_image(images).numpy();b=model.encode_text(texts).numpy()
    a/=np.linalg.norm(a,axis=1,keepdims=True);b/=np.linalg.norm(b,axis=1,keepdims=True)
    olda=saved['media'][[ix[r['filename']] for r in records]];oldb=saved['CLIPScore_candidate'][:len(records)]
    errors={'max_image_embedding_abs_error':float(np.max(abs(a-olda))),'max_text_embedding_abs_error':float(np.max(abs(b-oldb))),
      'max_score_abs_error':float(np.max(abs(2.5*np.maximum(np.sum(a*b,1),0)-2.5*np.maximum(np.sum(olda*oldb,1),0))))}
    result={'n_caption_pairs':len(records),'n_images':len({r['filename'] for r in records}),'weight_loading':'same verified HF weights mapped into original OpenAI architecture, strict state_dict match','errors':errors,'passed':all(v<1e-4 for v in errors.values())}
    (ROOT/'official_clip_verification.json').write_text(json.dumps(result,indent=2));print(result);assert result['passed']

if __name__=='__main__':main()
