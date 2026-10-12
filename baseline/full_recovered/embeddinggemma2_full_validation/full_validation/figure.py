"""Publication-style summary; uses measured cluster confidence intervals."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    sugar=json.loads((ROOT/'results.json').read_text())['cohorts']['unseen'];human=json.loads((ROOT/'human_results.json').read_text())['cohorts']['unseen']
    keys=['clip-CLIPScore-512','clip_large-CLIPScore-768','specs-NativeSPECS-512','siglip-SigLIP2-768','gemma-CLIPLiteral-768','gemma-CLIPLiteral-256']
    names=['CLIP B/32','CLIP L/14','SPECS','SigLIP 2 Base','Gemma 768d','Gemma 256d'];colors=['#78909c']*4+['#6753c7','#ab94df'];y=np.arange(len(keys))
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    for ax,data,field,scale,title,xlabel in [(axes[0],sugar,'accuracy',100,f'SugarCrepe: {sugar["n_pairs"]:,} pairs / {sugar["n_images"]:,} new images','Caption-ranking accuracy (%)'),
      (axes[1],human,'kendall_tau_c',1,f'Flickr8k-Expert: {human["n_images"]} new images','Kendall tau-c with individual expert ratings')]:
        vals=np.array([data['metrics'][k][field]*scale for k in keys]);ci=np.array([data['metrics'][k]['ci95'] for k in keys])*scale
        ax.barh(y,vals,color=colors,height=.62);ax.errorbar(vals,y,xerr=np.stack([vals-ci[:,0],ci[:,1]-vals]),fmt='none',ecolor='#253445',capsize=3,lw=1.2)
        ax.set_yticks(y,names);ax.invert_yaxis();ax.set_title(title,fontsize=11,pad=12);ax.set_xlabel(xlabel,fontsize=10);ax.set_xlim(0,100 if scale==100 else 1)
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
        for i,v in enumerate(vals):ax.text(v+(1.5 if scale==100 else .015),i,f'{v:.1f}' if scale==100 else f'{v:.3f}',va='center',fontsize=9)
    fig.suptitle('CLIPScore encoder substitutions: image-disjoint continuation cohorts',fontsize=13)
    fig.savefig(ROOT/'figure_main.png',dpi=200);fig.savefig(ROOT/'figure_main.svg');plt.close(fig)

if __name__=='__main__':main()
