import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import read

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    r=Path(a.results);u=read(r/'adapter_uncertainty.json')['results'];m=read(r/'all_metrics.json')['cohorts']['unseen900']
    fig,axes=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
    for i,key in enumerate(['historical560','historical_unseen3949']):
        z=u[key];delta=100*z['delta'];lo,hi=[100*x for x in z['paired_image_bootstrap_95_percentile_delta_ci']]
        axes[0].errorbar(delta,i,xerr=[[delta-lo],[hi-delta]],fmt='o',color='#2463a2',capsize=4)
    axes[0].set(yticks=[0,1],yticklabels=['SugarCrepe: 560 images','SugarCrepe: 1,005 images'],xlabel='Adapter − original accuracy (percentage points)',title='Compositional discrimination')
    delta=m['gemma-adapter-ensemble5']['kendall_tau_c_individual']-m['gemma-CLIPLiteral-768-float32']['kendall_tau_c_individual'];lo,hi=u['human_unseen900']['paired_image_bootstrap_95_percentile_tau_c_delta_ci']
    axes[1].errorbar(delta,0,xerr=[[delta-lo],[hi-delta]],fmt='o',color='#b74935',capsize=4)
    axes[1].set(yticks=[0],yticklabels=['Expert: 900 images'],xlabel='Adapter − original Kendall τ-c',title='Human alignment',xlim=(-.16,.025))
    for ax in axes:ax.axvline(0,color='.4',ls='--',lw=1);ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Five-seed adapter ensemble: exploratory historical diagnostics',fontsize=12)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=220);fig.savefig(out.with_suffix('.pdf'));print(out)

if __name__=='__main__':main()
