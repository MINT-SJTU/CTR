"""Generate separate, data-derived figures. No synthetic trajectories or error bars."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
D=json.loads((ROOT/'data/experiments.json').read_text())
f=pd.DataFrame(D['e2a_paired_freeze']['rows'])
labels={'L_to_R':r'$L\to R$','R_to_L':r'$R\to L$','Sync':'Sync','Mix3':'Mix-3','Ours':'CTR'}
markers=['o','s','^','D','P'];lines=['-','--',':','-.','-']
for arm in ['L','R']:
    fig,ax=plt.subplots(figsize=(3.8,3.0))
    for (group,label),marker,line in zip(labels.items(),markers,lines):
        s=f[(f.group==group)&(f.frozen_arm==arm)].groupby('freeze_duration_s').delta_wait_s.mean()
        ax.plot(s.index,s.values,marker=marker,linestyle=line,linewidth=1.5,markersize=4.5,label=label)
    other='right' if arm=='L' else 'left'
    ax.set_xlabel('Forced pause duration (s)',fontsize=10)
    ax.set_ylabel('Extra waiting of '+other+' arm (s)',fontsize=9)
    ax.set_title(('Left' if arm=='L' else 'Right')+' arm paused',fontsize=11)
    ax.set_xticks([1,2,4]);ax.set_ylim(0,5.1)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8,frameon=False,loc='upper left')
    fig.tight_layout()
    fig.savefig(ROOT/'figures'/f'freeze_{arm}.pdf',bbox_inches='tight')
    fig.savefig(ROOT/'figures'/f'freeze_{arm}.png',dpi=160,bbox_inches='tight')
    plt.close(fig)
print('Created two separate freeze-response plots from the uploaded results.')
