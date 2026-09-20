#!/usr/bin/env python3
"""Render the article's standalone figures from analysis.json (requires matplotlib)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

out=Path('docs/blog/colts-chiefs-jones-2026')
d=json.loads((out/'analysis.json').read_text());m=d['model']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'figure.facecolor':'#111823',
                    'axes.facecolor':'#111823','text.color':'#e7eef9','axes.labelcolor':'#e7eef9',
                    'xtick.color':'#b8c5d6','ytick.color':'#b8c5d6','axes.edgecolor':'#42526b',
                    'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(10,5.3),layout='constrained')
x=np.linspace(-10,65,1500);y=norm.pdf(x,m['mu'],m['sigma'])
ax.plot(x,y,color='#67dfc3',lw=2.5,label='Published raw model')
ax.fill_between(x,y,where=x<=31.5,color='#67dfc3',alpha=.25)
ax.plot(x,norm.pdf(x,m['base_mean'],m['sigma']),color='#fac77b',lw=2,ls='--',label='Without defense adjustment')
ax.axvline(31.5,color='#e7eef9',ls=':',lw=1.5)
ax.text(32.3,.037,'Under 31.5\nneeds 31 or fewer',fontsize=10)
ax.set(title='Jones passing attempts: the assumption behind the under',xlabel='Passing attempts (continuous Normal approximation)',ylabel='Probability density')
ax.set_xlim(-5,62);ax.set_ylim(0,.047)
ax.text(4,.043,'Shaded area: 67.0% raw probability\nPublished calibration lowers it to 63.6%',fontsize=11)
ax.legend(loc='upper right',bbox_to_anchor=(1,.69),facecolor='#111823',labelcolor='#e7eef9',fontsize=10)
ax.spines[['top','right']].set_visible(False)
fig.savefig(out/'distribution.svg');fig.savefig(out/'distribution.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,4.4),layout='constrained')
sc=d['scenarios'][:3];labels=['Published model','Half defense adjustment','No defense adjustment']
raw=[s['raw']*100 for s in sc];cal=[s['calibrated']*100 for s in sc];ys=np.arange(3)
ax.barh(ys+.16,raw,height=.28,color='#526981',label='Raw Normal probability')
ax.barh(ys-.16,cal,height=.28,color='#67dfc3',label='After calibration')
for i in range(3):
 ax.text(raw[i]+.7,ys[i]+.16,f'{raw[i]:.1f}%',va='center',fontsize=10)
 label = f'{cal[i]:.2f}%' if i == 1 else f'{cal[i]:.1f}%'
 ax.text(cal[i]+.7,ys[i]-.16,label,va='center',fontsize=10)
ax.axvline(d['price']['break_even']*100,color='#fac77b',ls='--',label='52.4% break-even at −110')
ax.set_yticks(ys,labels);ax.invert_yaxis();ax.set_xlim(0,80)
ax.set(title='The calibrated edge depends on the defense adjustment',xlabel='Probability of under 31.5 attempts (%)')
ax.legend(loc='upper center',bbox_to_anchor=(.42,-.18),ncol=1,frameon=False,labelcolor='#e7eef9')
ax.spines[['top','right']].set_visible(False)
fig.savefig(out/'sensitivity.svg');fig.savefig(out/'sensitivity.png',dpi=170);plt.close(fig)
print('Rendered distribution and sensitivity as SVG and PNG.')
