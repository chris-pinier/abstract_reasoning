from pathlib import Path
import os,json
os.environ['MPLCONFIGDIR']='/tmp/elife-mpl-cache'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#a5adb3','text.color':'#20303b','axes.labelcolor':'#20303b','xtick.color':'#425360','ytick.color':'#425360'})
MODELS=['Phi-4','Gemma-2-2B','Gemma-2-9B','Gemma-2-27B','Llama-3.2-3B','Llama-3.3-70B','Qwen2.5-72B','DeepSeek-R1-70B']
MC=json.loads((OUT/'model_palette.json').read_text())
CONDS=['frp','response','rest'];CC=dict(zip(CONDS,['#0072B2','#D55E00','#009E73']))
REGIONS=['frontal','temporal','central','occipital','parietal']
rsa=pd.read_csv(OUT/'data/matched_rsa.csv');beh=pd.read_csv(OUT/'data/behavior_profile_comparison.csv',index_col=0)
outputs=[]
def finish(fig,name,note):
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight')
 fig.savefig(OUT/(name+'.png'),bbox_inches='tight',dpi=180)
 outputs.append({'name':name,'note':note});plt.close(fig)
def foot(fig,text):fig.text(.025,.018,text,fontsize=8,color='#586773')

means=rsa.groupby(['channel_group','eeg_type']).rsa_similarity.mean().unstack().reindex(index=REGIONS,columns=CONDS)
values=np.vstack([means.values,means.mean().values]);x=np.array([0,1,2,3,4,5.5]);width=.23
fig,ax=plt.subplots(figsize=(10,5.5));fig.subplots_adjust(left=.09,right=.98,top=.79,bottom=.19)
ax.axvspan(4.9,6.08,color='#f2f5f7',zorder=0)
for j,c in enumerate(CONDS):
 bars=ax.bar(x+(j-1)*width,values[:,j],width=width*.9,color=CC[c],label=c.upper() if c=='frp' else c.title(),zorder=3)
 for b,v in zip(bars,values[:,j]):ax.text(b.get_x()+b.get_width()/2,v+(.018 if v>=0 else -.02),f'{v:.2f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=8,color=CC[c])
ax.axhline(0,color='#697780',lw=.9);ax.set(xticks=x,xticklabels=['Frontal *','Temporal','Central','Occipital','Parietal','Regional\nmean'],ylim=(-.30,.75),ylabel='Mean LLM-human RSA (Pearson r)',xlim=(-.6,6.1))
ax.grid(axis='y',color='#e5eaee',lw=.7);ax.set_axisbelow(True)
fig.suptitle('06  |  Neural correspondence by region and EEG condition',x=.04,ha='left',fontsize=15,fontweight='bold')
fig.legend(handles=[Patch(color=CC[c],label=c.upper() if c=='frp' else c.title()) for c in CONDS],loc='upper left',bbox_to_anchor=(.085,.905),ncol=3,frameon=False)
foot(fig,'Matched dataset. Bars average eight models; no uncertainty intervals are implied. * Primary region.')
finish(fig,'06b_matched_rsa_grouped_bars','Same cell values as 06_matched_rsa_heatmap. Bars group the three EEG conditions within each regional category. Regional mean is separated and shaded. No standard errors or confidence intervals have been added.')

order=beh.sort_values('accuracy',ascending=False).index.tolist()
fig,ax=plt.subplots(figsize=(9,8));fig.subplots_adjust(left=.25,right=.96,top=.84,bottom=.17)
for i,m in enumerate(order):
 if i%2==0:ax.axhspan(i-.48,i+.48,color='#f3f5f7',zorder=0)
 for j,c in enumerate(CONDS):
  g=rsa[rsa.label.eq(m)&rsa.eeg_type.eq(c)].set_index('channel_group').reindex(REGIONS)
  yy=i+(j-1)*.26
  for k,reg in enumerate(REGIONS):
   ax.scatter(g.loc[reg,'rsa_similarity'],yy+(k-2)*.026,s=38 if reg=='frontal' else 23,marker='D' if reg=='frontal' else 'o',color=CC[c],alpha=1 if reg=='frontal' else .38,zorder=4 if reg=='frontal' else 3)
  ax.scatter(g.rsa_similarity.mean(),yy,marker='s',s=55,facecolors='white',edgecolors=CC[c],lw=1.6,zorder=5)
ax.axvline(0,color='#7b8891',ls='--',lw=.9)
ax.set(yticks=range(8),yticklabels=order,ylim=(7.6,-.6),xlim=(-.35,.76),xticks=[-.3,-.15,0,.15,.3,.45,.6,.75],xlabel='LLM-human RSA (Pearson r)')
ax.grid(axis='x',color='#e1e6eb',lw=.6);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0,pad=10)
fig.suptitle('07  |  Model correspondence across EEG conditions',x=.04,ha='left',fontsize=15,fontweight='bold')
fig.text(.04,.91,'Within each model: FRP above response above rest; regional values retained',fontsize=9,color='#586773')
fig.legend(handles=[Patch(color=CC[c],label=c.upper() if c=='frp' else c.title()) for c in CONDS],loc='upper center',bbox_to_anchor=(.6,.897),ncol=3,frameon=False,fontsize=9)
fig.legend(handles=[Line2D([],[],ls='',marker='D',color='#526472',label='Frontal (primary)'),Line2D([],[],ls='',marker='o',color='#aab5bd',label='Other regions'),Line2D([],[],ls='',marker='s',markerfacecolor='white',markeredgecolor='#526472',label='Regional mean')],loc='lower center',bbox_to_anchor=(.58,.077),ncol=3,frameon=False,fontsize=8.5)
foot(fig,'Same matched dataset as Figures 6 and 7. Regional points are descriptive values, not uncertainty intervals.')
finish(fig,'07b_model_rsa_combined','Single shared axis, model categories, three vertically offset conditions in consistent EEG colors. Each condition retains all five regional points, the frontal diamond and an open square for the mean. Values unchanged from 07_model_rsa_regions.')

rank1=beh.accuracy.rank(ascending=False);rank2=beh.original_profile_r.rank(ascending=False)
fig,ax=plt.subplots(figsize=(9,6));fig.subplots_adjust(left=.36,right=.86,top=.79,bottom=.15)
for y in range(1,9):ax.axhline(y,color='#edf0f2',lw=.7,zorder=0)
for m in order:
 a,b=rank1[m],rank2[m];color=MC[m]
 ax.plot([0,1],[a,b],color=color,lw=2.3,alpha=.85,zorder=2)
 ax.scatter([0,1],[a,b],s=175,facecolors='white',edgecolors=color,lw=1.8,zorder=3)
 for xx,yy in [(0,a),(1,b)]:ax.text(xx,yy,f'{int(yy)}',ha='center',va='center',fontsize=8,color='#20303b',zorder=4)
 ax.text(-.065,a-.04,m,ha='right',va='center',fontsize=10,color=color,fontweight='bold')
 ax.text(-.065,a+.24,f'{beh.loc[m,"accuracy"]*100:.1f}%',ha='right',va='center',fontsize=8,color='#687680')
 ax.text(1.065,b,f'r = {beh.loc[m,"original_profile_r"]:.3f}',ha='left',va='center',fontsize=9,color=color)
ax.set(xlim=(-.025,1.025),ylim=(8.55,.4),xticks=[],yticks=[]);ax.spines[:].set_visible(False)
ax.text(0,1.06,'Accuracy',transform=ax.transAxes,ha='center',fontsize=11,fontweight='bold')
ax.text(1,1.06,'Human-profile similarity',transform=ax.transAxes,ha='center',fontsize=11,fontweight='bold')
fig.suptitle('03  |  High accuracy does not guarantee a human-like profile',x=.04,ha='left',fontsize=14,fontweight='bold')
fig.text(.04,.90,'Follow each colored line from left to right. Circled numbers show rank: 1 = highest.',fontsize=9,color='#586773')
foot(fig,'Current manuscript values. Ranks hide effect sizes; Phi-4 and DeepSeek are almost tied (both r = 0.688).')
finish(fig,'03e_accuracy_profile_rank_clean','Original Figure 3C values and ranks preserved. Each model appears once on the left, colored to match its line. Endpoint circles display ranks; left percentages and right correlations retain actual values. The near-tie between Phi-4 and DeepSeek is explicitly noted.')
(OUT/'variants_notes.json').write_text(json.dumps(outputs,indent=2))
print('\n'.join(o['name'] for o in outputs))
