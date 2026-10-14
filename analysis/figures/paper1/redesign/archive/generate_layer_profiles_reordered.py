from pathlib import Path
import os,json
os.environ['MPLCONFIGDIR']='/tmp/elife-mpl-cache'
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parent
layers=pd.read_csv(B/'data/layers.csv')
MC=json.loads((B/'model_palette.json').read_text())
MODELS=['Gemma-2-2B','Gemma-2-9B','Gemma-2-27B','Phi-4','Llama-3.2-3B','Llama-3.3-70B','DeepSeek-R1-70B','Qwen2.5-72B']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
 'axes.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
 'axes.spines.right':False,'axes.edgecolor':'#91999f','text.color':'#20303b',
 'axes.labelcolor':'#20303b','xtick.color':'#425360','ytick.color':'#425360'})
fig,axs=plt.subplots(2,4,figsize=(11,5.8),sharex=True,sharey=True);fig.subplots_adjust(left=.08,right=.98,bottom=.17,top=.83,hspace=.42,wspace=.16)
peaks=[]
for ax,m in zip(axs.flat,MODELS):
 for other in MODELS:
  g=layers[layers.model.eq(other)].sort_values('layer');ax.plot(g.layer/g.layer.max(),g.similarity,color='#dfe4e8',lw=.7,zorder=1)
 g=layers[layers.model.eq(m)].sort_values('layer');x=g.layer/g.layer.max();ax.plot(x,g.similarity,color=MC[m],lw=2,zorder=3)
 peak=g.loc[g.similarity.idxmax()];dep=peak.layer/g.layer.max();peaks.append({'model':m,'layer':int(peak.layer),'normalized_depth':dep,'r':peak.similarity})
 ax.scatter(dep,peak.similarity,s=35,color=MC[m],edgecolors='white',zorder=4);ax.axvline(dep,color=MC[m],ls=':',lw=.8)
 ax.set_title(m.replace('DeepSeek-R1-70B','DS-R1-Llama-70B'),loc='left',fontsize=10);ax.text(.97,.05,f'Peak: {dep:.0%} depth',transform=ax.transAxes,ha='right',fontsize=8)
 ax.set(xlim=(0,1),ylim=(-.025,.82),xticks=[0,.5,1]);ax.grid(axis='y',color='#edf0f2')
fig.supxlabel('Relative layer depth',y=.09);fig.supylabel('Pearson r with task-design RDM',x=.01)

out=B/'no_titles/05_layer_profiles_no_title'
fig.savefig(out.with_suffix('.pdf'),bbox_inches='tight',facecolor='white',dpi=600)
fig.savefig(out.with_suffix('.png'),bbox_inches='tight',facecolor='white',dpi=180)
plt.close(fig)
