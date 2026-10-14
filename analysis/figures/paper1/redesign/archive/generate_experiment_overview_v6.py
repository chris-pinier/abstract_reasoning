from pathlib import Path
import os,json
os.environ['MPLCONFIGDIR']='/tmp/elife-mpl-cache'
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.lines import Line2D
from PIL import Image
B=Path(__file__).resolve().parent
O=B/'experiment_overview';O.mkdir(exist_ok=True)
colors=json.loads((B/'model_palette.json').read_text())
beh=pd.read_csv(B/'data/behavior_profile_comparison.csv',index_col=0)
rsa=pd.read_csv(B/'data/matched_rsa.csv')
layers=pd.read_csv(B/'data/layers.csv')
peaks=pd.read_csv(B/'data/selected_layers.csv')
regions=['frontal','temporal','central','occipital','parietal'];conditions=['frp','response','rest']
cc={'frp':'#0072B2','response':'#D55E00','rest':'#009E73'}
ink='#20303b';muted='#607480'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'axes.titlesize':12,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#abb5bd','text.color':ink,'axes.labelcolor':ink,'xtick.color':muted,'ytick.color':muted,'pdf.fonttype':42})
f=plt.figure(figsize=(14,14),facecolor='white')
f.text(.04,.97,'Abstract pattern reasoning in humans and language models',fontsize=21,fontweight='bold',va='top')
f.text(.04,.929,'A common task links behavior, model representations and fixation-locked EEG',fontsize=12,color=muted)
def label(letter,title,x,y):
 f.text(x,y,letter,fontsize=16,fontweight='bold',color='#0072B2',va='top')
 f.text(x+.027,y-.001,title,fontsize=13,fontweight='bold',va='top')
def box(ax,x,y,w,h,title,body,color):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.025',facecolor='#f3f6f8',edgecolor='#dce4e9',lw=1))
 ax.text(x+.025,y+h-.07,title,fontsize=11,fontweight='bold',color=color,va='top')
 ax.text(x+.025,y+h-.20,body,fontsize=9,color=ink,va='top',linespacing=1.15)
def arrow(ax,p,q,color='#8296a3'):
 ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=12,lw=1.4,color=color,connectionstyle='arc3,rad=0'))
label('A','From human and model responses to representational similarity',.04,.897)
im=f.add_axes([.04,.736,.225,.137]);im.imshow(Image.open(B.parents[1]/'Versions/eLife/figures/trial_example2.png'));im.axis('off')
# All EEG traces and matrices below are schematic, not measured data.
from matplotlib.patches import Rectangle
flow=f.add_axes([0,0,1,1],zorder=-1);flow.set(xlim=(0,1),ylim=(0,1));flow.axis('off')
starts=[.10,.85,1.10];cols=['#0072B2','#008778','#69a8d0']
t=np.linspace(0,1.9,1000)
y=.20*np.sin(2*np.pi*7*t)+.13*np.sin(2*np.pi*13*t+.7)+.1*np.cos(2*np.pi*3*t)
xs=[.30,.452,.605];w=.12
for x,title in zip(xs,['1  Mark fixations','2  Align epochs','3  Average within trial']):
 f.text(x,.872,title,fontsize=9,fontweight='bold')
def trace_axes(x):
 ax=f.add_axes([x,.775,w,.075]);ax.set_yticks([]);ax.spines[['left','right','top']].set_visible(False);ax.tick_params(axis='x',labelsize=7,length=2);return ax
ax=trace_axes(xs[0]);ax.plot(t,y,color=ink,lw=.8)
for i,(st,col) in enumerate(zip(starts,cols)):
 ax.add_patch(Rectangle((st,.53+i*.18),.6,.12,color=col,alpha=.28,lw=0))
 ax.axvline(st,color=col,ls=':',lw=.8)
ax.set(xlim=(0,1.9),ylim=(-.55,1.12),xticks=[0,.6,1.2,1.8]);ax.set_xlabel('Time in trial (s)',fontsize=7,labelpad=2)
epoch_t=np.linspace(0,.6,220)
epochs=np.array([np.interp(st+epoch_t,t,y) for st in starts])
ax=trace_axes(xs[1])
for i,(ep,col) in enumerate(zip(epochs,cols)):ax.plot(epoch_t*1000,ep+(2-i)*.65,color=col,lw=1)
ax.axvline(0,color=muted,lw=.7);ax.set(xlim=(-10,610),xticks=[0,600]);ax.set_xlabel('From fixation onset (ms)',fontsize=7,labelpad=2)
ax=trace_axes(xs[2])
for ep in epochs:ax.plot(epoch_t*1000,ep,color='#adcde0',lw=.65,alpha=.8)
ax.plot(epoch_t*1000,epochs.mean(axis=0),color=cc['frp'],lw=2)
ax.set(xlim=(0,600),xticks=[0,600],ylim=(-.6,.6));ax.set_xlabel('From fixation onset (ms)',fontsize=7,labelpad=2)
for p,q in [((.269,.815),(.293,.815)),((.425,.815),(.446,.815)),((.578,.815),(.599,.815))]:arrow(flow,p,q)
f.text(.30,.744,'Sequence fixations ≥100 ms  •  0–600 ms EEG windows may overlap',fontsize=8,color=muted)
f.text(.30,.730,'Pattern means → session RDMs → participant RDMs → group RDM',fontsize=8,color=ink)
# Parallel LLM construction row. Synthetic activations illustrate the procedure.
llm_color='#746044'
f.text(.04,.712,'Instructions + one solved example',fontsize=8.5,color=muted)
f.text(.04,.688,'Target sequence',fontsize=9,fontweight='bold',color=cc['frp'])
flow.add_patch(FancyBboxPatch((.04,.651),.224,.025,boxstyle='round,pad=.003',facecolor='#dceef8',edgecolor='none'))
f.text(.045,.660,'star star star guitar star star star ?',fontsize=7.8)
f.text(.04,.632,'Options: truck / star / cube / guitar',fontsize=8)
f.text(.04,.611,'400 independent trial prompts',fontsize=7.5,color=muted)
for x,title in zip(xs,['1  Read internal activations','2  Build trial representations','3  Construct pattern means']):
 f.text(x,.712,title,fontsize=8.5,fontweight='bold')
rng=np.random.default_rng(13)
def feature_grid(rect,arr):
 aa=f.add_axes(rect);aa.imshow(arr,cmap='BrBG',aspect='auto');aa.set_xticks([]);aa.set_yticks([])
 for sp in aa.spines.values():sp.set_visible(False)
 return aa
# Transformer blocks and corresponding target-token hidden states.
for k in range(3):
 yy=.678-k*.022
 flow.add_patch(FancyBboxPatch((.30,yy),.045,.016,boxstyle='round,pad=.001',facecolor='#dceef8',edgecolor=cc['frp'],lw=.5))
 f.text(.3225,yy+.005,'Block '+str(k+1) if k<2 else 'Block L',ha='center',fontsize=6)
 arrow(flow,(.348,yy+.008),(.364,yy+.008),color=llm_color)
 feature_grid([.367,yy,.052,.016],rng.normal(size=(3,15)))
f.text(.30,.620,'Target-sequence tokens at each layer',fontsize=7,color=muted)
# Retain token positions by flattening the token-by-feature array.
feature_grid([.454,.646,.040,.043],rng.normal(size=(8,18)))
arrow(flow,(.497,.668),(.522,.668),color=llm_color)
feature_grid([.526,.661,.045,.014],rng.normal(size=(1,144)))
f.text(.454,.633,'Tokens × features',fontsize=6.5)
f.text(.531,.641,'Flatten',fontsize=7,fontweight='bold')
f.text(.452,.611,'One vector / trial / layer',fontsize=7.5,color=muted)
features=np.array([[np.sin(k*.65+j*.6) for j in range(12)] for k in range(8)])
aa=f.add_axes([.614,.632,.097,.06]);aa.imshow(features,cmap='BrBG',aspect='auto');aa.set_xticks([]);aa.set_yticks([])
aa.set_xlabel('Selected-layer features',fontsize=7,labelpad=2);aa.set_ylabel('8 patterns',fontsize=7,labelpad=2)
f.text(.605,.611,'Average trials at the selected layer',fontsize=7.5,color=muted)
for p,q in [((.269,.662),(.293,.662)),((.425,.662),(.446,.662)),((.578,.662),(.599,.662))]:arrow(flow,p,q,color=llm_color)
rdms=[1-np.corrcoef(features),1-np.corrcoef(features+.3*np.cos(np.arange(8)[:,None]+np.arange(12)[None,:]))]
for yy,title,rdm in zip([.795,.633],['Human group RDM','LLM RDM'],rdms):
 f.text(.762,yy+.062,title,fontsize=8,fontweight='bold')
 mat=f.add_axes([.773,yy,.057,.057]);mat.imshow(rdm,cmap='Blues',vmin=0,vmax=2);mat.axis('off')
 arrow(flow,(.835,yy+.028),(.863,.742))
flow.add_patch(FancyBboxPatch((.867,.712),.086,.062,boxstyle='round,pad=0.004',facecolor='#f3f6f8',edgecolor='#dce4e9'))
f.text(.91,.751,'RSA',ha='center',fontsize=12,fontweight='bold',color=cc['frp'])
f.text(.91,.731,'Pearson r',ha='center',fontsize=9)
f.text(.91,.683,'28 matched\npattern pairs',ha='center',fontsize=8,color=muted,linespacing=1.3)
arrow(flow,(.726,.823),(.766,.823))
arrow(flow,(.726,.662),(.766,.662),color=llm_color)
f.text(.04,.594,'Traces, activations and matrices are schematic. RDM distance = 1 − r; RSA compares corresponding upper-triangle entries.',fontsize=8,color=muted)
f.lines.append(Line2D([.04,.96],[.58,.58],transform=f.transFigure,color='#dce4e9',lw=1))

# Preserve the original results panels and their data, with space for the new row.
original_text=f.text
original_axes=f.add_axes
def lower_text(x,y,*args,**kwargs): return original_text(x,y*.84,*args,**kwargs)
def lower_axes(rect,*args,**kwargs):
 x,y,w,h=rect
 return original_axes([x,y*.84,w,h*.84],*args,**kwargs)
f.text=lower_text
f.add_axes=lower_axes
# B: accuracy trajectories over categorical pattern types.
label('B','Accuracy across abstract patterns',.04,.669)
order=beh.sort_values('accuracy',ascending=False).index.tolist()
mp=pd.read_csv(B/'data/model_pattern_accuracy.csv',index_col=0)
hp=pd.read_csv(B/'data/human_pattern_summary.csv').set_index('pattern').reindex(mp.index)
ax=f.add_axes([.087,.410,.417,.170])
x=np.arange(len(mp))
for m in order:
 ax.plot(x,100*mp[m],color=colors[m],marker='o',markersize=3,lw=1.5,label=m)
ax.fill_between(x,100*hp['ci_low'],100*hp['ci_high'],color=ink,alpha=.15,lw=0,zorder=1)
ax.plot(x,100*hp['mean'],color=ink,marker='D',markersize=4,lw=2.4,label='Humans',zorder=6)
ax.axhline(25,color='#9aa8b0',ls=':',lw=1)
ax.set(ylim=(-3,105),xlim=(-.2,7.2),yticks=[0,25,50,75,100],ylabel='Accuracy (%)',xticks=x,xticklabels=mp.index)
ax.tick_params(axis='x',labelsize=8,rotation=35)
for tick in ax.get_xticklabels():tick.set_ha('right')
ax.grid(axis='y',color='#edf0f3',lw=.7)
h,l=ax.get_legend_handles_labels()
# Matplotlib fills columns vertically: reference/singletons, Gemma, Llama-based.
legend_order=['Humans','Phi-4','Qwen2.5-72B','Gemma-2-2B','Gemma-2-9B','Gemma-2-27B','Llama-3.2-3B','Llama-3.3-70B','DeepSeek-R1-70B']
legend_handles=dict(zip(l,h))
f.legend([legend_handles[m] for m in legend_order],[m.replace('DeepSeek-R1-70B','DS-R1-Llama-70B') for m in legend_order],loc='upper left',bbox_to_anchor=(.062,.643*.84),ncol=3,frameon=False,fontsize=8.3,columnspacing=1.2,handlelength=1.8,labelspacing=.55)
f.text(.04,.343,'Human band: 95% bootstrap CI (N = 25). Dotted line: 25% chance.',fontsize=8.3,color=muted)

# C: model traces with the selected layers marked, no human EEG used in selection.
label('C','Task structure emerges inside the models',.55,.669)
ax=f.add_axes([.585,.389,.375,.222])
for m in order:
 g=layers[layers.model.eq(m)].sort_values('layer');x=g.layer/g.layer.max()
 ax.plot(x,g.similarity,color=colors[m],lw=1.6,alpha=.95)
 p=peaks[peaks.model.eq(m)].iloc[0]
 ax.scatter(p.normalized_depth,p.r,s=32,color=colors[m],edgecolors='white',lw=.7,zorder=4)
ax.set(xlim=(0,1),ylim=(-.03,.83),xticks=[0,.25,.5,.75,1],yticks=[0,.2,.4,.6,.8],xlabel='Relative layer depth',ylabel='Task-design RSA (r)')
ax.grid(axis='y',color='#edf0f3',lw=.7)
f.text(.55,.343,'Dots mark selected layers. Selection uses task structure, independently of EEG.',fontsize=8.3,color=muted)
f.lines.append(Line2D([.04,.96],[.321*.84,.321*.84],transform=f.transFigure,color='#dce4e9',lw=1))

# D: frontal summary and per-model observations from the matched dataset.
label('D','Frontal correspondence is strongest for FRP',.04,.304)
ax=f.add_axes([.095,.103,.36,.155])
means=rsa.groupby(['channel_group','eeg_type']).rsa_similarity.mean().unstack().reindex(index=regions,columns=conditions)
for j,c in enumerate(conditions):
 vals=rsa[rsa.channel_group.eq('frontal')&rsa.eeg_type.eq(c)].set_index('label').reindex(order).rsa_similarity.to_numpy()
 ax.scatter(j+np.linspace(-.14,.14,8),vals,color=cc[c],s=28,alpha=.55,zorder=3)
 ax.plot([j-.2,j+.2],[vals.mean()]*2,color=cc[c],lw=3,zorder=4)
 ax.text(j,vals.max()+.05,f'{vals.mean():.2f}',ha='center',fontsize=15,color=cc[c],fontweight='bold')
ax.axhline(0,color='#8a969f',lw=.7)
ax.set(xlim=(-.5,2.5),ylim=(-.09,.83),xticks=[0,1,2],xticklabels=['FRP','Response','Rest'],ylabel='LLM-human RSA (r)',yticks=[0,.2,.4,.6,.8])
for t,c in zip(ax.get_xticklabels(),conditions):t.set_color(cc[c]);t.set_fontweight('bold')
ax.grid(axis='y',color='#edf0f3',lw=.7)
f.text(.04,.064,'Dots: eight models. Horizontal marks: means. Matched N = 24.',fontsize=8.5,color=muted)
f.text(.04,.043,'Frontal FRP pattern-label test: p = .010; Holm-adjusted p = .030.',fontsize=8.5,color=muted)

# E: grouped bars reproduce 06b's regional and across-region means.
label('E','Regional context matters',.55,.304)
ax=f.add_axes([.595,.105,.365,.145])
values=np.vstack([means.values,means.mean().values])
x=np.array([0,1,2,3,4,5.5]);width=.23
ax.axvspan(4.9,6.08,color='#f2f5f7',zorder=0)
for j,c in enumerate(conditions):
 bars=ax.bar(x+(j-1)*width,values[:,j],width=width*.9,color=cc[c],label='FRP' if c=='frp' else c.title(),zorder=3)
 for bar,value in zip(bars,values[:,j]):
  ax.text(bar.get_x()+bar.get_width()/2,value+(.02 if value>=0 else -.025),f'{value:.2f}',ha='center',va='bottom' if value>=0 else 'top',fontsize=6.5,color=cc[c])
ax.axhline(0,color='#697780',lw=.9)
ax.set(xticks=x,xticklabels=['Frontal','Temporal','Central','Occipital','Parietal','Regional\nmean'],ylim=(-.30,.75),yticks=[-.2,0,.2,.4,.6],ylabel='Mean LLM-human RSA (r)',xlim=(-.6,6.1))
ax.tick_params(axis='x',labelsize=7.5)
ax.tick_params(axis='y',labelsize=8)
ax.yaxis.label.set_size(9)
ax.grid(axis='y',color='#e5eaee',lw=.7);ax.set_axisbelow(True)
ax.legend(loc='lower left',bbox_to_anchor=(-.01,1.03),ncol=3,frameon=False,fontsize=8.5,handlelength=1.6,columnspacing=1.5)
f.text(.55,.043,'Bars average eight models; shaded group: mean across the five regions.',fontsize=8.3,color=muted)

out=O/'main_experiment_overview_v6_frp_llm_construction'
f.savefig(out.with_suffix('.pdf'),bbox_inches='tight',dpi=600,facecolor='white')
f.savefig(out.with_suffix('.png'),bbox_inches='tight',dpi=220,facecolor='white')
plt.close(f)
print(out.with_suffix('.pdf'))
