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
f=plt.figure(figsize=(14,11.5),facecolor='white')
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
label('A','From gaze fixations to representational similarity',.04,.897)
a=f.add_axes([.04,.714,.92,.151]);a.set(xlim=(0,1),ylim=(0,1));a.axis('off')
im=f.add_axes([.045,.757,.205,.09]);im.imshow(Image.open(B.parent/'trial_example2.png'));im.axis('off')
a.text(.005,.15,'Sequence fixations ≥100 ms',fontsize=9,fontweight='bold')
a.text(.005,.025,'8 patterns · 400 task items',fontsize=9,color=muted)
# Miniature waveforms are explanatory schematics, not measured EEG.
for x,w,title in [(.26,.19,'FIXATION EPOCHS'),(.49,.18,'TRIAL FRP'),(.71,.12,'PATTERN RDM'),(.875,.11,'RSA')]:
 a.add_patch(FancyBboxPatch((x,.23),w,.72,boxstyle='round,pad=0.01,rounding_size=0.025',facecolor='#f3f6f8',edgecolor='#dce4e9'))
 a.text(x+w/2,.84,title,ha='center',fontsize=8.5,fontweight='bold',color=cc['frp'])
for x1,x2 in [(.22,.25),(.455,.48),(.675,.70),(.835,.865)]:arrow(a,(x1,.55),(x2,.55))
t=np.linspace(0,1,120)
wave=np.exp(-((t-.25)/.13)**2)-.8*np.exp(-((t-.52)/.18)**2)+.3*np.exp(-((t-.82)/.13)**2)
for i in range(3):
 a.plot(.28+t*.15,.39+i*.11+.048*wave,color=cc['frp'],alpha=.4+i*.25,lw=1.2)
a.text(.355,.27,'0–600 ms · no baseline',ha='center',fontsize=8)
a.plot(.51+t*.14,.53+.12*wave,color=cc['frp'],lw=2)
a.text(.58,.29,'Average within trial',ha='center',fontsize=8)
# Symmetric correlation-distance example derived from schematic pattern features.
v=np.array([[np.sin(k*.65+j*.6) for j in range(12)] for k in range(8)])
d=1-np.corrcoef(v)
mini=f.add_axes([.705,.773,.07,.064]);mini.imshow(d,cmap='Blues',vmin=0,vmax=2);mini.axis('off')
a.text(.77,.29,'1 − Pearson r',ha='center',fontsize=8)
a.text(.935,.60,'EEG ↔ LLM',ha='center',fontsize=8.5,fontweight='bold')
a.text(.935,.42,'28 pattern pairs',ha='center',fontsize=8)
a.text(.935,.29,'Pearson r',ha='center',fontsize=8)
a.text(.27,.045,'Regional traces → pattern means / session → session RDMs → participant → group',fontsize=8.5,color=ink)
f.text(.96,.699,'Schematic waveforms and RDM; LLM layers selected using task structure.',ha='right',fontsize=8,color=muted)
f.lines.append(Line2D([.04,.96],[.69,.69],transform=f.transFigure,color='#dce4e9',lw=1))

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
f.legend([legend_handles[m] for m in legend_order],legend_order,loc='upper left',bbox_to_anchor=(.062,.643),ncol=3,frameon=False,fontsize=8.3,columnspacing=1.2,handlelength=1.8,labelspacing=.55)
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
f.lines.append(Line2D([.04,.96],[.321,.321],transform=f.transFigure,color='#dce4e9',lw=1))

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

# E: spatial context keeps heterogeneous comparison conditions visible.
label('E','Regional context matters',.55,.304)
ax=f.add_axes([.655,.09,.26,.184])
im=ax.imshow(means.values,cmap='RdBu_r',vmin=-1,vmax=1,aspect='auto')
for i in range(5):
 for j in range(3):
  v=means.iloc[i,j];ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=11,color='white' if abs(v)>.55 else ink)
ax.set(yticks=range(5),yticklabels=[r.title() for r in regions],xticks=range(3),xticklabels=['FRP','Response','Rest']);ax.tick_params(length=0,pad=7);ax.spines[:].set_visible(False)
cax=f.add_axes([.93,.09,.012,.184]);cb=f.colorbar(im,cax=cax,ticks=[-1,0,1]);cb.set_label('Mean RSA (r)',fontsize=9);cb.ax.tick_params(labelsize=8)
f.text(.55,.043,'Means across eight models; color describes effect size, not significance.',fontsize=8.3,color=muted)

out=O/'main_experiment_overview_v2_frp_accuracy'
f.savefig(out.with_suffix('.pdf'),bbox_inches='tight',dpi=600,facecolor='white')
f.savefig(out.with_suffix('.png'),bbox_inches='tight',dpi=220,facecolor='white')
plt.close(f)
(O/'v2_caption_and_data_notes.txt').write_text("""Overview variant 2. A: Task example followed by schematic fixation epochs (0–600 ms, without baseline correction), trial FRPs, pattern RDMs and EEG–LLM RSA. Sequence fixations must last at least 100 ms. Epochs are averaged within trial; regional time courses are averaged within pattern and session before session RDMs are formed. Session RDMs are averaged within participant, then participants receive equal weight in the group RDM. Compare the 28 unique off-diagonal entries with task-selected model RDMs using Pearson r. Waveforms and the illustrated RDM are schematic, not measurements.
B: Accuracy by pattern for all eight models and the equal-participant human means (N=25), from model_pattern_accuracy.csv and human_pattern_summary.csv. Lines connect categorical pattern types, not a time series. Human shading shows marginal 95% percentile bootstrap confidence intervals for the equally weighted participant mean (10,000 resamples within each pattern; seed 20260930). The band represents uncertainty in the mean, not the spread of individual participant accuracies. Model lines have no uncertainty intervals. Uniform four-choice baseline =25%.
C–E: Unchanged from the first overview: task-design layer RSA; matched frontal EEG–LLM RSA; and regional RSA means. Matched EEG N=24, 96 sessions. D's pattern-label p-values test alignment, not differences between conditions. Colors follow model_palette.json. Human means use equal participant weighting (overall 82.47%).
""")
print(out.with_suffix('.pdf'))
