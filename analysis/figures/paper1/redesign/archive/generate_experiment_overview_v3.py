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
label('A','Comparing human and model representations',.04,.897)
# Larger stimulus image, with no explanatory text underneath.
im=f.add_axes([.04,.714,.27,.159]);im.imshow(Image.open(B.parents[1]/'Versions/eLife/figures/trial_example2.png'));im.axis('off')
# The two modalities yield the same eight-pattern geometry before RSA.
a=f.add_axes([.325,.714,.635,.16]);a.set(xlim=(0,1),ylim=(0,1));a.axis('off')
for y,title,body,color in [(.57,'HUMAN FRP','Fixation-locked EEG → trial averages\nRegional pattern means / session\nRDMs: session → participant → group',cc['frp']),(.065,'LLM REPRESENTATIONS','Sequence-token activations\nTask-selected layer (no EEG used)\nAverage activations within pattern',colors['DeepSeek-R1-70B'])]:
 a.text(0,y+.34,title,fontsize=10,fontweight='bold',color=color,va='top')
 a.text(0,y+.21,body,fontsize=8,va='top',linespacing=1.45)
# Synthetic matrices illustrate the operations; no measured values are implied.
features=np.array([[np.sin(k*.65+j*.6) for j in range(12)] for k in range(8)])
rdms=[1-np.corrcoef(features),1-np.corrcoef(features+.3*np.cos(np.arange(8)[:,None]+np.arange(12)[None,:]))]
for i,(y,rdm) in enumerate(zip([.818,.733],rdms)):
 mat=f.add_axes([.565,y,.058,.058]);mat.imshow(rdm,cmap='Blues',vmin=0,vmax=2);mat.axis('off')
 # Outline the upper triangle to identify the entries used for RSA.
 from matplotlib.patches import Polygon
 mat.add_patch(Polygon([(.5,-.5),(7.5,-.5),(7.5,6.5)],closed=True,fill=False,edgecolor='#D55E00',lw=1.2))
 vals=rdm[np.triu_indices(8,1)]
 strip=f.add_axes([.67,y+.014,.115,.025]);strip.imshow(vals[None,:],cmap='Blues',vmin=0,vmax=2,aspect='auto');strip.axis('off')
 a.annotate('',xy=(.53,.77-i*.53),xytext=(.48,.77-i*.53),arrowprops=dict(arrowstyle='-|>',color='#8296a3',lw=1.3))
 a.annotate('',xy=(.81,.49),xytext=(.735,.77-i*.53),arrowprops=dict(arrowstyle='-|>',color='#8296a3',lw=1.3))
f.text(.594,.803,'8 × 8 RDM',ha='center',fontsize=8)
f.text(.594,.718,'8 × 8 RDM',ha='center',fontsize=8)
f.text(.727,.86,'28 unique pattern pairs',ha='center',fontsize=8.3,fontweight='bold')
f.text(.727,.793,'Same pair order',ha='center',fontsize=8,color=muted)
a.add_patch(FancyBboxPatch((.82,.24),.165,.52,boxstyle='round,pad=0.01,rounding_size=0.025',facecolor='#f3f6f8',edgecolor='#dce4e9'))
a.text(.90,.65,'RSA',ha='center',fontsize=13,fontweight='bold',color=cc['frp'])
a.text(.90,.49,'Pearson r',ha='center',fontsize=10)
a.text(.90,.32,'Human ↔ LLM',ha='center',fontsize=8.5)
f.text(.96,.699,'RDM distance = 1 − r. Matrices are schematic; RSA compares corresponding upper-triangle entries.',ha='right',fontsize=8,color=muted)
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

out=O/'main_experiment_overview_v3_grouped_rsa'
f.savefig(out.with_suffix('.pdf'),bbox_inches='tight',dpi=600,facecolor='white')
f.savefig(out.with_suffix('.png'),bbox_inches='tight',dpi=220,facecolor='white')
plt.close(f)
(O/'v3_caption_and_data_notes.txt').write_text("""Overview variant 3. A: Enlarged task example and parallel human/LLM representational pipelines. Human fixation-locked trial averages yield regional pattern means within session, then session RDMs are averaged within participant and across participants. LLM sequence-token activations are averaged within pattern at a layer selected independently of EEG. Both modalities produce 8-by-8 correlation-distance RDMs. Their 28 unique upper-triangular entries are extracted in the same pattern-pair order and compared with Pearson correlation. Matrices and entry strips are schematic, not measured data.
B: Accuracy by pattern for all eight models and the equal-participant human means (N=25), from model_pattern_accuracy.csv and human_pattern_summary.csv. Lines connect categorical pattern types, not a time series. Human shading shows marginal 95% percentile bootstrap confidence intervals for the equally weighted participant mean (10,000 resamples within each pattern; seed 20260930). The band represents uncertainty in the mean, not the spread of individual participant accuracies. Model lines have no uncertainty intervals. Uniform four-choice baseline =25%.
C–D: Unchanged from the previous overview: task-design layer RSA and matched frontal EEG–LLM RSA. E: Grouped bars recreate 06b_matched_rsa_grouped_bars_no_title from the same matched_rsa.csv data. Three EEG conditions are grouped within each of five regions; the shaded final group is the equal-weight mean across regions. Labels show arithmetic means across eight models. No error bars or inferential interpretation are implied. Matched EEG N=24, 96 sessions. D's pattern-label p-values test alignment, not differences between conditions. Colors follow model_palette.json. Human means use equal participant weighting (overall 82.47%).
""")
print(out.with_suffix('.pdf'))
