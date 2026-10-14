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
label('A','One task, two routes to representational geometry',.04,.897)
a=f.add_axes([.04,.714,.92,.151]);a.set(xlim=(0,1),ylim=(0,1));a.axis('off')
# Actual task image remains separate from the explanatory pipeline.
im=f.add_axes([.045,.731,.245,.124]);im.imshow(Image.open(B.parent/'trial_example2.png'));im.axis('off')
f.text(.05,.722,'8 pattern types  |  400 task items',fontsize=10,fontweight='bold')
box(a,.32,.56,.29,.40,'HUMANS','25 participants: behavior\n24 matched participants: EEG',cc['frp'])
box(a,.32,.015,.29,.40,'LANGUAGE MODELS','8 models; icon-name sequences\nTask-selected internal layer',colors['DeepSeek-R1-70B'])
arrow(a,(.27,.55),(.31,.75));arrow(a,(.27,.45),(.31,.23))
box(a,.72,.10,.27,.8,'COMPARE PATTERN GEOMETRY','8-pattern RDMs\n28 unique pattern pairs\nPearson RSA',ink)
arrow(a,(.62,.75),(.71,.64));arrow(a,(.62,.22),(.71,.35))
f.lines.append(Line2D([.04,.96],[.69,.69],transform=f.transFigure,color='#dce4e9',lw=1))

# B: paired metric columns with a common order, using equal-participant human reference.
label('B','Behavior: accuracy and human-like difficulty',.04,.669)
f.text(.04,.63,'82.5%',fontsize=23,fontweight='bold',color=ink)
f.text(.145,.631,'human mean',fontsize=10,color=muted)
f.text(.275,.63,'40.6%',fontsize=23,fontweight='bold',color=muted)
f.text(.377,.631,'model mean',fontsize=10,color=muted)
order=beh.sort_values('accuracy',ascending=False).index.tolist()
ax1=f.add_axes([.175,.389,.137,.216]);ax2=f.add_axes([.372,.389,.137,.216],sharey=ax1)
for j,(ax,col,mult) in enumerate([(ax1,'accuracy',100),(ax2,'original_profile_r',1)]):
 for i,m in enumerate(order):
  v=beh.loc[m,col]*mult
  ax.hlines(i,0,v,color=colors[m],alpha=.24,lw=2)
  ax.scatter(v,i,s=35,color=colors[m],zorder=3)
  ax.annotate(f'{v:.1f}' if j==0 else f'{v:.2f}',(v,i),xytext=(5,0),textcoords='offset points',va='center',fontsize=8.3,color='black')
 ax.set(ylim=(7.65,-.65),yticks=range(8),xlim=(0,105 if j==0 else 1.03),xticks=[0,50,100] if j==0 else [0,.5,1],xlabel='Accuracy (%)' if j==0 else 'Human-profile r')
 ax.tick_params(axis='y',length=0);ax.grid(axis='x',color='#edf0f3',lw=.7);ax.set_axisbelow(True)
ax1.set_yticklabels(order,fontsize=8.5)
for t,m in zip(ax1.get_yticklabels(),order):t.set_color(colors[m]);t.set_fontweight('bold')
plt.setp(ax2.get_yticklabels(),visible=False)
ax1.axvline(25,color='#adb6bc',ls=':',lw=1)
f.text(.04,.343,'Profile correlations match the rank plot; r across eight patterns.',fontsize=8.3,color=muted)

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

out=O/'main_experiment_overview'
f.savefig(out.with_suffix('.pdf'),bbox_inches='tight',dpi=600,facecolor='white')
f.savefig(out.with_suffix('.png'),bbox_inches='tight',dpi=220,facecolor='white')
plt.close(f)
caption='''Experiment overview - draft main figure.
A. Participants and eight language models solve abstract sequence-completion problems spanning eight pattern types and 400 task items. Human EEG and task-selected model representations yield eight-pattern RDMs compared over 28 unique pairs. The stimulus image is the actual task example; the arrows and boxes summarize the analysis, not its exact temporal order. Behavioral N=25; the matched EEG analysis includes 24 participants and 96 complete participant-session cells.
B. Model accuracy and human-profile correlation, preserving the exact values used in the currently accepted family-color rank plot. Human overall mean is 82.47%; the mean across eight model accuracies is 40.59%. The human overall accuracy averages participants equally. Profile correlations retain the older manuscript/rank-plot definition, rather than the separately recomputed equal-participant sensitivity version (03D). These behavioral estimands still need to be reconciled before publication; this overview introduces no new choice of profile-correlation values. Dotted accuracy line: uniform four-choice baseline. No uncertainty intervals are shown here.
C. Layer-wise Pearson correlation with the binary task-design RDM. Dots mark task-optimal layers, selected independently of human EEG. Colors identify the same models as panel B. Layer depth uses the source convention, index divided by maximum index.
D. Frontal LLM-human RSA from the matched analysis: each point is one model, horizontal segments and printed values show arithmetic means. Points do not represent independent human participants. Exact one-sided frontal FRP pattern-label permutation p=.010; Holm-adjusted across the three frontal EEG conditions p=.030, as currently reported in generated_hierarchical_results.tex. These p-values test pattern-identity alignment, not a difference between conditions.
E. Mean RSA across models in five regional views, using the same matched dataset as D. This preserves the temporal region's positive rest association and the negative rest estimates in central/occipital regions. Color is descriptive and does not imply cellwise significance.
This is a proposed summary figure, not a replacement of any manuscript figure. Participant reliability, individual-participant RSA, and gaze/timing diagnostics remain supporting analyses in the full figure set.
'''
(O/'caption_and_data_notes.txt').write_text(caption)
(O/'provenance.json').write_text(json.dumps({'inputs':['../data/behavior_profile_comparison.csv','../data/matched_rsa.csv','../data/layers.csv','../data/selected_layers.csv','../model_palette.json','../../trial_example2.png'],'behavior_r_column':'original_profile_r','EEG_dataset':'matched 24 participants, 96 complete sessions','design':'five-panel narrative overview'},indent=2))
print(out.with_suffix('.pdf'))
