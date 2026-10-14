from pathlib import Path
import os, json, pickle, warnings, hashlib, shutil
os.environ['MPLCONFIGDIR']='/tmp/elife-mpl-cache'
import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors

OUT=Path(__file__).resolve().parent
FIG=OUT.parent
ROOT=FIG.parents[4]
AN=ROOT/'other/final_analysis-primary_regions'
DRIVE=Path('/Volumes/Realtek 1Tb/PhD Data/experiment1-analysis')
RD=DRIVE/'ANNs/analyzed/RSA-seq_tokens-metric_correlation'
DATA=OUT/'data'; DATA.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
 'axes.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,
 'axes.spines.right':False,'axes.edgecolor':'#91999f','text.color':'#20303b',
 'axes.labelcolor':'#20303b','xtick.color':'#425360','ytick.color':'#425360'})
PATS=['AAABAAAB','ABABCDCD','ABBAABBA','ABBACDDC','ABBCABBC','ABCAABCA','ABCDDCBA','ABCDEEDC']
MODELS=['Phi-4','Gemma-2-2B','Gemma-2-9B','Gemma-2-27B','Llama-3.2-3B','Llama-3.3-70B','Qwen2.5-72B','DeepSeek-R1-70B']
COLORS=[json.loads((OUT/'model_palette.json').read_text())[m] for m in MODELS]
MC=dict(zip(MODELS,COLORS)); HC='#182a38'
REGIONS=['frontal','temporal','central','occipital','parietal']
CONDS=['frp','response','rest']; CC={'frp':'#0072B2','response':'#D55E00','rest':'#009E73'}
outputs=[]; sources=[]
def source(p):
 p=Path(p);sources.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()});return p
def short(s):
 s=str(s).lower()
 if 'deepseek' in s:return MODELS[7]
 if 'phi' in s:return MODELS[0]
 if 'qwen' in s:return MODELS[6]
 for m in MODELS[1:6]:
  if m.lower() in s:return m
 raise ValueError(s)
def save(fig,name,caption):
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',facecolor='white',dpi=600)
 fig.savefig(OUT/(name+'.png'),bbox_inches='tight',dpi=180,facecolor='white')
 outputs.append({'name':name,'caption':caption});plt.close(fig)
def grid(ax):
 ax.grid(axis='x',color='#e6eaed',lw=.7);ax.set_axisbelow(True)
def footer(fig,text):fig.text(.02,.012,text,fontsize=8,color='#566570',va='bottom')

# Recover exact numeric artists; no digitizing of published pixels.
with warnings.catch_warnings():
 warnings.simplefilter('ignore')
 figs=pickle.load(open(source(DRIVE/'comparison/saved_figs.pickle'),'rb'))
a=figs['Accuracy by Pattern scatter'].axes[0]
human={t.get_text():np.asarray(c.get_offsets())[:,1].astype(float) for t,c in zip(a.get_xticklabels(),a.collections[::2])}
assert all(len(v)==25 for v in human.values())
hum_mean=pd.Series({p:human[p].mean() for p in PATS})
pd.DataFrame([(p,i,float(v)) for p in PATS for i,v in enumerate(human[p])],columns=['pattern','point_index_not_participant_id','accuracy']).to_csv(DATA/'human_pattern_points.csv',index=False)
a=figs['Accuracy by Pattern line - All ANNs'].axes[0]
order=[t.get_text() for t in a.get_xticklabels()]
lines=[l for l in a.lines if len(l.get_ydata())==8][1:]
names=a.get_legend_handles_labels()[1][1:]
model_patterns=pd.DataFrame({short(n):dict(zip(order,l.get_ydata())) for n,l in zip(names,lines)}).reindex(PATS)
model_patterns.to_csv(DATA/'model_pattern_accuracy.csv')
old_ids=['DeepSeek-R1-70B','Gemma-2-27B','Gemma-2-2B','Gemma-2-9B','Llama-3.2-3B','Llama-3.3-70B','Phi-4','Qwen2.5-72B']
oldpts=np.asarray(figs['ANN Accuracy Correlation with Humans vs. ANN Overall Accuracy scatter'].axes[0].collections[0].get_offsets())
beh=pd.DataFrame(oldpts,columns=['accuracy','original_profile_r'],index=old_ids)
beh['equal_participant_profile_r']=[np.corrcoef(model_patterns[m],hum_mean)[0,1] for m in beh.index]
assert np.allclose(beh.accuracy,model_patterns.mean().reindex(beh.index))
beh.to_csv(DATA/'behavior_profile_comparison.csv')
plt.close('all')

# 1: preserve stimulus pixels, add vector explanation outside stimulus content.
stim=Image.open(source(FIG/'trial_example2.png'))
fig=plt.figure(figsize=(10,6.4));ax=fig.add_axes([.02,.25,.96,.65]);ax.imshow(stim);ax.axis('off')
w,h=stim.size
xs=np.array([.093,.209,.326,.44,.559,.675,.794,.913])*w
for x,letter in zip(xs,'ABCAABCA'):ax.text(x,.45*h,letter,ha='center',va='center',fontsize=15,fontweight='bold',color=HC)
ax.text(.01*w,.10*h,'Sequence',fontsize=11,fontweight='bold')
ax.text(.01*w,.57*h,'Response options',fontsize=11,fontweight='bold')
ax.annotate('Correct: A',xy=(.69*w,.76*h),xytext=(.84*w,.87*h),fontsize=11,
 arrowprops={'arrowstyle':'->','color':HC},color=HC)
fig.text(.04,.94,'01  |  A worked example of the task',fontsize=16,fontweight='bold')
ta=fig.add_axes([.12,.09,.8,.1]);ta.set_xlim(-.15,.75);ta.set_ylim(0,1);ta.axis('off')
ta.plot([0,.6],[.5,.5],color=CC['frp'],lw=10,solid_capstyle='butt');ta.plot([0,0],[.25,.8],color=HC)
ta.text(0,.03,'Fixation onset',ha='center',fontsize=9);ta.text(.6,.03,'600 ms',ha='center',fontsize=9)
ta.text(.3,.82,'EEG epoch following a valid fixation on a sequence icon',ha='center',fontsize=10)
footer(fig,'Letters, answer annotation and timeline are explanatory additions; the stimulus image is retained.')
save(fig,'01_trial_example','Original stimulus with pattern letters, correct-answer annotation and a schematic fixation-locked EEG window. The letters and annotations are not part of the experimental display. Image content is raster; annotations are vector.')

# 2: equal participant means, marginal bootstrap intervals; identity not inferred from artist order.
rng=np.random.default_rng(20260930)
ci={p:np.quantile(human[p][rng.integers(0,25,(10000,25))].mean(axis=1),[.025,.975]) for p in PATS}
pd.DataFrame([{'pattern':p,'mean':hum_mean[p],'ci_low':ci[p][0],'ci_high':ci[p][1]} for p in PATS]).to_csv(DATA/'human_pattern_summary.csv',index=False)
fig,ax=plt.subplots(figsize=(9.5,6));fig.subplots_adjust(left=.16,right=.98,top=.84,bottom=.23)
y=np.arange(8)
for j,m in enumerate(MODELS):ax.scatter(model_patterns[m]*100,y+(j-3.5)*.055,s=24,color=MC[m],alpha=.9,label=m,zorder=3)
ax.errorbar(hum_mean*100,y+.28,xerr=np.array([[hum_mean[p]-ci[p][0] for p in PATS],[ci[p][1]-hum_mean[p] for p in PATS]])*100,fmt='D',ms=5,color=HC,capsize=3,lw=1.4,label='Humans: mean and 95% CI',zorder=5)
ax.axvline(25,color='#808890',ls='--',lw=1);ax.text(25,-.7,'25% choice baseline',ha='center',fontsize=8,color='#66717a')
ax.set(yticks=y,yticklabels=PATS,xlim=(-2,103),ylim=(7.7,-1),xlabel='Accuracy (%)');grid(ax)
fig.suptitle('02  |  Accuracy across the eight pattern types',x=.16,ha='left',fontsize=15,fontweight='bold')
handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.04),ncol=3,frameon=False,fontsize=8)
footer(fig,'Humans: N=25, equal participant weighting. Model points: one value per model and pattern.')
save(fig,'02_pattern_accuracy','Human means give each of 25 participants equal weight. Whiskers are marginal 95% percentile bootstrap CIs (10,000 resamples of participant values within each pattern; seed 20260930). Model points are descriptive accuracies, without sampling intervals. The 25% line is the uniform four-choice baseline; invalid model outputs can produce below-baseline accuracy. Overall human mean across patterns: 82.47%.')

# 3A / 3D: aligned axes preserve metric units, no connecting unlike quantities.
ordered=beh.sort_values('accuracy',ascending=False).index.tolist()
def aligned(col,name,subtitle):
 fig,axs=plt.subplots(1,2,figsize=(9,4.7),sharey=True);fig.subplots_adjust(left=.22,right=.94,wspace=.30,top=.78,bottom=.17)
 for j,(metric,mult,title,lim) in enumerate([('accuracy',100,'Task accuracy (%)',(0,100)),(col,1,'Human-profile similarity (r)',(-.05,1))]):
  ax=axs[j]
  for i,m in enumerate(ordered):
   v=beh.loc[m,metric]*mult;ax.hlines(i,0,v,color=MC[m],lw=2,alpha=.35);ax.scatter(v,i,s=55,color=MC[m],zorder=3)
   ax.annotate(f'{v:.1f}%' if j==0 else f'{v:.2f}',(v,i),xytext=(7,0),textcoords='offset points',va='center',fontsize=9)
  ax.set(yticks=range(8),yticklabels=ordered,ylim=(7.6,-.6),xlim=lim,xlabel=title);grid(ax);ax.tick_params(axis='y',length=0)
 axs[0].axvline(25,color='#8b949a',lw=1,ls='--');axs[1].axvline(0,color='#8b949a',lw=1)
 fig.suptitle('03  |  Accuracy and human-like difficulty are distinct',x=.04,ha='left',fontsize=15,fontweight='bold')
 fig.text(.04,.87,subtitle,fontsize=9,color='#566570')
 footer(fig,'Models ordered by accuracy. The two columns use separate scales; intervals are not estimated.')
 save(fig,name,'Aligned dot plots, ordered by accuracy. '+subtitle+' Exact values are annotated. Separate axes avoid implying that accuracy and correlation are the same quantity.')
aligned('original_profile_r','03a_accuracy_profile_aligned','Layout comparison: values preserved from the current manuscript scatterplot.')

# 3B: compact scatter without arbitrary quadrants.
fig,ax=plt.subplots(figsize=(7,5));fig.subplots_adjust(left=.13,right=.97,bottom=.18,top=.84)
offsets={'Phi-4':(8,12),'Gemma-2-2B':(8,4),'Gemma-2-9B':(8,8),'Gemma-2-27B':(8,9),'Llama-3.2-3B':(8,-12),'Llama-3.3-70B':(-8,-15),'Qwen2.5-72B':(-8,16),'DeepSeek-R1-70B':(-8,-18)}
for m,row in beh.iterrows():
 x,y=row.accuracy*100,row.original_profile_r;ax.scatter(x,y,color=MC[m],s=65,zorder=3);dx,dy=offsets[m]
 ax.annotate(m,(x,y),xytext=(dx,dy),textcoords='offset points',ha='right' if dx<0 else 'left',fontsize=8.5)
ax.axvline(25,color='#8b949a',ls='--',lw=1);ax.set(xlim=(0,92),ylim=(-.02,.84),xlabel='Task accuracy (%)',ylabel='Human-profile similarity (Pearson r)');grid(ax)
fig.suptitle('03  |  Compact, directly labeled scatter',fontsize=15,fontweight='bold')
footer(fig,'Current manuscript values. Dashed line: uniform four-choice baseline. Vertical scale is shown explicitly.')
save(fig,'03b_accuracy_profile_compact_scatter','Compact scatter with direct model labels. Current manuscript values are preserved. The vertical axis uses the displayed range around the data, rather than the full possible correlation range. This remains the clearest option for judging the relationship between the two metrics.')

# 3C: rank comparison. Explain the loss of distances and near ties.
fig,ax=plt.subplots(figsize=(9,5.5));fig.subplots_adjust(left=.29,right=.71,top=.80,bottom=.16)
r1=beh.accuracy.rank(ascending=False);r2=beh.original_profile_r.rank(ascending=False)
for m in MODELS:
 ax.plot([0,1],[r1[m],r2[m]],color=MC[m],lw=1.8,alpha=.85,marker='o',ms=6)
 ax.text(-.055,r1[m],f'{m}  {beh.loc[m,"accuracy"]*100:.1f}%',ha='right',va='center',fontsize=9)
 ax.text(1.055,r2[m],f'{m}  {beh.loc[m,"original_profile_r"]:.3f}',ha='left',va='center',fontsize=9)
ax.set(xlim=(-.03,1.03),ylim=(8.6,.4),xticks=[0,1],xticklabels=['Accuracy rank','Human-profile rank'],yticks=[])
ax.spines[['left','bottom']].set_visible(False);ax.tick_params(axis='x',length=0)
fig.suptitle('03  |  How model rankings change',fontsize=15,fontweight='bold')
fig.text(.5,.87,'Rank 1 is highest; numbers beside model names retain the original values.',ha='center',fontsize=9)
footer(fig,'Ranks discard effect-size differences. Phi-4 and DeepSeek have almost identical profile correlations.')
save(fig,'03c_accuracy_profile_rank','Rank comparison using current manuscript values. It emphasizes changes in ordering, but overstates tiny rank differences if interpreted without the printed values: Phi-4 and DeepSeek correlations differ by about 0.00006. This is a descriptive alternative, not evidence of statistically distinguishable model ranks.')
aligned('equal_participant_profile_r','03d_accuracy_profile_equal_participant','Analysis sensitivity: correlations recomputed against Figure 2\'s equally weighted human profile.')

# 4: actual trial RDM values; summarize individual matrices without treating pairs as replicates.
def matrix(path):
 with h5py.File(source(path)) as f:
  pats=np.array([x.decode() if isinstance(x,bytes) else str(x) for x in f['pattern_descriptors/patterns'][:]])
  v=f['dissimilarities'][0];n=len(pats);mat=np.zeros((n,n));ii=np.triu_indices(n,1);mat[ii]=v;mat+=mat.T
  ix=np.array([i for p in PATS for i in np.where(pats==p)[0]]);return mat[np.ix_(ix,ix)],pats[ix]
mat,patlabels=matrix(RD/'best_layer/rdm-ANN-ANN_avg-sequence_lvl.hdf5')
np.save(DATA/'grand_mean_trial_rdm.npy',mat)
pairs=[]
for p in sorted((RD/'best_layer').glob('rdm-ANN-*-sequence_lvl.hdf5')):
 if 'ANN_avg' in p.name:continue
 mm,pp=matrix(p);ii=np.triu_indices(len(pp),1);same=pp[ii[0]]==pp[ii[1]]
 pairs.append({'model':short(p.name),'within':mm[ii][same].mean(),'between':mm[ii][~same].mean()})
pairdf=pd.DataFrame(pairs);assert len(pairdf)==8;pairdf.to_csv(DATA/'within_between_distances.csv',index=False)
fig=plt.figure(figsize=(12,5.6));gs=fig.add_gridspec(1,3,width_ratios=[1,1,0.62],left=.09,right=.96,top=.82,bottom=.26,wspace=.43)
ax0=fig.add_subplot(gs[0]);ax1=fig.add_subplot(gs[1]);ax2=fig.add_subplot(gs[2])
design=(patlabels[:,None]!=patlabels[None,:]).astype(int)
ax0.imshow(design,cmap=ListedColormap(['#223948','#e8edf1']),vmin=0,vmax=1,interpolation='none')
im=ax1.imshow(mat,cmap='viridis',vmin=0,vmax=float(mat.max()),interpolation='none',rasterized=True)
for ax,title in [(ax0,'A  Task-design reference'),(ax1,'B  Mean selected-layer RDM')]:
 ax.set_title(title,loc='left',fontsize=11,pad=12);ax.set(xticks=np.arange(8)*50+24.5,xticklabels=PATS,yticks=np.arange(8)*50+24.5,yticklabels=PATS)
 ax.tick_params(axis='x',rotation=90,labelsize=8);ax.tick_params(axis='y',labelsize=8)
 for t in np.arange(1,8)*50-.5:ax.axhline(t,color='white',lw=.35);ax.axvline(t,color='white',lw=.35)
ax0.legend(handles=[Rectangle((0,0),1,1,color='#223948',label='Same pattern'),Rectangle((0,0),1,1,color='#e8edf1',label='Different pattern')],loc='upper center',bbox_to_anchor=(.5,-.28),frameon=False,fontsize=8)
cb=fig.colorbar(im,ax=ax1,fraction=.045,pad=.025);cb.set_label('Correlation distance (1 - r)',fontsize=8);cb.ax.tick_params(labelsize=7)
for _,r in pairdf.iterrows():ax2.plot([0,1],[r['within'],r['between']],color=MC[r.model],marker='o',ms=4,lw=1,alpha=.9)
ax2.set(xticks=[0,1],xticklabels=['Within\npattern','Between\npatterns'],ylabel='Mean distance',xlim=(-.2,1.2),ylim=(0,max(pairdf['between'])*1.15));ax2.set_title('C  Each model',loc='left',fontsize=11,pad=12);ax2.grid(axis='y',color='#e6eaed')
fig.suptitle('04  |  Trial representations cluster by abstract pattern',x=.04,ha='left',fontsize=15,fontweight='bold')
footer(fig,'400 trials; 50 per pattern. Panel C: one connected pair per model; self-distances excluded.')
save(fig,'04_task_structure_rdms','Reference uses a binary same/different-pattern key; measured LLM RDM uses correlation distance. Panel C averages unique within- and between-pattern pairs separately for each of eight models; these are descriptive summaries, not independent pair-level inferential tests. The full trial matrix is preserved, not replaced by an RDM of pattern-averaged features.')

# 5: identical axes with selected-layer markers.
layers=pd.read_csv(source(RD/'layer-wise-similarity.csv'));layers['model']=layers.id.map(short);layers.to_csv(DATA/'layers.csv',index=False)
fig,axs=plt.subplots(2,4,figsize=(11,5.8),sharex=True,sharey=True);fig.subplots_adjust(left=.08,right=.98,bottom=.17,top=.83,hspace=.42,wspace=.16)
peaks=[]
for ax,m in zip(axs.flat,MODELS):
 for other in MODELS:
  g=layers[layers.model.eq(other)].sort_values('layer');ax.plot(g.layer/g.layer.max(),g.similarity,color='#dfe4e8',lw=.7,zorder=1)
 g=layers[layers.model.eq(m)].sort_values('layer');x=g.layer/g.layer.max();ax.plot(x,g.similarity,color=MC[m],lw=2,zorder=3)
 peak=g.loc[g.similarity.idxmax()];dep=peak.layer/g.layer.max();peaks.append({'model':m,'layer':int(peak.layer),'normalized_depth':dep,'r':peak.similarity})
 ax.scatter(dep,peak.similarity,s=35,color=MC[m],edgecolors='white',zorder=4);ax.axvline(dep,color=MC[m],ls=':',lw=.8)
 ax.set_title(m,loc='left',fontsize=10);ax.text(.97,.05,f'Peak: {dep:.0%} depth',transform=ax.transAxes,ha='right',fontsize=8)
 ax.set(xlim=(0,1),ylim=(-.025,.82),xticks=[0,.5,1]);ax.grid(axis='y',color='#edf0f2')
fig.supxlabel('Relative layer depth',y=.09);fig.supylabel('Pearson r with task-design RDM',x=.01)
fig.suptitle('05  |  Task structure across model depth',x=.04,ha='left',fontsize=15,fontweight='bold')
footer(fig,'Colored line: named model. Gray lines: other models. Dot: task-selected layer, independent of EEG.')
pd.DataFrame(peaks).to_csv(DATA/'selected_layers.csv',index=False)
save(fig,'05_layer_profiles','One panel per model with shared axes. Gray trajectories provide context; the colored dot marks the maximum task-design similarity. Normalized depth follows the existing figure convention: layer index divided by the maximum index, not an independently redefined depth measure.')

# 6/7 share the same matched participant/session analysis.
matched=AN/'llm_human_rsa-matched_followups-primary_session_subject_rdm'
rsa=pd.read_csv(source(matched/'matched_hierarchical_llm_human_rsa.csv'));rsa['label']=rsa.model.map(short);rsa.to_csv(DATA/'matched_rsa.csv',index=False)
means=rsa.groupby(['channel_group','eeg_type']).rsa_similarity.mean().unstack().reindex(index=REGIONS,columns=CONDS)
assert len(rsa)==120
fig,ax=plt.subplots(figsize=(7.2,5.3));fig.subplots_adjust(left=.22,right=.85,bottom=.16,top=.81)
hm=np.vstack([means.values,np.full((1,3),np.nan),means.mean().values]);cm=plt.get_cmap('RdBu_r').copy();cm.set_bad('white')
im=ax.imshow(hm,cmap=cm,vmin=-1,vmax=1,aspect='auto')
for i,row in enumerate(hm):
 for j,v in enumerate(row):
  if np.isfinite(v):ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=14,color='white' if abs(v)>.55 else HC)
ax.set(xticks=range(3),xticklabels=['FRP','Response','Rest'],yticks=[0,1,2,3,4,6],yticklabels=['Frontal *','Temporal','Central','Occipital','Parietal','Regional mean'])
ax.tick_params(length=0,pad=8);ax.spines[:].set_visible(False)
cb=fig.colorbar(im,ax=ax,fraction=.05,pad=.05);cb.set_label('Mean LLM-human RSA (r)',fontsize=9)
fig.suptitle('06  |  Neural correspondence across conditions',x=.04,ha='left',fontsize=15,fontweight='bold')
fig.text(.04,.87,'Matched participants and sessions in all three conditions',fontsize=9,color='#566570')
footer(fig,'* Primary region. Each cell averages eight model correlations; color does not indicate significance.')
save(fig,'06_matched_rsa_heatmap','Same 24 participants and 96 complete participant-session cells across conditions. Each regional cell is the arithmetic mean of eight model RSA values. The regional mean is separated from anatomical regions. Frontal is marked as primary. No cellwise significance is implied.')

fig,axs=plt.subplots(1,3,figsize=(10,5.8),sharex=True,sharey=True);fig.subplots_adjust(left=.20,right=.97,top=.8,bottom=.22,wspace=.12)
for ax,cond in zip(axs,CONDS):
 for i,m in enumerate(ordered):
  g=rsa[rsa.label.eq(m)&rsa.eeg_type.eq(cond)].set_index('channel_group').reindex(REGIONS)
  for j,reg in enumerate(REGIONS):
   ax.scatter(g.loc[reg,'rsa_similarity'],i+(j-2)*.085,s=35 if reg=='frontal' else 23,marker='D' if reg=='frontal' else 'o',color=CC[cond],alpha=1 if reg=='frontal' else .4,zorder=3)
  ax.scatter(g.rsa_similarity.mean(),i,s=65,facecolors='white',edgecolors=HC,marker='s',lw=1.2,zorder=4)
 ax.axvline(0,color='#8b949a',ls='--',lw=.8);ax.set(xlim=(-.4,.8),xticks=[-.3,0,.3,.6],yticks=range(8),yticklabels=ordered,ylim=(7.6,-.6),xlabel='RSA (Pearson r)');ax.set_title(cond.upper() if cond=='frp' else cond.title(),color=CC[cond],fontweight='bold');grid(ax)
fig.suptitle('07  |  Model correspondence and regional variation',x=.04,ha='left',fontsize=15,fontweight='bold')
fig.text(.04,.87,'Same matched dataset as Figure 6; model order follows task accuracy',fontsize=9,color='#566570')
fig.legend(handles=[Line2D([],[],marker='D',ls='',color=HC,label='Frontal (primary)'),Line2D([],[],marker='o',ls='',color='#9da9b2',label='Other regional values'),Line2D([],[],marker='s',ls='',markerfacecolor='white',color=HC,label='Mean of five regions')],loc='lower center',bbox_to_anchor=(.5,.075),ncol=3,frameon=False,fontsize=9)
footer(fig,'Small points are regional values, not participant observations or confidence intervals.')
save(fig,'07_model_rsa_regions','Five regional values per model and condition; frontal values are diamonds and regional means are open squares. Uses exactly the matched data underlying Figure 6. This deliberately replaces the condition-specific participant sets used by the previous Figure 7, so response means change from about .285 to .192. The five regions are not treated as independent participant replicates.')

# 8: retain the original Fisher-z summary estimator and its existing participant CIs.
diag=AN/'diagnostics-hierarchical_quality'
part=pd.read_csv(source(diag/'nuisance_rdm_diagnostics_participant.csv'));summ=pd.read_csv(source(diag/'nuisance_rdm_diagnostics_summary.csv'))
part=part[part.eeg_type.eq('frp')];summ=summ[summ.eeg_type.eq('frp')]
part.to_csv(DATA/'gaze_participant_rsa.csv',index=False);summ.to_csv(DATA/'gaze_summary.csv',index=False)
metrics=['n_valid_fixations','rt','mean_fixation_duration','adjacent_overlap_fraction','mean_fixated_stimulus_index']
titles=['Fixation\ncount','Response\ntime','Fixation\nduration','Epoch\noverlap','Fixated\nposition']
fig,axs=plt.subplots(1,5,figsize=(12,5.2),sharey=True,sharex=True);fig.subplots_adjust(left=.10,right=.98,top=.72,bottom=.23,wspace=.14)
for ax,metric,title in zip(axs,metrics,titles):
 for i,reg in enumerate(REGIONS):
  vals=part[part.nuisance.eq(metric)&part.channel_group.eq(reg)].rdm_rsa_similarity.to_numpy()
  s=summ[summ.nuisance.eq(metric)&summ.channel_group.eq(reg)].iloc[0]
  assert len(vals)==s.n==24
  ax.scatter(vals,i+rng.uniform(-.13,.13,len(vals)),s=10,color='#9aa7b2',alpha=.5,zorder=2)
  ax.errorbar(s.mean_r,i,xerr=[[s.mean_r-s.ci95_low_r],[s.ci95_high_r-s.mean_r]],fmt='D' if reg=='frontal' else 'o',ms=5,color=CC['frp'],capsize=3,lw=1.8,zorder=4)
 ax.axvline(0,color='#8b949a',ls='--',lw=.8);ax.set(xlim=(-1,1),xticks=[-1,0,1],yticks=range(5),yticklabels=[r.title()+(' *' if r=='frontal' else '') for r in REGIONS],ylim=(4.6,-.6));ax.set_title(title,fontsize=11,fontweight='bold');grid(ax)
fig.suptitle('08  |  FRP associations with gaze and timing',x=.04,ha='left',fontsize=15,fontweight='bold')
fig.text(.04,.85,'Participant values and 95% confidence intervals reveal variation hidden by the heatmap',fontsize=9,color='#566570')
fig.supxlabel('Neural-behavioral RDM correlation (Pearson r)',y=.12)
footer(fig,'N=24. Blue: Fisher-z mean and t-based 95% CI, back-transformed to r. Gray: participants. * Primary region.')
save(fig,'08_gaze_timing_intervals','Participant dots and the existing Fisher-z mean / t-based 95% confidence intervals, back-transformed to r. N=24 in every region-measure cell. Intervals are marginal, not simultaneous or multiplicity-adjusted. The source heatmap also uses Fisher-z summaries; these are preserved. Associations do not establish that gaze/timing explains away LLM-FRP correspondence.')

# Audit and review booklet: each figure on its own landscape page, with context.
notes=[
 'Temporary design drafts; manuscript source and production figures are unchanged.',
 'Figure 2: the recovered plot contains 25 participant values per pattern. Their equal-weight mean is 82.4708%; the old pooled-trial human line averages 86.7976%. Point indices are not participant IDs, and no paired cross-pattern analysis is inferred from their order.',
 'Figure 3A-C preserve the exact current scatterplot points to isolate layout choice. Figure 3D is a separately labeled sensitivity calculation using the equal-participant profile from Figure 2. It changes correlations (e.g., Phi-4: .688 to .784; Llama-70B: .266 to .426). Choose and document the behavioral estimand before publication.',
 'Figures 6-7 both use matched participants/sessions. The original Figure 7 used condition-specific sets. This accounts for the response mean discrepancy (.285 versus .192). The draft is harmonized to Figure 6 and the matched comparison described in the text.',
 'Figure 8 retains the saved Fisher-z estimator and existing participant confidence intervals. It does not silently switch to the arithmetic mean of raw correlations.',
 'All plot labels and marks are vector in PDF. Figure 1 retains the original stimulus raster; the 400-by-400 empirical RDM is rasterized without interpolation. No new raw-data cleaning or model inference was run.',
 'DeepSeek-R1-70B abbreviates DeepSeek-R1-Distill-Llama-70B throughout. Each output has a PNG preview and source data in data/.',
]
manifest={'outputs':outputs,'notes':notes,'sources':sources,'bootstrap_seed':20260930,'bootstrap_resamples':10000}
(OUT/'provenance.json').write_text(json.dumps(manifest,indent=2))
(OUT/'README.txt').write_text('MAIN-TEXT FIGURE REDESIGNS - TEMPORARY REVIEW\n\n'+'\n\n'.join(notes)+'\n\n'+'\n\n'.join(o['name']+'\n'+o['caption'] for o in outputs))
def wrap(c,text,x,y,width,size=10,leading=15):
 c.setFont('Helvetica',size);words=text.split();line=''
 for word in words:
  new=(line+' '+word).strip()
  if c.stringWidth(new,'Helvetica',size)>width:
   c.drawString(x,y,line);y-=leading;line=word
  else:line=new
 if line:c.drawString(x,y,line);y-=leading
 return y
book=OUT/'00_review_all_figures.pdf';c=canvas.Canvas(str(book),pagesize=(900,650));c.setTitle('Main-text figure redesigns - temporary review')
c.setFillColor(colors.HexColor('#20303b'));c.setFont('Helvetica-Bold',24);c.drawString(40,605,'Main-text figure redesigns')
c.setFont('Helvetica',12);c.drawString(40,580,'Eight figures, four behavioral comparison options | 30 September 2026')
y=545
for i,n in enumerate(notes):y=wrap(c,f'{i+1}. {n}',40,y,810,10,15)-13
c.showPage()
for o in outputs:
 c.setFillColor(colors.HexColor('#20303b'));c.setFont('Helvetica-Bold',14);c.drawString(35,617,o['name'].replace('_',' '))
 image=Image.open(OUT/(o['name']+'.png'));iw,ih=image.size;scale=min(830/iw,450/ih)
 c.drawImage(ImageReader(image),450-iw*scale/2,155+(450-ih*scale)/2,width=iw*scale,height=ih*scale)
 wrap(c,o['caption'],40,120,810,10,14);c.setFont('Helvetica',8);c.drawString(40,25,'Temporary draft | Standalone figure PDFs preserve vector plotting elements.');c.showPage()
c.save()
assert len(outputs)==11
for p in OUT.glob('*.pdf'):
 reader=PdfReader(p);assert len(reader.pages)==(12 if p==book else 1)
print(json.dumps({'output_dir':str(OUT),'pdf_count':len(list(OUT.glob('*.pdf'))),'book_pages':len(PdfReader(book).pages),'human_mean':hum_mean.mean(),'rsa_means':means.mean().to_dict()},indent=2))
