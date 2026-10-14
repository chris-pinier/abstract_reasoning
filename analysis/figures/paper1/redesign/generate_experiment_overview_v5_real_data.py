from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/elife-mpl-cache')
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.lines import Line2D
from PIL import Image
B=Path(__file__).resolve().parent
O=Path(os.environ.get('PAPER_FIGURE_OUTPUT',str(B.parent/'generated')));O.mkdir(parents=True,exist_ok=True)
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
def label(letter,title,x,y):
 f.text(x,y,letter,fontsize=16,fontweight='bold',color='#0072B2',va='top')
 f.text(x+.027,y-.001,title,fontsize=13,fontweight='bold',va='top')
def box(ax,x,y,w,h,title,body,color):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.025',facecolor='#f3f6f8',edgecolor='#dce4e9',lw=1))
 ax.text(x+.025,y+h-.07,title,fontsize=11,fontweight='bold',color=color,va='top')
 ax.text(x+.025,y+h-.20,body,fontsize=9,color=ink,va='top',linespacing=1.15)
def arrow(ax,p,q,color='#8296a3',head_size=12,line_width=1.4):
 f.add_artist(FancyArrowPatch(p,q,transform=f.transFigure,arrowstyle='-|>',mutation_scale=head_size,lw=line_width,color=color,connectionstyle='arc3,rad=0',shrinkA=0,shrinkB=0,clip_on=False,zorder=20))
label('A','From human and model responses to representational similarity',.04,.897)
im=f.add_axes([.04,.736,.225,.137]);im.imshow(Image.open(B.parent/'assets/trial_example2.png'));im.axis('off')
# Measured examples: participant 1 / session 1 / trial 0, and matched Gemma-2-2B item.
real=np.load(B/'data/real_figure_examples.npz')
from matplotlib.patches import Rectangle, Polygon
flow=f.add_axes([0,0,1,1],zorder=-1);flow.set(xlim=(0,1),ylim=(0,1));flow.axis('off')
starts=real['starts'];cols=['#0072B2','#008778','#69a8d0','#935ba5']
t=real['trace_t'];y=real['trace_uv']
xs=[.30,.452,.614];w=.116
# Shared row bounds: paired human/LLM panels have identical geometry.
row_height=.089; human_y=.767; llm_y=.614
for x,title in zip(xs,['1  Mark fixations','2  Align + average epochs','3  Construct pattern means']):
 f.text(x,.880 if '\n' in title else .872,title,fontsize=9,fontweight='bold',va='top' if '\n' in title else 'baseline')
def trace_axes(x):
 ax=f.add_axes([x,human_y,w,row_height]);ax.set_yticks([]);ax.spines[['left','right','top']].set_visible(False);ax.tick_params(axis='x',labelsize=7,length=2);return ax
ax=trace_axes(xs[0]);ax.plot(t,y,color=ink,lw=.8)
for i,(st,col) in enumerate(zip(starts,cols)):
 ax.add_patch(Rectangle((st,19+i*3),.6,2.3,color=col,alpha=.50,lw=0))
 ax.axvline(st,color=col,ls=':',lw=.8)
ax.set(xlim=(0,t[-1]),ylim=(-8,32),xticks=[0,2,4]);ax.set_xlabel('From first retained fixation (s)',fontsize=7,labelpad=2)
epoch_t=real['epoch_times'];epochs=real['epochs_uv']
ax=trace_axes(xs[1])
for ep,col in zip(epochs,cols):ax.plot(epoch_t*1000,ep,color=col,lw=.85,alpha=.8)
ax.plot(epoch_t*1000,real['mean_uv'],color='black',lw=2,zorder=5)
ax.set(xlim=(0,600),xticks=[0,600],ylim=(-8,16))
ax.set_xlabel('From fixation onset (ms)',fontsize=7,labelpad=2)
ax.text(.98,.96,'Black: trial mean',transform=ax.transAxes,ha='right',va='top',fontsize=6)
hp_example=np.load(B/'data/real_human_pattern_means.npz')
ax=f.add_axes([.614,human_y,.114,row_height])
limit=np.abs(hp_example['means_uv']).max()
human_image=ax.imshow(hp_example['means_uv'],aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit,extent=[0,600,8,0])
ax.set_xticks([0,600]);ax.set_yticks(np.arange(8)+.5,range(1,9));ax.tick_params(labelsize=6,length=0,pad=2)
ax.set_xlabel('Time from fixation (ms)',fontsize=6.5,labelpad=2)
ax.set_ylabel('Abstract pattern',fontsize=7,labelpad=2)
for sp in ax.spines.values():sp.set_visible(False)
# Identical connector lengths, positioned in clear gaps including axis-label space.
connector_length=.018
connector_starts=[.2735,.425,.572,.7345]
def row_connectors(y):
 for x in connector_starts:arrow(flow,(x,y),(x+connector_length,y))
row_connectors(human_y+row_height/2)
# Parallel LLM construction row using measured activations.
llm_color=muted
trial_bounds=im.get_position()
flow.add_patch(FancyBboxPatch((trial_bounds.x0,.598),trial_bounds.width,trial_bounds.height,boxstyle='round,pad=0,rounding_size=.005',facecolor='#f8fafb',edgecolor='#aabac5',lw=1))
f.text(.05,.716,'Instructions +\none solved example',fontsize=11,fontweight='bold',color=cc['frp'],va='top',linespacing=1.2)
flow.add_patch(FancyBboxPatch((.097,.651),.158,.025,boxstyle='round,pad=.003',facecolor='#dceef8',edgecolor='none'))
f.text(.05,.660,'Sequence:',fontsize=7.8)
f.text(.102,.660,'star guitar truck star star guitar truck ?',fontsize=7.8)
f.text(.05,.632,'Options: eye truck guitar star',fontsize=8)
for x,title in zip(xs,['1  Read internal activations','2  Build trial representations','3  Construct pattern means']):
 f.text(x,llm_y+row_height+.016,title,fontsize=8.5,fontweight='bold')
act_vmin,act_vmax=np.quantile(real['activation_views'],[.02,.98])
def feature_grid(rect,arr):
 aa=f.add_axes(rect);aa.imshow(arr,cmap='Blues',aspect='auto',vmin=act_vmin,vmax=act_vmax);aa.set_xticks([]);aa.set_yticks([])
 for sp in aa.spines.values():sp.set_visible(False)
 return aa
# Transformer blocks and corresponding target-token hidden states.
for k in range(3):
 yy=llm_y+(2-k)*.032
 flow.add_patch(FancyBboxPatch((.30,yy),.049,.025,boxstyle='round,pad=0',facecolor='#dceef8',edgecolor=cc['frp'],lw=.5))
 f.text(.3245,yy+.0095,'Layer '+str([1,7,13][k]),ha='center',fontsize=6)
 arrow(flow,(.352,yy+.0125),(.360,yy+.0125),color=llm_color,head_size=7,line_width=.9)
 feature_grid([.364,yy,.052,.025],real['activation_views'][k])
# Show the selected-layer token-by-feature representation.
feature_grid([.452,llm_y,w,row_height],real['activation_views'][-1])
f.text(.452,.601,'Tokens × feature subset',fontsize=6.5)
features=real['pattern_mean_view'][:,np.linspace(0,real['pattern_mean_view'].shape[1]-1,32,dtype=int)];llm_rdm=real['llm_rdm'];human_rdm=real['human_rdm']
aa=f.add_axes([.614,llm_y,.114,row_height]);aa.imshow(features,cmap='Blues',aspect='auto',vmin=np.quantile(features,.02),vmax=np.quantile(features,.98));aa.set_xticks([]);aa.set_yticks([])
for sp in aa.spines.values():sp.set_visible(False)
aa.set_xlabel('Layer 13 feature subset',fontsize=7,labelpad=2);aa.set_ylabel('8 patterns',fontsize=7,labelpad=0)
row_connectors(llm_y+row_height/2)
rdms=[human_rdm,llm_rdm]
for yy,title,rdm in zip([human_y,llm_y],['Human group RDM','LLM RDM'],rdms):
 f.text(.759,yy+row_height+.016,title,fontsize=8,fontweight='bold')
 mat=f.add_axes([.759,yy,row_height,row_height]);mat.imshow(rdm,cmap='Blues');mat.axis('off')
 mat.add_patch(Polygon([(.5,-.5),(7.5,-.5),(7.5,6.5)],fill=False,edgecolor='#bd7100',lw=2.5))
# Extract both upper triangles in the identical row-major pattern-pair order.
upper=np.triu_indices(8,1)
vectors=[human_rdm[upper],llm_rdm[upper]]
for xx,name,vec in zip([.876,.927],['Human','LLM'],vectors):
 va=f.add_axes([xx,.691,.018,.122])
 va.imshow(vec[:,None],cmap='Blues',aspect='auto',vmin=0,vmax=vec.max())
 va.set_xticks([]);va.set_yticks([])
 for sp in va.spines.values():sp.set_edgecolor('#bd7100');sp.set_linewidth(1)
 f.text(xx+.009,.824,name,ha='center',fontsize=7,fontweight='bold')
flow.add_patch(FancyArrowPatch((.898,.752),(.923,.752),arrowstyle='<->',mutation_scale=10,lw=1.3,color=ink))
f.text(.911,.847,'RSA',ha='center',fontsize=11,fontweight='bold',color=cc['frp'])
arrow(flow,(.853,.8115),(.853+connector_length,.8115))
arrow(flow,(.853,.6585),(.936,.685))
f.lines.append(Line2D([.04,.96],[.574,.574],transform=f.transFigure,color='#dce4e9',lw=1))

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

# E: grouped bars reproduce 06b's regional and across-region means.
label('E','Regional context matters',.55,.304)
ax=f.add_axes([.595,.105,.365,.145])
values=np.vstack([means.values,means.mean().values])
ci=pd.read_csv(B/'data/panel_e_participant_bootstrap_ci.csv')
label_y=np.column_stack([ci[ci.eeg_type.eq(c)].set_index('region').reindex(regions+['regional_mean']).ci_high.to_numpy() for c in conditions])+.025
for i in range(6):
 for j in range(1,3):
  for prev in range(j):
   if abs(label_y[i,j]-label_y[i,prev])<.065:label_y[i,j]=label_y[i,prev]+.065
x=np.array([0,1,2,3,4,5.5]);width=.23
ax.axvspan(4.9,6.08,color='#f2f5f7',zorder=0)
for j,c in enumerate(conditions):
 bars=ax.bar(x+(j-1)*width,values[:,j],width=width*.9,color=cc[c],label='FRP' if c=='frp' else c.title(),zorder=3)
 rows=ci[ci.eeg_type.eq(c)].set_index('region').reindex(regions+['regional_mean'])
 assert np.allclose(rows['mean'],values[:,j])
 ax.errorbar(x+(j-1)*width,values[:,j],yerr=np.vstack([values[:,j]-rows.ci_low.to_numpy(),rows.ci_high.to_numpy()-values[:,j]]),fmt='none',ecolor=ink,elinewidth=.8,capsize=2,capthick=.8,zorder=5)
 for xpos,value,ly in zip(x+(j-1)*width,values[:,j],label_y[:,j]):
  ax.text(xpos,ly,f'{value:.2f}',ha='center',va='bottom',fontsize=6.5,color=cc[c])
ax.axhline(0,color='#697780',lw=.9)
ax.set(xticks=x,xticklabels=['Frontal','Temporal','Central','Occipital','Parietal','Regional\nmean'],ylim=(-.45,.78),yticks=[-.4,-.2,0,.2,.4,.6],ylabel='Mean LLM-human RSA (r)',xlim=(-.6,6.1))
ax.tick_params(axis='x',labelsize=7.5)
ax.tick_params(axis='y',labelsize=8)
ax.yaxis.label.set_size(9)
ax.grid(axis='y',color='#e5eaee',lw=.7);ax.set_axisbelow(True)
ax.legend(loc='lower left',bbox_to_anchor=(-.01,1.03),ncol=3,frameon=False,fontsize=8.5,handlelength=1.6,columnspacing=1.5)

from matplotlib.transforms import Bbox
clean_bounds=Bbox.from_extents(.015*14,.045*14,.98*14,.919*14)
out=O/'main_experiment_overview_v5_frp_llm_construction_no_title_real_data_panel_e_ci'
f.savefig(out.with_suffix('.pdf'),bbox_inches=clean_bounds,dpi=600,facecolor='white')
f.savefig(out.with_suffix('.png'),bbox_inches=clean_bounds,dpi=220,facecolor='white')
plt.close(f)
print(out.with_suffix('.pdf'))
