from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/elife-mpl-cache')
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
B=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#91999f','text.color':'#20303b','axes.labelcolor':'#20303b','xtick.color':'#425360','ytick.color':'#425360'})
PATS=['AAABAAAB','ABABCDCD','ABBAABBA','ABBACDDC','ABBCABBC','ABCAABCA','ABCDDCBA','ABCDEEDC']
mat=np.load(B/'data/grand_mean_trial_rdm.npy');assert mat.shape==(400,400)
patlabels=np.repeat(PATS,50)
pairdf=pd.read_csv(B/'data/within_between_distances.csv')
MC=json.loads((B/'model_palette.json').read_text())
fig=plt.figure(figsize=(12,5.6));gs=fig.add_gridspec(1,3,width_ratios=[1,1,0.62],left=.09,right=.96,top=.82,bottom=.26,wspace=.43)
ax0=fig.add_subplot(gs[0]);ax1=fig.add_subplot(gs[1]);ax2=fig.add_subplot(gs[2])
design=(patlabels[:,None]!=patlabels[None,:]).astype(int)
ax0.imshow(design,cmap=ListedColormap(['#223948','#e8edf1']),vmin=0,vmax=1,interpolation='none')
im=ax1.imshow(mat,cmap='Blues',vmin=0,vmax=float(mat.max()),interpolation='none',rasterized=True)
for ax,title in [(ax0,'A  Task-design reference'),(ax1,'B  Mean selected-layer RDM')]:
 ax.set_title(title,loc='left',fontsize=11,pad=12);ax.set(xticks=np.arange(8)*50+24.5,xticklabels=PATS,yticks=np.arange(8)*50+24.5,yticklabels=PATS)
 ax.tick_params(axis='x',rotation=90,labelsize=8);ax.tick_params(axis='y',labelsize=8)
 for t in np.arange(1,8)*50-.5:ax.axhline(t,color='white',lw=.35);ax.axvline(t,color='white',lw=.35)
ax0.legend(handles=[Rectangle((0,0),1,1,color='#223948',label='Same pattern'),Rectangle((0,0),1,1,color='#e8edf1',label='Different pattern')],loc='upper center',bbox_to_anchor=(.5,-.28),frameon=False,fontsize=8)
cax=ax1.inset_axes([0,-.34,1,.035]);cb=fig.colorbar(im,cax=cax,orientation="horizontal");cb.set_label('Correlation distance (1 - r)',fontsize=8);cb.ax.tick_params(labelsize=7)
for _,r in pairdf.iterrows():ax2.plot([0,1],[r['within'],r['between']],color=MC[r.model],marker='o',ms=4,lw=1,alpha=.9)
ax2.set(xticks=[0,1],xticklabels=['Within\npattern','Between\npatterns'],ylabel='Mean distance',xlim=(-.2,1.2),ylim=(0,max(pairdf['between'])*1.15));ax2.set_title('C  Each model',loc='left',fontsize=11,pad=12);ax2.grid(axis='y',color='#e6eaed')


fig.canvas.draw()
assert np.allclose(ax0.get_window_extent().size,ax1.get_window_extent().size)
O=Path(os.environ.get('PAPER_FIGURE_OUTPUT',str(B.parent/'generated')));O.mkdir(parents=True,exist_ok=True)
for suffix in ['pdf','png']:
 fig.savefig(O/('04_task_structure_rdms_no_title.'+suffix),bbox_inches='tight',facecolor='white',dpi=300)
plt.close(fig)
