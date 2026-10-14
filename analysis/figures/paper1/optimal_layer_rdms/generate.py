"""Regenerate the appendix RDM figure from the exact archived numeric inputs."""
from pathlib import Path
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/elife-mpl-cache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,FormatStrFormatter
B=Path(__file__).resolve().parent
meta=json.loads((B/'provenance.json').read_text())
colors=json.loads((B/'model_palette.json').read_text())
rdms=np.load(B/'rdms.npz')
models=[x['model'] for x in meta['models']]+['Human frontal FRP']
layers={x['model']:x['layer_zero_based'] for x in meta['models']}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(3,3,figsize=(11,10.9),layout='constrained')
for i,(ax,m) in enumerate(zip(axes.flat,models)):
 a=rdms[m]
 im=ax.imshow(a,cmap='Blues',vmin=0,vmax=a.max(),interpolation='nearest')
 name=m.replace('DeepSeek-R1-70B','DS-R1-Llama-70B')
 suffix=f'Layer {layers[m]}' if m in layers else '24 participants'
 ax.set_title(f'{chr(65+i)}  {name}\n{suffix}',fontsize=11,color=colors.get(m,'#20303b'),pad=8)
 ax.set_xticks(range(8),meta['patterns'],rotation=55,ha='right',fontsize=7)
 ax.set_yticks(range(8),meta['patterns'],fontsize=7)
 ax.tick_params(length=0,pad=3)
 for spine in ax.spines.values():spine.set_visible(False)
 cb=fig.colorbar(im,ax=ax,fraction=.045,pad=.025,shrink=.8)
 cb.set_ticks([0,a.max()/2,a.max()]);cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
 cb.ax.tick_params(labelsize=7,length=2);cb.outline.set_visible(False)
 cb.set_label('1 − r',fontsize=8,labelpad=3)
O=Path(os.environ.get('PAPER_FIGURE_OUTPUT',str(B.parent/'generated')));O.mkdir(parents=True,exist_ok=True)
out=O/'optimal_layer_llm_human_rdms'
fig.savefig(out.with_suffix('.pdf'),bbox_inches='tight')
fig.savefig(out.with_suffix('.png'),bbox_inches='tight',dpi=180)
plt.close(fig)
print(out.with_suffix('.pdf'))
