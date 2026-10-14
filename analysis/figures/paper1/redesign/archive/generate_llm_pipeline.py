from pathlib import Path
import os
os.environ['MPLCONFIGDIR']='/tmp/elife-mpl-cache'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
B=Path(__file__).resolve().parent;O=B/'llm_pipeline';O.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'font.size':11})
ink='#20303b';muted='#607480';blue='#0072B2';green='#008778';gold='#bd7100'
f=plt.figure(figsize=(15,10),facecolor='white');bg=f.add_axes([0,0,1,1]);bg.set(xlim=(0,1),ylim=(0,1));bg.axis('off')
def text(x,y,s,size=11,color=ink,weight='normal',ha='left'):
 bg.text(x,y,s,fontsize=size,color=color,fontweight=weight,va='top',ha=ha,linespacing=1.4)
def arrow(x,y,u,v):bg.add_patch(FancyArrowPatch((x,y),(u,v),arrowstyle='-|>',mutation_scale=14,color=muted,lw=1.5))
def card(x,y,num,title):
 bg.add_patch(FancyBboxPatch((x,y),.29,.35,boxstyle='round,pad=.008,rounding_size=.012',facecolor='#f6f8fa',edgecolor='#dce4e9'))
 text(x+.012,y+.327,num,15,blue,'bold');text(x+.043,y+.325,title,12,ink,'bold')
def matrix(rect,arr,cmap='Blues'):
 a=f.add_axes(rect);a.imshow(arr,cmap=cmap,aspect='auto');a.set_xticks([]);a.set_yticks([])
 for sp in a.spines.values():sp.set_visible(False)
 return a
rng=np.random.default_rng(13)
text(.04,.962,'From language-model activations to pattern geometry',22,ink,'bold')
text(.04,.915,'What is extracted, how a layer is selected, and what enters the comparison with human EEG',12,muted)
for x,n,t in zip([.04,.355,.67],['1','2','3'],['Present one task trial','Read internal activations','Build trial representations']):card(x,.505,n,t)
for x,n,t in zip([.04,.355,.67],['4','5','6'],['Select a task-informative layer','Construct the pattern RDM','Compare with human geometry']):card(x,.105,n,t)
text(.055,.792,'Instructions + one solved example',10,muted)
text(.055,.752,'Target sequence',10,blue,'bold')
bg.add_patch(FancyBboxPatch((.053,.677),.263,.047,boxstyle='round,pad=.003',facecolor='#dceef8',edgecolor='none'))
text(.06,.711,'star star star guitar star star star ?',9,ink)
text(.055,.657,'Options: truck / star / cube / guitar',9)
text(.055,.615,'Tokenize the full prompt',11,ink,'bold')
text(.055,.577,'400 independent trials per model\nNo previous-trial context',10,muted)
# Transformer layers with selected-token feature grids.
for k in range(4):
 y=.752-k*.039
 bg.add_patch(FancyBboxPatch((.375,y),.098,.029,boxstyle='round,pad=.002',facecolor='#dceef8',edgecolor=blue,lw=.6))
 text(.424,y+.023,'Block '+str(k+1) if k<3 else '… Block L',9,ink,ha='center')
 arrow(.479,y+.014,.502,y+.014)
 matrix([.507,y,.117,.028],rng.normal(size=(3,15)))
text(.375,.8,'Transformer',10,ink,'bold');text(.507,.8,'Layer outputs',10,blue,'bold')
text(.375,.578,'Retain the target-sequence token span\nHidden features at every layer',10,muted)
# Token-feature array to flattened vector.
matrix([.689,.674,.101,.098],rng.normal(size=(8,18)))
text(.687,.792,'Tokens × hidden features',10,blue,'bold')
arrow(.798,.721,.83,.721)
matrix([.835,.703,.1,.035],rng.normal(size=(1,144)))
text(.834,.682,'Flatten',10,ink,'bold')
text(.689,.628,'One vector per trial, per layer',11,ink,'bold')
text(.689,.586,'Keep token-position information\nStack trial vectors for each layer',10,muted)
# Layer selection: reference and trial matrix.
labels=np.repeat(np.arange(8),8);ref=(labels[:,None]!=labels[None,:]).astype(float)
matrix([.059,.263,.075,.103],ref)
text(.055,.397,'Task-design RDM',10,ink,'bold')
text(.055,.245,'Same pattern: 0\nDifferent pattern: 1',8,muted)
a=f.add_axes([.172,.274,.136,.093]);xx=np.linspace(0,1,25);yy=.7*np.sin(np.pi*xx)**1.3;a.plot(xx,yy,color=blue,lw=2);a.scatter([xx[12]],[yy[12]],color=gold,s=40,zorder=3);a.set(xticks=[0,1],yticks=[],xlabel='Relative layer depth');a.xaxis.label.set_size(8);a.tick_params(labelsize=8);a.spines[['top','right','left']].set_visible(False)
text(.169,.397,'RSA at each layer',10,ink,'bold')
text(.055,.19,'Correlate 400 × 400 trial RDMs with the\ntask-design RDM; retain the best layer.',10)
text(.055,.132,'Layer selection uses no human EEG.',10,green,'bold')
# Pattern means -> matrix.
features=rng.normal(size=(8,24));rdm=1-np.corrcoef(features)
matrix([.375,.264,.097,.104],features)
text(.374,.397,'8 pattern-mean vectors',10,ink,'bold')
arrow(.48,.315,.51,.315)
matrix([.524,.264,.083,.104],rdm)
text(.523,.247,'8 × 8 RDM',9,blue,'bold')
text(.375,.206,'Average trial vectors within each pattern\nat the selected layer.',10)
text(.375,.149,'Dissimilarity = 1 − Pearson r\nRepeat separately for each model.',10,muted)
# Human comparison matrices with shared triangular mask.
for x,title,mat in [(.689,'LLM',rdm),(.825,'Human EEG',1-np.corrcoef(features+.5*rng.normal(size=features.shape)))]:
 text(x,.397,title,10,ink,'bold')
 ax=matrix([x,.275,.074,.1],mat)
 ax.add_patch(Polygon([(.5,-.5),(7.5,-.5),(7.5,6.5)],fill=False,edgecolor=gold,lw=1.5))
arrow(.768,.324,.82,.324)
text(.689,.247,'Extract the same 28 unique pattern pairs',10)
text(.689,.201,'Pearson correlation between RDM entries',10,blue,'bold')
text(.689,.157,'One RSA value per model, EEG measure\nand regional group.',10,muted)
# Process sequence is read rowwise; explicit transition avoids crossing figures.
arrow(.332,.686,.35,.686);arrow(.647,.686,.665,.686)
arrow(.332,.286,.35,.286);arrow(.647,.286,.665,.286)
text(.50,.484,'Repeat across layers → choose a layer → compare pattern geometry',10,muted,ha='center')
text(.04,.06,'Answers are scored separately for behavioral accuracy; the RSA features are internal activations.',11,ink,'bold')
text(.04,.027,'Schematic illustration: token grids, matrices and layer trajectory are not measured data. Eight models are analyzed separately.',9,muted)
out=O/'llm_representation_extraction_pipeline'
f.savefig(out.with_suffix('.pdf'),bbox_inches='tight');f.savefig(out.with_suffix('.png'),dpi=180,bbox_inches='tight');plt.close(f)
print(out.with_suffix('.pdf'))
