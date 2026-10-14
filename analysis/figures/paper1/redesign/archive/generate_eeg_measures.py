from pathlib import Path
import os
os.environ['MPLCONFIGDIR']='/tmp/elife-mpl-cache'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle
B=Path(__file__).resolve().parent;O=B/'eeg_measures';O.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'font.size':10})
ink='#20303b';muted='#607480';cols=['#009E73','#0072B2','#D55E00']
f=plt.figure(figsize=(15,10),facecolor='white');bg=f.add_axes([0,0,1,1]);bg.set(xlim=(0,1),ylim=(0,1));bg.axis('off')
def text(x,y,s,size=11,color=ink,bold=False,ha='left'):
 bg.text(x,y,s,fontsize=size,color=color,fontweight='bold' if bold else 'normal',va='top',ha=ha,linespacing=1.4)
def arrow(p,q):bg.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=14,color=muted,lw=1.5))
def axes(rect):
 a=f.add_axes(rect);a.set_yticks([]);a.spines[['left','top','right']].set_visible(False);a.spines['bottom'].set_color('#c3cdd4');a.tick_params(axis='x',labelsize=8,length=3);return a
text(.04,.963,'Three EEG measures, three temporal anchors',23,bold=True)
text(.04,.914,'Preprocessed continuous EEG → trial-level time courses → a common representational analysis',12,muted)
# Trial order schematic, explicitly not to scale.
phases=[(.04,.19,'Fixation cross',cols[0]),(.25,.25,'Trial starts → encoding',muted),(.52,.31,'Full sequence + choices',cols[1]),(.85,.11,'Response',cols[2])]
for x,w,t,c in phases:
 bg.add_patch(FancyBboxPatch((x,.839),w,.031,boxstyle='round,pad=.002',facecolor=c,edgecolor='none',alpha=.12));text(x+w/2,.862,t,10,c,True,'center')
for p,q in [((.233,.855),(.247,.855)),((.503,.855),(.517,.855)),((.833,.855),(.847,.855))]:arrow(p,q)
text(.96,.825,'Trial phases shown schematically; durations are not to scale.',8,muted,ha='right')
xs=[.04,.355,.67]
for x,c,title,sub in zip(xs,cols,['Pre-trial rest EEG','Fixation-related EEG (FRP)','Response-locked EEG'],['Before the trial-start marker','After each retained sequence fixation','Before the response marker']):
 bg.add_patch(FancyBboxPatch((x,.335),.29,.456,boxstyle='round,pad=.008,rounding_size=.012',facecolor='#f6f8fa',edgecolor='#dce4e9'))
 text(x+.012,.771,title,14,c,True);text(x+.012,.734,sub,10,muted)
# Single-window extraction before event.
for x,col,anchor,seed in [(xs[0],cols[0],'Trial start',0),(xs[2],cols[2],'Response',2)]:
 a=axes([x+.023,.571,.243,.113]);t=np.linspace(-850,200,500);y=.18*np.sin(t/31+seed)+.1*np.cos(t/14)+.16*np.sin(t/75)
 a.plot(t,y,color=ink,lw=.85);a.axvspan(-600,0,color=col,alpha=.18);a.axvline(0,color=col,lw=1.4)
 a.set(xlim=(-850,200),ylim=(-.7,.7),xticks=[-600,0]);a.text(-300,.5,'600 ms',ha='center',fontsize=10,color=col);a.text(15,.51,anchor,fontsize=8,color=col)
 a.set_xlabel('Time relative to marker (ms)',fontsize=8,labelpad=3)
 a=axes([x+.068,.423,.152,.062]);tt=np.linspace(-600,0,250);yy=.18*np.sin(tt/31+seed)+.1*np.cos(tt/14)+.16*np.sin(tt/75);a.plot(tt,yy,color=col,lw=1.7);a.set(xticks=[-600,0],xlim=(-600,0))
 arrow((x+.145,.531),(x+.145,.501))
text(.055,.394,'One pre-trial segment per trial',10,cols[0],True)
text(.055,.367,'Fixation cross displayed before trial onset',9,muted)
text(.685,.394,'One pre-response segment per trial',10,cols[2],True)
text(.685,.367,'Omissions: end of the response window',9,muted)
# FRP: multiple overlapping windows -> aligned mean.
x=xs[1];a=axes([x+.023,.571,.243,.113]);t=np.linspace(0,1.3,600);y=.2*np.sin(2*np.pi*7*t)+.1*np.cos(2*np.pi*11*t)+.13*np.sin(2*np.pi*3*t)
a.plot(t,y,color=ink,lw=.85);starts=[.1,.39,.67];shades=['#0072B2','#4798c7','#91c4e0']
for i,(st,c) in enumerate(zip(starts,shades)):
 a.add_patch(Rectangle((st,.5+i*.17),.6,.11,color=c,alpha=.45,lw=0));a.axvline(st,color=c,lw=.8,ls=':');a.text(st,1.1,'f'+str(i+1),fontsize=8,color=c,ha='center')
a.set(xlim=(0,1.3),ylim=(-.65,1.4),xticks=[0,.6,1.2]);a.set_xlabel('Time within the decision phase (s)',fontsize=8,labelpad=3)
a=axes([x+.068,.423,.152,.062]);tt=np.linspace(0,.6,250);eps=np.array([np.interp(st+tt,t,y) for st in starts])
for ep in eps:a.plot(tt*1000,ep,color='#a9cfe4',lw=.7)
a.plot(tt*1000,eps.mean(0),color=cols[1],lw=2);a.set(xlim=(0,600),xticks=[0,600])
text(x+.016,.518,'Align to fixation onset, then average',9,cols[1])
text(x+.016,.394,'One trial FRP from multiple epochs',10,cols[1],True)
text(x+.016,.367,'Sequence fixations ≥100 ms; windows may overlap',8.7,muted)
# Shared analysis with explicit ordering.
text(.04,.3,'Shared construction of representational geometry',14,bold=True)
labels=[('Regional time courses','Average electrodes at each time point'),('8 pattern vectors','Average trials within pattern + session'),('Session RDM','Pairwise correlation distance: 1 − r'),('Participant → group','Average RDMs within, then across people')]
for i,(title,body) in enumerate(labels):
 x=.04+i*.237
 bg.add_patch(FancyBboxPatch((x,.183),.211,.074,boxstyle='round,pad=.006',facecolor='#edf3f7',edgecolor='#d4e0e8'))
 text(x+.01,.245,title,11,ink,True)
 # Wrap explanatory copy deliberately for narrow cards.
 lines=body.replace(' at each','\nat each').replace(' within pattern','\nwithin pattern').replace(': 1',':\n1').replace(' within, then','\nwithin, then')
 text(x+.01,.218,lines,8.7,muted)
 if i<3:arrow((x+.215,.22),(x+.229,.22))
text(.04,.153,'Repeat separately for each EEG measure and each of five electrode groups: frontal, temporal, central, occipital and parietal.',10,muted)
text(.04,.109,'Same 600-ms duration does not imply equivalent measurements.',12,ink,True)
text(.04,.079,'FRP averages multiple post-fixation windows; response and rest use one window per trial. No fixation-specific baseline correction is applied to FRPs.',9.5,muted)
text(.04,.047,'Rest denotes the pre-trial fixation-cross interval, not a separate resting-state recording. Waveforms and timing are schematic, not measured data.',9.5,muted)
out=O/'three_eeg_measures_construction';f.savefig(out.with_suffix('.pdf'),bbox_inches='tight');f.savefig(out.with_suffix('.png'),bbox_inches='tight',dpi=180);plt.close(f)
(O/'caption_and_sources.txt').write_text('Construction of three EEG measures. Pre-trial rest: −600 to 0 ms relative to trial start, during the fixation cross. FRP: 0–600 ms from each retained sequence fixation; epochs are aligned and averaged within trial and may overlap. Response: −600 to 0 ms before response, or response-window end for omissions. Regional electrode averages yield trial time courses. Trial vectors are averaged within pattern and session before computing correlation-distance RDMs. Session RDMs are averaged within participant, then participant RDMs equally across the group. Schematics do not show measured EEG or trial durations. Source: Versions/eLife/04-methods.tex, Trial-level EEG features and Session/Participant RDM construction; existing frp-extraction-pipeline caption.\n')
print(out.with_suffix('.pdf'))
