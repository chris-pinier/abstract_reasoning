from pathlib import Path
import os,json
os.environ['MPLCONFIGDIR']='/tmp/elife-mpl-cache'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle,Circle
B=Path(__file__).resolve().parent;O=B/'frp_pipeline';O.mkdir(exist_ok=True)
real=np.load(B/'data/real_figure_examples.npz')
ink='#20303b';muted='#607480';blue='#0072B2';teal='#008778';orange='#D55E00'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':ink,'pdf.fonttype':42})
f=plt.figure(figsize=(15,11));bg=f.add_axes([0,0,1,1]);bg.set(xlim=(0,1),ylim=(0,1));bg.axis('off')
def text(x,y,s,size=11,color=ink,weight='normal',ha='left',va='top'):
 bg.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,va=va,linespacing=1.35)
def card(x,y,w,h):bg.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.012',facecolor='#f4f7f9',edgecolor='#dce5eb',lw=1))
def arrow(p,q,color='#8c9eaa',ms=15):bg.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=ms,color=color,lw=1.7))
def step(n,x,y,title,color=blue):
 bg.text(x,y,str(n),fontsize=12,color='white',fontweight='bold',ha='center',va='center',bbox=dict(boxstyle='circle,pad=.34',fc=color,ec='none'))
 text(x+.025,y+.009,title,13,weight='bold')
def axis(rect):
 a=f.add_axes(rect);a.spines[['top','right','left']].set_visible(False);a.tick_params(left=False,labelleft=False,labelsize=8);a.spines['bottom'].set_color('#a8b5bf');return a
text(.035,.97,'From eye movements to fixation-related potentials',23,weight='bold')
text(.035,.928,'How fixation onsets become trial-level EEG responses and pattern-level representations',12,color=muted)

# INPUT AND PREPROCESSING
card(.035,.724,.275,.152);step(1,.055,.85,'Record EEG and gaze')
text(.055,.814,'EEG: 64 electrodes, 2,048 Hz\nEyeLink: dominant eye, 2,000 Hz\nMatch fixation times to trial EEG',11)
card(.345,.724,.615,.152);step(2,.365,.85,'Preprocess continuous EEG')
text(.365,.813,'Average reference → notches at 50, 100, 150, 200, 250 Hz → ICA',10.7)
text(.365,.777,'Band-pass 0.1–100 Hz → interpolate bad electrodes → average reference',10.7)
text(.365,.746,'ICA fitted on a 1–100 Hz copy; retain components labeled “brain”.',9,color=muted)
arrow((.313,.795),(.337,.795))
arrow((.5,.716),(.5,.689));text(.515,.705,'Clean continuous EEG',9,color=muted)

# THREE CORE STEPS
for x in [.035,.355,.675]:card(x,.294,.29,.382)
step(3,.055,.651,'Select sequence fixations')
text(.055,.614,'During the decision phase',10,color=muted)
# abstract sequence display, explicitly schematic
seq=list('ABCAABC?');xs=np.linspace(.062,.294,8)
for x,s in zip(xs,seq):
 bg.add_patch(Rectangle((x-.012,.541),.025,.044,facecolor='white',edgecolor='#90a7b6',lw=1))
 text(x,.563,s,11,weight='bold',ha='center',va='center')
for x,s in zip([.103,.151,.199,.247],['D','C','B','A']):
 bg.add_patch(Rectangle((x-.016,.462),.032,.045,facecolor='white',edgecolor='#c8d1d8'))
 text(x,.484,s,10,color='#8c9aa5',ha='center',va='center')
for x in [xs[2],xs[4],xs[5],xs[6]]:bg.scatter(x,.563,s=850,facecolors='none',edgecolors=teal,lw=1.7)
text(.055,.439,'Retain',10,color=teal,weight='bold')
text(.055,.414,'Mean gaze inside a sequence box\nFixation duration ≥ 100 ms',10)
text(.055,.362,'Exclude response-option fixations',10,color=orange)
text(.055,.322,'Letter labels simplify the actual trial.',8.5,color=muted)

step(4,.375,.651,'Extract fixation-locked EEG')
text(.375,.614,'One 0–600 ms epoch per retained fixation',9.5,color=muted)
a=axis([.38,.452,.242,.123]);t=real['trace_t'];y=real['trace_uv']
a.plot(t,y,color=ink,lw=.8);starts=real['starts'];cols=[blue,teal,'#6BAED6','#935ba5']
for i,(st,col) in enumerate(zip(starts,cols)):
 a.add_patch(Rectangle((st,19+i*2),.6,.9,facecolor=col,alpha=.3,edgecolor=col,lw=.8))
 a.axvline(st,color=col,ls=':',lw=.8)
 a.text(st,28,f'f{i+1}',fontsize=8,color=col,ha='center')
a.set(xlim=(0,t[-1]),ylim=(-8,32),xticks=[0,2,4],xlabel='Time from first retained fixation (s)')
a.xaxis.label.set_size(8)
a.set_yticks([0,10]);a.tick_params(axis='y',labelleft=True,labelsize=7);a.set_ylabel('Fz (µV)',fontsize=8)
text(.375,.397,'Start at fixation onset (t = 0)',10,weight='bold')
text(.375,.368,'Fixed window, even if fixation ends sooner\nNo fixation-specific baseline correction',9.5)
text(.375,.314,'Windows can overlap in continuous time.',9,color=muted)

step(5,.695,.651,'Average within each trial')
text(.695,.614,'Align all retained epochs at fixation onset',9.5,color=muted)
a=axis([.70,.452,.24,.123]);tt=real['epoch_times'];ys=real['epochs_uv']
for v in ys:a.plot(tt*1000,v,color='#a2c8df',lw=.8,alpha=.75)
a.plot(tt*1000,real['mean_uv'],color=blue,lw=2.6);a.axvline(0,color=muted,lw=.7)
a.set(xlim=(0,600),ylim=(-8,16),xticks=[0,300,600],xlabel='Time from fixation onset (ms)');a.xaxis.label.set_size(9)
a.set_yticks([-5,0,10]);a.tick_params(axis='y',labelleft=True,labelsize=7);a.set_ylabel('Fz (µV)',fontsize=8)
text(.695,.398,'One trial-level FRP per electrode',10,weight='bold')
text(.695,.368,'Each retained fixation contributes equally\nTrials with no valid epochs have no FRP',9.5)
text(.695,.316,'Pale: individual epochs  •  Blue: their mean',8.8,color=muted)
arrow((.325,.508),(.351,.508));arrow((.645,.508),(.671,.508))
# input gaze routes to selection; cleaned EEG supplies epoch extraction
arrow((.174,.716),(.174,.687))

# HIERARCHY
step(6,.055,.254,'Turn trial FRPs into representational geometry',teal)
text(.055,.221,'Repeat separately for frontal, temporal, central, occipital and parietal channel groups.',10,color=muted)
labels=[('REGIONAL TIME COURSE','Average electrodes\nat each time point'),('PATTERN TIME COURSES','Average trials within\npattern and session'),('SESSION RDM','8 × 8 distances\nCorrelation distance: 1 − r'),('PARTICIPANT RDM','Average that participant’s\nsession RDMs'),('GROUP RDM','Average participant RDMs\nwith equal weights')]
starts=np.linspace(.043,.791,5)
for i,((title,body),x) in enumerate(zip(labels,starts)):
 card(x,.093,.164,.102)
 text(x+.009,.178,title,8.6,color=teal,weight='bold')
 text(x+.009,.147,body,8.7)
 if i<4:arrow((x+.168,.144),(starts[i+1]-.011,.144),ms=11)
text(.035,.049,'FRPs summarize ongoing sequence inspection; they are not isolated responses to a single visual event.',10.5,weight='bold')
text(.035,.021,'Panels 4–5: measured EEG, participant 1 / session 1 / trial 1, electrode Fz; all four retained sequence fixations. Panel 3 uses simplified symbols for the same trial.',9,color=muted)
p=O/'frp_extraction_pipeline_real_data';f.savefig(p.with_suffix('.pdf'),bbox_inches='tight',dpi=600,facecolor='white');f.savefig(p.with_suffix('.png'),bbox_inches='tight',dpi=220,facecolor='white');plt.close(f)
(O/'caption_and_sources.txt').write_text('''FRP extraction and processing. Continuous EEG is initially average-referenced excluding flagged bad electrodes, notch-filtered at 50-Hz multiples from 50 to 250 Hz, cleaned with extended Infomax ICA fitted to a 1–100 Hz copy (only ICLabel brain components retained), band-pass filtered at 0.1–100 Hz, interpolated at bad channels, and average-referenced again. Eye-tracker-labeled fixations during the decision phase are assigned by mean gaze position to stimulus bounding boxes. Retained sequence fixations last at least 100 ms; response-option fixations do not contribute to sequence FRPs. EEG is cropped from fixation onset through 600 ms without baseline correction. These windows can extend beyond the fixation and overlap one another. All retained sequence-fixation epochs are equally averaged within trial, yielding a multichannel trial FRP; trials without valid epochs have no FRP. Regional electrode averages yield time-course features. Within each session, trials are averaged within each of eight patterns, then pairwise correlation distances form a session RDM. Session RDMs are averaged within participant; participant RDMs are averaged equally at group level. All waveforms, gaze markers and timing are schematic.

Sources:
04-methods.tex: EEG preprocessing and RSA sections.
abstract_reasoning/analysis/src/ar_analysis/analysis_config.py: MIN_FIXATION_DURATION = 0.1.
abstract_reasoning/analysis/src/ar_analysis/data_loader/human_data.py: analyze_trial epoch extraction and within-trial concatenation/average (around lines 2780–2930).
abstract_reasoning/analysis/scripts/frp_hierarchical_rdm.py: load_trial_frp_features validates baseline=None, non-negative onset, 600-ms coverage, and regional channel averaging.

The >=100-ms inclusion threshold is explicit in the current code but absent from the current methods prose; consider adding it when finalizing the manuscript. This infographic has not been inserted into the paper.
''')
print(p.with_suffix('.pdf'))
