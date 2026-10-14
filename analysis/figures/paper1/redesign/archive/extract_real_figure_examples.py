from pathlib import Path
import os
os.environ['MNE_DONTWRITE_HOME']='true';os.environ['MPLCONFIGDIR']='/tmp/elife-mpl-cache'
import pickle,json,h5py,numpy as np,pandas as pd
B=Path(__file__).resolve().parent;D=B/'data';ROOT=next(p for p in B.parents if (p/'other/final_analysis-primary_regions').exists())
DRIVE=Path('/Volumes/Realtek 1Tb/PhD Data')
H=DRIVE/'demo3_draft_export/session_analysis-100ms_min_fix/subj_01/sess_01'
all_epochs=pickle.load((H/'eeg_fixation_epochs.pkl').open('rb'))
frps=pickle.load((H/'sess_frps.pkl').open('rb'))
events=pd.read_parquet(H/'fixation_events.parquet');manifest=pd.read_csv(H/'sess_frp_manifest.csv')
trial=0;channel='Fz'
ev=events[events.trial_N.eq(trial)&events.stim_type.eq('sequence')&events.duration.ge(.1)].sort_values('onset')
assert len(ev)==4
collected=[];counts={}
for row in ev.itertuples():
 obj=all_epochs[trial][row.target_ind];k=counts.get(row.target_ind,0);counts[row.target_ind]=k+1
 assert obj.baseline is None and obj.tmin==0
 collected.append(obj.get_data()[k,obj.ch_names.index(channel)])
epochs=np.array(collected);sfreq=obj.info['sfreq'];times=obj.times
mean=epochs.mean(0);actual=frps['sequence'][trial].get_data()[frps['sequence'][trial].ch_names.index(channel)]
assert np.allclose(mean,actual,rtol=1e-9,atol=1e-14)
onsets=ev.onset.to_numpy();relative=onsets-onsets[0]
# Initially preserve only recorded windows; never interpolate unobserved gaps.
offsets=(onsets*sfreq).astype(int);offsets-=offsets[0]
sums=np.zeros(offsets[-1]+len(times));count=np.zeros_like(sums)
for offset,ep in zip(offsets,epochs):sums[offset:offset+len(ep)]+=ep;count[offset:offset+len(ep)]+=1
trace=np.divide(sums,count,out=np.full_like(sums,np.nan),where=count>0);t=np.arange(len(trace))/sfreq
continuous=False
# Recover the exact continuous recording only if its samples reproduce all saved epochs.
raw_file=DRIVE/'experiment1/data/Lab/raw-BIDS/derivatives/preprocessed/sub-01_ses-01_task-AbsPattComp_prepro-eeg-raw.fif'
import mne
from scipy.signal import correlate
raw=mne.io.read_raw_fif(raw_file,preload=False,verbose='ERROR');record=raw.get_data(picks=[channel])[0]
a=epochs[0];ac=a-a.mean();corr=correlate(record,ac,mode='valid',method='fft');cs=np.r_[0.,np.cumsum(record)];cs2=np.r_[0.,np.cumsum(record**2)];var=cs2[len(a):]-cs2[:-len(a)]-(cs[len(a):]-cs[:-len(a)])**2/len(a)
score=corr/np.sqrt(np.maximum(var,1e-30)*np.sum(ac**2));start=int(np.argmax(score))
max_error=max(float(np.max(np.abs(record[start+off:start+off+len(ep)]-ep))) for off,ep in zip(offsets,epochs))
if max_error<1e-10:
 trace=record[start:start+len(trace)];continuous=True
print('Human: epochs',epochs.shape,'continuous verified',continuous,'max raw error V',max_error)
# Use the same task item for the LLM example.
item=int(manifest.loc[manifest.trial_N.eq(trial),'item_id'].iloc[0]);pattern=manifest.loc[manifest.trial_N.eq(trial),'pattern'].iloc[0]
L=DRIVE/'experiment1/data/ANNs/local_run/sessions-1_to_5-masked_idx(7)-sequence_tokens_acts/google--gemma-2-2b-it'
responses=pd.read_csv(L/'responses.csv');idx=responses.index[responses.item_id.eq(item)].tolist();assert len(idx)==1;idx=idx[0]
activations=[];selected_all=None
for layer in [1,7,13]:
 arr=pickle.load((L/f'acts_by_layer-model.layers.{layer}.pkl').open('rb'))
 activations.append(np.asarray(arr[idx]).squeeze(0))
 if layer==13:selected_all=np.array([np.asarray(v).reshape(-1) for v in arr])
RD=DRIVE/'experiment1-analysis/ANNs/analyzed/RSA-seq_tokens-metric_correlation/best_layer'
with h5py.File(RD/'dataset-ANN-google--gemma-2-2b-it-model.layers.13-pattern_lvl.hdf5') as f:
 pattern_means=f['measurements'][()];patterns=[x.decode() for x in f['obs_descriptors/patterns'][()]]
recomputed=np.array([selected_all[responses.pattern.eq(p)].mean(0) for p in patterns])
assert np.allclose(recomputed,pattern_means,rtol=1e-5,atol=1e-5),np.max(abs(recomputed-pattern_means))
with h5py.File(RD/'rdm-ANN-google--gemma-2-2b-it-model.layers.13-pattern_lvl.hdf5') as f:
 labels=[x.decode() for x in f['pattern_descriptors/patterns'][()]];llm=np.zeros((8,8));llm[np.triu_indices(8,1)]=f['dissimilarities'][()].ravel();llm+=llm.T
 assert labels==patterns
assert np.allclose(1-np.corrcoef(pattern_means),llm,atol=5e-6)
# Deterministic, evenly spaced hidden units; retain every token in the illustration.
hidden_indices=np.linspace(0,activations[0].shape[1]-1,24,dtype=int)
act_view=np.stack([a[:,hidden_indices] for a in activations])
flat_indices=np.array([token*activations[0].shape[1]+unit for token in range(activations[0].shape[0]) for unit in hidden_indices])
# Display the same feature subset in the pattern means (all tokens, 24 units/token).
mean_view=pattern_means[:,flat_indices]
payload=dict(epoch_times=times,epochs_uv=epochs*1e6,mean_uv=mean*1e6,trace_t=t,trace_uv=trace*1e6,starts=relative,durations=ev.duration.to_numpy(),activation_views=act_view,flat_view=act_view[-1].reshape(1,-1),pattern_mean_view=mean_view,llm_rdm=llm,human_rdm=np.load(D/'real_human_frontal_rdm.npy'))
np.savez_compressed(D/'real_figure_examples.npz',**payload)
meta=dict(human_subject=1,human_session=1,human_trial_index=trial,item_id=item,pattern=pattern,channel=channel,n_sequence_fixations=len(ev),epoch_start=0,epoch_end=float(times[-1]),baseline=None,eeg_epoch_source=str(H/'eeg_fixation_epochs.pkl'),continuous_raw_source=str(raw_file) if continuous else None,continuous_verified=continuous,raw_match_max_error_volts=max_error,trace_note='Continuous raw EEG verified against saved epochs.' if continuous else 'Only retained recorded windows shown; gaps contain no interpolated data.',human_rdm='Matched 24-participant frontal FRP group RDM; sessions averaged within participant.',llm='Gemma-2-2B',llm_row_index=int(idx),llm_layers_zero_based=[1,7,13],llm_selected_layer_zero_based=13,activation_source=str(L),prompt=responses.loc[idx,'prompt'],patterns=patterns,hidden_unit_indices=hidden_indices.tolist(),feature_view='All nine sequence tokens; 24 evenly spaced hidden units displayed; full features used to calculate RDMs.',checks=['Saved trial FRP equals mean of retained sequence epochs','Raw activations reproduce saved pattern-mean dataset','Full pattern-mean vectors reproduce saved LLM RDM'])
(D/'real_figure_examples_metadata.json').write_text(json.dumps(meta,indent=2));print(json.dumps({k:v for k,v in meta.items() if k not in ['prompt','hidden_unit_indices']},indent=2))
