from pathlib import Path
import os
os.environ['MNE_DONTWRITE_HOME']='true';os.environ['MPLCONFIGDIR']='/tmp/elife-mpl-cache'
import mne,numpy as np,pandas as pd,json
from scipy.signal import correlate
B=Path(__file__).resolve().parent;D=B/'data';prior=np.load(D/'real_figure_examples.npz');meta=json.loads((D/'real_figure_examples_metadata.json').read_text())
path=Path(meta['continuous_raw_source']);raw=mne.io.read_raw_fif(path,preload=False,verbose='ERROR');v=raw.get_data(picks=['Fz'])[0];sf=raw.info['sfreq'];a=prior['epochs_uv'][0]/1e6;ac=a-a.mean();c=correlate(v,ac,mode='valid',method='fft');cs=np.r_[0,np.cumsum(v)];cs2=np.r_[0,np.cumsum(v*v)];var=cs2[len(a):]-cs2[:-len(a)]-(cs[len(a):]-cs[:-len(a)])**2/len(a);start=int(np.argmax(c/np.sqrt(np.maximum(var,1e-30)*np.sum(ac**2))))
assert np.allclose(v[start:start+len(a)],a,rtol=1e-9,atol=1e-14)
events_path=Path('/Volumes/Realtek 1Tb/PhD Data/experiment1/data/Lab/raw-BIDS/sub-01/ses-01/eeg/sub-01_ses-01_task-AbsPattComp_events.tsv')
ev=pd.read_csv(events_path,sep='\t',encoding='utf-8-sig');ev['raw_index']=ev['sample']-raw.first_samp
starts=ev[(ev.trial_type=='trial_start')&(ev.raw_index<start)];trialstart=int(starts.iloc[-1].raw_index)
end=int(ev[(ev.trial_type=='trial_end')&(ev.raw_index>start)].iloc[0].raw_index)
trial=ev[(ev.raw_index>=trialstart)&(ev.raw_index<=end)];print(trial.to_string(index=False))
response=trial[trial.trial_type.isin(['a','x','m','l'])];assert len(response)==1
response_index=int(response.iloc[0].raw_index)
# Match manuscript's event-locked window convention: round(-.6*sfreq), 1230 samples.
window_offsets=np.arange(int(round(-.6*sf)),int(round(-.6*sf))+1230)
context_offsets=np.arange(int(round(-.85*sf)),int(round(.2*sf))+1)
result=dict(epoch_time_ms=window_offsets/sf*1000,context_time_ms=context_offsets/sf*1000)
for name,anchor in [('rest',trialstart),('response',response_index)]:
 result[name+'_epoch_uv']=v[anchor+window_offsets]*1e6
 result[name+'_context_uv']=v[anchor+context_offsets]*1e6
np.savez_compressed(D/'real_three_eeg_measures.npz',**result)
info=dict(subject=1,session=1,trial_index=0,channel='Fz',raw_source=str(path),events_source=str(events_path),trial_start_raw_index=trialstart,response_raw_index=response_index,response_event=str(response.iloc[0].trial_type),first_fixation_raw_index=start,baseline=None,window='-600 ms to 0 ms, 1230 samples at 2048 Hz (sample rounding follows analysis convention)',FRP_source=str(D/'real_figure_examples.npz'),validation='First fixation epoch matches continuous raw EEG exactly; event markers bracket that same recorded trial.')
(D/'real_three_eeg_measures_metadata.json').write_text(json.dumps(info,indent=2));print(info)
