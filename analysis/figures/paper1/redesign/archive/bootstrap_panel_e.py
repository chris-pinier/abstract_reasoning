from pathlib import Path
import pickle,json,h5py,numpy as np,pandas as pd
B=Path(__file__).resolve().parent
ROOT=next(p for p in B.parents if (p/'other/final_analysis-primary_regions').exists())
AN=ROOT/'other/final_analysis-primary_regions'
FOLLOW=AN/'llm_human_rsa-matched_followups-primary_session_subject_rdm'
RD=Path('/Volumes/Realtek 1Tb/PhD Data/experiment1-analysis/ANNs/analyzed/RSA-seq_tokens-metric_correlation/best_layer')
meta=json.loads((FOLLOW/'matched_hierarchical_followup_metadata.json').read_text())
pats=meta['patterns'];subjects=meta['matched_subjects'];sessions=meta['matched_sessions'];regions=meta['channel_groups'];conditions=['frp','response','rest'];upper=np.triu_indices(8,1)
models=[];model_sources=[]
for p in sorted(RD.glob('rdm-*pattern_lvl.hdf5')):
 with h5py.File(p) as f:
  labels=[x.decode() for x in f['pattern_descriptors/patterns'][()]]
  mat=np.zeros((8,8));mat[upper]=f['dissimilarities'][()].ravel();mat+=mat.T
  order=[labels.index(x) for x in pats];mat=mat[np.ix_(order,order)]
  models.append(mat[upper]);model_sources.append(str(p))
models=np.array(models);assert len(models)==8
center=models-models.mean(1,keepdims=True);model_unit=center/np.linalg.norm(center,axis=1,keepdims=True)
indices=np.load(FOLLOW/'bootstrap_participant_indices.npy');assert indices.shape==(10000,24)
observed=pd.read_csv(B/'data/matched_rsa.csv').groupby(['channel_group','eeg_type']).rsa_similarity.mean()
boot=np.empty((len(indices),5,3));point=np.empty((5,3));human_arrays=[]
for j,c in enumerate(conditions):
 x=pickle.load((AN/f'hierarchical_rdm/primary_regions-{c}/hierarchical_{c}_rdm_results.pickle').open('rb'))
 assert list(x['patterns'])==pats
 for i,g in enumerate(regions):
  sr=x['channel_groups'][g]['session_rdms']
  participant=np.array([np.mean([sr[tuple(key)] for key in sessions if key[0]==sub],axis=0)[upper] for sub in subjects])
  v=participant.mean(0);v-=v.mean();point[i,j]=(model_unit @ (v/np.linalg.norm(v))).mean()
  assert abs(point[i,j]-observed.loc[g,c])<1e-8,(g,c,point[i,j],observed.loc[g,c])
  v=participant[indices].mean(1);v-=v.mean(1,keepdims=True);v/=np.linalg.norm(v,axis=1,keepdims=True)
  boot[:,i,j]=(v@model_unit.T).mean(1)
  if c=='frp' and g=='frontal':np.save(B/'data/real_human_frontal_rdm.npy',np.mean([np.mean([sr[tuple(key)] for key in sessions if key[0]==sub],axis=0) for sub in subjects],axis=0))
point=np.vstack([point,point.mean(0)])
boot=np.concatenate([boot,boot.mean(1,keepdims=True)],axis=1)
lo,hi=np.quantile(boot,[.025,.975],axis=0)
rows=[dict(region=g,eeg_type=c,mean=point[i,j],ci_low=lo[i,j],ci_high=hi[i,j]) for i,g in enumerate(regions+['regional_mean']) for j,c in enumerate(conditions)]
pd.DataFrame(rows).to_csv(B/'data/panel_e_participant_bootstrap_ci.csv',index=False)
np.savez_compressed(B/'data/panel_e_participant_bootstrap.npz',replicates=boot,point=point,low=lo,high=hi)
(B/'data/panel_e_participant_bootstrap_metadata.json').write_text(json.dumps(dict(method='Paired participant bootstrap; session RDMs averaged within participant; group RDM rebuilt per draw; Pearson RSA calculated per fixed model then arithmetic mean across eight models. Regional mean recomputed in each draw. Percentile 95% intervals.',n=24,n_bootstrap=10000,indices=str(FOLLOW/'bootstrap_participant_indices.npy'),model_sources=model_sources,matched_sessions=sessions),indent=2))
print(pd.DataFrame(rows).round(3).to_string(index=False))
