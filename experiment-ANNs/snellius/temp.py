import os
from pathlib import Path
import pickle
from pprint import pprint
import pandas as pd
import re

WD = Path.cwd()
# ----------------------------------------
results_dir = WD / "results/20250206-171916/Qwen--Qwen2.5-7B-Instruct"
act_files = [f for f in results_dir.glob("*.pkl")]
responses = pd.read_csv(results_dir/"responses.csv")

with open(results_dir/"tokens.pkl", 'rb') as f:
    tokens = pickle.load(f)

responses['response']
responses['cleaned_response'] = responses['response'].str.extract(r'Answer: (\w+)', flags=re.IGNORECASE)
responses['correct'] = responses['cleaned_response'] == responses['solution']
responses['correct'].mean()

idx = 0
with open(act_files[idx], 'rb') as f:
    acts = pickle.load(f)

print(type(acts))

pprint(list(acts.keys()))

for k, v in acts.items():
    print(k, type(v), len(v))
    print(v[1].shape)

layer1 = list(acts.keys())[0]
prompt_acts = acts[layer1][0]
response_acts = acts[layer1][1:]
response_acts = np.concatenate(response_acts, axis=1)

assert prompt_acts.shape[1] == len(tokens['prompt'][idx])
assert response_acts.shape[1] == len(tokens['response'][idx])
acts[layer1] = [prompt_acts, response_acts]


# * ----------------------------------------
results_dir = WD / "results"

for child in results_dir.glob("*"):
    if child.is_dir() and child not in ['downloaded', 'new']:
    
        model_name = next(child.walk())[1][0]
        
        responses_file = list(child.rglob("responses.csv"))[0]
        responses = pd.read_csv(responses_file)
        responses['cleaned_response'] = responses['response'].str.extract(r'Answer: (\w+)', flags=re.IGNORECASE)
        responses['correct'] = responses['cleaned_response'] == responses['solution']
        
        print(f"{model_name}: {responses['correct'].mean():.2%}")


# * ----------------------------------------

res_dir = WD / "results/20250207-095625/meta-llama--Llama-3.2-3B-Instruct"
sum(f.stat().st_size for f in res_dir.rglob('*') if f.is_file()) / (1024 * 1024)
layer_acts_files = list(res_dir.glob("layers_acts_*.pkl"))

acts = []
for f in layer_acts_files:
    with open(f, 'rb') as f:
        _acts = pickle.load(f)
        # print(f, len(_acts))
        acts.append(_acts)
    