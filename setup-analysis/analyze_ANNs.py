from pathlib import Path
import os
import pandas as pd
import pickle


WD = Path.cwd()


ann_dir = WD.parent / "experiment-ANNs"

with open(ann_dir / "config/instructions.txt", "r") as f:
    instructions = f.read()

seq_prompts = pd.read_csv(ann_dir / "sequences/sequence_prompts.csv")

ann_res_dir = ann_dir / "results/local_run"

responses_files = list(ann_res_dir.glob("*responses.pkl"))

df_resp = pd.DataFrame(columns=["response", "model"])

for resp_file in responses_files:
    with open(resp_file, "rb") as f:
        resps = pickle.load(f)

    resps = [r.replace(seq_prompts.loc[i, "prompt"], "") for i, r in enumerate(resps)]
    model_str = resp_file.stem.split("--")[1].replace("-responses", "")

    df = pd.DataFrame(resps, columns=["response"])
    df["model"] = model_str

    df_resp = pd.concat([df_resp, df])

print(seq_prompts["prompt"][0])
print(df["response"][0])
df["response"][0].replace(seq_prompts["prompt"][0], "")

df_resp.reset_index(drop=True, inplace=True)
# df_resp['response'] = df_resp['response'].str.replace(instructions, '')
print(instructions)

print(df_resp.loc[0, "response"])
df_resp.loc[500]

print(seq_prompts.loc[3, "prompt"])

df_resp["model"].unique()

for resp in df_resp.query("model.str.contains('QwQ')")["response"]:
    print(resp, f"\n{'-'*50}\n")

df_resp.query("model.str.contains('QwQ')")["response"]


seq_prompts["solution"].iloc[:5]
