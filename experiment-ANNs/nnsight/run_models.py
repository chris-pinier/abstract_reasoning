from nnsight import CONFIG as nnsight_CONFIG
from pathlib import Path
import tomllib
from box import Box
import os
from nnsight import LanguageModel
import numpy as np
import pandas as pd
import torch
from tokenization_utils import locate_target_tokens, clean_tokens

WD = Path(__file__).parent
config_dir = WD.parents[1] / "config"

with open(config_dir / "layer_hook_config.toml", "rb") as f:
    layer_hook_conf = tomllib.load(f)

with open(config_dir / "credentials.toml", "rb") as f:
    credentials = Box(tomllib.load(f))

nnsight_CONFIG.set_default_api_key(credentials.api_keys.nnsight)

# * llama3.1 70b is a gated model and you need access via your huggingface token
os.environ["HF_TOKEN"] = credentials.api_keys.huggingface

model_id = "meta-llama/Meta-Llama-3.1-70B"

hooked_layers = layer_hook_conf[model_id]["layers"]
hooked_layers = [int(l.replace("model.layers.", "")) for l in hooked_layers]

# * We'll never actually load the parameters so no need to specify a device_map.
model = LanguageModel(model_id)

layers_acts = []


prompts_file = WD.parent / "sequence_prompts/sequence_prompts-masked_idx(7).csv"
df_prompts = pd.read_csv(prompts_file)
prompts = df_prompts["prompt"].to_list()


# * ----------------------------------------
# * ----------------------------------------
token_inds = []
tokenizer = model.tokenizer

for prompt in prompts:
    # * Tokenize the prompt
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids  # .to(device)
    attention_mask = encoded.attention_mask  # .to(device)
    input_tokens = np.array(tokenizer.convert_ids_to_tokens(input_ids[0]))

    # *  Decode and include special tokens

    # * Note:
    # * temperature, top_p, and top_k are only active when do_sample=True
    # * See: https://github.com/huggingface/transformers/issues/22405

    # * Locate the target tokens (i.e., the tokens of the sequence) in the prompt
    _tok_prefix = "Here is the puzzle you must now solve:\nSequence: "
    _tok_prefix_ind = prompt.index(_tok_prefix)
    tok_prefix = prompt[: _tok_prefix_ind + len(_tok_prefix)]

    _tok_suffix = "\nOptions:"
    _tok_suffix_ind = prompt[_tok_prefix_ind:].index(_tok_suffix)
    tok_suffix = prompt[_tok_prefix_ind + _tok_suffix_ind :]

    inds = locate_target_tokens(
        tok_prefix,
        tok_suffix,
        clean_tokens(input_tokens, tokenizer),
    )
    inds = slice(*inds)

    token_inds.append(inds)

# * ----------------------------------------
# * ----------------------------------------
import nnsight
prompts = prompts[:3]  # ! TEMP
hooked_layers = hooked_layers[:3]  # ! TEMP

model = LanguageModel(model_id)

acts = []
outputs = []
inputs = []

with model.session(remote=True) as session:
    _acts = nnsight.list().save()
    # * All we need to specify using NDIF vs executing locally is remote=True.
    with model.trace(prompts, remote=True) as runner:
        # inputs.append(model.model.inputs.save())

        masks = []

        # for attention_mask in inputs[0][1]["attention_mask"]:
        for attention_mask in model.model.inputs[0][1]["attention_mask"]:
            masks.append(torch.where(attention_mask == 1)[0])

        for hooked_layer_N in hooked_layers:
            layer_acts = model.model.layers[hooked_layer_N].output #.save()

            _layer_acts = []

            for i in range(len(prompts)):
                prompt_acts = layer_acts[0][i][token_inds[i]]
                #prompt_acts = prompt_acts.cpu().float().numpy().save()
                _layer_acts.append(prompt_acts)

            # layer_acts = np.array(layer_acts)
            _acts.append(_layer_acts)

        # outputs.append(model.output.save())

    # _acts = np.array(acts)
    # acts.append(_acts)

len(acts[0])

acts = np.array([np.array(acts[i]) for i in range(len(acts))])
np.save(WD / f"{model_id.replace("/", "--")}-acts", acts)

acts[0]
[a.shape for a in acts[0]]
acts[0][0]


# acts[0].shape
# acts[0][0].shape
len(acts)
acts[0].shape

acts[0]
len(acts)
acts[0]


layers_acts_per_prompt = {}
for i in range(len(prompts)):
    acts = [act[0][i][masks[i]].float().numpy() for act in layers_acts.values()]

    # * turn into np array of shape (n_layers, n_tokens, layer_output_dim)
    acts = np.array(acts)
    layers_acts_per_prompt[i] = acts

layers_acts_per_prompt[1]
