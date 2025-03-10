from nnsight import CONFIG as nnsight_CONFIG
from pathlib import Path
import tomllib
from box import Box
import os
from nnsight import LanguageModel

WD = Path(__file__).parent

with open(WD.parents[1] / "config/credentials.toml", "rb") as f:
    credentials = Box(tomllib.load(f))

nnsight_CONFIG.set_default_api_key(credentials.api_keys.nnsight)

# llama3.1 70b is a gated model and you need access via your huggingface token
os.environ["HF_TOKEN"] = credentials.api_keys.HF

# We'll never actually load the parameters so no need to specify a device_map.
llama = LanguageModel("meta-llama/Meta-Llama-3.1-70B")

# All we need to specify using NDIF vs executing locally is remote=True.
with llama.trace("The Eiffel Tower is in the city of", remote=True) as runner:
    hidden_states = llama.model.layers[-1].output.save()

    output = llama.output.save()

print(hidden_states)

print(output["logits"])
