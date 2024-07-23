from pathlib import Path
import json
import torch
from huggingface_hub import notebook_login
import numpy as np
import argparse
from transformers import AutoModel, AutoTokenizer

parser = argparse.ArgumentParser()

wd = Path(__file__).parent

with open(wd / "config/private.json") as f:
    private = json.load(f)

api_keys = {k: v.get("key") for k, v in private["API"].items()}


def test2():
    # Initialize model and tokenizer
    model_name = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    activations = {}

    # Define the hook function to include the layer index
    def hook(layer_index):
        def layer_hook(module, input, output):
            layer_name = f"Layer_{layer_index}"
            activations[layer_name] = output[0].detach().cpu().numpy()

        return layer_hook

    # Register hooks using the defined hook function
    for index, layer in enumerate(model.layers):
        if index == 31:
            layer.register_forward_hook(hook(index))

    {f"layer_{i}": layer for i, layer in enumerate(model.layers)}
    
    
    # * Prepare the input
    sequence = "camera eye eye eye camera smile tree tree tree ?"
    options = ["camera", "eye", "bone", "smile", "tree"]
    input_sequence = f"{sequence} {' '.join(options)}"  # Sequence including options
    inputs = tokenizer(input_sequence, return_tensors="pt")

    print("Words split:", input_sequence.split())
    print("Number of split words:", len(input_sequence.split()))

    print("Token IDs:", inputs["input_ids"])
    print("Number of tokens:", len(inputs["input_ids"][0]))
    print("Decoded tokens:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

    # * Perform inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    outputs.keys()
    outputs["last_hidden_state"]
    len(outputs["past_key_values"])
    outputs["past_key_values"][31][1].shape

    dir(activations[layer_name][1])
    activations[layer_name][1].seen_tokens
    len(activations[layer_name][1].value_cache)
    len(activations[layer_name][1].key_cache)

    len(activations["layers.30.mlp.act_fn"]["past_key_values"][0][0])
    activations["layers.30.mlp.act_fn"]["past_key_values"][0][0].shape

    option_inds = np.arange(-4, 0)
    activations["layers.30.mlp.act_fn"]["last_hidden_state"][0, option_inds].shape
    activations["layers.30.mlp.act_fn"]["last_hidden_state"].shape
    activations["layers.30.mlp.act_fn"]["last_hidden_state"].detach().cpu().shape

    activations[layer_name][0].shape

    selected_token_idx = 1
    activations[layer_name][0][0][selected_token_idx].shape

    # Extract and process activations
    # Assuming you are interested in the activations from the specified layer
    print(activations[layer_name].shape)  # Shape of the tensor for verification

    # Remove the hook when done to prevent memory leaks
    hook.remove()
