import transformers
from pathlib import Path
import json
import torch
from huggingface_hub import notebook_login
import numpy as np
import argparse

parser = argparse.ArgumentParser()

wd = Path(__file__).parent

with open(wd / "config/private.json") as f:
    private = json.load(f)

api_keys = {k: v.get("key") for k, v in private["API"].items()}

model_id = "meta-llama/Meta-Llama-3-8B"
# model_id = "microsoft/Phi-3-mini-128k-instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    # model_kwargs={"torch_dtype": torch.float32},
    # device=torch.device("mps"),
    # device="mps",
    # model_kwargs={"torch_dtype": torch.bfloat16},
    # device_map="auto",
)


def get_activation(layer_name):
    def hook(model, input, output):
        activations[layer_name] = output  # .detach()

    return hook


# Dictionary to hold the activations
activations = {}

# Select a layer from which to capture the activations
module = "layers.31.mlp.act_fn"

pipeline.model.register_forward_hook(get_activation(module))

pipeline("Hey how are you doing today?")


# !
def testing():
    import torch
    from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

    model_name = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # * Dictionary to hold the activations
    activations = {}

    def get_activation(layer_name):
        def hook(model, input, output):
            activations[layer_name] = output.detach()

        return hook

    # * Select a layer from which to capture the activations
    # * layer_index = 15  # Change this based on which layer you're interested in
    module = model.layers[31].mlp.act_fn

    model.register_forward_hook(get_activation(module))

    # * Example puzzle input
    sequence = "camera eye eye eye camera smile tree tree tree ?"
    options = ["camera", "eye", "bone", "smile", "tree"]
    input_sequence = f"{sequence} {' '.join(options)}"  # Adding options to the sequence

    # * Tokenize the input
    inputs = tokenizer(input_sequence, return_tensors="pt")

    # * Perform inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    # * Extract the activations for each token in the sequence
    # * Assuming single batch
    token_activations = activations[f"decoder_layer_{layer_index}"][0]

    # * Mapping tokens to their activations
    tokens = tokenizer.tokenize(input_sequence)
    token_to_activation = {
        token: activation for token, activation in zip(tokens, token_activations)
    }

    # * Display the activations for each token
    for token, activation in token_to_activation.items():
        print(f"Activation for token '{token}': {activation.shape}")
        # Here you might want to analyze or save the activations further

    # *
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Example puzzle input
    sequence = "Complete the following sequence: 'camera eye eye eye camera smile tree tree tree',"
    options = ["camera", "eye", "bone", "smile"]
    input_sequence = f"{sequence} with one of the following options: {' '.join(options)}"  # Adding options to the sequence

    # Tokenize the input
    inputs = tokenizer(input_sequence, return_tensors="pt")

    generate_ids = model.generate(
        inputs.input_ids,
        max_new_tokens=10,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    tokenizer.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    # *


def test2():

    import torch
    from transformers import AutoModel, AutoTokenizer

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
