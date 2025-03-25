from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from pathlib import Path
import pickle
import tomllib
import copy
import pandas as pd
from tqdm.auto import tqdm
from datetime import datetime
import json
import contextlib
import argparse
import numpy as np
import json
from utils import reformat_act_files
from tokenization_utils import locate_target_tokens, clean_tokens


def get_timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


WD = Path(__file__).resolve().parent
MODELS_DIR = WD / "models"
MAX_NEW_TOKENS = 500


def make_layer_hook_fn(hooked_layers_outputs, layer_name):
    """
    Returns a hook function that captures the correct layer_name.
    """

    def hook_fn(module, input, output):
        # print(f"HOOK TRIGGERED for {layer_name}, output shape={tuple(output.shape)}")

        # * Index 0 should correspond to the layer output, other indices are auxiliary
        # * data such as attention weights, but that might depend on models / model types
        # print(f"{output[0].shape = }")
        hooked_layers_outputs[layer_name].append(
            output[0].clone().detach().cpu().float().numpy()
        )

    return hook_fn


def get_layer_outputs(model, hook_layer_names):
    hooked_layers_outputs = {}
    hook_handles = []

    for hook_layer_name in hook_layer_names:
        # * Find the layer
        layer_to_hook = None
        for name, module in model.named_modules():
            if name == hook_layer_name:
                layer_to_hook = module
                break
        if layer_to_hook is None:
            print(f"WARNING: Layer '{hook_layer_name}' not found.")
            continue

        # * Initialize storage for this layer
        hooked_layers_outputs[hook_layer_name] = []

        # * Make a hook function bound to the correct name
        hook_fn = make_layer_hook_fn(hooked_layers_outputs, hook_layer_name)
        hook_handle = layer_to_hook.register_forward_hook(hook_fn)
        hook_handles.append(hook_handle)

        print(f"Successfully hooked layer: {hook_layer_name}")

    return hooked_layers_outputs, hook_handles


def make_sublayer_hook_fn(hooked_layers_outputs, layer_name):
    """
    Returns a hook function that captures the correct layer_name.
    """

    def hook_fn(module, input, output):
        # print(f"HOOK TRIGGERED for {layer_name}, output shape={tuple(output.shape)}")

        hooked_layers_outputs[layer_name].append(
            output.clone().detach().cpu().float().numpy()
        )

    return hook_fn


def get_sublayer_outputs(model, hook_layer_names):
    hooked_layers_outputs = {}
    hook_handles = []

    for hook_layer_name in hook_layer_names:
        # * Find the layer
        layer_to_hook = None
        for name, module in model.named_modules():
            if name == hook_layer_name:
                layer_to_hook = module
                break
        if layer_to_hook is None:
            print(f"WARNING: Layer '{hook_layer_name}' not found.")
            continue

        # * Initialize storage for this layer
        hooked_layers_outputs[hook_layer_name] = []

        # * Make a hook function bound to the correct name
        hook_fn = make_sublayer_hook_fn(hooked_layers_outputs, hook_layer_name)
        hook_handle = layer_to_hook.register_forward_hook(hook_fn)
        hook_handles.append(hook_handle)

        print(f"Successfully hooked layer: {hook_layer_name}")

    return hooked_layers_outputs, hook_handles


def get_device_type():
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # * Check for MPS support
    elif torch.backends.mps.is_built():
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            print(
                "MPS not available, falling back to CPU. Pytorch might not have been built with MPS support."
            )
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print(f"Device used: {device}")
    return device


def export_model_architecture_str(model_id, pipe=None):
    model_id_str = model_id.replace("/", "--")

    if pipe is None:
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype="auto",  # torch.bfloat16,
            # device=device,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=1.0,
            top_p=1.0,
            top_k=None,
            device_map="auto",
            return_full_text=False,
            # return_tensors = True,
            # return_text = True,
            # generate_kwargs = kwargs,
        )

    with open(WD / f"models_architectures/{model_id_str}.txt", "w") as f:
        with contextlib.redirect_stdout(f):
            print(pipe.model)


def run_model_and_extract_sublayer_output(
    model_id: str, prompt_file: Path, hook_layer_names: list
):
    timestamp = get_timestamp()
    model_id_str = model_id.replace("/", "--")

    save_dir = WD / f"results/{timestamp}/{model_id_str}"
    save_dir.mkdir(parents=True)

    device = get_device_type()

    # ! TEMP
    # if device.type == "cpu":
    #     raise ValueError("Device is CPU, but we need a GPU.")

    df_prompts = pd.read_csv(prompt_file).iloc[:2]

    # * Note:
    # * temperature, top_p, and top_k are only active when do_sample=True
    # * See: https://github.com/huggingface/transformers/issues/22405
    model_inference_kwargs = dict(
        torch_dtype="auto",
        do_sample=False,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        top_p=1.0,
        top_k=None,
        device_map="auto",
        return_full_text=False,
    )

    pipe = pipeline(
        "text-generation",
        model=model_id,
        **model_inference_kwargs,
        # return_tensors = True,
        # return_text = True,
        # generate_kwargs = kwargs,
    )

    export_model_architecture_str(model_id, pipe)

    # pipe.model.config.use_cache = False

    # * Access the model
    # model = pipe.model

    # * Access the tokenizer
    tokenizer = pipe.tokenizer

    # * ---------------------------------------- *
    hooked_layers_outputs, hook_handles = get_layer_activations(
        pipe.model, hook_layer_names
    )

    # * Run inference
    tokens = {"prompt": [], "response": []}

    for i in tqdm(range(len(df_prompts))):
        row = df_prompts.iloc[i : i + 1].copy()
        prompt = row["prompt"].item()
        response = pipe(prompt)[0]["generated_text"]
        row["response"] = response

        # * Tokenize the prompt and response
        prompt_tokens = tokenizer.tokenize(prompt)
        response_tokens = tokenizer.tokenize(response)

        tokens["prompt"].append(prompt_tokens)
        tokens["response"].append(response_tokens)

        # * --- Incremental Saving ---
        # * Save the response (append mode)
        if i == 0:  # * Write header only on the first iteration
            row.to_csv(save_dir / "responses.csv", index=False)
        else:
            row.to_csv(save_dir / "responses.csv", index=False, mode="a", header=False)

        # * Save activations incrementally
        with open(
            save_dir / f"layers_acts_{i:04d}.pkl", "wb"
        ) as f:  # Unique filename per prompt
            pickle.dump(hooked_layers_outputs, f)

        # * Clear the hooked_layers_outputs for the next prompt
        for layer_name in hooked_layers_outputs:
            hooked_layers_outputs[layer_name] = []
        # * --- End Incremental Saving ---

        for k in hooked_layers_outputs.keys():
            hooked_layers_outputs[k] = []

    # with open(save_dir / "tokens.json", "w") as f:
    #     json.dump(tokens, f, indent=4)

    with open(save_dir / "tokens.pkl", "wb") as f:
        pickle.dump(tokens, f)

    with open(save_dir / "run_info.json", "w") as f:
        json.dump(
            {
                "model_id": model_id,
                "hooked_layers": hook_layer_names,
                "prompt_file": str(prompt_file),
                "timestamp": timestamp,
                "model_inference_kwargs": model_inference_kwargs,
                "item_ids": df_prompts["item_id"].tolist(),
                "masked_idx": df_prompts["masked_idx"].tolist(),
            },
            f,
            indent=4,
        )

    reformat_act_files(save_dir)


def run_model_and_extract_layer_token_output(
    model_id: str, prompt_file: Path, hook_layer_names: list
):
    timestamp = get_timestamp()
    model_id_str = model_id.replace("/", "--")

    save_dir = WD / f"results/{timestamp}/{model_id_str}"
    save_dir.mkdir(parents=True, exist_ok=True)

    device = get_device_type()

    # ! TEMP
    # if device.type == "cpu":
    #     raise ValueError("Device is CPU, but we need a GPU.")
    # ! TEMP

    df_prompts = pd.read_csv(prompt_file)

    # *  --- Load Model and Tokenizer ---

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",  # Use "auto" for automatic precision
        device_map="auto",  # Let HF handle device placement
    )

    model = model.eval()  # Put the model in evaluation mode

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # * ---------------------------------------- *
    # TODO: Rewrite / reimplement code below (taken from run_models() )
    hooked_layers_outputs, hook_handles = get_layer_outputs(model, hook_layer_names)

    run_info_file = save_dir / "run_info.json"
    with open(run_info_file, "w") as f:
        json.dump(
            {
                "model_id": model_id,
                "hooked_layers": hook_layer_names,
                "prompt_file": str(prompt_file),
                "timestamp": timestamp,
                "item_ids": df_prompts["item_id"].tolist(),
                "masked_idx": df_prompts["masked_idx"].tolist(),
            },
            f,
            indent=4,
        )

    tokens_file = save_dir / "tokens.jsonl"
    responses_file = save_dir / "responses.csv"

    # * Run inference
    for i in tqdm(range(len(df_prompts))):
        # * Convert the row to a DataFrame for easier csv export
        row = pd.DataFrame(df_prompts.iloc[i].copy()).T

        prompt = row["prompt"].item()

        # * Tokenize the prompt
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        input_tokens = np.array(tokenizer.convert_ids_to_tokens(input_ids[0]))

        # *  Decode and include special tokens

        # * Note:
        # * temperature, top_p, and top_k are only active when do_sample=True
        # * See: https://github.com/huggingface/transformers/issues/22405

        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            # top_p=1.0,
            # top_k=None,
            # max_new_tokens=MAX_NEW_TOKENS,
            # return_full_text=False,
        )

        output_tokens = tokenizer.convert_ids_to_tokens(
            output_ids[0], skip_special_tokens=False
        )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        row["response"] = output_text
        # tokens["prompt"].append(input_tokens)
        # tokens["response"].append(output_tokens)

        # * Locate the target tokens (i.e., the tokens of the sequence) in the prompt
        _tok_prefix = "Here is the puzzle you must now solve:\nSequence: "
        _tok_prefix_ind = prompt.index(_tok_prefix)
        tok_prefix = prompt[: _tok_prefix_ind + len(_tok_prefix)]

        _tok_suffix = "\nOptions:"
        _tok_suffix_ind = prompt[_tok_prefix_ind:].index(_tok_suffix)
        tok_suffix = prompt[_tok_prefix_ind + _tok_suffix_ind :]

        tok_indices = locate_target_tokens(
            tok_prefix,
            tok_suffix,
            clean_tokens(input_tokens, tokenizer),
        )

        # * Extract the activations for the target tokens
        for layer_name, layer_acts in hooked_layers_outputs.items():
            prompt_acts = layer_acts[0]
            sequence_acts = prompt_acts[:, slice(*tok_indices), :]

            # * Check that the "sequence activations" have the correct shape
            assert sequence_acts.shape[1] == input_tokens[slice(*tok_indices)].shape[0]

            hooked_layers_outputs[layer_name] = sequence_acts

        # * --- Incremental Saving ---
        # * Save the response
        if i == 0:  # * Write header only on the first iteration
            row.to_csv(responses_file, index=False)
        else:
            row.to_csv(responses_file, index=False, mode="a", header=False)

        # * Save the tokens
        tokens = {"prompt": list(input_tokens), "response": output_tokens}

        with open(tokens_file, "a", encoding="utf-8") as f:
            json.dump(tokens, f, ensure_ascii=False)
            f.write("\n")

        # * Save the activations
        with open(save_dir / f"layers_acts_{row.index[0]:04d}.pkl", "wb") as f:
            pickle.dump(hooked_layers_outputs, f)
        # * --- End Incremental Saving ---

        # * Clear the hooked_layers_outputs for the next prompt
        for layer_name in hooked_layers_outputs.keys():
            hooked_layers_outputs[layer_name] = []

    reformat_act_files(save_dir)


if __name__ == "__main__":
    # parser.add_argument("--action", required=True, help="The action to perform.")

    parser = argparse.ArgumentParser(description="Run inference on a model.")
    parser.add_argument("--model_id", required=True, help="The model ID to use.")

    args = parser.parse_args()

    model_id = args.model_id  # * Get model_id from command line argument

    with open(WD / "config/layer_hook_config.toml", "rb") as f:
        layer_hook_config = tomllib.load(f)

    hook_layer_names = layer_hook_config[model_id]["layers"]

    prompt_file = WD / "sequence_prompts-masked_idx(7).csv"

    # run_model(model_id, prompt_file, hook_layer_name)
    run_model_and_extract_layer_token_output(model_id, prompt_file, hook_layer_names)

    # export_model_architecture_str("meta-llama/Llama-3.3-70B-Instruct")
