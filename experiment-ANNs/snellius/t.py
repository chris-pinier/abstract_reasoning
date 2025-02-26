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

def get_timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


WD = Path(__file__).resolve().parent
MODELS_DIR = WD / "models"
MAX_NEW_TOKENS = 50

def make_hook_fn(hooked_layers_outputs, layer_name):
    """
    Returns a hook function that captures the correct layer_name.
    """

    def hook_fn(module, input, output):
        # print(f"HOOK TRIGGERED for {layer_name}, output shape={tuple(output.shape)}")
        hooked_layers_outputs[layer_name].append(output.clone().detach().cpu().float().numpy())

    return hook_fn


def get_layer_activations(pipe, hook_layer_names):
    hooked_layers_outputs = {}
    hook_handles = []

    for hook_layer_name in hook_layer_names:
        # * Find the layer
        layer_to_hook = None
        for name, module in pipe.model.named_modules():
            if name == hook_layer_name:
                layer_to_hook = module
                break
        if layer_to_hook is None:
            print(f"WARNING: Layer '{hook_layer_name}' not found.")
            continue

        # * Initialize storage for this layer
        hooked_layers_outputs[hook_layer_name] = []

        # * Make a hook function bound to the correct name
        hook_fn = make_hook_fn(hooked_layers_outputs, hook_layer_name)
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
            torch_dtype="auto", #torch.bfloat16,
            # device=device,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=1.0,
            top_p=1.0,
            top_k=None,
            device_map="auto",
            return_full_text = False,
            # return_tensors = True,
            # return_text = True,
            # generate_kwargs = kwargs,
        )

    with open(WD / f"models_architectures/{model_id_str}.txt", "w") as f:
        with contextlib.redirect_stdout(f):
            print(pipe.model)

    def generate_text(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS, temperature: float = 1.0, top_p: float = 1.0, top_k=None) -> Dict:
        """
        Generates text based on the given prompt.

        Args:
            prompt: The input text prompt.
            max_new_tokens: The maximum number of tokens to generate.
            temperature: Controls randomness (higher = more random).
            top_p: Nucleus sampling (limit to top_p probability mass).
            top_k: Top-k sampling (limit to top k tokens).

        Returns:
            The generated text and tokens (excluding the prompt).
        """
        inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Tokenize and move to device
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]


        with torch.no_grad():  # Disable gradient calculation
            output = model.generate(
                input_ids=input_ids,
                attention_mask = attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=False,  # You specified do_sample=False in the pipeline
                pad_token_id=tokenizer.eos_token_id,  # Prevent padding issues
                return_dict_in_generate=True # need to have this to access the sequences.
            )

        generated_tokens = output.sequences[0, input_ids.shape[-1]:] # Get *only* generated tokens, which start after the prompt.
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        inputs = {"text": prompt, "tokens": inputs['input_ids']}
        outputs = {"text": generated_text, "tokens":generated_tokens}
        
        return {"input":inputs, "outputs":outputs}


def run_model(model_id: str, prompt_file: Path, hook_layer_names:list):
    timestamp = get_timestamp()
    model_id_str = model_id.replace("/", "--")
    
    save_dir = WD / f"results/{timestamp}/{model_id_str}"
    save_dir.mkdir(parents=True)

    device = get_device_type()

    # ! TEMP
    # if device.type == "cpu":
    #     raise ValueError("Device is CPU, but we need a GPU.")

    df_prompts = pd.read_csv(prompt_file).iloc[:2]

    # pipe = pipeline(
    #     "text-generation",
    #     model=model_id,
    #     torch_dtype="auto", #torch.bfloat16,
    #     # device=device,
    #     do_sample=False,
    #     max_new_tokens=MAX_NEW_TOKENS,
    #     temperature=1.0,
    #     top_p=1.0,
    #     top_k=None,
    #     device_map="auto",
    #     return_full_text = False,
    #     # return_tensors = True,
    #     # return_text = True,
    #     # generate_kwargs = kwargs,
    # )


    # --- Load Model and Tokenizer (instead of pipeline) ---
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",  # Use "auto" for automatic precision
        device_map="auto",  # Let HF handle device placement
    )
    model = model.eval()  # Put the model in evaluation mode

    # --- Text Generation Function (instead of pipe) ---


    export_model_architecture_str(model_id, pipe)
    
    # pipe.model.config.use_cache = False

    # * Access the model
    # model = pipe.model

    # * Access the tokenizer
    tokenizer = pipe.tokenizer

    # * ---------------------------------------- *
    hooked_layers_outputs, hook_handles = get_layer_activations(pipe, hook_layer_names)

    # * Run inference
    tokens = {'prompt': [], 'response': []}

    for i in tqdm(range(len(df_prompts))):
        row = df_prompts.iloc[i : i + 1].copy()
        prompt = row["prompt"].item()
        response = pipe(prompt)[0]["generated_text"]
        row["response"] = response

        # * Tokenize the prompt and response
        prompt_tokens = tokenizer.tokenize(prompt)
        response_tokens = tokenizer.tokenize(response)
        
        tokens['prompt'].append(prompt_tokens)
        tokens['response'].append(response_tokens)

        # * --- Incremental Saving ---
        # * Save the response (append mode)
        if i == 0:  # * Write header only on the first iteration
            row.to_csv(save_dir / "responses.csv", index=False)
        else:
            row.to_csv(save_dir / "responses.csv", index=False, mode="a", header=False)

        # * Reformat activations
        # for layer_name in hooked_layers_outputs.keys():
            # prompt_acts = hooked_layers_outputs[layer_name][0]
            
            # response_acts = hooked_layers_outputs[layer_name][1:]
            # response_acts = np.concatenate(response_acts, axis=1)

            # if prompt_acts.shape[2] != response_acts.shape[2]:
            #     raise ValueError(f"Prompt and response activations have different sequence lengths for layer {layer_name}.")
            # if len(prompt_tokens) != prompt_acts.shape[1]:
            #     raise ValueError(f"Prompt tokens and activations have different lengths for layer {layer_name}.")
            # if len(response_tokens) != response_acts.shape[1]:
            #     raise ValueError(f"Response tokens and activations have different lengths for layer {layer_name}.")
            
            # hooked_layers_outputs[layer_name] = [prompt_acts, response_acts]

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
                "item_ids": df_prompts['item_id'].tolist(),
                "masked_idx": df_prompts['masked_idx'].tolist(),
            },
            f,
            indent=4,
        )

    reformat_act_files(save_dir)


if __name__ == "__main__":
    
    # parser.add_argument("--action", required=True, help="The action to perform.")
   
    parser = argparse.ArgumentParser(description="Run inference on a model.")
    parser.add_argument("--model_id", required=True, help="The model ID to use.")

    args = parser.parse_args()

    model_id = args.model_id # * Get model_id from command line argument

    with open(WD / "config/layer_hook_config.toml", "rb") as f:
        layer_hook_config = tomllib.load(f)

    hook_layer_names = layer_hook_config[model_id]["layers"]

    prompt_file = WD / "sequence_prompts-masked_idx(7).csv"
    run_model(model_id, prompt_file, hook_layer_names)

    # export_model_architecture_str("meta-llama/Llama-3.3-70B-Instruct")