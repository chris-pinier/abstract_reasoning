from transformers import AutoModelForCausalLM, AutoTokenizer  # , BitsAndBytesConfig
import torch
from pathlib import Path
from argparse import ArgumentParser
import tomllib
from tqdm.auto import tqdm
import pickle
import pandas as pd

WD = Path(__file__).resolve().parent
MODELS_DIR = WD / "models"


def load_model_and_tokenizer(
    model_name, cache_dir=None, use_layer_hooks=False, hook_layer_names=None
):
    """
    Loads a pre-trained model and tokenizer, with optional layer hooking for multiple layers.

    Args:
        model_name (str or Path): The name of the pre-trained model.
        cache_dir (str or Path, optional): The directory to cache the model. Defaults to None.
        use_layer_hooks (bool): Whether to hook layers for output capture. Defaults to False.
        hook_layer_names (list of str): Names of the layers to hook. Required if use_layer_hooks is True.

    Returns:
        tuple: A tuple containing the loaded model, tokenizer, and optionally, a dictionary of hooked layer outputs.
               Returns None, None, None if there's an error or if layer hooking is not enabled.
    """
    try:
        device = get_device_type()

        quantization_config = None  # BitsAndBytesConfig(
        #     load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        # )

        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quantization_config, cache_dir=cache_dir
        )

        tokenizer.to(device)
        model.to(device)

        hooked_layers_outputs = {}  # Dictionary to store outputs, keyed by layer name
        hook_handles = []  # List to store hook handles

        if use_layer_hooks:
            if hook_layer_names is None or not isinstance(hook_layer_names, list):
                raise ValueError(
                    "hook_layer_names must be a list of layer names when use_layer_hooks is True"
                )

            for hook_layer_name in hook_layer_names:
                # Find the layer by name
                layer_to_hook = None
                for name, module in model.named_modules():
                    if name == hook_layer_name:
                        layer_to_hook = module
                        break

                if layer_to_hook is None:
                    print(
                        f"WARNING: Layer '{hook_layer_name}' not found in the model. Skipping."
                    )
                    continue

                # Initialize an empty list for this layer's outputs
                hooked_layers_outputs[hook_layer_name] = []

                # Define the hook function
                def hook_fn(module, input, output, layer_name=hook_layer_name):
                    # Store the output in the dictionary under the corresponding layer name
                    hooked_layers_outputs[layer_name].append(
                        output.clone().detach().cpu()
                    )

                # Register the hook
                hook_handle = layer_to_hook.register_forward_hook(
                    lambda module, input, output: hook_fn(
                        module, input, output, hook_layer_name
                    )
                )
                hook_handles.append(hook_handle)

                print(f"Successfully hooked layer: {hook_layer_name}")

            return model, tokenizer, hooked_layers_outputs, hook_handles

        print(f"Successfully loaded model and tokenizer for {model_name}")
        return model, tokenizer, None, None  # No hooked outputs if not enabled

    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None, None, None


def get_device_type():
    # Check for MPS support
    if torch.cuda.is_available():
        device = torch.device("cuda")

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

    return device


def run_model(
    model_name,
    prompts,
    cache_dir=MODELS_DIR,
    use_layer_hooks=False,
    hook_layer_names=None,
):
    # Load model and tokenizer with optional layer hooking
    model, tokenizer, hooked_layers_outputs, hook_handles = load_model_and_tokenizer(
        model_name=model_name,
        cache_dir=cache_dir,
        use_layer_hooks=use_layer_hooks,
        hook_layer_names=hook_layer_names,
    )

    # If model is loaded successfully, move it to the device
    if model:
        model.eval()
        model_responses = []

        for prompt in tqdm(prompts, desc="Generating responses"):
            messages = [
                {"role": "user", "content": prompt},
            ]

            # Tokenize the input
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                tokenize=False,
                # return_attention_mask=True,
            )

            inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Generate output
            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=200,
                    do_sample=False,
                    # top_k=50,
                    # top_p=0.95,
                    temperature=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Decode output
            decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
            model_responses.append(decoded_output)

        return hooked_layers_outputs, decoded_output
    else:
        print("Model loading failed, so cannot proceed with generation.")
        return None, None


def setup_parser():
    parser = ArgumentParser(description="Load a pre-trained model and tokenizer.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The name of the pre-trained model to load.",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default=MODELS_DIR,
        help="The directory to cache the model. Defaults to the 'models' directory in the current working directory.",
    )
    parser.add_argument(
        "--use_layer_hooks",
        action="store_true",
        help="Enable layer hooking for capturing outputs from specific layers.",
    )
    parser.add_argument(
        "--hook_layer_names",
        nargs="+",
        help="Names of the layers to hook. Required if use_layer_hooks is True.",
    )

    return parser


if __name__ == "__main__":
    prompts = ["Give me a short introduction to large language model."]

    # setup_parser()
    # args = setup_parser().parse_args()

    # ! ################################################################################
    # ! TEMP CONFIG
    model_name = "meta-llama/Llama-3.2-3B-Instruct"

    with open(WD / "config/layer_hook_config.toml", "rb") as f:
        layer_hook_config = tomllib.load(f)

    use_layer_hooks = layer_hook_config[model_name]["hook_enabled"]
    hook_layer_names = layer_hook_config[model_name]["layers"]

    hooked_layers_outputs, decoded_outputs = run_model(
        model_name=model_name,
        prompts=prompts,
        cache_dir=MODELS_DIR,
        use_layer_hooks=use_layer_hooks,
        hook_layer_names=hook_layer_names,
    )

    fpath = WD / "{model_name}_hooked_layers_outputs.pkl"

    with open(WD / "hooked_layers_outputs.pkl", "wb") as f:
        pickle.dump(hooked_layers_outputs, f)

    fpath = WD / "{model_name}_decoded_outputs.csv"
    pd.DataFrame(decoded_outputs, columns=["decoded_outputs"]).to_csv(
        fpath, index=False
    )

    # ! ################################################################################
