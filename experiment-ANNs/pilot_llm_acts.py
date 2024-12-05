import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import code
import argparse
import requests
import re
from typing import List, Dict
from huggingface_hub import HfApi, ModelFilter
import inspect

# * ----------------------------------------
# model_methods = [m for m in dir(transformers) if m.lower().startswith("automodel")]
# hf_api = HfApi()

# models = hf_api.list_models(
#     filter=ModelFilter(task="text-generation", model_name="llama3")
# )

# * ----------------------------------------


def run_pipeline(model_id, prompt):
    # model_id = "meta-llama/Meta-Llama-3-8B"
    # model_id = "microsoft/Phi-3-small-8k-instruct"
    model_id = "microsoft/Phi-3-mini-128k-instruct"

    # * pinning a revision
    # model_id = "microsoft/Phi-3-mini-128k-instruct@main"  # Replace 'main' with the specific hash or tag
    
    pipeline = transformers.pipeline(
        task="text-generation",
        model=model_id,
        # model_kwargs={"torch_dtype": torch.bfloat16},
        # device_map="auto"
        device="mps",
        framework="pt",
        # generation_kwargs={"temperature": 0.7, "top_k": 50}
        # model_kwargs = {}
        trust_remote_code=True,
    )

    print((list(inspect.signature(transformers.pipeline(task='text-generation',model=model_id)).parameters.keys())))

    model = pipeline.model

    output = pipeline("hey, who are you?")

def search(patt, text: List | str):
    if isinstance(text, str):
        return re.search(patt, text, re.IGNORECASE)

    elif isinstance(text, list):
        return [t for t in text if re.search(patt, t, re.IGNORECASE)]


def main(model_name):    
    activations = {}


    # * Define the hook function to include the layer index
    def hook(layer_index):
        def layer_hook(module, input, output):
            layer_name = f"Layer_{layer_index}"
            activations[layer_name] = output  # .detach().cpu().numpy()

        return layer_hook

    # * Assuming the model uses a transformer architecture, register hooks on each layer
    # * This needs adjustment based on the actual model architecture
    # if hasattr(model, "transformer"):
    #     for index, layer in enumerate(model.transformer.h):
    #         if index == 31:  # Adjust this index based on your specific needs
    #             layer.register_forward_hook(hook(index))

    # * Check if CUDA is available
    print(f"{torch.cuda.is_available() = }")

    # * Initialize model and tokenizer
    # model_name = "meta-llama/Meta-Llama-3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # * Register hooks using the defined hook function
    for index, layer in enumerate(model.model.layers):
        if index == 31:
            layer.register_forward_hook(hook(index))
            print("LAYER HOOKED")

    # * Prepare the input
    sequence = "camera eye eye eye camera smile tree tree tree ?"
    options = ["camera", "eye", "bone", "smile", "tree"]
    input_sequence = f"Sequence:{sequence}\nOptions: {', '.join(options)}"  # Sequence including options

    input_sequence = (
        "Finish the sequence by replacing the question mark (?) by the correct options out of the four given.\n"
        + input_sequence
    )

    inputs = tokenizer(input_sequence, return_tensors="pt")

    model.eval()
    # * Generate a response from the model
    with torch.no_grad():
        outputs = model(**inputs)

    # * Since outputs are generated by AutoModelForCausalLM, we need logits from outputs
    generated_text_ids = outputs.logits.argmax(-1)
    generated_text = tokenizer.decode(generated_text_ids[0], skip_special_tokens=True)

    print(f"\n\nInput_sequence: {input_sequence}\n\n")
    print(f"Model's response: {generated_text}")

    activations = activations

    code.interact(local=locals())


def what_happens_inside_pipeline():
    # * for ref, see: https://huggingface.co/docs/transformers/en/conversations
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # Prepare the input as before
    chat = [
        {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
        {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
    ]

    # 1: Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

    # 2: Apply the chat template
    formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    print("Formatted chat:\n", formatted_chat)

    # 3: Tokenize the chat (This can be combined with the previous step using tokenize=True)
    inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
    # Move the tokenized inputs to the same device the model is on (GPU/CPU)
    inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
    print("Tokenized inputs:\n", inputs)

    # 4: Generate text from the model
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.)
    print("Generated tokens:\n", outputs)

    # 5: Decode the output back to a string
    decoded_output = tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
    print("Decoded output:\n", decoded_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLMs & extract their activations")
    parser.add_argument("model_name", help="model name")
    parser.add_argument("--layers", nargs="+", help="")
    args = parser.parse_args()

    main(args.model_name)