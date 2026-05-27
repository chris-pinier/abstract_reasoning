from collections import defaultdict
from typing import *
import functools

import einops
from tqdm import tqdm
import torch
import torch.nn.functional as F
import nnsight
from nnterp import StandardizedTransformer
from nnterp.prompt_utils import get_first_tokens

from .eval_utils import spearman_rho_torch
from .dataset_builder import AbstractTask

def flush_torch_ram(func):
    """Decorator to flush torch RAM after function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
    return wrapper

def convert_bfloat(func):
    """Decorator to convert bfloat tensors to their corresponding float types."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, torch.Tensor) and 'bfloat' in str(result.dtype):
            target_dtype = str(result.dtype).split('.')[-1].replace('bfloat', 'float')
            return result.to(getattr(torch, target_dtype))
        return result
    return wrapper

def condense_matrix(X, n=None):
    '''
    Condense a square matrix into a condensed vector
    '''
    n = X.size(0) if n is None else n
    inds = torch.triu_indices(n, n, offset=1)
    return X[inds[0], inds[1]]
    
@torch.no_grad
def get_completions(
    model: StandardizedTransformer, 
    dataset: AbstractTask,
    logging: bool = True,
) -> List[str]:
    with model.lm.session(remote=model.remote_run) as sess:
        completion_ids = []
        for idx, (prompts, _) in enumerate(dataset):
            if logging:
                print(f"Batch: {idx+1} / {len(dataset)}")
            with model.lm.trace(prompts) as t:
                logits = model.lm.lm_head.output[:, -1]
                completion_ids.extend([logits.log_softmax(dim=-1).argmax(dim=-1).save()])
    return model.lm.tokenizer.batch_decode(torch.cat(completion_ids))

@torch.no_grad
def intervene_with_vec(
    model: StandardizedTransformer,
    dataset: AbstractTask,
    vector: torch.Tensor,
    layers: List[int] = None,
    token: int = -1,
    remote: bool = True,
    logging: bool = True
) -> Dict[int, List[str]]:
    T = token
    with model.lm.session(remote=remote) as sess:
        intervention_top_ind = {layer: [] for layer in layers}
        for prompts, _ in dataset:
            for layer in layers:
                if logging:
                    print(f'Layer: {layer}')
                with model.lm.trace(prompts) as t:
                    # Add the vector to the residual stream, at the last sequence position
                    hidden_states = model.lm.model.layers[layer].output[0]
                    hidden_states[:, T] += vector
                    # Get correct logprobs
                    logits = model.lm.lm_head.output[:, -1]
                    intervention_top_ind[layer].extend([logits.log_softmax(dim=-1).argmax(dim=-1).save()])
    
    # Decode the completions
    return {layer: model.lm.tokenizer.batch_decode(torch.stack(ind).squeeze()) for layer, ind in intervention_top_ind.items()}

@torch.no_grad
def intervene_and_get_probs(
    model: StandardizedTransformer,
    dataset: AbstractTask,
    vector: torch.Tensor,
    layer: int,
    token: int = -1,
    remote: bool = True,
    logging: bool = True
) -> Tuple[List[str], List[float]]:
    T = token
    with model.lm.session(remote=remote) as sess:
        completion_ids = nnsight.list().save()
        y_logits = nnsight.list().save()
        for prompts, y in dataset:
            y_ids = model.config['get_first_token_ids'](y)
            with model.lm.trace(prompts) as t:
                # Add the vector to the residual stream, at the last sequence position
                hidden_states = model.lm.model.layers[layer].output[0]
                hidden_states[:, T] += vector
                # Get correct logprobs
                logits = model.lm.lm_head.output[:, -1].log_softmax(dim=-1)
                completion_ids.extend(logits.argmax(dim=-1).tolist())
                y_logits.extend(logits[torch.arange(len(prompts)), y_ids].tolist())

    completions = model.lm.tokenizer.batch_decode(torch.tensor(completion_ids))
    y_probs = torch.tensor(y_logits).exp().tolist()
    return completions, y_probs

OUT_PROJ_NAMES = ['o_proj', 'c_proj', 'out_proj', 'dense']

def get_out_proj(model: StandardizedTransformer, layer: int):
    """Get the output projection module for a given layer, regardless of architecture."""
    attn = model.layers[layer].self_attn
    for name in OUT_PROJ_NAMES:
        if hasattr(attn, name):
            return getattr(attn, name)
    raise ValueError(
        f"Could not find output projection in self_attn. "
        f"Tried: {OUT_PROJ_NAMES}. Available: {[n for n, _ in attn._module.named_children()]}"
    )

def get_att_out_proj_input(
    model: StandardizedTransformer,
    layer: int,
    token: int = -1,
) -> torch.Tensor:
    """Get the input to the output projection, reshaped to (batch, n_heads, d_head)."""
    proj = get_out_proj(model, layer)
    z = proj.input[:, token]
    return z.view(z.shape[0], model.num_heads, -1)

def get_avg_att_output(
    model: StandardizedTransformer,
    layer: int,
    heads: List[int],
    token: int = -1,
) -> torch.Tensor:
    heads_to_ablate = list(set(range(model.num_heads)) - set(heads))
    proj = get_out_proj(model, layer)
    z = get_att_out_proj_input(model, layer, token)
    z_ablated = z.mean(dim=0)
    z_ablated[heads_to_ablate, :] = 0.0
    z_ablated = z_ablated.view(-1)
    return proj(z_ablated)

def get_att_output_per_item(
    model: StandardizedTransformer,
    layer: int,
    heads: List[int],
    token: int = -1,
) -> torch.Tensor:
    heads_to_ablate = list(set(range(model.num_heads)) - set(heads))
    proj = get_out_proj(model, layer)
    z = get_att_out_proj_input(model, layer, token)
    z_ablated = z.clone()
    z_ablated[:, heads_to_ablate, :] = 0.0
    z_ablated = z_ablated.view(z.shape[0], -1)
    return proj(z_ablated)

@torch.no_grad
def get_summed_vec_per_item(
    model: StandardizedTransformer,
    dataset: AbstractTask,
    heads: Dict[int, List[int]],
    token: int = -1,
) -> torch.Tensor:
    with model.session() as sess:
        all_head_outputs = [].save()
        
        for batched_prompts, _ in tqdm(dataset, total=len(dataset), desc="Getting summed vector per item"):
            
            with model.trace(batched_prompts) as t:
                batch_head_outputs = []
                for layer, head_list in sorted(heads.items()):
                    out_proj_output = get_att_output_per_item(model, layer, head_list, token=token)
                    batch_head_outputs.append(out_proj_output.cpu())
                
                # Stack and sum for this batch (all on CPU now)
                batch_summed = torch.stack(batch_head_outputs).sum(dim=0)
                all_head_outputs.append(batch_summed)

    return torch.cat(all_head_outputs, dim=0).to(model.device)

@torch.no_grad
@convert_bfloat
def get_summed_vec_simmat(
    model: StandardizedTransformer,
    dataset: AbstractTask,
    heads: Dict[int, List[int]],
    token: int = -1,
    logging: bool = True
) -> torch.Tensor:
    with model.lm.session(remote=model.remote_run) as sess:
        all_head_outputs = nnsight.list()
        
        for idx, (batched_prompts, _) in enumerate(dataset):
            if logging:
                print(f"Batch: {idx+1} / {len(dataset)}")
            
            with model.lm.trace(batched_prompts) as t:
                batch_head_outputs = []
                for layer, head_list in heads.items():
                    out_proj_output = get_att_output_per_item(model, layer, head_list, token=token)
                    batch_head_outputs.append(out_proj_output)
                
                # Stack and sum for this batch
                batch_summed = torch.stack(batch_head_outputs).sum(dim=0)
                batch_summed = batch_summed if model.remote_run else batch_summed.cpu()
                all_head_outputs.append(batch_summed)
        
        simmat = compute_similarity_matrix(torch.cat(all_head_outputs, dim=0))
        simmat_condensed = condense_matrix(simmat, n=len(dataset.prompts)).save()

    return simmat_condensed.cpu()

@torch.no_grad
def get_avg_summed_vec(
    model: StandardizedTransformer, 
    dataset: AbstractTask,
    heads: List[Tuple[int, int, float]],
    token: int = -1,
    remote: bool = True,
    logging: bool = True
) -> torch.Tensor:
    '''
    Get the relation vector by summing the output of the most influential attention heads for the 
    model's output tokens given a prompt or list of prompts. Either per item or averaged across the batch.

    Args:
        model (StandardizedTransformer): The model object
        prompts (List[str]): The prompt or list of prompts to run the model on
        average_heads (bool): Whether to average across the heads in a batch or get a relation vector per item (default: False)
    Returns:
        torch.Tensor: The relation vectors for the model's output tokens (shape: [batch_size, resid_dim])
    '''
    # Turn head_list into a dict of {layer: heads we need in this layer}
    head_dict = defaultdict(set)
    for layer, head in heads:
        head_dict[layer].add(head)
    head_dict = dict(head_dict)

    relation_vec_list = []
    with model.lm.session(remote=remote) as sess:
        with model.lm.trace(dataset.prompts) as runner:
            for layer, head_list in head_dict.items():        
                out_proj_output = get_avg_att_output(model, layer, head_list, token=token)
                relation_vec_list.append(out_proj_output.save())

    # Sum all the attention heads per item
    relation_vec_tensor = torch.stack(relation_vec_list) # (len(head_list), D_MODEL) if average_heads else (len(head_list), B, D_MODEL)
    relation_vecs = torch.sum(relation_vec_tensor, dim=0)

    return relation_vecs

@flush_torch_ram
def compute_similarity_matrix(vectors: torch.Tensor) -> torch.Tensor:
    norm_v = F.normalize(vectors, p=2, dim=1)
    return torch.matmul(norm_v, torch.transpose(norm_v, 0, 1))

@flush_torch_ram
@torch.no_grad 
@convert_bfloat
def get_all_head_simmats(
    model: StandardizedTransformer,
    dataset: AbstractTask,
    token: int = -1,
) -> torch.Tensor:
    """Get similarity matrices for all (layer, head) combinations.
    
    Returns:
        Tensor of shape (n_layers, n_heads, n_items, n_items)
    """
    layers = list(range(model.num_layers))
    heads = list(range(model.num_heads))

    with model.session() as sess:

        simmat_dict = {(layer, head): [] for layer in layers for head in heads}
        for prompts, _ in tqdm(dataset, total=len(dataset), desc="Getting hidden states"):
            with model.trace(prompts) as t:
                for layer in layers:
                    att_out = get_att_out_proj_input(model, layer, token)
                    for head in heads:
                        act = att_out[:, head].cpu()
                        simmat_dict[(layer, head)].append(act)

        simmats = [[None for _ in heads] for _ in layers].save()
        for (layer, head), v in tqdm(simmat_dict.items(), desc="Computing similarity matrices"):
            v = torch.concat(v).to(model.device)
            simmat = compute_similarity_matrix(v)
            simmats[layer][head] = condense_matrix(simmat, n=len(dataset.prompts)).cpu()
    
    return torch.stack([torch.stack(simmats[layer]) for layer in range(len(layers))])


@flush_torch_ram
@torch.no_grad 
@convert_bfloat
def get_head_simmats(
    model: StandardizedTransformer,
    dataset: AbstractTask,
    heads: List[Tuple[int, int]],
    token: int = -1,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """Get similarity matrices for specific (layer, head) pairs.
    
    Args:
        pairs: List of (layer, head) tuples, e.g. [(60, 26), (55, 63), (60, 40)]
    
    Returns:
        Dict mapping (layer, head) -> condensed similarity matrix
    """
    layer_to_heads = defaultdict(set)
    for layer, head in heads:
        layer_to_heads[layer].add(head)
    layers_sorted = sorted(layer_to_heads.keys())

    with model.session() as sess:

        act_dict = {(l, h): [] for l, h in heads}
        for prompts, _ in tqdm(dataset, total=len(dataset), desc="Getting hidden states"):
            with model.trace(prompts) as t:
                for layer in layers_sorted:
                    att_out = get_att_out_proj_input(model, layer, token)
                    for head in layer_to_heads[layer]:
                        act = att_out[:, head].cpu()
                        act_dict[(layer, head)].append(act)

        simmats = {}.save()
        for (layer, head), v in tqdm(act_dict.items(), desc="Computing similarity matrices"):
            v = torch.concat(v).to(model.device)
            simmats[(layer, head)] = compute_similarity_matrix(v)
    
    return simmats

@torch.no_grad
def get_rsa(
    model: StandardizedTransformer,
    dataset: AbstractTask,
    design_matrix: torch.Tensor,
    layers: Optional[List[int]] = None,
    token: int = -1,
    remote: bool = True,
    logging: bool = True
) -> torch.Tensor:
    layers = range(model.config['n_layers']) if (layers is None) else layers
    heads = range(model.config['n_heads'])
    N_HEADS = model.config['n_heads']
    D_HEAD = model.config['resid_dim'] // N_HEADS

    with model.lm.session(remote=remote) as sess:        
            
        print(f"Extracting hidden states ...")
        simmat_dict = {(layer, head): [] for layer in layers for head in heads}
        for batched_prompts, _ in dataset:            
            # Collect the hidden states for each head
            with model.lm.trace(batched_prompts) as t:
                for layer in layers:
                    # Get hidden states, reshape to get head dimension, store the mean tensor
                    out_proj = model.config['out_proj'](layer)
                    z = out_proj.inputs[0][0][:, token]
                    z_reshaped = z.reshape(len(batched_prompts), N_HEADS, D_HEAD)
                    for head in heads:
                        z_head = z_reshaped[:, head]
                        simmat_dict[(layer, head)].extend([z_head])

        print(f"Computing similarity matrices ...")
        for k, v in simmat_dict.items():
            simmat_dict[k] = compute_similarity_matrix(torch.concat(v))
        
        # Get the upper triangular indices of the similarity matrix
        n = len(dataset.prompts)
        inds = torch.triu_indices(n, n, offset=1)
        design_matrix_condensed = design_matrix[inds[0], inds[1]]

        print(f"Computing RSA ...")
        rsa_vals = nnsight.list([[] for _ in layers]).save()
        for i, layer in enumerate(layers):
            for head in heads:
                v_condensed = simmat_dict[(layer, head)][inds[0], inds[1]]
                rho = spearman_rho_torch(v_condensed, design_matrix_condensed)
                rsa_vals[i].append(rho)
    
    return torch.tensor(rsa_vals)

@flush_torch_ram
@torch.no_grad
def calculate_AP(
    model: StandardizedTransformer,
    dataset: AbstractTask,
    layers: Optional[List[int]] = None,
) -> torch.Tensor:
    '''
    Returns a tensor of shape (layers, heads), containing the CIE for each head.

    Inputs:
        model: LanguageModel
            the transformer you're doing this computation with
        dataset: AbstractTask
            the dataset of clean prompts from which we'll extract the function vector (we'll also create a
            corrupted version of this dataset for interventions)
        layers: Optional[List[int]]
            the layers which this function will calculate the score for (if None, we assume all layers)
    '''
    dataset.create_corrupted_dataset()
    layers = range(model.num_layers) if (layers is None) else layers
    heads = range(model.num_heads)

    N_HEADS = model.num_heads
    T = -1 # values taken from last token
        
    with model.session() as sess:
        z_dict = {(layer, head): [] for layer in layers for head in heads}
        correct_probs_corrupted = [].save()
        
        for (prompts, completions), (prompts_corrupted, _) in dataset:
            correct_completion_ids = torch.tensor([
                model.tokenizer.encode(c, add_special_tokens=False)[0] for c in completions
            ])
            # Run a forward pass on corrupted prompts, where we don't intervene or store activations (just so we can
            # get the correct-token logprobs to compare with our intervention)  
            with model.trace(prompts_corrupted) as t:
                probs = model.next_token_probs[torch.arange(len(prompts)), correct_completion_ids]
                correct_probs_corrupted.extend([probs])

            # Run a forward pass on clean prompts, where we store attention head outputs
            with model.trace(prompts) as t:
                for layer in layers:
                    # Get hidden states, reshape to get head dimension, store the mean tensor
                    z = get_att_out_proj_input(model, layer, T)
                    for head in heads:
                        z_head = z[:, head]
                        z_dict[(layer, head)].extend([z_head])

        # Get the mean of the head activations
        z_dict = {
            k: torch.cat(v).mean(dim=0)
            for k, v in z_dict.items()
        }
        
        # For each head, run a forward pass on corrupted prompts (here we need multiple different forward passes
        correct_probs_dict = {(layer, head): [] for layer in layers for head in heads}.save()
        for (prompts, completions), (prompts_corrupted, _) in dataset:
            correct_completion_ids = torch.tensor([
                model.tokenizer.encode(c, add_special_tokens=False)[0] for c in completions
            ])
            # For each head, run a forward pass on corrupted prompts (here we need multiple different forward passes,
            # because we're doing different interventions each time)
            for layer in tqdm(layers, desc="Patching layers"):
                for head in heads:
                    with model.trace(prompts_corrupted) as t:
                        # Get hidden states, reshape to get head dimension, then set it to the a-vector
                        z = get_att_out_proj_input(model, layer, T)
                        z[:, head] = z_dict[(layer, head)]
                        # Get probs at the end, which we'll compare with our corrupted probs
                        correct_probs = model.next_token_probs[torch.arange(len(prompts)), correct_completion_ids]
                        correct_probs_dict[(layer, head)].append(correct_probs.cpu())

    # Get difference between intervention probs and corrupted probs, and take mean over batch dim
    all_correct_probs_intervention = einops.rearrange(
        torch.stack([torch.cat(v) for v in correct_probs_dict.values()]),
        "(layers heads) batch -> layers heads batch",
        layers = len(layers),
    )
    correct_probs_corrupted = torch.cat(correct_probs_corrupted).cpu()
    return all_correct_probs_intervention - correct_probs_corrupted # [layers heads batch]
    

@flush_torch_ram
@torch.no_grad
def calculate_cross_format_CIE(
    model: StandardizedTransformer,
    clean_dataset: AbstractTask,
    target_dataset: AbstractTask,
    layers: Optional[List[int]] = None,
) -> torch.Tensor:
    '''
    Returns a tensor of shape (layers, heads), containing the Cross-Format CIE for each head.
    
    Extracts mean activations from clean_dataset.
    Patches them into corrupted version of target_dataset.
    '''
    target_dataset.create_corrupted_dataset()
    layers = range(model.config['n_layers']) if (layers is None) else layers
    heads = range(model.config['n_heads'])

    N_HEADS = model.config['n_heads']
    D_HEAD = model.config['resid_dim'] // N_HEADS
    T = -1 # values taken from last token
        
    with model.lm.session(remote=model.remote_run) as sess:
        z_dict = {(layer, head): [] for layer in layers for head in heads}
        
        # 1. Extract activations from Clean Dataset
        print("Extracting activations from clean dataset ...")
        for prompts, _ in clean_dataset:
            with model.lm.trace(prompts) as t:
                for layer in layers:
                    out_proj = model.config['out_proj'](layer)
                    z = out_proj.inputs[0][0][:, T]
                    z_reshaped = z.reshape(len(prompts), N_HEADS, D_HEAD)
                    for head in heads:
                        z_head = z_reshaped[:, head]
                        z_dict[(layer, head)].extend([z_head])

        # Get the mean of the head activations
        z_dict = {
            k: torch.cat(v).mean(dim=0)
            for k, v in z_dict.items()
        }
        
        # 2. Compute Baseline on Corrupted Target Dataset
        # We need P(y | corrupted_prompt)
        correct_logprobs_corrupted = []
        print("Computing baseline on corrupted target dataset ...")
        for (prompts, completions), (prompts_corrupted, _) in target_dataset:
            correct_completion_ids = model.config['get_first_token_ids'](completions)
            with model.lm.trace(prompts_corrupted) as t:
                logits = model.lm.lm_head.output[:, -1]
                correct_logprobs_corrupted.extend([logits.log_softmax(dim=-1)[torch.arange(len(prompts)), correct_completion_ids].save()])
        
        # 3. Compute Patched Performance on Corrupted Target Dataset
        # We need P(y | corrupted_prompt, patched_activation)
        correct_logprobs_dict = {(layer, head): [] for layer in layers for head in heads}
        print("Computing patched performance ...")
        
        for (prompts, completions), (prompts_corrupted, _) in target_dataset:
            correct_completion_ids = model.config['get_first_token_ids'](completions)
            
            for layer in layers:
                print(f"Layer: {layer}")
                for head in heads:
                    with model.lm.trace(prompts_corrupted) as t:
                        out_proj = model.config['out_proj'](layer)
                        z = out_proj.inputs[0][0][:, T]
                        z.reshape(len(prompts), N_HEADS, D_HEAD)[:, head] = z_dict[(layer, head)]
                        
                        logits = model.lm.lm_head.output[:, -1]
                        correct_logprobs = logits.log_softmax(dim=-1)[torch.arange(len(prompts)), correct_completion_ids]
                        correct_logprobs = correct_logprobs if model.remote_run else correct_logprobs.cpu()
                        correct_logprobs_dict[(layer, head)].append(correct_logprobs.save())

    # Get difference between intervention logprobs and corrupted logprobs, and take mean over batch dim
    all_correct_logprobs_intervention = einops.rearrange(
        torch.stack([torch.cat([t.value for t in v]) for v in correct_logprobs_dict.values()]),
        "(layers heads) batch -> layers heads batch",
        layers = len(layers),
    )
    correct_logprobs_corrupted = torch.cat([p.value for p in correct_logprobs_corrupted]) if model.remote_run else torch.cat([p.value for p in correct_logprobs_corrupted]).cpu()
    logprobs_diff = all_correct_logprobs_intervention - correct_logprobs_corrupted # shape [layers heads batch]

    # Return mean effect of intervention, over the batch dimension
    return logprobs_diff.mean(dim=-1)