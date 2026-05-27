import itertools
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from rich.console import Console
from rich.panel import Panel
from matplotlib import pyplot as plt
from nnterp import StandardizedTransformer

from .query_utils import get_completions, get_all_head_simmats, get_summed_vec_per_item, compute_similarity_matrix
from .eval_utils import SimilarityMatrix, create_design_matrix, rsa as compute_rsa
from .dataset_builder import AbstractTask


class Acc:
    def __init__(self, model: StandardizedTransformer, datasets: AbstractTask):
        self.model = model
        self.datasets = datasets
        self.errors = []
        self.Ys = []
        self.prompts = []

    @property
    def completions(self):
        if not hasattr(self, '_completions'): self._completions = get_completions(self.model, self.datasets)
        return self._completions
    
    @completions.setter
    def completions(self, completions: List[str]): self._completions = completions

    def load_completions(self, path: str):
        with open(path, 'r') as f:
            completions = [line.strip('\n') for line in f.readlines()]
        assert len(completions) == len(self.datasets.prompts), f'Expected {len(self.datasets.prompts)} completions, got {len(completions)}'
        self.completions = completions
    
    def save_completions(self, path: str):
        with open(path, 'w') as f:
            for completion in self.completions:
                f.write(completion + '\n')
    
    @property
    def acc_df(self):
        if not hasattr(self, '_acc_df'): self._acc_df = self._create_acc_df()
        return self._acc_df
    
    def _create_acc_df(self):
        if not hasattr(self, 'acc'): self._evaluate()
        df = pd.DataFrame(self.acc)
        df['dataset'] = df['dataset'].astype(str)
        df['concept'] = df['dataset'].apply(lambda x: [d.task.concept for d in self.datasets.datasets if d.dataset_name == x][0])
        df['list'] = df['dataset'].apply(lambda x: [d.task.sequence_name for d in self.datasets.datasets if d.dataset_name == x][0])
        df['shuffle_prop'] = df['dataset'].apply(lambda x: [d.task.shuffle_prop for d in self.datasets.datasets if d.dataset_name == x][0])
        return df[['dataset', 'concept', 'list', 'shuffle_prop', 'accuracy']]

    def _evaluate(self):
        self.acc, self.correct = [], []
        for i, dataset in enumerate(self.datasets.datasets):
            model_completions = self._completions[dataset.size * i : dataset.size * (i+1)]
            acc, correct = accuracy_completions(self.model, model_completions, dataset.completions, return_correct=True)
            self.acc.append({'dataset': dataset.dataset_name, 'accuracy': acc})
            self.correct.append(correct)
    
    def print_errors(self, dataset_index: int = None):
        console = Console()
        for i, dataset in enumerate(self.datasets.datasets):
            if dataset_index is not None and i != dataset_index:
                continue
            model_completions = self._completions[dataset.size * i : dataset.size * (i+1)]
            correct = self.correct[i]
            incorrect_indices = [j for j, c in enumerate(correct) if not c]
            if incorrect_indices:
                console.print(f"[bold]Errors in dataset {dataset.dataset_name}:[/bold]")
                for n, idx in enumerate(incorrect_indices):
                    y = self.datasets.datasets[i].completions[idx]
                    tokenized_completion = self.model.config['get_first_token'](y)
                    panel = Panel.fit(
                        f"[bold]Prompt:[/bold] {dataset.prompts[idx]}\n\n"
                        f"[bold]Model Completion:[/bold] [u]{model_completions[idx]}[/u]\n"
                        f"[bold]Expected Completion:[/bold] [u]{y}[/u] ([u]{tokenized_completion}[/u])",
                        title=f"Error {n+1} in {dataset.dataset_name}",
                        title_align="left",
                        padding=(1, 2),
                        border_style="red"
                    )
                    console.print(panel)


class RSA:
    def __init__(
            self,      
            model: StandardizedTransformer, 
            dataset: AbstractTask
    ):
        self.model = model
        self.dataset = dataset
        self.design_matrix = create_design_matrix(self.dataset.prompts.pattern)
        self.sorted_by = None

    def __repr__(self):
        return f'RSA(\n\tmodel={self.model}\n\tsimmats={self.simmats.shape}\n\tdatasets={self.dataset}\n\tsorted_by={self.sorted_by})'
        
    @property
    def simmats(self):
        if not hasattr(self, '_simmats'): 
            self._simmats = get_all_head_simmats(self.model, self.dataset).numpy() 
        return self._simmats
    
    @simmats.setter
    def simmats(self, simmats: np.ndarray): self._simmats = simmats

    @property
    def rsa(self):
        if not hasattr(self, '_rsa'): 
            self._rsa = self.batch_rsa(self.simmats, self.design_matrix)
        return self._rsa

    @rsa.setter
    def rsa(self, rsa: np.ndarray): self._rsa = rsa

    def _apply_sort(self, matrix: np.ndarray) -> np.ndarray:
        """Apply sort order to a similarity matrix (condensed or square)."""
        if matrix.ndim == 1:
            n = len(self.dataset.prompts.names)
            triu_i, triu_j = np.triu_indices(n, k=1)
            square = np.zeros((n, n))
            square[triu_i, triu_j] = matrix
            square[triu_j, triu_i] = matrix
            np.fill_diagonal(square, 1)
            matrix = square
        if getattr(self, '_sort_idx', None) is not None:
            matrix = matrix[self._sort_idx][:, self._sort_idx]
        return matrix

    def _sorted(self, items: list) -> list:
        """Apply sort order to a list (e.g. names, pattern)."""
        if getattr(self, '_sort_idx', None) is not None:
            return [items[i] for i in self._sort_idx]
        return items

    def _make_simmat(self, matrix: np.ndarray) -> SimilarityMatrix:
        """Build a SimilarityMatrix with sort-aware names and pattern."""
        return SimilarityMatrix(
            sim_mat=self._apply_sort(matrix),
            tasks=self._sorted(self.dataset.prompts.names),
            attribute_list=self._sorted(self.dataset.prompts.pattern),
        )

    def get_cv_simmat(self, top_k: int = 5):
        heads = [(layer, head) for layer, head in self.top_heads(top_k)]
        heads_dict = defaultdict(set)
        for layer, head in heads:
            heads_dict[layer].add(head)
        cv_vecs = get_summed_vec_per_item(self.model, self.dataset, heads_dict)
        cv_simmat = compute_similarity_matrix(cv_vecs).cpu().numpy()
        cv_rsa = compute_rsa(self.design_matrix, cv_simmat)
        return self._make_simmat(cv_simmat), cv_rsa

    def top_heads(self, k: int = 1, return_rsa: bool = False) -> List[Tuple[int, int]] | List[Tuple[int, int, float]]:
        """
        Get the k layer and head indices with the highest RSA correlation.
        Returns:
            List[Tuple[int, int]] | List[Tuple[int, int, float]]: List of k (layer_index, head_index) tuples in descending order
        """
        indices = np.argsort(self.rsa, axis=None)[-k:][::-1] 
        if return_rsa:
            return [(int(layer), int(head), self.rsa[layer, head].item()) 
                    for layer, head in [np.unravel_index(idx, self.rsa.shape) for idx in indices]]
        return [(int(layer), int(head)) 
                for layer, head in [np.unravel_index(idx, self.rsa.shape) for idx in indices]]

    def batch_rsa(self, simmats: torch.Tensor, design_matrix: np.ndarray) -> np.ndarray:
        n_layers, n_components = simmats.shape[0], simmats.shape[1]
        rsa_results = [[[] for _ in range(n_components)] for _ in range(n_layers)]
        for layer in tqdm(range(n_layers), desc="Computing RSA"):
            for head in range(n_components):
                rsa_results[layer][head] = compute_rsa(design_matrix, simmats[layer, head])            
        rsa_results = np.nan_to_num(np.array(rsa_results), nan=0.0) # Replace NaN values with 0
        return rsa_results
    
    # def filter_simmats_by_tasks(self, tasks: Union[str, List[str]]) -> 'RSA':
    #     tasks_to_keep = set(tasks) if isinstance(tasks, list) else {tasks}
        
    #     # Filter datasets
    #     datasets_to_keep = [d for d in self.dataset.datasets if d.dataset_name in tasks_to_keep]
    #     if not datasets_to_keep: raise ValueError(f'No datasets found for tasks: {tasks_to_keep}')
    #     datasets = ICLDatasetCollection(datasets_to_keep)

    #     # Vectorized calculation of flattened item indices
    #     item_indices_to_keep = [i for i, task in enumerate(self.dataset.prompts.names) if task in tasks_to_keep]
    #     pairs = np.array(list(itertools.combinations(item_indices_to_keep, 2)))
    #     i, j = pairs[:, 0], pairs[:, 1]
    #     n_total = len(self.dataset.prompts.names)
    #     flattened_indices_to_keep = (n_total * i - i * (i + 1) // 2 + j - 1 - i).astype(int)

    #     _rsa = RSA(self.model, datasets)
    #     _rsa.simmats = self.simmats[..., flattened_indices_to_keep]
    #     return _rsa
    
    _SORT_FIELDS = {'pattern', 'alphabet', 'format'}

    def sort_by(self, by: str = 'pattern'):
        """Set a sort order for lazy reordering of similarity matrices.

        Args:
            by: Prompts field to sort by — 'pattern', 'alphabet', or 'format'.
                Pass None to revert to original ordering.
        """
        if by is None:
            self._sort_idx = None
            self.sorted_by = None
            return
        assert by in self._SORT_FIELDS, \
            f"by must be one of {self._SORT_FIELDS}, got '{by}'"
        keys = getattr(self.dataset.prompts, by)
        self._sort_idx = np.argsort(keys, kind='stable')
        self.sorted_by = by

    def get_head_simmat(self, layer: int, head: int) -> SimilarityMatrix:
        """Get a SimilarityMatrix for a single head, applying lazy sort if set."""
        return self._make_simmat(self.simmats[layer, head])

    def plot_rsa(self, title: str = None):
        plt.imshow(self.rsa.T, cmap='viridis')
        plt.xlabel('Heads')
        plt.ylabel('Layers')
        plt.colorbar()
        plt.title(title if title else '')
        plt.show()

    def plot_top_heads(self, k: int = 5, labels: List[str] = None):
        top_heads = self.top_heads(k)
        fig, axes = plt.subplots(nrows=1, ncols=k, figsize=(10 * k, 10),)
        vmin = min([
            np.min(self.simmats[layer, head]) for layer, head in top_heads
        ])
        vmax = max([
            np.max(self.simmats[layer, head]) for layer, head in top_heads
        ])
  
        for i, (layer, head) in enumerate(top_heads):
            simmat = self.get_head_simmat(layer, head)
            simmat.plot(
                axis=axes[i],
                norm=(vmin, vmax),
                title=f'Layer {layer}, Head {head} - RSA: {self.rsa[layer, head]:.3f}'
            )        
            if labels is not None:
                axes[i].set_xticklabels(labels, rotation=90, ha='right')
                axes[i].set_yticklabels(labels, rotation=0, ha='right')
        
        plt.show()

    def load_simmats(self, path: str):
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.lib.npyio.NpzFile):
            full = np.zeros(data['full_shape'], dtype=data['simmats'].dtype)
            for idx, simmat in zip(data['indices'], data['simmats']):
                full[idx[0], idx[1]] = simmat
            self.simmats = full
        else:
            self.simmats = data

    def load_rsa(self, path: str): self.rsa = np.load(path)

    def save_simmats(self, path: str, top_n: int = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if top_n is None:
            np.save(path, self.simmats)
            return
        heads = self.top_heads(top_n)
        indices = np.array(heads, dtype=np.intp)
        top_simmats = np.stack([self.simmats[l, h] for l, h in heads])
        np.savez(path, simmats=top_simmats, indices=indices, full_shape=self.simmats.shape)

    def save_rsa(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.rsa)