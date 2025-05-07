import os

from typing import Optional, Any, cast, Callable

import torch

import numpy as np
import torch
import validators
import yaml
import pickle

from torch.optim.optimizer import Optimizer

from typing import Dict, Union, List, TypeVar, Tuple

import numpy as np
import matplotlib.pyplot as plt
import umap
import seaborn as sns
import re


from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity


T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")

NdTensor = Union[np.ndarray, torch.Tensor, List[T]]

Device = Union[torch.device, str]

ModuleOrOptimizer = Union[torch.nn.Module, Optimizer]

# word2idx, idx2word, embedding vectors
Embeddings = Tuple[Dict[str, int], Dict[int, str], np.ndarray]

ValidationResult = Union[validators.ValidationFailure, bool]

GenericDict = Dict[K, V]


def is_file(inp: Optional[str]) -> ValidationResult:
    if not inp:
        return False
    return os.path.isfile(inp)


def to_device(
    tt: torch.Tensor, device: Optional[Device] = "cpu", non_blocking: bool = False
) -> torch.Tensor:
    return tt.to(device, non_blocking=non_blocking)


def t_(
    data: NdTensor,
    dtype: torch.dtype = torch.float,
    device: Optional[Device] = "cpu",
    requires_grad: bool = False,
) -> torch.Tensor:
    """Convert a list or numpy array to torch tensor. If a torch tensor
    is passed it is cast to  dtype, device and the requires_grad flag is
    set IN PLACE.

    Args:
        data: (list, np.ndarray, torch.Tensor): Data to be converted to
            torch tensor.
        dtype: (torch.dtype): The type of the tensor elements
            (Default value = torch.float)
        device: (torch.device, str): Device where the tensor should be
            (Default value = 'cpu')
        requires_grad: bool): Trainable tensor or not? (Default value = False)

    Returns:
        (torch.Tensor): A tensor of appropriate dtype, device and
            requires_grad containing data

    """

    if isinstance(device, str):
        device = torch.device(device)

    tt = torch.as_tensor(data, dtype=dtype, device=device).requires_grad_(requires_grad)

    return tt


def t(
    data: NdTensor,
    dtype: torch.dtype = torch.float,
    device: Device = "cpu",
    requires_grad: bool = False,
) -> torch.Tensor:
    """Convert a list or numpy array to torch tensor. If a torch tensor
    is passed it is cast to  dtype, device and the requires_grad flag is
    set. This always copies data.

    Args:
        data: (list, np.ndarray, torch.Tensor): Data to be converted to
            torch tensor.
        dtype: (torch.dtype): The type of the tensor elements
            (Default value = torch.float)
        device: (torch.device, str): Device where the tensor should be
            (Default value = 'cpu')
        requires_grad: (bool): Trainable tensor or not? (Default value = False)

    Returns:
        (torch.Tensor): A tensor of appropriate dtype, device and
            requires_grad containing data

    """
    tt = torch.tensor(data, dtype=dtype, device=device, requires_grad=requires_grad)

    return tt


def mktensor(
    data: NdTensor,
    dtype: torch.dtype = torch.float,
    device: Device = "cpu",
    requires_grad: bool = False,
    copy: bool = True,
) -> torch.Tensor:
    """Convert a list or numpy array to torch tensor. If a torch tensor
        is passed it is cast to  dtype, device and the requires_grad flag is
        set. This can copy data or make the operation in place.

    Args:
        data: (list, np.ndarray, torch.Tensor): Data to be converted to
            torch tensor.
        dtype: (torch.dtype): The type of the tensor elements
            (Default value = torch.float)
        device: (torch.device, str): Device where the tensor should be
            (Default value = 'cpu')
        requires_grad: (bool): Trainable tensor or not? (Default value = False)
        copy: (bool): If false creates the tensor inplace else makes a copy
            (Default value = True)

    Returns:
        (torch.Tensor): A tensor of appropriate dtype, device and
            requires_grad containing data

    """
    tensor_factory = t if copy else t_

    return tensor_factory(data, dtype=dtype, device=device, requires_grad=requires_grad)


def from_checkpoint(
    checkpoint_file: Optional[str],
    obj: ModuleOrOptimizer,
    map_location: Optional[Device] = None,
) -> ModuleOrOptimizer:  # noqa: E501
    if checkpoint_file is None:
        return obj

    if not is_file(checkpoint_file):
        print(
            f"The checkpoint {checkpoint_file} you are trying to load "
            "does not exist. Continuing without loading..."
        )

        return obj
    state_dict = torch.load(checkpoint_file, map_location=map_location)

    if isinstance(obj, torch.nn.Module):
        if "model" in state_dict:
            state_dict = state_dict["model"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    if isinstance(obj, torch.optim.Optimizer) and "optimizer" in state_dict:
        state_dict = state_dict["optimizer"]
    obj.load_state_dict(state_dict)  # type: ignore

    return obj


def rotate_tensor(l: torch.Tensor, n: int = 1) -> torch.Tensor:
    return torch.cat((l[n:], l[:n]))


def shift_tensor(l: torch.Tensor, n: int = 1) -> torch.Tensor:
    out = rotate_tensor(l, n=n)
    out[-n:] = 0

    return out


def safe_mkdirs(path: str) -> None:
    """! Makes recursively all the directory in input path"""
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except Exception as e:
            print(e)
            raise IOError((f"Failed to create recursive directories: {path}"))


def yaml_load(fname: str) -> GenericDict:
    with open(fname, "r") as fd:
        data = yaml.load(fd, Loader=yaml.Loader) # Changed from data = yaml.load(fd)
    return data


def yaml_dump(data: GenericDict, fname: str) -> None:
    with open(fname, "w") as fd:
        yaml.dump(data, fd)


def pickle_load(fname: str) -> Any:
    with open(fname, "rb") as fd:
        data = pickle.load(fd)
    return data


def pickle_dump(data: Any, fname: str) -> None:
    with open(fname, "wb") as fd:
        pickle.dump(data, fd)


def pad_mask(lengths: torch.Tensor, max_length: Optional[int] = None, device="cpu"):
    """lengths is a torch tensor"""
    if max_length is None:
        max_length = cast(int, torch.max(lengths).item())
    max_length = cast(int, max_length)
    idx = torch.arange(0, max_length).unsqueeze(0).to(device)
    mask = (idx < lengths.unsqueeze(1)).float()
    return mask


def print_separator(
    symbol: str = "*", n: int = 10, print_fn: Callable[[str], None] = print
):
    print_fn(symbol * n)



# ==== Function to bin predictions ====
def bin_predictions(predictions):
    if isinstance(predictions, torch.Tensor):  # Convert tensor to NumPy
        predictions = predictions.detach().cpu().numpy()

    bins = [-np.inf, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, np.inf]
    labels = np.array([-3, -2, -1, 0, 1, 2, 3])

    bin_indices = np.digitize(predictions, bins, right=True) - 1  # Adjust indices
    return labels[bin_indices]  # Map to correct labels


# ==== Function to plot UMAP embeddings ====
def plot_umap(embeddings, predictions, title_before, title_after, ax1, ax2, save_path):
    # Convert predictions to NumPy if it's a tensor
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    
    # Fit and transform embeddings
    embedding_before = reducer.fit_transform(embeddings["before"])
    embedding_after = reducer.fit_transform(embeddings["after"])
    
    # Scatter plots
    scatter1 = ax1.scatter(embedding_before[:, 0], embedding_before[:, 1], c=predictions, cmap="viridis", alpha=0.7)
    scatter2 = ax2.scatter(embedding_after[:, 0], embedding_after[:, 1], c=predictions, cmap="viridis", alpha=0.7)
    
    # Titles and aesthetics
    ax1.set_title(title_before)
    ax2.set_title(title_after)
    ax1.set_xticks([]), ax1.set_yticks([])
    ax2.set_xticks([]), ax2.set_yticks([])
    
    # Add colorbars
    plt.colorbar(scatter1, ax=ax1, label="Binned Prediction Labels")
    plt.colorbar(scatter2, ax=ax2, label="Binned Prediction Labels")
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Figure saved to {save_path}")
def compute_metrics_embeddings(embeddings, labels):

        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        # Check for NaN or Inf
        if np.isnan(embeddings).any() or np.isinf(embeddings).any():
            print("Warning: Embeddings contain NaN or Inf. Skipping metric computation.")
            embeddings = fix_nan_inf(embeddings)

        # Reduce embeddings to match label count if necessary
        if embeddings.shape[0] != labels.shape[0]:
            embeddings = embeddings.reshape(1, -1)
        
        if embeddings.shape[0] < 2 or len(set(labels)) < 2:
            return None, None
        
        silhouette = silhouette_score(embeddings, labels)
        # Normalize embeddings before cosine similarity
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Compute cosine similarity within clusters
        cos_sim_within = []
        unique_labels = np.unique(labels)

        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            if len(cluster_indices) > 1:
                cluster_embeddings = norm_embeddings[cluster_indices]
                cos_sim_matrix = cosine_similarity(cluster_embeddings)
                cos_sim_within.append(np.mean(cos_sim_matrix[np.triu_indices_from(cos_sim_matrix, k=1)]))

        mean_cosine_similarity = np.mean(cos_sim_within) if cos_sim_within else None

        # Compute Davies-Bouldin Index (Lower is better)
        dbi = davies_bouldin_score(embeddings, labels)

        
        
        return silhouette, mean_cosine_similarity, dbi

def truncate_embeddings(embeddings_list):
        min_dim = min(emb.shape[1] for emb in embeddings_list)  # Find minimum feature dimension
        return [emb[:, :min_dim] for emb in embeddings_list]  # Truncate all to the same size
    

def fix_nan_inf(embeddings):
        """
        Intelligently replace NaNs and Infs in embeddings:
        - NaNs are replaced with the mean of the non-NaN values in the corresponding dimension
        - Infs are replaced with the max/min finite value based on the sign
        """
        # Create a copy to avoid modifying the original array
        fixed_embeddings = embeddings.copy()
        
        # Handle NaNs
        for dim in range(fixed_embeddings.shape[1]):
            column = fixed_embeddings[:, dim]
            nan_mask = np.isnan(column)
            
            if nan_mask.any():
                # Replace NaNs with the mean of non-NaN values in that dimension
                non_nan_mean = np.nanmean(column)
                fixed_embeddings[nan_mask, dim] = non_nan_mean
        
        # Handle Infs
        for dim in range(fixed_embeddings.shape[1]):
            column = fixed_embeddings[:, dim]
            pos_inf_mask = np.isinf(column) & (column > 0)
            neg_inf_mask = np.isinf(column) & (column < 0)
            
            if pos_inf_mask.any():
                # Replace positive infinities with the max finite value
                max_finite = np.max(column[~np.isinf(column)])
                fixed_embeddings[pos_inf_mask, dim] = max_finite
            
            if neg_inf_mask.any():
                # Replace negative infinities with the min finite value
                min_finite = np.min(column[~np.isinf(column)])
                fixed_embeddings[neg_inf_mask, dim] = min_finite
        
        return fixed_embeddings