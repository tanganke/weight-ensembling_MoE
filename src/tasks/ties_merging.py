import copy
import logging
from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

log = logging.getLogger(__name__)


def normalize(tensor: Tensor, dim: int = 0, eps: float = 1e-8) -> Tensor:
    """
    Normalizes a tensor along a given dimension.

    Args:
        tensor (Tensor): The tensor to normalize.
        dim (int, optional): The dimension along which to normalize the tensor. Defaults to 0.
        eps (float, optional): A small value to add to the denominator to avoid division by zero. Defaults to 1e-8.

    Returns:
        Tensor: The normalized tensor.
    """
    return tensor / torch.clamp(torch.norm(tensor, dim=dim, keepdim=True), min=eps)


def state_dict_to_vector(state_dict: Dict[str, Tensor], remove_keys: List[str] = []):
    R"""
    Converts a PyTorch state dictionary to a 1D tensor.

    Args:
        state_dict (Dict[str, Tensor]): A dictionary containing the state of a PyTorch model.
        remove_keys (List[str], optional): A list of keys to remove from the state dictionary before converting it to a tensor. Defaults to [].

    Returns:
        Tensor: A 1D tensor containing the values of the state dictionary, sorted by key.
    """
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in shared_state_dict:
            del shared_state_dict[key]
    sorted_shared_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return torch.nn.utils.parameters_to_vector([value.reshape(-1) for key, value in sorted_shared_state_dict.items()])


def vector_to_state_dict(vector: Tensor, state_dict: Dict[str, Tensor], remove_keys: List[str] = []) -> Dict[str, Tensor]:
    """
    Converts a 1D tensor to a PyTorch state dictionary.

    Args:
        vector (Tensor): A 1D tensor containing the values of the state dictionary, sorted by key.
        state_dict (Dict[str, Tensor]): A dictionary containing the state of a PyTorch model.
        remove_keys (List[str], optional): A list of keys to remove from the state dictionary before converting it to a tensor. Defaults to [].

    Returns:
        Dict[str, Tensor]: A dictionary containing the state of a PyTorch model, with the values of the state dictionary replaced by the values in the input tensor.
    """
    # create a reference dict to define the order of the vector
    reference_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        if key in reference_dict:
            del reference_dict[key]
    sorted_reference_dict = OrderedDict(sorted(reference_dict.items()))

    # create a shared state dict using the refence dict
    torch.nn.utils.vector_to_parameters(vector, sorted_reference_dict.values())

    return sorted_reference_dict


def topk_values_mask(M: torch.Tensor, K: float = 0.7, return_mask: bool = False):
    """
    Returns a tensor with the top k values of each row of the input tensor M, where k is a fraction of the number of columns.

    Args:
        M (torch.Tensor): A 2D tensor of shape (n, d) where n is the number of rows and d is the number of columns.
        K (float, optional): The fraction of the number of columns to keep. Defaults to 0.7.
        return_mask (bool, optional): Whether to return the mask tensor used to select the top k values. Defaults to False.

    Returns:
        torch.Tensor: A tensor of the same shape as M with the top k values of each row.
        torch.Tensor: A 1D tensor of shape (n,) containing the mean of the mask tensor for each row.
        torch.Tensor: A tensor of the same shape as M with True for the top k values of each row and False otherwise. Only returned if return_mask is True.
    """
    assert M.dim() <= 2, "M must be a 1D or 2D tensor"
    if K > 1:
        K /= 100
        log.warning("K is a percentage, not a fraction. Dividing by 100.")

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    _, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements
    if k <= 0:
        k = 1

    # Find the k-th smallest element by magnitude for each row
    # kthvalue: https://pytorch.org/docs/stable/generated/torch.kthvalue.html
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask: torch.Tensor = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)


def resolve_zero_signs(sign_to_mult: Tensor, method="majority"):
    """
    Resolves zero signs in a tensor of signs that will be multiplied together.

    Args:
        sign_to_mult (torch.Tensor): A 1D tensor of signs to be multiplied together.
        method (str, optional): The method to use for resolving zero signs. Can be "majority" or "minority". Defaults to "majority".

    Returns:
        torch.Tensor: A 1D tensor of signs with zero signs resolved according to the specified method.
    """
    majority_sign = torch.sign(sign_to_mult.sum())
    if majority_sign == 0:
        majority_sign = 1

    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    else:
        raise ValueError(f"Method {method} is not defined.")
    return sign_to_mult


def resolve_sign(M: Tensor) -> Tensor:
    """
    Computes the majority sign of the input tensor and resolves zero signs using the "majority" method.

    Args:
        Tensor (torch.Tensor): A 2D tensor of shape (n, d) where n is the number of rows and d is the number of columns.

    Returns:
        torch.Tensor: A 1D tensor of shape (d,) containing the majority sign of each column, with zero signs resolved using the "majority" method.
    """
    sign_to_mult = torch.sign(M.sum(dim=0))  # \gamma_m^p
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult


def disjoint_merge(M: Tensor, merge_func: str, sign_to_mult: Tensor):
    """
    Merges the entries of a tensor M that correspond to disjoint sets.

    Args:
        M (torch.Tensor): A 2D tensor of shape (n, d) where n is the number of sets and d is the number of entries per set.
        merge_func (str): The merge function to use. Can be "mean", "sum", or "max".
        sign_to_mult (torch.Tensor, optional): A 1D tensor of signs to be multiplied with the selected entries. Defaults to None.

    Returns:
        torch.Tensor: A 1D tensor of shape (d,) containing the merged entries.
    """
    # Extract the merge function from the input string
    merge_func = merge_func.split("-")[-1].lower()
    assert merge_func in [
        "mean",
        "sum",
        "max",
    ], f"Merge method {merge_func} is not defined."

    # If sign is provided then we select the corresponding entries and aggregate.
    # If `sign_to_mult` is not None, the function creates a boolean tensor `rows_to_keep` that has the same shape as `M`.
    # The values in `rows_to_keep` are `True` for entries in `M` that have the same sign as the corresponding entry in `sign_to_mult`, and `False` otherwise.
    if sign_to_mult is not None:
        rows_to_keep = torch.where(sign_to_mult.unsqueeze(0) > 0, M > 0, M < 0)
        selected_entries = M * rows_to_keep
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = M != 0
        selected_entries = M * rows_to_keep

    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(non_zero_counts, min=1)
    elif merge_func == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    elif merge_func == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        disjoint_aggs *= sign_to_mult
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return disjoint_aggs


def ties_merging(
    flat_task_checks: Tensor,
    reset_thresh: float = None,
    merge_func: str = "mean",
):
    """
    Merges the task checks of a flat tensor using the TIES algorithm.

    Args:
        flat_task_checks (torch.Tensor): A 2D tensor of shape (n, d) where n is the number of tasks and d is the number of checks per task.
        reset_thresh (float, optional): The threshold for resetting the task checks (the top-K% parameters to be keeped, if this is 1, keep all the parameters).
            Should be a float between 0 and 1.
            If None, no resetting is performed. Defaults to None.
        merge_func (str, optional): The merge function to use for aggregating the task checks.
            Can be "mean", "sum", or "max". Defaults to "".

    Returns:
        torch.Tensor: A 1D tensor of shape (d,) containing the merged task checks.
    """
    all_checks = flat_task_checks.clone()

    # 1. Trim
    updated_checks, *_ = topk_values_mask(all_checks, K=reset_thresh, return_mask=False)

    # 2. Elect
    log.debug(f"RESOLVING SIGN")
    final_signs = resolve_sign(updated_checks)
    assert final_signs is not None

    # 3. Disjoint Merge
    log.debug(f"Disjoint AGGREGATION: {merge_func}")
    merged_tv = disjoint_merge(updated_checks, merge_func, final_signs)

    return merged_tv


def _test_ties_merging():
    # Create a tensor of flat task checks with positive and negative entries
    flat_task_checks = torch.tensor([[1, -2, 3, -4], [5, 6, -7, -8], [-9, 10, 11, -12]])

    # Test with reset_thresh=None and merge_func="mean"
    merged_tv = ties_merging(flat_task_checks, reset_thresh=0.7, merge_func="mean")

    # Test with reset_thresh=2 and merge_func="max"
    merged_tv = ties_merging(flat_task_checks, reset_thresh=0.7, merge_func="max")

    # Test with reset_thresh=3 and merge_func="sum"
    merged_tv = ties_merging(flat_task_checks, reset_thresh=0.7, merge_func="sum")


if __name__ == "__main__":
    _test_ties_merging()
