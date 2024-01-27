R"""
This module provides functions for fusing the weights of multiple deep neural networks block-wisely.
"""
import logging
from copy import deepcopy
# imports
from typing import Any, Dict, Iterator, List

import torch
from torch import Tensor, nn
from torch.func import functional_call

from .ties_merging_utils import StateDict, check_parameterNamesMatch
from .utils import timeit_context

__all__ = [
    "get_block_wise_weights_by_partitions",
    "get_block_wise_weights_by_block_size",
    "fuse_weights",
    "num_partitions_per_model",
    "BlockWiseMergedModel",
]

log = logging.getLogger(__name__)


# To construct a unified model by fusing the weights of multiple deep neural networks block-wise, where each layer is split into several partitions
def _get_block_wise_weights_by_partitions(weights: Tensor, num_partitions: int, num_models: int) -> Tensor:
    """
    Returns a tensor of block-wise weights for a given tensor of weights.

    Args:
        weights (Tensor): The tensor of weights to compute block-wise weights for.
        num_partitions (int): The number of partitions to split the tensor into.
        num_models (int): The number of models to compute block-wise weights for.

    Returns:
        Tensor: A tensor of block-wise weights with shape (num_models, num_partitions) if the weights tensor can be evenly split,
        otherwise a tensor of shape (num_models, 1) with a value of 1.0.
    """
    if weights.dim() >= 1 and weights.size(0) % num_partitions == 0:
        return torch.ones(num_models, num_partitions, dtype=torch.float32) / num_models
    else:
        return torch.ones(num_models, 1, dtype=torch.float32) / num_models


def get_block_wise_weights_by_partitions(params: List[Tensor] | Dict[str, Tensor], num_partitions: int, num_models: int):
    """
    Returns a list or dictionary of block-wise weights for a given list or dictionary of tensors of weights.

    Args:
        params (List[Tensor] | Dict[str, Tensor]): The list or dictionary of tensors of weights to compute block-wise weights for.
        num_partitions (int): The number of partitions to split the tensors into.
        num_models (int): The number of models to compute block-wise weights for.

    Returns:
        List[Tensor] | Dict[str, Tensor]: A list or dictionary of tensors of block-wise weights with the same length or keys as the input list or dictionary if the weights tensors can be evenly split,
        otherwise a list or dictionary of tensors of size 1 with a value of 1.0.
    """
    if isinstance(params, list):
        return [_get_block_wise_weights_by_partitions(p, num_partitions, num_models) for p in params]
    elif isinstance(params, dict):
        return {k: _get_block_wise_weights_by_partitions(p, num_partitions, num_models) for k, p in params.items()}
    else:
        raise TypeError(f"params must be a list or a dict, but got {type(params)}")


def _get_block_wise_weights_by_block_size(weights: Tensor, block_size: int, num_models: int, init_value: float = None) -> Tensor:
    """
    Returns a tensor of block-wise weights for a given tensor of weights.

    Args:
        weights (Tensor): The tensor of weights to compute block-wise weights for.
        block_size (int): The size of each block.
        num_models (int): The number of models to compute block-wise weights for.

    Returns:
        Tensor: A tensor of block-wise weights with shape (num_models, num_partitions) if the weights tensor can be evenly split,
        otherwise a tensor of shape (num_models, 1) with a value of 1.0.
    """
    if weights.dim() >= 1 and weights.size(0) % block_size == 0:
        size = (num_models, weights.size(0) // block_size)
    else:
        size = (num_models, 1)
    if init_value is None:
        return torch.ones(*size, dtype=torch.float32) / num_models
    else:
        return torch.ones(*size, dtype=torch.float32) * init_value


def get_block_wise_weights_by_block_size(params: List[Tensor] | Dict[str, Tensor], block_size: int, num_models: int, init_value: float = None):
    """
    Returns a list or dictionary of block-wise weights for a given list or dictionary of tensors of weights.

    Args:
        params (List[Tensor] | Dict[str, Tensor]): The list or dictionary of tensors of weights to compute block-wise weights for.
        block_size (int): The size of each block.
        num_models (int): The number of models to compute block-wise weights for.
        init_value (float, optional): The initial value of the block-wise weights. Defaults to None.

    Returns:
        List[Tensor] | Dict[str, Tensor]: A list or dictionary of tensors of block-wise weights with the same length or keys as the input list or dictionary if the weights tensors can be evenly split,
        otherwise a list or dictionary of tensors of size 1 with a value of 1.0.
    """
    if isinstance(params, list):
        return [_get_block_wise_weights_by_block_size(p, block_size, num_models, init_value=init_value) for p in params]
    elif isinstance(params, dict):
        return {k: _get_block_wise_weights_by_block_size(p, block_size, num_models, init_value=init_value) for k, p in params.items()}
    else:
        raise TypeError(f"params must be a list or a dict, but got {type(params)}")


def _fuse_weights(block_wise_weights: Tensor, model_weights: List[Tensor]) -> Tensor:
    """
    Fuses a list of model weights using block-wise weights.

    Args:
        block_wise_weights (Tensor): A tensor of block-wise weights with shape (num_models, num_partitions).
        model_weights (List[Tensor]): A list of tensors of model weights to fuse.

    Returns:
        Tensor: A tensor of fused model weights with the same shape as the input tensors.
    """
    assert block_wise_weights.dim() == 2, f"block_wise_weights must be a 2D tensor, but got {block_wise_weights.dim()}D"
    assert block_wise_weights.size(0) == len(
        model_weights
    ), f"block_wise_weights must have a size of {len(model_weights)} in the first dimension, but got {block_wise_weights.size(0)}"

    num_models, num_partitions = block_wise_weights.size()
    if num_partitions == 1:
        return sum(block_wise_weights[i, 0] * model_weights[i].to(block_wise_weights.device) for i in range(num_models))
    else:
        ans = torch.zeros_like(model_weights[0], device=block_wise_weights.device)
        block_size = model_weights[0].size(0) // num_partitions
        for i in range(num_models):
            for j in range(num_partitions):
                ans[block_size * j : block_size * (j + 1)] += block_wise_weights[i, j] * model_weights[i][block_size * j : block_size * (j + 1)].to(
                    block_wise_weights.device
                )
        return ans


def fuse_weights(block_wise_weight: Dict[str, Tensor], state_dicts: List[StateDict]):
    """
    Returns a state dict of fused weights for a given list of state dicts of weights.

    Args:
        block_wise_weight (Dict[str, Tensor]): The dictionary of tensors of block-wise weights to fuse.
        state_dicts (List[STATE_DICT]): The list of state dicts of weights to fuse.

    Returns:
        STATE_DICT: A state dict of fused weights.
    """
    assert isinstance(state_dicts, list), f"state_dicts must be a list, but got {type(state_dicts)}"
    check_parameterNamesMatch([block_wise_weight] + state_dicts)
    return {k: _fuse_weights(block_wise_weight[k], [state_dict[k] for state_dict in state_dicts]) for k in state_dicts[0]}


def num_partitions_per_model(block_wise_weight: Dict[str, Tensor]):
    """
    Returns the number of partitions per model for a given dictionary of tensors of block-wise weights.

    Args:
        block_wise_weight (Dict[str, Tensor]): The dictionary of tensors of block-wise weights.

    Returns:
        int: The number of partitions per model.
    """
    assert isinstance(block_wise_weight, dict), f"block_wise_weight must be a dict, but got {type(block_wise_weight)}"
    total_num_partitions = 0
    for k, weight in block_wise_weight.items():
        assert weight.dim() == 2, f"block_wise_weight[{k}] must be a 2D tensor, but got {weight.dim()}D"
        total_num_partitions += weight.size(1)
    return total_num_partitions


def remove_dot_from_keys(tensor_dict: Dict[str, Any]):
    return {k.replace(".", "|"): v for k, v in tensor_dict.items()}


class BlockWiseMergedModel(nn.Module):
    def __init__(
        self,
        pretrained_model: nn.Module,
        block_wise_weight: Dict[str, Tensor],
        task_vectors: List[StateDict],
        clamp_weights: bool = True,
    ):
        super().__init__()
        self.model = deepcopy(pretrained_model)
        for p in self.model.parameters():
            p.requires_grad = False

        self.block_wise_weight = {k: nn.Parameter(v, requires_grad=True) for k, v in block_wise_weight.items()}
        self.task_vectors = task_vectors
        self.pretrained_state_vector = self.model.state_dict()
        check_parameterNamesMatch([self.pretrained_state_vector] + self.task_vectors)
        self.clamp_weights = clamp_weights
        self._state_dict = None

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        return iter(self.block_wise_weight.values())

    def _get_merged_vector(self):
        """
        Returns the fused weights of the model by applying the block-wise fusion algorithm.

        Returns:
            The fused weights of the model.
        """
        if self.clamp_weights:
            block_wise_weight = {k: torch.clamp(p, min=0, max=1) for k, p in self.block_wise_weight.items()}
        else:
            block_wise_weight = self.block_wise_weight
        return fuse_weights(block_wise_weight, self.task_vectors)

    def merge_weights(self):
        """
        Merges the weights of the model.
        Call this after each update step.
        """
        with timeit_context(loglevel=logging.DEBUG):
            log.debug("Merging weights")
            task_vector = self._get_merged_vector()
            self._state_dict = {
                k: (task_vector[k] + self.pretrained_state_vector[k]).cuda(non_blocking=True) for k in self.pretrained_state_vector.keys()
            }

    def forward(self, *args, **kwargs):
        if self._state_dict is None:
            self.merge_weights()
        return functional_call(
            self.model,
            self._state_dict,
            args=args,
            kwargs=kwargs,
        )
