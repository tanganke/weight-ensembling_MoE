import logging
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, Iterator, List

import torch
from torch import Tensor, nn
from torch.func import functional_call

from .ties_merging_utils import check_parameterNamesMatch
from .type import StateDict
from .utils import timeit_context

__all__ = ["get_layer_wise_weights", "fuse_weights", "LayerWiseMergedModel"]

log = logging.getLogger(__name__)


def get_layer_wise_weights(num_models: int, num_layers: int, init_values: float = None):
    """
    Return a tensor of layer-wise weights for the given number of models and layers.

    Args:
        num_models (int): The number of models to fuse.
        num_layers (int): The number of layers in each model.
        init_values (float, optional): The initial value for each weight. Defaults to 1.0 / num_models.

    Returns:
        Tensor: A tensor of shape (num_models, num_layers) containing the layer-wise weights.
    """
    assert num_models >= 1, f"num_models must be >= 1, got {num_models}"
    assert num_layers >= 1, f"num_layers must be >= 1, got {num_layers}"
    if init_values is None:
        init_values = 1.0 / num_models
    return torch.full((num_models, num_layers), init_values, dtype=torch.float32)


def _fuse_weights(layer_wise_weight: Tensor, tensors: List[Tensor]):
    """
    Fuse the layer-wise weights with the given state dictionaries.

    Args:
        layer_wise_weight (Tensor): A tensor of shape (num_models,) containing the layer-wise weights.
        state_dicts (List[Tensor]): A list of state dictionaries, each containing the weights for a single layer.

    Returns:
        Tensor: A tensor of shape (num_params,) containing the fused weights.
    """
    assert len(layer_wise_weight) == len(tensors), f"layer_wise_weight.shape={layer_wise_weight.shape}, len(tensors)={len(tensors)}"
    return sum(layer_wise_weight[i] * w.to(layer_wise_weight.device) for i, w in enumerate(tensors))


def fuse_weights(layer_wise_weight: Tensor, state_dicts: List[StateDict]) -> StateDict:
    """
    Fuse the weights of multiple models using layer-wise fusion.

    Args:
        layer_wise_weight (Tensor): A tensor of shape (num_models, num_layers) representing the weight of each layer for each model.
        state_dicts (List[StateDict]): A list of state dictionaries, one for each model.

    Returns:
        A dictionary mapping each weight tensor key to the fused weight tensor.
    """
    num_models = len(state_dicts)
    num_layers = len(state_dicts[0])
    assert layer_wise_weight.shape == (
        num_models,
        num_layers,
    ), f"layer_wise_weight.shape={layer_wise_weight.shape}, expected (num_models, num_layers): ({num_models}, {num_layers})"
    return {k: _fuse_weights(layer_wise_weight[:, i], [state_dict[k] for state_dict in state_dicts]) for i, k in enumerate(state_dicts[0].keys())}


class LayerWiseMergedModel(nn.Module):
    def __init__(
        self,
        pretrained_model: nn.Module,
        layer_wise_weight: Tensor,
        task_vectors: List[StateDict],
        clamp_weights: bool = True,
    ):
        super().__init__()
        self._model = (pretrained_model,)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.layer_wise_weight = nn.Parameter(layer_wise_weight, requires_grad=True)
        self.task_vectors = task_vectors
        self.pretrained_state_dict = self.model.state_dict()
        check_parameterNamesMatch(self.task_vectors)
        self.merged_state_dict = None
        self.clamp_weights = clamp_weights

    @property
    def model(self):
        return self._model[0]

    def merge_weights(self):
        """
        Merges the weights of the model.
        Call this after each update step.
        """
        if self.clamp_weights:
            layer_wise_weight = self.layer_wise_weight.clamp(0, 1)
        else:
            layer_wise_weight = self.layer_wise_weight
        device = layer_wise_weight.device
        task_vector = fuse_weights(layer_wise_weight, self.task_vectors)
        self.merged_state_dict = {k: self.pretrained_state_dict[k].to(device, non_blocking=True) for k in self.pretrained_state_dict.keys()}
        for k in task_vector.keys():
            self.merged_state_dict[k] += task_vector[k]

    def forward(self, *args, **kwargs):
        if self.merged_state_dict is None:
            self.merge_weights()
        return functional_call(
            self.model,
            self.merged_state_dict,
            args=args,
            kwargs=kwargs,
            tie_weights=False,
        )

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            attr = getattr(self.model, name)
            if isinstance(attr, Callable):
                warnings.warn(f"forwarding `{name}` to the underlying model", UserWarning)
            return attr

    def __setattr__(self, name: str, value: Any) -> None:
        try:
            super().__setattr__(name, value)
        except AttributeError:
            setattr(self.model, name, value)
