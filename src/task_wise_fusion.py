R"""
```python
# Get the task-wise weights
task_wise_weights = get_task_wise_weights(num_models)

# Define the task vectors (in this case, we'll use the state_dict of the pretrained model)
task_vectors = ...

# Initialize the TaskWiseMergedModel
merged_model = TaskWiseMergedModel(pretrained_model, task_wise_weights, task_vectors)

# Now you can use the merged_model like a regular PyTorch model
outputs = merged_model(inputs)
```
"""
import logging
import types
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, Iterator, List

import torch
from torch import Tensor, nn
from torch.func import functional_call

from .ties_merging_utils import check_parameterNamesMatch
from .type import StateDict
from .utils import timeit_context

log = logging.getLogger(__name__)

__all__ = ["get_task_wise_weights", "fuse_weights", "TaskWiseMergedModel"]


def get_task_wise_weights(num_models: int, init_values: float = None):
    """
    This function generates a tensor of weights for each model.

    Args:
        num_models (int): The number of models.
        init_values (float, optional): The initial value for each weight. Defaults to None.

    Returns:
        Tensor: A tensor of weights for each model.
    """
    assert num_models >= 1, f"num_models must be >= 1, got {num_models}"
    if init_values is None:
        init_values = 1.0 / num_models
    return torch.full((num_models,), init_values, dtype=torch.float32)


def _fuse_weights(task_wise_weight: Tensor, tensors: List[Tensor]):
    """
    This function fuses the weights of the models.

    Args:
        task_wise_weight (Tensor): The weights for each model.
        tensors (List[Tensor]): The list of tensors to be fused.

    Returns:
        Tensor: The fused weights.
    """
    device = task_wise_weight.device
    return sum(task_wise_weight[i] * w.to(device) for i, w in enumerate(tensors))


def fuse_weights(task_wise_weight: Tensor, state_dicts: List[StateDict]) -> StateDict:
    """
    This function fuses the weights of the models and returns a state dictionary.

    Args:
        task_wise_weight (Tensor): The weights for each model. on cuda or cpu.
        state_dicts (List[StateDict]): The list of state dictionaries. on cpu.

    Returns:
        StateDict: The fused state dictionary.
    """
    num_models = len(state_dicts)
    assert task_wise_weight.dim() == 1, f"task_wise_weight must be a 1D tensor, got {task_wise_weight.dim()}"
    assert num_models == task_wise_weight.size(
        0
    ), f"num_models must be equal to the number of state_dicts, got {num_models} and {task_wise_weight.size(0)}"
    return {k: _fuse_weights(task_wise_weight, [sd[k] for sd in state_dicts]) for k in state_dicts[0].keys()}


class TaskWiseMergedModel(nn.Module):
    def __init__(
        self,
        pretrained_model: nn.Module,
        task_wise_weight: Tensor,
        task_vectors: List[StateDict],
        clamp_weights: bool = True,
    ):
        super().__init__()
        self._model = (pretrained_model,)  # self._model should be on cpu

        self.task_wise_weight = nn.Parameter(task_wise_weight, requires_grad=True)
        self.task_vectors = task_vectors  # should be on cpu
        self.pretrained_state_dict: StateDict = self.model.state_dict(keep_vars=False)
        check_parameterNamesMatch(self.task_vectors)
        self.clamp_weights = clamp_weights
        self.merged_state_dict = None

    @property
    def model(self):
        return self._model[0]

    def merge_weights(self):
        if self.clamp_weights:
            task_wise_weight = self.task_wise_weight.clamp(0, 1)
        else:
            task_wise_weight = self.task_wise_weight
        device = task_wise_weight.device
        task_vector = fuse_weights(task_wise_weight, self.task_vectors)
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
