import logging
from pathlib import Path
from typing import Dict, List, Tuple

import open_clip.model
import torch
from torch import Tensor, nn
from typing_extensions import TypeAlias

StateDict: TypeAlias = Dict[str, Tensor]

log = logging.getLogger(__name__)


class TaskVector:
    def __init__(
        self,
        pretrained_checkpoint: str = None,
        finetuned_checkpoint: str = None,
        vector: StateDict = None,
    ):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.

        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.

        Args:
            pretrained_checkpoint (str, optional): Path to the pretrained checkpoint file. Defaults to None.
            finetuned_checkpoint (str, optional): Path to the finetuned checkpoint file. Defaults to None.
            vector (STATE_DICT, optional): The task vector state dict. Defaults to None.

        Raises:
            ValueError: If both `pretrained_checkpoint` and `vector` are None.
            ValueError: If both `finetuned_checkpoint` and `vector` are None.
            ValueError: If both `pretrained_checkpoint` and `finetuned_checkpoint` are None.
            ValueError: If `pretrained_checkpoint` or `finetuned_checkpoint` is not a valid file path.
        """
        if isinstance(pretrained_checkpoint, Path):
            pretrained_checkpoint = str(pretrained_checkpoint)
        if isinstance(finetuned_checkpoint, Path):
            finetuned_checkpoint = str(finetuned_checkpoint)

        if vector is not None:
            self.vector = vector
        else:
            # construct the task vector from the pretrained and finetuned checkpoints
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                log.info("TaskVector: " + finetuned_checkpoint)
                pretrained_state_dict = torch.load(pretrained_checkpoint, map_location="cpu").state_dict()
                finetuned_state_dict = torch.load(finetuned_checkpoint, map_location="cpu").state_dict()
                self.vector = {}
                for key in pretrained_state_dict:
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]

    def __add__(self, other: "TaskVector"):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other: "TaskVector"):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = -self.vector[key]
        return TaskVector(vector=new_vector)

    def weightmerging(self, taskvectors, coefficients):
        with torch.no_grad():
            new_vector = {}
            for key in taskvectors[0].vector:
                new_vector[key] = sum(coefficients[k] * taskvectors[k][key] for k in range(len(taskvectors)))
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint: str, scaling_coef: float = 1.0) -> nn.Module:
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model: nn.Module = torch.load(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f"Warning: key {key} is present in the pretrained state dict but not in the task vector")
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model


def state_dict_mean(state_dicts: List[StateDict]) -> StateDict:
    """Compute the mean of a list of state dicts."""
    with torch.no_grad():
        mean_state_dict = {}
        for key in state_dicts[0]:
            mean_state_dict[key] = sum(state_dict[key] for state_dict in state_dicts) / len(state_dicts)
    return mean_state_dict
