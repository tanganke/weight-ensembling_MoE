import logging
from typing import Iterator, List, Optional

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from .task_vectors import StateDict

log = logging.getLogger(__name__)


class ConcreteMask(nn.Module):
    """
    This class represents a ConcreteMask, which is a type of mask that can be applied to a state dictionary / task vector.
    It is used to create a mask for each parameter in the state dictionary and apply it to the state dictionary.

    Attributes:
        temperature (float): The temperature parameter for the RelaxedBernoulli distribution.
        masks (nn.ParameterDict): A dictionary of masks for each parameter in the state dictionary.
    """

    def __init__(
        self,
        temperature: float,
        state_dict: StateDict,
        init_value: float = 5.0,
        draw_sample: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        masks = {}
        for k, v in state_dict.items():
            masks[k] = nn.Parameter(torch.ones_like(v) * init_value, requires_grad=True)
            init_device = v.device
        self.masks = masks
        self.draw_sample = draw_sample

    def _draw_mask(self, binary_mask: Optional[bool] = False):
        """
        Draws a mask based on the current state of the object.

        This function uses a relaxed Bernoulli distribution to draw a mask. If `binary_mask` is True,
        the function will return a binary mask. Otherwise, it will return a mask based on the probabilities
        from the distribution.

        Parameters:
            binary_mask (bool, optional): If True, the function will return a binary mask. Defaults to False.

        Returns:
            dict: A dictionary where the keys are the same as the keys in `self.masks` and the values are the drawn masks.
        """
        concrete_masks = {}
        for k in self.masks.keys():
            concrete_dist = torch.distributions.RelaxedBernoulli(
                self.temperature,
                logits=self.masks[k],
            )
            if binary_mask == True:
                concrete_mask: Tensor = (concrete_dist.sample()).detach_() > 0.5
            else:
                if self.draw_sample:
                    # this is slow on cpu
                    concrete_mask = concrete_dist.rsample()
                else:
                    concrete_mask = concrete_dist.probs
            concrete_masks[k] = concrete_mask
        return concrete_masks

    def _apply_mask(self, concrete_masks, state_dict: StateDict):
        """
        This method applies the mask to the state dictionary and rescale it.

        Args:
            concrete_masks (StateDict): The concrete masks to be applied.
            state_dict (StateDict): The state dictionary to which the mask will be applied.

        Returns:
            StateDict: The state dictionary after the mask has been applied.
        """
        new_state_dict = {}
        for k, v in state_dict.items():
            concrete_mask = concrete_masks[k]
            new_state_dict[k] = v * concrete_mask / concrete_mask.mean()
        return new_state_dict

    def apply_mask(self, state_dicts: List[StateDict], concrete_masks: Optional[StateDict] = None):
        """
        This method applies the mask to the state dictionary and rescales it.

        Args:
            state_dict (StateDict): The state dictionary to which the mask will be applied.

        Returns:
            StateDict: The state dictionary after the mask has been applied and rescaled.
        """
        # draw common mask
        if concrete_masks is None:
            concrete_masks = self._draw_mask()

        _mask_on_device = {}

        def mask_on_device(device: torch.device):
            if device in _mask_on_device:
                return _mask_on_device[device]
            else:
                _mask_on_device[device] = {k: v.to(device, non_blocking=True) for k, v in concrete_masks.items()}
                return _mask_on_device[device]

        # mask and rescale
        new_state_dicts = []
        for state_dict in state_dicts:
            device = next(iter(state_dict.values())).device
            new_state_dict = self._apply_mask(mask_on_device(device), state_dict)
            new_state_dicts.append(new_state_dict)
        return new_state_dicts

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.masks.values()

    def to(self, device):
        for k in self.masks.keys():
            self.masks[k] = self.masks[k].to(device)
        return self
