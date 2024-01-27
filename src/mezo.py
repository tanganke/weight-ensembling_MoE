from typing import Callable

import numpy as np
import torch
from torch.optim.optimizer import Optimizer

from .utils import timeit_context


class MeZO(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-5,
        weight_decay: float = 0,
        eps: float = 1e-3,
    ):
        defaults = dict(eps=eps, lr=lr, weight_decay=weight_decay)
        super(MeZO, self).__init__(params, defaults)

    def step(
        self,
        closure: Callable,
    ):
        assert isinstance(closure, Callable), "closure should be provided"
        zo_random_seed = np.random.randint(1000000000)

        # First function evaluation
        self._zo_perturb_parameters(zo_random_seed, scaling_factor=1)

        with torch.inference_mode():
            loss1 = closure()
            if loss1 is None:
                raise ValueError("closure returned None (should return loss)")

        # Second function evaluation
        self._zo_perturb_parameters(zo_random_seed, scaling_factor=-2)

        with torch.inference_mode():
            loss2 = closure()

        # Compute projected gradient
        projected_grad = (loss1 - loss2) / (2 * self.defaults["eps"])
        if isinstance(projected_grad, torch.Tensor):
            projected_grad = projected_grad.item()

        # Reset model back to its parameters at start of step
        self._zo_perturb_parameters(zo_random_seed, scaling_factor=1)

        with timeit_context("MeZO update"):
            self._zo_update_parameters(zo_random_seed, projected_grad)
        return loss1, projected_grad

    def _zo_perturb_parameters(self, random_seed: int, scaling_factor: float):
        """
        Perturbs the parameters of the Zeroth Order Optimization algorithm.

        Args:
            random_seed (int): The random seed to use for the perturbation.
            scaling_factor (float): The scaling factor to use for the perturbation.

        Returns:
            None
        """
        torch.manual_seed(random_seed)

        for group in self.param_groups:
            for param in group["params"]:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                param.data = param.data + group["eps"] * scaling_factor * z

    def _zo_update_parameters(self, random_seed: int, projected_grad: float):
        # Update parameters
        torch.manual_seed(random_seed)
        for group in self.param_groups:
            for param in group["params"]:
                z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                param.data = param.data - group["lr"] * (projected_grad * z + group["weight_decay"] * param.data)
