from copy import deepcopy
from typing import Any, Dict, List, Optional, cast

import torch
from torch import Tensor, nn


def get_by_name(model: nn.Module, name: str):
    name = name.split(".")
    for n in name[:-1]:
        model = getattr(model, n)
    return getattr(model, name[-1])


def set_by_name(model: nn.Module, name: str, obj: Any):
    name = name.split(".")
    for n in name[:-1]:
        model = getattr(model, n)
    if isinstance(p := getattr(model, name[-1]), nn.Parameter):
        # if obj is a Parameter, set the data of the parameter
        p.data = obj
    else:
        setattr(model, name[-1], obj)


def human_readable(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "%.2f%s" % (num, ["", "K", "M", "B", "T", "P"][magnitude])


def print_trainable_parameters(model, verbose: bool = False):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            if verbose:
                print(f"{name}: {param.numel()}")
            trainable_params += param.numel()

    print(
        f"trainable params: {human_readable(trainable_params)} || all params: {human_readable(all_param)} || trainable%: {100 * trainable_params / all_param}"
    )
