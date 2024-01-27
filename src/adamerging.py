from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .heads import get_classification_head


def del_attr(obj, names: List[str]):
    """
    Deletes an attribute from an object recursively.

    Args:
        obj (object): Object to delete attribute from.
        names (list): List of attribute names to delete recursively.
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names: List[str], val):
    """
    Sets an attribute of an object recursively.

    Args:
        obj (object): Object to set attribute of.
        names (list): List of attribute names to set recursively.
        val (object): Value to set the attribute to.
    """
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)


def make_functional(mod: nn.Module):
    """
    Converts a PyTorch module into a functional module by removing all parameters and returning their names.

    Args:
        mod (nn.Module): PyTorch module to be converted.

    Returns:
        Tuple[Tensor]: Tuple containing the original parameters of the module.
        List[str]: List containing the names of the removed parameters.
    """
    orig_params = tuple(mod.parameters())
    names: List[str] = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names


def load_weights(mod, names, params):
    """
    Loads weights into a PyTorch module.

    Args:
        mod (nn.Module): PyTorch module to load weights into.
        names (list): List of parameter names to load weights into.
        params (tuple): Tuple of weights to load into the module.
    """
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, initial_weights=None):
        """
        Initializes a wrapper for a PyTorch model.

        Args:
            model (nn.Module): PyTorch model to wrap.
            initial_weights (optional): Initial weights for the model. Defaults to None.
        """
        super(ModelWrapper, self).__init__()
        self.model = model

        if hasattr(self.model, "transformer"):
            delattr(self.model, "transformer")

    def forward(self, images: Tensor) -> Tensor:
        """
        Forward pass of the wrapped PyTorch model.

        Args:
            images (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        features = self.model(images)
        return features


class AdaMerging(torch.nn.Module):  # Task-wise AdaMerging
    def __init__(self, paramslist: List[Tuple[Tensor]], model, names: List[str], exam_datasets: List[str], cfg):
        """
        Initializes an AdaMerging model.

        Args:
            paramslist (list): List of parameters for the model.
            model (nn.Module): PyTorch model to use.
            names (list): List of parameter names for the model.
            exam_datasets (list): List of exam datasets.
            args (argparse.Namespace): Arguments for the model.
        """
        super(AdaMerging, self).__init__()
        self.paramslist = paramslist
        self.model = model
        self.names = names
        self.pretrain_lambdas = torch.ones(1, 1)
        if cfg.sam_retraining:
            prior = 0.2
        else:
            prior = 0.3
        rlambdas = torch.ones(1, len(paramslist) - 1) * prior  # (1 * tasks)
        self.lambdas_raw = torch.nn.Parameter(rlambdas)

        self.classifier = []
        for dataset_name in exam_datasets:
            classification_head = get_classification_head(cfg, dataset_name)
            layer_name = "classifier_{}".format(dataset_name)
            self.add_module(layer_name, classification_head.to(cfg.device))
            self.classifier.append(layer_name)

    def lambdas(self):
        """
        Computes the lambdas for the model.

        Returns:
            Tensor: Tensor containing the lambdas.
        """
        task_lambdas = torch.clamp(self.lambdas_raw, min=0.0, max=1.0)
        lambdass = torch.cat((self.pretrain_lambdas, task_lambdas), 1)
        return lambdass

    def collect_trainable_params(self):
        """
        Collects the trainable parameters of the model.

        Returns:
            list: List of trainable parameters.
        """
        return [self.lambdas_raw]

    def get_classification_head(self, dataset_name: str):
        """
        Gets the classification head for a given dataset.

        Args:
            dataset_name (str): Name of the dataset.

        Returns:
            nn.Module: Classification head for the dataset.
        """
        layer_name = "classifier_{}".format(dataset_name)
        classification_head = getattr(self, layer_name)
        return classification_head

    def get_image_encoder(self):
        """
        Gets the image encoder for the model.

        Returns:
            nn.Module: Image encoder for the model.
        """
        alph = self.lambdas()
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[0].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        return self.model

    def forward(self, inp: Tensor, dataset_name: str):
        """
        Forward pass of the AdaMerging model.

        Args:
            inp (Tensor): Input tensor.
            dataset_name (str): Name of the dataset.

        Returns:
            Tensor: Output tensor.
        """
        alph = self.lambdas()
        params = tuple(sum(tuple(pi * lambdasi for pi, lambdasi in zip(p, alph[0].cpu()))) for j, p in enumerate(zip(*self.paramslist)))

        params = tuple(p.cuda(0) for p in params)
        load_weights(self.model, self.names, params)
        feature = self.model(inp)

        layer_name = "classifier_{}".format(dataset_name)
        classification_head = getattr(self, layer_name)
        out = classification_head(feature)

        return out


def softmax_entropy(x: Tensor):
    """
    Computes the softmax entropy of a tensor.

    Args:
        x (Tensor): Input tensor.

    Returns:
        Tensor: Softmax entropy of the input tensor.
    """
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
