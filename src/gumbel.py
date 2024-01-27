"""
The `gumbel_sigmoid` function is adapted from https://github.com/AngelosNal/PyTorch-Gumbel-Sigmoid/
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def sample_gumbel(size, eps: float = 1e-20, device=torch.device("cpu")):
    U = torch.rand(size, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits: Tensor, temperature: float):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits: Tensor, temperature: float, hard: bool = False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y.view(-1, latent_dim * categorical_dim)

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, latent_dim * categorical_dim)


def gumbel_sigmoid(logits: Tensor, tau: float = 1, hard: bool = False, threshold: float = 0.5) -> Tensor:
    """
    Samples from the Gumbel-Sigmoid distribution and optionally discretizes.
    The discretization converts the values greater than `threshold` to 1 and the rest to 0.
    The code is adapted from the official PyTorch implementation of gumbel_softmax:
    https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax

    Args:
        logits: `[..., num_features]` unnormalized log probabilities
        tau: non-negative scalar temperature
        hard: if ``True``, the returned samples will be discretized,
                but will be differentiated as if it is the soft sample in autograd
        threshold: threshold for the discretization,
                    values greater than this will be set to 1 and the rest to 0

    Returns:
        Sampled tensor of same shape as `logits` from the Gumbel-Sigmoid distribution.
        If ``hard=True``, the returned samples are descretized according to `threshold`, otherwise they will
        be probability distributions.
    """
    gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0, 1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()

    if hard:
        # Straight through.
        indices = (y_soft > threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

