from copy import deepcopy
from typing import Any, Dict, List, Optional, cast

import torch
from torch import Tensor, nn

from .utils import get_by_name, set_by_name


def merge_fusion_block(model: nn.Module):
    for n, m in model.named_modules():
        if isinstance(m, FusionBlock):
            block = deepcopy(m.models[0])
            block.load_state_dict(m.merged_state_dict())
            set_by_name(model, n, block)
    return model


class FusionBlock(nn.Module):
    def __init__(
        self,
        models: List[nn.Module],
        fix_models: bool = True,
        init_lambda: float = 0.1,
        offload_device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_local_experts = len(models) - 1
        self.offload_device = offload_device

        self.models = nn.ModuleList(models)

        for m in self.models:
            for p in m.parameters():
                p.requires_grad_(not fix_models)

        self.weight = nn.Parameter(torch.ones(self.num_local_experts) * init_lambda, requires_grad=True)

        self._merged_state_dict = None

    def to(self, device, *args, **kwargs):
        self.weight.to(device, *args, **kwargs)
        if self.offload_device is not None:
            self.models.to(self.offload_device, *args, **kwargs)
        else:
            self.models.to(device, *args, **kwargs)

    @property
    def device(self):
        return self.weight.device

    def merged_state_dict(self):
        final_task_vector = {}
        task_vectors = [cast(nn.Module, m).state_dict(keep_vars=True) for m in self.models[1:]]
        for k in task_vectors[0].keys():
            final_task_vector[k] = torch.stack([d[k] for d in task_vectors], dim=0)
        final_task_vector = {
            k: torch.sum(
                v * self.weight.view(-1, *([1] * (v.dim() - 1))).to(self.offload_device if self.offload_device is not None else self.device),
                dim=0,
            )
            for k, v in final_task_vector.items()
        }
        final_state_dict = self.models[0].state_dict(keep_vars=True)
        for k, v in final_task_vector.items():
            final_state_dict[k] = (final_state_dict[k] + v).to(self.device)
        return final_state_dict

    def forward(self, *args, **kwargs):
        if self.offload_device is not None:
            self.to(self.device)
        if self._merged_state_dict is None:
            final_state_dict = self.merged_state_dict()
        else:
            final_state_dict = self._merged_state_dict
        outputs = torch.func.functional_call(
            self.models[0],
            final_state_dict,
            args=args,
            kwargs=kwargs,
            strict=False,
        )
        return outputs

    def merge_weights(self):
        self._merged_state_dict = self.merged_state_dict()


def build_fusion_block(
    pretrained_model: nn.Module,
    finetuned_models: List[nn.Module],
    fusion_block_kwargs: Dict[str, Any],
):
    pretrained_sd = pretrained_model.state_dict()
    models = [deepcopy(pretrained_model)]
    for ft_model in finetuned_models:
        ft_sd = ft_model.state_dict()
        for key in ft_sd.keys():
            ft_sd[key] = ft_sd[key] - pretrained_sd[key]
        _model = deepcopy(ft_model)
        _model.load_state_dict(ft_sd)
        models.append(_model)
    return FusionBlock(models, **fusion_block_kwargs)
