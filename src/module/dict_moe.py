import logging
from copy import deepcopy
from typing import List, Optional

import torch
import torch.func
from torch import Tensor, nn
from torch.nn import functional as F

from ..tasks.arithmetic import state_dict_sub, state_dict_weighted_sum

log = logging.getLogger(__name__)


def join_list(list_of_list: List[List]):
    ans = []
    for l in list_of_list:
        ans.extend(l)
    return ans


class DictMoEGate2(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        init_lambda: float,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = hidden_size
        self.num_experts = num_experts

        if num_hidden_layers > 1:
            self.fc1 = nn.Sequential(*join_list([(nn.Linear(hidden_size, hidden_size, bias=True), nn.ReLU()) for _ in range(num_hidden_layers - 1)]))
        else:
            self.fc1 = nn.Identity()
        self.fc2 = nn.Linear(hidden_size, num_experts, bias=True)

        for p in self.fc1.parameters():
            nn.init.zeros_(p)
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, init_lambda)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        gate_weights = self.fc2(hidden_states)
        return gate_weights


class DictMoEGate(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        init_lambda: float,
        num_hidden_layers: int = 2,
    ):
        super().__init__()
        assert num_hidden_layers <= 2
        self.input_dim = hidden_size
        self.num_experts = num_experts
        self.num_hidden_layers = num_hidden_layers

        if num_hidden_layers == 2:
            self.fc1 = nn.Linear(hidden_size, hidden_size, bias=True)
            nn.init.normal_(self.fc1.weight, std=0.01)
            nn.init.zeros_(self.fc1.bias)
        elif num_hidden_layers == 1:
            self.fc1 = nn.Identity()

        if num_hidden_layers >= 1:
            self.fc2 = nn.Linear(hidden_size, num_experts, bias=True)
            nn.init.normal_(self.fc2.weight, std=0.01)
            nn.init.constant_(self.fc2.bias, init_lambda)

        if num_hidden_layers == 0:
            self.weight = nn.Parameter(torch.ones(num_experts) * init_lambda, requires_grad=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if self.num_hidden_layers == 0:
            return self.weight

        if self.num_hidden_layers == 2:
            hidden_states = F.relu(self.fc1(hidden_states))
        gate_weights = self.fc2(hidden_states)
        return gate_weights


class DictMoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        base_model: nn.Module,
        expert_models: List[nn.Module],
        init_lambda: float = 0.2,
        fix_base_model_and_experts: bool = True,
        batch_first: bool = False,
        router_hidden_layers: int = 2,
    ):
        super().__init__()
        self.num_experts = len(expert_models)
        self.input_dim = hidden_size
        self.batch_first = batch_first

        self.gate = DictMoEGate(
            hidden_size,
            self.num_experts,
            init_lambda=init_lambda,
            num_hidden_layers=router_hidden_layers,
        )

        self.base_model = deepcopy(base_model)
        experts = [deepcopy(e) for e in expert_models]
        base_sd = self.base_model.state_dict()
        experts_params = []
        experts_sd = [e.state_dict() for e in experts]
        for name in base_sd.keys():
            task_vectors = []
            for e_sd in experts_sd:
                with torch.no_grad():
                    _task_vector = e_sd[name] - base_sd[name]
                    task_vectors.append(_task_vector)
            task_vectors = torch.stack(task_vectors)
            experts_params.append(nn.Parameter(task_vectors, requires_grad=not fix_base_model_and_experts))
        self.expert_parms = nn.ParameterList(experts_params)

        if fix_base_model_and_experts:
            for p in self.base_model.parameters():
                p.requires_grad_(False)
            for p in self.expert_parms.parameters():
                p.requires_grad_(False)

    def forward(self, hidden_states: Tensor):
        if not self.batch_first:
            hidden_states = hidden_states.permute(1, 0, 2)
        batch_size, seq_len, hidden_size = hidden_states.shape
        gate_weights: Tensor = self.gate(hidden_states)
        if self.gate.num_hidden_layers == 0:
            base_sd = self.base_model.state_dict(keep_vars=True)
            sd = {}
            for param_idx, (name, param) in enumerate(base_sd.items()):
                expert_params: nn.Parameter = self.expert_parms[param_idx]
                task_vector = expert_params * gate_weights.view([-1] + [1] * (expert_params.dim() - 1))
                task_vector = task_vector.sum(dim=0)
                sd[name] = param + task_vector
            final_hidden_states = torch.func.functional_call(self.base_model, sd, hidden_states)
        else:
            gate_weights = gate_weights.mean(dim=1)
            final_hidden_states = []
            base_sd = self.base_model.state_dict(keep_vars=True)
            for sample_idx in range(batch_size):
                sd = {}
                for param_idx, (name, param) in enumerate(base_sd.items()):
                    expert_params: nn.Parameter = self.expert_parms[param_idx]
                    task_vector = expert_params * gate_weights[sample_idx].view([-1] + [1] * (expert_params.dim() - 1))
                    task_vector = task_vector.sum(dim=0)
                    sd[name] = param + task_vector
                _final_hidden_states = torch.func.functional_call(self.base_model, sd, hidden_states[sample_idx : sample_idx + 1])
                final_hidden_states.append(_final_hidden_states)
            final_hidden_states = torch.cat(final_hidden_states, dim=0)
        if not self.batch_first:
            final_hidden_states = final_hidden_states.permute(1, 0, 2)
        return final_hidden_states
