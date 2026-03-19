import random

import torch
import torch.nn as nn
from torch.distributions import Normal

from flow_models.flows_utils import LinearMaskedCouplingVP, LinearMaskedCouplingCondiNVP, LinearMaskedCouplingNVP


random.seed(923)
torch.manual_seed(923)
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NICE(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim=256, n_transforms=2, device=DEVICE):
        super(NICE, self).__init__()
        self.state_size, self.action_size = state_size, action_size
        self.prior = Normal(loc=torch.zeros(action_size).to(device), scale=torch.ones(action_size).to(device),
                            validate_args=False)

        module_list = []
        mask = torch.arange(action_size).float() % 2
        for _ in range(n_transforms * 2):
            module_list.append(LinearMaskedCouplingVP(state_size, action_size, hidden_dim, mask))
            mask = 1 - mask
        self.net = nn.ModuleList(module_list)
        self.s = nn.Parameter(torch.randn(action_size))

    def forward(self, stt, z):
        action = z.clone() / torch.exp(self.s)
        log_prob = self.prior.log_prob(z).sum(dim=1, keepdims=True)
        for i in range(len(self.net)):
            action, _ = self.net[i].inverse(stt, action)
        log_tanh = torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        log_prob -= log_tanh.sum(dim=1, keepdim=True)
        log_prob -= self.s.sum()
        return torch.tanh(action), log_prob

    def inverse(self, stt, act):
        log_prob = - torch.log(1 - act.pow(2) + 1e-7).sum(dim=1, keepdim=True)
        # act = torch.atanh(torch.clip(act, min=-1 + 1e-7, max=1 - 1e-7))
        act = torch.atanh(act)
        z = act.clone()
        for i in range(len(self.net)):
            z, _ = self.net[len(self.net) - 1 - i](stt, z)
        z *= torch.exp(self.s)
        log_prob += self.prior.log_prob(z).sum(dim=1, keepdims=True)
        log_prob -= self.s.sum()
        return z, log_prob


class RealNVP(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim=256, n_transforms=2, device=DEVICE):
        super(RealNVP, self).__init__()
        self.state_size, self.action_size = state_size, action_size
        self.prior = Normal(loc=torch.zeros(action_size).to(device), scale=torch.ones(action_size).to(device),
                            validate_args=False)

        module_list = []
        mask = torch.arange(action_size).float() % 2
        for _ in range(n_transforms * 2):
            module_list.append(LinearMaskedCouplingNVP(state_size, action_size, hidden_dim, mask))
            mask = 1 - mask

        self.net = nn.ModuleList(module_list)

    def forward(self, stt, z):
        action = z.clone()
        log_prob = self.prior.log_prob(z).sum(dim=1, keepdims=True)
        for i in range(len(self.net)):
            action, log_jacobian = self.net[i].inverse(stt, action)
            log_prob += log_jacobian
        log_tanh = torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        log_prob -= log_tanh.sum(dim=1, keepdim=True)
        return torch.tanh(action), log_prob

    def inverse(self, stt, act):
        log_prob = - torch.log(1 - act.pow(2) + 1e-7).sum(dim=1, keepdim=True)
        # act = torch.atanh(torch.clip(act, min=-1 + 1e-7, max=1 - 1e-7))
        act = torch.atanh(act)
        z = act.clone()
        for i in range(len(self.net)):
            z, log_jacobian = self.net[len(self.net) - 1 - i](stt, z)
            log_prob += log_jacobian
        log_prob += self.prior.log_prob(z).sum(dim=1, keepdims=True)
        return z, log_prob


class SDNVP(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim=256, n_transforms=2, device=DEVICE):
        super(SDNVP, self).__init__()
        self.state_size, self.action_size = state_size, action_size
        self.prior = Normal(loc=torch.zeros(action_size).to(device), scale=torch.ones(action_size).to(device),
                            validate_args=False)

        module_list = []
        mask = torch.arange(action_size).float() % 2
        for _ in range(n_transforms * 2):
            module_list.append(LinearMaskedCouplingCondiNVP(state_size, action_size, hidden_dim, mask))
            mask = 1 - mask

        self.net = nn.ModuleList(module_list)

    def forward(self, stt, z):
        action = z.clone()
        log_prob = self.prior.log_prob(z).sum(dim=1, keepdims=True)
        for i in range(len(self.net)):
            action, log_jacobian = self.net[i].inverse(stt, action)
            log_prob += log_jacobian
        log_tanh = torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        log_prob -= log_tanh.sum(dim=1, keepdim=True)
        return torch.tanh(action), log_prob

    def inverse(self, stt, act):
        log_prob = - torch.log(1 - act.pow(2) + 1e-7).sum(dim=1, keepdim=True)
        # act = torch.atanh(torch.clip(act, min=-1 + 1e-7, max=1 - 1e-7))
        act = torch.atanh(act)
        z = act.clone()
        for i in range(len(self.net)):
            z, log_jacobian = self.net[len(self.net) - 1 - i](stt, z)
            log_prob += log_jacobian
        log_prob += self.prior.log_prob(z).sum(dim=1, keepdims=True)
        return z, log_prob