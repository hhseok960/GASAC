import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from flow_models.flows import RealNVP, NICE, SDNVP

random.seed(923)
torch.manual_seed(923)
torch.backends.cudnn.benchmark = False


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim=256, init_alpha=1., nf_archi="SD_NVP", device=DEVICE):
        super(Actor, self).__init__()
        assert nf_archi in ("RealNVP", "SD_NVP", "NICE")
        self.device = device

        if nf_archi == "SD_NVP":
            self.policy = SDNVP(state_size=state_size, action_size=action_size, hidden_dim=hidden_dim,
                                device=self.device).to(self.device)
        elif nf_archi == "RealNVP":
            self.policy = RealNVP(state_size=state_size, action_size=action_size, hidden_dim=hidden_dim,
                                  device=self.device).to(self.device)
        else:
            self.policy = NICE(state_size=state_size, action_size=action_size, hidden_dim=hidden_dim,
                               device=self.device).to(self.device)
        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True

    def forward(self, stt):
        z = self.policy.prior.sample((stt.shape[0],)).to(self.device)
        return self.policy(stt, z)

    def inverse(self, stt, act):
        return self.policy.inverse(stt, act)


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, stt, act):
        output = F.relu(self.fc1(torch.cat([stt, act], dim=1)))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output


class Discriminator(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim=256):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, stt, act):
        output = F.relu(self.fc1(torch.cat([stt, act], dim=1)))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output


class DiscriminatorSN(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim=256):
        super(DiscriminatorSN, self).__init__()
        self.fc1 = spectral_norm(nn.Linear(state_size + action_size, hidden_dim))
        self.fc2 = spectral_norm(nn.Linear(hidden_dim, hidden_dim))
        self.fc3 = spectral_norm(nn.Linear(hidden_dim, 1))

    def forward(self, stt, act):
        output = F.relu(self.fc1(torch.cat([stt, act], dim=1)))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output