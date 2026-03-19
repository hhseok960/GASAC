import os
import random
import torch
import torch.nn as nn


random.seed(923)
torch.manual_seed(923)
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LinearMaskedCouplingVP(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim, mask):
        super(LinearMaskedCouplingVP, self).__init__()
        self.register_buffer('mask', mask)
        self.trans_net = nn.Sequential(nn.Linear(state_size + action_size, hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, action_size), )

        self.trans_net.apply(self.init_weights)

    def forward(self, stt, act):
        act_masked = act * self.mask
        trans = self.trans_net(torch.cat([stt, act_masked], dim=1))
        z = act_masked + (1 - self.mask) * (act - trans)
        return z, 0

    def inverse(self, stt, z):
        z_masked = z * self.mask
        trans = self.trans_net(torch.cat([stt, z_masked], dim=1))
        act = z_masked + (1 - self.mask) * (z + trans)
        return act, 0

    def init_weights(self, module, scale=0.01):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            print("Init processed")
            module.weight.data.normal_(0, scale)
        else:
            pass


class LinearMaskedCouplingNVP(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim, mask):
        super(LinearMaskedCouplingNVP, self).__init__()
        self.register_buffer('mask', mask)
        self.scale_net = nn.Sequential(nn.Linear(state_size + action_size, hidden_dim), nn.Tanh(),
                                       nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                       nn.Linear(hidden_dim, action_size), )

        self.shift_net = nn.Sequential(nn.Linear(state_size + action_size, hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, action_size), )

        self.scale_net.apply(self.init_weights)
        self.shift_net.apply(self.init_weights)

    def forward(self, stt, act):
        act_masked = act * self.mask  # apply mask
        scale = self.scale_net(torch.cat([stt, act_masked], dim=1))
        shift = self.shift_net(torch.cat([stt, act_masked], dim=1))
        z = act_masked + (1 - self.mask) * (act - shift) * torch.exp(- scale)
        log_jacobian = - torch.sum((1 - self.mask) * scale, dim=1, keepdim=True)
        return z, log_jacobian

    def inverse(self, stt, z):
        z_masked = z * self.mask  # apply mask
        scale = self.scale_net(torch.cat([stt, z_masked], dim=1))
        shift = self.shift_net(torch.cat([stt, z_masked], dim=1))
        act = z_masked + (1 - self.mask) * (z * scale.exp() + shift)
        log_jacobian = - torch.sum((1 - self.mask) * scale, dim=1, keepdim=True)
        return act, log_jacobian

    def init_weights(self, module, scale=0.01):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            print("Init processed")
            module.weight.data.normal_(0, scale)
        else:
            pass


class LinearMaskedCouplingCondiNVP(nn.Module):
    def __init__(self, state_size, action_size, hidden_dim, mask):
        super(LinearMaskedCouplingCondiNVP, self).__init__()
        self.register_buffer('mask', mask)
        self.scale_net = nn.Sequential(nn.Linear(state_size, hidden_dim), nn.Tanh(),
                                       nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                                       nn.Linear(hidden_dim, action_size), )

        self.shift_net = nn.Sequential(nn.Linear(state_size + action_size, hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, action_size), )

        self.scale_net.apply(self.init_weights)
        self.shift_net.apply(self.init_weights)

    def forward(self, stt, act):
        act_masked = act * self.mask  # apply mask
        scale = self.scale_net(stt)
        shift = self.shift_net(torch.cat([stt, act_masked], dim=1))

        z = (act_masked + (1 - self.mask) * (act - shift)) * torch.exp(- scale)
        log_jacobian = - torch.sum(scale, dim=1, keepdim=True)
        return z, log_jacobian

    def inverse(self, stt, z):
        z_masked = z * self.mask  # apply mask
        scale = self.scale_net(stt)
        shift = self.shift_net(torch.cat([stt, z_masked], dim=1))

        act = z * scale.exp() + (1 - self.mask) * shift
        log_jacobian = - torch.sum(scale, dim=1, keepdim=True)
        return act, log_jacobian

    def init_weights(self, module, scale=0.01):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            print("Init processed")
            module.weight.data.normal_(0, scale)
        else:
            pass
