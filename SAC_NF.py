import os
import random
from collections import deque
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from flow_models.flows import RealNVP, NICE, SDNVP

random.seed(923)
torch.manual_seed(923)
torch.backends.cudnn.benchmark = False

GAMMA = 0.99
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


class SACAgent:
    def __init__(self, state_size, action_size, nf_archi, buffer_size=int(1e6), batch_size=256,
                 tau=0.005, lr_policy=0.0003, lr_q_func=0.0003, lr_alpha=0.0003, device=DEVICE):
        self.device = device

        self.state_size, self.action_size = state_size, action_size
        self.target_entropy = - action_size
        self.replay_memory = deque(maxlen=buffer_size)

        random.seed(923)
        np.random.seed(923)
        self.main_critic1 = Critic(state_size, action_size).to(self.device)
        self.target_critic1 = Critic(state_size, action_size).to(self.device)
        self.main_critic2 = Critic(state_size, action_size).to(self.device)
        self.target_critic2 = Critic(state_size, action_size).to(self.device)
        self.target_critic1.load_state_dict(self.main_critic1.state_dict())
        self.target_critic2.load_state_dict(self.main_critic2.state_dict())

        self.nf_archi = nf_archi
        self.actor = Actor(state_size, action_size, nf_archi=self.nf_archi, device=self.device).to(self.device)

        self.optimizer_q1 = optim.Adam(self.main_critic1.parameters(), lr=lr_q_func)
        self.optimizer_q2 = optim.Adam(self.main_critic2.parameters(), lr=lr_q_func)
        self.optimizer_policy = optim.Adam(self.actor.parameters(), lr=lr_policy)
        self.optimizer_log_alpha = optim.Adam([self.actor.log_alpha], lr=lr_alpha)

        self.batch_size = batch_size
        self.tau = tau

    def update_network(self):
        if len(self.replay_memory) <= self.batch_size:
            return

        mini_batch = random.sample(self.replay_memory, self.batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for transitions in mini_batch:
            s, a, r, s_n, d = transitions
            states.append(s)
            actions.append(a)
            rewards.append([r])
            next_states.append(s_n)
            dones.append([1.0 if d else 0.0])
        states, actions = torch.cat(states).to(self.device), torch.cat(actions).to(self.device)
        rewards, next_states = torch.Tensor(rewards).to(self.device), torch.cat(next_states).to(self.device)
        dones = torch.Tensor(dones).to(self.device)

        self.main_critic1.train()
        self.main_critic2.train()
        self.target_critic1.eval()
        self.target_critic2.eval()
        self.actor.eval()

        # Critic Update
        next_actions, log_probs = self.actor(next_states)

        entropies = -self.actor.log_alpha.exp() * log_probs
        q1_vals = self.target_critic1(next_states, next_actions)
        q2_vals = self.target_critic2(next_states, next_actions)
        min_q = torch.min(torch.cat([q1_vals, q2_vals], dim=1), dim=1, keepdim=True)[0]
        td_target = rewards + (1 - dones) * GAMMA * (min_q + entropies)

        loss_q1 = F.mse_loss(self.main_critic1(states, actions), td_target.detach())
        self.optimizer_q1.zero_grad()
        loss_q1.backward()
        self.optimizer_q1.step()

        loss_q2 = F.mse_loss(self.main_critic2(states, actions), td_target.detach())
        self.optimizer_q2.zero_grad()
        loss_q2.backward()
        self.optimizer_q2.step()

        # Actor update
        self.main_critic1.eval()
        self.main_critic2.eval()
        self.target_critic1.eval()
        self.target_critic2.eval()
        self.actor.train()

        actions, log_probs = self.actor(states)
        entropy = -self.actor.log_alpha.exp() * log_probs

        q1_vals, q2_vals = self.main_critic1(states, actions), self.main_critic2(states, actions)
        min_q = torch.min(torch.cat([q1_vals, q2_vals], dim=1), dim=1, keepdim=True)[0]
        loss_pi = torch.mean(-min_q - entropy)

        self.optimizer_policy.zero_grad()
        loss_pi.backward()
        self.optimizer_policy.step()

        self.optimizer_log_alpha.zero_grad()
        loss_alpha = -(self.actor.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()
        loss_alpha.backward()
        self.optimizer_log_alpha.step()

        self.soft_update()

    def decide_action(self, state):
        with torch.no_grad():
            action, _ = self.actor(state.to(self.device))
        return action

    def soft_update(self):
        for param_target, param in zip(self.target_critic1.parameters(), self.main_critic1.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
        for param_target, param in zip(self.target_critic2.parameters(), self.main_critic2.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def memorize(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def save_model(self, file_name):
        save_path = f"./weights_SAC_{self.nf_archi}"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(self.main_critic1.state_dict(), f"{save_path}/critic1_{file_name}.pt")
        torch.save(self.main_critic2.state_dict(), f"{save_path}/critic2_{file_name}.pt")
        torch.save(self.actor.state_dict(), f"{save_path}/actor_{file_name}.pt")