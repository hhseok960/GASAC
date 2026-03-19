import os
import random
from collections import deque
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from gasac.base import Actor, Critic, Discriminator, DiscriminatorSN

random.seed(923)
torch.manual_seed(923)
torch.backends.cudnn.benchmark = False

GAMMA = 0.99
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GASACAgent:
    def __init__(self, state_size, action_size, nf_archi, sn=True, buffer_size=int(1e6), batch_size=256, balance=0.5,
                 tau=0.005, lr_policy=0.0003, lr_q_func=0.0003, lr_disc=0.0003, lr_alpha=0.0003, device=DEVICE):
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
        if sn:
            self.discriminator = DiscriminatorSN(state_size, action_size).to(self.device)
            print("Spectal Normalized Discriminator Initialized")
        else:
            self.discriminator = Discriminator(state_size, action_size).to(self.device)
            print("Discriminator Initialized")

        self.optimizer_q1 = optim.Adam(self.main_critic1.parameters(), lr=lr_q_func)
        self.optimizer_q2 = optim.Adam(self.main_critic2.parameters(), lr=lr_q_func)
        self.optimizer_policy = optim.Adam(self.actor.parameters(), lr=lr_policy)
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=lr_disc)
        self.optimizer_log_alpha = optim.Adam([self.actor.log_alpha], lr=lr_alpha)

        self.batch_size = batch_size
        self.tau = tau
        self.balance = balance
        self.it = 0

    def update_network(self):
        if len(self.replay_memory) <= self.batch_size:
            return
        self.it += 1
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

        # Importance Sampling based Discriminator Update
        action_G, log_probs_G = self.actor(states[:64])
        loss_D_pi = torch.mean(torch.log(1 - torch.sigmoid(self.discriminator(states[:64], action_G))))

        action_D = torch.rand(action_G.shape).to(self.device) * 2.0 - 1.0

        q1_targets = self.target_critic1(states[:64], action_D.detach())
        q2_targets = self.target_critic2(states[:64], action_D.detach())
        min_q_targets = torch.min(torch.cat([q1_targets, q2_targets], dim=1), dim=1, keepdim=True)[0]
        min_q_targets /= self.actor.log_alpha.exp().detach()

        decimal = torch.floor(torch.log10(torch.abs(min_q_targets))).max()
        IS_weight = torch.exp(min_q_targets * (10 ** -decimal))
        IS_weight = IS_weight.detach()

        loss_D_IS = torch.log(torch.sigmoid(self.discriminator(states[:64], action_D)))
        loss_D_q = torch.sum(IS_weight * loss_D_IS) / torch.sum(IS_weight)
        loss_var = - (loss_D_pi + loss_D_q)
        self.optimizer_d.zero_grad()
        loss_var.backward()
        self.optimizer_d.step()

        # Actor Update
        self.main_critic1.eval()
        self.main_critic2.eval()
        self.target_critic1.eval()
        self.target_critic2.eval()
        self.actor.train()

        actions, log_probs = self.actor(states)
        entropy = -self.actor.log_alpha.exp() * log_probs

        q1_vals, q2_vals = self.main_critic1(states, actions), self.main_critic2(states, actions)
        min_q = torch.min(torch.cat([q1_vals, q2_vals], dim=1), dim=1, keepdim=True)[0]

        loss_pi_explicit = torch.mean(-min_q - entropy)

        if self.it % 1 == 0:
            # Generator Update
            actions_G, _ = self.actor(states[:64])
            loss_pi_implicit = - self.balance * torch.mean(torch.log(torch.sigmoid(self.discriminator(states[:64], actions_G))))
            loss_pi_explicit += loss_pi_implicit

        self.optimizer_policy.zero_grad()
        loss_pi_explicit.backward()
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
        save_path = f"./weights_VanillaGASAC_{self.nf_archi}"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(self.main_critic1.state_dict(), f"{save_path}/critic1_{file_name}.pt")
        torch.save(self.main_critic2.state_dict(), f"{save_path}/critic2_{file_name}.pt")
        torch.save(self.discriminator.state_dict(), f"{save_path}/discriminator_{file_name}.pt")
        torch.save(self.actor.state_dict(), f"{save_path}/actor_{file_name}.pt")