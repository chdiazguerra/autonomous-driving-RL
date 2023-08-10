import pickle

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from reinforcement.utils import StepLR
from reinforcement.buffer import ReplayBuffer

def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)
    
class ValueNetwork(nn.Module):
    def __init__(self, obs_dim):
        super(ValueNetwork, self).__init__()
        
        self.fc1 = nn.Linear(obs_dim, 64)
        init_weight(self.fc1)
        
        self.fc2 = nn.Linear(64, 64)
        init_weight(self.fc2)

        self.out_left = nn.Linear(64, 1)
        self.out_straight = nn.Linear(64, 1)
        self.out_right = nn.Linear(64, 1)
        init_weight(self.out_left, "xavier uniform")
        init_weight(self.out_straight, "xavier uniform")
        init_weight(self.out_right, "xavier uniform")
        
    def forward(self, x, command):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)

        out_left = self.out_left(x)
        out_straight = self.out_straight(x)
        out_right = self.out_right(x)
        
        x = torch.stack((out_left, out_straight, out_right), dim=0)
        x = torch.gather(x, 0, command.view(1,-1,1)).view(-1, 1)

        return x
    
class QvalueNetwork(nn.Module):
    def __init__(self, obs_dim, nb_actions=2):
        super(QvalueNetwork, self).__init__()
        
        self.fc1 = nn.Linear(obs_dim+nb_actions, 64)
        init_weight(self.fc1)
        
        self.fc2 = nn.Linear(64, 64)
        init_weight(self.fc2)

        self.out_left = nn.Linear(64, 1)
        self.out_straight = nn.Linear(64, 1)
        self.out_right = nn.Linear(64, 1)
        init_weight(self.out_left, "xavier uniform")
        init_weight(self.out_straight, "xavier uniform")
        init_weight(self.out_right, "xavier uniform")
        
    def forward(self, x, command, a):
        x = torch.cat([x, a], dim=1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)

        out_left = self.out_left(x)
        out_straight = self.out_straight(x)
        out_right = self.out_right(x)
        
        x = torch.stack((out_left, out_straight, out_right), dim=0)
        x = torch.gather(x, 0, command.view(1,-1,1)).view(-1, 1)

        return x
    
class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, nb_actions=2):
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(obs_dim, 64)
        init_weight(self.fc1)
        self.fc2 = nn.Linear(64, 64)
        init_weight(self.fc1)

        self.mu_left = nn.Linear(64, nb_actions)
        self.mu_straight = nn.Linear(64, nb_actions)
        self.mu_right = nn.Linear(64, nb_actions)

        init_weight(self.mu_left, "xavier uniform")
        init_weight(self.mu_straight, "xavier uniform")
        init_weight(self.mu_right, "xavier uniform")

        self.log_std_left = nn.Linear(64, nb_actions)
        self.log_std_straight = nn.Linear(64, nb_actions)
        self.log_std_right = nn.Linear(64, nb_actions)

        init_weight(self.log_std_left, "xavier uniform")
        init_weight(self.log_std_straight, "xavier uniform")
        init_weight(self.log_std_right, "xavier uniform")

        self.nb_actions = nb_actions
        
    def forward(self, x, command):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        mu_left = self.mu_left(x)
        mu_straight = self.mu_straight(x)
        mu_right = self.mu_right(x)

        mu = torch.stack((mu_left, mu_straight, mu_right), dim=0)
        mu = torch.gather(mu, 0, command.expand((-1,self.nb_actions)).view(1,-1,self.nb_actions)).view(-1,self.nb_actions)

        log_std_left = self.log_std_left(x)
        log_std_straight = self.log_std_straight(x)
        log_std_right = self.log_std_right(x)

        log_std = torch.stack((log_std_left, log_std_straight, log_std_right), dim=0)
        log_std = torch.gather(log_std, 0, command.expand((-1,self.nb_actions)).view(1,-1,self.nb_actions)).view(-1,self.nb_actions)

        std = log_std.clamp(min=-20, max=2).exp()
        
        return mu, std

    def sample_or_likelihood(self, states, command):
        mu, std = self(states, command)
        dist = torch.distributions.Normal(mu, std)

        u = dist.rsample()
        action = torch.tanh(u)
        log_prob = dist.log_prob(value=u)

        log_prob -= torch.log(1 - action ** 2 + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob

class SACAgent:
    def __init__(self, obs_dim=260, nb_actions=2, device='cpu', lr=1e-3, alpha=0.2,
                 batch_size=64, gamma=0.99, tau=0.005, buffer_size=10000, action_clip=(-1,1),
                 collision_percentage=0.2, sch_gamma = 0.9, sch_steps=500):
        
        self.device = device

        self.policy_network = PolicyNetwork(obs_dim, nb_actions).to(self.device)
        self.q_value_network1 = QvalueNetwork(obs_dim, nb_actions).to(self.device)
        self.q_value_network2 = QvalueNetwork(obs_dim, nb_actions).to(self.device)
        self.value_network = ValueNetwork(obs_dim).to(self.device)
        self.value_target_network = ValueNetwork(obs_dim).to(self.device)

        self.hard_update()

        self.value_opt = torch.optim.Adam(self.value_network.parameters(), lr=lr)
        self.q_value1_opt = torch.optim.Adam(self.q_value_network1.parameters(), lr=lr)
        self.q_value2_opt = torch.optim.Adam(self.q_value_network2.parameters(), lr=lr)
        self.policy_opt = torch.optim.Adam(self.policy_network.parameters(), lr=lr)

        self.sch_value = StepLR(self.value_opt, sch_steps, gamma=sch_gamma)
        self.sch_q1 = StepLR(self.q_value1_opt, sch_steps, gamma=sch_gamma)
        self.sch_q2 = StepLR(self.q_value2_opt, sch_steps, gamma=sch_gamma)
        self.sch_policy = StepLR(self.policy_opt, sch_steps, gamma=sch_gamma)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.action_clip = action_clip

        self.collision_percentage = collision_percentage
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size, self.device)
        self.buffer_collision = ReplayBuffer(buffer_size, self.device)

        #Training variables
        self.tr_step = 0
        self.tr_steps_vec = []
        self.avg_reward_vec = []
        self.std_reward_vec = []
        self.success_rate_vec = []
        self.episode_nb = 0

    def hard_update(self):
        self.value_target_network.load_state_dict(self.value_network.state_dict())

    def soft_update(self):
        for param, target_param in zip(self.value_network.parameters(), self.value_target_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def reset_noise(self):
        pass

    def store_transition(self, obs, action, reward, next_obs, done, info=None):
        self.buffer.add((obs, action, reward, next_obs, done))

    def store_transition_collision(self, obs, action, reward, next_obs, done, info=None):
        self.buffer_collision.add((obs, action, reward, next_obs, done))

    def select_action(self, obs, noise=True):
        obs = torch.from_numpy(obs).to(self.device).unsqueeze(0)
        comm = obs[:, -1:].long()
        obs = obs[:, :-1]

        with torch.no_grad():
            if noise:
                action, _ = self.policy_network.sample_or_likelihood(obs, comm)
            else:
                action, _ = self.policy_network(obs, comm)
        action = action.cpu().data.numpy().flatten()

        return np.clip(action, *self.action_clip)
    
    def get_batch(self):
        col_batch_size = int(self.collision_percentage*self.batch_size)
        if col_batch_size < len(self.buffer_collision):
            batch_size = self.batch_size - col_batch_size
            states_col, actions_col, rewards_col, next_states_col, dones_col = self.buffer_collision.sample(col_batch_size)
            states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
            states = np.concatenate((states_col, states))
            actions = np.concatenate((actions_col, actions))
            rewards = np.concatenate((rewards_col, rewards))
            next_states = np.concatenate((next_states_col, next_states))
            dones = np.concatenate((dones_col, dones))
        else:
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        return states, actions, rewards, next_states, dones  
    
    def update(self, soft_update=True):
        
        states, actions, rewards, next_states, dones = self.get_batch()
        comms = states[:, -1:]
        states = states[:, :-1]
        next_comms = next_states[:, -1:]
        next_states = next_states[:, :-1]
        

        states = torch.from_numpy(states).float().to(self.device)
        comms = torch.from_numpy(comms).long().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        next_comms = torch.from_numpy(next_comms).long().to(self.device)
        dones = torch.from_numpy(dones.astype(np.uint8)).float().to(self.device)

        reparam_actions, log_probs = self.policy_network.sample_or_likelihood(states, comms)

        q1 = self.q_value_network1(states, comms, reparam_actions)
        q2 = self.q_value_network2(states, comms, reparam_actions)
        q = torch.min(q1, q2)
        target_value = q.detach() - self.alpha * log_probs.detach()

        value = self.value_network(states, comms)
        value_loss = F.mse_loss(value, target_value)

        with torch.no_grad():
            target_q = rewards + self.gamma * self.value_target_network(next_states, next_comms) * (1 - dones)

        q1 = self.q_value_network1(states, comms, actions)
        q2 = self.q_value_network2(states, comms, actions)
        q1_loss = F.mse_loss(q1, target_q)
        q2_loss = F.mse_loss(q2, target_q)

        policy_loss = (self.alpha * log_probs - q).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 5e-3)
        self.policy_opt.step()

        self.value_opt.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 5e-3)
        self.value_opt.step()

        self.q_value1_opt.zero_grad()
        q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_value_network1.parameters(), 5e-3)
        self.q_value1_opt.step()

        self.q_value2_opt.zero_grad()
        q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_value_network2.parameters(), 5e-3)
        self.q_value2_opt.step()

        if soft_update:
            self.soft_update()
        else:
            self.hard_update()

        self.sch_value.step()
        self.sch_q1.step()
        self.sch_q2.step()
        self.sch_policy.step()

    def save(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
    