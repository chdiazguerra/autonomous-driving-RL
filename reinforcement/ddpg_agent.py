import pickle

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from reinforcement.utils import OUNoise, StepLR
from reinforcement.buffer import ReplayBuffer

class Actor(nn.Module):
    def __init__(self, obs_dim, nb_actions=2):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(obs_dim, 64)
        self.norm1 = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, 64)
        self.norm2 = nn.LayerNorm(64)

        self.out_left = nn.Linear(64, nb_actions)
        self.out_straight = nn.Linear(64, nb_actions)
        self.out_right = nn.Linear(64, nb_actions)

        self.out_left.weight.data.normal_(0, 1e-4)
        self.out_straight.weight.data.normal_(0, 1e-4)
        self.out_right.weight.data.normal_(0, 1e-4)

        self.nb_actions = nb_actions
        
    def forward(self, x, command):
        x = self.norm1(self.fc1(x))
        x = torch.relu(x)
        x = self.norm2(self.fc2(x))
        x = torch.relu(x)
        out_left = self.out_left(x)
        out_straight = self.out_straight(x)
        out_right = self.out_right(x)

        x = torch.stack((out_left, out_straight, out_right), dim=0)
        x = torch.gather(x, 0, command.expand((-1,self.nb_actions)).view(1,-1,self.nb_actions)).view(-1,self.nb_actions)

        x = torch.tanh(x)

        return x
    
class Critic(nn.Module):
    def __init__(self, obs_dim, nb_actions=2):
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(obs_dim+nb_actions, 64)
        self.norm1 = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, 64)
        self.norm2 = nn.LayerNorm(64)

        self.out_left = nn.Linear(64, 1)
        self.out_straight = nn.Linear(64, 1)
        self.out_right = nn.Linear(64, 1)
        
    def forward(self, x, command, a):
        x = torch.cat([x,a], 1)
        x = self.norm1(self.fc1(x))
        x = torch.relu(x)
        x = self.norm2(self.fc2(x))
        x = torch.relu(x)

        out_left = self.out_left(x)
        out_straight = self.out_straight(x)
        out_right = self.out_right(x)
        
        x = torch.stack((out_left, out_straight, out_right), dim=0)
        x = torch.gather(x, 0, command.view(1,-1,1)).view(-1, 1)

        return x
    
class Environment(nn.Module):
    def __init__(self, emb_size=260, nb_actions=2):
        super().__init__()
        self.fc1_transition = nn.Linear(nb_actions, 128)
        self.fc2_transition = nn.Linear(128+emb_size, 512)
        self.out_transition = nn.Linear(512, emb_size)

        self.fc1_reward = nn.Linear(nb_actions, 128)
        self.fc2_reward = nn.Linear(128+emb_size*2, 512)
        self.fc3_reward = nn.Linear(512, 256)
        self.out_reward = nn.Linear(256, 1)
    
    def forward(self, emb, action):
        o = self.transition_model(emb, action)
        r = self.reward_model(emb, action, o)

        return o, r
    
    def transition_model(self, emb, action):
        o = F.relu(self.fc1_transition(action))
        o = torch.cat((o, emb), dim=1)
        o = F.relu(self.fc2_transition(o))
        o = self.out_transition(o)
        return o
    
    def reward_model(self, emb, action, next_emb):
        r = F.relu(self.fc1_reward(action))
        r = torch.cat((r, emb, next_emb), dim=1)
        r = F.relu(self.fc2_reward(r))
        r = F.relu(self.fc3_reward(r))
        r = self.out_reward(r)
        return r

class DDPGAgent:
    """DDPG Agent, with behavior cloning and deductive learning options

    Args
    ----
    obs_dim : int
        Observation dimension
    nb_actions : int
        Number of actions
    device : str
        Device to use for training
    lr_actor : float
        Learning rate for the actor
    lr_critic : float
        Learning rate for the critic
    batch_size : int
        Batch size for training
    gamma : float
        Discount factor
    tau : float
        Soft update factor
    clip_norm : float
        Gradient clipping norm
    buffer_size : int
        Replay buffer size
    action_clip : tuple
        Action clipping range
    collision_percentage : float
        Percentage of collision data in the batch
    noise_sigma : float
        Noise sigma for the Ornstein-Uhlenbeck process
    noise_decay : float
        Noise decay for the Ornstein-Uhlenbeck process
    sch_gamma : float
        Learning rate scheduler gamma
    sch_steps : int
        Learning rate scheduler steps
    use_expert_data : bool
        Whether to use expert data
    expert_percentage : float
        Percentage of expert data in the batch
    lambda_bc : float
        Behavior cloning loss weight
    lambda_env : float
        Deductive learning loss weight
    """

    def __init__(self, obs_dim=260, nb_actions=2, device='cpu', lr_actor=1e-4, lr_critic=1e-3,
                 batch_size=64, gamma=0.95, tau=0.005, clip_norm=5e-3, buffer_size=20000, action_clip=(-1,1),
                 collision_percentage=0.2, noise_sigma=0.4, noise_decay=1/350, sch_gamma = 0.9, sch_steps=500,
                 use_expert_data=False, expert_percentage=0.25, lambda_bc=0.1, use_env_model=False, lambda_env=0.2,
                 env_steps=10):
        
        self.device = device

        self.actor = Actor(obs_dim, nb_actions).to(self.device)
        self.critic = Critic(obs_dim, nb_actions).to(self.device)
        self.actor_target = Actor(obs_dim, nb_actions).to(self.device)
        self.critic_target = Critic(obs_dim, nb_actions).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.actor_sch = StepLR(self.actor_optimizer, sch_steps, sch_gamma)
        self.critic_sch = StepLR(self.critic_optimizer, sch_steps, sch_gamma)

        self.gamma = gamma
        self.tau = tau
        self.clip_norm = clip_norm
        self.action_clip = action_clip
        self.use_expert_data = use_expert_data
        self.expert_percentage = expert_percentage
        self.lambda_bc = lambda_bc
        self.use_env_model = use_env_model
        self.lambda_env = lambda_env
        self.env_steps = env_steps

        self.noise = OUNoise(mu=np.zeros(nb_actions), sigma=noise_sigma, noise_decay=noise_decay)

        self.collision_percentage = collision_percentage
        self.batch_size = batch_size
        self.buffer = ReplayBuffer(buffer_size)
        self.buffer_collision = ReplayBuffer(buffer_size)
        if self.use_expert_data:
            self.buffer_expert = ReplayBuffer(buffer_size)

        if use_env_model:
            self.env_model = Environment(emb_size=obs_dim, nb_actions=nb_actions).to(self.device)
            self.env_model_optimizer = torch.optim.Adam(self.env_model.parameters(), lr=1e-4)

        self.hard_update()

        #Training variables
        self.tr_step = 0
        self.tr_steps_vec = []
        self.avg_reward_vec = []
        self.std_reward_vec = []
        self.success_rate_vec = []
        self.episode_nb = 0

    def hard_update(self):
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def soft_update(self):
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def reset_noise(self):
        self.noise.reset()

    def store_transition(self, obs, action, reward, next_obs, done, info=None):
        self.buffer.add((obs, action, reward, next_obs, done))

    def store_transition_collision(self, obs, action, reward, next_obs, done, info=None):
        self.buffer_collision.add((obs, action, reward, next_obs, done))

    def select_action(self, obs, noise=True):
        obs = torch.from_numpy(obs).to(self.device).unsqueeze(0)
        comm = obs[:, -1:].long()
        obs = obs[:, :-1]

        with torch.no_grad():
            action = self.actor(obs, comm).cpu().data.numpy().flatten()
        if noise:
            action += self.noise()

        return np.clip(action, *self.action_clip)

    def _compute_critic_loss(self, states, comms, actions, rewards, next_states, next_comms, dones):
        next_actions = self.actor_target(next_states, next_comms)
        next_q = self.critic_target(next_states, next_comms, next_actions)
        q_target = rewards + (1 - dones) * self.gamma * next_q
        q_target = q_target.detach()

        q_values = self.critic(states, comms, actions)
        critic_loss = F.mse_loss(q_values, q_target)
        return critic_loss
    
    def _compute_actor_loss(self, states, comms, expert_batch=None):
        policy_loss = -self.critic(states, comms, self.actor(states, comms)).sum()

        if expert_batch is not None:
            states, actions, rewards, next_states, dones = expert_batch
            comms = states[:, -1:]
            states = states[:, :-1]            

            states = torch.from_numpy(states).float().to(self.device)
            comms = torch.from_numpy(comms).long().to(self.device)
            actions = torch.from_numpy(actions).float().to(self.device)

            policy_loss = self.lambda_bc*self._compute_bc_loss(states, comms, actions, sum_red=True) + (1-self.lambda_bc)*policy_loss

        if self.use_env_model:
            policy_loss = self.lambda_env*self._compute_env_loss(states, comms) + (1-self.lambda_env)*policy_loss

        return policy_loss/self.batch_size
    
    def _compute_bc_loss(self, states, comms, actions, sum_red=False):
        actions_pred = self.actor(states, comms)
        bc_loss = F.mse_loss(actions_pred, actions, reduction='sum' if sum_red else 'mean')
        return bc_loss
    
    def _compute_env_loss(self, states, comms):

        action = self.actor(states, comms)
        loss = 0
        for i in range(self.env_steps):
            states, rew_pred = self.env_model(states, action)
            loss += (self.gamma**i)*rew_pred
            action = self.actor(states, comms)
        return -loss.sum()
    
    def _update_env_model(self, states, act, rew, next_states):
        self.env_model_optimizer.zero_grad()
        transition = self.env_model.transition_model(states, act)
        reward = self.env_model.reward_model(states, act, next_states)
        t_loss = F.mse_loss(transition, next_states)
        r_loss = F.mse_loss(reward, rew)
        loss = t_loss + r_loss
        loss.backward()
        self.env_model_optimizer.step()
    
    def pretrain_update(self):
        states, actions, rewards, next_states, dones = self.buffer_expert.sample(self.batch_size)
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

        critic_loss = self._compute_critic_loss(states, comms, actions, rewards, next_states, next_comms, dones)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        policy_loss = self._compute_bc_loss(states, comms, actions)
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.hard_update()

        if self.use_env_model:
            self._update_env_model(states, actions, rewards, next_states)
    
    def get_batch(self):
        col_batch_size = int(self.collision_percentage*self.batch_size)
        batch_size = self.batch_size - col_batch_size
        expert_batch_size = int(self.expert_percentage*self.batch_size) if self.use_expert_data else 0
        batch_size = batch_size - expert_batch_size

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        if col_batch_size < len(self.buffer_collision) and col_batch_size > 0:
            states_col, actions_col, rewards_col, next_states_col, dones_col = self.buffer_collision.sample(col_batch_size)
            states = np.concatenate((states_col, states))
            actions = np.concatenate((actions_col, actions))
            rewards = np.concatenate((rewards_col, rewards))
            next_states = np.concatenate((next_states_col, next_states))
            dones = np.concatenate((dones_col, dones))
        
        if expert_batch_size > 0 and self.use_expert_data:
            expert_batch = self.buffer_expert.sample(expert_batch_size)
        else:
            expert_batch = None

        return (states, actions, rewards, next_states, dones), expert_batch
    
    def update(self, soft_update=True):
        (states, actions, rewards, next_states, dones), expert_batch = self.get_batch()
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

        if self.use_env_model:
            self._update_env_model(states, actions, rewards, next_states)


        critic_loss = self._compute_critic_loss(states, comms, actions, rewards, next_states, next_comms, dones)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # if self.clip_norm is not None:
        #     torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_norm)
        self.critic_optimizer.step()

        policy_loss = self._compute_actor_loss(states, comms, expert_batch=expert_batch)
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        if self.clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_norm)
        self.actor_optimizer.step()


        # next_actions = self.actor_target(next_states, next_comms)
        # next_q = self.critic_target(next_states, next_comms, next_actions)
        # q_target = rewards + (1 - dones) * self.gamma * next_q
        # q_target = q_target.detach()

        # q_values = self.critic(states, comms, actions)

        # critic_loss = F.mse_loss(q_values, q_target)
        # self.critic_optimizer.zero_grad()
        # critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_norm)
        # self.critic_optimizer.step()

        # policy_loss = -self.critic(states, comms, self.actor(states, comms)).mean()
        # self.actor_optimizer.zero_grad()
        # policy_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_norm)
        # self.actor_optimizer.step()

        if soft_update:
            self.soft_update()
        else:
            self.hard_update()

        self.actor_sch.step()
        self.critic_sch.step()

    def save(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    def load_expert_data(self, expert_data_path):
        with open(expert_data_path, 'rb') as f:
            expert_data = pickle.load(f)
        for transition in expert_data:
            self.buffer_expert.add(transition)

    def save_actor(self, save_path):
        torch.save(self.actor.state_dict(), save_path)

    def load_actor(self, load_path):
        self.actor.load_state_dict(torch.load(load_path))
    