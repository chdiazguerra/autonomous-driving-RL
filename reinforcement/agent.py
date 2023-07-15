import pickle
import copy

import numpy as np
import torch
import torch.nn.functional as F

from models.agent_parts import Actor, TwinCritic, Environment
from reinforcement.buffer import ReplayBuffer


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.6, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state

class TD3ColDeductiveAgent:
    def __init__(self, obs_size=256, device='cpu', actor_lr=1e-3, critic_lr=1e-3,
                 pol_freq_update=2, policy_noise=0.2, noise_clip=0.5, act_noise=0.1, gamma=0.99,
                 tau=0.005, l2_reg=1e-5, env_steps=8, env_w=0.2, lambda_bc=0.1, lambda_a=0.9, lambda_q=1.0,
                 exp_buff_size=100000, actor_buffer_size=50000, exp_prop=0.25, batch_size=64,
                 save_path='agent.pkl'):
        assert device in ['cpu', 'cuda'], "device must be either 'cpu' or 'cuda'"

        self.actor = Actor(obs_size).to(device)
        self.actor_target = Actor(obs_size).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=l2_reg)

        self.critic = TwinCritic(obs_size).to(device)
        self.critic_target = TwinCritic(obs_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=l2_reg)

        self.env_model = Environment(obs_size).to(device)
        self.env_model_optimizer = torch.optim.Adam(self.env_model.parameters(), lr=1e-3)

        self.pol_freq_update = pol_freq_update
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.act_noise = act_noise
        self.gamma = gamma
        self.tau = tau
        self.env_steps = env_steps
        self.env_w = env_w
        self.lambda_a = lambda_a
        self.lambda_bc = lambda_bc
        self.lambda_q = lambda_q

        self.policy_clip_min = torch.tensor([-0.75, -1.0]).to(device)
        self.policy_clip_max = torch.tensor([0.75, 1.0]).to(device)

        self.policy_clip_max_np = self.policy_clip_max.cpu().numpy()
        self.policy_clip_min_np = self.policy_clip_min.cpu().numpy()

        self.expert_buffer = ReplayBuffer(exp_buff_size, device)
        self.actor_buffer = ReplayBuffer(actor_buffer_size, device)
        self.batch_size = batch_size
        self.exp_batch_size = int(exp_prop*batch_size)
        self.actual_batch_size = batch_size - self.exp_batch_size

        self.device = device
        self.save_path = save_path

        self.ou_noise = OUNoise(2, sigma=act_noise)

    def select_action(self, obs, prev_action, eval=False):
        emb, command = obs
        emb = torch.FloatTensor(emb.reshape(1, -1)).to(self.device)
        command = torch.tensor(command).reshape(1, 1).to(self.device)
        prev_action = torch.FloatTensor(prev_action).reshape(1, -1).to(self.device)

        with torch.no_grad():
            action = self.actor(emb, command, prev_action).cpu().numpy().flatten()
        if not eval:
            #noise = np.random.normal(0, self.act_noise, size=action.shape)
            noise = self.ou_noise.sample()
            action = (action + noise).clip(self.policy_clip_min_np, self.policy_clip_max_np)
        return action.tolist()
    
    def reset_noise(self):
        self.ou_noise.reset()
    
    def store_transition(self, p_act, obs, act, rew, next_obs, done):
        self.actor_buffer.store_transition((p_act, obs, act, rew, next_obs, done))

    def update_step(self, it, is_pretraining):
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        if is_pretraining:
            p_act_exp, obs_exp, act_exp, rew_exp, next_obs_exp, done_exp = self.expert_buffer.sample(self.batch_size)
            obs = obs_exp
            next_obs = next_obs_exp
            act = act_exp
            rew = rew_exp

            critic_loss = self._compute_critic_loss(obs_exp, act_exp, rew_exp, next_obs_exp, done_exp)
        else:
            p_act_exp, obs_exp, act_exp, rew_exp, next_obs_exp, done_exp = self.expert_buffer.sample(self.exp_batch_size)
            p_act_act, obs_act, act_act, rew_act, next_obs_act, done_act = self.actor_buffer.sample(self.actual_batch_size)

            emb_exp, command_exp = obs_exp
            emb_act, command_act = obs_act
            obs = (torch.cat((emb_exp, emb_act), dim=0), torch.cat((command_exp, command_act), dim=0))
            p_act = torch.cat((p_act_exp, p_act_act), dim=0)
            act = torch.cat((act_exp, act_act), dim=0)
            rew = torch.cat((rew_exp, rew_act), dim=0)
            next_emb_exp, next_command_exp = next_obs_exp
            next_emb_act, next_command_act = next_obs_act
            next_obs = (torch.cat((next_emb_exp, next_emb_act), dim=0), torch.cat((next_command_exp, next_command_act), dim=0))
            done = torch.cat((done_exp, done_act), dim=0)

            critic_loss = self.lambda_q*self._compute_critic_loss(obs, act, rew, next_obs, done)

        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        if it%self.pol_freq_update==0:
            if is_pretraining:
                actor_loss = self._compute_bc_loss(obs_exp, act_exp, p_act_exp)
            else:
                actor_loss = self.lambda_bc*self._compute_bc_loss(obs_exp, act_exp, p_act_exp)
                actor_loss += self.lambda_a*self._compute_actor_loss(obs, p_act)
                actor_loss += self.env_w*self._compute_env_loss(obs, p_act)

            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self._update_env_model(obs, act, rew, next_obs)

    
    def _compute_bc_loss(self, obs, act, p_act):
        emb, command = obs
        pi_s = self.actor(emb, command, p_act)
        return torch.mean((pi_s-act)**2)
    
    def _compute_critic_loss(self, obs, act, rew, next_obs, done):
        emb, command = obs
        emb_, command_ = next_obs
        noise = torch.randn_like(act)*self.policy_noise
        noise = noise.clamp(-self.noise_clip, self.noise_clip).to(self.device)
        next_act = (self.actor_target(emb_, command_, act)+noise).clamp(self.policy_clip_min, self.policy_clip_max)

        q1, q2 = self.critic_target(emb_, command_, next_act)
        q = torch.min(q1, q2)
        q_target = rew + (self.gamma*(1-done)*q).detach()

        current_q1, current_q2 = self.critic(emb, command, act)

        critic_loss = F.mse_loss(current_q1, q_target) + F.mse_loss(current_q2, q_target)

        return critic_loss
    
    def _compute_actor_loss(self, obs, p_act):
        emb, command = obs
        loss = -self.critic.critic1(emb, command, self.actor(emb, command, p_act)).mean()
        return loss
    
    def _compute_env_loss(self, obs, p_act):
        emb, command = obs
        action = self.actor(emb, command, p_act)
        loss = 0
        for i in range(self.env_steps):
            emb, rew_pred = self.env_model(emb, action)
            loss += self.gamma**i*rew_pred
            action = self.actor(emb, command, action)
        return -loss.mean()
    
    def _update_env_model(self, obs, act, rew, next_obs):
        emb, _ = obs
        next_emb, _ = next_obs

        self.env_model_optimizer.zero_grad()
        transition = self.env_model.transition_model(emb, act)
        reward = self.env_model.reward_model(emb, act, next_emb)
        t_loss = F.mse_loss(transition, next_emb)
        r_loss = F.mse_loss(reward, rew)
        loss = t_loss + r_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.env_model.parameters(), 1.0)
        self.env_model_optimizer.step()

    def change_opt_lr(self, actor_lr, critic_lr):
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = actor_lr
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = critic_lr

    def load_exp_buffer(self, data):
        for d in data:
            self.expert_buffer.store_transition(d)

    def save(self):
        with open(self.save_path, 'wb') as f:
            pickle.dump(self, f)
