import torch

from models.agent_parts import Actor, TwinCritic, Environment


class TD3ColDeductiveAgent:
    def __init__(self, obs_size=256, device='cpu', actor_lr=1e-3, critic_lr=1e-3,
                 pretraining=True):
        assert device in ['cpu', 'cuda'], "device must be either 'cpu' or 'cuda'"

        self.actor = Actor(obs_size).to(device)
        self.actor_target = Actor(obs_size).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = TwinCritic(obs_size).to(device)
        self.critic_target = TwinCritic(obs_size).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.env_model = Environment(obs_size).to(device)

        self.device = device
        self.pretraining = pretraining

    def set_pretraining(self, pretraining):
        self.pretraining = pretraining

    
