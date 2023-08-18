import numpy as np
import torch

class OUNoise:
    def __init__(self, mu, sigma=0.4, theta=.6, dt=0.05, x0=None, noise_decay=0.0):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.noise_decay = noise_decay*sigma
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
        self.sigma = max(self.sigma-self.noise_decay, 0.01)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
    
class StepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.9, last_epoch=-1, min_lr=1e-6, verbose=False):
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group['lr'] for group in self.optimizer.param_groups]
        return [max(self.min_lr, group['lr'] * self.gamma)
                for group in self.optimizer.param_groups]