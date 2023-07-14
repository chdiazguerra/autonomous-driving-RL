import numpy as np

import torch

class ReplayBuffer(object):
    """Buffer to store tuples of experience replay"""
    
    def __init__(self, max_size=20000, device='cpu'):
        """
        Args:
            max_size (int): total amount of tuples to store
        """
        assert device in ['cpu', 'cuda'], "device must be either 'cpu' or 'cuda'"
        
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        self.device = device

    def store_transition(self, data):
        """Add experience tuples to buffer
        
        Args:
            data (tuple): experience replay tuple (prev_act, obs, action, reward, next_obs, done)
        """
        
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        """Samples a random amount of experiences from buffer of batch size
        
        Args:
            batch_size (int): size of sample
        """
        
        ind = np.random.choice(len(self.storage), size=batch_size, replace=False) #np.random.randint(0, len(self.storage), size=batch_size)
        prev_actions, embs, commands, actions, rewards, embs_, commands_, dones = [], [], [], [], [], [], [], []

        for i in ind: 
            pa, o, a, r, o_, d = self.storage[i]
            emb, command = o
            emb_, command_ = o_
            prev_actions.append(pa)
            embs.append(emb)
            commands.append(command)
            actions.append(a)
            embs_.append(emb_)
            commands_.append(command_)
            rewards.append(r)
            dones.append(d)

        prev_actions = torch.FloatTensor(prev_actions).to(self.device)
        embs = torch.FloatTensor(np.concatenate(embs)).to(self.device)
        commands = torch.LongTensor(commands).view(-1, 1).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        embs_ = torch.FloatTensor(np.concatenate(embs_)).to(self.device)
        commands_ = torch.LongTensor(commands_).reshape(-1, 1).to(self.device)
        rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(self.device)
        dones = torch.FloatTensor(dones).reshape(-1, 1).to(self.device)

        return prev_actions, (embs, commands), actions, rewards, (embs_, commands_), dones