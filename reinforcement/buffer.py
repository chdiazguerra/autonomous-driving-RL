import numpy as np

import torch

class ReplayBuffer(object):
    """Buffer to store tuples of experience replay"""
    
    def __init__(self, max_size=20000):
        """
        Args:
            max_size (int): total amount of tuples to store
        """
        
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        """Add experience tuples to buffer
        (state, action, reward, next_state, done)
        
        Args:
            data (tuple): experience replay tuple
        """
        
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def __len__(self):
        """Returns the current size of buffer
        
        Returns:
            int: current size of buffer
        """
        
        return len(self.storage)

    def sample(self, batch_size):
        """Samples a random amount of experiences from buffer of batch size
        
        Args:
            batch_size (int): size of sample
        Returns:
            tuple: states, actions, rewards, next_states, dones
        """
        
        ind = np.random.choice(len(self.storage), size=batch_size, replace=False)
        states, actions, next_states, rewards, dones = [], [], [], [], []

        for i in ind: 
            s, a, r, s_, d = self.storage[i]
            states.append(np.array(s, copy=False, dtype=np.float32))
            actions.append(np.array(a, copy=False, dtype=np.float32))
            next_states.append(np.array(s_, copy=False, dtype=np.float32))
            rewards.append(np.array(r, copy=False, dtype=np.float32))
            dones.append(np.array(d, copy=False, dtype=np.float32))

        return np.array(states), np.array(actions), np.array(rewards).reshape(-1, 1), np.array(next_states), np.array(dones).reshape(-1, 1)
