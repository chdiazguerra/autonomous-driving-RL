import torch
from torch import nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, emb_size=256):
        super().__init__()
        self.fc1 = nn.Linear(emb_size+2, 256)
        self.fc2 = nn.Linear(256, 256)

        self.fc3_left = nn.Linear(256, 256)
        self.out_left = nn.Linear(256, 2)

        self.fc3_right = nn.Linear(256, 256)
        self.out_right = nn.Linear(256, 2)

        self.fc3_straight = nn.Linear(256, 256)
        self.out_straight = nn.Linear(256, 2)

    def forward(self, emb, command, action):
        x = torch.cat((emb, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x_left = F.relu(self.fc3_left(x))
        x_left = self.out_left(x_left)

        x_straight = F.relu(self.fc3_straight(x))
        x_straight = self.out_straight(x_straight)

        x_right = F.relu(self.fc3_right(x))
        x_right = self.out_right(x_right)

        x = torch.stack((x_left, x_straight, x_right), dim=0)

        x = torch.gather(x, 0, command.expand((-1,2)).view(1,-1,2)).squeeze(0)

        x = torch.tanh(x)

        return x
    
class Critic(nn.Module):
    def __init__(self, emb_size=256):
        super().__init__()
        self.fc1 = nn.Linear(emb_size+2, 256)
        self.fc2 = nn.Linear(256, 256)

        self.fc3_left = nn.Linear(256, 256)
        self.out_left = nn.Linear(256, 1)

        self.fc3_right = nn.Linear(256, 256)
        self.out_right = nn.Linear(256, 1)

        self.fc3_straight = nn.Linear(256, 256)
        self.out_straight = nn.Linear(256, 1)

    def forward(self, emb, command, action):
        x = torch.cat((emb, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x_left = F.relu(self.fc3_left(x))
        x_left = self.out_left(x_left)

        x_straight = F.relu(self.fc3_straight(x))
        x_straight = self.out_straight(x_straight)

        x_right = F.relu(self.fc3_right(x))
        x_right = self.out_right(x_right)

        x = torch.stack((x_left, x_straight, x_right), dim=0)

        x = torch.gather(x, 0, command.view(1,-1,1)).squeeze(0)

        return x
    
class TwinCritic(nn.Module):
    def __init__(self, emb_size=256):
        super().__init__()
        self.critic1 = Critic(emb_size)
        self.critic2 = Critic(emb_size)

    def forward(self, emb, command, action):
        return self.critic1(emb, command, action), self.critic2(emb, command, action)
    
class Environment(nn.Module):
    def __init__(self, emb_size=256):
        super().__init__()
        self.fc1_transition = nn.Linear(2, 128)
        self.fc2_transition = nn.Linear(128+emb_size, 512)
        self.out_transition = nn.Linear(512, emb_size)

        self.fc1_reward = nn.Linear(2, 128)
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
