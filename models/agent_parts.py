import torch
from torch import nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, emb_size=256):
        super().__init__()
        self.fc1 = nn.Linear(emb_size+3, 256)
        self.fc2 = nn.Linear(256, 256)

        self.fc3_left = nn.Linear(256, 256)
        self.out_left = nn.Linear(256, 3)

        self.fc3_right = nn.Linear(256, 256)
        self.out_right = nn.Linear(256, 3)

        self.fc3_straight = nn.Linear(256, 256)
        self.out_straight = nn.Linear(256, 3)

    def forward(self, img, action, command):
        x = torch.cat((img, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        if command==-1:
            x = F.relu(self.fc3_left(x))
            x = self.out_left(x)
        elif command==0:
            x = F.relu(self.fc3_straight(x))
            x = self.out_straight(x)
        elif command==1:
            x = F.relu(self.fc3_right(x))
            x = self.out_right(x)

        return x
    
class Critic(nn.Module):
    def __init__(self, emb_size=256):
        super().__init__()
        self.fc1 = nn.Linear(emb_size+3, 256)
        self.fc2 = nn.Linear(256, 256)

        self.fc3_left = nn.Linear(256, 256)
        self.out_left = nn.Linear(256, 1)

        self.fc3_right = nn.Linear(256, 256)
        self.out_right = nn.Linear(256, 1)

        self.fc3_straight = nn.Linear(256, 256)
        self.out_straight = nn.Linear(256, 1)

    def forward(self, emb, action, command):
        x = torch.cat((emb, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        if command==-1:
            x = F.relu(self.fc3_left(x))
            x = self.out_left(x)
        elif command==0:
            x = F.relu(self.fc3_straight(x))
            x = self.out_straight(x)
        elif command==1:
            x = F.relu(self.fc3_right(x))
            x = self.out_right(x)

        return x
    
class TwinCritic(nn.Module):
    def __init__(self, emb_size=256):
        super().__init__()
        self.critic1 = Critic(emb_size)
        self.critic2 = Critic(emb_size)

    def forward(self, emb, action, command):
        return self.critic1(emb, action, command), self.critic2(emb, action, command)
    
class Environment(nn.Module):
    def __init__(self, emb_size=256):
        super().__init__()
        self.fc1_transition = nn.Linear(3, 128)
        self.fc2_transition = nn.Linear(128+emb_size, 512)
        self.out_transition = nn.Linear(512, 200)

        self.fc1_reward = nn.Linear(3, 128)
        self.fc2_reward = nn.Linear(128+emb_size*2, 512)
        self.fc3_reward = nn.Linear(512, 256)
        self.out_reward = nn.Linear(256, 1)
    
    def forward(self, emb, action):
        o = F.relu(self.fc1_transition(action))
        o = torch.cat((o, emb), dim=1)
        o = F.relu(self.fc2_transition(o))
        o = self.out_transition(o)

        r = F.relu(self.fc1_reward(action))
        r = torch.cat((r, emb, o), dim=1)
        r = F.relu(self.fc2_reward(r))
        r = F.relu(self.fc3_reward(r))
        r = self.out_reward(r)

        return o, r
