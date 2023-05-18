import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, action_size):
        """Initialize parameters and build model.
        Params
        ======
            action_size (int): Dimension of each action
        """
        super(Actor, self).__init__()

        self.resnet18 = torch.nn.Sequential(*(list(models.resnet18(weights=models.ResNet18_Weights.DEFAULT).children())[:-2]))
        for param in self.resnet18.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(512 * 7 * 7, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, action_size)  

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.resnet18(state)
        x = x.view(-1, 512 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, action_size):
        """Initialize parameters and build model.
        Params
        ======
            action_size (int): Dimension of each action
        """
        super(Critic, self).__init__()

        self.resnet18 = torch.nn.Sequential(*(list(models.resnet18(weights=models.ResNet18_Weights.DEFAULT).children())[:-2]))
        for param in self.resnet18.parameters():
            param.requires_grad = False
            
        self.fc1 = nn.Linear(512 * 7 * 7 + action_size, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = self.resnet18(state)
        x = x.view(-1, 512 * 7 * 7)
        x = torch.cat([x, action], dim=1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))

        return self.fc3(x)
