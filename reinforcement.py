import random
import math
import numpy as np
from collections import namedtuple
from nets.dqn import dqn

import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torch import Tensor


# Transition
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Hyperparameter
GAMMA = 0.1
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000

class DQNAgent:
    def __init__(self, obs_dim, action_dim, batch_size=16, device='cuda'):
        self.memory = ReplayMemory(1000)
        self.batch_size = batch_size
        self.device = device
        
        # networks: dqn, dqn_target
        self.dqn = dqn(obs_dim, action_dim).to(device)
        self.dqn_target = dqn(obs_dim, action_dim).to(device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())
        self.criterion = nn.MSELoss()

        self.step_done = 0

    def train(self):
        self.dqn.train()

    def eval(self):
        self.dqn.eval()

    def select_action(self, state: np.ndarray) -> np.ndarray:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.step_done / EPS_DECAY)
        if random.random() < eps_threshold:
            action = torch.randint(0, 6, size=(1,)).squeeze()
        else:
            qval = self.dqn(state)
            _, predicted = torch.max(qval.data, 1)
            action = predicted[0]
        
        self.step_done += 1
        return action.item()

    def update_model(self) -> torch.Tensor:
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        if all(x is None for x in batch.next_state):    # if all the next_state are None
            return

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.tensor(batch.action, device=self.device).view(self.batch_size, -1)
        reward_batch = torch.tensor(batch.reward, device=self.device)

        # torch.Size([16, 1])
        state_action_values = self.dqn(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.dqn_target(non_final_next_states).max(1)[0]

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute loss
        loss = self.criterion(state_action_values.squeeze(), expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def target_hard_update(self):
        self.dqn_target.load_state_dict(self.dqn.state_dict())


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

def transform_action(action, x, y):
    '''
        Map actions to a specified range
    '''
    return x + action * (y - x)

def modify_image(image: Tensor, brightness_factor: float, saturation_factor: float, contrast_factor: float, sharpness_factor: float):
    """
        Adjusting the contrast, saturation, brightness and sharpness
    """
    # Adjust saturation & brightness & contrast & sharpness
    bright_img = TF.adjust_brightness(image, brightness_factor)
    saturation_img = TF.adjust_saturation(bright_img, saturation_factor)
    contrast_img = TF.adjust_contrast(saturation_img, contrast_factor)
    sharpness_img = TF.adjust_sharpness(contrast_img, sharpness_factor)
    
    return sharpness_img

def distortion_image(image: Tensor, max_kernel_size=5):
    '''
        Apply random blurring
    '''
    # Random blurring
    kernel_size = torch.randint(1, max_kernel_size+1, size=(1,)).item()
    kernel_size += kernel_size % 2 - 1  # Ensure odd kernel size
    image = TF.gaussian_blur(image, kernel_size)

    return image

def get_score(Avg_iou, precison, recall):
    return 0.25 * Avg_iou + 0.25 * precison + 0.5 * recall

def get_reward(RL_score, Origin_score, Distortion_score, EPS):
    score = RL_score*2 - Origin_score - Distortion_score
    reward = 1.0 if score > -EPS else -1.0
    return reward