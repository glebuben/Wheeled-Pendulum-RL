import torch
from torch import nn
import numpy as np

OBSERVATION_DIM = 4


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        observation_dim=OBSERVATION_DIM,
        hidden_dim=128,
        action_dim=1,
        action_bound=1.0,
    ):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(observation_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)
        self.fc_value = nn.Linear(hidden_dim, 1)
        self.action_bound = action_bound
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.tanh(self.fc_mean(x)) * self.action_bound
        value = self.fc_value(x)
        # log_std = self.fc_log_std(x)
        # Fixed small std for stability
        log_std = -2.0 * torch.ones_like(mean) + np.log(self.action_bound)
        return mean, log_std, value
