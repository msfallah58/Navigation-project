import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Actor (policy) network"""

    def __init__(self, state_size, action_size, fc1_units=128, fc2_units=64):
        """Initialise parameters and build model

                Params
                ======
                    state_size (int): Dimension of state space
                    action_size (int): Dimension of action space
                    fc1_unit (int): Number of nodes in first hidden layer, default is 128
                    fc2_unit (int): Number of nodes in second hidden layer, default is 64
                returns action distribution
                """
        super().__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.std = nn.Parameter(torch.zeros(action_size))

    def forward(self, state):
        """Build a network that maps state --> action values"""
        x = F.relu(F.dropout(self.fc1(state)))
        x = F.relu(F.dropout(self.fc2(x)))
        mean = torch.tanh(self.fc3(x))
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        return dist


class Critic(nn.Module):
    """Critic model"""

    def __init__(self, state_size, fc1_units=128, fc2_units=64):
        """Initialise parameters and build model

                Params
                ======
                    state_size (int): Dimension of state space
                    action_size (int): Dimension of action space
                    fc1_unit (int): Number of nodes in first hidden layer, default is 128
                    fc2_unit (int): Number of nodes in second hidden layer, default is 64

                returns critic value
                """
        super().__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

    def forward(self, state):
        """Build a network that maps state --> action values"""
        x = F.relu(F.dropout(self.fc1(state)))
        x = F.relu(F.dropout(self.fc2(x)))
        return self.fc3(x)
