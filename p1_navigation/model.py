import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    """Actor (Policy) Model"""

    def __init__(self, state_size=37, action_size=4, fc1_units=128, fc2_units=64):

        """Initialise parameters and build model

        Params
        ======
            state_size (int): Dimension of state space
            action_size (int): Dimension of action space
            seed (int): Random seed
            fc1_unit (int): Number of nodes in first hidden layer, default is 128
            fc2_unit (int): Number of nodes in second hidden layer
        """
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fc1 = nn.Linear(state_size, fc1_units).to(self.device)
        self.fc2 = nn.Linear(fc1_units, fc2_units).to(self.device)
        self.fc3 = nn.Linear(fc2_units, action_size).to(self.device)

    def forward(self, state):
        """Build a network that maps state --> action values"""
        x = F.relu(self.fc1(state)).to(self.device)
        x = F.relu(self.fc2(x)).to(self.device)
        return self.fc3(x).to(self.device)
