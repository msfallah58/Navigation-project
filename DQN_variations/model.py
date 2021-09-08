import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    """Actor (Policy) model without advantage function"""

    def __init__(self, state_size=37, action_size=4, fc1_units=64, fc2_units=64):
        """Initialise parameters and build model

        Params
        ======
            state_size (int): Dimension of state space
            action_size (int): Dimension of action space
            fc1_unit (int): Number of nodes in first hidden layer, default is 128
            fc2_unit (int): Number of nodes in second hidden layer
        """
        super().__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state --> action values"""
        x = F.relu(F.dropout(self.fc1(state)))
        x = F.relu(F.dropout(self.fc2(x)))
        return self.fc3(x)


class Dueling_Network(nn.Module):
    """Actor (Policy) model without advantage function"""

    def __init__(self, state_size=37, action_size=4, fc1_units=64, fc2_units=64, output_feature=1):
        """Initialise parameters and build model

        Params
        ======
            state_size (int): Dimension of state space
            action_size (int): Dimension of action space
            fc1_unit (int): Number of nodes in first hidden layer, default is 128
            fc2_unit (int): Number of nodes in second hidden layer
        """

        super().__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2_adv = nn.Linear(fc1_units, fc2_units)
        self.fc2_val = nn.Linear(fc1_units, fc2_units)
        self.fc3_adv = nn.Linear(fc2_units, action_size)
        self.fc3_val = nn.Linear(fc2_units, output_feature)

    def forward(self, state):
        """Build a network that maps state --> action values"""
        x = F.relu(F.dropout(self.fc1(state)))
        adv = F.relu(F.dropout(self.fc2_adv(x)))
        val = F.relu(F.dropout(self.fc2_val(x)))

        adv = self.fc3_adv(adv)
        val = self.fc3_val(val).exapnd(x.size(0), self.num_actions)

        x = val + adv - adv.mean(1).unsqueeze(1)
        return self.fc3(x)
