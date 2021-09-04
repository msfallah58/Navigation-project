import torch
import numpy as np
import random
from collections import namedtuple, deque

from model import Network

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # mini batch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 0.001
UPDATE_EVERY = 10


class Agent:
    """Interacts with and learns from environment"""

    def __init__(self, seed, state_size, action_size, model_choice="DQN"):
        """
        :param model_choice: DQN or DDQN (string)
        :param state_size: size of state space (int)
        :param action_size: size of action space (int)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.network_local = Network(seed, state_size, action_size).to(device)
        self.network_target = Network(seed, state_size, action_size).to(device)
        self.optimiser = optim.Adam(self.network_local.parameters(), lr=LR)
        self.model_choice = model_choice
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialise time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def select_action(self, state, eps):
        """
        Returns actions for given state as per policy
        :param state: current state (array_like)
        :param eps: epsilon (float), for epsilon-greedy action selection
        :return: selected action (int)
        """

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.network_local.eval()
        with torch.no_grad():
            action_values = self.network_local(state)
        self.network_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples

        :param experiences: (Tuple[torch.Variable]): tuple of (s, a, r, s', done)
        :param gamma: discount factor (float)
        :return:
        """

        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        if self.model_choice == "DDQN":
            pass

        q_targets_next = self.network_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))
        # Get expected Q values from local model
        q_expected = self.network_local(states).gather(1, actions)
        # Compute loss
        loss = F.mse_loss(q_targets, q_expected)

        # Minimise the loss
        loss.backward()
        self.optimiser.step()

        # update target network
        self.soft_update(self.network_local, self.network_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """
        soft update model parameters
        theta_target = tau*theta_local + (1-tau)*theta_target

        :param local_model: weights will be copied from (Pytorch model)
        :param target_model: weights will be copied to (Pytorch model)
        :param tau: interpolation parameter (float)
        :return:
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """ Fixed-size buffer to store experience tuples"""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Initialise a ReplayBuffer object

        :param action_size: dimension of action space (int)
        :param buffer_size: maximum size of buffer (int)
        :param batch_size: size of each training batch (int)
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float() \
            .to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float() \
            .to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)
