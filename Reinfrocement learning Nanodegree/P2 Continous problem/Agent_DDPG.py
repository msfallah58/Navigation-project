import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import copy
from networks import Actor_Network, Critic_Network
from collections import namedtuple, deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # mini batch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR_ACTOR = 1e-4  # learning rate of actor
LR_CRITIC = 1e-4  # learning rate of critic


class Agent:
    """Interacts with and learns from environment"""

    def __init__(self, state_size, action_size, n_agents, random_seed=12345):
        """
        :param model_choice: DQN or DDQN (string)
        :param state_size: size of state space (int)
        :param action_size: size of action space (int)
        :param n_agents: number of agents (int)
        :param random_seed: random seed (int)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents
        self.seed = random.seed(random_seed)

        # Actor Network
        self.actor_network_local = Actor_Network(state_size, action_size, random_seed).to(device)
        self.actor_network_target = Actor_Network(state_size, action_size, random_seed).to(device)
        self.actor_optimiser = optim.Adam(self.actor_network_local.parameters(), lr=LR_ACTOR)

        # Critic Network
        self.critic_network_local = Critic_Network(state_size, action_size, random_seed).to(device)
        self.critic_network_target = Critic_Network(state_size, action_size, random_seed).to(device)
        self.critic_optimiser = optim.Adam(self.critic_network_local.parameters(), lr=LR_CRITIC)

        # Noise process
        self.noise = OUNoise((n_agents, action_size), random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        # Initialise time step (for updating every UPDATE_EVERY steps)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        for i in range(self.n_agents):
            self.memory.add(state[i, :], action[i, :], reward[i], next_state[i, :], done[i])

        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def select_action(self, state, add_noise=True):
        """
        Returns actions for given state as per current policy
        :param state: current state (array_like)
        :param add_noise: add noise to the network (boolean)
        :return: selected actions (int)
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_network_local.eval()
        with torch.no_grad():
            action = self.actor_network_local(state).cpu().data.numpy()
        self.actor_network_local.train()

        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """
        Update policy and value parameters using given batch of experience tuples
        Q_targets = r (reward) + γ (discount factor) * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) --> action
            critic_target (state, action) --> Q-value

        :param experiences: (Tuple[torch.Variable]): tuple of (s, a, r, s', done)
        :param gamma: discount factor (float)
        :return:
        """

        states, actions, rewards, next_states, dones = experiences

        # --------------------------------- update critic ----------------------------------#
        # Get predicted next-state actions and Q values (for next states) from target models
        next_actions = self.actor_network_target(next_states)
        Q_targets_next = self.critic_network_target(next_states, next_actions)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_network_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimise the loss
        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_network_local.parameters(), 1)
        self.critic_optimiser.step()

        # -------------------------------- update actor -------------------------------------#
        # Compute actor loss
        actions_pred = self.actor_network_local(states)
        actor_loss = -self.critic_network_local(states, actions_pred).mean()
        # Minimise the loss
        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actor_network_local.parameters(), 1)
        self.actor_optimiser.step()

        # ------------------------------ update target networks -----------------------------#
        self.soft_update(self.critic_network_local, self.critic_network_target, TAU)
        self.soft_update(self.actor_network_local, self.actor_network_target, TAU)

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


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state


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
