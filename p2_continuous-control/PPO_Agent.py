import torch
import torch.optim as optim
import numpy as np
from networks import Actor, Critic
from collections import deque, namedtuple
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LR = 3e-4
LAMBDA = 0.96
EPSILON = 0.1
BETA = 0.01
UPDATE_EVERY = 100


class Agent:
    """Interacts with and learns from environment"""

    def __init__(self, state_size, action_size, GAMMA=0.99, gae_lambda=0.95,
                 policy_clip=0.2, BATCH_SIZE=64, BUFFER_SIZE=int(1e5), n_epochs=10):
        """
        :param model_choice: DQN or DDQN (string)
        :param state_size: size of state space (int)
        :param action_size: size of action space (int)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.policy_clip = policy_clip
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        # Actor and Critic Networks
        self.actor_network = Actor(state_size, action_size).to(device)
        self.critic_network = Critic(state_size, action_size).to(device)
        self.optimiser_actor = optim.Adam(self.actor_network.parameters(), lr=LR)
        self.optimiser_critic = optim.Adam(self.critic_network.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE)
        # Initialise time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, prob, val, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, prob, val, done)

        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn

            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def select_action(self, state):
        """
        Returns actions for given state as per policy
        :param state: current state (array_like)
        :param eps: epsilon (float), for epsilon-greedy action selection
        :return: selected action (int)
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        dist = self.actor_network(state)
        value = self.critic_network(state)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).detach().cpu().numpy()
        action = torch.squeeze(action).detach().cpu().numpy()
        value = torch.squeeze(value).item()

        return action, probs, value

    def learn(self, experiences):
        """
        Update value parameters using given batch of experience tuples

        :param experiences: (Tuple[torch.Variable]): tuple of (s, a, r, s', done)
        :param gamma: discount factor (float)
        :return:
        """

        for _ in range(self.n_epochs):

            states, actions, rewards, probs, vals, dones = experiences
            advantage = np.zeros(len(rewards), dtype=np.float32)

            for t in range(len(rewards) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(rewards) - 1):
                    a_t += discount * (rewards[k] + self.gamma * vals[k + 1] * (1 - int(dones[k])) - vals[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(device)

            dist = self.actor_network(states)
            critic_value = self.critic_network(states)
            critic_value = torch.squeeze(critic_value)
            old_probs = probs
            new_probs = dist.log_prob(actions)
            prob_ratio = new_probs.exp() / old_probs.exp()

            print("prob_ratio = ", list(prob_ratio.size()))
            print("prob_ratio = ", prob_ratio)
            print("advantage = ", list(advantage.size()))
            print("advantage = ", advantage)

            weighted_probs = advantage * prob_ratio
            weighted_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage
            actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
            returns = advantage + vals
            critic_loss = (returns - critic_value) ** 2
            critic_loss = critic_loss.mean()
            total_loss = actor_loss + 0.5 * critic_loss
            self.optimiser_actor.zero_grad()
            self.optimiser_critic.zero_grad()
            total_loss.backward()
            self.optimiser_actor.step()
            self.optimiser_critic.step()


class ReplayBuffer:
    """ Fixed-size buffer to store experience tuples"""

    def __init__(self, action_size, buffer_size, batch_size):
        """
        Initialise a ReplayBuffer object

        :param action_size: dimension of action space (int)
        :param buffer_size: maximum size of buffer (int)
        :param batch_size: size of each training batch (int)
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "prob", "val", "done"])

    def add(self, state, action, reward, prob, val, done):
        """Add a new experience to memory"""
        e = self.experience(state, action, reward, prob, val, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        probs = torch.from_numpy(np.vstack([e.prob for e in experiences if e is not None])).float() \
            .to(device)
        vals = torch.from_numpy(np.vstack([e.val for e in experiences if e is not None])).float() \
            .to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float() \
            .to(device)

        return states, actions, rewards, probs, vals, dones

    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)
