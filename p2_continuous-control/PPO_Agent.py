import torch
import torch.optim as optim
import numpy as np
from networks import Actor, Critic
from collections import deque, namedtuple

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LR = 1e-4
UPDATE_EVERY = 50


class Agent:
    """Interacts with and learns from environment"""

    def __init__(self, state_size, action_size, GAMMA=0.99, gae_lambda=0.95,
                 policy_clip=0.2, BATCH_SIZE=64, n_epochs=10):
        """
        :param state_size: size of state space (int)
        :param action_size: size of action space (int)
        :param GAMMA: discount factor (float)
        :param gae_lambda: smoothing factor of GAE
        :param policy_clip: clipping factor of the surrogate objective function
        :param BATCH_SIZE: batch size (int)
        :param BEFFER_SIZE: buffer size(int)
        :param n_epochs: number of epochs for training (int)

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
        self.memory = ReplayBuffer(action_size, BATCH_SIZE)
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
                self.learn()
                self.memory.clear_buffer()

    def select_action(self, state):
        """
        Returns actions, probabilities and critic value for given state as per policy
        :param state: current state (array_like)
        :return: selected action (int), probability of actions (float) and critic value (float)
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        dist = self.actor_network(state)
        value = self.critic_network(state)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).detach().cpu().numpy()
        action = torch.squeeze(action).detach().cpu().numpy()
        value = torch.squeeze(value).item()

        return action, probs, value

    def learn(self):
        """
        Update actor and critic networks using given batch of experience tuples

        :return:
        """

        for _ in range(self.n_epochs):

            states, actions, rewards, probs, vals, dones, batches = self.memory.rollouts()
            advantage = np.zeros(len(rewards), dtype=np.float32)

            # calculate the advantage for batch
            for t in range(len(rewards) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(rewards) - 1):
                    a_t += discount * (rewards[k] + self.gamma * vals[k + 1] * (1 - int(dones[k])) - vals[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = (advantage - advantage.mean())/advantage.std()
            advantage = torch.tensor(advantage).to(device)

            for batch in batches:
                states_batch = states[batch]
                # get the distribution of actions and the value of critic
                dist = self.actor_network(states_batch)
                critic_value = self.critic_network(states_batch)
                critic_value = torch.squeeze(critic_value)

                # ca\lculate old and new probabilities and their ratio
                old_probs = probs[batch]
                actions_batch = actions[batch]
                new_probs = dist.log_prob(actions_batch)
                prob_ratio = torch.exp(new_probs - old_probs)
                # calculate the surrogate objective and loss functions
                weighted_probs = prob_ratio * advantage[batch].unsqueeze(dim=1)
                clipped = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip)
                weighted_clipped_probs = clipped * advantage[batch].unsqueeze(dim=1)
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + vals[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()
                total_loss = actor_loss + 0.5 * critic_loss
                # do backpropagation and optimisation
                self.optimiser_actor.zero_grad()
                self.optimiser_critic.zero_grad()
                total_loss.backward()
                self.optimiser_actor.step()
                self.optimiser_critic.step()


class ReplayBuffer:
    """ Fixed-size buffer to store experience tuples"""

    def __init__(self, action_size, batch_size):
        """
        Initialise a ReplayBuffer object

        :param action_size: dimension of action space (int)
        :param buffer_size: maximum size of buffer (int)
        :param batch_size: size of each training batch (int)
        """

        self.action_size = action_size
        self.buffer = deque()
        self.batch_size = batch_size
        self.trajectory = namedtuple("Experience", field_names=["state", "action", "reward", "prob", "val", "done"])

    def add(self, state, action, reward, prob, val, done):
        """Add a new experience to memory"""
        e = self.trajectory(state, action, reward, prob, val, done)
        self.buffer.append(e)

    def rollouts(self):
        """Sample from the experience and returns a batch of states, actions, rewards, probabilities, values, dones"""
        states = torch.from_numpy(np.vstack([e.state for e in self.buffer if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in self.buffer if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in self.buffer if e is not None])).float().to(device)
        probs = torch.from_numpy(np.vstack([e.prob for e in self.buffer if e is not None])).float() \
            .to(device)
        vals = torch.from_numpy(np.vstack([e.val for e in self.buffer if e is not None])).float() \
            .to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in self.buffer if e is not None]).astype(np.uint8)).float() \
            .to(device)
        n_states = len(states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]
        return states, actions, rewards, probs, vals, dones, batches

    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.buffer)

    def clear_buffer(self):
        self.buffer.clear()
