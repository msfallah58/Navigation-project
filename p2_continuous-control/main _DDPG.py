# 1- Start the Environment
from unityagents import UnityEnvironment
import numpy as np
from Agent_DDPG import Agent
import torch
import matplotlib.pyplot as plt
from collections import deque


# Define environment configuration
# TODO: Add the correct path of environment if needed
environment = UnityEnvironment(file_name="/home/saber/deep-reinforcement-learning/p2_continuous-control/Reacher20_Linux"
                                         "/Reacher.x86_64", no_graphics=False)
brain_name = environment.brain_names[0]
brain = environment.brains[brain_name]

# Get environment information
# Reset the environment
env_info = environment.reset(train_mode=True)[brain_name]
# number of agents
number_of_agents = len(env_info.agents)
states = env_info.vector_observations[0]
state_size = len(states)
action_size = brain.vector_action_space_size

figure_file = 'plots/unityagent.png'

agent = Agent(state_size, action_size, number_of_agents)


def train_agent(env=environment, num_agents=number_of_agents, n_episodes=400, t_max=1000):
    """
    The function trains the network for given number of episodes
    :param env: the environment
    :param agent: the agent
    :param num_agents: number of agents (int)
    :param n_episodes: number of episodes (int)
    :param t_max: Maximum episode time step (int)
    :return:
    """
    scores_window = deque(maxlen=100)
    scores_episode = []

    for episode in range(n_episodes):
        states = env.reset(train_mode=True)[brain_name].vector_observations  # Get initial state
        agent.reset()
        score = np.zeros(num_agents)

        for t_step in range(t_max):
            actions = agent.select_action(states)  # Get the actions
            env_info = env.step(actions)[brain_name]  # Send the actions to the environment
            next_states = env_info.vector_observations  # Get the next state
            rewards = env_info.rewards  # Get the reward
            dones = env_info.local_done  # See if episode has finished

            agent.step(states, actions, rewards, next_states, dones)
            score += rewards
            states = next_states

            if np.any(dones):  # if done, break
                break

        scores_episode.append(np.mean(score))
        scores_window.append(np.mean(score))

        print('\rEpisode: \t{} \tScore: \t{:.2f} \tAverage Score: \t{:.2f}'.format(episode, np.mean(score),
                                                                                   np.mean(scores_window)), end="")

        if np.mean(scores_window) >= 30.0:
            print(
                '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
            torch.save(agent.actor_network_local.state_dict(), 'saved_actor_network_DDPG.pth')
            torch.save(agent.critic_network_local.state_dict(), 'saved_critic_network_DDPG.pth')
            break

    return scores_episode


scores = train_agent()
environment.close()

fig_2 = plt.figure(2)
ax = fig_2.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.title('DDPG agent')

plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
