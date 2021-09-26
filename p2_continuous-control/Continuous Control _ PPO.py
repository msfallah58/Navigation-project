# 1- Start the Environment
from unityagents import UnityEnvironment
import numpy as np
from PPO_Agent import Agent
import torch
import matplotlib.pyplot as plt

# Define environment
environment = UnityEnvironment(file_name="/home/saber/deep-reinforcement-learning/p2_continuous-control/Reacher_Linux"
                                         "/Reacher.x86_64")
brain_name = environment.brain_names[0]
brain = environment.brains[brain_name]

env_info = environment.reset(train_mode=True)[brain_name]
states = env_info.vector_observations
state_size = states.shape[1]
action_size = brain.vector_action_space_size

agent_local = Agent(state_size, action_size)
n_steps = 0
score_history = []
figure_file = 'plots/unityagent.png'
learn_iters = 0
avg_score = 0


def train_agent(env=environment, agent=agent_local, n_episodes=50, t_max=1000):
    """
    The function trains the network for given number of episodes
    :param env: the environment
    :param agent: the agent
    :param n_episodes: number of episodes (int)
    :param t_max: Maximum episode time step (int)
    :return:
    """
    for i in range(n_episodes + 1):
        state = env.reset(train_mode=True)[brain_name].vector_observations[0]  # Get initial state
        done = False
        score = 0
        n_steps = 0
        best_score = 0 if i == 0 else best_score

        while (not done) and (n_steps < t_max):
            action, prob, val = agent.select_action(state)
            env_info = env.step(action)[brain_name]  # Send the action to the environment
            next_state = env_info.vector_observations[0]  # Get the next state
            reward = env_info.rewards[0]  # Get the reward
            done = env_info.local_done[0]  # See if episode has finished
            n_steps += 1
            score += reward
            agent.step(state, action, reward, prob, val, done)
            state = next_state
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'best score %.1f' % best_score, 'time_steps', n_steps)

        if best_score >= 30:
            torch.save(agent.actor_network.state_dict(), 'actor_network_PPO.pth')
            torch.save(agent.critic_network.state_dict(), 'critic_network_PPO.pth')
            break

    return score_history


score = train_agent()
environment.close()

fig_1 = plt.figure(1)
ax = fig_1.add_subplot(111)
plt.plot(np.arange(len(score)), score)
plt.title('PPO agent')

plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
