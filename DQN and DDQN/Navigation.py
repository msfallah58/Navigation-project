import matplotlib.pyplot as plt
import numpy as np
import torch
from unityagents import UnityEnvironment

from agent_dqn import Agent
model_choice = "DDQN"

agent_global = Agent(state_size=37, action_size=4, model_choice=model_choice)

# TODO: update the path according to your local folder
environment = UnityEnvironment(
    file_name="/home/saber/deep-reinforcement-learning/p1_navigation/Banana_Linux/Banana.x86_64", worker_id=0)
brain_name = environment.brain_names[0]
brain = environment.brains[brain_name]


# Train the Agent with chosen Deep Q_learning Technique
def train_agent(env, agent, n_episodes=1000, eps_start=1.0, eps_decay=0.995, eps_min=0.01):
    """The function trains an agent using DQN or Double DQN (DDQN)
    techniques. The technique can be chosen when the Agent class is called.

    Params
    =========
        n_episodes (int): maximum number of training episodes
        eps_start (float): starting value of epsilon for epsilon-greedy
        eps_min (float): minimum value of epsilon
        eps_decay (float): the decay factor for decreasing epsilon

    """
    eps = eps_start
    scores = []

    for i_episode in range(1, n_episodes + 1):
        state = env.reset(train_mode=True)[brain_name].vector_observations[0]  # Get initial state
        score = 0  # Initialise the score

        while True:
            action = agent.select_action(state, eps)  # Select action
            env_info = env.step(action)[brain_name]  # Send the action to the environment
            next_state = env_info.vector_observations[0]  # Get the next state
            reward = env_info.rewards[0]  # Get the reward
            done = env_info.local_done[0]  # See if episode has finished
            agent.step(state, action, reward, next_state, done)

            score += reward  # Update the score
            state = next_state  # Roll over the state to next time step
            if done:
                break  # If episodes end, stop
        scores.append(score)  # Add the score of the episode to scores
        eps = max(eps_min, eps_decay * eps)  # Apply epsilon decay
        average_size = 100
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores[-average_size:])), end="")
        if i_episode % average_size == 0:
            print(
                '\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode - average_size, np.mean(scores[-average_size:])))

        if np.mean(scores[-average_size:]) >= 13.0:
            if model_choice == "DQN":
                torch.save(agent.network_local.state_dict(), 'saved_network_DQN.pth')
                break
            else:
                torch.save(agent.network_local.state_dict(), 'saved_network_DDQN.pth')
                break
    return scores


scores = train_agent(environment, agent_global)
environment.close()

# plotting per episode score achieved by the agent
fig_1 = plt.figure(1)
ax = fig_1.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
if model_choice == "DQN":
    plt.title('DQN agent')
else:
    plt.title('DDQN agent')

plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
