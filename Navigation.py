import torch
import numpy as np
# import matplotlib.pyplot as plt
from agent_dqn import Agent
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

from pyvirtualdisplay import Display

display = Display(visible=0, size=(1400, 900))
display.start()

is_ipython = 'inline' in plt.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


def train_agent(n_episodes=1000, eps_start=1.0, eps_decay=0.995, eps_min=0.01):
    """The function trains an agent to navigate in a large, square world
    and collect as many yellow banana as possible. Either DQN or DDQN can be used

    Params
    =========
        n_episodes (int): maximum number of training episodes
        eps_start (float): starting value of epsilon for epsilon-greedy
        eps_min (float): minimum value of epsilon
        eps_decay (float): the decay factor for decreasing epsilon

    """
    env = UnityEnvironment(file_name="/data/Banana_Linux_NoVis/Banana.x86_64")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    scores = []
    eps = eps_start
    agent = Agent(state_size=37, action_size=4)

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # Reset the environment
        state = env_info.vector_observations[0]  # Get initial state
        score = 0  # Initialise the score
        while True:
            action = agent.select_action(state, eps)  # Select action
            env_info = env.step(action)[brain_name]  # Send the action to the environment
            next_state = env_info.vector_observations[0]  # Get the next state
            reward = env_info.rewards[0]  # Get the reward
            done = env_info.local_done[0]  # See if episode has finished
            score += reward  # Update the score
            state = next_state  # Roll over the state to next time step
            if done:
                break  # If episodes end, stop
        scores.append(score)  # Add the score of the episode to scores
        eps = max(eps_min, eps_decay * eps)  # Apply epsilon decay

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode - 100, np.mean(scores[-100:])))

        if np.mean(scores[-100:]) >= 13.0:
            torch.save(agent.network_local.state_dict(), 'saved_network.pth')
            break
    return scores


scores = train_agent()
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
