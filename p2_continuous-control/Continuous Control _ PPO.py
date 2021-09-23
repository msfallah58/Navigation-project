# 1- Start the Environment
from unityagents import UnityEnvironment
import numpy as np
from PPO_Agent import Agent
from utils import plot_learning_curve

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


def train_agent(env=environment, agent=agent_local, n_episodes=1000, t_max=1000):
    for i in range(n_episodes + 1):
        state = env.reset(train_mode=True)[brain_name].vector_observations[0]  # Get initial state
        done = False
        score = 0
        n_steps = 0
        learn_iters = 0
        learning_step = 100
        avg_score = 0
        best_score = 0

        while (not done) or (n_steps > t_max):
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
            #agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
              'time_steps', n_steps, 'learning_steps', learn_iters)
    return score_history


score = train_agent()
environment.close()

x = [i + 1 for i in range(len(score))]
plot_learning_curve(x, score_history, figure_file)
