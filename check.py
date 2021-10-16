# 1- Start the Environment
from unityagents import UnityEnvironment
import numpy as np
from Agent_DDPG import Agent
import torch
import matplotlib.pyplot as plt
from collections import deque


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# environment configuration
env = UnityEnvironment(file_name="/home/saber/deep-reinforcement-learning/p2_continuous-control/Reacher20_Linux"
                                         "/Reacher.x86_64", no_graphics=False)

brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# environment information
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
# number of agents in the environment
n_agents = len(env_info.agents)
print('Number of agents:', n_agents)
# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)
# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)