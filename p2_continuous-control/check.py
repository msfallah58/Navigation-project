import numpy as np
import random
import torch

action = torch.arange(10)
print(action)
action = [torch.squeeze(t).item() for t in action]
print(action)