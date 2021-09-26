import numpy as np
import random
import torch
tensor1 = torch.tensor([[1,2], [2, 3]])
tensor2 = torch.tensor([[2], [3]])
print("Tensor 1 =", tensor1)
print("Tensor 2 =", tensor2)
print(torch.mul(tensor1, tensor2).size())
print(torch.mul(tensor1, tensor2))
