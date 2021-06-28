import numpy as np

import torch
a=torch.tensor([[2],[3],[4]])
print(a.size())
# torch.Size([3, 1])
a.expand(3,2)
print(a)