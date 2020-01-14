"""
run as:
python pytorch01.py
https://www.youtube.com/watch?v=BzcBsTou0C0
"""

import torch

x = torch.Tensor([5, 3])
y = torch.Tensor([2, 1])

print(x * y)

x = torch.zeros([2, 5])

x.shape

y = torch.rand([2, 5])
print(y)
print(y.view([1, 10]))
