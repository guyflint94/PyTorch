import torch

goal1 = torch.tensor([[2, 5, 3],
                     [8, 0, 7]])

left_ok = goal1 > 0
print(left_ok)