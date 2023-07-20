import torch


# Example usage:
t = torch.tensor([0, 1, 0, 1, 2, 3])
mapped_t = torch.where(t > 0, 0.1, 10)
print(mapped_t)