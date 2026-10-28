# Lesson 4 — PyTorch tensor indexing and gather for MAPPO
# dim=0 means "fix a column and move along rows"
# dim=1 means "fix a row and move along the columns"
# gather() is extremely useful when extracting selected Q-values, action scores,
# or probabilities from tensors in other RL implementations.

import torch

probs = torch.tensor([
    [0.7, 0.2, 0.1],
    [0.1, 0.6, 0.3],
    [0.2, 0.2, 0.6]
])

# Suppose the taxis sample these actions:
actions = torch.tensor([0, 2, 1])
"""That means Taxi 0 chose action 0, Taxi 1 chose action 2, and Taxi 2 chose action 1.
Our goal: retrieve the probability of the action each taxi actually chose."""

# Gather the probability for each taxi's chosen action
# Originally, actions has shape [3]. After unsqueeze(1), its shape becomes [3, 1].
# We add a dimension so the indices match the two-dimensional structure of probs.

selected_probs = probs.gather(
    dim=1,
    index=actions.unsqueeze(1)
)

print(selected_probs, '\n')

# Indexing

print(probs[0])
# tensor([0.7, 0.2, 0.1])

print(probs[0, 1])
# tensor(0.2)
