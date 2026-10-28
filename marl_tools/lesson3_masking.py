# Lesson 3 — Action masking in PyTorch for MAPPO
# 1. Why do we need action masking?
# Imagine a taxi is at an intersection. Its possible actions are: left (valid), right (blocked), wait (valid)
# Your policy might output these probabilities: P(a)=[0.6, 0.3, 0.1]
# Without masking, the taxi has a 30% chance of choosing the blocked road.
# Action masking prevents the agent from selecting invalid actions.

# A common approach to implement this is to mask the logits before applying softmax.


import torch
from torch.distributions import Categorical

# Neural network output: raw scores (logits)
logits = torch.tensor([2.0, 1.0, 0.1])

# True = valid action, False = invalid action
action_mask = torch.tensor([True, False, True])

# Make invalid actions impossible
masked_logits = logits.masked_fill(~action_mask, float("-inf"))

# Convert the remaining scores into probabilities
probs = torch.softmax(masked_logits, dim=0)

# Create the distribution and sample an action
dist = Categorical(probs=probs)
action = dist.sample()

print("Masked logits:", masked_logits)
print("Probabilities:", probs)
print("Sampled action:", action.item())


"""
Expected probabilities are approximately:
tensor([0.8699, 0.0000, 0.1301])

A. masked_fill(~action_mask, float("-inf"))
- ~action_mask reverses the Boolean mask: [False, True, False].
- masked_fill replaces the invalid action's logit with negative infinity.
- Softmax assigns that action probability zero.

B. Why mask logits rather than probabilities?
It's a convenient way to make invalid actions have exactly zero probability after softmax, 
while automatically renormalizing the valid actions so their probabilities sum to one.
"""
