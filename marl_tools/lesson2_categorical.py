# Lesson 2 - Categorical: choosing an action in MAPPO
# The key idea :
# Softmax gives you probabilities. But in Reinforcement Learning, your agent needs to select an action.
# Categorical treats these probabilities as a distribution from which we can sample an action.

import torch
from torch.distributions import Categorical


import torch
from torch.distributions import Categorical

probs = torch.tensor([0.659, 0.242, 0.099])

# Create a probability distribution
dist = Categorical(probs=probs)

# Sample an action
action = dist.sample()

# Calculate the log probability of that action
log_prob = dist.log_prob(action)

print("Action:", action.item())
print("Log probability:", log_prob.item())


"""
- probs is a tensor containing the probabilities.
- dist is a probability distribution object that knows how to use those probabilities to sample actions and calculate their log probabilities.
Think of probs as the data and dist as the tool that operates on that data.

dist holds the information defining the categorical distribution, including its probabilities.
print(dist.probs)
# tensor([0.659, 0.242, 0.099])
Now you can use that distribution to perform operations:
action = dist.sample()
log_prob = dist.log_prob(action)
Why is this useful? In MAPPO, the neural network produces probabilities, 
and the distribution object helps us sample actions and evaluate the probabilities of those actions.

dist.log_prob(action)
dist is the object that knows the probability distribution.
.log_prob() is a method belonging to that object.
action is the action whose probability we want to evaluate.
log_prob is the resulting number stored in a variable.
"""

