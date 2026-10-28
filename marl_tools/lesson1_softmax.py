# Lesson 1- Softmax: from neural network output to action probabilities
# lets say your neural network produces three numbers called logits:
#  [2.0, 1.0, 0.1]
# These are raw scores, not probabilities. Softmax converts them into probabilities that sum to 1.

import torch
import torch.nn.functional as F

# Output of our neural network
logits = torch.tensor([2.0, 1.0, 0.1])

# Convert logits into probabilities
probs = F.softmax(logits, dim=0)

print("Logits:", logits)
print("Probabilities:", probs)
print("Sum of probabilities:", probs.sum())

# Expert insight: Softmax doesn't select the action. It defines a probability distribution over actions.
# We'll use Categorical in Lesson 2 to actually sample an action.
