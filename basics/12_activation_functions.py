import torch
import torch.nn as nn
import torch.nn.functional as F  # gives access to many useful functions used when building and training neural networks.

# Option 1 (Create nn modules)
# nn.Module is PyTorch's base class for neural networks. By inheriting from it, your class can manage layers, parameters, and training-related functionality.
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()  # replaces negative values with zero and leaves positive values unchanged:
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

# Option 2 (use activation functions directly in forward pass)
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out


"""
   *** class NeuralNet(nn.Module): ***
   
nn.Module is PyTorch's base class for neural networks.

Inheriting from it gives your class important functionality, such as:
- Tracking trainable parameters (weights and biases).
- Allowing model.parameters() to provide parameters to the optimizer.
- Managing layers and model operations.
"""

"""
   *** Why do we use super()? ***
   
   super(NeuralNet, self).__init__()
   
   This calls the initialization method of the parent class, nn.Module.
    It lets PyTorch initialize the internal machinery needed to manage your model's layers and parameters.
    Do we have to write it? Yes, you should call super().__init__() when defining a normal custom PyTorch model.
    The modern, shorter syntax is:
    
    super().__init__()
    
    Both forms do the same thing in this case.
"""


### Another example

class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.linear1 = nn.Linear(input_size, 5)  # First hidden layer: 5 neurons
        self.linear2 = nn.Linear(5, 3)           # Second hidden layer: 3 neurons
        self.linear3 = nn.Linear(3, 1)           # Output layer: 1 neuron

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x

model = NeuralNet(input_size=3)