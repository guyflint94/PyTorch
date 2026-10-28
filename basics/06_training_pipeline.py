import torch
import torch.nn as nn

# 1. Design model (input size, output size, forward pass)
# 2. Construct the loss and the optimizer
# 3. Training loop
#   3.1. forward pass - compute prediction
#   3.2. backward pass - gradients
#   3.3. update weights

# 32-bit floating-point values are commonly used for neural-network calculations.
# size is [4, 1]. 4 samples and 1 feature per sample
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)  # input
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)  # target values

X_test = torch.tensor([5], dtype=torch.float32)  # Create a test input

n_samples, n_features = X.shape
print(f"n_samples: {n_samples}, n_features: {n_features}")

input_size = n_features  # 1
output_size = n_features  # 1
model = nn.Linear(input_size, output_size)  # nn.Linear creates a linear layer. Its mathematical equation is y_hat = wx+b

print(f"Prediction before training: f(5) = {model(X_test).item():.3f}")  # .item() extracts the single numerical value from the output tensor.

# Training
learning_rate = 0.01
n_iter = 100

# torch.optim.SGD(...) creates an optimizer.
# model.parameters() gives the optimizer access to the model's trainable weight and bias.
# lr=learning_rate sets the learning rate to 0.01.
# The optimizer uses the gradients calculated by backpropagation to update the parameters.
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iter):
    # prediction = forward pass
    y_pred = model(X)

    # loss
    l = loss(y_pred, Y)

    # zero gradients
    optimizer.zero_grad()

    # gradients = backward pass
    l.backward() # dl/dw

    # update weights
    optimizer.step()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f"Epoch {epoch+1}: w={w[0][0].item():.3f}, Loss {l.item():.8f}")

print(f"Prediction after training: f(5) = {model(X_test).item():.3f}")