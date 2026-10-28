import torch
import numpy as np

### Basic types of tensors
x = torch.empty(2, 3)  # Creates a tensor with uninitialized values, not necessarily zeros.
print(f"empty: {x}")
print(f"type(x): {type(x)}", '\n')

y = torch.zeros(2, 3)  # create an empty tensor filled with 0's size 2x3
print(f"zeros: {y}", '\n')

o = torch.ones(2)  # Creates a 1D tensor with 2 elements, shape (2,)
print(f"ones: {o}", '\n')

z = torch.rand(2, 2)  # create a tensor with values between 0-1 size 2x2
print(f"rand: {z}")
print(f"z size: {z.size()}", '\n')

t = torch.tensor([2.5, 0.2])  # a way to create a tensor with specific values
print(f"t: {t}", '\n')


### BASIC OPERATIONS
x = torch.rand(2, 2)
y = torch.rand(2, 2)
print(f"x: {x}", '\n')
print(f"y: {y}", '\n')

a = torch.add(x, y)  # addition 1
print(f"addition1: {a}", '\n')

a = x + y  # addition 2
print(f"addition2: {a}", '\n')

y.add_(x)  # this operation does y = y + x and modifies y
print(f"y: {y}", '\n')

# in PyTorch every operation with an underscore _ will do an inplace operation

s = x - y  # subtraction 1
print(f"subtraction1: {s}", '\n')

s = torch.sub(x, y)  # subtraction 2
print(f"subtraction2: {s}", '\n')

m = torch.mul(x, y)  # multiplication 1
print(f"multiplication1: {m}", '\n')

m = x * y  # multiplication 2
print(f"multiplication2: {m}", '\n')

y.mul_(x)  # this operation does y = y * x and modifies y
print(f"y: {y}", '\n')


### SAME WITH Division /


### Slicing same as we already know


### Tensor shapes
x = torch.rand(2, 3, 4)

print(f"x.shape: {x.shape}")              # torch.Size([2, 3, 4])
print(f"x.ndim: {x.ndim}")               # 3 dimensions
print(f"x.numel(): {x.numel()}")            # 24 total elements
print(f"x.dtype: {x.dtype}")              # torch.float32


### Reshaping
x = torch.rand(4, 4)
print(f"x: {x}", '\n')

y = x.view(-1, 2)  # Infers the number of rows needed to produce 2 columns.
print(f"y: {y}", '\n')

y = x.view(16)  # Changes the shape to (16,), a 1D tensor
print(f"y: {y}", '\n')


### Converting Numpy <-> PyTorch
a = torch.ones(5)
print(f"a as pytorch: {a}", '\n')
print(f"a type: {type(a)}", '\n')

b = a.numpy()
print(f"b as numpy: {b}", '\n')
print(f"b type: {type(b)}", '\n')

a = np.ones(5)
print(f"a as numpy: {a}", '\n')

b = torch.from_numpy(a)
print(f"b as numpy: {b}", '\n')

# If we modify a or b it will also modify the other one even though from different libraries


### GPU Acceleration
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)  # if you want to create a tensor on the GPU

    # or do this: first just create a tensor and then move it on the device
    y = torch.ones(5)
    y = y.to(device) # move the tensor to be on the GPU

    z = x + y  # this will be performed on the GPU











