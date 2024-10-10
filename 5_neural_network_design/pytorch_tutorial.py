'''
Authors: Danielle and Huldar
Date: 2024-10-09
Project:
Stuff for assignment 5 in data mining
Done from https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
'''
# Imports
import torch
import numpy as np
import time # For knowing the runtime

start_t = time.time()

# Tensors
#----------------------------------------
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# We move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
  print(f"Device tensor is stored on: {tensor.device}")# We move our tensor to the GPU if available

# Indexing and slicing
tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)

# Tensor concatenation
t1 = torch.cat([tensor, tensor, tensor], dim=0)
print(t1)

# This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")

# This computes the tensor multiplication
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")

# In place operations
print(tensor, "\n")
tensor.add_(5)
print(tensor)

x = tensor
y = tensor
x.copy_(y)
x.t_()

# Bridge to numpy
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy() # Note only a memory reference, not a copy
print(f"n: {n}")

# Change tensor, see how numpy array reacts
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# Numpy array to tensor
n = np.ones(5)
t = torch.from_numpy(n) # Note only a memory reference, not a copy

# Change numpy array, see how tensor reacts
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")































end_t = time.time()
runtime = end_t-start_t
print("Runtime: " + str(runtime*1000) + "ms")