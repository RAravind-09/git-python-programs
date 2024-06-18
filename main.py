
# Tensors are a specialized data structure that are very similar to arrays and matrices.
# In PyTorch, we use tensors to encode the inputs and outputs of a model,
# as well as the model’s parameters.
#
# Tensors are similar to NumPy’s ndarrays,
# except that tensors can run on GPUs or other specialized hardware to accelerate computing.
# If you’re familiar with ndarrays, you’ll be right at home with the Tensor API.
# If not, follow along in this quick API walkthrough.

import torch
import numpy as np

# Tensor Initialization
# Tensors can be initialized in various ways. Take a look at the following examples:
#
# Directly from data
#
# Tensors can be created directly from data. The data type is automatically inferred.


data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)


    #     From a NumPy array
    #
    # Tensors can be created from NumPy arrays (and vice versa - see Bridge with NumPy).
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print("np" , np_array)
print("torch" , x_np)

    # From another tensor:
    #
    # The new tensor retains the properties (shape, datatype) of the argument tensor,
    # unless explicitly overridden.


x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

    # With random or constant values:
    #
    # shape is a tuple of tensor dimensions. In the functions below,
    # it determines the dimensionality of the output tensor.

shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

    # Tensor Attributes
    # Tensor attributes describe their shape, datatype, and the device on which they are stored.

tensor = torch.rand(3, 4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# NumPy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n)

# Changes in the NumPy array reflects in the tensor.
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")