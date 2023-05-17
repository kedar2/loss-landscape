"""
Experiments on sampling activation regions and determining the probability of global minima within the region.
"""

import sys
import os
if os.path.basename(os.getcwd()) == 'experiments':
    sys.path.append('..')

import numpy as np
import torch
import models
from utils import generate_heatmap
from math import ceil
from scipy.optimize import linprog

def sample_activation_region(n: int=100, input_dim: int=100, hidden_dim: int=12) -> np.array:
    """
    Given specifications for an MLP, sample a random data point and initialization
    and compute the activation matrix A for the region.

    Args:
        n (int): The number of samples in the dataset.
        input_dim (int): The dimension of the input data.
        hidden_dim (int): The dimension of the hidden layer.
    
    Returns:
        A (np.array): The activation matrix.
        X (np.array): The input data.
        v (np.array): The frozen last layer.
    """

    model = models.MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, num_hidden_layers=1)
    X = torch.rand(n, input_dim)
    _ = model(X)
    X = X.numpy()
    A = model.activations[0].float().numpy()
    v = model.last_layer.view(-1).numpy()
    return A, X, v

def random_region_global_minima(n: int=2, input_dim: int=100, hidden_dim: int=12) -> bool:
    """
    Given specifications for an MLP, sample a random data point and initialization
    and determine whether the region contains a zero loss global minimum.

    Determine existence of global minima by solving the linear program
    min 0
    s.t. (A * (XW)) v = y,
            (2A - 1) * (XW) > 0,
    over the matrix variable W.
    Here * denotes the Hadamard (elementwise) product.

    Args:
        n (int): The number of samples in the dataset.
        input_dim (int): The dimension of the input data.
        hidden_dim (int): The dimension of the hidden layer.
    
    Returns:
        (bool): Whether the region contains a global minima.
    """

    A, X, v = sample_activation_region(n=n, input_dim=input_dim, hidden_dim=hidden_dim)
    y = np.random.randn(n) # Random outputs
    
    X = np.hstack((X, np.ones((n, 1)))) # Add bias term to X
    input_dim += 1 # Account for bias term

    equality_tensor = np.einsum('jk,ji,i->jki', X, A, v) # Equality constraint of LP
    equality_matrix = equality_tensor.reshape(n, input_dim * hidden_dim) # Flatten to matrix

    inequality_tensor = np.einsum('jk,ji,il->ljki', X, 2 * A - 1, np.eye(hidden_dim)) # Inequality constraint of LP
    inequality_matrix = inequality_tensor.reshape(n * hidden_dim, input_dim * hidden_dim) # Flatten to matrix

    # Determine whether LP is feasible
    res = linprog(c=np.zeros(input_dim * hidden_dim),
                  A_eq=equality_matrix,
                  b_eq=y,
                  A_ub=-inequality_matrix,
                  b_ub=np.zeros(n * hidden_dim))
    
    return res.success

if __name__ == "__main__":
    exists = random_region_global_minima(n=1, input_dim=1, hidden_dim=1)
    print(exists)