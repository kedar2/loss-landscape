"""
Experiments on sampling activation regions and determining probability of full rank.
"""

import sys
import os
if os.path.basename(os.getcwd()) == 'experiments':
    sys.path.append('..')

import numpy as np
from matplotlib import pyplot as plt
import torch
import metrics
import models

def random_binary_matrix_invertibility(n: int=100, num_trials: int=1000) -> float:
    """
    Given a matrix of size n x n with entries sampled from a Bernoulli distribution
    with parameter 0.5, compute the probability that the matrix is invertible.
    """
    num_invertible = 0
    for _ in range(num_trials):
        A = np.random.binomial(1, 0.5, size=(n, n))
        if np.linalg.matrix_rank(A) == n:
            num_invertible += 1
    return num_invertible / num_trials

def prob_jacobian_full_rank(n: int=100, input_dim: int=100, hidden_dim: int=12, num_trials: int=100) -> float:
    """
    Given specifications for an MLP, compute the probability that the Jacobian is of full rank
    with respect to the parameters at a random initialization.

    Args:
        n: The number of samples in the dataset.
        input_dim: The dimension of the input data.
        hidden_dim: The dimension of the hidden layer.
        num_trials: The number of trials to simulate.
    """

    num_full_rank = 0
    for _ in range(num_trials):
        model = models.MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, num_hidden_layers=1)
        X = torch.randn(n, input_dim)
        J = metrics.jacobian(model, X)
        if torch.linalg.matrix_rank(J).item() == n:
            num_full_rank += 1
    return num_full_rank / num_trials






def main():
    p = prob_jacobian_full_rank(input_dim=100, hidden_dim=25, num_trials=100)
    print(p)

if __name__ == "__main__":
    main()