"""
Experiments on sampling activation regions and determining probability of full rank.
"""

import sys
import os
if os.path.basename(os.getcwd()) == 'experiments':
    sys.path.append('..')

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import metrics
import models
from utils import generate_heatmap
from math import ceil

def random_binary_matrix_invertibility(n: int=100, num_trials: int=1000) -> float:
    """
    Given a matrix of size n x n with entries sampled from a Bernoulli distribution
    with parameter 0.5, compute the probability that the matrix is invertible.

    Args:
        n (int): The dimension of the matrix.
        num_trials (int): The number of trials to simulate.
    
    Returns:
        (float): The probability that the matrix is invertible (i.e. has full rank
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
        n (int): The number of samples in the dataset.
        input_dim (int): The dimension of the input data.
        hidden_dim (int): The dimension of the hidden layer.
        num_trials (int): The number of trials to simulate.
    
    Returns:
        (float): The probability that the Jacobian is of full rank.
    """

    num_full_rank = 0
    for _ in range(num_trials):
        model = models.MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=1, num_hidden_layers=1)
        X = torch.randn(n, input_dim)
        J = metrics.jacobian(model, X)
        if torch.linalg.matrix_rank(J).item() == n:
            num_full_rank += 1
    return num_full_rank / num_trials

def experiment_varying_input_dim(ratio=0.5):
    """
    Vary the number of samples in the dataset and plot the probability that the Jacobian is of full rank.
    The input dimension is proportional to the number of samples.

    Args:
        ratio (int): The ratio between the input dimension and the number of samples.
    """
    n_values = list(range(10, 110, 10))
    hidden_dim_values = list(range(1, 21, 1))
    prob_full_rank = lambda n, hidden_dim: prob_jacobian_full_rank(n=n, input_dim=ceil(n * ratio), hidden_dim=hidden_dim)
    generate_heatmap(n_values, hidden_dim_values, prob_full_rank, xlabel='Dataset size', ylabel='Network width', title='Probability of Full Rank Jacobian')

def experiment_input_dim_constant(input_dim: int=1):
    """
    Vary the number of samples in the dataset and plot the probability that the Jacobian is of full rank.
    The input dimension is kept constant.

    Args:
        input_dim (int): The input dimension.
    """
    n_values = list(range(10, 110, 10))
    hidden_dim_values = list(range(10, 110, 10))
    prob_full_rank = lambda n, hidden_dim: prob_jacobian_full_rank(n=n, input_dim=input_dim, hidden_dim=hidden_dim)
    generate_heatmap(n_values, hidden_dim_values, prob_full_rank, xlabel='Dataset size', ylabel='Network width', title='')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Experiments on sampling activation regions and determining probability of full rank.')
    parser.add_argument('--experiment', type=str, default='varying_input_dim', help='The experiment to run.')
    parser.add_argument('--ratio', type=float, default=0.5, help='The ratio between the input dimension and the number of samples.') # only used for experiment='varying_input_dim'
    parser.add_argument('--input_dim', type=int, default=100, help='The input dimension.')
    args = parser.parse_args()
    if args.experiment == 'varying_input_dim':
        experiment_varying_input_dim(ratio=args.ratio)
    elif args.experiment == 'input_dim_constant':
        experiment_input_dim_constant(input_dim=args.input_dim)
    else:
        raise ValueError('Invalid experiment name.')

if __name__ == "__main__":
    main()