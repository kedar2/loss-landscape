"""
Experiments on the feasibility of random linear programs.
"""

import numpy as np
from scipy.optimize import linprog
from matplotlib import pyplot as plt
import seaborn as sns

def randomlp(m: int, n: int):
    """
    Simulates a random linear program Ax=b, x>=0, where A is m x n.
    The entries of A and b are taken to be standard Gaussian iid random variables.

    Args:
        m (int): number of constraints
        n (int): number of variables

    Returns:
        feasibility (bool): True if the linear program is feasible, False otherwise
    """

    A = np.random.randn(m, n)
    b = np.random.randn(m)

    # Solve the linear program
    res = linprog(np.zeros(n), A_ub=A, b_ub=b, bounds=(0, None))
    feasibility = res.success
    return feasibility
def simulate_probability_lp(m: int, n: int, num_simulations: int=1000):
    """
    Simulates the probability that a random linear program Ax=b, x>=0, where A is m x n, is feasible.
    The entries of A and b are taken to be standard Gaussian iid random variables.

    Args:
        m (int): number of constraints
        n (int): number of variables
        num_simulations (int): number of simulations to run
    Returns:
        prob (float): probability that the linear program is feasible
    """

    # Simulate the linear program
    feasibility = np.zeros(num_simulations)
    for i in range(num_simulations):
        feasibility[i] = randomlp(m, n)

    # Compute the probability
    prob = np.mean(feasibility)
    return prob
def generate_probability_table(dim: int=50, num_simulations: int=1000):
    """
    Generates a table of the probability that a random linear program Ax=b, x>=0, where A is m x n, is feasible.
    The entries of A and b are taken to be standard Gaussian iid random variables.

    Args:
        dim (int): max number of constraints and variables
        num_simulations (int): number of simulations to run
    Returns:
        prob (np.ndarray): probability that the linear program is feasible
    """

    # Generate the table
    m = np.arange(1, dim+1)
    n = np.arange(1, dim+1)
    prob = np.zeros((len(m), len(n)))
    for i, m_i in enumerate(m):
        for j, n_j in enumerate(n):
            prob[i, j] = simulate_probability_lp(m_i, n_j, num_simulations=num_simulations)
    return prob
def generate_probability_heatmap(dim: int=50, num_simulations: int=1000):
    """
    Generates a heatmap of the probability that a random linear program Ax=b, x>=0, where A is m x n, is feasible.
    The entries of A and b are taken to be standard Gaussian iid random variables.

    Args:
        dim (int): max number of constraints and variables
        num_simulations (int): number of simulations to run
    Returns:
        prob (np.ndarray): probability that the linear program is feasible
    """

    # Generate the table
    table = generate_probability_table(dim, num_simulations=num_simulations)

    # Plot the heatmap
    ax = sns.heatmap(table)
    ax.invert_yaxis()
    plt.xlabel("Number of variables")
    plt.ylabel("Number of constraints")
    plt.title("Probability that a random linear program is feasible")
    plt.show()

    return table

def run():
    import argparse
    parser = argparse.ArgumentParser(description="Experiments on the feasibility of random linear programs.")
    parser.add_argument("--dim", type=int, default=50, help="max number of constraints and variables")
    parser.add_argument("--num_simulations", type=int, default=100, help="number of simulations to run")
    args = parser.parse_args()

    # Run the experiment
    generate_probability_heatmap(dim=args.dim, num_simulations=args.num_simulations)

if __name__ == "__main__":
    run()