import numpy as np
import torch
from typing import Tuple

class RandomPolynomialMapping:
    """
    Polynomial mapping with random coefficients.
    For multidimensional inputs, sums the entries after applying the polynomial.
    """
    
    def __init__(self, coeff_lb: int, coeff_ub: int, degree: int):
        self.coeffs = np.random.uniform(coeff_lb, coeff_ub, degree + 1)
        self.degree = degree
    def __call__(self, X):
        y = np.sum(np.polyval(self.coeffs, X), axis=1).reshape(-1, 1)
        return y
    def generate_random_data(self, d: int, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a dataset (X, y) where X is random and y is the polynomial mapping of X.
        
        Args:
            d (int): Dimension of the input.
            n (int): Number of data points.
        
        Returns:
            X (torch.Tensor): Input data.
            y (torch.Tensor): Output data.
        """

        X = (np.random.rand(n, d) * 2 - 1) * 2
        y = self(X)

        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        return X, y