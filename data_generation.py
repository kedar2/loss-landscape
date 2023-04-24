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
        return np.sum(np.polyval(self.coeffs, X), axis=1)
    def generate_gaussian_data(self, d: int, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a dataset (X, y) where X is Gaussian and y is the polynomial mapping of X.
        
        Args:
            d (int): Dimension of the input.
            n (int): Number of data points.
        
        Returns:
            X (torch.Tensor): Input data.
            y (torch.Tensor): Output data.
        """

        X = np.random.normal(size=(n, d))
        y = self(X)

        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        return X, y
    
if __name__ == "__main__":
    poly = RandomPolynomialMapping(-1, 1, 2)
    X, y = poly.generate_gaussian_data(10, 1000)
    print("Shape of X: ", X.shape)
    print("Shape of y: ", y.shape)