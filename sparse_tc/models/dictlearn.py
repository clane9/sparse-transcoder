"""
A pytorch implementation of online dictionary learning

References:
    https://github.com/jsulam/Online-Dictionary-Learning-demo/blob/master/Dictionary_Model.py
    https://www.ceremade.dauphine.fr/~carlier/FISTA
"""
import math

import torch


def fista(
    A: torch.Tensor, Y: torch.Tensor, lambd: float, num_iters: int = 20
) -> torch.Tensor:
    """
    Fast iterative shrinkage-thresholding algorithm to solve the lasso problem:

        min_X 1/2 || X A - Y ||_2^2 + lambd || X ||_1

    Returns the solution x.
    """
    if Y.ndim == 1:
        Y = Y[None, :]
        squeeze = True
    else:
        squeeze = False

    # Fix memory layout to be column major if necessary
    At = A.contiguous().t()
    A = A.t().contiguous().t()

    # Max eigenvalue of A^T A is the Lipschitz constant.
    L = torch.linalg.norm(A, ord=2) ** 2
    eta = 1 / L

    # Initialization
    X = torch.zeros((Y.size(0), A.size(0)), dtype=A.dtype, device=A.device)
    Z = X.clone()
    t = 1.0

    for _ in range(num_iters):
        Xprev = X.clone()

        # Proximal gradient step
        residual = Z @ A - Y
        X = soft_threshold(Z - eta * (residual @ At), eta * lambd)

        # Acceleration
        tprev = t
        t = (1.0 + math.sqrt(1.0 + 4.0 * t**2)) / 2.0
        Z = X + ((tprev - 1) / t) * (X - Xprev)

    if squeeze:
        X = X.squeeze(0)
    return X


def soft_threshold(x: torch.Tensor, lambd: float) -> torch.Tensor:
    """
    Soft-threshold x by lambd.
    """
    return x.sign() * torch.relu(x.abs() - lambd)
