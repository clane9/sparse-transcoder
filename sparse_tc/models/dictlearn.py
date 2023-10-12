"""
Sparse coding with online sparse dictionary learning.

References:
    https://github.com/jsulam/Online-Dictionary-Learning-demo/blob/master/Dictionary_Model.py
    https://www.ceremade.dauphine.fr/~carlier/FISTA
"""
import math
from typing import Optional, Tuple

import torch

from .base import SparseCoder


class SparseDictEncoder(SparseCoder):
    """
    A sparse dictionary encoder-decoder. Sparse codes are computed with FISTA.

    Args:
        in_shape: input feature shape
        latent_dim: sparse latent dimension
        lambd: l1 penalty
        num_iters: number of fista iterations
        update_eta_freq: how often to update the fista step size
    """

    def __init__(
        self,
        in_shape: Tuple[int, ...],
        latent_dim: int,
        lambd: float = 0.1,
        num_iters: int = 20,
        update_eta_freq: int = 100,
    ):
        super().__init__(in_shape=in_shape, latent_dim=latent_dim, bias=False)
        self.lambd = lambd
        self.num_iters = num_iters
        self.update_eta_freq = update_eta_freq

        self._step = 0
        self._eta = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input x and return sparse codes, shape (batch_size, latent_dim).
        """
        A = self.proj.weight.detach()
        x = x.detach()

        if self._step % self.update_eta_freq == 0:
            self._eta = fista_step_size(A)
        self._step += 1

        code = fista(A, x, lambd=self.lambd, num_iters=self.num_iters, eta=self._eta)
        return code


def fista(
    A: torch.Tensor,
    Y: torch.Tensor,
    lambd: float,
    num_iters: int = 20,
    eta: Optional[float] = None,
) -> torch.Tensor:
    """
    Fast iterative shrinkage-thresholding algorithm to solve the lasso problem:

        min_X 1/2 || X A^T - Y ||_2^2 + lambd || X ||_1

    Returns the solution x.
    """
    if Y.ndim == 1:
        Y = Y[None, :]
        squeeze = True
    else:
        squeeze = False

    if eta is None:
        eta = fista_step_size(A)

    # Fix memory layout to be column major if necessary
    At = A.contiguous().t()
    A = A.t().contiguous().t()

    # Initialization
    X = torch.zeros((Y.size(0), A.size(1)), dtype=A.dtype, device=A.device)
    Z = X.clone()
    t = 1.0

    for _ in range(num_iters):
        Xprev = X.clone()

        # Proximal gradient step
        residual = Z @ At - Y
        X = soft_threshold(Z - eta * (residual @ A), eta * lambd)

        # Acceleration
        tprev = t
        t = (1.0 + math.sqrt(1.0 + 4.0 * t**2)) / 2.0
        Z = X + ((tprev - 1) / t) * (X - Xprev)

    if squeeze:
        X = X.squeeze(0)
    return X


def fista_step_size(A: torch.Tensor) -> float:
    """
    Get the default FISTA step size which is 1 over the gradient Lipschitz constant.
    I.e. the largest eigenvalue of A^T A.
    """
    L = (torch.linalg.norm(A, ord=2) ** 2).item()
    return 1 / L


def soft_threshold(x: torch.Tensor, lambd: float) -> torch.Tensor:
    """
    Soft-threshold x by lambd.
    """
    return x.sign() * torch.relu(x.abs() - lambd)
