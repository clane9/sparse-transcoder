"""
Sparse coding with a multi-layer sparse autoencoder.

References:
    https://transformer-circuits.pub/2023/monosemantic-features/index.html
    https://github.com/jsulam/Online-Dictionary-Learning-demo
"""

from typing import Callable, Optional, Tuple

import torch
from torch import nn

from .base import SparseCoder

Layer = Callable[[], nn.Module]


class SparseAutoencoder(SparseCoder):
    """
    A multi-layer sparse auto-encoder.

    Args:
        in_shape: input feature shape
        latent_dim: sparse latent dimension
        depth: number of layers
    """

    def __init__(
        self,
        in_shape: Tuple[int, ...],
        latent_dim: int,
        depth: int = 1,
    ):
        super().__init__(in_shape=in_shape, latent_dim=latent_dim, bias=True)

        self.blocks = nn.ModuleList(
            [SparseLayer(self.in_features, self.latent_dim) for _ in range(depth)]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input x and return sparse codes, shape (batch_size, latent_dim).
        """
        # Each block acts a bit like one iteration of iterative hard thresholding.
        # See work by Jeremias Sulam for more details.
        #
        # Note that the Anthropic sparse autoencoder subtracts the decoder bias from x
        # before embedding. We achieve the same thing by initializing the code to 0 and
        # subtracting the reconstruction at every step.
        B = x.size(0)
        code = torch.zeros(B, self.latent_dim, device=x.device)
        for block in self.blocks:
            recon = self.proj(code)
            code = block(x - recon, pre_code=code)
        return code


class SparseLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        latent_dim: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim

        self.fc = nn.Linear(in_features, latent_dim, bias=bias)
        self.act = nn.ReLU()

    def forward(
        self, x: torch.Tensor, pre_code: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        code = self.fc(x)
        # Add previous code before relu.
        # Resembles the iterative hard thresholding update.
        if pre_code is not None:
            code = code + pre_code
        code = self.act(code)
        return code
