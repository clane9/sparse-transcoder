"""
A sparse transcoder for translating between high-dimensional feature spaces.

A sparse transcoder consists of two sparse autoencoders, an encoder and a decoder.
Transcoding is achieved by encoding with the encoder and decoding with the decoder.
"""

from typing import Tuple

import torch
from torch import nn

from .autoencoder import SparseAutoencoder


class SparseTranscoder(nn.Module):
    """
    A sparse transcoder for translating between high-dimensional feature spaces.
    """

    perm_indices: torch.Tensor

    def __init__(self, encoder: SparseAutoencoder, decoder: SparseAutoencoder):
        assert (
            encoder.latent_dim == decoder.latent_dim
        ), "encoder and decoder latent dimensions must match"
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.register_buffer(
            "perm_indices", torch.empty(encoder.latent_dim, dtype=torch.int64)
        )
        self.set_perm()

    def set_perm(self):
        """
        Set the permutation indices for aligning the encoder latent space to the
        decoder. The encoder latent dimensions are sorted so that their loadings
        approximately match the decoder.
        """
        perm_indices = permute_align(self.encoder.loadings(), self.decoder.loadings())
        self.perm_indices.copy_(perm_indices)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        code = self.encoder.encode(x)
        code = code[:, self.perm_indices]
        out = self.decoder.decode(code)
        return out, code


def permute_align(input: torch.Tensor, target: torch.Tensor):
    """
    Given 1D arrays `input`, `target` find indices such that `input[indices] ~= target`.

    Example::

    >>> import torch
    >>> target = torch.rand(10)
    >>> input = target[torch.randperm(10)]
    >>> indices = permute_align(input, target)
    >>> print(torch.allclose(input[indices], target))
    True
    """
    # First sort input then unsort by target's order.
    ind1 = torch.argsort(input)
    ind2 = torch.argsort(target)
    rank2 = torch.argsort(ind2)
    indices = ind1[rank2]
    return indices
