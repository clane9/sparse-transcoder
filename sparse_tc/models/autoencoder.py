"""
A sparse auto-encoder for learning sparse representations.

References:
    https://transformer-circuits.pub/2023/monosemantic-features/index.html
    https://github.com/jsulam/Online-Dictionary-Learning-demo
"""

from typing import Callable, Optional, Tuple

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

Layer = Callable[[], nn.Module]


class LitSparseAE(pl.LightningModule):
    """
    A lightning module for sparse auto-encoder.
    """

    def __init__(
        self,
        in_features: int,
        latent_ratio: float = 8.0,
        depth: int = 1,
        lambd: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.coder = SparseAutoencoder(
            in_features=in_features,
            latent_ratio=latent_ratio,
            depth=depth,
        )

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        recon, code = self.coder(batch)
        loss = self.loss_fn(batch, recon, code)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        recon, code = self.coder(batch)
        loss = self.loss_fn(batch, recon, code)
        self.log("val_loss", loss)
        return loss

    def loss_fn(self, x: torch.Tensor, recon: torch.Tensor, code: torch.Tensor):
        # Note that codes are non-negative.
        return F.mse_loss(recon, x) + self.hparams.lambd * torch.mean(code)

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [scheduler]


class SparseAutoencoder(nn.Module):
    """
    A multi-layer sparse auto-encoder.

    Args:
        in_shape: input feature shape
        latent_ratio: latent dimension is `latent_ratio * in_features`
        depth: number of layers
    """

    def __init__(
        self,
        in_shape: Tuple[int, ...],
        latent_ratio: float = 8.0,
        depth: int = 1,
    ):
        super().__init__()
        self.in_shape = in_shape
        self.in_features = np.prod(in_shape)
        self.latent_ratio = latent_ratio
        self.latent_dim = int(latent_ratio * self.in_features)

        self.flat = nn.Flatten()
        self.blocks = nn.ModuleList(
            [SparseLayer(self.in_features, self.latent_dim) for _ in range(depth)]
        )
        self.proj = nn.Linear(self.latent_dim, self.in_features)
        self.unflat = nn.Unflatten(1, in_shape)
        # Just for recording sparse code stats
        self.stats = nn.BatchNorm1d(self.latent_dim, affine=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict reconstruction (batch_size, *in_shape) and sparse codes (batch_size,
        latent_dim) for input x.
        """
        code = self.encode(x)
        recon = self.decode(code)

        # Update code stats.
        self.stats(code.detach())
        return recon, code

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input x and return sparse codes, shape (batch_size, latent_dim).
        """
        # Each block acts a bit like one iteration of iterative hard thresholding.
        # See work by Jeremias Sulam for more details.
        B = x.size(0)
        x = self.flat(x)
        code = torch.zeros(B, self.latent_dim, device=x.device)
        for block in self.blocks:
            recon = self.proj(code)
            code = block(x - recon, pre_code=code)
        return code

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        """
        Decode the sparse codes and return reconstructions, shape (batch_size, *in_shape).
        """
        recon = self.proj(code)
        recon = self.unflat(recon)
        return recon

    def loadings(self) -> torch.Tensor:
        """
        Get the "loadings" on each atom. (I.e. the running mean of the codes.)
        """
        return self.stats.running_mean

    def atoms(self) -> torch.Tensor:
        """
        Get the dictionary atoms, shape `(latent_dim, in_shape)`.
        """
        atoms = self.proj.weight.detach().t()
        atoms = self.unflat(atoms)
        return atoms


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
