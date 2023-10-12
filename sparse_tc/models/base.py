from typing import Tuple, Union

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim


class LitSparseCoder(pl.LightningModule):
    """
    A lightning module for a sparse coder.
    """

    def __init__(
        self,
        coder: "SparseCoder",
        *,
        lambd: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.coder = coder

    def training_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        recon, code = self.coder(batch)
        loss = self.loss_fn(batch, recon, code)
        sparsity = torch.sum(code.detach() > 0, dim=1).float().mean()
        self.log("train_loss", loss)
        self.log("train_sparsity", sparsity)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        recon, code = self.coder(batch)
        loss = self.loss_fn(batch, recon, code)
        sparsity = torch.sum(code.detach() > 0, dim=1).float().mean()
        self.log("val_loss", loss)
        self.log("val_sparsity", sparsity)
        return loss

    def loss_fn(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        code: torch.Tensor,
    ):
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


class SparseCoder(nn.Module):
    """
    A sparse coding base module.

    Args:
        in_shape: input feature shape
        latent_dim: sparse latent dimension
        bias: include bias in linear projection
    """

    def __init__(self, in_shape: Tuple[int, ...], latent_dim: int, bias: bool = True):
        super().__init__()
        self.in_shape = in_shape
        self.in_features = np.prod(in_shape)
        self.latent_dim = latent_dim

        self.flat = nn.Flatten()
        self.proj = nn.Linear(self.latent_dim, self.in_features, bias=bias)
        self.unflat = nn.Unflatten(1, in_shape)

        # Just for recording sparse code stats
        self.loading_ema = EMA(self.latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict reconstruction (batch_size, *in_shape) and sparse codes (batch_size,
        latent_dim) for input x.
        """
        x = self.flat(x)
        code = self.encode(x)
        recon = self.decode(code)

        # Update code stats.
        self.loading_ema(code.mean(dim=0))
        return recon, code

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the flattened input x and return sparse codes, shape (batch_size,
        latent_dim).
        """
        raise NotImplementedError

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        """
        Decode the sparse codes and return reconstructions, shape (batch_size, *in_shape).
        """
        recon = self.proj(code)
        recon = self.unflat(recon)
        return recon

    def loading(self) -> torch.Tensor:
        """
        Get the "loading" on each atom. (I.e. the running mean of the codes.)
        """
        return self.loading_ema.running_mean

    def atoms(self) -> torch.Tensor:
        """
        Get the dictionary atoms, shape `(latent_dim, in_shape)`.
        """
        atoms = self.proj.weight.detach().t()
        atoms = self.unflat(atoms)
        return atoms


class EMA(nn.Module):
    """
    Track the exponential moving average of a tensor.
    """

    step: torch.Tensor
    running_mean: torch.Tensor

    def __init__(self, shape: Union[int, Tuple[int, ...]], momentum: float = 0.9):
        super().__init__()
        self.shape = shape
        self.momentum = momentum

        self.register_buffer("running_mean", torch.empty(shape))
        self.register_buffer("step", torch.tensor(0, dtype=torch.int64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.update(x)
        return self.running_mean

    def update(self, x: torch.Tensor):
        x = x.detach()
        if self.step == 0:
            self.running_mean.data.copy_(x)
        else:
            self.running_mean.data.mul_(self.momentum).add_(x, alpha=1 - self.momentum)
        self.step.add_(1)

    def extra_repr(self) -> str:
        return f"shape={self.shape}, momentum={self.momentum:.2f}"
