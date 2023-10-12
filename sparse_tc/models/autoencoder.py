"""
A sparse auto-encoder for learning sparse representations.

References:
    https://transformer-circuits.pub/2023/monosemantic-features/index.html
    https://github.com/jsulam/Online-Dictionary-Learning-demo
"""

from typing import Callable, Optional, Tuple, Union

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
        in_shape: int,
        latent_dim: int,
        depth: int = 1,
        lambd: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.main = SparseAutoencoder(
            in_shape=in_shape,
            latent_dim=latent_dim,
            depth=depth,
        )

    def forward(
        self, batch: Union[Tuple[torch.Tensor, ...], torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        return self.main(batch)

    def training_step(self, batch, batch_idx):
        recon, code = self(batch)
        loss = self.loss_fn(batch, recon, code)
        sparsity = torch.sum(code.detach() > 0, dim=1).float().mean()
        self.log("train_loss", loss)
        self.log("train_sparsity", sparsity)
        return loss

    def validation_step(self, batch, batch_idx):
        recon, code = self(batch)
        loss = self.loss_fn(batch, recon, code)
        sparsity = torch.sum(code.detach() > 0, dim=1).float().mean()
        self.log("val_loss", loss)
        self.log("val_sparsity", sparsity)
        return loss

    def loss_fn(
        self,
        batch: Union[Tuple[torch.Tensor, ...], torch.Tensor],
        recon: torch.Tensor,
        code: torch.Tensor,
    ):
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        # Note that codes are non-negative.
        return F.mse_loss(recon, batch) + self.hparams.lambd * torch.mean(code)

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
        latent_dim: sparse latent dimension
        depth: number of layers
    """

    def __init__(
        self,
        in_shape: Tuple[int, ...],
        latent_dim: int,
        depth: int = 1,
    ):
        super().__init__()
        self.in_shape = in_shape
        self.in_features = np.prod(in_shape)
        self.latent_dim = latent_dim

        self.flat = nn.Flatten()
        self.blocks = nn.ModuleList(
            [SparseLayer(self.in_features, self.latent_dim) for _ in range(depth)]
        )
        self.proj = nn.Linear(self.latent_dim, self.in_features)
        self.unflat = nn.Unflatten(1, in_shape)

        # Just for recording sparse code stats
        self.loading_ema = EMA(self.latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict reconstruction (batch_size, *in_shape) and sparse codes (batch_size,
        latent_dim) for input x.
        """
        code = self.encode(x)
        recon = self.decode(code)

        # Update code stats.
        self.loading_ema(code.mean(dim=0))
        return recon, code

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
        step = self.step.item()
        if step == 0:
            self.running_mean.data.copy_(x)
        else:
            self.running_mean.data.mul_(self.momentum).add_(x, alpha=1 - self.momentum)
        self.step.add_(1)

    def extra_repr(self) -> str:
        return f"shape={self.shape}, momentum={self.momentum:.2f}"
