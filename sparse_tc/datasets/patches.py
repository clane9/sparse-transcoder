import logging
import math
import os
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset, TensorDataset

from .image import cifar10, imagenet
from .registry import DatasetPair, register_dataset


def extract_patches(
    dataset: Dataset,
    patch_size: int = 32,
    num_patches: int = 50000,
    seed: int = 42,
) -> torch.Tensor:
    """
    Extract patches from a dataset of images.

    Args:
        dataset: a dataset of images
        patch_size: square patch size
        num_images: number of images to sample from the dataset
        seed: random seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    def get_image(idx) -> torch.Tensor:
        sample = dataset[idx]
        if isinstance(sample, tuple):
            return sample[0]
        return sample

    img = get_image(0)
    H = img.size(1)
    num_images = math.ceil(num_patches / ((H // patch_size) ** 2))
    indices = rng.permutation(len(dataset))[:num_images]

    images = torch.stack([get_image(idx) for idx in indices])
    patches = rearrange(
        images,
        "b c (h p1) (w p2) -> (b h w) c p1 p2",
        p1=patch_size,
        p2=patch_size,
    )
    return patches


def image_patch_dataset(
    ds_factory: Callable[[], DatasetPair],
    out_dir: Optional[Path] = None,
    patch_size: int = 16,
    num_train: int = 50000,
    num_val: int = 10000,
):
    if out_dir is not None:
        train_path = out_dir / "train_patches.pt"
        val_path = out_dir / "val_patches.pt"
        if train_path.exists() and val_path.exists():
            logging.info(f"Loading cached dataset from {out_dir}")
            train_ds = TensorDataset(torch.load(train_path))
            val_ds = TensorDataset(torch.load(val_path))
            return train_ds, val_ds

    train_images, val_images = ds_factory()

    logging.info("Extracting patches")
    train_patches = extract_patches(train_images, patch_size, num_train)
    val_patches = extract_patches(val_images, patch_size, num_val)
    logging.info(
        f"\n  Train patches: {train_patches.shape}"
        f"\n  Val patches: {val_patches.shape}"
    )

    if out_dir is not None:
        logging.info(f"Saving patches to {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(train_patches, train_path)
        torch.save(val_patches, val_path)

    train_ds = TensorDataset(train_patches)
    val_ds = TensorDataset(train_patches)
    return train_ds, val_ds


@register_dataset("imagenet_patches")
def imagenet_patches(
    root: Optional[str] = None,
    patch_size: int = 16,
    image_size: int = 224,
    **kwargs,
) -> DatasetPair:
    """
    A dataset of image patches extracted from ImageNet.
    """
    logging.info(
        "Loading ImageNet patches "
        f"(patch_size={patch_size}, image_size={image_size})"
    )
    if kwargs:
        logging.warning("Extra unused kwargs: %s", kwargs)
    root = root or os.environ.get("IMAGENET_ROOT")
    cache_dir = os.environ.get("SPARSE_TC_CACHE_DIR") or ".cache"

    out_dir = (
        Path(cache_dir) / "datasets" / f"imagenet_patches_p-{patch_size}_i-{image_size}"
    )
    train_ds, val_ds = image_patch_dataset(
        ds_factory=partial(imagenet, root=root, image_size=image_size, crop_pct=1.0),
        out_dir=out_dir,
        patch_size=patch_size,
    )
    return train_ds, val_ds


@register_dataset("cifar10_patches")
def cifar10_patches(
    root: Optional[str] = None,
    patch_size: int = 16,
    **kwargs,
) -> DatasetPair:
    """
    A dataset of image patches extracted from ImageNet.
    """
    logging.info(f"Loading CIFAR10 patches (patch_size={patch_size})")
    if kwargs:
        logging.warning("Extra unused kwargs: %s", kwargs)
    root = root or os.environ.get("CIFAR10_ROOT")
    cache_dir = os.environ.get("SPARSE_TC_CACHE_DIR") or ".cache"

    out_dir = Path(cache_dir) / "datasets" / f"cifar10_patches_p-{patch_size}"
    train_ds, val_ds = image_patch_dataset(
        ds_factory=partial(cifar10, root=root),
        out_dir=out_dir,
        patch_size=patch_size,
    )
    return train_ds, val_ds
