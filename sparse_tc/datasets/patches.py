import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset, TensorDataset

from .image import imagenet
from .registry import DatasetPair, register_dataset


def extract_patches(
    dataset: Dataset,
    patch_size: int = 32,
    num_images: int = 100,
    seed: int = 42,
):
    """
    Extract patches from a dataset of images.

    Args:
        dataset: a dataset of images
        patch_size: square patch size
        num_images: number of images to sample from the dataset
        seed: random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))[:num_images]

    def get_image(idx):
        sample = dataset[idx]
        if isinstance(sample, tuple):
            return sample[0]
        return sample

    images = torch.stack([get_image(idx) for idx in indices])
    patches = rearrange(
        images,
        "b c (h p1) (w p2) -> (b h w) c p1 p2",
        p1=patch_size,
        p2=patch_size,
    )
    return patches


@register_dataset("imagenet_patches")
def imagenet_patches(
    root: Optional[str] = None,
    patch_size: int = 16,
    image_size: int = 224,
    num_images: int = 100,
    seed: int = 42,
    cache_dir: Optional[Union[str, Path]] = None,
    **kwargs,
) -> DatasetPair:
    """
    A dataset of image patches extracted from ImageNet.
    """
    logging.info(
        "Loading ImageNet patches "
        f"(patch_size={patch_size}, image_size={image_size}, "
        f"num_images={num_images}, seed={seed})"
    )
    if kwargs:
        logging.warning("Extra unused kwargs: %s", kwargs)
    root = root or os.environ.get("IMAGENET_ROOT")
    cache_dir = cache_dir or os.environ.get("SPARSETC_CACHE_DIR") or ".cache"

    ds_dir = (
        Path(cache_dir)
        / "datasets"
        / "imagenet_patches"
        / f"ps-{patch_size}_is-{image_size}_n-{num_images}_s-{seed}"
    )
    train_path = ds_dir / "train.pt"
    val_path = ds_dir / "val.pt"
    if train_path.exists() and val_path.exists():
        logging.info(f"Loading cached dataset from {ds_dir}")
        train_ds = TensorDataset(torch.load(train_path))
        val_ds = TensorDataset(torch.load(val_path))
        return train_ds, val_ds

    train_in1k, val_in1k = imagenet(root=root, image_size=image_size, crop_pct=1.0)

    logging.info("Extracting patches")
    train_patches = extract_patches(train_in1k, patch_size, num_images, seed)
    val_patches = extract_patches(val_in1k, patch_size, num_images, seed)
    logging.info(
        f"\n  Train patches: {train_patches.shape}"
        f"\n  Val patches: {val_patches.shape}"
    )

    logging.info(f"Saving patches to {ds_dir}")
    ds_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train_patches, train_path)
    torch.save(val_patches, val_path)

    train_ds = TensorDataset(train_patches)
    val_ds = TensorDataset(train_patches)
    return train_ds, val_ds
