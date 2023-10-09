import logging
import os
from pathlib import Path
from typing import Optional

from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageNet
from torchvision.transforms import InterpolationMode

from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .registry import DatasetPair, register_dataset


@register_dataset("imagenet")
def imagenet(
    root: Optional[str] = None,
    image_size: int = 224,
    crop_pct: float = 0.875,
    **kwargs,
) -> DatasetPair:
    """
    Load ImageNet dataset.
    """
    root = root or os.environ.get("IMAGENET_ROOT")
    logging.info(
        f"Loading ImageNet (root={root}, image_size={image_size}, crop_pct={crop_pct})"
    )
    if root is None:
        raise ValueError(
            "Either root or environment variable IMAGENET_ROOT is required"
        )
    if kwargs:
        logging.warning("Extra unused kwargs: %s", kwargs)

    transform = transforms.Compose(
        [
            transforms.Resize(
                int(image_size / crop_pct), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            ),
        ]
    )

    train_in1k = ImageNet(root, split="train", transform=transform)
    val_in1k = ImageNet(root, split="val", transform=transform)
    return train_in1k, val_in1k


@register_dataset("cifar10")
def cifar10(root: Optional[str] = None, **kwargs) -> DatasetPair:
    """
    Load CIFAR-10 dataset.
    """
    root = root or os.environ.get("CIFAR10_ROOT")
    if root is None:
        cache_dir = os.environ.get("SPARSE_TC_CACHE_DIR") or ".cache"
        root = Path(cache_dir) / "datasets" / "cifar10"
    logging.info(f"Loading CIFAR-10 (root={root})")
    if kwargs:
        logging.warning("Extra unused kwargs: %s", kwargs)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )

    train_cifar10 = CIFAR10(root, train=True, download=True, transform=transform)
    val_cifar10 = CIFAR10(root, train=False, download=True, transform=transform)
    return train_cifar10, val_cifar10
