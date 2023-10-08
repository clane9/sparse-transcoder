import logging
import os
from typing import Optional

from torchvision import transforms
from torchvision.datasets import ImageNet
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
    logging.info(f"Loading ImageNet (image_size={image_size}, crop_pct={crop_pct})")
    if kwargs:
        logging.warning("Extra unused kwargs: %s", kwargs)
    root = root or os.environ.get("IMAGENET_ROOT")
    if root is None:
        raise ValueError(
            "Either root or environment variable IMAGENET_ROOT is required"
        )

    logging.info(f"Loading ImageNet from {root}")
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
