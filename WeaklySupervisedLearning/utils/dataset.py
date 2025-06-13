#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defines a PyTorch Dataset for image/mask slices, suitable for training,
validation, and testing. Produces image tensors, one-hot masks, true sizes,
and ±10% bounds per sample.
"""

from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
from torch import einsum
from PIL import Image, ImageOps
from torch.utils.data import Dataset

# Converts integer mask labels to one-hot [K, H, W]
from utils.utils import class2one_hot  


def make_dataset(root: Union[str, Path],
                 subset: str
                 ) -> List[Tuple[Path, Path]]:
    """Create a list of (image_path, mask_path) pairs for a given subset.

    Args:
        root: Base directory containing 'train', 'val', and 'test' subfolders.
        subset: One of 'train', 'val', or 'test'.

    Returns:
        A list of tuples, each containing (image_path, mask_path).

    Raises:
        AssertionError: If `subset` is invalid or image and mask counts differ.
    """
    assert subset in ['train', 'val', 'test'], f"Invalid subset: {subset}"
    root = Path(root)
    img_dir = root / subset / 'images'
    msk_dir = root / subset / 'masks'
    images = sorted(img_dir.glob("*.png"))
    masks = sorted(msk_dir.glob("*.png"))
    assert len(images) == len(masks), (
        f"Number of images ({len(images)}) != masks ({len(masks)})"
    )
    return list(zip(images, masks))


class SliceDataset(Dataset):
    """PyTorch Dataset for loading image/mask pairs with optional transforms,
    histogram equalization, and one-hot encoding of masks."""

    def __init__(self,
                 subset: str,
                 root_dir: Union[str, Path],
                 transform: Callable = None,
                 mask_transform: Callable = None,
                 augment: bool = False,
                 equalize: bool = False):
        """
        Args:
            subset: 'train', 'val', or 'test'.
            root_dir: Base dataset directory.
            transform: Function to apply to the image.
            mask_transform: Function to apply to the mask.
            augment: If True, apply augmentations (reserved for future use).
            equalize: If True, apply histogram equalization to the image.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mask_transform = mask_transform
        self.augment = augment      # placeholder for future augmentation
        self.equalize = equalize    # histogram equalization flag

        # Build file list and report dataset size
        self.files = make_dataset(self.root_dir, subset)
        print(f">> Created {subset} dataset with {len(self)} samples")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Load and process the sample at index `idx`, returning a dictionary:

            img         Tensor[C, H, W]
            full_mask   Tensor[K, H, W] (one-hot)
            weak_mask   Tensor[K, H, W] (same as full_mask)
            path        str            (image file path)
            true_size   Tensor[K]      (pixel count per class)
            bounds      Tensor[K, 2]   (±10% bounds on true_size)

        Args:
            idx: Sample index.

        Returns:
            A dictionary with keys 'img', 'full_mask', 'weak_mask', 'path',
            'true_size', and 'bounds'.
        """
        img_path, mask_path = self.files[idx]

        # 1) Load RGB image and grayscale mask (0 or 255)
        img = Image.open(img_path).convert('RGB')
        msk_i = Image.open(mask_path).convert('L')

        # 2) Optional histogram equalization
        if self.equalize:
            img = ImageOps.equalize(img)

        # 3) Apply image transforms (e.g., resize, normalize, ToTensor)
        if self.transform:
            img = self.transform(img)

        # 4) Apply mask transform or convert to one-hot
        if self.mask_transform:
            mask = self.mask_transform(msk_i)  # expects Tensor[K, H, W]
        else:
            arr = (np.array(msk_i) > 0).astype(np.int64)  # [H, W] ∈ {0,1}
            mask = class2one_hot(torch.from_numpy(arr), K=2)  # [2, H, W]

        # 5) full_mask and weak_mask are identical here
        full_mask = mask
        weak_mask = mask

        # 6) Compute pixel counts per class and ±10% bounds
        true_size = einsum("kwh->k", full_mask)  # Tensor[K]
        bounds = einsum("k,b->kb", true_size,
                        torch.tensor([0.9, 1.1], dtype=torch.float32))

        return {
            "img": img,
            "full_mask": full_mask,
            "weak_mask": weak_mask,
            "path": str(img_path),
            "true_size": true_size,
            "bounds": bounds
        }
