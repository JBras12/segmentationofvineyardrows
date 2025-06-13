#!/usr/bin/env python3

from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Callable, Iterable, List, Set, Tuple, TypeVar, cast

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor, einsum

# Custom tqdm for fixed-width progress bars
tqdm_ = partial(
    tqdm,
    ncols=125,
    leave=True,
    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]'
)


def weights_init(m: nn.Module) -> None:
    """Initialize weights for Conv2d/ConvTranspose2d and BatchNorm2d layers.

    Args:
        m (nn.Module): Layer/module to initialize.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Type variables for generic map functions
A = TypeVar("A")
B = TypeVar("B")

def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    """Sequentially apply `fn` to each element of `iter`.

    Args:
        fn (Callable[[A], B]): Function to apply.
        iter (Iterable[A]): Input iterable.

    Returns:
        List[B]: Results as a list.
    """
    return list(map(fn, iter))


def mmap_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    """Apply `fn` in parallel using multiprocessing.Pool.

    Args:
        fn (Callable[[A], B]): Function to apply.
        iter (Iterable[A]): Input iterable.

    Returns:
        List[B]: Results as a list.
    """
    return Pool().map(fn, iter)


def starmmap_(fn: Callable[[Tuple[A]], B], iter: Iterable[Tuple[A]]) -> List[B]:
    """Apply `fn(*args)` in parallel to each tuple in `iter` via Pool.starmap.

    Args:
        fn (Callable[[Tuple[A]], B]): Function accepting unpacked tuple args.
        iter (Iterable[Tuple[A]]): Iterable of argument tuples.

    Returns:
        List[B]: Results as a list.
    """
    return Pool().starmap(fn, iter)


def uniq(a: Tensor) -> Set:
    """Return the set of unique values in `a` (on CPU)."""
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    """Check if all values of `a` are in `sub`."""
    return uniq(a).issubset(sub)


def eq(a: Tensor, b: Tensor) -> bool:
    """Check if all elements of tensors `a` and `b` are equal."""
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis: int = 1) -> bool:
    """Check if `t` is a simplex along `axis` (sums to 1)."""
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis: int = 1) -> bool:
    """Check if `t` is one-hot: a simplex and contains only 0/1."""
    return simplex(t, axis) and sset(t, [0, 1])


def class2one_hot(seg: Tensor, K: int) -> Tensor:
    """Convert integer class tensor to one-hot format of size K.

    Args:
        seg (Tensor[B, H, W]): class values in {0,…,K-1}.
        K (int): Number of classes.

    Returns:
        Tensor[B, K, H, W]: One-hot encoded representation.

    Raises:
        AssertionError: if `seg` contains values outside [0,K-1].
    """
    assert sset(seg, list(range(K))), (uniq(seg), K)
    b, *img_shape = seg.shape
    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device)
    res.scatter_(1, seg[:, None, ...], 1)
    assert res.shape == (b, K, *img_shape)
    assert one_hot(res)
    return res


def probs2class(probs: Tensor) -> Tensor:
    """Convert softmax probabilities to class labels via argmax.

    Args:
        probs (Tensor[B, K, H, W]): Probability map.

    Returns:
        Tensor[B, H, W]: Integer class labels.
    """
    b, _, *img_shape = probs.shape
    assert simplex(probs)
    res = probs.argmax(dim=1)
    assert res.shape == (b, *img_shape)
    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    """Convert softmax probabilities to one-hot format.

    Args:
        probs (Tensor[B, K, H, W]): Probability map.

    Returns:
        Tensor[B, K, H, W]: One-hot equivalent.
    """
    _, K, *_ = probs.shape
    assert simplex(probs)
    res = class2one_hot(probs2class(probs), K)
    assert res.shape == probs.shape
    assert one_hot(res)
    return res


def saveImages(net: nn.Module,
               img_batch: Iterable[dict],
               batch_size: int,
               epoch: int,
               dataset: str,
               mode: str,
               device: torch.device) -> None:
    """Save a composite image of [RGB | prediction | weak mask] for each patch in validation.

    Args:
        net (nn.Module): Model for inference.
        img_batch (Iterable[dict]): List of dicts with keys "img", "weak_mask", "full_mask".
        batch_size (int): Batch size (not used here).
        epoch (int): Current epoch.
        dataset (str): Dataset name (e.g. "val").
        mode (str): Validation mode (e.g. "epoch").
        device (torch.device): Device for computation.
    """
    path = Path('results/images') / dataset / mode
    path.mkdir(parents=True, exist_ok=True)

    net.eval()
    desc = f">> Validation ({epoch:4d})"
    log_dice = torch.zeros(len(img_batch), device=device)
    iterator = tqdm_(enumerate(img_batch), total=len(img_batch), desc=desc)

    for j, data in iterator:
        img = data["img"].to(device)        # [B,3,H,W]
        weak_mask = data["weak_mask"].to(device)  # [B,2,H,W]
        full_mask = data["full_mask"].to(device)  # [B,2,H,W]

        # Inference with temperature scaling (factor 5)
        probs = F.softmax(5 * net(img), dim=1)
        segmentation = probs2class(probs)[:, None, ...].float()
        fg_mask = weak_mask[:, [1], ...].float()

        # Compute Dice for class 1
        log_dice[j] = dice_coef(probs2one_hot(probs), full_mask)[0, 1]

        B = img.size(0)
        for k in range(B):
            img_k = img[k]
            seg_k = segmentation[k]
            fg_k  = fg_mask[k]

            seg_k_color = seg_k.repeat(3, 1, 1)
            fg_k_color  = fg_k.repeat(3, 1, 1)
            grid = torch.stack([img_k, seg_k_color, fg_k_color], dim=0)

            torchvision.utils.save_image(
                grid,
                path / f"{j:03d}_{k}_Ep_{epoch:04d}.png",
                nrow=3,
                padding=2,
                normalize=False,
                pad_value=0
            )

        iterator.set_postfix({"DSC": f"{log_dice[:j+1].mean():05.3f}"})
    iterator.close()


def meta_dice(sum_str: str,
              label: Tensor,
              pred: Tensor,
              smooth: float = 1e-8) -> Tensor:
    """Generic Dice coefficient computation via einsum reduction.

    Args:
        sum_str (str): Reduction string for einsum (e.g., "bk...->bk").
        label (Tensor[B,K,...]): Ground truth one-hot.
        pred (Tensor[B,K,...]): Predicted one-hot.
        smooth (float): Small value to avoid division by zero.

    Returns:
        Tensor: Dice score for each batch/class as specified by sum_str.
    """
    assert label.shape == pred.shape
    assert one_hot(label)
    assert one_hot(pred)

    inter_size = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    sum_sizes = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(torch.float32)
    dices = (2 * inter_size + smooth) / (sum_sizes + smooth)
    return dices


# Aliases for standard Dice coefficient reductions
dice_coef  = partial(meta_dice, "bk...->bk")
dice_batch = partial(meta_dice, "bk...->k")


def intersection(a: Tensor, b: Tensor) -> Tensor:
    """Logical AND intersection between two one-hot binary tensors."""
    assert a.shape == b.shape
    assert sset(a, [0, 1]) and sset(b, [0, 1])
    res = a & b
    assert sset(res, [0, 1])
    return res


def union(a: Tensor, b: Tensor) -> Tensor:
    """Logical OR union between two one-hot binary tensors."""
    assert a.shape == b.shape
    assert sset(a, [0, 1]) and sset(b, [0, 1])
    res = a | b
    assert sset(res, [0, 1])
    return res
