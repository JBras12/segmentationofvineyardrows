#!/usr/bin/env python3
"""
Training script for semantic segmentation networks.

Supports two architectures (toy and residual U-Net), three loss modes
(full, unconstrained, constrained), and periodically saves validation images.
"""

import argparse
from pathlib import Path
from typing import Any, Tuple
from operator import itemgetter

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, einsum
from torchvision import transforms
from torch.utils.data import DataLoader

# Project imports
from utils.dataset import SliceDataset
from utils.ShallowNet import shallowCNN
from utils.residual_unet import ResidualUNet
from utils.utils import (
    weights_init,
    saveImages,
    class2one_hot,
    probs2one_hot,
    one_hot,
    tqdm_,
    dice_coef
)
from utils.losses import (
    CrossEntropy,
    PartialCrossEntropy,
    NaiveSizeLoss
)

def setup(args) -> Tuple[nn.Module, torch.optim.Optimizer, torch.device, DataLoader, DataLoader]:
    """Set up the model, optimizer, device, and DataLoaders.

    Args:
        args: argparse.Namespace with `gpu`, `dataset`, `batch_size`.

    Returns:
        net: ready nn.Module for training.
        optimizer: Adam optimizer.
        device: torch.device (cpu or cuda).
        train_loader, val_loader: DataLoader for train/val splits.
    """
    gpu = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")

    num_classes = 2
    if args.dataset == 'TOY':
        # Toy model for quick testing
        net = shallowCNN(1, initial_kernels=4, num_classes=num_classes)
        net.apply(weights_init)
    else:
        # Deep residual U-Net
        net = ResidualUNet(in_dim=3, out_dim=num_classes)
        net.init_weights()
    net.to(device)

    # Adam optimizer with fixed learning rate
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

    # Custom transforms: RGB conversion, CHW format, normalization, to tensor
    transform = transforms.Compose([
        lambda img: img.convert('RGB'),
        lambda img: np.array(img).transpose(2, 0, 1),
        lambda arr: arr.astype(np.float32) / 255.0,
        lambda nd: torch.tensor(nd)
    ])
    mask_transform = transforms.Compose([
        lambda img: np.array(img),
        lambda nd: nd / 255,
        lambda nd: torch.tensor(nd, dtype=torch.int64)[None],
        lambda t: class2one_hot(t, K=2),
        itemgetter(0)
    ])

    # Dataset and DataLoader setup
    root_dir = Path("data") / args.dataset
    train_set = SliceDataset(
        'train', root_dir,
        transform=transform,
        mask_transform=mask_transform,
        augment=True, equalize=False
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=5,
        shuffle=True
    )
    val_set = SliceDataset(
        'val', root_dir,
        transform=transform,
        mask_transform=mask_transform,
        augment=False, equalize=False
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        num_workers=5,
        shuffle=False
    )

    return net, optimizer, device, train_loader, val_loader

def runTraining(args) -> None:
    """Runs full training and validation loop.

    Args:
        args: argparse.Namespace with `epochs`, `dataset`, `mode`, `batch_size`, `gpu`.
    """
    print(f">>> Setting up to train on {args.dataset} with mode={args.mode}")
    net, optimizer, device, train_loader, val_loader = setup(args)

    # Loss functions
    ce_loss    = CrossEntropy(idk=[0, 1])
    partial_ce = PartialCrossEntropy()
    sizeLoss   = NaiveSizeLoss()

    # Epoch loop
    for epoch in range(args.epochs):
        net.train()
        N = len(train_loader)
        # Logging tensors per batch
        log_ce       = torch.zeros(N, device=device)
        log_sizeloss = torch.zeros(N, device=device)
        log_sizediff = torch.zeros(N, device=device)
        log_dice     = torch.zeros(N, device=device)

        desc = f">> Training   ({epoch+1:4d}/{args.epochs})"
        iterator = tqdm_(enumerate(train_loader), total=N, desc=desc)

        for j, data in iterator:
            img       = data["img"].to(device)        # [B,3,H,W]
            full_mask = data["full_mask"].to(device)  # [B,2,H,W]
            weak_mask = data["weak_mask"].to(device)  # [B,2,H,W]
            bounds    = data["bounds"].to(device)     # [B,2,2]

            optimizer.zero_grad()

            # Forward pass
            logits        = net(img)
            pred_softmax  = F.softmax(5 * logits, dim=1)      # scaled softmax
            pred_seg      = probs2one_hot(pred_softmax)       # [B,2,H,W]

            # 1) Dice for class 1
            dices         = dice_coef(pred_seg, full_mask)[:, 1]
            log_dice[j]   = dices.mean()

            # 2) Mean size difference
            pred_size     = einsum("bkwh->bk", pred_seg)[:, 1].float()
            true_size_fg  = data["true_size"][:, 1].to(device).float()
            log_sizediff[j] = (pred_size - true_size_fg).mean()

            # 3) Loss selection based on mode
            if args.mode == 'full':
                loss_ce       = ce_loss(pred_softmax, full_mask)
                loss_size_val = torch.tensor(0.0, device=device)
            elif args.mode == 'unconstrained':
                loss_ce       = partial_ce(pred_softmax, weak_mask)
                loss_size_val = torch.tensor(0.0, device=device)
            else:  # constrained
                loss_ce_raw   = partial_ce(pred_softmax, weak_mask)
                loss_size_raw = sizeLoss(pred_softmax, bounds)
                loss_size_val = loss_size_raw.mean()
                loss_ce       = loss_ce_raw

            loss = loss_ce + loss_size_val

            # Logging
            log_ce[j]       = loss_ce.item()
            log_sizeloss[j] = loss_size_val.item()

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Update tqdm
            postfix = {
                "DSC":      f"{log_dice[:j+1].mean():05.3f}",
                "SizeDiff": f"{log_sizediff[:j+1].mean():+07.1f}",
                "LossCE":   f"{log_ce[:j+1].mean():5.2e}"
            }
            if args.mode == 'constrained':
                postfix["LossSize"] = f"{log_sizeloss[:j+1].mean():5.2e}"
            iterator.set_postfix(postfix)

        iterator.close()

        # Save validation images every 10 epochs
        if (epoch + 1) % 10 == 0:
            saveImages(net, val_loader, args.batch_size, epoch+1, args.dataset, args.mode, device)

def main() -> None:
    """Main entry: parses arguments and launches training."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     default=200, type=int)
    parser.add_argument('--dataset',    default='dataset', type=str)
    parser.add_argument('--mode',       default='constrained', choices=['full','unconstrained','constrained'])
    parser.add_argument('--gpu',        action='store_true')
    parser.add_argument('-b','--batch-size', default=4, type=int,
                        help='Batch size for training and validation')
    args = parser.parse_args()
    runTraining(args)

if __name__ == '__main__':
    main()
