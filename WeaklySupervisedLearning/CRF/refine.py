#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module to refine segmentation masks using DenseCRF on image patches.

This module splits each input image and its predicted mask into fixed-size
patches, applies DenseCRF to improve spatial consistency, and then reassembles
the refined patches back into the original image dimensions. Processing is
parallelized using Python multiprocessing.
"""

import os
import math
from glob import glob
from multiprocessing import Pool, cpu_count

import numpy as np
import cv2
from tqdm import tqdm

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import (
    unary_from_softmax,
    create_pairwise_gaussian,
    create_pairwise_bilateral
)

# ── CONFIGURATION ──────────────────────────────────
IMGS_DIR    = "/home/rodrigo/JoseBras/Imagens"
PREDS_DIR   = "/home/rodrigo/JoseBras/PredictMasks/Predict_Masks_WSL_resnet50"
OUT_DIR     = "/home/rodrigo/JoseBras/refinadas"

EXT_IMG     = ".tif"
EXT_MASK    = ".png"

# Fixed patch size (height, width)
PATCH_SIZE  = (512, 512)

# Number of CRF iterations
N_ITERS     = 10

# CRF parameters
SXY_GAUSS    = (3, 3)
COMPAT_GAUSS = 3
SXY_BILAT    = (80, 80)
SCHAN_BILAT  = (13, 13, 13)
COMPAT_BILAT = 10

# Number of parallel processes (leaving one core free)
N_WORKERS = max(1, cpu_count() - 1)

os.makedirs(OUT_DIR, exist_ok=True)

# Precompute a single Gaussian pairwise kernel to reuse for all patches
GAUSS_KERNEL = create_pairwise_gaussian(
    sdims=SXY_GAUSS,
    shape=(PATCH_SIZE[0], PATCH_SIZE[1])
)

def refine_crf(image: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Apply DenseCRF to a probability patch to produce a refined binary mask.

    Args:
        image (np.ndarray): BGR image patch of shape (H, W, 3).
        probs (np.ndarray): Softmax probabilities of shape (2, H, W),
            where probs[0] is background and probs[1] is foreground.

    Returns:
        np.ndarray: Refined binary mask of shape (H, W), values in {0,1}.
    """
    H, W = probs.shape[1:]
    unary = unary_from_softmax(probs)  # Convert softmax to unary energy
    d = dcrf.DenseCRF2D(W, H, 2)
    d.setUnaryEnergy(unary)

    # Add Gaussian pairwise potential (spatial only)
    d.addPairwiseEnergy(GAUSS_KERNEL, compat=COMPAT_GAUSS)

    # Add bilateral pairwise potential (spatial + color)
    bilat = create_pairwise_bilateral(
        sdims=SXY_BILAT,
        schan=SCHAN_BILAT,
        img=image[:, :, :3],
        chdim=2
    )
    d.addPairwiseEnergy(bilat, compat=COMPAT_BILAT)

    # Run inference
    Q = d.inference(N_ITERS)
    refined = np.array(Q).reshape((2, H, W))
    # Choose the class with highest probability
    return np.argmax(refined, axis=0)

def split_image_and_mask(
    img: np.ndarray,
    msk: np.ndarray,
    patch_h: int,
    patch_w: int
):
    """Split an image and mask into reflect-padded patches.

    Args:
        img (np.ndarray): BGR image of shape (H, W, 3).
        msk (np.ndarray): Float32 mask of shape (H, W), values ∈ [0,1].
        patch_h (int): Patch height in pixels.
        patch_w (int): Patch width in pixels.

    Returns:
        patches (list of tuples): Each tuple is (i, j, patch_img, patch_msk).
        orig_size (tuple): Original image dimensions (H, W).
        grid_size (tuple): Number of patches along height and width (nH, nW).
    """
    H, W = img.shape[:2]
    nH = math.ceil(H / patch_h)
    nW = math.ceil(W / patch_w)
    pad_h = nH * patch_h - H
    pad_w = nW * patch_w - W

    # Reflect padding to avoid border artifacts
    img_pad = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    msk_pad = cv2.copyMakeBorder(msk, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

    patches = []
    for i in range(nH):
        for j in range(nW):
            y, x = i * patch_h, j * patch_w
            p_img = img_pad[y:y+patch_h, x:x+patch_w]
            p_msk = msk_pad[y:y+patch_h, x:x+patch_w]
            patches.append((i, j, p_img, p_msk))
    return patches, (H, W), (nH, nW)

def stitch_patches(
    patches_refined: list,
    orig_size: tuple,
    grid_size: tuple,
    patch_h: int,
    patch_w: int
) -> np.ndarray:
    """Reconstruct the full mask from a list of refined patches.

    Args:
        patches_refined (list): Tuples (i, j, ref_patch) where ref_patch is (H, W) uint8.
        orig_size (tuple): Original dimensions (H, W).
        grid_size (tuple): Number of patches (nH, nW).
        patch_h (int): Patch height.
        patch_w (int): Patch width.

    Returns:
        np.ndarray: Full assembled mask, cropped to original size, values ∈ [0,255].
    """
    H, W = orig_size
    nH, nW = grid_size
    full = np.zeros((nH * patch_h, nW * patch_w), dtype=np.uint8)
    for i, j, ref in patches_refined:
        y, x = i * patch_h, j * patch_w
        full[y:y+patch_h, x:x+patch_w] = ref
    return full[:H, :W]

def process_pair(pair: tuple) -> None:
    """Process one image/mask pair: CRF-refine and save the result.

    Args:
        pair (tuple): (img_path (str), mask_path (str)).

    Raises:
        FileNotFoundError: if either image or mask file cannot be read.
    """
    img_path, mask_path = pair
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or msk is None:
        raise FileNotFoundError(f"Failed to read {img_path} or {mask_path}")
    msk = msk.astype(np.float32) / 255.0

    patch_h, patch_w = PATCH_SIZE
    patches, orig_size, grid = split_image_and_mask(img, msk, patch_h, patch_w)

    refined_list = []
    for i, j, p_img, p_msk in patches:
        # Stack background and foreground probabilities
        probs = np.stack([1 - p_msk, p_msk], axis=0).astype(np.float32)
        ref_patch = refine_crf(p_img, probs)
        # Convert binary 0/1 to 0/255 for saving
        refined_list.append((i, j, (ref_patch * 255).astype(np.uint8)))

    full_refined = stitch_patches(refined_list, orig_size, grid, patch_h, patch_w)
    out_path = os.path.join(OUT_DIR, os.path.basename(mask_path))
    cv2.imwrite(out_path, full_refined)

def main() -> None:
    """Entry point: prepare file pairs and run parallel refinement."""
    img_paths  = sorted(glob(os.path.join(IMGS_DIR, '*' + EXT_IMG)))
    mask_paths = sorted(glob(os.path.join(PREDS_DIR, '*' + EXT_MASK)))
    if len(img_paths) != len(mask_paths):
        raise ValueError("Number of images and masks differ")

    pairs = list(zip(img_paths, mask_paths))
    with Pool(processes=N_WORKERS) as pool:
        for _ in tqdm(
            pool.imap_unordered(process_pair, pairs),
            total=len(pairs),
            desc="Refining masks in parallel"
        ):
            pass

    print("Done. Refined masks saved to:", OUT_DIR)

if __name__ == '__main__':
    main()
