# -*- coding: utf-8 -*-
"""
Module for semantic segmentation inference on large images using overlapping patches.

This module loads a U-Net model with a specified backbone and applies it to
high-resolution images by dividing them into overlapping patches, aggregating
probabilities, and producing binary masks using a fixed threshold.
"""

import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import segmentation_models_pytorch as smp
import concurrent.futures

# --- Global Parameters ---
PATCH_SIZE = 512         # patch dimension in pixels
THRESHOLD = 0.5          # probability threshold for binary mask

# ImageNet normalization parameters (mean and std)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Transform pipeline: resize → tensor → normalization
image_transforms = T.Compose([
    T.Resize((PATCH_SIZE, PATCH_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def load_model(model_path: str, backbone: str, device: torch.device) -> torch.nn.Module:
    """Load a pretrained U-Net model and its weights from a state_dict file.

    Args:
        model_path (str): Path to the .pth file containing model.state_dict().
        backbone (str): Name of the encoder architecture compatible with SMP.
        device (torch.device): Device for model allocation (CPU or GPU).

    Returns:
        torch.nn.Module: Model in evaluation mode (`model.eval()`), ready for inference.

    Raises:
        FileNotFoundError: If `model_path` does not exist.
        RuntimeError: If loading state_dict fails or shapes mismatch.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Initialize U-Net model
    model = smp.Unet(
        encoder_name=backbone,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    )
    # Load weights and move to device
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"Model loaded on {device} (backbone={backbone}, classes=1)")
    return model

def predict_large_image(
    model: torch.nn.Module,
    image_path: Path,
    transform: T.Compose,
    device: torch.device,
    patch_size: int = PATCH_SIZE
) -> (np.ndarray, np.ndarray):
    """Perform inference on a large image by processing overlapping patches.

    The image is divided into patches with 50% overlap, each passed through
    the model individually. Predicted probabilities are aggregated and
    thresholded to produce a binary mask.

    Args:
        model (torch.nn.Module): Loaded segmentation model.
        image_path (Path): Path to input image file.
        transform (T.Compose): Transform pipeline for each patch.
        device (torch.device): Device for inference.
        patch_size (int): Size of each square patch (default=512).

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            img_rgb: Original image as an RGB numpy array.
            mask: Binary mask (dtype uint8, values 0 or 255).

    Raises:
         If the image cannot be read, returns (None, None).
    """
    # Read image in BGR then convert to RGB
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        # Failed to load image
        print(f"Failed to read image at {image_path}")
        return None, None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    # Accumulators for probabilities and overlap counts
    accum = np.zeros((h, w), dtype=np.float32)
    count = np.zeros((h, w), dtype=np.float32)
    step = patch_size // 2  # 50% overlap

    # Iterate over patches
    for i in range(0, h, step):
        for j in range(0, w, step):
            i_end = min(i + patch_size, h)
            j_end = min(j + patch_size, w)
            patch = img_rgb[i:i_end, j:j_end]
            ph, pw = patch.shape[:2]

            # Apply transform and add batch dimension
            patch_pil = Image.fromarray(patch)
            inp = transform(patch_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(inp)                    # shape: 1×1×H×W
                prob = torch.sigmoid(out)[0, 0]     # extract probabilities
                prob = prob.cpu().numpy()

            # Resize edge patches back to original spatial size
            if ph != patch_size or pw != patch_size:
                prob = cv2.resize(prob, (pw, ph), interpolation=cv2.INTER_NEAREST)

            # Aggregate probabilities and track count
            accum[i:i+ph, j:j+pw] += prob
            count[i:i+ph, j:j+pw] += 1

    # Compute average probability and apply threshold
    avg = accum / count
    mask = (avg > THRESHOLD).astype(np.uint8) * 255
    return img_rgb, mask

def process_image_file(
    model: torch.nn.Module,
    img_file: Path,
    output_path: Path,
    transform: T.Compose,
    device: torch.device
) -> None:
    """Process a single image file: perform inference and save the mask.

    Args:
        model (torch.nn.Module): Loaded segmentation model.
        img_file (Path): Path to the input image.
        output_path (Path): Directory to save the mask file.
        transform (T.Compose): Transform pipeline for patches.
        device (torch.device): Device for inference.

    Returns:
        None

    Side Effects:
        Writes a PNG mask file named `<original_stem>.png` to `output_path`.
    """
    img_rgb, mask = predict_large_image(model, img_file, transform, device)
    if mask is not None:
        out_file = output_path / f"{img_file.stem}.png"
        cv2.imwrite(str(out_file), mask)  # Save binary mask
        print(f"{img_file.name} → mask saved as {out_file.name}")
    else:
        # Image read error; skip
        print(f"Ignored {img_file.name}")

def process_folder(
    input_dir: str,
    output_dir: str,
    model: torch.nn.Module,
    transform: T.Compose,
    device: torch.device
) -> None:
    """Recursively process all supported image files in a directory.

    Args:
        input_dir (str): Path to directory containing images.
        output_dir (str): Directory to save generated masks.
        model (torch.nn.Module): Loaded segmentation model.
        transform (T.Compose): Transform pipeline for patches.
        device (torch.device): Device for inference.

    Returns:
        None

    Side Effects:
        Creates `output_dir` if it doesn't exist and writes mask files.
    """
    inp = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Filter supported image extensions
    imgs = [
        f for f in inp.iterdir()
        if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    ]
    print(f"{len(imgs)} images found in {input_dir}")

    # Parallel processing using threads (I/O bound for image reading/writing)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_image_file, model, f, out, transform, device)
            for f in imgs
        ]
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    # Execution-specific parameters
    model_path = "/home/rodrigo/JoseBras/TreinosFinais/resnet152/model_se_resnet101_aug.pth"
    input_dir = "/home/rodrigo/JoseBras/Imagens"
    output_dir = "/home/rodrigo/JoseBras/TreinosFinais/Predicts_Aug/se_resnet101"
    backbone = 'se_resnet101'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_path, backbone, device)
    process_folder(input_dir, output_dir, model, image_transforms, device)
