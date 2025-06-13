"""
Script to train and validate a U-Net semantic segmentation model
with the se_resnet101 backbone. Applies data augmentation only during training,
evaluates IoU and pixel-wise accuracy, and implements early stopping.
"""

import os
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import segmentation_models_pytorch as smp
from tqdm import tqdm

# Directories for dataset splits – adjust these to your environment
train_images_dir      = "/home/rodrigo/JoseBras/dataset/train/images"
train_annotations_dir = "/home/rodrigo/JoseBras/dataset/train/masks"
val_images_dir        = "/home/rodrigo/JoseBras/dataset/val/images"
val_annotations_dir   = "/home/rodrigo/JoseBras/dataset/val/masks"
test_images_dir       = "/home/rodrigo/JoseBras/dataset/test/images"
test_annotations_dir  = "/home/rodrigo/JoseBras/dataset/test/masks"

# Main hyperparameters
BACKBONE    = 'se_resnet101'
img_size    = (512, 512)
batch_size  = 6
num_classes = 1

# ImageNet normalization parameters
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

def train_augmentations(
    image: Image.Image,
    mask: Image.Image
) -> tuple[Image.Image, Image.Image]:
    """Apply synchronized random flips and rotation to an image and its mask.

    Args:
        image (PIL.Image): RGB input image.
        mask  (PIL.Image): Grayscale segmentation mask.

    Returns:
        Tuple[PIL.Image, PIL.Image]: Augmented image and mask.
    """
    # Random horizontal flip
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask  = TF.hflip(mask)
    # Random vertical flip
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask  = TF.vflip(mask)
    # Random rotation between -30° and +30°
    angle = random.uniform(-30, 30)
    image = TF.rotate(image, angle, interpolation=Image.BILINEAR)
    mask  = TF.rotate(mask,  angle, interpolation=Image.NEAREST)
    return image, mask

# Transforms for validation and test (no augmentation)
image_transforms = T.Compose([
    T.Resize(img_size),
    T.ToTensor(),
    T.Normalize(mean=imagenet_mean, std=imagenet_std),
])
mask_transforms = T.Compose([
    T.Resize(img_size, interpolation=Image.NEAREST),
    T.ToTensor(),
])

class SegmentationDataset(Dataset):
    """Dataset for paired image/mask semantic segmentation."""

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        image_transform=None,
        mask_transform=None,
        augmentations=None
    ):
        """
        Args:
            images_dir (str): Directory containing RGB image files.
            masks_dir  (str): Directory containing grayscale mask files.
            image_transform: torchvision transform for images.
            mask_transform: torchvision transform for masks.
            augmentations: function to augment image/mask pairs (training only).
        """
        self.image_transform = image_transform
        self.mask_transform  = mask_transform
        self.augmentations   = augmentations

        # Collect and sort paths
        self.image_paths = []
        self.mask_paths  = []
        for root, _, files in os.walk(images_dir):
            for fname in sorted(files):
                self.image_paths.append(os.path.join(root, fname))
        for root, _, files in os.walk(masks_dir):
            for fname in sorted(files):
                self.mask_paths.append(os.path.join(root, fname))

        assert len(self.image_paths) == len(self.mask_paths), \
            "Number of images and masks must be equal."

    def __len__(self) -> int:
        """Return the number of image/mask pairs."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load an image/mask pair, apply augmentations (if any) and transforms.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (image_tensor, mask_tensor).
        """
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask  = Image.open(self.mask_paths[idx]).convert("L")

        if self.augmentations:
            image, mask = self.augmentations(image, mask)
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

# Instantiate datasets
train_dataset = SegmentationDataset(
    train_images_dir, train_annotations_dir,
    image_transform=image_transforms,
    mask_transform=mask_transforms,
    augmentations=train_augmentations
)
val_dataset = SegmentationDataset(
    val_images_dir, val_annotations_dir,
    image_transform=image_transforms,
    mask_transform=mask_transforms
)
test_dataset = SegmentationDataset(
    test_images_dir, test_annotations_dir,
    image_transform=image_transforms,
    mask_transform=mask_transforms
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                          shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                          shuffle=False, num_workers=4)

# Build U-Net model with se_resnet101 backbone
model = smp.Unet(
    encoder_name=BACKBONE,
    encoder_weights="imagenet",
    in_channels=3,
    classes=num_classes,
    activation=None
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # Move model to GPU if available

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6
) -> float:
    """Compute batch-average Intersection-over-Union (IoU).

    Args:
        pred (Tensor): Model logits of shape (B,1,H,W).
        target (Tensor): Ground-truth masks of same shape.
        threshold (float): Binarization cutoff.
        eps (float): Small constant to avoid zero-division.

    Returns:
        float: Mean IoU over the batch.
    """
    pred_bin   = torch.sigmoid(pred) > threshold
    target_bin = target > threshold
    intersection = (pred_bin & target_bin).sum(dim=(1,2,3))
    union        = (pred_bin.sum(dim=(1,2,3)) +
                    target_bin.sum(dim=(1,2,3)) -
                    intersection)
    return ((intersection + eps) / (union + eps)).mean().item()

def accuracy_score(
    outputs: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """Compute batch-average pixel-wise accuracy.

    Args:
        outputs (Tensor): Model logits.
        masks (Tensor): Ground-truth masks.
        threshold (float): Binarization cutoff.

    Returns:
        float: Mean accuracy over the batch.
    """
    preds      = torch.sigmoid(outputs) > threshold
    true_masks = masks > threshold
    return (preds == true_masks).float().mean().item()

def train_one_epoch(
    loader: DataLoader,
    model: torch.nn.Module,
    criterion,
    optimizer=None
) -> tuple[float, float, float]:
    """Run one epoch of training or validation.

    If `optimizer` is provided, this performs a training epoch; otherwise, it runs validation.

    Args:
        loader (DataLoader): Yields (images, masks) batches.
        model (nn.Module): The segmentation model.
        criterion: Loss function.
        optimizer: Optimizer instance or None for validation.

    Returns:
        Tuple[float, float, float]: (average_loss, average_iou, average_accuracy).
    """
    running_loss = running_iou = running_acc = 0.0
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    for images, masks in tqdm(loader,
                              desc="Training" if is_train else "Validation",
                              leave=False):
        images, masks = images.to(device), masks.to(device)
        if is_train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)
        if is_train:
            loss.backward()
            optimizer.step()

        bs = images.size(0)
        running_loss += loss.item() * bs
        running_iou  += iou_score(outputs, masks) * bs
        running_acc  += accuracy_score(outputs, masks) * bs

    total = len(loader.dataset)
    return running_loss/total, running_iou/total, running_acc/total

# Count model parameters
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# Training settings and early stopping
num_epochs          = 100
early_stop_patience = 20
early_stop_counter  = 0
best_val_loss       = float('inf')

# Initialize training log file
with open("model_se_resnet101.txt", "w") as f:
    f.write("Training log:\n")

# Main training/validation loop
for epoch in range(1, num_epochs + 1):
    train_loss, train_iou, train_acc = train_one_epoch(
        train_loader, model, criterion, optimizer
    )
    val_loss, val_iou, val_acc = train_one_epoch(
        val_loader, model, criterion
    )

    log_line = (
        f"Epoch {epoch}/{num_epochs}: "
        f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
        f"Train IoU={train_iou:.4f}, Val IoU={val_iou:.4f}, "
        f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}\n"
    )
    print(log_line.strip())
    with open("model_se_resnet101.txt", "a") as f:
        f.write(log_line)

    if val_loss < best_val_loss:
        best_val_loss      = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), "model_se_resnet101_best.pth")
    else:
        early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping after {early_stop_patience} epochs without improvement.")
            break

    torch.cuda.empty_cache()

# Save final model weights
torch.save(model.state_dict(), "model_se_resnet101.pth")
print("Final model saved as 'model_se_resnet101.pth'")
