"""
Script for training and validating a semantic segmentation model using U-Net
with EfficientNet-B6 backbone. Applies data augmentation during training and
evaluates IoU and accuracy metrics. Includes early stopping, logging, and weight saving.
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

# Directories with data – adjust as needed
train_images_dir      = "/home/rodrigo/JoseBras/dataset/train/images"
train_annotations_dir = "/home/rodrigo/JoseBras/dataset/train/masks"
val_images_dir        = "/home/rodrigo/JoseBras/dataset/val/images"
val_annotations_dir   = "/home/rodrigo/JoseBras/dataset/val/masks"
test_images_dir       = "/home/rodrigo/JoseBras/dataset/test/images"
test_annotations_dir  = "/home/rodrigo/JoseBras/dataset/test/masks"

# Hyperparameters
BACKBONE    = 'efficientnet-b6'
IMG_SIZE    = (512, 512)
BATCH_SIZE  = 6
NUM_CLASSES = 1

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def train_augmentations(image: Image.Image, mask: Image.Image):
    """Apply synchronized data augmentation to image/mask pairs.

    Performs random horizontal flip, vertical flip and random rotation in [-30°, +30°].

    Args:
        image (PIL.Image): RGB input image.
        mask (PIL.Image): Grayscale mask image.

    Returns:
        tuple[PIL.Image, PIL.Image]: Transformed image and mask.
    """
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask  = TF.hflip(mask)
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask  = TF.vflip(mask)
    angle = random.uniform(-30, 30)
    image = TF.rotate(image, angle, interpolation=Image.BILINEAR)
    mask  = TF.rotate(mask,  angle, interpolation=Image.NEAREST)
    return image, mask

# Transforms without augmentation for validation/testing
image_transforms = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
mask_transforms = T.Compose([
    T.Resize(IMG_SIZE, interpolation=Image.NEAREST),
    T.ToTensor(),
])

class SegmentationDataset(Dataset):
    """Semantic segmentation dataset that loads image/mask pairs and applies optional augmentations.

    Args:
        images_dir (str): Path to directory with input images.
        masks_dir (str): Path to directory with corresponding masks.
        image_transform (callable, optional): Transform applied to images.
        mask_transform (callable, optional): Transform applied to masks.
        augmentations (callable, optional): Function to augment image/mask pairs.
    """

    def __init__(self,
                 images_dir: str,
                 masks_dir: str,
                 image_transform=None,
                 mask_transform=None,
                 augmentations=None):
        self.image_transform = image_transform
        self.mask_transform  = mask_transform
        self.augmentations   = augmentations

        # Get sorted lists of file paths
        self.image_paths = sorted([
            os.path.join(root, f)
            for root, _, files in os.walk(images_dir)
            for f in files
        ])
        self.mask_paths = sorted([
            os.path.join(root, f)
            for root, _, files in os.walk(masks_dir)
            for f in files
        ])
        assert len(self.image_paths) == len(self.mask_paths), \
            "Different number of images and masks!"

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """Load and return one sample (image and mask tensors).

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Transformed image and mask.
        """
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask  = Image.open(self.mask_paths[idx]).convert("L")

        # Apply augmentations if provided
        if self.augmentations:
            image, mask = self.augmentations(image, mask)

        # Apply transforms
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

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=4)

# Model instantiation
model = smp.Unet(
    encoder_name=BACKBONE,
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
    activation=None
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def iou_score(pred: torch.Tensor,
              target: torch.Tensor,
              threshold: float = 0.5,
              eps: float = 1e-6) -> float:
    """Compute Intersection-over-Union (IoU) metric.

    Args:
        pred (torch.Tensor): Model output logits of shape (B, C, H, W).
        target (torch.Tensor): Ground truth masks of same shape.
        threshold (float): Probability threshold for binarization.
        eps (float): Small epsilon to avoid division by zero.

    Returns:
        float: Mean IoU over the batch.
    """
    pred_bin = torch.sigmoid(pred) > threshold
    target_bin = target > threshold
    intersection = (pred_bin & target_bin).sum(dim=(1,2,3))
    union = pred_bin.sum(dim=(1,2,3)) + target_bin.sum(dim=(1,2,3)) - intersection
    return ((intersection + eps) / (union + eps)).mean().item()

def accuracy_score(outputs: torch.Tensor,
                   masks: torch.Tensor,
                   threshold: float = 0.5) -> float:
    """Compute pixel-wise accuracy.

    Args:
        outputs (torch.Tensor): Model output logits (B, C, H, W).
        masks (torch.Tensor): Ground truth masks (B, C, H, W).
        threshold (float): Threshold for binarization.

    Returns:
        float: Mean accuracy over the batch.
    """
    preds = torch.sigmoid(outputs) > threshold
    true_masks = masks > threshold
    return (preds == true_masks).float().mean().item()

def train_one_epoch(loader: DataLoader,
                    model: torch.nn.Module,
                    criterion,
                    optimizer=None) -> tuple[float, float, float]:
    """Run one epoch of training or validation.

    If `optimizer` is None, runs in evaluation mode; otherwise performs training.

    Args:
        loader (DataLoader): DataLoader for train or validation set.
        model (torch.nn.Module): Model to train/validate.
        criterion: Loss function.
        optimizer (Optimizer, optional): Optimizer for weight updates.

    Returns:
        tuple[float, float, float]:
            Average loss, average IoU, average accuracy for the epoch.
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

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_iou  += iou_score(outputs, masks) * batch_size
        running_acc  += accuracy_score(outputs, masks) * batch_size

    total_samples = len(loader.dataset)
    return (running_loss / total_samples,
            running_iou  / total_samples,
            running_acc  / total_samples)

# Count parameters
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# Training hyperparameters and early stopping setup
num_epochs          = 100
early_stop_patience = 20
early_stop_counter  = 0
best_val_loss       = float('inf')

# Initialize log file
with open("model_efficientnet-b6.txt", "w") as f:
    f.write("Training log:\n")

for epoch in range(1, num_epochs + 1):
    # Train and validate one epoch
    train_loss, train_iou, train_acc = train_one_epoch(
        train_loader, model, criterion, optimizer
    )
    val_loss, val_iou, val_acc = train_one_epoch(
        val_loader, model, criterion
    )

    # Log and print metrics
    log_line = (f"Epoch {epoch}/{num_epochs}: "
                f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                f"Train IoU={train_iou:.4f}, Val IoU={val_iou:.4f}, "
                f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}\n")
    print(log_line.strip())
    with open("model_efficientnet-b6.txt", "a") as f:
        f.write(log_line)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), "model_efficientnet-b6_best.pth")
    else:
        early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping after {early_stop_patience} epochs without improvement.")
            break

    torch.cuda.empty_cache()

# Save final model
torch.save(model.state_dict(), "model_efficientnet-b6.pth")
print("Final model saved as 'model_efficientnet-b6.pth'")
