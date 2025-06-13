"""
Module for training and validating a semantic segmentation U-Net model with ResNet-50 backbone.

Applies synchronized data augmentation during training, computes IoU and accuracy metrics,
implements early stopping, logs progress to a text file, and saves both the best model
and the final model weights.
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

# Directories – adjust as needed
train_images_dir      = "/home/rodrigo/JoseBras/dataset/train/images"
train_annotations_dir = "/home/rodrigo/JoseBras/dataset/train/masks"
val_images_dir        = "/home/rodrigo/JoseBras/dataset/val/images"
val_annotations_dir   = "/home/rodrigo/JoseBras/dataset/val/masks"
test_images_dir       = "/home/rodrigo/JoseBras/dataset/test/images"
test_annotations_dir  = "/home/rodrigo/JoseBras/dataset/test/masks"

# Hyperparameters
BACKBONE    = 'resnet50'
IMG_SIZE    = (512, 512)
BATCH_SIZE  = 6
NUM_CLASSES = 1

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def train_augmentations(
    image: Image.Image,
    mask: Image.Image
) -> tuple[Image.Image, Image.Image]:
    """Apply synchronized random flips and rotation to an image and its mask.

    Args:
        image (PIL.Image.Image): Input RGB image.
        mask  (PIL.Image.Image): Input grayscale mask.

    Returns:
        Tuple[PIL.Image.Image, PIL.Image.Image]: Transformed image and mask.
    """
    # Horizontal flip with 50% probability
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask  = TF.hflip(mask)
    # Vertical flip with 50% probability
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask  = TF.vflip(mask)
    # Random rotation between -30° and +30°
    angle = random.uniform(-30, 30)
    image = TF.rotate(image, angle, interpolation=Image.BILINEAR)
    mask  = TF.rotate(mask,  angle, interpolation=Image.NEAREST)
    return image, mask

# Transforms without augmentation for validation/test
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
    """Dataset for semantic segmentation that loads image–mask pairs synchronously.

    Args:
        images_dir (str): Directory containing input images.
        masks_dir (str): Directory containing corresponding masks.
        image_transform (callable, optional): Transform applied to each image.
        mask_transform (callable, optional): Transform applied to each mask.
        augmentations (callable, optional): Synchronous augmentation function.
    """
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        image_transform=None,
        mask_transform=None,
        augmentations=None
    ):
        self.image_transform = image_transform
        self.mask_transform  = mask_transform
        self.augmentations   = augmentations

        # Collect ordered lists of file paths
        self.image_paths = []
        self.mask_paths  = []
        for root, _, files in os.walk(images_dir):
            for fname in sorted(files):
                self.image_paths.append(os.path.join(root, fname))
        for root, _, files in os.walk(masks_dir):
            for fname in sorted(files):
                self.mask_paths.append(os.path.join(root, fname))

        assert len(self.image_paths) == len(self.mask_paths), \
            "Different number of images and masks!"

    def __len__(self) -> int:
        """Total number of samples in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and return a transformed sample (image and mask).

        Args:
            idx (int): Sample index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tensors for image and mask.
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

# Instantiate datasets and loaders
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

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=4
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=4
)

# Model, device, loss and optimizer
model = smp.Unet(
    encoder_name=BACKBONE,
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES,
    activation=None
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6
) -> float:
    """Compute the mean Intersection-over-Union (IoU) over a batch.

    Args:
        pred (torch.Tensor): Model logits, shape (B, C, H, W).
        target (torch.Tensor): Ground truth masks, same shape as `pred`.
        threshold (float): Threshold for binarization.
        eps (float): Small epsilon to avoid division by zero.

    Returns:
        float: Mean IoU over the batch.
    """
    pred_bin   = torch.sigmoid(pred) > threshold
    target_bin = target > threshold
    inter = (pred_bin & target_bin).sum(dim=(1,2,3))
    union = pred_bin.sum(dim=(1,2,3)) + target_bin.sum(dim=(1,2,3)) - inter
    return ((inter + eps) / (union + eps)).mean().item()

def accuracy_score(
    outputs: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5
) -> float:
    """Compute pixel-wise accuracy over a batch.

    Args:
        outputs (torch.Tensor): Model logits, shape (B, C, H, W).
        masks (torch.Tensor): Ground truth masks, same shape as `outputs`.
        threshold (float): Threshold for binarization.

    Returns:
        float: Mean accuracy over the batch.
    """
    preds     = torch.sigmoid(outputs) > threshold
    true_masks = masks > threshold
    return (preds == true_masks).float().mean().item()

def train_one_epoch(
    loader: DataLoader,
    model: torch.nn.Module,
    criterion,
    optimizer=None
) -> tuple[float, float, float]:
    """Run one epoch of training or validation.

    If `optimizer` is provided, runs training; otherwise runs validation.

    Args:
        loader (DataLoader): DataLoader for training or validation.
        model (torch.nn.Module): Model to train or evaluate.
        criterion: Loss function.
        optimizer (torch.optim.Optimizer, optional): Optimizer for training.

    Returns:
        Tuple[float, float, float]: (average_loss, average_iou, average_accuracy).
    """
    running_loss = running_iou = running_acc = 0.0
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    for images, masks in tqdm(
        loader,
        desc="Training" if is_train else "Validation",
        leave=False
    ):
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

# Count parameters and set up early stopping
total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

num_epochs          = 100
early_stop_patience = 20
early_stop_counter  = 0
best_val_loss       = float('inf')

with open("model_resnet50.txt", "w") as f:
    f.write("Training log:\n")

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
    with open("model_resnet50.txt", "a") as f:
        f.write(log_line)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), "model_resnet50_best.pth")
    else:
        early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping after {early_stop_patience} epochs without improvement.")
            break

    torch.cuda.empty_cache()

torch.save(model.state_dict(), "model_resnet50.pth")
print("Final model saved as 'model_resnet50.pth'")
