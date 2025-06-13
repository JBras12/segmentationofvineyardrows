#!/usr/bin/env python3

from torch import einsum
import torch.nn.functional as F

from .utils import simplex, sset


class CrossEntropy:
    """Cross-entropy loss for “weak” one-hot masks, focusing on selected classes."""

    def __init__(self, idk: list[int], **kwargs):
        """
        Initialize the CrossEntropy loss.

        Args:
            idk (List[int]): Indices of the classes to include in the loss.
            **kwargs: Optional keyword arguments for logging or metadata.
        """
        self.idk = idk
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self,
                 pred_softmax: torch.Tensor,
                 weak_target: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean cross-entropy over the specified classes.

        This performs:
          - Validation of shapes and value ranges.
          - Extraction of log-probabilities for selected classes.
          - Element-wise multiplication with the weak one-hot mask.
          - Normalization by the total mask weight.

        Args:
            pred_softmax (Tensor[B, K, H, W]):
                Model predictions after softmax (sum to 1 over K).
            weak_target (Tensor[B, K, H, W]):
                Weak one-hot or pseudo one-hot mask (values 0 or 1).

        Returns:
            Tensor: Scalar loss normalized by the sum of mask weights.

        Raises:
            AssertionError:
                If `pred_softmax` and `weak_target` shapes differ,
                if `pred_softmax` is not a valid probability simplex,
                or if `weak_target` contains values other than 0 or 1.
        """
        # Validate shapes and values
        assert pred_softmax.shape == weak_target.shape, "Incompatible dimensions"
        assert simplex(pred_softmax), "pred_softmax is not a valid simplex"
        assert sset(weak_target, [0, 1]), "weak_target must contain only 0 or 1"

        # Select log-probs and mask for the classes of interest
        log_p = (pred_softmax[:, self.idk, ...] + 1e-10).log()
        mask = weak_target[:, self.idk, ...].float()

        # Compute summed cross-entropy: -sum(mask * log_p)
        loss = -einsum("bkwh,bkwh->", mask, log_p)
        # Normalize by total mask weight
        loss /= mask.sum() + 1e-10
        return loss


class PartialCrossEntropy(CrossEntropy):
    """Partial cross-entropy that always computes loss for class index 1."""

    def __init__(self, **kwargs):
        """
        Initialize to compute cross-entropy only on class 1.

        Args:
            **kwargs: Optional keyword arguments for logging.
        """
        super().__init__(idk=[1], **kwargs)


class NaiveSizeLoss:
    """Quadratic penalty for predicted class size falling outside given bounds.

    penalty = 0,                            if lower ≤ size ≤ upper  
            = (size - upper)²,              if size > upper  
            = (lower - size)²,              if size < lower  
    """

    def __init__(self, **kwargs):
        """
        Initialize the size loss.

        Args:
            **kwargs: Optional keyword arguments for logging.
        """
        self.idk = [1]  # Default to class index 1
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self,
                 pred_softmax: torch.Tensor,
                 bounds: torch.Tensor) -> torch.Tensor:
        """
        Compute size-based penalty for each sample in the batch.

        Args:
            pred_softmax (Tensor[B, K, H, W]):
                Model predictions after softmax.
            bounds (Tensor[B, K, 2]):
                Lower and upper size bounds per class for each batch.

        Returns:
            Tensor[B, 1]: Penalty per sample, normalized by patch area and scaled by 1/100.

        Raises:
            AssertionError:
                If `pred_softmax` is not a valid simplex or `bounds` has unexpected shape.
        """
        assert simplex(pred_softmax), "pred_softmax is not a valid simplex"
        B, K, H, W = pred_softmax.shape
        assert bounds.shape == (B, K, 2), "bounds has unexpected shape"

        # Predicted size per class = sum of probabilities over H×W
        pred_size = einsum("bkwh->bk", pred_softmax)[:, self.idk]  # Shape [B, len(idk)]

        # Extract corresponding lower/upper bounds
        upper = bounds[:, self.idk, 1]
        lower = bounds[:, self.idk, 0]
        assert (upper >= 0).all() and (lower >= 0).all()

        # Quadratic penalty for exceeding upper or falling below lower
        loss = F.relu(pred_size - upper) ** 2 + F.relu(lower - pred_size) ** 2
        # Normalize by patch area
        loss /= (H * W)
        # Additional scaling factor
        return loss / 100
