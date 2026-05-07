"""BBox loss functions for camouflaged object detection.

Composite loss combining coordinate regression (SmoothL1) with
box-level overlap supervision (GIoU) — tuned for small, subtle targets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def box_giou(pred, gt):
    """Generalized IoU for boxes in (x1, y1, x2, y2) format.

    GIoU = IoU - |C \ (A ∪ B)| / |C|  where C is the smallest enclosing box.
    Ranges from -1 (worst) to 1 (best).

    Args:
        pred: (..., 4) tensor
        gt:   (..., 4) tensor
    Returns:
        giou: (...,) tensor in [-1, 1]
    """
    # Intersection
    ix1 = torch.max(pred[..., 0], gt[..., 0])
    iy1 = torch.max(pred[..., 1], gt[..., 1])
    ix2 = torch.min(pred[..., 2], gt[..., 2])
    iy2 = torch.min(pred[..., 3], gt[..., 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    # Areas
    area_pred = (pred[..., 2] - pred[..., 0]).clamp(min=0) * (
        pred[..., 3] - pred[..., 1]
    ).clamp(min=0)
    area_gt = (gt[..., 2] - gt[..., 0]).clamp(min=0) * (
        gt[..., 3] - gt[..., 1]
    ).clamp(min=0)
    union = area_pred + area_gt - inter
    iou = inter / (union + 1e-6)

    # Smallest enclosing box
    ex1 = torch.min(pred[..., 0], gt[..., 0])
    ey1 = torch.min(pred[..., 1], gt[..., 1])
    ex2 = torch.max(pred[..., 2], gt[..., 2])
    ey2 = torch.max(pred[..., 3], gt[..., 3])
    area_enclose = (ex2 - ex1).clamp(min=0) * (ey2 - ey1).clamp(min=0)

    giou = iou - (area_enclose - union) / (area_enclose + 1e-6)
    return giou


def box_diou(pred, gt):
    """Distance IoU for boxes in (x1, y1, x2, y2) format.

    DIoU = IoU - ρ²(b_pred, b_gt) / c²  where ρ is the center distance
    and c is the diagonal of the smallest enclosing box.
    Ranges from -1 (worst) to 1 (best).

    Args:
        pred: (..., 4) tensor
        gt:   (..., 4) tensor
    Returns:
        diou: (...,) tensor in [-1, 1]
    """
    # Intersection & IoU (as above)
    ix1 = torch.max(pred[..., 0], gt[..., 0])
    iy1 = torch.max(pred[..., 1], gt[..., 1])
    ix2 = torch.min(pred[..., 2], gt[..., 2])
    iy2 = torch.min(pred[..., 3], gt[..., 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    area_pred = (pred[..., 2] - pred[..., 0]).clamp(min=0) * (
        pred[..., 3] - pred[..., 1]
    ).clamp(min=0)
    area_gt = (gt[..., 2] - gt[..., 0]).clamp(min=0) * (
        gt[..., 3] - gt[..., 1]
    ).clamp(min=0)
    union = area_pred + area_gt - inter
    iou = inter / (union + 1e-6)

    # Center coords
    cx_pred = (pred[..., 0] + pred[..., 2]) / 2.0
    cy_pred = (pred[..., 1] + pred[..., 3]) / 2.0
    cx_gt = (gt[..., 0] + gt[..., 2]) / 2.0
    cy_gt = (gt[..., 1] + gt[..., 3]) / 2.0
    rho2 = (cx_pred - cx_gt) ** 2 + (cy_pred - cy_gt) ** 2

    # Enclosing box diagonal
    ex1 = torch.min(pred[..., 0], gt[..., 0])
    ey1 = torch.min(pred[..., 1], gt[..., 1])
    ex2 = torch.max(pred[..., 2], gt[..., 2])
    ey2 = torch.max(pred[..., 3], gt[..., 3])
    c2 = (ex2 - ex1).clamp(min=0) ** 2 + (ey2 - ey1).clamp(min=0) ** 2

    diou = iou - rho2 / (c2 + 1e-6)
    return diou


class BBoxLoss(nn.Module):
    """Composite BBox loss for camouflaged object detection.

    L = λ_smooth * SmoothL1(pred, gt) + λ_iou * (1 - GIoU(pred, gt))

    SmoothL1 provides per-coordinate regression signal; GIoU provides
    box-level overlap supervision that is scale-invariant and handles
    non-overlapping predictions gracefully.
    """

    def __init__(self, smooth_l1_weight=1.0, giou_weight=1.0, use_diou=False):
        super().__init__()
        self.smooth_l1_weight = smooth_l1_weight
        self.giou_weight = giou_weight
        self.use_diou = use_diou

    def forward(self, pred, gt):
        """Compute composite loss.

        Args:
            pred: (B, T, 4) predicted BBoxes in [0, 1]
            gt:   (B, T, 4) ground-truth BBoxes in [0, 1]

        Returns:
            dict with keys: loss (total), smooth_l1, giou, mean_iou
        """
        l1 = F.smooth_l1_loss(pred, gt, beta=0.1)

        if self.use_diou:
            iou_metric = box_diou(pred, gt)
        else:
            iou_metric = box_giou(pred, gt)
        # GIoU/DIoU loss: 1 - GIoU (ranges from 0 to 2, lower is better)
        iou_loss = (1.0 - iou_metric).mean()

        total = self.smooth_l1_weight * l1 + self.giou_weight * iou_loss

        # Monitoring: actual IoU (not GIoU) for human readability
        with torch.no_grad():
            actual_iou = _box_iou(pred, gt).mean()

        return {
            "loss": total,
            "smooth_l1": l1.detach(),
            "giou_loss": iou_loss.detach(),
            "mean_iou": actual_iou,
        }


def _box_iou(pred, gt):
    """Plain IoU (for monitoring only)."""
    ix1 = torch.max(pred[..., 0], gt[..., 0])
    iy1 = torch.max(pred[..., 1], gt[..., 1])
    ix2 = torch.min(pred[..., 2], gt[..., 2])
    iy2 = torch.min(pred[..., 3], gt[..., 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    area_pred = (pred[..., 2] - pred[..., 0]).clamp(min=0) * (
        pred[..., 3] - pred[..., 1]
    ).clamp(min=0)
    area_gt = (gt[..., 2] - gt[..., 0]).clamp(min=0) * (
        gt[..., 3] - gt[..., 1]
    ).clamp(min=0)
    union = area_pred + area_gt - inter
    return inter / (union + 1e-6)
