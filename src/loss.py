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

    L = lambda_smooth * SmoothL1(pred, gt) + lambda_iou * (1 - GIoU(pred, gt))
        + lambda_center * |center_pred - center_gt|^2
        + lambda_logwh * (|log(w_pred) - log(w_gt)|^2 + |log(h_pred) - log(h_gt)|^2)

    SmoothL1 provides per-coordinate regression signal; GIoU provides
    box-level overlap supervision that is scale-invariant and handles
    non-overlapping predictions gracefully.
    Center loss provides explicit position supervision for the bbox center.
    Log-WH loss provides relative scale calibration by penalizing errors
    in log(width) and log(height), directly targeting size accuracy.
    Objectness BCE provides foreground/background discrimination.
    """

    def __init__(self, smooth_l1_weight=1.0, giou_weight=1.0, use_diou=False,
                 center_weight=0.0, log_wh_weight=0.0, objectness_weight=0.0):
        super().__init__()
        self.smooth_l1_weight = smooth_l1_weight
        self.giou_weight = giou_weight
        self.use_diou = use_diou
        self.center_weight = center_weight
        self.log_wh_weight = log_wh_weight
        self.objectness_weight = objectness_weight

    def forward(self, pred, gt):
        """Compute composite loss.

        Args:
            pred: (B, T, 4) predicted BBoxes in [0, 1], OR
                  tuple of ((B,T,4) bbox, (B,T,1) objectness) when objectness_head active
            gt:   (B, T, 4) ground-truth BBoxes in [0, 1]

        Returns:
            dict with keys: loss (total), smooth_l1, giou_loss, mean_iou,
                            [+ center_loss, objectness_loss, objectness_acc]
        """
        obj_pred = None
        if isinstance(pred, tuple):
            pred, obj_pred = pred

        l1 = F.smooth_l1_loss(pred, gt, beta=0.1)

        if self.use_diou:
            iou_metric = box_diou(pred, gt)
        else:
            iou_metric = box_giou(pred, gt)
        iou_loss = (1.0 - iou_metric).mean()

        total = self.smooth_l1_weight * l1 + self.giou_weight * iou_loss

        result = {
            "loss": total,
            "smooth_l1": l1.detach(),
            "giou_loss": iou_loss.detach(),
        }

        # Center loss: MSE between predicted and GT bbox centers
        if self.center_weight > 0:
            cx_pred = (pred[..., 0] + pred[..., 2]) / 2.0
            cy_pred = (pred[..., 1] + pred[..., 3]) / 2.0
            cx_gt = (gt[..., 0] + gt[..., 2]) / 2.0
            cy_gt = (gt[..., 1] + gt[..., 3]) / 2.0
            center_loss = F.mse_loss(cx_pred, cx_gt) + F.mse_loss(cy_pred, cy_gt)
            total = total + self.center_weight * center_loss
            result["center_loss"] = center_loss.detach()

        # Log-WH size loss: MSE on log(width) and log(height)
        # Penalizes relative scale error — a 2x size error costs the same
        # regardless of absolute bbox dimensions
        if self.log_wh_weight > 0:
            eps = 1e-6
            w_pred = (pred[..., 2] - pred[..., 0]).clamp(min=eps)
            h_pred = (pred[..., 3] - pred[..., 1]).clamp(min=eps)
            w_gt = (gt[..., 2] - gt[..., 0]).clamp(min=eps)
            h_gt = (gt[..., 3] - gt[..., 1]).clamp(min=eps)
            log_wh_loss = F.mse_loss(torch.log(w_pred), torch.log(w_gt)) + \
                          F.mse_loss(torch.log(h_pred), torch.log(h_gt))
            total = total + self.log_wh_weight * log_wh_loss
            result["log_wh_loss"] = log_wh_loss.detach()

        # Objectness auxiliary loss
        if obj_pred is not None and self.objectness_weight > 0:
            gt_area = (gt[..., 2] - gt[..., 0]) * (gt[..., 3] - gt[..., 1])
            gt_obj = (gt_area > 0.001).float().unsqueeze(-1)  # (B, T, 1)
            obj_loss = F.binary_cross_entropy_with_logits(obj_pred, gt_obj)
            total = total + self.objectness_weight * obj_loss
            obj_acc = ((torch.sigmoid(obj_pred) > 0.5).float() == gt_obj).float().mean()
            result["objectness_loss"] = obj_loss.detach()
            result["objectness_acc"] = obj_acc.detach()

        result["loss"] = total

        # Monitoring: actual IoU (not GIoU) for human readability
        with torch.no_grad():
            actual_iou = _box_iou(pred, gt).mean()

        result["mean_iou"] = actual_iou
        return result


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
