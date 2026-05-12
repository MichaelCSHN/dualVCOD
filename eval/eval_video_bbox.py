import time
from collections import defaultdict
import torch


def bbox_iou(pred, gt):
    """Pairwise IoU for boxes in (x1, y1, x2, y2) format.

    Args:
        pred: (..., 4) tensor
        gt:   (..., 4) tensor

    Returns:
        iou:  (...,)  tensor
    """
    ix1 = torch.max(pred[..., 0], gt[..., 0])
    iy1 = torch.max(pred[..., 1], gt[..., 1])
    ix2 = torch.min(pred[..., 2], gt[..., 2])
    iy2 = torch.min(pred[..., 3], gt[..., 3])

    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    area_pred = (pred[..., 2] - pred[..., 0]).clamp(min=0) * (pred[..., 3] - pred[..., 1]).clamp(min=0)
    area_gt = (gt[..., 2] - gt[..., 0]).clamp(min=0) * (gt[..., 3] - gt[..., 1]).clamp(min=0)
    union = area_pred + area_gt - inter

    return inter / (union + 1e-6)


def compute_metrics(pred_bboxes, gt_bboxes):
    """Compute mean IoU and Recall@0.5 across all (B, T) predictions.

    Returns dict with keys: mean_iou, recall_at_0_5, per_frame_ious.
    """
    ious = bbox_iou(pred_bboxes, gt_bboxes)
    mean_iou = ious.mean().item()
    recall = (ious >= 0.5).float().mean().item()
    return {"mean_iou": mean_iou, "recall@0.5": recall, "per_frame_ious": ious}


def categorize_size(area):
    """Categorize normalized bbox area."""
    if area < 0.01:
        return "tiny"
    elif area < 0.05:
        return "small"
    elif area < 0.15:
        return "medium"
    else:
        return "large"


def classify_error(iou, area_ratio, center_err):
    """Classify per-frame prediction error type."""
    if iou >= 0.5:
        return "good"
    if area_ratio > 2.0:
        return "pred_too_large"
    if area_ratio < 0.5:
        return "pred_too_small"
    if center_err > 0.15:
        return "center_shift"
    return "scale_mismatch"


def compute_per_frame_metrics(pred_bboxes, gt_bboxes):
    """Comprehensive per-frame validation metrics.

    Args:
        pred_bboxes: (N, T, 4) or (N*T, 4) tensor
        gt_bboxes:   same shape tensor

    Returns dict with keys:
        per_frame_mIoU, bad_frame_rate, R@0.5,
        IoU_by_size (dict), area_ratio_mean, area_ratio_median,
        pred_ratio_gt1_5, pred_ratio_lt0_67,
        center_error_mean, center_error_median,
        width_error_mean, height_error_mean,
        error_counts (dict of error_type -> count),
        IoU_by_AR (dict: tall/square/wide),
        size_distribution (dict),
    """
    if pred_bboxes.dim() == 3:
        N, T, _ = pred_bboxes.shape
        pred_bboxes = pred_bboxes.reshape(N * T, 4)
        gt_bboxes = gt_bboxes.reshape(N * T, 4)

    ious = bbox_iou(pred_bboxes, gt_bboxes)  # (N*T,)

    # Core metrics
    per_frame_mIoU = ious.mean().item()
    bad_frame_rate = (ious < 0.5).float().mean().item()
    r05 = (ious >= 0.5).float().mean().item()

    # Bbox areas
    pred_w = (pred_bboxes[:, 2] - pred_bboxes[:, 0]).clamp(min=0)
    pred_h = (pred_bboxes[:, 3] - pred_bboxes[:, 1]).clamp(min=0)
    gt_w = (gt_bboxes[:, 2] - gt_bboxes[:, 0]).clamp(min=0)
    gt_h = (gt_bboxes[:, 3] - gt_bboxes[:, 1]).clamp(min=0)
    pred_area = pred_w * pred_h
    gt_area = gt_w * gt_h
    area_ratio = pred_area / (gt_area + 1e-8)

    # Center error
    pred_cx = (pred_bboxes[:, 0] + pred_bboxes[:, 2]) / 2
    pred_cy = (pred_bboxes[:, 1] + pred_bboxes[:, 3]) / 2
    gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2
    gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2
    center_err = ((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2).sqrt()

    # Width/height errors
    w_err = pred_w - gt_w
    h_err = pred_h - gt_h

    # Aspect ratios
    gt_ar = gt_w / (gt_h + 1e-8)
    pred_ar = pred_w / (pred_h + 1e-8)

    # IoU by object size
    iou_by_size = defaultdict(list)
    for i in range(len(ious)):
        size_cat = categorize_size(gt_area[i].item())
        iou_by_size[size_cat].append(ious[i].item())

    # IoU by aspect ratio
    iou_by_ar = defaultdict(list)
    for i in range(len(ious)):
        ar = gt_ar[i].item()
        if ar < 0.67:
            ar_cat = "tall"
        elif ar > 1.5:
            ar_cat = "wide"
        else:
            ar_cat = "square"
        iou_by_ar[ar_cat].append(ious[i].item())

    # Error classification
    error_counts = defaultdict(int)
    for i in range(len(ious)):
        err = classify_error(ious[i].item(), area_ratio[i].item(), center_err[i].item())
        error_counts[err] += 1

    # Size distribution
    size_dist = defaultdict(int)
    for i in range(len(gt_area)):
        size_dist[categorize_size(gt_area[i].item())] += 1

    return {
        "per_frame_mIoU": per_frame_mIoU,
        "bad_frame_rate": bad_frame_rate,
        "R@0.5": r05,
        # Size-stratified IoU
        "IoU_tiny": sum(iou_by_size["tiny"]) / max(len(iou_by_size["tiny"]), 1),
        "IoU_small": sum(iou_by_size["small"]) / max(len(iou_by_size["small"]), 1),
        "IoU_medium": sum(iou_by_size["medium"]) / max(len(iou_by_size["medium"]), 1),
        "IoU_large": sum(iou_by_size["large"]) / max(len(iou_by_size["large"]), 1),
        # Area ratio
        "area_ratio_mean": area_ratio.mean().item(),
        "area_ratio_median": area_ratio.median().item(),
        "pred_gt_1_5": (area_ratio > 1.5).float().mean().item(),
        "pred_lt_0_67": (area_ratio < 0.67).float().mean().item(),
        # Center error
        "center_error_mean": center_err.mean().item(),
        "center_error_median": center_err.median().item(),
        # Width/height errors
        "width_error_mean": w_err.mean().item(),
        "height_error_mean": h_err.mean().item(),
        # Error counts
        "n_good": error_counts["good"],
        "n_pred_too_large": error_counts["pred_too_large"],
        "n_pred_too_small": error_counts["pred_too_small"],
        "n_center_shift": error_counts["center_shift"],
        "n_scale_mismatch": error_counts["scale_mismatch"],
        "total_frames": len(ious),
        # IoU by AR
        "IoU_tall": sum(iou_by_ar["tall"]) / max(len(iou_by_ar["tall"]), 1),
        "IoU_square": sum(iou_by_ar["square"]) / max(len(iou_by_ar["square"]), 1),
        "IoU_wide": sum(iou_by_ar["wide"]) / max(len(iou_by_ar["wide"]), 1),
        # Size distribution
        "size_distribution": dict(size_dist),
    }


def benchmark_fps(model, dataloader, device="cuda", num_iters=50):
    """Measure throughput in frames per second.

    Includes the full forward pass; warm-up iterations are excluded.
    """
    model.eval()
    model.to(device)

    for i, (frames, _) in enumerate(dataloader):
        if i >= 3:
            break
        _ = model(frames.to(device))

    if device == "cuda":
        torch.cuda.synchronize()

    total_frames = 0
    t0 = time.time()

    with torch.no_grad():
        for i, (frames, _) in enumerate(dataloader):
            if i >= num_iters:
                break
            _ = model(frames.to(device))
            total_frames += frames.size(0) * frames.size(1)

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - t0
    return total_frames / elapsed if elapsed > 0 else 0.0


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
