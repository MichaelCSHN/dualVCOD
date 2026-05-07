import time
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
