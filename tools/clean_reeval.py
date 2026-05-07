"""Clean Re-Evaluation — load clean_seed42 checkpoint, evaluate on MoCA Val set.

Runs:
  1. Fresh prediction (model output)
  2. GT from CSV
  3. Random prediction control
  4. All-zero prediction control
  5. Shuffled prediction control (with dataset center prior interpretation)
  6. Delta between checkpoint metric and clean re-eval metric

Does NOT load any old leaked checkpoint. Does NOT claim publication-grade completeness.
"""

import sys
import os
import random
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset_real import RealVideoBBoxDataset, collate_video_clips
from src.model import MicroVCOD
from eval.eval_video_bbox import compute_metrics, count_parameters

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")


def load_checkpoint(path, model):
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    return state


def collect_predictions(model, loader):
    """Run model and collect all predictions + ground truths."""
    model.eval()
    all_preds, all_gts = [], []
    with torch.no_grad():
        for frames, gt_bboxes in loader:
            frames = frames.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                pred = model(frames)
            all_preds.append(pred.float().cpu())
            all_gts.append(gt_bboxes)
    preds = torch.cat(all_preds, dim=0)
    gts = torch.cat(all_gts, dim=0)
    return preds, gts


def center_prior_prediction(gts):
    """Predict the dataset center prior bbox for every sample.

    The center prior is the mean bbox across all GT samples.
    For camouflaged animal detection, animals tend to be near center.
    """
    mean_bbox = gts.mean(dim=0)  # (T, 4) average per timestep
    return mean_bbox.unsqueeze(0).expand(gts.shape[0], -1, -1).clone()


def random_uniform_prediction(gts):
    """Random uniform bboxes in [0, 1], clipped to valid range."""
    preds = torch.rand_like(gts)
    # Ensure x1 < x2, y1 < y2
    x1, y1, x2, y2 = preds[..., 0], preds[..., 1], preds[..., 2], preds[..., 3]
    x_min = torch.min(x1, x2)
    x_max = torch.max(x1, x2)
    y_min = torch.min(y1, y2)
    y_max = torch.max(y1, y2)
    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)


def zero_prediction(gts):
    """All-zero bbox (degenerate, IoU always 0)."""
    return torch.zeros_like(gts)


def shuffle_prediction(preds, seed=42):
    """Shuffle predictions across samples to break temporal coherence.

    This tests whether the model benefits from temporal ordering.
    If shuffled mIoU ≈ real mIoU, the model isn't using temporal info.
    If shuffled mIoU ≈ center prior mIoU, temporal info is critical.
    """
    rng = random.Random(seed)
    idx = list(range(preds.shape[0]))
    rng.shuffle(idx)
    return preds[idx].clone()


def main():
    parser = argparse.ArgumentParser(description="Clean Re-Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to clean checkpoint. Defaults to clean_seed42_best_miou.pth")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--T", type=int, default=5)
    args = parser.parse_args()

    ckpt_path = args.checkpoint or os.path.join(CHECKPOINT_DIR, "clean_seed42_best_miou.pth")
    if not os.path.isfile(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return 1

    print("=" * 72)
    print("  CLEAN RE-EVALUATION  |  MicroVCOD  |  MoCA Val Set")
    print("=" * 72)
    print(f"  device     : {DEVICE}")
    print(f"  checkpoint : {ckpt_path}")
    print()

    # ── Dataset (MoCA val only, aug=False) ──────────────────────────
    print("  Loading MoCA val set ...")
    full_ds = RealVideoBBoxDataset(
        [r"D:\ML\COD_datasets\MoCA"],
        T=args.T,
        target_size=224,
        augment=False,
    )
    from tools.train import split_by_video
    _, val_idx = split_by_video(full_ds, val_ratio=0.2, seed=42)
    val_ds = Subset(full_ds, val_idx)
    print(f"  Val samples : {len(val_ds)}")
    print()

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_video_clips, num_workers=2, pin_memory=True,
    )

    # ── Model ───────────────────────────────────────────────────────
    model = MicroVCOD(T=args.T, pretrained_backbone=False).to(DEVICE)
    state = load_checkpoint(ckpt_path, model)
    ckpt_miou = state["miou"]
    ckpt_recall = state["recall"]
    ckpt_epoch = state["epoch"]
    print(f"  Checkpoint epoch : {ckpt_epoch}")
    print(f"  Checkpoint mIoU  : {ckpt_miou:.4f}")
    print(f"  Checkpoint R@0.5 : {ckpt_recall:.4f}")
    print()

    # ── Fresh predictions ───────────────────────────────────────────
    print("  Running fresh predictions ...")
    preds, gts = collect_predictions(model, val_loader)
    fresh_metrics = compute_metrics(preds, gts)
    fresh_miou = fresh_metrics["mean_iou"]
    fresh_recall = fresh_metrics["recall@0.5"]
    print(f"  Fresh mIoU       : {fresh_miou:.4f}")
    print(f"  Fresh R@0.5      : {fresh_recall:.4f}")
    print()

    # ── GT from CSV (already loaded as gts above) ───────────────────
    print(f"  GT samples       : {gts.shape[0]}")
    print(f"  GT bbox range    : [{gts.min():.4f}, {gts.max():.4f}]")
    print(f"  GT mean center   : x={(gts[..., 0].mean() + gts[..., 2].mean()) / 2:.4f}, y={(gts[..., 1].mean() + gts[..., 3].mean()) / 2:.4f}")
    print()

    # ── Negative controls ───────────────────────────────────────────
    print("-" * 72)
    print("  NEGATIVE CONTROLS")
    print("-" * 72)

    # 1. Random prediction
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)
    random_preds = random_uniform_prediction(gts)
    random_metrics = compute_metrics(random_preds, gts)
    print(f"  [Control 1] Random uniform bbox:")
    print(f"    mIoU       : {random_metrics['mean_iou']:.4f}")
    print(f"    R@0.5      : {random_metrics['recall@0.5']:.4f}")

    # 2. All-zero prediction
    zero_preds = zero_prediction(gts)
    zero_metrics = compute_metrics(zero_preds, gts)
    print(f"  [Control 2] All-zero bbox:")
    print(f"    mIoU       : {zero_metrics['mean_iou']:.4f}  (expected: 0.0)")
    print(f"    R@0.5      : {zero_metrics['recall@0.5']:.4f}  (expected: 0.0)")

    # 3. Center prior prediction
    center_preds = center_prior_prediction(gts)
    center_metrics = compute_metrics(center_preds, gts)
    print(f"  [Control 3] Dataset center prior:")
    print(f"    mIoU       : {center_metrics['mean_iou']:.4f}")
    print(f"    R@0.5      : {center_metrics['recall@0.5']:.4f}")
    print(f"    Prior bbox : [{center_preds[0, 0, 0]:.4f}, {center_preds[0, 0, 1]:.4f}, {center_preds[0, 0, 2]:.4f}, {center_preds[0, 0, 3]:.4f}]")

    # 4. Shuffled prediction
    shuffled_preds = shuffle_prediction(preds, seed=42)
    shuffled_metrics = compute_metrics(shuffled_preds, gts)
    print(f"  [Control 4] Shuffled prediction (temporal coherence broken):")
    print(f"    mIoU       : {shuffled_metrics['mean_iou']:.4f}")
    print(f"    R@0.5      : {shuffled_metrics['recall@0.5']:.4f}")
    print()

    # ── Interpretation ──────────────────────────────────────────────
    print("-" * 72)
    print("  INTERPRETATION")
    print("-" * 72)

    # Shuffled vs real: how much does temporal info help?
    temporal_gain = fresh_miou - shuffled_metrics["mean_iou"]
    print(f"  Fresh mIoU              : {fresh_miou:.4f}")
    print(f"  Shuffled mIoU            : {shuffled_metrics['mean_iou']:.4f}")
    print(f"  Temporal gain            : {temporal_gain:+.4f}  (fresh - shuffled)")

    # Shuffled vs center prior: does shuffled retain any spatial info?
    spatial_residual = shuffled_metrics["mean_iou"] - center_metrics["mean_iou"]
    print(f"  Center prior mIoU        : {center_metrics['mean_iou']:.4f}")
    print(f"  Shuffled - center        : {spatial_residual:+.4f}")

    # Interpretation guidance
    print()
    if temporal_gain > 0.05:
        print(f"  Temporal gain ({temporal_gain:+.4f}) > 0.05: model BENEFITS from temporal ordering.")
    elif temporal_gain > 0.01:
        print(f"  Temporal gain ({temporal_gain:+.4f}) marginal: model may be learning temporal cues weakly.")
    else:
        print(f"  Temporal gain ({temporal_gain:+.4f}) negligible: model relies primarily on spatial cues.")

    if spatial_residual > 0.02:
        print(f"  Spatial residual ({spatial_residual:+.4f}) > 0.02: shuffled preds retain spatial info beyond center prior.")
        print(f"    This is expected — model bbox distribution is learned from training data spatial prior.")
    elif spatial_residual > -0.02:
        print(f"  Spatial residual ({spatial_residual:+.4f}) near 0: shuffled preds ≈ center prior.")
        print(f"    Temporal information is the primary driver of model performance.")
    else:
        print(f"  Spatial residual ({spatial_residual:+.4f}) negative: shuffled performs WORSE than center prior.")
        print(f"    Shuffling may have created invalid bbox pairs, check temporal stride consistency.")

    print()

    # ── Checkpoint metric delta ─────────────────────────────────────
    print("-" * 72)
    print("  CHECKPOINT METRIC DELTA")
    print("-" * 72)
    delta_miou = fresh_miou - ckpt_miou
    delta_recall = fresh_recall - ckpt_recall
    print(f"  Checkpoint mIoU          : {ckpt_miou:.4f}  (epoch {ckpt_epoch})")
    print(f"  Clean re-eval mIoU       : {fresh_miou:.4f}")
    print(f"  Delta mIoU               : {delta_miou:+.4f}")
    print(f"  Checkpoint R@0.5         : {ckpt_recall:.4f}")
    print(f"  Clean re-eval R@0.5      : {fresh_recall:.4f}")
    print(f"  Delta R@0.5              : {delta_recall:+.4f}")

    if abs(delta_miou) < 0.005:
        print(f"  Delta within 0.005: checkpoint metric matches clean re-eval — consistent.")
    elif abs(delta_miou) < 0.02:
        print(f"  Delta within 0.02: minor discrepancy, likely data-loading order or augmentation OFF variance.")
    else:
        print(f"  Delta > 0.02: significant discrepancy — investigate DataLoader determinism.")
    print()

    # ── Summary ─────────────────────────────────────────────────────
    print("=" * 72)
    print("  CLEAN RE-EVAL SUMMARY")
    print("=" * 72)
    print(f"  {'Metric':30s} {'mIoU':>10s} {'R@0.5':>10s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    print(f"  {'Fresh prediction':30s} {fresh_miou:10.4f} {fresh_recall:10.4f}")
    print(f"  {'Random uniform':30s} {random_metrics['mean_iou']:10.4f} {random_metrics['recall@0.5']:10.4f}")
    print(f"  {'All-zero':30s} {zero_metrics['mean_iou']:10.4f} {zero_metrics['recall@0.5']:10.4f}")
    print(f"  {'Center prior':30s} {center_metrics['mean_iou']:10.4f} {center_metrics['recall@0.5']:10.4f}")
    print(f"  {'Shuffled prediction':30s} {shuffled_metrics['mean_iou']:10.4f} {shuffled_metrics['recall@0.5']:10.4f}")
    print(f"  {'Checkpoint (epoch '+str(ckpt_epoch)+')':30s} {ckpt_miou:10.4f} {ckpt_recall:10.4f}")
    print(f"  {'Delta (fresh - ckpt)':30s} {delta_miou:+10.4f} {delta_recall:+10.4f}")
    print(f"  {'Temporal gain':30s} {temporal_gain:+10.4f}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
