"""M4 Final Benchmark — load best checkpoint, evaluate on MoCA Val set.

Reports: mean BBox IoU, Recall@0.5, FPS (end-to-end), parameter count.
"""

import sys
import os
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Subset

from src.dataset_real import RealVideoBBoxDataset, collate_video_clips
from src.model import MicroVCOD
from eval.eval_video_bbox import compute_metrics, count_parameters

# ── Memory safety ───────────────────────────────────────────────────
torch.cuda.set_per_process_memory_fraction(0.45, 0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")


def load_checkpoint(path, model, device="cuda"):
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    print(f"  loaded checkpoint : {os.path.basename(path)}")
    print(f"  epoch {state['epoch']}  |  mIoU={state['miou']:.4f}  |  R@0.5={state['recall']:.4f}")
    return state


@torch.no_grad()
def evaluate_full(model, loader):
    """Accumulate all predictions and compute final metrics."""
    model.eval()
    all_preds, all_gts = [], []

    for frames, gt_bboxes in loader:
        frames = frames.to(DEVICE)
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            pred = model(frames)
        all_preds.append(pred.float().cpu())
        all_gts.append(gt_bboxes)

    preds = torch.cat(all_preds, dim=0)
    gts = torch.cat(all_gts, dim=0)
    return compute_metrics(preds, gts)


def benchmark_fps(model, loader, num_iters=30):
    """End-to-end FPS including dataloading + forward pass."""
    model.eval()
    torch.cuda.synchronize()

    total_frames = 0
    t0 = time.time()

    for i, (frames, _) in enumerate(loader):
        if i >= num_iters:
            break
        frames = frames.to(DEVICE)
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            _ = model(frames)
        total_frames += frames.size(0) * frames.size(1)

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    return total_frames / elapsed if elapsed > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="VCOD Benchmark")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint. Defaults to best_greenvcod_box_miou.pth in checkpoints/")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--T", type=int, default=5)
    args = parser.parse_args()

    ckpt_path = args.checkpoint or os.path.join(CHECKPOINT_DIR, "best_greenvcod_box_miou.pth")
    if not os.path.isfile(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        print("Run tools/train.py first to generate checkpoints.")
        return 1

    print("=" * 75)
    print("  M4 Final Benchmark  |  MicroVCOD  |  MoCA Val Set")
    print("=" * 75)
    print(f"  device      : {DEVICE}")
    print(f"  checkpoint  : {ckpt_path}")
    print()

    # ── Dataset ──────────────────────────────────────────────────────
    print("  Loading MoCA dataset ...")
    full_ds = RealVideoBBoxDataset(
        [r"D:\ML\COD_datasets\MoCA"],
        T=args.T,
        target_size=224,
    )
    print(f"  Total samples : {len(full_ds)}")

    from tools.train import split_by_video
    train_idx, val_idx = split_by_video(full_ds, val_ratio=0.2, seed=42)
    val_ds = Subset(full_ds, val_idx)
    print(f"  Val samples   : {len(val_ds)}")
    print()

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_video_clips, num_workers=2,
        pin_memory=True, prefetch_factor=2,
    )

    # ── Model ────────────────────────────────────────────────────────
    model = MicroVCOD(T=args.T, pretrained_backbone=False).to(DEVICE)
    n_params = count_parameters(model)
    state = load_checkpoint(ckpt_path, model, DEVICE)
    print()

    # ── Metrics ──────────────────────────────────────────────────────
    print("  Computing metrics ...")
    metrics = evaluate_full(model, val_loader)
    print(f"  mean BBox IoU  : {metrics['mean_iou']:.4f}")
    print(f"  Recall@0.5     : {metrics['recall@0.5']:.4f}")
    print()

    # ── FPS ──────────────────────────────────────────────────────────
    print("  Benchmarking FPS ...")
    fps_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_video_clips, num_workers=0,
    )
    fps = benchmark_fps(model, fps_loader, num_iters=min(30, len(fps_loader)))
    print(f"  FPS (end-to-end): {fps:.1f} frames/sec")
    print()

    # ── Summary table ────────────────────────────────────────────────
    print("=" * 75)
    print("  PHASE 1 FINAL RESULTS")
    print("=" * 75)
    print(f"  {'Metric':25s} │ {'Value':>15s}")
    print(f"  {'─'*25}─┼─{'─'*15}")
    print(f"  {'mean BBox IoU':25s} │ {metrics['mean_iou']:15.4f}")
    print(f"  {'Recall@0.5':25s} │ {metrics['recall@0.5']:15.4f}")
    print(f"  {'FPS (end-to-end)':25s} │ {fps:14.1f}")
    print(f"  {'Parameters':25s} │ {n_params:15,}")
    print(f"  {'Model':25s} │ {'MicroVCOD':>15s}")
    print(f"  {'Backbone':25s} │ {'MobileNetV3-Small+FPN':>15s}")
    print(f"  {'Device':25s} │ {'RTX 4090':>15s}")
    print(f"  {'Precision':25s} │ {'AMP fp16':>15s}")
    print(f"{'=' * 75}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
