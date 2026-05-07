"""M2 Real-Data Validation — MicroVCOD on MoCA / MoCA_Mask / CAD.

Feeds a real video clip (T=5 frames) through the upgraded backbone and
reports output shapes, IoU metrics, FPS, and parameter count.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from src.dataset_real import RealVideoBBoxDataset, collate_video_clips
from src.model import MicroVCOD, MicroVCOD_Lite
from eval.eval_video_bbox import compute_metrics, benchmark_fps, count_parameters


def test_on_dataset(ds_path, ds_name, T=5, target_size=224, B=2):
    print(f"\n{'─' * 64}")
    print(f"  [{ds_name}]  {ds_path}")
    print(f"{'─' * 64}")

    ds = RealVideoBBoxDataset([ds_path], T=T, target_size=target_size,
                               dataset_names=[ds_name])
    dl = DataLoader(ds, batch_size=B, shuffle=True, collate_fn=collate_video_clips)

    try:
        frames, gt_bboxes = next(iter(dl))
    except StopIteration:
        print(f"  [SKIP] No valid clips found (need >= {T} annotated frames per window)")
        return None

    actual_B, actual_T = frames.shape[0], frames.shape[1]
    print(f"  actual batch  : ({actual_B}, {actual_T}, 3, {target_size}, {target_size})")
    print(f"  gt_bboxes     : {list(gt_bboxes.shape)}  range [{gt_bboxes.min():.4f}, {gt_bboxes.max():.4f}]")

    # Build model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MicroVCOD(T=actual_T, pretrained_backbone=True).to(device)
    n_params = count_parameters(model)

    # Forward pass
    model.eval()
    with torch.no_grad():
        pred = model(frames.to(device))
    print(f"  output        : {list(pred.shape)}")
    print(f"  params        : {n_params:,}")

    # Validation checks
    assert pred.shape == (actual_B, actual_T, 4), f"Shape error: {pred.shape}"
    assert not torch.isnan(pred).any(), "NaN in output"
    assert not torch.isinf(pred).any(), "Inf in output"
    assert (pred >= 0).all() and (pred <= 1).all(), "BBox out of [0,1]"
    print("  [PASS] shape / NaN / range checks")

    # Metrics
    metrics = compute_metrics(pred.cpu(), gt_bboxes)
    print(f"  mean IoU      : {metrics['mean_iou']:.4f}")
    print(f"  recall@0.5    : {metrics['recall@0.5']:.4f}")

    # FPS benchmark
    fps = benchmark_fps(model, dl, device=device, num_iters=min(30, len(dl)))
    print(f"  FPS           : {fps:.1f} frames/sec  (on {device})")

    return {
        "dataset": ds_name,
        "n_params": n_params,
        "mean_iou": metrics["mean_iou"],
        "recall": metrics["recall@0.5"],
        "fps": fps,
        "n_samples": len(ds),
        "batch_shape": (actual_B, actual_T),
    }


def main():
    print("=" * 64)
    print("  M2 Real-Data Validation  |  MicroVCOD")
    print("=" * 64)

    datasets = [
        (r"D:\ML\COD_datasets\MoCA", "MoCA"),
        (r"D:\ML\COD_datasets\MoCA_Mask", "MoCA_Mask"),
        (r"D:\ML\COD_datasets\CamouflagedAnimalDataset", "CAD"),
    ]

    results = []
    for path, name in datasets:
        if not os.path.isdir(path):
            print(f"\n  [{name}] path not found, skipping: {path}")
            continue
        r = test_on_dataset(path, name, T=5, target_size=224, B=4)
        if r:
            results.append(r)

    # Summary
    print(f"\n{'=' * 64}")
    print(f"  M2 SUMMARY")
    print(f"{'=' * 64}")
    print(f"  {'Dataset':12s} {'Samples':>6s} {'Params':>10s} {'FPS':>8s} {'mIoU':>8s} {'R@0.5':>8s}")
    print(f"  {'─'*12} {'─'*6} {'─'*10} {'─'*8} {'─'*8} {'─'*8}")
    for r in results:
        print(f"  {r['dataset']:12s} {r['n_samples']:>6d} {r['n_params']:>10,} {r['fps']:>7.1f}  {r['mean_iou']:>7.4f}  {r['recall']:>7.4f}")

    print(f"\n  All checks passed. Pipeline validated on real data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
