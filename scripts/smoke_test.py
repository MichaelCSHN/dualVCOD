"""M1 Smoke Test — validates the full VCOD pipeline with synthetic data.

Feeds (B,T,C,H,W) tensors through the DummyVCOD model and checks:
  - No dimension errors
  - BBox predictions are in [0,1]
  - IoU / Recall@0.5 are computable
  - FPS benchmark runs without error
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from src.dataloader import SyntheticVideoDataset, collate_video_clips
from src.model import MicroVCOD_Lite
from eval.eval_video_bbox import compute_metrics, benchmark_fps, count_parameters


def main():
    print("=" * 64)
    print("  dualVCOD  |  M1 Pipeline Smoke Test")
    print("=" * 64)

    B, T, C, H, W = 4, 5, 3, 224, 224
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n  device     | {device}")
    print(f"  input      | ({B}, {T}, {C}, {H}, {W})")

    # -- 1. Dataloader --------------------------------------------------
    print(f"\n-- [1/5] Dataloader")
    ds = SyntheticVideoDataset(num_samples=60, T=T, H=H, W=W, C=C)
    dl = DataLoader(ds, batch_size=B, shuffle=False, collate_fn=collate_video_clips)
    frames, gt_bboxes = next(iter(dl))
    print(f"  frames     : {list(frames.shape)}")
    print(f"  gt_bboxes  : {list(gt_bboxes.shape)}")
    print(f"  gt range   : [{gt_bboxes.min():.3f}, {gt_bboxes.max():.3f}]")

    # -- 2. Model -------------------------------------------------------
    print(f"\n-- [2/5] Model (MicroVCOD_Lite | lightweight VCOD)")
    model = MicroVCOD_Lite(T=T, in_channels=C).to(device)
    n_params = count_parameters(model)
    print(f"  params     | {n_params:,}")

    # -- 3. Forward Pass ------------------------------------------------
    print(f"\n-- [3/5] Forward Pass")
    model.eval()
    with torch.no_grad():
        pred = model(frames.to(device))
    print(f"  output     : {list(pred.shape)}")
    print(f"  pred range : [{pred.min():.4f}, {pred.max():.4f}]")

    assert pred.shape == (B, T, 4), f"Expected ({B},{T},4), got {pred.shape}"
    assert not torch.isnan(pred).any(), "NaN in output"
    assert not torch.isinf(pred).any(), "Inf in output"
    assert (pred >= 0).all() and (pred <= 1).all(), "BBox out of [0,1]"
    print("  [PASS] shape / NaN / range checks")

    # -- 4. Evaluation Metrics ------------------------------------------
    print(f"\n-- [4/5] Evaluation")
    metrics = compute_metrics(pred.cpu(), gt_bboxes)
    print(f"  mean BBox IoU | {metrics['mean_iou']:.4f}")
    print(f"  Recall@0.5    | {metrics['recall@0.5']:.4f}")
    print(f"  [PASS] metrics computed (random dummy model, low IoU expected)")

    # -- 5. FPS Benchmark -----------------------------------------------
    print(f"\n-- [5/5] FPS Benchmark (on {device})")
    fps = benchmark_fps(model, dl, device=device, num_iters=30)
    print(f"  throughput  | {fps:.1f} frames/sec")
    print(f"  [PASS] FPS benchmark completed")

    # -- Summary --------------------------------------------------------
    print(f"\n{'=' * 64}")
    print(f"  SMOKE TEST PASSED")
    print(f"{'=' * 64}")
    print(f"  input shape : ({B}, {T}, {C}, {H}, {W})")
    print(f"  output shape: ({B}, {T}, 4)")
    print(f"  parameters  : {n_params:,}")
    print(f"  mean IoU    : {metrics['mean_iou']:.4f}")
    print(f"  Recall@0.5  : {metrics['recall@0.5']:.4f}")
    print(f"  FPS         : {fps:.1f}")
    print(f"  device      : {device}")
    print(f"{'=' * 64}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
