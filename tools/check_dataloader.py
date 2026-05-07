"""Dataloader dry-run validation — checks tensor shapes, bbox validity, no CUDA import.

Usage:
    python tools/check_dataloader.py
    python tools/check_dataloader.py --batches 10 --batch_size 16
"""

import sys
import os
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, ConcatDataset

from src.dataset_real import RealVideoBBoxDataset, collate_video_clips


DATASET_ROOTS = [
    r"D:\ML\COD_datasets\MoCA",
    r"D:\ML\COD_datasets\MoCA_Mask",
    r"D:\ML\COD_datasets\CamouflagedAnimalDataset",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def check_batch(frames, bboxes, batch_idx):
    """Validate a single batch. Returns list of error strings."""
    errors = []

    # ── Shape checks ───────────────────────────────────────────────
    if frames.dim() != 5:
        errors.append(f"frames.dim()={frames.dim()}, expected 5 (B,T,C,H,W)")
        return errors  # can't check further

    B, T, C, H, W = frames.shape

    if C != 3:
        errors.append(f"channels={C}, expected 3")
    if H != 224 or W != 224:
        errors.append(f"spatial=({H},{W}), expected (224,224)")

    if bboxes.dim() != 3:
        errors.append(f"bboxes.dim()={bboxes.dim()}, expected 3 (B,T,4)")
        return errors

    if bboxes.shape[0] != B:
        errors.append(f"batch size mismatch: frames={B}, bboxes={bboxes.shape[0]}")
    if bboxes.shape[1] != T:
        errors.append(f"T mismatch: frames={T}, bboxes={bboxes.shape[1]}")
    if bboxes.shape[2] != 4:
        errors.append(f"bbox coords={bboxes.shape[2]}, expected 4")

    # ── Value checks ───────────────────────────────────────────────
    if torch.isnan(frames).any():
        errors.append("frames contain NaN")
    if torch.isinf(frames).any():
        errors.append("frames contain Inf")
    if torch.isnan(bboxes).any():
        errors.append("bboxes contain NaN")
    if torch.isinf(bboxes).any():
        errors.append("bboxes contain Inf")

    # BBox range [0, 1]
    bmin, bmax = bboxes.min().item(), bboxes.max().item()
    if bmin < 0.0:
        n_neg = (bboxes < 0.0).sum().item()
        errors.append(f"bboxes below 0: {n_neg} values (min={bmin:.6f})")
    if bmax > 1.0:
        n_oob = (bboxes > 1.0).sum().item()
        errors.append(f"bboxes above 1: {n_oob} values (max={bmax:.6f})")

    # Frame range sanity
    fmin, fmax = frames.min().item(), frames.max().item()
    if fmin < 0.0 or fmax > 1.0:
        errors.append(f"frames out of [0,1]: min={fmin:.4f}, max={fmax:.4f}")

    # ── Semantic checks ─────────────────────────────────────────────
    # All-zero bboxes in any frame
    for b in range(B):
        for t in range(T):
            bbox = bboxes[b, t]
            if (bbox == 0.0).all():
                errors.append(f"batch {batch_idx}, sample {b}, frame {t}: all-zero bbox")
            # x1 < x2, y1 < y2
            if bbox[0] >= bbox[2]:
                errors.append(
                    f"batch {batch_idx}, sample {b}, frame {t}: "
                    f"x1={bbox[0]:.4f} >= x2={bbox[2]:.4f}"
                )
            if bbox[1] >= bbox[3]:
                errors.append(
                    f"batch {batch_idx}, sample {b}, frame {t}: "
                    f"y1={bbox[1]:.4f} >= y2={bbox[3]:.4f}"
                )

    return errors


def run_check(num_batches=8, batch_size=16, T=5):
    print("=" * 70)
    print("  DATALOADER DRY-RUN — Tensor Shape & BBox Validity Check")
    print("=" * 70)
    print(f"  device     : {DEVICE}")
    print(f"  batches    : {num_batches}")
    print(f"  batch_size : {batch_size}")
    print(f"  T          : {T}")
    print()

    # ── Build datasets (no augmentation for deterministic check) ───
    print("  Loading datasets (augmentation OFF, deterministic mode) ...")
    train_sets = []
    for root in DATASET_ROOTS:
        if os.path.isdir(root):
            ds = RealVideoBBoxDataset([root], T=T, target_size=224, augment=False)
            train_sets.append(ds)
            # Count videos
            videos = set()
            for s in ds.samples:
                dir_path = s.get("video_dir", s["frame_dir"])
                videos.add(os.path.basename(dir_path.rstrip("/\\")))
            print(f"    {os.path.basename(root):12s} : {len(videos):4d} videos, {len(ds):5d} windows")
        else:
            print(f"    {os.path.basename(root):12s} : [NOT FOUND]")

    joint_ds = ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]
    print(f"    {'TOTAL':12s} : {len(joint_ds):5d} windows")
    print()

    loader = DataLoader(
        joint_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_video_clips,
        num_workers=0,
        pin_memory=False,
    )

    # ── Run batches ────────────────────────────────────────────────
    total_errors = []
    batch_shapes = []

    print(f"  Running {num_batches} batches ...")
    print()

    for batch_idx, (frames, bboxes) in enumerate(loader):
        if batch_idx >= num_batches:
            break

        B, T_batch, C, H, W = frames.shape
        batch_shapes.append((B, T_batch, C, H, W))

        errors = check_batch(frames, bboxes, batch_idx)
        total_errors.extend(errors)

        # Per-batch summary
        bbox_range = f"[{bboxes.min().item():.4f}, {bboxes.max().item():.4f}]"
        err_str = f"{len(errors)} errors" if errors else "OK"
        print(f"  batch {batch_idx:3d} | frames ({B},{T_batch},{C},{H},{W}) "
              f"| bboxes ({B},{T_batch},4) | range {bbox_range} | {err_str}")
        for err in errors[:5]:
            print(f"          ! {err}")
        if len(errors) > 5:
            print(f"          ... and {len(errors) - 5} more errors")

    print()

    # ── Summary ────────────────────────────────────────────────────
    print("=" * 70)
    print("  DRY-RUN SUMMARY")
    print("=" * 70)

    # Shape consistency
    if batch_shapes:
        first_shape = batch_shapes[0]
        shape_consistent = all(s == first_shape for s in batch_shapes)
        print(f"  Shape consistency : {'PASS' if shape_consistent else 'FAIL'}")
        print(f"  All batches shape : {first_shape}  (B,T,C,H,W)")
    else:
        print(f"  Shape consistency : NO DATA")
        shape_consistent = False

    # Error summary
    if total_errors:
        print(f"\n  [FAIL] {len(total_errors)} validation errors found:")
        # Group by type
        error_types = defaultdict(int)
        for e in total_errors:
            key = e.split(":")[0] if ":" in e else e[:60]
            error_types[key] += 1
        for k, v in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"    {v:4d}x  {k}")
    else:
        print(f"\n  [PASS] All checks passed — {num_batches} batches clean")

    print()
    print("=" * 70)
    print("  DATALOADER PRE-CHECK COMPLETE")
    print("=" * 70)

    return len(total_errors) == 0 and shape_consistent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataloader dry-run validator")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--T", type=int, default=5)
    args = parser.parse_args()

    success = run_check(
        num_batches=args.batches,
        batch_size=args.batch_size,
        T=args.T,
    )
    sys.exit(0 if success else 1)
