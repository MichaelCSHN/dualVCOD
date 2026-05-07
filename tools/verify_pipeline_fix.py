"""Verify Phase 1 pipeline fixes: CAD threshold=10 + MoCA_Mask TestDataset isolation."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from collections import defaultdict

from src.dataset_real import mask_to_bbox, RealVideoBBoxDataset


def verify_cad_threshold():
    """Check that all 9 CAD categories produce valid bboxes with threshold=10."""
    print("=" * 70)
    print("  VERIFY: CAD mask threshold (10) — all 9 categories")
    print("=" * 70)

    cad_root = r"D:\ML\COD_datasets\CamouflagedAnimalDataset"
    results = {}

    for animal in sorted(os.listdir(cad_root)):
        animal_dir = os.path.join(cad_root, animal)
        gt_dir = os.path.join(animal_dir, "groundtruth")
        if not os.path.isdir(gt_dir):
            continue

        gt_files = sorted(os.listdir(gt_dir))
        total = len(gt_files)
        valid = 0
        empty = 0
        for gf in gt_files:
            mask = cv2.imread(os.path.join(gt_dir, gf), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            bbox = mask_to_bbox(mask)
            if bbox is not None:
                valid += 1
            else:
                empty += 1

        results[animal] = {"total": total, "valid": valid, "empty": empty}
        status = "OK" if valid > 0 else "BLOCKED"
        print(f"  {animal:20s}  GT={total:3d}  valid={valid:3d}  empty={empty:3d}  [{status}]")

    all_ok = all(v["valid"] > 0 for v in results.values())
    print(f"\n  All 9 categories indexed: {'YES' if all_ok else 'NO — ' + str(sum(1 for v in results.values() if v['valid'] == 0)) + ' blocked'}")
    print()
    return results


def verify_moca_mask_isolation():
    """Confirm TestDataset_per_sq is excluded from training index."""
    print("=" * 70)
    print("  VERIFY: MoCA_Mask TestDataset_per_sq hard isolation")
    print("=" * 70)

    root = r"D:\ML\COD_datasets\MoCA_Mask"

    # Check what videos exist on disk
    disk_videos = {"TrainDataset_per_sq": [], "TestDataset_per_sq": []}
    for split in ["TrainDataset_per_sq", "TestDataset_per_sq"]:
        split_dir = os.path.join(root, split)
        if os.path.isdir(split_dir):
            disk_videos[split] = sorted(os.listdir(split_dir))

    print(f"  On disk:")
    print(f"    TrainDataset_per_sq: {len(disk_videos['TrainDataset_per_sq'])} videos")
    print(f"    TestDataset_per_sq:  {len(disk_videos['TestDataset_per_sq'])} videos")

    # Now index via the dataset class
    ds = RealVideoBBoxDataset([root], T=5, target_size=224, augment=False)

    # Extract video names from indexed samples
    indexed_videos = set()
    for sample in ds.samples:
        dir_path = sample.get("video_dir", sample["frame_dir"])
        vname = os.path.basename(dir_path.rstrip("/\\"))
        indexed_videos.add(vname)

    # Check which TestDataset_per_sq videos are in the index
    test_vids_on_disk = set(disk_videos["TestDataset_per_sq"])
    leaked = test_vids_on_disk & indexed_videos
    train_indexed = set(disk_videos["TrainDataset_per_sq"]) & indexed_videos

    print(f"\n  Indexed by dataset:")
    print(f"    Total samples:  {len(ds)}")
    print(f"    Unique videos: {len(indexed_videos)}")
    print(f"    Train videos indexed: {len(train_indexed)} / {len(disk_videos['TrainDataset_per_sq'])}")
    print(f"    Test videos LEAKED:   {len(leaked)} / {len(disk_videos['TestDataset_per_sq'])}")

    if leaked:
        print(f"\n  [FAIL] TestDataset_per_sq videos in training: {sorted(leaked)}")
    else:
        print(f"\n  [PASS] Zero TestDataset_per_sq videos in training index")

    print()
    return len(leaked) == 0


def final_counts():
    """Print final Train/Val video counts across all datasets."""
    print("=" * 70)
    print("  FINAL: Train/Val Video Counts (post-fix)")
    print("=" * 70)

    DATASET_ROOTS = [
        r"D:\ML\COD_datasets\MoCA",
        r"D:\ML\COD_datasets\MoCA_Mask",
        r"D:\ML\COD_datasets\CamouflagedAnimalDataset",
    ]

    total_train_samples = 0
    total_train_videos = 0

    for root in DATASET_ROOTS:
        if not os.path.isdir(root):
            print(f"  {os.path.basename(root):12s} : [NOT FOUND]")
            continue

        ds = RealVideoBBoxDataset([root], T=5, target_size=224, augment=False)
        videos = set()
        for s in ds.samples:
            dir_path = s.get("video_dir", s["frame_dir"])
            videos.add(os.path.basename(dir_path.rstrip("/\\")))

        print(f"  {os.path.basename(root):12s} : {len(videos):4d} videos, {len(ds):5d} windows")
        total_train_samples += len(ds)
        total_train_videos += len(videos)

    # MoCA validation split
    moca_ds = RealVideoBBoxDataset([DATASET_ROOTS[0]], T=5, target_size=224, augment=False)

    # Inline split_by_video to avoid CUDA import
    video_to_indices = defaultdict(list)
    for i in range(len(moca_ds)):
        s = moca_ds.samples[i]
        dir_path = s.get("video_dir", s["frame_dir"])
        vname = os.path.basename(dir_path.rstrip("/\\"))
        video_to_indices[vname].append(i)

    import random
    videos_list = sorted(video_to_indices.keys())
    rng = random.Random(42)
    rng.shuffle(videos_list)
    n_val = max(1, int(len(videos_list) * 0.2))
    val_videos = set(videos_list[:n_val])
    train_videos = set(videos_list[n_val:])

    val_windows = sum(len(video_to_indices[v]) for v in val_videos & set(video_to_indices.keys()))
    train_windows = sum(len(video_to_indices[v]) for v in train_videos & set(video_to_indices.keys()))

    print(f"\n  MoCA validation split (seed=42, 80/20 by video):")
    print(f"    Train: {len(train_videos)} videos, {train_windows} windows")
    print(f"    Val:   {len(val_videos)} videos, {val_windows} windows")

    print(f"\n  [JOINT TRAIN] {total_train_videos} videos, {total_train_samples} windows")
    print(f"  [VAL]         {len(val_videos)} videos, {val_windows} windows")
    print()


if __name__ == "__main__":
    cad_results = verify_cad_threshold()
    isolation_ok = verify_moca_mask_isolation()
    final_counts()
    print("=" * 70)
    print("  PIPELINE FIX VERIFICATION COMPLETE")
    print("=" * 70)
