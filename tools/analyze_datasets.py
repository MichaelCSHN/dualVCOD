"""Dataset structure probe — scans all 3 COD dataset directories.

Prints directory organization, annotation formats, image resolutions,
frame counts, and annotation density per video sequence.
"""

import os
import csv
import json
import cv2
from collections import defaultdict


def probe_moca(root):
    """Probe MoCA (JPEGImages + CSV annotations)."""
    print("=" * 64)
    print("  MoCA  —  JPEGImages + annotations.csv")
    print("=" * 64)

    jpeg_root = os.path.join(root, "JPEGImages")
    csv_path = os.path.join(root, "Annotations", "annotations.csv")

    video_names = sorted(os.listdir(jpeg_root))
    print(f"\n  Video sequences: {len(video_names)}")

    # Parse CSV
    annotations = defaultdict(lambda: defaultdict(list))
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            # row: metadata_id, file_list, flags, temporal_coords, spatial_coords, metadata
            file_path = row[1].strip("/")
            spatial = row[4]
            parts = file_path.split("/")
            video = parts[0]
            frame_idx = int(parts[1].replace(".jpg", ""))
            coords = json.loads(spatial)
            if coords[0] == 2:  # rectangle
                annotations[video][frame_idx] = coords[1:5]  # [x, y, w, h]

    total_ann = sum(len(a) for a in annotations.values())
    print(f"  Total annotations: {total_ann}")

    # Sample
    sample_vids = video_names[:3]
    for v in sample_vids:
        frame_dir = os.path.join(jpeg_root, v)
        n_frames = len(os.listdir(frame_dir))
        n_annot = len(annotations[v])

        # Get a sample image resolution
        sample_frame = sorted(os.listdir(frame_dir))[0]
        img = cv2.imread(os.path.join(frame_dir, sample_frame))
        h, w = img.shape[:2]

        ann_frames = sorted(annotations[v].keys())
        ann_interval = ann_frames[1] - ann_frames[0] if len(ann_frames) > 1 else "N/A"

        print(f"\n  [{v}]")
        print(f"    frames: {n_frames}  |  annotated: {n_annot}  |  resolution: {w}x{h}")
        print(f"    annotation interval: every {ann_interval} frames")
        if ann_frames:
            bbox = annotations[v][ann_frames[0]]
            print(f"    example bbox [x,y,w,h]: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")

    print(f"\n  Annotation format: CSV with VIA spatial_coordinates [shape_id, x, y, w, h]")
    print(f"  Frame format: JPG, numbered 00000.jpg .. {max(len(os.listdir(os.path.join(jpeg_root, v))) for v in video_names) - 1:05d}.jpg")
    return annotations


def probe_moca_mask(root):
    """Probe MoCA_Mask (Imgs + GT mask pairs)."""
    print("\n" + "=" * 64)
    print("  MoCA_Mask  —  Train/Test with Imgs + GT masks")
    print("=" * 64)

    for split_name, split_dir in [("Train", "TrainDataset_per_sq"), ("Test", "TestDataset_per_sq")]:
        full = os.path.join(root, split_dir)
        if not os.path.isdir(full):
            continue
        videos = sorted(os.listdir(full))
        print(f"\n  [{split_name}] {len(videos)} sequences")

        for v in videos[:3]:
            imgs_dir = os.path.join(full, v, "Imgs")
            gt_dir = os.path.join(full, v, "GT")
            n_imgs = len(os.listdir(imgs_dir))
            n_gt = len(os.listdir(gt_dir))

            sample_img = sorted(os.listdir(imgs_dir))[0]
            img = cv2.imread(os.path.join(imgs_dir, sample_img))
            h, w = img.shape[:2]

            gt_files = sorted(os.listdir(gt_dir))
            ann_interval = "N/A"
            if len(gt_files) > 1:
                a = int(gt_files[0].replace(".png", ""))
                b = int(gt_files[1].replace(".png", ""))
                ann_interval = b - a

            print(f"    [{v}] frames: {n_imgs}  gt: {n_gt}  res: {w}x{h}  interval: {ann_interval}")

    print(f"\n  Annotation format: PNG mask images (binary), paired with JPG frames")
    print(f"  Mask → BBox conversion: findContours → boundingRect → normalize")


def probe_cad(root):
    """Probe CamouflagedAnimalDataset (frames + groundtruth masks)."""
    print("\n" + "=" * 64)
    print("  CAD  —  {animal}/frames/ + {animal}/groundtruth/")
    print("=" * 64)

    animals = sorted(os.listdir(root))
    print(f"\n  Animal categories: {len(animals)}")

    for a in animals[:5]:
        a_dir = os.path.join(root, a)
        if not os.path.isdir(a_dir):
            continue
        frame_dir = os.path.join(a_dir, "frames")
        gt_dir = os.path.join(a_dir, "groundtruth")

        frames = sorted(os.listdir(frame_dir))
        gts = sorted(os.listdir(gt_dir))
        n_frames = len(frames)
        n_gt = len(gts)

        sample_img = cv2.imread(os.path.join(frame_dir, frames[0]))
        h, w = sample_img.shape[:2]

        ann_interval = "N/A"
        if len(gts) > 1:
            a_n = int(gts[0].split("_")[0])
            b_n = int(gts[1].split("_")[0])
            ann_interval = b_n - a_n

        print(f"\n  [{a}]")
        print(f"    frames: {n_frames}  |  gt: {n_gt}  |  resolution: {w}x{h}")
        print(f"    frame naming: {frames[0]} .. {frames[-1]}")
        print(f"    gt naming:    {gts[0]} .. {gts[-1]}")
        print(f"    annotation interval: every {ann_interval} frames")

    print(f"\n  Annotation format: PNG mask images (binary)")
    print(f"  Frame format: PNG, named {{animal}}_{{:03d}}.png")
    print(f"  Mask → BBox conversion required")


def main():
    datasets = {
        "MoCA": r"D:\ML\COD_datasets\MoCA",
        "MoCA_Mask": r"D:\ML\COD_datasets\MoCA_Mask",
        "CAD": r"D:\ML\COD_datasets\CamouflagedAnimalDataset",
    }
    for name, path in datasets.items():
        if not os.path.isdir(path):
            print(f"[SKIP] {name}: path not found ({path})")
            continue
        if name == "MoCA":
            probe_moca(path)
        elif name == "MoCA_Mask":
            probe_moca_mask(path)
        elif name == "CAD":
            probe_cad(path)

    print("\n" + "=" * 64)
    print("  SUMMARY")
    print("=" * 64)
    print(f"""
  Dataset     | Annotations    | Resolution   | Interval
  ------------|----------------|--------------|--------
  MoCA        | CSV (VIA BBox)  | 1280x720     | every  5 frames
  MoCA_Mask   | PNG masks       | 1280x720     | every  5 frames
  CAD         | PNG masks       |  640x360     | every ~5 frames (varies)
""")


if __name__ == "__main__":
    main()
