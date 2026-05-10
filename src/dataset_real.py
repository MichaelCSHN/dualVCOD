"""RealVideoBBoxDataset — unified video BBox dataset adapter.

Handles MoCA (CSV BBox), MoCA_Mask (PNG mask → BBox), and CAD (PNG mask → BBox).
All frames are spatially aligned to a fixed target size with BBox coordinates
scaled and normalized to [0, 1].
"""

import os
import csv
import json
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict


# ── Mask → BBox conversion ────────────────────────────────────────────

def mask_to_bbox(mask):
    """Convert binary mask to normalized BBox (x1, y1, x2, y2) in [0, 1].

    Returns None if the mask is empty.
    """
    if mask.ndim == 3:
        mask = mask[..., 0] if mask.shape[-1] == 3 else mask.squeeze(-1)
    mask = (mask > 0).astype(np.uint8)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None

    y1, y2 = np.where(rows)[0][[0, -1]].astype(np.float32)
    x1, x2 = np.where(cols)[0][[0, -1]].astype(np.float32)
    h, w = mask.shape
    return np.array([x1 / w, y1 / h, x2 / w, y2 / h], dtype=np.float32)


# ── MoCA CSV parser ───────────────────────────────────────────────────

def parse_moca_csv(csv_path):
    """Parse MoCA VIA annotations.csv → {video_name: {frame_idx: [x,y,w,h]}}."""
    annotations = defaultdict(dict)
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            file_path = row[1].strip("/")
            spatial = row[4]
            parts = file_path.split("/")
            video = parts[0]
            frame_idx = int(parts[1].replace(".jpg", ""))
            coords = json.loads(spatial)
            if coords[0] == 2:  # rectangle
                annotations[video][frame_idx] = [
                    float(coords[1]),
                    float(coords[2]),
                    float(coords[3]),
                    float(coords[4]),
                ]
    return dict(annotations)


# ── Dataset ───────────────────────────────────────────────────────────


class RealVideoBBoxDataset(Dataset):
    """Unified video BBox dataset for MoCA, MoCA_Mask, and CAD.

    Each item is a clip of T annotated frames with their BBoxes.
    Frames are resized to a fixed spatial size with BBox coordinates
    scaled and normalized to [0, 1].

    Args:
        dataset_paths: list of root paths to include
        T: number of temporal frames per clip
        target_size: spatial size (H=W=target_size) after resize
        dataset_names: optional list of names matching dataset_paths
    """

    def __init__(self, dataset_paths, T=5, target_size=224, dataset_names=None, augment=False,
                 temporal_stride=1):
        self.T = T
        self.target_size = target_size
        self.augment = augment
        self.temporal_stride = temporal_stride
        self.samples = []  # list of {path, video, start_frame, annot_interval, bbox_map}

        if dataset_names is None:
            dataset_names = [f"ds_{i}" for i in range(len(dataset_paths))]

        for ds_path, ds_name in zip(dataset_paths, dataset_names):
            self._index_dataset(ds_path, ds_name)

        if not self.samples:
            raise RuntimeError("No valid video sequences found in given paths")

    def _index_dataset(self, root, ds_name):
        """Discover video sequences and build sample windows."""
        # ── Heuristic: detect dataset type ──
        if os.path.isfile(os.path.join(root, "Annotations", "annotations.csv")):
            self._index_moca(root)
        elif os.path.isdir(os.path.join(root, "TrainDataset_per_sq")):
            self._index_moca_mask(root)
        elif any(
            os.path.isdir(os.path.join(root, d, "frames"))
            for d in os.listdir(root)
        ):
            self._index_cad(root)
        else:
            print(f"[WARN] Unknown dataset format: {root}")

    def _index_moca(self, root):
        csv_path = os.path.join(root, "Annotations", "annotations.csv")
        ann = parse_moca_csv(csv_path)
        jpeg_root = os.path.join(root, "JPEGImages")

        for video in sorted(ann.keys()):
            frame_dir = os.path.join(jpeg_root, video)
            if not os.path.isdir(frame_dir):
                continue
            ann_frames = sorted(ann[video].keys())
            if len(ann_frames) < self.T:
                continue

            # BBox map: frame_index → normalized bbox
            # Original frames: 1280×720
            W_orig, H_orig = 1280, 720
            bbox_map = {}
            for fi in ann_frames:
                x, y, w, h = ann[video][fi]
                bbox = np.array(
                    [x / W_orig, y / H_orig, (x + w) / W_orig, (y + h) / H_orig],
                    dtype=np.float32,
                )
                bbox_map[fi] = np.clip(bbox, 0.0, 1.0)

            interval = ann_frames[1] - ann_frames[0] if len(ann_frames) > 1 else 1

            for start_idx in ann_frames[: len(ann_frames) - self.T + 1]:
                needed_frames = [start_idx + i * interval * self.temporal_stride for i in range(self.T)]
                if all(f in bbox_map for f in needed_frames):
                    self.samples.append(
                        {
                            "frame_dir": frame_dir,
                            "start_frame": start_idx,
                            "annot_interval": interval,
                            "bbox_map": bbox_map,
                            "frame_ext": ".jpg",
                        }
                    )

    def _index_moca_mask(self, root):
        # Hard-isolate: TestDataset_per_sq must NOT enter training
        for split in ["TrainDataset_per_sq"]:
            split_dir = os.path.join(root, split)
            if not os.path.isdir(split_dir):
                continue
            for video in sorted(os.listdir(split_dir)):
                imgs_dir = os.path.join(split_dir, video, "Imgs")
                gt_dir = os.path.join(split_dir, video, "GT")
                if not os.path.isdir(imgs_dir) or not os.path.isdir(gt_dir):
                    continue

                gt_files = sorted(os.listdir(gt_dir))
                gt_indices = [int(f.replace(".png", "")) for f in gt_files]
                if len(gt_indices) < self.T:
                    continue

                interval = gt_indices[1] - gt_indices[0] if len(gt_indices) > 1 else 1

                # Pre-compute bbox map from masks
                bbox_map = {}
                for fi in gt_indices:
                    mask_path = os.path.join(gt_dir, f"{fi:05d}.png")
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        continue
                    bbox = mask_to_bbox(mask)
                    if bbox is not None:
                        bbox_map[fi] = bbox

                if len(bbox_map) < self.T:
                    continue

                video_dir = os.path.join(split_dir, video)
                for start_idx in gt_indices[: len(gt_indices) - self.T + 1]:
                    needed = [start_idx + i * interval * self.temporal_stride for i in range(self.T)]
                    if all(f in bbox_map for f in needed):
                        self.samples.append(
                            {
                                "frame_dir": imgs_dir,
                                "video_dir": video_dir,
                                "start_frame": start_idx,
                                "annot_interval": interval,
                                "bbox_map": bbox_map,
                                "frame_ext": ".jpg",
                            }
                        )

    def _index_cad(self, root):
        for animal in sorted(os.listdir(root)):
            animal_dir = os.path.join(root, animal)
            if not os.path.isdir(animal_dir):
                continue
            frame_dir = os.path.join(animal_dir, "frames")
            gt_dir = os.path.join(animal_dir, "groundtruth")
            if not os.path.isdir(frame_dir) or not os.path.isdir(gt_dir):
                continue

            gt_files = sorted(os.listdir(gt_dir))
            # Build: frame_index → gt_filename
            gt_lookup = {}
            for f in gt_files:
                num = f.split("_")[0]
                if num.isdigit():
                    gt_lookup[int(num)] = f

            gt_indices = sorted(gt_lookup.keys())
            if len(gt_indices) < self.T:
                continue

            # Build: frame_index → frame_filename
            frame_files = sorted(os.listdir(frame_dir))
            frame_lookup = {}
            for f in frame_files:
                base = f.replace(".png", "").replace(".jpg", "")
                # Try "{animal}_NNN" pattern
                prefix = f"{animal}_"
                if base.startswith(prefix):
                    num_str = base[len(prefix):]
                    if num_str.isdigit():
                        frame_lookup[int(num_str)] = f
                # Try "NNN_gt" style (less common but possible)
                parts = base.split("_")
                if not frame_lookup or int(num_str) not in frame_lookup:
                    for p in parts:
                        if p.isdigit():
                            frame_lookup.setdefault(int(p), f)

            interval = gt_indices[1] - gt_indices[0] if len(gt_indices) > 1 else 1

            # Pre-compute bbox map from masks
            bbox_map = {}
            for fi in gt_indices:
                gt_fname = gt_lookup.get(fi)
                if gt_fname is None:
                    continue
                mask_path = os.path.join(gt_dir, gt_fname)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                bbox = mask_to_bbox(mask)
                if bbox is not None:
                    bbox_map[fi] = bbox

            if len(bbox_map) < self.T:
                continue

            for start_idx in gt_indices[: len(gt_indices) - self.T + 1]:
                needed = [start_idx + i * interval * self.temporal_stride for i in range(self.T)]
                if all(f in bbox_map for f in needed):
                    self.samples.append(
                        {
                            "frame_dir": frame_dir,
                            "video_dir": animal_dir,
                            "start_frame": start_idx,
                            "annot_interval": interval,
                            "bbox_map": bbox_map,
                            "frame_lookup": frame_lookup,
                            "frame_ext": ".png",
                        }
                    )

    # ── Item access ───────────────────────────────────────────────────

    def _resolve_frame_path(self, sample, frame_idx):
        """Find the actual frame file given a frame index."""
        frame_dir = sample["frame_dir"]
        ext = sample["frame_ext"]

        # Use pre-built lookup if available (CAD-style)
        lookup = sample.get("frame_lookup")
        if lookup and frame_idx in lookup:
            return os.path.join(frame_dir, lookup[frame_idx])

        # Direct 5-digit naming (MoCA/MoCA_Mask style)
        direct = os.path.join(frame_dir, f"{frame_idx:05d}{ext}")
        if os.path.exists(direct):
            return direct

        # Pattern-based search
        existing = set(os.listdir(frame_dir))
        for fname in sorted(existing):
            if f"_{frame_idx:03d}" in fname or fname.startswith(f"{frame_idx}_"):
                return os.path.join(frame_dir, fname)

        # Fallback: try 0-padded variations
        for pad in [3, 4, 5]:
            fpath = os.path.join(frame_dir, f"{frame_idx:0{pad}d}{ext}")
            if os.path.exists(fpath):
                return fpath

        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        interval = sample["annot_interval"]

        # ── Temporal stride jittering ──────────────────────────────
        if self.augment:
            stride = random.choice([1, 2, 3])
        else:
            stride = 1

        frame_indices = []
        for t in range(self.T):
            fi = sample["start_frame"] + t * interval * stride
            if fi in sample["bbox_map"]:
                frame_indices.append(fi)
            else:
                frame_indices = [sample["start_frame"] + t * interval for t in range(self.T)]
                break

        # ── Spatial augmentation params (consistent across clip) ──
        do_flip = self.augment and random.random() < 0.5
        if self.augment:
            jitter_brightness = 1.0 + random.uniform(-0.15, 0.15)
            jitter_contrast = 1.0 + random.uniform(-0.15, 0.15)
            jitter_saturation = 1.0 + random.uniform(-0.15, 0.15)

        frames = []
        bboxes = []
        for fi in frame_indices:
            fpath = self._resolve_frame_path(sample, fi)

            img = cv2.imread(fpath) if fpath is not None else np.zeros(
                (self.target_size, self.target_size, 3), dtype=np.uint8
            )
            if img is None:
                img = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.target_size, self.target_size))

            bbox = sample["bbox_map"].get(fi, np.array([0, 0, 1, 1], dtype=np.float32))

            # ── Apply spatial augmentations ────────────────────────
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (C, H, W)

            if do_flip:
                img_t = torch.flip(img_t, dims=[-1])  # horizontal flip
                bbox = np.array([1.0 - bbox[2], bbox[1], 1.0 - bbox[0], bbox[3]], dtype=np.float32)

            if self.augment:
                img_t = self._color_jitter(img_t, jitter_brightness, jitter_contrast, jitter_saturation)

            frames.append(img_t)
            bboxes.append(torch.from_numpy(bbox))

        frames = torch.stack(frames, dim=0)  # (T, C, H, W)
        bboxes = torch.stack(bboxes, dim=0)  # (T, 4)
        return frames, bboxes

    @staticmethod
    def _color_jitter(img, brightness, contrast, saturation):
        """Apply consistent brightness/contrast/saturation jitter to a (C,H,W) image."""
        img = img * brightness
        mean = img.mean(dim=[-2, -1], keepdim=True)
        img = (img - mean) * contrast + mean
        gray = img.mean(dim=0, keepdim=True)
        img = gray + (img - gray) * saturation
        return torch.clamp(img, 0.0, 1.0)


def collate_video_clips(batch):
    frames, bboxes = zip(*batch)
    return torch.stack(frames, dim=0), torch.stack(bboxes, dim=0)
