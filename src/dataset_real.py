"""RealVideoBBoxDataset — unified video BBox dataset adapter.

Handles MoCA (CSV BBox), MoCA_Mask (PNG mask → BBox), and CAD (PNG mask → BBox).
All frames are spatially aligned to a fixed target size with BBox coordinates
scaled and normalized to [0, 1].
"""

import os
import csv
import json
import random
import hashlib
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import io as tvio
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
                 temporal_stride=1, cache_dir=None, resized_root=None, jitter_strength=0.15,
                 return_mask=False, bg_mix_prob=0.0, dense_target_mode="hard",
                 mask_hw_s4=None,
                 zoom_enabled=False, zoom_context_factor=2.0,
                 zoom_prob_tiny=0.8, zoom_prob_small=0.5,
                 zoom_prob_medium=0.2, zoom_prob_large=0.0):
        self.T = T
        self.target_size = target_size
        self.augment = augment
        self.temporal_stride = temporal_stride
        self.cache_dir = cache_dir
        self.resized_root = resized_root
        self.jitter_strength = jitter_strength
        self.return_mask = return_mask
        self.bg_mix_prob = bg_mix_prob
        self.dense_target_mode = dense_target_mode
        self.mask_hw_s4 = mask_hw_s4
        self.zoom_enabled = zoom_enabled
        self.zoom_context_factor = zoom_context_factor
        self.zoom_prob_tiny = zoom_prob_tiny
        self.zoom_prob_small = zoom_prob_small
        self.zoom_prob_medium = zoom_prob_medium
        self.zoom_prob_large = zoom_prob_large
        self._src_roots = []  # list of (source_root, dataset_basename) for path mapping
        self.samples = []  # list of {path, video, start_frame, annot_interval, bbox_map}

        if dataset_names is None:
            dataset_names = [f"ds_{i}" for i in range(len(dataset_paths))]

        for ds_path, ds_name in zip(dataset_paths, dataset_names):
            self._src_roots.append((os.path.normpath(ds_path), os.path.basename(ds_path)))
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
                            "gt_lookup": gt_lookup,
                            "frame_ext": ".png",
                        }
                    )

    # ── Image cache / resized path ──────────────────────────────────

    def _to_resized_path(self, fpath):
        """Map a source frame path to its pre-resized JPEG counterpart."""
        fpath_norm = os.path.normpath(fpath)
        for src_root, ds_name in self._src_roots:
            if fpath_norm.startswith(src_root):
                rel = os.path.relpath(fpath_norm, src_root)
                stem = os.path.splitext(rel)[0]
                return os.path.join(self.resized_root, ds_name, stem + ".jpg")
        return None

    def _cache_path_for(self, frame_path):
        """Compute cache file path for a given source frame path + target_size."""
        key = f"{frame_path}_{self.target_size}"
        h = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.npy")

    def _load_or_decode(self, fpath):
        """Load frame, preferring pre-resized JPEG when resized_root is set.

        With resized_root: torchvision decode_image → RGB tensor (C,H,W) uint8.
        Without: cv2 imread → cvtColor BGR2RGB → resize → numpy (H,W,C) uint8.
        """
        if fpath is None:
            if self.resized_root:
                return torch.zeros(3, self.target_size, self.target_size, dtype=torch.uint8)
            return np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)

        # ── Pre-resized path (torchvision, no cvtColor, no resize) ──
        if self.resized_root is not None:
            rpath = self._to_resized_path(fpath)
            if rpath is not None and os.path.exists(rpath):
                data = tvio.read_file(rpath)
                img = tvio.decode_image(data)  # (C, H, W) uint8, RGB
                return img
            # fall through to cv2 path if resized file missing

        # ── Cache path (.npy) ──
        if self.cache_dir:
            cache_path = self._cache_path_for(fpath)
            if os.path.exists(cache_path):
                return np.load(cache_path)

        # ── Original cv2 path ──
        img = cv2.imread(fpath)
        if img is None:
            if self.resized_root:
                return torch.zeros(3, self.target_size, self.target_size, dtype=torch.uint8)
            return np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.target_size, self.target_size))

        if self.cache_dir:
            cache_path = self._cache_path_for(fpath)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            tmp_path = cache_path + ".tmp"
            np.save(tmp_path, img)
            try:
                os.replace(tmp_path, cache_path)  # atomic rename
            except OSError:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        return img

    # ── Mask loading (for dense_fg_aux supervision) ─────────────────────

    def _resolve_mask_path(self, sample, frame_idx):
        """Find GT mask file for a given frame index. Returns None if no mask file."""
        video_dir = sample.get("video_dir")
        if video_dir is None:
            return None  # MoCA CSV — generate from bbox

        # MoCA_Mask: GT/{fi:05d}.png
        mask_path = os.path.join(video_dir, "GT", f"{frame_idx:05d}.png")
        if os.path.exists(mask_path):
            return mask_path

        # CAD: groundtruth/{gt_lookup[fi]}
        gt_lookup = sample.get("gt_lookup", {})
        if frame_idx in gt_lookup:
            mask_path = os.path.join(video_dir, "groundtruth", gt_lookup[frame_idx])
            if os.path.exists(mask_path):
                return mask_path

        return None

    def _load_mask(self, sample, frame_idx, bbox_norm, hw=28):
        """Load or generate dense targets for a frame at hw×hw resolution.

        dense_target_mode dispatch:
          "hard"                    — binary mask (real PNG or bbox rectangle)
          "soft_mask"               — Gaussian blur on real PNG masks; hard bbox fallback
          "soft_mask_adaptive"      — size-dependent Gaussian blur on real PNG masks
          "soft_bbox"               — hard real PNG masks; Gaussian falloff from bbox
          "ce"                      — center+extent 5-ch target (bbox-derived, no PNG needed)
        """
        if self.dense_target_mode in ("ce",):
            if hw != 28:
                raise NotImplementedError("CE targets only supported at hw=28")
            return self._make_ce_targets(bbox_norm)

        mask_path = self._resolve_mask_path(sample, frame_idx)

        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.resize(mask, (hw, hw), interpolation=cv2.INTER_AREA)
                binary = (mask > 0).astype(np.float32)
                if self.dense_target_mode in ("soft_mask",):
                    return self._soften_mask(binary, hw=hw)
                if self.dense_target_mode in ("soft_mask_adaptive",):
                    return self._soften_mask_adaptive(binary, bbox_norm, hw=hw)
                return binary

        # No real mask — generate from bbox (always hard rectangle)
        return self._bbox_to_mask(bbox_norm, hw, hw)

    @staticmethod
    def _soften_mask(mask, sigma=1.0, hw=28):
        """Gaussian blur on binary mask to soften boundaries.

        sigma=1.0 at 28×28 gives ~3-pixel transition zone at edges.
        """
        ksize = max(3, 2 * int(3.0 * sigma) + 1)
        if ksize % 2 == 0:
            ksize += 1
        ksize = min(ksize, 15)  # raised from 7 for 56×56 support
        blurred = cv2.GaussianBlur(mask, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        blurred = np.maximum(blurred, mask)  # preserve interior at 1.0
        return np.minimum(blurred, 1.0)

    @staticmethod
    def _soften_mask_adaptive(mask, bbox, hw=28):
        """Size-adaptive Gaussian blur: sigma proportional to object size at 28×28.

        Uses normalized bbox area to select sigma:
          tiny   (area < 0.01): sigma=0   — hard mask, no blur
          small  (area < 0.05): sigma=0.5 — gentle boundary softening
          medium (area < 0.15): sigma=1.0 — moderate softening (≈E-46)
          large  (area ≥ 0.15): sigma=1.5 — stronger boundary softening for large objects

        Interior pixels are always preserved at 1.0 via np.maximum(blurred, mask).
        Bbox-only samples never reach this method (hard rectangles used instead).
        """
        area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        if area < 0.01:
            return mask  # tiny: hard binary
        elif area < 0.05:
            return RealVideoBBoxDataset._soften_mask(mask, sigma=0.5, hw=hw)
        elif area < 0.15:
            return RealVideoBBoxDataset._soften_mask(mask, sigma=1.0, hw=hw)
        else:
            return RealVideoBBoxDataset._soften_mask(mask, sigma=1.5, hw=hw)

    @staticmethod
    def _make_ce_targets(bbox, h=28, w=28):
        """Generate center+extent targets (5, H, W) from normalized bbox.

        Channel 0: Center heatmap — 2D Gaussian peak at bbox center (logit target
                   for BCEWithLogitsLoss, values in [0, 1]).
        Channels 1-4: Per-pixel distance to left/right/top/bottom edges, normalized
                      to [0, 1]. Zero outside bbox; masked out in loss.

        Center Gaussian σ = max(bbox_w, bbox_h) / 4, clamped to [0.5, 4.0] at 28×28.
        """
        x1, y1, x2, y2 = bbox
        # Center coords at 28×28
        cx = (x1 + x2) / 2.0 * w
        cy = (y1 + y2) / 2.0 * h
        bw = max((x2 - x1) * w, 1.0)
        bh = max((y2 - y1) * h, 1.0)
        sigma = float(np.clip(max(bw, bh) / 4.0, 0.5, 4.0))

        y, x_coord = np.mgrid[0:h, 0:w]
        center = np.exp(-((x_coord - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))
        center = center.astype(np.float32)

        # Bbox mask for extent supervision
        ix1 = max(0, int(x1 * w))
        iy1 = max(0, int(y1 * h))
        ix2 = min(w, int(np.ceil(x2 * w)))
        iy2 = min(h, int(np.ceil(y2 * h)))
        extent_mask = np.zeros((h, w), dtype=np.float32)
        if ix2 > ix1 and iy2 > iy1:
            extent_mask[iy1:iy2, ix1:ix2] = 1.0

        # Per-pixel distances to edges (normalized)
        col = np.arange(w, dtype=np.float32).reshape(1, w)  # (1, w)
        row = np.arange(h, dtype=np.float32).reshape(h, 1)  # (h, 1)

        left   = np.clip((col - x1 * w) / w, 0, None)  # (1, w)
        right  = np.clip((x2 * w - col) / w, 0, None)
        top    = np.clip((row - y1 * h) / h, 0, None)  # (h, 1)
        bottom = np.clip((y2 * h - row) / h, 0, None)

        left   = np.broadcast_to(left, (h, w))   * extent_mask
        right  = np.broadcast_to(right, (h, w))  * extent_mask
        top    = np.broadcast_to(top, (h, w))    * extent_mask
        bottom = np.broadcast_to(bottom, (h, w)) * extent_mask

        targets = np.stack([center, left, right, top, bottom], axis=0)  # (5, H, W)
        return targets.astype(np.float32)

    @staticmethod
    def _bbox_to_gaussian_mask(bbox, h, w, sigma_factor=0.3):
        """Generate 2D Gaussian falloff from bbox center.

        sigma_x/y proportional to bbox dimensions × sigma_factor.
        Clamped to [1.0, 5.0] pixels at target resolution.
        Peak=1.0 at center, falls off toward edges and beyond.
        """
        cx = (bbox[0] + bbox[2]) / 2.0 * w
        cy = (bbox[1] + bbox[3]) / 2.0 * h
        bw = max((bbox[2] - bbox[0]) * w, 1.0)
        bh = max((bbox[3] - bbox[1]) * h, 1.0)
        sigma_x = np.clip(bw * sigma_factor, 1.0, 5.0)
        sigma_y = np.clip(bh * sigma_factor, 1.0, 5.0)

        y, x = np.mgrid[0:h, 0:w]
        g = np.exp(-((x - cx) ** 2 / (2 * sigma_x ** 2) + (y - cy) ** 2 / (2 * sigma_y ** 2)))
        g_max = float(g.max())
        return (g / g_max).astype(np.float32)

    @staticmethod
    def _bbox_to_mask(bbox, h, w):
        """Generate binary mask from normalized bbox [x1,y1,x2,y2]."""
        x1 = max(0, int(bbox[0] * w))
        y1 = max(0, int(bbox[1] * h))
        x2 = min(w, int(np.ceil(bbox[2] * w)))
        y2 = min(h, int(np.ceil(bbox[3] * h)))
        mask = np.zeros((h, w), dtype=np.float32)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1.0
        return mask

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
            s = self.jitter_strength
            jitter_brightness = 1.0 + random.uniform(-s, s)
            jitter_contrast = 1.0 + random.uniform(-s, s)
            jitter_saturation = 1.0 + random.uniform(-s, s)

        # ── Scale-adaptive zoom (E-54): consistent across clip ──
        zoom_crop = None
        if self.augment and self.zoom_enabled:
            mid_fi = frame_indices[len(frame_indices) // 2]
            mid_bbox = sample["bbox_map"].get(mid_fi)
            if mid_bbox is not None:
                mid_area = (mid_bbox[2] - mid_bbox[0]) * (mid_bbox[3] - mid_bbox[1])
                if mid_area < 0.01:
                    zoom_prob = self.zoom_prob_tiny
                elif mid_area < 0.05:
                    zoom_prob = self.zoom_prob_small
                elif mid_area < 0.15:
                    zoom_prob = self.zoom_prob_medium
                else:
                    zoom_prob = self.zoom_prob_large
                if random.random() < zoom_prob:
                    zoom_crop = self._compute_zoom_params(mid_bbox, self.zoom_context_factor)

        # ── Background mixing (E-42): shared bg for temporal consistency ──
        do_bg_mix = (
            self.augment and self.bg_mix_prob > 0
            and random.random() < self.bg_mix_prob
        )
        bg_frame = None
        if do_bg_mix:
            exclude_dir = sample.get("frame_dir", "")
            bg_frame = self._pick_background(exclude_dir)

        frames = []
        bboxes = []
        masks_s8 = [] if self.return_mask else None
        masks_s4 = [] if self.mask_hw_s4 is not None else None
        for fi in frame_indices:
            fpath = self._resolve_frame_path(sample, fi)
            img = self._load_or_decode(fpath)

            bbox_before_flip = sample["bbox_map"].get(
                fi, np.array([0, 0, 1, 1], dtype=np.float32)
            ).copy()

            # ── Convert to float tensor (C, H, W) ──────────────────
            if isinstance(img, torch.Tensor):
                img_t = img.float() / 255.0
            else:
                img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

            # ── Scale-adaptive zoom (E-54) ──
            if zoom_crop is not None:
                img_t, bbox_before_flip = self._apply_zoom(img_t, bbox_before_flip, zoom_crop)

            # ── Background mixing: replace background before flip/jitter ──
            if bg_frame is not None:
                full_mask = self._load_full_mask(sample, fi, bbox_before_flip)
                if full_mask.sum() > 0 and full_mask.sum() < full_mask.size:
                    img_t = self._composite_frame(img_t, full_mask, bg_frame)
                # else: mask empty or full-frame → skip compositing

            # ── Horizontal flip ────────────────────────────────────
            bbox = bbox_before_flip.copy()
            if do_flip:
                img_t = torch.flip(img_t, dims=[-1])
                bbox = np.array(
                    [1.0 - bbox[2], bbox[1], 1.0 - bbox[0], bbox[3]],
                    dtype=np.float32,
                )

            # ── Color jitter ────────────────────────────────────────
            if self.augment:
                img_t = self._color_jitter(
                    img_t, jitter_brightness, jitter_contrast, jitter_saturation
                )

            # ── Dense masks ─────────────────────────────────────────
            if self.return_mask:
                mask_s8 = self._load_mask(sample, fi, bbox, hw=28)
                if do_flip and self.dense_target_mode not in ("ce",):
                    mask_s8 = np.fliplr(mask_s8).copy()
                masks_s8.append(torch.from_numpy(mask_s8))

            if self.mask_hw_s4 is not None:
                mask_s4 = self._load_mask(sample, fi, bbox, hw=self.mask_hw_s4)
                if do_flip and self.dense_target_mode not in ("ce",):
                    mask_s4 = np.fliplr(mask_s4).copy()
                masks_s4.append(torch.from_numpy(mask_s4))

            frames.append(img_t)
            bboxes.append(torch.from_numpy(bbox))

        frames = torch.stack(frames, dim=0)  # (T, C, H, W)
        bboxes = torch.stack(bboxes, dim=0)  # (T, 4)
        if masks_s4 is not None:
            masks_s4 = torch.stack(masks_s4, dim=0)  # (T, 56, 56)
            masks_s8 = torch.stack(masks_s8, dim=0)  # (T, 28, 28)
            return frames, bboxes, masks_s4, masks_s8
        if self.return_mask:
            masks_s8 = torch.stack(masks_s8, dim=0)  # (T, 28, 28)
            return frames, bboxes, masks_s8
        return frames, bboxes

    # ── Background mixing (E-42: camouflage-aware augmentation) ──────────

    def _load_full_mask(self, sample, frame_idx, bbox_norm):
        """Load binary mask at target_size resolution for compositing.

        Real masks (MoCA_Mask/CAD) loaded from PNG → target_size.
        Bbox-only samples fall back to rectangle mask.
        """
        mask_path = self._resolve_mask_path(sample, frame_idx)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.resize(mask, (self.target_size, self.target_size),
                                  interpolation=cv2.INTER_AREA)
                return (mask > 0).astype(np.float32)
        return self._bbox_to_mask(bbox_norm, self.target_size, self.target_size)

    # ── Scale-adaptive zoom (E-54) ───────────────────────────────────────

    @staticmethod
    def _compute_zoom_params(mid_bbox, context_factor):
        """Compute zoom crop region from middle-frame bbox.

        Expands the bbox by context_factor (e.g. 2.0 = 2× bbox size),
        clamped to [0, 1]. Returns normalized [cx1, cy1, cx2, cy2] or None.
        """
        cx = (mid_bbox[0] + mid_bbox[2]) / 2.0
        cy = (mid_bbox[1] + mid_bbox[3]) / 2.0
        bw = max(mid_bbox[2] - mid_bbox[0], 0.01)
        bh = max(mid_bbox[3] - mid_bbox[1], 0.01)
        crop_size = max(bw, bh) * context_factor
        half = crop_size / 2.0
        cx1 = max(0.0, cx - half)
        cy1 = max(0.0, cy - half)
        cx2 = min(1.0, cx + half)
        cy2 = min(1.0, cy + half)
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        return np.array([cx1, cy1, cx2, cy2], dtype=np.float32)

    @staticmethod
    def _apply_zoom(img_t, bbox, zoom_crop):
        """Crop + resize a single frame and transform its bbox.

        Args:
            img_t: (C, H, W) float tensor in [0, 1]
            bbox:  (4,) numpy float32 in normalized [0, 1]
            zoom_crop: (4,) numpy float32 [cx1, cy1, cx2, cy2] normalized

        Returns:
            (img_t_resized, bbox_transformed)
        """
        C, H, W = img_t.shape
        cx1, cy1, cx2, cy2 = zoom_crop
        ix1 = max(0, int(cx1 * W))
        iy1 = max(0, int(cy1 * H))
        ix2 = min(W, int(cx2 * W))
        iy2 = min(H, int(cy2 * H))
        cropped = img_t[:, iy1:iy2, ix1:ix2].unsqueeze(0)
        resized = F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)
        img_out = resized.squeeze(0)
        crop_w = cx2 - cx1
        crop_h = cy2 - cy1
        new_bbox = np.array([
            (bbox[0] - cx1) / crop_w,
            (bbox[1] - cy1) / crop_h,
            (bbox[2] - cx1) / crop_w,
            (bbox[3] - cy1) / crop_h,
        ], dtype=np.float32)
        new_bbox = np.clip(new_bbox, 0.0, 1.0)
        return img_out, new_bbox

    def _pick_background(self, exclude_dir):
        """Pick a random frame from a different video as compositing background.

        Returns float tensor (3, H, W) in [0,1], or None on failure.
        """
        for _ in range(50):
            bg_sample = random.choice(self.samples)
            if bg_sample.get("frame_dir", "") == exclude_dir:
                continue
            bg_fi = bg_sample["start_frame"]
            fpath = self._resolve_frame_path(bg_sample, bg_fi)
            if fpath is None:
                continue
            img = self._load_or_decode(fpath)
            if img is None:
                continue
            if isinstance(img, torch.Tensor):
                return img.float() / 255.0
            else:
                img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                return img_t
        return None

    @staticmethod
    def _composite_frame(frame, mask, background):
        """Composite foreground (where mask==1) onto background.

        Args:
            frame: (C, H, W) float tensor in [0,1]
            mask: (H, W) float32 numpy, 1=foreground, 0=background
            background: (C, H, W) float tensor in [0,1]

        Returns composited (C, H, W) float tensor.
        """
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)  # (1, H, W)
        return frame * mask_t + background * (1.0 - mask_t)

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


def collate_video_clips_with_masks(batch):
    frames, bboxes, masks = zip(*batch)
    return torch.stack(frames, dim=0), torch.stack(bboxes, dim=0), torch.stack(masks, dim=0)


def collate_video_clips_ms(batch):
    """Collate multi-scale: (frames, bboxes, masks_s4, masks_s8)."""
    frames, bboxes, masks_s4, masks_s8 = zip(*batch)
    return (
        torch.stack(frames, dim=0),
        torch.stack(bboxes, dim=0),
        torch.stack(masks_s4, dim=0),
        torch.stack(masks_s8, dim=0),
    )
