"""Micro-benchmark: compare frame decode pipelines.

Tests 4 pipelines on 10,000+ randomly sampled training frames:
  1. cv2.imread + cvtColor(BGR2RGB) + resize(224)     [current]
  2. cv2.imread + resize(224), no BGR2RGB              [no color conv]
  3. cv2.imread only (pre-resized 224 JPEG)            [pre-resized]
  4. torchvision.io.read_file + decode_jpeg + resize   [torchvision]

Reports: total_time, frames/s, ms/frame, per-pipeline breakdown.
"""

import os
import sys
import time
import random
import argparse
import cv2
import numpy as np
import torch
from torchvision import io as tvio
from torchvision.transforms import functional as F_tv

ROOTS = [
    r"C:\datasets\MoCA",
    r"C:\datasets\MoCA_Mask",
    r"C:\datasets\CamouflagedAnimalDataset",
]

TARGET_SIZE = 224
N_FRAMES = 10_000


def collect_frame_paths(roots):
    """Walk dataset roots and collect all image frame paths (not masks/GT)."""
    paths = []
    for root in roots:
        if not os.path.isdir(root):
            print(f"  SKIP (not found): {root}")
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip GT/mask directories
            if os.path.basename(dirpath) in ("GT", "groundtruth", "Annotations"):
                continue
            for fn in filenames:
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    paths.append(os.path.join(dirpath, fn))
    return paths


def pipeline_current(fpath):
    """cv2.imread + cvtColor(BGR2RGB) + resize(224)."""
    img = cv2.imread(fpath)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))
    return img


def pipeline_no_color(fpath):
    """cv2.imread + resize(224), keep BGR."""
    img = cv2.imread(fpath)
    if img is None:
        return None
    img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))
    return img


def pipeline_preresized(fpath):
    """cv2.imread only — frame already 224x224."""
    img = cv2.imread(fpath)
    return img


def pipeline_torchvision_full(fpath):
    """torchvision.io.read_file + decode_image + resize(224) — RGB output."""
    data = tvio.read_file(fpath)
    img = tvio.decode_image(data)
    img = F_tv.resize(img, [TARGET_SIZE, TARGET_SIZE])
    return img


def pipeline_torchvision_preresized(fpath):
    """torchvision.io.read_file + decode_image only — frame already 224."""
    data = tvio.read_file(fpath)
    img = tvio.decode_image(data)
    return img


PIPELINES = {
    "1_current_imread_cvt_resize": pipeline_current,
    "2_imread_resize_no_cvt": pipeline_no_color,
    "3_preresized_imread_only": pipeline_preresized,
    "4_torchvision_full_decode_resize": pipeline_torchvision_full,
    "5_torchvision_preresized_decode_only": pipeline_torchvision_preresized,
}


def benchmark(paths, n_frames, warmup=50):
    """Run each pipeline on the same frames, report timing."""
    random.shuffle(paths)
    paths = paths[:n_frames]
    results = {}

    for name, fn in PIPELINES.items():
        # Warmup
        for p in paths[:warmup]:
            fn(p)

        t0 = time.perf_counter()
        ok = 0
        for p in paths:
            img = fn(p)
            if img is not None:
                ok += 1
        elapsed = time.perf_counter() - t0

        n = len(paths)
        results[name] = {
            "total_time_s": round(elapsed, 3),
            "frames": ok,
            "frames_per_s": round(ok / elapsed, 1) if elapsed > 0 else 0,
            "ms_per_frame": round(elapsed / ok * 1000, 3) if ok > 0 else 0,
            "has_decode": "imread" in name or "torchvision" in name,
            "has_resize": "preresized" not in name,
            "has_bgr2rgb": "current" in name,
        }
        print(f"  {name}: {ok}/{n} ok, {elapsed:.2f}s, {ok/elapsed:.1f} fps, {elapsed/ok*1000:.2f} ms/frame")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_FRAMES)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 70)
    print("Micro-benchmark: frame decode pipelines")
    print(f"  target_size={TARGET_SIZE}, n_frames={args.n}, warmup={args.warmup}")
    print("=" * 70)

    # Collect
    print("\n[1] Collecting frame paths ...")
    paths = collect_frame_paths(ROOTS)
    print(f"  Found {len(paths)} total frames")
    if len(paths) < args.n:
        print(f"  WARNING: only {len(paths)} available, using all")
        args.n = len(paths)

    # Benchmark
    print(f"\n[2] Benchmarking {args.n} frames per pipeline ...")
    results = benchmark(paths, args.n, args.warmup)

    # Summary
    print("\n[3] Summary")
    print("-" * 70)
    header = f"{'Pipeline':<38} {'fps':>8} {'ms/f':>8} {'time_s':>8}"
    print(header)
    print("-" * 70)
    baseline_fps = None
    for name in PIPELINES:
        r = results[name]
        marker = ""
        if baseline_fps is None:
            baseline_fps = r["frames_per_s"]
            marker = " ← baseline"
        speedup = r["frames_per_s"] / baseline_fps if baseline_fps > 0 else 1.0
        print(f"{name:<38} {r['frames_per_s']:>8.1f} {r['ms_per_frame']:>8.2f} {r['total_time_s']:>8.1f}  ({speedup:.2f}x){marker}")

    # Save if requested
    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
