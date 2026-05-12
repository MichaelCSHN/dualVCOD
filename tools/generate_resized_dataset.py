"""Generate pre-resized 224x224 JPEG dataset from original frames.

Reads each frame with cv2, resizes to 224x224, saves as JPEG quality=95.
Mirrors original directory structure under C:\datasets_224.
Skips mask/GT/annotation directories.

Frame directories by dataset:
  MoCA:     {root}/JPEGImages/{video}/*.jpg  (1280x720)
  MoCA_Mask:{root}/TrainDataset_per_sq/{video}/Imgs/*.jpg  (varies)
  CAD:      {root}/{animal}/frames/*.png  (varies)

Output: C:\datasets_224\{dataset}\{relative_path}.jpg
"""

import os
import sys
import cv2
import time
import argparse
from multiprocessing import Pool, cpu_count
from collections import deque

SOURCE_ROOTS = [
    r"C:\datasets\MoCA",
    r"C:\datasets\MoCA_Mask",
    r"C:\datasets\CamouflagedAnimalDataset",
]

OUTPUT_ROOT = r"C:\datasets_224"
JPEG_QUALITY = 95
TARGET_SIZE = 224

SKIP_DIRS = {"GT", "groundtruth", "Annotations", "annotations"}


def collect_frames(root):
    """Collect (src_path, rel_path_stem) for all frames, skipping masks/GT."""
    frames = []
    root_name = os.path.basename(root)

    for dirpath, dirnames, filenames in os.walk(root):
        # Skip mask/annotation directories
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS
                       and "groundtruth" not in d.lower()
                       and "gt" != d.lower()]

        for fn in filenames:
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            src = os.path.join(dirpath, fn)
            rel = os.path.relpath(src, root)
            stem = os.path.splitext(rel)[0]
            out_name = stem + ".jpg"  # Always JPEG output
            frames.append((src, out_name, root_name))

    return frames


def process_one(args):
    """Read, resize, write one frame. Returns (src, ok, err)."""
    src, out_rel, root_name = args
    dst = os.path.join(OUTPUT_ROOT, root_name, out_rel)

    # Skip if already exists and is non-zero
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return (src, True, "skipped")

    os.makedirs(os.path.dirname(dst), exist_ok=True)

    img = cv2.imread(src)
    if img is None:
        return (src, False, "imread_failed")

    img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LINEAR)

    # Write directly (cv2 needs extension to determine codec)
    ok = cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return (src, False, "imwrite_failed")

    return (src, True, "ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=min(8, cpu_count()))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("Pre-resize dataset: 224x224 JPEG (quality={})".format(JPEG_QUALITY))
    print(f"  Source roots: {SOURCE_ROOTS}")
    print(f"  Output root:  {OUTPUT_ROOT}")
    print(f"  Workers:      {args.workers}")
    print("=" * 70)

    # Collect
    all_frames = []
    for root in SOURCE_ROOTS:
        if not os.path.isdir(root):
            print(f"  SKIP (not found): {root}")
            continue
        frames = collect_frames(root)
        print(f"  {os.path.basename(root)}: {len(frames)} frames")
        all_frames.extend(frames)

    print(f"\n  Total: {len(all_frames)} frames")
    if args.dry_run:
        print("  DRY RUN — showing first 5:")
        for src, out_rel, root_name in all_frames[:5]:
            dst = os.path.join(OUTPUT_ROOT, root_name, out_rel)
            print(f"    {src}\n      → {dst}")
        return

    # Process
    t0 = time.perf_counter()
    ok = skipped = failed = 0

    with Pool(args.workers) as pool:
        for src, success, msg in pool.imap_unordered(process_one, all_frames, chunksize=50):
            if success:
                if msg == "skipped":
                    skipped += 1
                else:
                    ok += 1
            else:
                failed += 1
                print(f"  FAIL [{msg}]: {src}")

            total = ok + skipped + failed
            if total % 2000 == 0 or total == len(all_frames):
                elapsed = time.perf_counter() - t0
                fps = ok / elapsed if elapsed > 0 else 0
                print(f"  [{total}/{len(all_frames)}] ok={ok} skip={skipped} fail={failed}  {fps:.0f} fps  {elapsed:.0f}s elapsed")

    elapsed = time.perf_counter() - t0
    print(f"\n  Done: {ok} written, {skipped} skipped, {failed} failed  ({elapsed:.0f}s, {ok/elapsed:.0f} fps)")

    # Size comparison
    src_size = sum(os.path.getsize(f[0]) for f in all_frames if os.path.exists(f[0]))
    dst_size = 0
    for _, out_rel, root_name in all_frames:
        dst = os.path.join(OUTPUT_ROOT, root_name, out_rel)
        if os.path.exists(dst):
            dst_size += os.path.getsize(dst)
    print(f"  Source total: {src_size/1e9:.2f} GB")
    print(f"  Output total: {dst_size/1e9:.2f} GB")
    print(f"  Ratio:        {dst_size/src_size:.2f}x" if src_size > 0 else "")


if __name__ == "__main__":
    main()
