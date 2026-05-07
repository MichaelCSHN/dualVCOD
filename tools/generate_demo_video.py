"""Phase 2.1 Supplementary Demo Video Generator.

Reads a full MoCA video folder, runs MicroVCOD inference on every frame using
sliding T=5 windows, draws red prediction bounding boxes, and exports an MP4 video.

Usage:
    python tools/generate_demo_video.py --video flounder_6
    python tools/generate_demo_video.py --video seal --fps 30
    python tools/generate_demo_video.py --video cuttlefish_4 --output demo_cuttlefish.mp4
    python tools/generate_demo_video.py --video chameleon --no-gt
"""

import sys
import os
import argparse
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import torch
import numpy as np
from collections import defaultdict

from src.model import MicroVCOD

# ── Config ────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.45, 0)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_greenvcod_box_miou.pth")
MOCA_ROOT = r"D:\ML\COD_datasets\MoCA"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports", "paper_assets")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# BBox colors
GT_COLOR = (0, 200, 0)     # Green (BGR for cv2)
PRED_COLOR = (32, 0, 224)   # Red (BGR)


def parse_moca_csv(csv_path):
    """Parse MoCA VIA annotations.csv → {video_name: {frame_idx: [x, y, w, h]}}."""
    import csv
    import json
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
            if coords[0] == 2:
                annotations[video][frame_idx] = [
                    float(coords[1]),
                    float(coords[2]),
                    float(coords[3]),
                    float(coords[4]),
                ]
    return dict(annotations)


def xywh_to_xyxy(box, W, H):
    """Convert [x, y, w, h] to normalized [x1, y1, x2, y2]."""
    x, y, w, h = box
    return np.array([x / W, y / H, (x + w) / W, (y + h) / H], dtype=np.float32)


def load_model():
    print(f"  Loading checkpoint: {os.path.basename(CHECKPOINT_PATH)}")
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model = MicroVCOD(T=5, pretrained_backbone=False).to(DEVICE)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"    epoch={state['epoch']}  mIoU={state['miou']:.4f}  R@0.5={state['recall']:.4f}")
    return model, state


def preprocess_frame(img_bgr, target_size=224):
    """Resize and normalize a BGR frame → (1, C, H, W) tensor."""
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size, target_size))
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)  # (1, C, H, W)


@torch.no_grad()
def predict_window(model, frame_tensors):
    """Predict bboxes for a T=5 window. frame_tensors: list of 5 tensors (1,C,H,W)."""
    clip = torch.cat(frame_tensors, dim=0).unsqueeze(0).to(DEVICE)  # (1, T, C, H, W)
    with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
        pred = model(clip)
    return pred[0].float().cpu().numpy()  # (T, 4)


def draw_bbox(img, bbox_norm, color, thickness=3, label=None):
    """Draw a normalized bbox on a BGR image. Returns the image."""
    H, W = img.shape[:2]
    x1 = int(bbox_norm[0] * W)
    y1 = int(bbox_norm[1] * H)
    x2 = int(bbox_norm[2] * W)
    y2 = int(bbox_norm[3] * H)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def generate_demo_video(video_name, output_name=None, fps=30, show_gt=True,
                        target_size=224, codec="avc1"):
    """Generate a demo MP4 video for a MoCA video sequence.

    Args:
        video_name: name of the MoCA video folder (e.g. 'flounder_6')
        output_name: output MP4 filename (default: 'demo_{video_name}.mp4')
        fps: output video framerate
        show_gt: overlay ground truth (green) in addition to prediction (red)
        target_size: frame resize dimension (square)
        codec: FourCC codec ('avc1' for H.264, 'mp4v' for MPEG-4)
    """
    video_dir = os.path.join(MOCA_ROOT, "JPEGImages", video_name)
    if not os.path.isdir(video_dir):
        print(f"  ERROR: Video directory not found: {video_dir}")
        print(f"  Available videos (first 20):")
        jpeg_dir = os.path.join(MOCA_ROOT, "JPEGImages")
        for d in sorted(os.listdir(jpeg_dir))[:20]:
            print(f"    {d}")
        return None

    if output_name is None:
        output_name = f"demo_{video_name}.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    # ── Load model ──
    model, state = load_model()

    # ── Load GT annotations if available ──
    gt_annotations = {}
    if show_gt:
        csv_path = os.path.join(MOCA_ROOT, "Annotations", "annotations.csv")
        if os.path.isfile(csv_path):
            all_ann = parse_moca_csv(csv_path)
            gt_annotations = all_ann.get(video_name, {})
            if gt_annotations:
                print(f"  GT annotations loaded: {len(gt_annotations)} frames")
            else:
                print(f"  [WARN] No GT annotations found for '{video_name}'")

    # ── Discover frames ──
    frame_files = sorted([
        f for f in os.listdir(video_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if not frame_files:
        print(f"  ERROR: No frames found in {video_dir}")
        return None

    # Parse frame indices from filenames
    frame_indices = []
    for fname in frame_files:
        num_str = os.path.splitext(fname)[0]
        if num_str.isdigit():
            frame_indices.append(int(num_str))
        else:
            # Try extracting number from filename
            import re
            nums = re.findall(r"\d+", num_str)
            if nums:
                frame_indices.append(int(nums[-1]))

    # Sort frames by index
    paired = sorted(zip(frame_indices, frame_files), key=lambda x: x[0])
    frame_indices, frame_files = zip(*paired) if paired else ([], [])
    frame_indices = list(frame_indices)
    frame_files = list(frame_files)

    n_frames = len(frame_files)
    print(f"  Video: {video_name}")
    print(f"  Frames: {n_frames}")
    print(f"  Frame range: {frame_indices[0]} – {frame_indices[-1]}")
    print(f"  Output FPS: {fps}")
    print(f"  GT overlay: {'ON' if show_gt and gt_annotations else 'OFF'}")
    print()

    # ── Read first frame to get dimensions ──
    first_frame = cv2.imread(os.path.join(video_dir, frame_files[0]))
    if first_frame is None:
        print(f"  ERROR: Cannot read frames from {video_dir}")
        return None
    H_orig, W_orig = first_frame.shape[:2]
    print(f"  Original resolution: {W_orig}x{H_orig}")
    print(f"  Model input: {target_size}x{target_size}")

    # ── Initialize video writer ──
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W_orig, H_orig))
    if not writer.isOpened():
        # Fallback to MPEG-4
        print(f"  Codec '{codec}' failed, trying 'mp4v'...")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W_orig, H_orig))
    if not writer.isOpened():
        print(f"  ERROR: Cannot open video writer for {output_path}")
        return None

    # ── Pre-load all frames as tensors ──
    print("  Pre-loading and preprocessing frames...")
    frame_tensors = []
    for fname in frame_files:
        img = cv2.imread(os.path.join(video_dir, fname))
        if img is None:
            frame_tensors.append(None)
        else:
            tensor = preprocess_frame(img, target_size)
            frame_tensors.append(tensor)
    print(f"  Pre-loaded {len(frame_tensors)} frames")

    # ── Sliding-window inference ──
    # For each position, predict using window [i-2, i-1, i, i+1, i+2] (centered)
    # For edge frames, pad by mirroring
    T = 5
    half = T // 2

    print("  Running sliding-window inference...")
    t_start = time.time()

    predictions = {}  # frame_idx → normalized bbox

    for i in range(n_frames):
        # Build T=5 window centered on frame i
        window_indices = []
        for offset in range(-half, half + 1):
            src_idx = i + offset
            if src_idx < 0:
                src_idx = -src_idx  # mirror left edge
            elif src_idx >= n_frames:
                src_idx = 2 * (n_frames - 1) - src_idx  # mirror right edge
            src_idx = max(0, min(n_frames - 1, src_idx))
            window_indices.append(src_idx)

        window_tensors = [frame_tensors[wi] for wi in window_indices]
        if any(wt is None for wt in window_tensors):
            predictions[frame_indices[i]] = np.array([0, 0, 1, 1], dtype=np.float32)
            continue

        pred = predict_window(model, window_tensors)
        # Take the middle frame prediction (index 2)
        predictions[frame_indices[i]] = pred[half]

        if (i + 1) % 50 == 0 or i == 0 or i == n_frames - 1:
            elapsed = time.time() - t_start
            fps_infer = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"    Frame {i+1}/{n_frames}  ({100*(i+1)/n_frames:.0f}%)  "
                  f"{fps_infer:.1f} fps inference")

    total_time = time.time() - t_start
    print(f"  Inference complete: {total_time:.1f}s ({n_frames/total_time:.1f} fps avg)")

    # ── Render output video ──
    print("  Rendering output video...")
    t_render = time.time()

    for i, fname in enumerate(frame_files):
        img = cv2.imread(os.path.join(video_dir, fname))
        if img is None:
            continue

        fi = frame_indices[i]

        # Draw prediction (red)
        if fi in predictions:
            pred_bbox = predictions[fi]
            # Only draw if prediction is meaningful (not covering whole frame)
            pred_area = (pred_bbox[2] - pred_bbox[0]) * (pred_bbox[3] - pred_bbox[1])
            if 0.001 < pred_area < 0.99:
                draw_bbox(img, pred_bbox, PRED_COLOR, thickness=3, label="Ours")

        # Draw GT (green, dashed style using thinner line)
        if show_gt and fi in gt_annotations:
            gt_box = gt_annotations[fi]
            gt_norm = xywh_to_xyxy(gt_box, W_orig, H_orig)
            draw_bbox(img, gt_norm, GT_COLOR, thickness=2, label="GT")

        # Add frame counter overlay
        cv2.putText(img, f"Frame: {fi}  |  {video_name}",
                    (10, H_orig - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(img)

    writer.release()
    render_time = time.time() - t_render

    output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    duration_sec = n_frames / fps

    print(f"\n  Video saved: {output_path}")
    print(f"  Size: {output_size_mb:.1f} MB")
    print(f"  Duration: {duration_sec:.1f}s ({n_frames} frames @ {fps}fps)")
    print(f"  Resolution: {W_orig}x{H_orig}")
    print(f"  Total processing: {total_time + render_time:.1f}s")
    print(f"  Real-time factor: {(total_time + render_time) / duration_sec:.1f}x")

    return output_path


def list_available_videos():
    """Print available MoCA videos with frame counts."""
    jpeg_dir = os.path.join(MOCA_ROOT, "JPEGImages")
    if not os.path.isdir(jpeg_dir):
        print("MoCA JPEGImages directory not found.")
        return

    videos = []
    for d in sorted(os.listdir(jpeg_dir)):
        dpath = os.path.join(jpeg_dir, d)
        if os.path.isdir(dpath):
            n_frames = len([f for f in os.listdir(dpath)
                           if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            videos.append((d, n_frames))

    print(f"\nAvailable MoCA videos ({len(videos)} total):")
    print(f"{'Video Name':30s}  {'Frames':>8s}  {'Duration@30fps':>12s}")
    print("-" * 55)
    for name, n in videos:
        dur = n / 30.0
        print(f"  {name:30s}  {n:8d}  {dur:10.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="MicroVCOD Demo Video Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/generate_demo_video.py --video flounder_6
  python tools/generate_demo_video.py --video seal --fps 30
  python tools/generate_demo_video.py --video cuttlefish_4 --fps 60
  python tools/generate_demo_video.py --video chameleon --no-gt
  python tools/generate_demo_video.py --list
        """
    )
    parser.add_argument("--video", type=str, default=None,
                        help="MoCA video name (e.g. 'flounder_6', 'seal')")
    parser.add_argument("--output", type=str, default=None,
                        help="Output MP4 filename (default: demo_<video>.mp4)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Output video FPS (default: 30)")
    parser.add_argument("--no-gt", action="store_true",
                        help="Hide ground truth boxes")
    parser.add_argument("--list", action="store_true",
                        help="List available MoCA videos and exit")
    parser.add_argument("--codec", type=str, default="avc1",
                        help="Video codec FourCC (default: avc1)")
    args = parser.parse_args()

    print("=" * 70)
    print("  MicroVCOD — Supplementary Demo Video Generator")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    if args.list:
        list_available_videos()
        return 0

    if args.video is None:
        print("\n  Please specify a video with --video, or use --list to see available videos.\n")
        list_available_videos()
        return 1

    result = generate_demo_video(
        video_name=args.video,
        output_name=args.output,
        fps=args.fps,
        show_gt=not args.no_gt,
        codec=args.codec,
    )

    if result:
        print(f"\n  Demo video ready: {result}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
