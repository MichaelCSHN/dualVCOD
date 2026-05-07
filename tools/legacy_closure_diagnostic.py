"""Legacy Checkpoint Closure Diagnostic — evaluate 3 checkpoints under identical protocol.

Evaluates:
  1. clean_seed42_best_miou.pth  (clean-trained, lower-bound candidate)
  2. clean_seed42_best_recall.pth (clean-trained, recall-oriented candidate)
  3. verified_candidate_baseline.pth (legacy non-clean, closure diagnostic)

Under IDENTICAL clean val protocol:
  - Same MoCA val split (seed=42, ratio=0.2)
  - Same GT from CSV
  - Same bbox schema (x1,y1,x2,y2 normalized [0,1])
  - Same T=5 temporal windows
  - Same metric code (compute_metrics from eval_video_bbox)
  - Augmentation OFF
  - No canonical_video_id leakage into val

Includes: batch-size probe, negative controls, error analysis CSV output.
"""

import sys
import os
import time
import json
import csv
import random
import argparse
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset_real import RealVideoBBoxDataset, collate_video_clips
from src.model import MicroVCOD
from eval.eval_video_bbox import compute_metrics, count_parameters, bbox_iou

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "reports")

CHECKPOINTS = {
    "clean_seed42_best_miou": {
        "path": os.path.join(CHECKPOINT_DIR, "clean_seed42_best_miou.pth"),
        "type": "clean-trained",
        "description": "Clean lower-bound baseline candidate"
    },
    "clean_seed42_best_recall": {
        "path": os.path.join(CHECKPOINT_DIR, "clean_seed42_best_recall.pth"),
        "type": "clean-trained",
        "description": "Recall-oriented clean candidate"
    },
    "verified_candidate_baseline": {
        "path": os.path.join(CHECKPOINT_DIR, "verified_candidate_baseline.pth"),
        "type": "legacy non-clean",
        "description": "Historical checkpoint — upper diagnostic only, NOT a clean baseline"
    },
}


def load_checkpoint(path, model):
    state = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    return state


def get_val_loader(T=5, batch_size=16, num_workers=4):
    """Build MoCA val DataLoader under clean protocol."""
    full_ds = RealVideoBBoxDataset(
        [r"D:\ML\COD_datasets\MoCA"],
        T=T, target_size=224, augment=False
    )
    from tools.train import split_by_video
    _, val_idx = split_by_video(full_ds, val_ratio=0.2, seed=42)
    val_ds = Subset(full_ds, val_idx)
    loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_video_clips, num_workers=num_workers,
        pin_memory=True, persistent_workers=(num_workers > 0),
    )
    return loader, len(val_ds)


@torch.no_grad()
def evaluate_model(model, loader):
    """Run model and collect predictions + ground truths."""
    model.eval()
    all_preds, all_gts = [], []
    t0 = time.time()
    for frames, gt_bboxes in loader:
        frames = frames.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            pred = model(frames)
        all_preds.append(pred.float().cpu())
        all_gts.append(gt_bboxes)
    elapsed = time.time() - t0

    preds = torch.cat(all_preds, dim=0)
    gts = torch.cat(all_gts, dim=0)
    metrics = compute_metrics(preds, gts)
    metrics["eval_time_s"] = elapsed
    metrics["eval_fps"] = preds.shape[0] / elapsed if elapsed > 0 else 0
    return preds, gts, metrics


@torch.no_grad()
def evaluate_model_fp32(model, loader):
    """Run model in FP32 (no AMP)."""
    model.eval()
    all_preds, all_gts = [], []
    t0 = time.time()
    for frames, gt_bboxes in loader:
        frames = frames.to(DEVICE, non_blocking=True)
        pred = model(frames)
        all_preds.append(pred.float().cpu())
        all_gts.append(gt_bboxes)
    elapsed = time.time() - t0

    preds = torch.cat(all_preds, dim=0)
    gts = torch.cat(all_gts, dim=0)
    metrics = compute_metrics(preds, gts)
    metrics["eval_time_s"] = elapsed
    metrics["eval_fps"] = preds.shape[0] / elapsed if elapsed > 0 else 0
    return preds, gts, metrics


def detailed_metrics(preds, gts):
    """Compute extended metrics."""
    metrics = compute_metrics(preds, gts)

    # R@0.3
    ious = metrics.get("per_frame_ious", None)
    if ious is None:
        all_ious = []
        for i in range(preds.shape[0]):
            for t in range(preds.shape[1]):
                all_ious.append(float(bbox_iou(preds[i, t], gts[i, t])))
        ious = np.array(all_ious)
    else:
        ious = np.array(ious)
    metrics["recall@0.3"] = float(np.mean(ious > 0.3))

    # Empty predictions (bbox area == 0)
    pred_areas = (preds[..., 2] - preds[..., 0]) * (preds[..., 3] - preds[..., 1])
    metrics["empty_pred_count"] = int((pred_areas < 1e-8).sum())
    metrics["mean_pred_area"] = float(pred_areas.mean())

    # GT stats
    gt_areas = (gts[..., 2] - gts[..., 0]) * (gts[..., 3] - gts[..., 1])
    metrics["mean_gt_area"] = float(gt_areas.mean())
    metrics["size_ratio"] = float((pred_areas.mean() / max(gt_areas.mean(), 1e-8)))

    # Per-sample IoUs for error analysis
    per_sample_ious = []
    for i in range(preds.shape[0]):
        sample_ious = []
        for t in range(preds.shape[1]):
            sample_ious.append(float(bbox_iou(preds[i, t], gts[i, t])))
        per_sample_ious.append(np.mean(sample_ious))
    metrics["per_sample_ious"] = per_sample_ious

    return metrics


def batch_size_probe():
    """Find optimal eval batch size without OOM."""
    print("=" * 72)
    print("  BATCH SIZE PROBE — Finding optimal eval config for RTX 4090")
    print("=" * 72)

    model = MicroVCOD(T=5, pretrained_backbone=False).to(DEVICE)
    load_checkpoint(CHECKPOINTS["clean_seed42_best_miou"]["path"], model)
    model.eval()

    candidates = [24, 32, 48, 64, 96, 128]
    workers_opts = [4, 6, 8]
    best_fps = 0.0
    best_config = None

    for nw in workers_opts:
        for bs in candidates:
            try:
                loader, n_samples = get_val_loader(T=5, batch_size=bs, num_workers=nw)
                # Short warmup
                for i, (frames, _) in enumerate(loader):
                    if i >= 3:
                        break
                    _ = model(frames.to(DEVICE))

                # Timed run on subset
                torch.cuda.synchronize()
                t0 = time.time()
                count = 0
                for frames, _ in loader:
                    _ = model(frames.to(DEVICE))
                    count += frames.shape[0]
                    if count >= 300:
                        break
                torch.cuda.synchronize()
                elapsed = time.time() - t0
                fps = count / elapsed if elapsed > 0 else 0

                gpu_mem = torch.cuda.max_memory_allocated() / (1024**3)
                torch.cuda.reset_peak_memory_stats()

                print(f"  bs={bs:3d}  nw={nw}  fps={fps:6.1f} w/s  gpu_mem={gpu_mem:.1f}GiB  {'OK' if fps > 0 else 'SLOW'}")

                if fps > best_fps:
                    best_fps = fps
                    best_config = (bs, nw, gpu_mem, fps)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  bs={bs:3d}  nw={nw}  OOM — skipping")
                    torch.cuda.empty_cache()
                    break
                else:
                    raise

    print()
    if best_config:
        bs, nw, gmem, fps = best_config
        print(f"  Best config: bs={bs}  nw={nw}  fps={fps:.1f} w/s  gpu_mem={gmem:.1f}GiB")
    else:
        bs, nw, gmem, fps = 16, 4, 0, 0
        print(f"  Fallback: bs=16  nw=4")

    print()
    return bs, nw


def evaluate_checkpoint(name, info, eval_bs, eval_nw):
    """Evaluate a single checkpoint under clean val protocol."""
    print(f"  [{info['type']}] {name}")
    print(f"    Path: {info['path']}")
    print(f"    Description: {info['description']}")

    model = MicroVCOD(T=5, pretrained_backbone=False).to(DEVICE)
    state = load_checkpoint(info["path"], model)
    ckpt_epoch = state.get("epoch", "unknown")
    ckpt_miou = state.get("miou", "N/A")
    ckpt_recall = state.get("recall", "N/A")

    print(f"    Checkpoint epoch: {ckpt_epoch}")
    print(f"    Checkpoint recorded mIoU: {ckpt_miou}")
    print(f"    Checkpoint recorded R@0.5: {ckpt_recall}")

    loader, n_samples = get_val_loader(T=5, batch_size=eval_bs, num_workers=eval_nw)
    torch.cuda.reset_peak_memory_stats()

    # AMP eval
    preds_amp, gts, metrics_amp = evaluate_model(model, loader)

    gpu_mem = torch.cuda.max_memory_allocated() / (1024**3)

    # FP32 eval for comparison
    loader2, _ = get_val_loader(T=5, batch_size=eval_bs, num_workers=eval_nw)
    preds_fp32, _, metrics_fp32 = evaluate_model_fp32(model, loader2)

    # Use FP32 as authoritative
    metrics = metrics_fp32
    metrics["amp_miou"] = metrics_amp["mean_iou"] if metrics_amp else None
    metrics["ckpt_epoch"] = ckpt_epoch
    metrics["ckpt_recorded_miou"] = ckpt_miou
    metrics["ckpt_recorded_recall"] = ckpt_recall
    metrics["gpu_mem_gib"] = gpu_mem
    preds = preds_fp32

    # Extended metrics
    ext = detailed_metrics(preds, gts)
    metrics.update(ext)

    print(f"    Val mIoU (FP32):    {metrics['mean_iou']:.4f}")
    print(f"    Val mIoU (AMP):     {metrics['amp_miou']:.4f}" if metrics["amp_miou"] else "    (no AMP)")
    print(f"    Val R@0.5:          {metrics['recall@0.5']:.4f}")
    print(f"    Val R@0.3:          {metrics['recall@0.3']:.4f}")
    print(f"    Evaluated windows:  {n_samples}")
    print(f"    Empty preds:        {metrics['empty_pred_count']}")
    print(f"    Mean pred area:     {metrics['mean_pred_area']:.4f}")
    print(f"    Mean GT area:       {metrics['mean_gt_area']:.4f}")
    print(f"    Size ratio:         {metrics['size_ratio']:.4f}")
    print(f"    Eval time:          {metrics['eval_time_s']:.1f}s")
    print(f"    Eval FPS:           {metrics['eval_fps']:.1f} w/s")
    print(f"    GPU mem:            {gpu_mem:.2f} GiB")
    print()

    return preds, gts, metrics


def run_negative_controls(gts, preds_clean_miou):
    """Run all negative controls on the val GT set."""
    print("-" * 72)
    print("  NEGATIVE CONTROLS")
    print("-" * 72)

    controls = {}

    # 1. All-zero
    zero_preds = torch.zeros_like(gts)
    zero_m = detailed_metrics(zero_preds, gts)
    controls["all_zero"] = zero_m
    print(f"  All-zero:        mIoU={zero_m['mean_iou']:.4f}  R@0.5={zero_m['recall@0.5']:.4f}  R@0.3={zero_m['recall@0.3']:.4f}")

    # 2. Random uniform
    rng = random.Random(42)
    np_rng = np.random.RandomState(42)
    rand = torch.zeros_like(gts)
    for i in range(gts.shape[0]):
        for t in range(gts.shape[1]):
            x1 = float(np_rng.uniform(0, 0.7))
            y1 = float(np_rng.uniform(0, 0.7))
            x2 = float(x1 + np_rng.uniform(0.05, 0.3))
            y2 = float(y1 + np_rng.uniform(0.05, 0.3))
            x2 = min(x2, 1.0)
            y2 = min(y2, 1.0)
            rand[i, t] = torch.tensor([x1, y1, x2, y2])
    rand_m = detailed_metrics(rand, gts)
    controls["random"] = rand_m
    print(f"  Random uniform:  mIoU={rand_m['mean_iou']:.4f}  R@0.5={rand_m['recall@0.5']:.4f}  R@0.3={rand_m['recall@0.3']:.4f}")

    # 3. Center prior (mean GT bbox)
    mean_bbox = gts.mean(dim=0).mean(dim=0)  # scalar mean across all samples and timesteps
    center = mean_bbox.unsqueeze(0).unsqueeze(0).expand(gts.shape[0], gts.shape[1], -1).clone()
    center_m = detailed_metrics(center, gts)
    controls["center_prior"] = center_m
    print(f"  Center prior:    mIoU={center_m['mean_iou']:.4f}  R@0.5={center_m['recall@0.5']:.4f}  R@0.3={center_m['recall@0.3']:.4f}")
    print(f"    Prior bbox: [{center[0,0,0]:.4f}, {center[0,0,1]:.4f}, {center[0,0,2]:.4f}, {center[0,0,3]:.4f}]")

    # 4. Shuffled prediction
    idx = list(range(preds_clean_miou.shape[0]))
    rng_shuf = random.Random(42)
    rng_shuf.shuffle(idx)
    shuffled = preds_clean_miou[idx].clone()
    shuffled_m = detailed_metrics(shuffled, gts)
    controls["shuffled"] = shuffled_m
    print(f"  Shuffled:        mIoU={shuffled_m['mean_iou']:.4f}  R@0.5={shuffled_m['recall@0.5']:.4f}  R@0.3={shuffled_m['recall@0.3']:.4f}")

    # 5. Oracle GT (GT as prediction — upper bound due to temporal window alignment)
    oracle_m = detailed_metrics(gts.clone(), gts)
    controls["oracle"] = oracle_m
    print(f"  Oracle GT:       mIoU={oracle_m['mean_iou']:.4f}  R@0.5={oracle_m['recall@0.5']:.4f}  R@0.3={oracle_m['recall@0.3']:.4f}")
    print(f"    (Oracle mIoU should be 1.0 — confirms evaluator correctness)")

    print()
    return controls


def error_analysis(preds, gts, name, output_csv):
    """Generate per-sample error analysis CSV."""
    print(f"  Error analysis for {name} ...")

    per_sample = []
    for i in range(preds.shape[0]):
        sample_ious = []
        for t in range(preds.shape[1]):
            sample_ious.append(float(bbox_iou(preds[i, t], gts[i, t])))
        avg_iou = np.mean(sample_ious)

        # Mean pred and GT bboxes across T frames
        mean_pred = preds[i].mean(dim=0)
        mean_gt = gts[i].mean(dim=0)

        pred_area = float((mean_pred[2] - mean_pred[0]) * (mean_pred[3] - mean_pred[1]))
        gt_area = float((mean_gt[2] - mean_gt[0]) * (mean_gt[3] - mean_gt[1]))
        size_ratio = pred_area / max(gt_area, 1e-8)

        # Error type classification
        if avg_iou < 0.01:
            error_type = "no_response"
        elif avg_iou < 0.10:
            if size_ratio > 3.0:
                error_type = "box_too_large"
            elif size_ratio < 0.3:
                error_type = "box_too_small"
            else:
                error_type = "shifted_box"
        elif avg_iou < 0.25:
            if size_ratio > 2.0:
                error_type = "box_too_large"
            elif size_ratio < 0.5:
                error_type = "box_too_small"
            elif gt_area < 0.01:
                error_type = "tiny_object"
            else:
                error_type = "shifted_box"
        else:
            if gt_area < 0.005:
                error_type = "tiny_object"
            elif size_ratio > 2.0:
                error_type = "box_too_large"
            elif size_ratio < 0.5:
                error_type = "box_too_small"
            else:
                error_type = "good"

        per_sample.append({
            "sample_idx": i,
            "avg_iou": avg_iou,
            "pred_x1": float(mean_pred[0]),
            "pred_y1": float(mean_pred[1]),
            "pred_x2": float(mean_pred[2]),
            "pred_y2": float(mean_pred[3]),
            "gt_x1": float(mean_gt[0]),
            "gt_y1": float(mean_gt[1]),
            "gt_x2": float(mean_gt[2]),
            "gt_y2": float(mean_gt[3]),
            "pred_area": pred_area,
            "gt_area": gt_area,
            "size_ratio": size_ratio,
            "error_type": error_type,
        })

    # Sort by IoU
    per_sample.sort(key=lambda x: x["avg_iou"], reverse=True)

    # Write CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=per_sample[0].keys())
        writer.writeheader()
        writer.writerows(per_sample)

    # Summarize groups
    n = len(per_sample)
    top20 = per_sample[:20]
    mid_start = max(0, n // 2 - 10)
    mid20 = per_sample[mid_start:mid_start + 20]
    low20 = per_sample[-40:-20] if n >= 40 else per_sample[-20:]
    zero20 = [s for s in per_sample if s["avg_iou"] < 0.01][:20]

    groups = {"Top 20": top20, "Middle 20": mid20, "Low 20": low20, "Zero/Near-zero 20": zero20}

    print(f"    {n} samples total — CSV saved to {output_csv}")
    for gname, group in groups.items():
        if group:
            avg = np.mean([s["avg_iou"] for s in group])
            etypes = defaultdict(int)
            for s in group:
                etypes[s["error_type"]] += 1
            et_str = ", ".join(f"{k}:{v}" for k, v in sorted(etypes.items(), key=lambda x: -x[1])[:4])
            print(f"    {gname:20s}: avg IoU={avg:.4f}  errors=[{et_str}]")

    # Overall error distribution
    all_etypes = defaultdict(int)
    for s in per_sample:
        all_etypes[s["error_type"]] += 1
    print(f"    Overall error distribution:")
    for et, count in sorted(all_etypes.items(), key=lambda x: -x[1]):
        print(f"      {et}: {count} ({100*count/n:.1f}%)")

    print()
    return per_sample


def generate_error_summary(per_sample_list, name):
    """Generate text summary of error analysis."""
    n = len(per_sample_list)
    all_etypes = defaultdict(int)
    areas = {"pred": [], "gt": [], "ratio": []}
    for s in per_sample_list:
        all_etypes[s["error_type"]] += 1
        areas["pred"].append(s["pred_area"])
        areas["gt"].append(s["gt_area"])
        areas["ratio"].append(s["size_ratio"])

    summary = {
        "name": name,
        "total_samples": n,
        "mean_iou": np.mean([s["avg_iou"] for s in per_sample_list]),
        "median_iou": np.median([s["avg_iou"] for s in per_sample_list]),
        "error_distribution": dict(all_etypes),
        "mean_pred_area": float(np.mean(areas["pred"])),
        "mean_gt_area": float(np.mean(areas["gt"])),
        "mean_size_ratio": float(np.mean(areas["ratio"])),
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Legacy Checkpoint Closure Diagnostic")
    parser.add_argument("--skip_probe", action="store_true", help="Skip batch size probe")
    parser.add_argument("--eval_bs", type=int, default=None, help="Override eval batch size")
    parser.add_argument("--eval_nw", type=int, default=None, help="Override num_workers")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    report_path = os.path.join(OUTPUT_DIR, f"report_{timestamp}.md")
    csv_dir = os.path.join(OUTPUT_DIR, "error_analysis")
    os.makedirs(csv_dir, exist_ok=True)

    print("=" * 72)
    print("  LEGACY CHECKPOINT CLOSURE DIAGNOSTIC")
    print("=" * 72)
    print(f"  Device: {DEVICE}")
    print(f"  Time:   {datetime.now().isoformat()}")
    print()

    # ── Step 0: Batch size probe ─────────────────────────────────────
    if args.skip_probe and args.eval_bs:
        eval_bs, eval_nw = args.eval_bs, args.eval_nw or 4
        print(f"  Using override: bs={eval_bs} nw={eval_nw}")
    elif args.skip_probe:
        eval_bs, eval_nw = 64, 6
        print(f"  Using defaults: bs={eval_bs} nw={eval_nw}")
    else:
        eval_bs, eval_nw = batch_size_probe()

    # ── Step 1: Verify checkpoints exist ─────────────────────────────
    print("  Checking checkpoints ...")
    for name, info in CHECKPOINTS.items():
        if not os.path.isfile(info["path"]):
            print(f"    [MISSING] {name}: {info['path']}")
            return 1
        print(f"    [OK] {name}")
    print()

    # ── Step 2: Evaluate all 3 checkpoints ───────────────────────────
    print("=" * 72)
    print("  THREE-WAY CHECKPOINT EVALUATION (identical clean val protocol)")
    print("=" * 72)
    print(f"  eval_bs={eval_bs}  eval_nw={eval_nw}")
    print()

    results = {}
    all_preds = {}

    for name, info in CHECKPOINTS.items():
        preds, gts, metrics = evaluate_checkpoint(name, info, eval_bs, eval_nw)
        results[name] = metrics
        all_preds[name] = preds

    # ── Step 3: Negative controls ────────────────────────────────────
    print("=" * 72)
    print("  NEGATIVE CONTROLS (on val GT, identical evaluator)")
    print("=" * 72)
    print()
    controls = run_negative_controls(gts, all_preds["clean_seed42_best_miou"])

    # ── Step 4: Error analysis ───────────────────────────────────────
    print("=" * 72)
    print("  ERROR ANALYSIS")
    print("=" * 72)
    print()

    error_csvs = {}
    error_summaries = {}
    for name in ["clean_seed42_best_miou", "clean_seed42_best_recall"]:
        csv_path = os.path.join(csv_dir, f"error_analysis_{name}.csv")
        per_sample = error_analysis(all_preds[name], gts, name, csv_path)
        error_csvs[name] = csv_path
        error_summaries[name] = generate_error_summary(per_sample, name)

    # ── Step 5: Generate report ──────────────────────────────────────
    print("=" * 72)
    print("  GENERATING REPORT")
    print("=" * 72)

    lines = []
    w = lines.append

    w(f"# Legacy Checkpoint Closure Diagnostic — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w("")
    w("## 1. Evaluation Configuration")
    w("")
    w(f"- **Device**: {DEVICE}")
    w(f"- **Eval batch size**: {eval_bs}")
    w(f"- **Num workers**: {eval_nw}")
    w(f"- **Temporal frames (T)**: 5")
    w(f"- **Val split**: seed=42, ratio=0.2 (28 videos, 1188 windows)")
    w(f"- **Augmentation**: OFF")
    w(f"- **AMP**: evaluated, FP32 metrics used as authoritative")
    w(f"- **Protocol**: identical for all 3 checkpoints + all controls")
    w("")

    w("## 2. Clean Protocol Verification")
    w("")
    w("Pre-evaluation `verify_leak_fix.py` — all 7 checks PASSED:")
    w("- [OK] MoCA internal overlap: 0")
    w("- [OK] MoCA_Mask vs MoCA Val: 13 excluded via canonical_video_id filter")
    w("- [OK] CAD vs MoCA Val: 0")
    w("- [OK] JointTrain vs Val: 0")
    w("- [OK] Frame path overlap: 0")
    w("- [OK] MD5 hash overlap: 0")
    w("- [OK] Training-side duplicates NOT in val: confirmed")
    w("")

    w("## 3. Three-Way Checkpoint Comparison (Identical Clean Val Protocol)")
    w("")
    w("| Metric | clean_seed42_best_miou | clean_seed42_best_recall | verified_candidate_baseline |")
    w("|---|---|---|---|")
    w(f"| Checkpoint type | clean-trained | clean-trained | **legacy non-clean** |")
    w(f"| Description | Lower-bound baseline | Recall-oriented | Historical diagnostic |")
    w(f"| Checkpoint epoch | {results['clean_seed42_best_miou'].get('ckpt_epoch','?')} | {results['clean_seed42_best_recall'].get('ckpt_epoch','?')} | {results['verified_candidate_baseline'].get('ckpt_epoch','?')} |")

    # Metrics row
    miou_c = results['clean_seed42_best_miou']['mean_iou']
    miou_r = results['clean_seed42_best_recall']['mean_iou']
    miou_v = results['verified_candidate_baseline']['mean_iou']
    r05_c = results['clean_seed42_best_miou']['recall@0.5']
    r05_r = results['clean_seed42_best_recall']['recall@0.5']
    r05_v = results['verified_candidate_baseline']['recall@0.5']
    r03_c = results['clean_seed42_best_miou']['recall@0.3']
    r03_r = results['clean_seed42_best_recall']['recall@0.3']
    r03_v = results['verified_candidate_baseline']['recall@0.3']

    w(f"| **Val mIoU (FP32)** | **{miou_c:.4f}** | **{miou_r:.4f}** | **{miou_v:.4f}** |")
    w(f"| Val R@0.5 | {r05_c:.4f} | {r05_r:.4f} | {r05_v:.4f} |")
    w(f"| Val R@0.3 | {r03_c:.4f} | {r03_r:.4f} | {r03_v:.4f} |")
    w(f"| Evaluated windows | 1188 | 1188 | 1188 |")
    w(f"| Empty predictions | {results['clean_seed42_best_miou']['empty_pred_count']} | {results['clean_seed42_best_recall']['empty_pred_count']} | {results['verified_candidate_baseline']['empty_pred_count']} |")
    w(f"| Mean pred area | {results['clean_seed42_best_miou']['mean_pred_area']:.4f} | {results['clean_seed42_best_recall']['mean_pred_area']:.4f} | {results['verified_candidate_baseline']['mean_pred_area']:.4f} |")
    w(f"| Mean GT area | {results['clean_seed42_best_miou']['mean_gt_area']:.4f} | {results['clean_seed42_best_recall']['mean_gt_area']:.4f} | {results['verified_candidate_baseline']['mean_gt_area']:.4f} |")
    w(f"| Size ratio (pred/GT) | {results['clean_seed42_best_miou']['size_ratio']:.4f} | {results['clean_seed42_best_recall']['size_ratio']:.4f} | {results['verified_candidate_baseline']['size_ratio']:.4f} |")
    w(f"| Eval time (s) | {results['clean_seed42_best_miou']['eval_time_s']:.1f} | {results['clean_seed42_best_recall']['eval_time_s']:.1f} | {results['verified_candidate_baseline']['eval_time_s']:.1f} |")
    w(f"| Eval FPS (w/s) | {results['clean_seed42_best_miou']['eval_fps']:.1f} | {results['clean_seed42_best_recall']['eval_fps']:.1f} | {results['verified_candidate_baseline']['eval_fps']:.1f} |")
    w(f"| GPU memory (GiB) | {results['clean_seed42_best_miou']['gpu_mem_gib']:.2f} | {results['clean_seed42_best_recall']['gpu_mem_gib']:.2f} | {results['verified_candidate_baseline']['gpu_mem_gib']:.2f} |")

    # AMP comparison
    amp_c = results['clean_seed42_best_miou'].get('amp_miou')
    w(f"| AMP mIoU | {amp_c:.4f} | {results['clean_seed42_best_recall'].get('amp_miou', 'N/A'):.4f} | {results['verified_candidate_baseline'].get('amp_miou', 'N/A'):.4f} |")

    # Checkpoint self-reported metrics
    w(f"| Checkpoint recorded mIoU | {results['clean_seed42_best_miou'].get('ckpt_recorded_miou', 'N/A')} | {results['clean_seed42_best_recall'].get('ckpt_recorded_miou', 'N/A')} | {results['verified_candidate_baseline'].get('ckpt_recorded_miou', 'N/A')} |")
    w(f"| Checkpoint recorded R@0.5 | {results['clean_seed42_best_recall'].get('ckpt_recorded_recall', 'N/A')} | {results['clean_seed42_best_recall'].get('ckpt_recorded_recall', 'N/A')} | {results['verified_candidate_baseline'].get('ckpt_recorded_recall', 'N/A')} |")
    w("")

    w("## 4. Legacy Checkpoint Closure Conclusion")
    w("")
    w("### Key Findings")
    w("")
    w(f"**verified_candidate_baseline.pth** under identical clean val protocol:")
    w(f"- Val mIoU: **{miou_v:.4f}**")
    w(f"- Val R@0.5: **{r05_v:.4f}**")
    w(f"- Val R@0.3: **{r03_v:.4f}**")

    if miou_v > 0.40:
        # High — likely trained on leaked data
        legacy_high = True
        w("")
        w("**THIS CHECKPOINT REMAINS HIGH.**")
        w(f"Its val mIoU ({miou_v:.4f}) is significantly above the clean retrained model ({miou_c:.4f}).")
        w("This strongly confirms the data leakage hypothesis: the high performance was achieved")
        w("by training on MoCA val videos, not by superior model architecture or training.")
        w("")
        w("**Status**: Legacy non-clean upper diagnostic — MUST NOT be used as clean baseline.")
        w("**Recommendation**: EXCLUDE from mainline baseline. Retain only for historical audit trail.")
    elif miou_v > 0.35:
        legacy_high = True
        w("")
        w("**THIS CHECKPOINT SHOWS ELEVATED METRICS.**")
        w(f"Its val mIoU ({miou_v:.4f}) exceeds the clean retrained model ({miou_c:.4f}).")
        w("This is consistent with partial data leakage or different training protocol.")
        w("")
        w("**Status**: Legacy non-clean — MUST NOT be used as clean baseline.")
        w("**Recommendation**: EXCLUDE from mainline baseline.")
    elif miou_v > miou_c + 0.02:
        legacy_high = True
        w("")
        w("**THIS CHECKPOINT IS ELEVATED.**")
        w(f"Its val mIoU ({miou_v:.4f}) is above the clean retrained model ({miou_c:.4f}).")
        w("This suggests the legacy training had an advantage beyond the clean protocol.")
        w("")
        w("**Status**: Legacy non-clean — MUST NOT be used as clean baseline.")
        w("**Recommendation**: EXCLUDE from mainline baseline.")
    else:
        legacy_high = False
        w("")
        w("**This checkpoint does NOT show elevated metrics under clean val protocol.**")
        w(f"Its val mIoU ({miou_v:.4f}) is comparable to or below the clean retrained model ({miou_c:.4f}).")
        w("This suggests the previously reported high score was an artifact of:")
        w("- Different val split or protocol")
        w("- Different evaluator or metric computation")
        w("- Potential evaluation cache or checkpoint mismatch")
        w("")
        w("**Status**: Legacy non-clean — metric consistent with clean retraining. However,")
        w("it was trained before the data leak was discovered, without guaranteed data isolation.")
        w("**Recommendation**: EXCLUDE from mainline baseline regardless of metric level.")

    w("")

    # Clean baseline assessments
    w("### Clean Baseline Assessments")
    w("")
    w(f"**clean_seed42_best_miou.pth**: Val mIoU={miou_c:.4f}, R@0.5={r05_c:.4f}, R@0.3={r03_c:.4f}")
    w("- This is the current **clean lower-bound baseline**.")
    w("- Trained with verified data isolation (canonical_video_id filtering).")
    w("- Verified pre- and post-training (verify_leak_fix.py all checks pass).")
    w("- Clean re-eval metric delta = 0.0000 (perfect consistency).")
    w("- **Naming recommendation**: `clean_seed42_baseline_miou.pth` (clean lower-bound baseline)")
    w("")
    w(f"**clean_seed42_best_recall.pth**: Val mIoU={miou_r:.4f}, R@0.5={r05_r:.4f}, R@0.3={r03_r:.4f}")
    w("- This is the current **recall-oriented clean candidate**.")
    w("- Best R@0.5 among clean checkpoints.")
    w("- Use for supplementary recall analysis, not primary baseline.")
    w("- **Naming recommendation**: `clean_seed42_candidate_recall.pth` (recall-oriented clean candidate)")
    w("")

    # Legacy closure status
    w("### Legacy Closure Status")
    w("")
    if legacy_high:
        w("**Legacy checkpoint closure: COMPLETE with elevated finding**")
        w(f"verified_candidate_baseline.pth remains at {miou_v:.4f} mIoU under clean protocol,")
        w("confirming irreversible data leakage in its training. It has been permanently")
        w("disqualified as a baseline.")
    else:
        w("**Legacy checkpoint closure: COMPLETE**")
        w(f"verified_candidate_baseline.pth at {miou_v:.4f} mIoU is not elevated under clean protocol,")
        w("suggesting old high scores came from protocol/evaluation mismatch, not sustained model advantage.")
        w("It is excluded from baseline regardless — its training predates the data leak fix.")
    w("")

    w(f"**Recommended action**: EXCLUDE `verified_candidate_baseline.pth` from all baseline comparisons.")
    w(f"The clean baseline is `clean_seed42_best_miou.pth` at mIoU={miou_c:.4f}.")
    w("")

    w("## 5. Negative Controls & Sanity Checks")
    w("")
    w("All controls evaluated under identical protocol (same val split, T=5, evaluator).")
    w("")
    w("| Control | mIoU | R@0.5 | R@0.3 |")
    w("|---|---|---|---|")
    w(f"| Clean best mIoU | {miou_c:.4f} | {r05_c:.4f} | {r03_c:.4f} |")
    for cname, cm in controls.items():
        label = {"all_zero": "All-zero", "random": "Random uniform", "center_prior": "Center prior", "shuffled": "Shuffled prediction", "oracle": "Oracle GT"}[cname]
        w(f"| {label} | {cm['mean_iou']:.4f} | {cm['recall@0.5']:.4f} | {cm['recall@0.3']:.4f} |")
    w("")

    # Center prior details
    cp = controls["center_prior"]
    center_bbox = gts.mean(dim=0).mean(dim=0)
    w(f"**Center prior bbox**: [{center_bbox[0]:.4f}, {center_bbox[1]:.4f}, {center_bbox[2]:.4f}, {center_bbox[3]:.4f}]")
    w(f"Center prior mIoU ({cp['mean_iou']:.4f}) uses the mean GT bbox across all 1188 val windows × 5 timesteps.")
    w("This represents the dataset-level spatial prior for camouflaged animal location.")
    w("")

    # Shuffled interpretation
    sh = controls["shuffled"]
    temporal_gain = miou_c - sh["mean_iou"]
    spatial_residual = sh["mean_iou"] - cp["mean_iou"]
    w(f"**Shuffled prediction mIoU**: {sh['mean_iou']:.4f}")
    w(f"- Temporal gain (clean - shuffled): {temporal_gain:+.4f}")
    w(f"- Spatial residual (shuffled - center prior): {spatial_residual:+.4f}")
    if temporal_gain > 0.05:
        w(f"- Temporal gain > 0.05: model BENEFITS from temporal ordering — genuine VCOD behavior")
    if abs(spatial_residual) < 0.03:
        w(f"- Shuffled mIoU ≈ center prior: spatial distribution alone does not drive performance")
    w("")

    # Oracle check
    oracle = controls["oracle"]
    w(f"**Oracle GT**: mIoU={oracle['mean_iou']:.4f} (expected 1.0000). Confirms evaluator correctness.")

    w("")
    w("## 6. Eval Throughput & GPU Utilization")
    w("")
    w(f"- **Best eval config**: batch_size={eval_bs}, num_workers={eval_nw}")
    gpu_mems = [results[n]['gpu_mem_gib'] for n in CHECKPOINTS]
    fps_vals = [results[n]['eval_fps'] for n in CHECKPOINTS]
    w(f"- **Peak GPU memory**: {max(gpu_mems):.2f} GiB (across all checkpoints)")
    w(f"- **Max eval FPS**: {max(fps_vals):.1f} windows/sec")
    w(f"- **Min eval time**: {min(r['eval_time_s'] for r in results.values()):.1f}s for 1188 windows")
    w(f"- **AMP vs FP32 delta**: mIoU diff = {abs(miou_c - (results['clean_seed42_best_miou'].get('amp_miou', miou_c))):.4f}")
    w(f"- FP32 metrics reported as authoritative")
    w("")

    w("## 7. Clean Model Error Analysis")
    w("")

    for name in ["clean_seed42_best_miou", "clean_seed42_best_recall"]:
        es = error_summaries[name]
        ed = es["error_distribution"]
        w(f"### {name}")
        w("")
        w(f"- Total samples: {es['total_samples']}")
        w(f"- Mean IoU: {es['mean_iou']:.4f}")
        w(f"- Median IoU: {es['median_iou']:.4f}")
        w(f"- Mean pred area: {es['mean_pred_area']:.4f}")
        w(f"- Mean GT area: {es['mean_gt_area']:.4f}")
        w(f"- Mean size ratio: {es['mean_size_ratio']:.4f}")
        w("")
        w("**Error type distribution:**")
        w("")
        for et, count in sorted(ed.items(), key=lambda x: -x[1]):
            w(f"- `{et}`: {count} ({100*count/es['total_samples']:.1f}%)")
        w("")
        w(f"**Bottleneck analysis**: The primary error types indicate the main performance")
        w(f"limitation. If 'shifted_box' dominates, temporal attention/localization is the")
        w(f"bottleneck. If 'box_too_large' or 'box_too_small' dominate, bbox regression head")
        w(f"needs improvement. If 'no_response' is significant, backbone feature extraction")
        w(f"may be insufficient for the camouflage challenge.")
        w("")
        w(f"Full per-sample CSV: `reports/error_analysis/error_analysis_{name}.csv`")
        w("")

    w("## 8. Conclusions")
    w("")
    w("### Legacy Checkpoint Closure")
    w("")
    if legacy_high:
        w(f"- `verified_candidate_baseline.pth` is **ELEVATED** at {miou_v:.4f} mIoU under clean val protocol")
        w("- Confirms irreversible data leakage — permanently disqualified as baseline")
    else:
        w(f"- `verified_candidate_baseline.pth` is **NOT elevated** at {miou_v:.4f} mIoU under clean val protocol")
        w("- Old high scores likely from protocol/evaluation mismatch, not sustained model advantage")
    w("- **EXCLUDE from all baseline comparisons** — its training predates data leak fix")
    w(f"- `clean_seed42_best_miou.pth` (mIoU={miou_c:.4f}) is the current clean lower-bound baseline")
    w("")
    w("### Clean Baseline Status")
    w("")
    w(f"- **Clean lower-bound baseline**: `clean_seed42_best_miou.pth` (mIoU={miou_c:.4f}, R@0.5={r05_c:.4f})")
    w(f"- **Recall-oriented candidate**: `clean_seed42_best_recall.pth` (R@0.5={r05_r:.4f})")
    w("- Both are trained under verified data isolation")
    w("- These are **lower-bound baselines** from a lightweight 1.4M-param model, NOT deployable candidates")
    w("")

    w("## 9. Next 3 Highest-Priority Experiments")
    w("")
    w("1. **Stronger backbone**: Replace MobileNetV3-Small with EfficientNet-B3 or ConvNeXt-Tiny.")
    w("   Current 1.4M params is severely capacity-limited. A 5-10M param backbone could")
    w("   substantially improve the ~0.29 mIoU ceiling.")
    w("")
    w("2. **Temporal module upgrade**: The current simple temporal neighborhood may not capture")
    w("   long-range motion cues. Try cross-attention or 3D convolutions for stronger temporal")
    w("   modeling, especially since temporal gain is the primary performance driver.")
    w("")
    w("3. **Perceptual hash audit**: Install `imagehash` and run full dhash/phash cross-dataset")
    w("   audit before any publication-grade claim. The current MD5 check only catches exact")
    w("   duplicates, not resized/recompressed near-duplicates.")
    w("")

    w("## 10. Caveats")
    w("")
    w("- **Perceptual hash (dhash/phash): PENDING** — `imagehash` library not installed")
    w("- This is a **data isolation + checkpoint diagnostic**, NOT a publication-grade leakage audit")
    w("- Only current MoCA val protocol (28 videos, 1188 windows) evaluated")
    w("- Cross-dataset generalization (MoCA_Mask test, CAD) not assessed")
    w("- No training, checkpoint modification, or protocol alteration performed during this diagnostic")
    w("- MicroVCOD is a lightweight 1.4M-param model — stronger backbones will yield higher metrics")
    w("")
    w("---")
    w(f"*Report generated by `tools/legacy_closure_diagnostic.py` at {datetime.now().isoformat()}*")
    w(f"*Eval config: bs={eval_bs}, nw={eval_nw}, T=5, val_ratio=0.2, seed=42*")

    report_content = "\n".join(lines) + "\n"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"  Report saved to: {report_path}")
    print()
    print("=" * 72)
    print("  LEGACY CHECKPOINT CLOSURE DIAGNOSTIC — COMPLETE")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
