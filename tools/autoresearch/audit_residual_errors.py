"""Residual error audit: per-video metrics for two checkpoints.

Usage:
    python tools/autoresearch/audit_residual_errors.py \
        --ckpt-a PATH --label-a "MV3-Small E40" \
        --ckpt-b PATH --label-b "EffB0 E38" \
        --backbone-a mobilenet_v3_small --backbone-b efficientnet_b0 \
        --head-type dense_fg_aux --input-size 224 --T 5 \
        --resized-root C:\\datasets_224
"""
import os, sys, json, time, argparse
from collections import defaultdict
import numpy as np
import torch

# Project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.dataset_real import RealVideoBBoxDataset
from src.model import MicroVCOD
from eval.eval_video_bbox import bbox_iou, categorize_size, classify_error

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_VAL = r"C:\datasets\MoCA"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")


def _video_name_from_sample(sample):
    """Extract a canonical video name from a dataset sample dict."""
    for key in ("video_name", "video_id"):
        if key in sample:
            return str(sample[key])
    if "frame_dir" in sample and isinstance(sample["frame_dir"], str):
        return os.path.basename(sample["frame_dir"])
    return str(hash(sample.get("video_path", sample.get("frame_dir", ""))))


def split_by_video_np(ds, val_ratio=0.2, seed=42):
    video_to_indices = defaultdict(list)
    for i, s in enumerate(ds.samples):
        vn = _video_name_from_sample(s)
        video_to_indices[vn].append(i)
    videos = sorted(video_to_indices.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(videos)
    n_val = max(1, int(len(videos) * val_ratio))
    val_videos = set(videos[:n_val])
    val_idx = [i for v in val_videos for i in video_to_indices[v]]
    return val_idx, val_videos, video_to_indices


def evaluate_checkpoint(ckpt_path, backbone, head_type, input_size, T,
                        resized_root, val_idx, val_videos, video_to_indices, label):
    """Evaluate one checkpoint and return per-video metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Backbone: {backbone}")
    print(f"{'='*60}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_epoch = ckpt.get("epoch", -1)
    train_miou = ckpt.get("miou", 0.0)

    model = MicroVCOD(T=T, backbone_name=backbone, head_type=head_type)
    model.load_state_dict({k: v.to(DEVICE) for k, v in ckpt["model_state_dict"].items()})
    model.to(DEVICE)
    model.eval()

    from src.dataset_real import collate_video_clips as collate_fn
    from torch.utils.data import DataLoader, Subset

    ds = RealVideoBBoxDataset(
        [DEFAULT_VAL], T=T, target_size=input_size,
        augment=False, temporal_stride=1,
        resized_root=resized_root, return_mask=False
    )
    loader = DataLoader(
        Subset(ds, val_idx), batch_size=64, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True
    )

    # Collect per-sample predictions
    all_preds, all_gts, all_indices = [], [], []
    t0 = time.time()
    for batch_idx, (frames, gt_bboxes) in enumerate(loader):
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                pred = model(frames.to(DEVICE))
        all_preds.append(pred.float().cpu())
        all_gts.append(gt_bboxes)
        # Track which val indices this batch covers
        start = batch_idx * 64
        all_indices.extend(range(start, start + len(frames)))
    elapsed = time.time() - t0

    preds = torch.cat(all_preds, dim=0)  # (N_clips, T, 4)
    gts = torch.cat(all_gts, dim=0)      # (N_clips, T, 4)
    N_clips, T_dim, _ = preds.shape

    # Map val_idx samples back to videos (clip-level mapping)
    val_idx_set = set(val_idx)
    idx_to_video = {}
    for vn, indices in video_to_indices.items():
        for idx in indices:
            if idx in val_idx_set:
                idx_to_video[idx] = vn

    # Per-video aggregation (clip-level)
    per_video = {}
    val_pos_to_ds_idx = val_idx  # position in subset -> original dataset index

    for clip_i in range(N_clips):
        ds_idx = val_pos_to_ds_idx[clip_i]
        vn = idx_to_video.get(ds_idx, f"unknown_{ds_idx}")
        clip_ious = bbox_iou(preds[clip_i], gts[clip_i])  # (T,)
        clip_gt_w = (gts[clip_i, :, 2] - gts[clip_i, :, 0]).clamp(min=0)
        clip_gt_h = (gts[clip_i, :, 3] - gts[clip_i, :, 1]).clamp(min=0)
        clip_gt_area = clip_gt_w * clip_gt_h
        clip_pred_w = (preds[clip_i, :, 2] - preds[clip_i, :, 0]).clamp(min=0)
        clip_pred_h = (preds[clip_i, :, 3] - preds[clip_i, :, 1]).clamp(min=0)
        clip_pred_area = clip_pred_w * clip_pred_h
        clip_area_ratio = clip_pred_area / (clip_gt_area + 1e-8)
        clip_pred_cx = (preds[clip_i, :, 0] + preds[clip_i, :, 2]) / 2
        clip_pred_cy = (preds[clip_i, :, 1] + preds[clip_i, :, 3]) / 2
        clip_gt_cx = (gts[clip_i, :, 0] + gts[clip_i, :, 2]) / 2
        clip_gt_cy = (gts[clip_i, :, 1] + gts[clip_i, :, 3]) / 2
        clip_center_err = ((clip_pred_cx - clip_gt_cx) ** 2 + (clip_pred_cy - clip_gt_cy) ** 2).sqrt()

        if vn not in per_video:
            per_video[vn] = {"ious": [], "gt_areas": [], "area_ratios": [],
                            "center_errs": [], "n_frames": 0}
        for t in range(T_dim):
            per_video[vn]["ious"].append(clip_ious[t].item())
            per_video[vn]["gt_areas"].append(clip_gt_area[t].item())
            per_video[vn]["area_ratios"].append(clip_area_ratio[t].item())
            per_video[vn]["center_errs"].append(clip_center_err[t].item())
            per_video[vn]["n_frames"] += 1

    # Flatten for aggregate metrics
    preds_flat = preds.reshape(N_clips * T_dim, 4)
    gts_flat = gts.reshape(N_clips * T_dim, 4)
    ious = bbox_iou(preds_flat, gts_flat)
    gt_w = (gts_flat[:, 2] - gts_flat[:, 0]).clamp(min=0)
    gt_h = (gts_flat[:, 3] - gts_flat[:, 1]).clamp(min=0)
    gt_area = gt_w * gt_h
    pred_w = (preds_flat[:, 2] - preds_flat[:, 0]).clamp(min=0)
    pred_h = (preds_flat[:, 3] - preds_flat[:, 1]).clamp(min=0)
    pred_area = pred_w * pred_h
    area_ratio = pred_area / (gt_area + 1e-8)
    pred_cx = (preds_flat[:, 0] + preds_flat[:, 2]) / 2
    pred_cy = (preds_flat[:, 1] + preds_flat[:, 3]) / 2
    gt_cx = (gts_flat[:, 0] + gts_flat[:, 2]) / 2
    gt_cy = (gts_flat[:, 1] + gts_flat[:, 3]) / 2
    center_err = ((pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2).sqrt()

    # Compute per-video summary
    video_summary = {}
    for vn, data in per_video.items():
        ious_arr = np.array(data["ious"])
        mIoU = ious_arr.mean()
        bad_rate = (ious_arr < 0.5).mean()
        r05 = (ious_arr >= 0.5).mean()

        areas = np.array(data["gt_areas"])
        area_ratios = np.array(data["area_ratios"])
        center_errs = np.array(data["center_errs"])

        # Size breakdown
        iou_by_size = defaultdict(list)
        for i, a in enumerate(areas):
            iou_by_size[categorize_size(a)].append(ious_arr[i])
        IoU_tiny = np.mean(iou_by_size["tiny"]) if iou_by_size["tiny"] else 0
        IoU_small = np.mean(iou_by_size["small"]) if iou_by_size["small"] else 0
        IoU_medium = np.mean(iou_by_size["medium"]) if iou_by_size["medium"] else 0
        IoU_large = np.mean(iou_by_size["large"]) if iou_by_size["large"] else 0

        # Error classification
        error_counts = defaultdict(int)
        for i in range(len(ious_arr)):
            err = classify_error(ious_arr[i], area_ratios[i], center_errs[i])
            error_counts[err] += 1

        video_summary[vn] = {
            "n_frames": data["n_frames"],
            "mIoU": round(float(mIoU), 4),
            "bad_frame_rate": round(float(bad_rate), 4),
            "R@0.5": round(float(r05), 4),
            "IoU_tiny": round(float(IoU_tiny), 4),
            "IoU_small": round(float(IoU_small), 4),
            "IoU_medium": round(float(IoU_medium), 4),
            "IoU_large": round(float(IoU_large), 4),
            "mean_area_ratio": round(float(area_ratios.mean()), 4),
            "mean_center_error": round(float(center_errs.mean()), 4),
            "n_pred_too_large": error_counts["pred_too_large"],
            "n_pred_too_small": error_counts["pred_too_small"],
            "n_center_shift": error_counts["center_shift"],
            "n_scale_mismatch": error_counts["scale_mismatch"],
            "n_good": error_counts["good"],
        }

    # Aggregate metrics (same as reeval)
    pf = {}
    pf["per_frame_mIoU"] = float(ious.mean())
    pf["bad_frame_rate"] = float((ious < 0.5).float().mean())
    pf["R@0.5"] = float((ious >= 0.5).float().mean())

    iou_by_size_agg = defaultdict(list)
    for i in range(len(ious)):
        iou_by_size_agg[categorize_size(gt_area[i].item())].append(ious[i].item())
    for sz in ["tiny", "small", "medium", "large"]:
        vals = iou_by_size_agg[sz]
        pf[f"IoU_{sz}"] = float(np.mean(vals)) if vals else 0.0

    pf["area_ratio_mean"] = float(area_ratio.mean())
    pf["center_error_mean"] = float(center_err.mean())

    error_counts_agg = defaultdict(int)
    for i in range(len(ious)):
        err = classify_error(ious[i].item(), area_ratio[i].item(), center_err[i].item())
        error_counts_agg[err] += 1
    pf["n_pred_too_large"] = error_counts_agg["pred_too_large"]
    pf["n_pred_too_small"] = error_counts_agg["pred_too_small"]
    pf["n_center_shift"] = error_counts_agg["center_shift"]
    pf["n_scale_mismatch"] = error_counts_agg["scale_mismatch"]
    pf["n_good"] = error_counts_agg["good"]
    pf["total_frames"] = len(ious)

    print(f"  Aggregate: pf_mIoU={pf['per_frame_mIoU']:.4f} bad={pf['bad_frame_rate']:.4f} "
          f"R@0.5={pf['R@0.5']:.4f}")
    print(f"  IoU: tiny={pf['IoU_tiny']:.4f} small={pf['IoU_small']:.4f} "
          f"medium={pf['IoU_medium']:.4f} large={pf['IoU_large']:.4f}")
    print(f"  Errors: too_large={pf['n_pred_too_large']} too_small={pf['n_pred_too_small']} "
          f"center={pf['n_center_shift']} scale={pf['n_scale_mismatch']} "
          f"good={pf['n_good']}/{pf['total_frames']}")
    print(f"  Eval time: {elapsed:.1f}s")

    return {
        "label": label,
        "backbone": backbone,
        "ckpt_path": ckpt_path,
        "train_epoch": train_epoch,
        "train_val_miou": round(train_miou, 4),
        "aggregate": pf,
        "per_video": video_summary,
        "n_val_videos": len(val_videos),
        "eval_time_s": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-a", required=True)
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--backbone-a", required=True)
    parser.add_argument("--ckpt-b", required=True)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--backbone-b", required=True)
    parser.add_argument("--head-type", default="dense_fg_aux")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--resized-root", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # Build dataset once for video split
    ds = RealVideoBBoxDataset(
        [DEFAULT_VAL], T=args.T, target_size=args.input_size,
        augment=False, temporal_stride=1,
        resized_root=args.resized_root, return_mask=False
    )
    val_idx, val_videos, video_to_indices = split_by_video_np(ds, val_ratio=0.2, seed=42)
    print(f"Val split: {len(val_idx)} frames from {len(val_videos)} videos")

    # Evaluate both checkpoints
    result_a = evaluate_checkpoint(
        args.ckpt_a, args.backbone_a, args.head_type,
        args.input_size, args.T, args.resized_root,
        val_idx, val_videos, video_to_indices, args.label_a
    )
    result_b = evaluate_checkpoint(
        args.ckpt_b, args.backbone_b, args.head_type,
        args.input_size, args.T, args.resized_root,
        val_idx, val_videos, video_to_indices, args.label_b
    )

    # --- Comparison ---
    print(f"\n{'='*60}")
    print("RESIDUAL ERROR AUDIT")
    print(f"{'='*60}")

    a = result_a["aggregate"]
    b = result_b["aggregate"]
    a_vid = result_a["per_video"]
    b_vid = result_b["per_video"]

    print(f"\n--- Aggregate Comparison ---")
    print(f"{'Metric':<20} {'A: ' + args.label_a:<25} {'B: ' + args.label_b:<25} {'Delta (B-A)':>12}")
    print("-" * 82)
    for key, fmt in [("per_frame_mIoU", ".4f"), ("bad_frame_rate", ".4f"),
                      ("R@0.5", ".4f"), ("IoU_tiny", ".4f"),
                      ("IoU_small", ".4f"), ("IoU_medium", ".4f"),
                      ("IoU_large", ".4f"), ("area_ratio_mean", ".4f"),
                      ("center_error_mean", ".4f")]:
        va, vb = a[key], b[key]
        delta = vb - va
        print(f"{key:<20} {va:{fmt}}  {vb:{fmt}}  {delta:+{fmt}}")

    print(f"\n{'Error Type':<20} {'A':>8} {'B':>8} {'Delta':>8}")
    print("-" * 48)
    for ek in ["n_good", "n_pred_too_large", "n_pred_too_small",
                "n_center_shift", "n_scale_mismatch", "total_frames"]:
        va, vb = a[ek], b[ek]
        delta = vb - va
        print(f"{ek:<20} {va:>8} {vb:>8} {delta:>+8}")

    # Per-video comparison
    print(f"\n--- Per-Video Comparison (sorted by A mIoU) ---")
    all_videos = sorted(set(list(a_vid.keys()) + list(b_vid.keys())))
    print(f"{'Video':<30} {'A mIoU':>8} {'B mIoU':>8} {'Delta':>8} "
          f"{'A bad%':>7} {'B bad%':>7} {'A err':>6} {'B err':>6}")
    print("-" * 95)

    hard_videos = []
    a_wins = []
    b_wins = []
    ties = []

    for vn in all_videos:
        va = a_vid.get(vn, {})
        vb = b_vid.get(vn, {})
        miou_a = va.get("mIoU", 0)
        miou_b = vb.get("mIoU", 0)
        bad_a = va.get("bad_frame_rate", 0)
        bad_b = vb.get("bad_frame_rate", 0)
        delta = miou_b - miou_a
        n_err_a = va.get("n_pred_too_small", 0) + va.get("n_pred_too_large", 0) + va.get("n_center_shift", 0)
        n_err_b = vb.get("n_pred_too_small", 0) + vb.get("n_pred_too_large", 0) + vb.get("n_center_shift", 0)

        if abs(delta) < 0.01:
            ties.append(vn)
        elif delta > 0:
            b_wins.append((vn, delta))
        else:
            a_wins.append((vn, -delta))

        if miou_a < 0.5 or miou_b < 0.5:
            hard_videos.append((vn, miou_a, miou_b))

        vn_short = vn if len(vn) <= 28 else vn[:27] + "~"
        print(f"{vn_short:<30} {miou_a:>8.4f} {miou_b:>8.4f} {delta:>+8.4f} "
              f"{bad_a:>7.4f} {bad_b:>7.4f} {n_err_a:>6} {n_err_b:>6}")

    # Summary
    print(f"\n--- Head-to-Head Summary ---")
    print(f"  {args.label_a} wins: {len(a_wins)} videos (mean margin: {np.mean([d for _, d in a_wins]) if a_wins else 0:.4f})")
    print(f"  {args.label_b} wins: {len(b_wins)} videos (mean margin: {np.mean([d for _, d in b_wins]) if b_wins else 0:.4f})")
    print(f"  Ties (<0.01): {len(ties)} videos")
    if b_wins:
        print(f"\n  Top {args.label_b} wins:")
        for vn, delta in sorted(b_wins, key=lambda x: -x[1])[:5]:
            print(f"    {vn}: +{delta:.4f}")
    if a_wins:
        print(f"\n  Top {args.label_a} wins:")
        for vn, delta in sorted(a_wins, key=lambda x: -x[1])[:5]:
            print(f"    {vn}: +{delta:.4f}")

    print(f"\n--- Hard Videos (mIoU < 0.5 in either model) ---")
    if hard_videos:
        for vn, miou_a, miou_b in sorted(hard_videos, key=lambda x: min(x[1], x[2])):
            print(f"  {vn:<35} A={miou_a:.4f}  B={miou_b:.4f}")
    else:
        print("  None! All videos above 0.5 mIoU in both models.")

    # Save
    output = {
        "audit_config": {
            "ckpt_a": args.ckpt_a, "label_a": args.label_a, "backbone_a": args.backbone_a,
            "ckpt_b": args.ckpt_b, "label_b": args.label_b, "backbone_b": args.backbone_b,
        },
        "result_a": {k: v for k, v in result_a.items() if k != "per_video"},
        "result_b": {k: v for k, v in result_b.items() if k != "per_video"},
        "per_video_a": result_a["per_video"],
        "per_video_b": result_b["per_video"],
        "comparison": {
            "a_wins": [(vn, round(d, 4)) for vn, d in a_wins],
            "b_wins": [(vn, round(d, 4)) for vn, d in b_wins],
            "ties": ties,
            "hard_videos": [(vn, round(ma, 4), round(mb, 4)) for vn, ma, mb in hard_videos],
            "n_a_wins": len(a_wins),
            "n_b_wins": len(b_wins),
            "n_ties": len(ties),
            "n_hard_videos": len(hard_videos),
        },
    }

    out_path = args.output or "residual_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")

    return output


if __name__ == "__main__":
    main()
