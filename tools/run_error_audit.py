"""E-41: Validation Error Audit.
Diagnoses the ~0.31 mIoU ceiling by analyzing per-video/sample errors,
systematic biases, annotation quality, and E-31 vs E-40 differences.
"""

import os, sys, json, csv, time, hashlib
from collections import defaultdict
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import MicroVCOD
from src.dataset_real import RealVideoBBoxDataset, collate_video_clips
from src.loss import _box_iou as _bbox_iou_func

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE = 224
T = 5
BS = 64
NUM_WORKERS = 2
RESIZED_ROOT = r"C:\datasets_224"

VAL_DATASET = r"C:\datasets\MoCA"
VAL_RATIO = 0.2
SPLIT_SEED = 42

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "reports", "e41_error_audit")
OVERLAY_DIR = os.path.join(OUT_DIR, "worst_100_overlays")
os.makedirs(OVERLAY_DIR, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────

def bbox_iou(a, b):
    if isinstance(a, np.ndarray): a = torch.from_numpy(a)
    if isinstance(b, np.ndarray): b = torch.from_numpy(b)
    return float(_bbox_iou_func(a, b))

def bbox_area(b):
    return max(0.0, (b[2] - b[0]) * (b[3] - b[1]))

def bbox_center(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])

def _video_name_from_sample(sample):
    d = sample.get("frame_dir", "")
    parts = d.replace("\\", "/").split("/")
    if "MoCA_Mask" in d:
        for i, p in enumerate(parts):
            if p in ("TrainDataset_per_sq",):
                return parts[i + 1] if i + 1 < len(parts) else ""
    for i, p in enumerate(parts):
        if p in ("JPEGImages", "Imgs", "frames"):
            return parts[i + 1] if i + 1 < len(parts) else ""
    return ""

def split_by_video(ds, val_ratio=0.2, seed=42):
    video_to_indices = defaultdict(list)
    for i, s in enumerate(ds.samples):
        vn = _video_name_from_sample(s)
        video_to_indices[vn].append(i)
    videos = sorted(video_to_indices.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(videos)
    n_val = max(1, int(len(videos) * val_ratio))
    val_videos = set(videos[:n_val])
    train_idx, val_idx = [], []
    for vn in videos:
        for i in video_to_indices[vn]:
            (val_idx if vn in val_videos else train_idx).append(i)
    return train_idx, val_idx

# ── load model ────────────────────────────────────────────────────────

def load_model(ckpt_path, head_type):
    model = MicroVCOD(T=T, backbone_name="mobilenet_v3_small", head_type=head_type)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model

# ── run inference on val set ──────────────────────────────────────────

def run_val_inference(model, val_loader):
    all_preds, all_gts, all_indices = [], [], []
    for batch_idx, (frames, gt_bboxes) in enumerate(val_loader):
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                pred = model(frames.to(DEVICE))
        all_preds.append(pred.float().cpu())
        all_gts.append(gt_bboxes)
        start_idx = batch_idx * BS
        for i in range(frames.shape[0]):
            all_indices.append(start_idx + i)
    return (
        torch.cat(all_preds, dim=0),  # (N, T, 4)
        torch.cat(all_gts, dim=0),    # (N, T, 4)
        all_indices,
    )

# ── per-sample analysis ───────────────────────────────────────────────

def analyze_samples(preds, gts, dataset, indices):
    """Compute per-sample metrics. Returns list of dicts."""
    records = []
    for p_idx, (pred_t, gt_t, ds_idx) in enumerate(zip(preds, gts, indices)):
        sample = dataset.samples[ds_idx]
        video = _video_name_from_sample(sample)
        for t in range(T):
            pred = pred_t[t].numpy()
            gt = gt_t[t].numpy()
            iou = bbox_iou(pred, gt)

            # BBox areas
            gt_area_norm = bbox_area(gt)
            pred_area_norm = bbox_area(pred)
            area_ratio = pred_area_norm / max(gt_area_norm, 1e-8)

            # Center error in normalized coords
            c_pred = bbox_center(pred)
            c_gt = bbox_center(gt)
            center_err = np.linalg.norm(c_pred - c_gt)

            # Width/height error
            w_pred = pred[2] - pred[0]
            h_pred = pred[3] - pred[1]
            w_gt = gt[2] - gt[0]
            h_gt = gt[3] - gt[1]
            w_err = w_pred - w_gt
            h_err = h_pred - h_gt

            # Aspect ratio
            gt_ar = w_gt / max(h_gt, 1e-8)
            pred_ar = w_pred / max(h_pred, 1e-8)

            # Object size category
            if gt_area_norm < 0.01:
                size_cat = "tiny"
            elif gt_area_norm < 0.05:
                size_cat = "small"
            elif gt_area_norm < 0.15:
                size_cat = "medium"
            else:
                size_cat = "large"

            # Error type
            if iou >= 0.5:
                err_type = "good"
            elif area_ratio > 2.0:
                err_type = "pred_too_large"
            elif area_ratio < 0.5:
                err_type = "pred_too_small"
            elif center_err > 0.15:
                err_type = "center_shift"
            else:
                err_type = "scale_mismatch"

            records.append({
                "video": video,
                "ds_idx": ds_idx,
                "frame_t": t,
                "iou": round(iou, 4),
                "gt_area": round(gt_area_norm, 6),
                "pred_area": round(pred_area_norm, 6),
                "area_ratio": round(area_ratio, 3),
                "center_err": round(center_err, 4),
                "w_err": round(w_err, 4),
                "h_err": round(h_err, 4),
                "gt_ar": round(gt_ar, 3),
                "pred_ar": round(pred_ar, 3),
                "size_cat": size_cat,
                "err_type": err_type,
                "gt_x1": round(float(gt[0]), 4),
                "gt_y1": round(float(gt[1]), 4),
                "gt_x2": round(float(gt[2]), 4),
                "gt_y2": round(float(gt[3]), 4),
                "pred_x1": round(float(pred[0]), 4),
                "pred_y1": round(float(pred[1]), 4),
                "pred_x2": round(float(pred[2]), 4),
                "pred_y2": round(float(pred[3]), 4),
            })
    return records

# ── per-video aggregation ─────────────────────────────────────────────

def per_video_metrics(records):
    videos = defaultdict(list)
    for r in records:
        videos[r["video"]].append(r)
    result = []
    for vn, recs in sorted(videos.items()):
        ious = [r["iou"] for r in recs]
        miou = np.mean(ious)
        n_frames = len(recs)
        n_good = sum(1 for r in recs if r["iou"] >= 0.5)
        n_bad = n_frames - n_good
        err_types = defaultdict(int)
        for r in recs:
            if r["err_type"] != "good":
                err_types[r["err_type"]] += 1
        top_err = max(err_types, key=err_types.get) if err_types else "none"
        result.append({
            "video": vn, "n_frames": n_frames, "mIoU": round(miou, 4),
            "r05": round(n_good / max(n_frames, 1), 4),
            "n_bad": n_bad, "top_error": top_err,
        })
    result.sort(key=lambda x: x["mIoU"])
    return result

# ── draw overlay ──────────────────────────────────────────────────────

def draw_overlay(sample, t, pred, gt, iou, save_path):
    """Draw frame + bboxes overlay."""
    frame_dir = sample.get("frame_dir", "")
    ext = sample.get("frame_ext", ".jpg")
    fi = sample["start_frame"] + t * sample["annot_interval"]

    # Find frame path
    fpath = None
    lookup = sample.get("frame_lookup")
    if lookup and fi in lookup:
        fpath = os.path.join(frame_dir, lookup[fi])
    if fpath is None or not os.path.exists(fpath):
        direct = os.path.join(frame_dir, f"{fi:05d}{ext}")
        if os.path.exists(direct):
            fpath = direct
    if fpath is None or not os.path.exists(fpath):
        existing = sorted(os.listdir(frame_dir)) if os.path.isdir(frame_dir) else []
        for fname in existing:
            if f"_{fi:03d}" in fname or fname.startswith(f"{fi}_"):
                fpath = os.path.join(frame_dir, fname)
                break
    if fpath is None or not os.path.exists(fpath):
        for pad in [3, 4, 5]:
            fp = os.path.join(frame_dir, f"{fi:0{pad}d}{ext}")
            if os.path.exists(fp):
                fpath = fp
                break
    if fpath is None or not os.path.exists(fpath):
        return

    img = cv2.imread(fpath)
    if img is None:
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Draw GT in green
    gt_px = [int(gt[0] * w), int(gt[1] * h), int(gt[2] * w), int(gt[3] * h)]
    cv2.rectangle(img, (gt_px[0], gt_px[1]), (gt_px[2], gt_px[3]), (0, 255, 0), 2)

    # Draw Pred in red
    pred_px = [int(pred[0] * w), int(pred[1] * h), int(pred[2] * w), int(pred[3] * h)]
    cv2.rectangle(img, (pred_px[0], pred_px[1]), (pred_px[2], pred_px[3]), (255, 0, 0), 2)

    # IoU text
    cv2.putText(img, f"IoU={iou:.3f}", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# ── train/val distribution ────────────────────────────────────────────

def dist_analysis(train_ds, val_ds, val_indices):
    """Compare train vs val distributions."""
    def collect_stats(ds, indices=None):
        areas, ars, cx, cy = [], [], [], []
        sources = defaultdict(int)
        for i in (indices if indices is not None else range(len(ds))):
            s = ds.samples[i]
            d = s.get("frame_dir", "")
            if "MoCA_Mask" in d:
                src = "MoCA_Mask"
            elif "CamouflagedAnimal" in d:
                src = "CAD"
            else:
                src = "MoCA"
            sources[src] += 1
            for fi in list(s["bbox_map"].values())[:5]:
                b = fi if isinstance(fi, np.ndarray) else np.array(fi)
                area = max(0, (b[2] - b[0]) * (b[3] - b[1]))
                w_ = b[2] - b[0]
                h_ = b[3] - b[1]
                ar = w_ / max(h_, 1e-8)
                areas.append(area)
                ars.append(ar)
                cx.append((b[0] + b[2]) / 2)
                cy.append((b[1] + b[3]) / 2)
        return {
            "n_samples": len(indices) if indices else len(ds),
            "areas": areas,
            "ars": ars,
            "cx": cx,
            "cy": cy,
            "sources": dict(sources),
        }

    train_stats = collect_stats(train_ds)
    val_stats = collect_stats(val_ds, val_indices)
    return train_stats, val_stats

# ── systematic bias ───────────────────────────────────────────────────

def systematic_bias(records):
    """Analyze systematic prediction biases."""
    n = len(records)
    area_ratios = [r["area_ratio"] for r in records]
    center_errs = [r["center_err"] for r in records]
    w_errs = [r["w_err"] for r in records]
    h_errs = [r["h_err"] for r in records]

    # IoU vs object size
    size_bins = {"tiny": [], "small": [], "medium": [], "large": []}
    for r in records:
        size_bins[r["size_cat"]].append(r["iou"])

    # IoU vs aspect ratio
    ar_bins = {"tall": [], "square": [], "wide": []}
    for r in records:
        if r["gt_ar"] < 0.7:
            ar_bins["tall"].append(r["iou"])
        elif r["gt_ar"] > 1.4:
            ar_bins["wide"].append(r["iou"])
        else:
            ar_bins["square"].append(r["iou"])

    return {
        "area_ratio_mean": np.mean(area_ratios),
        "area_ratio_median": np.median(area_ratios),
        "area_ratio_p25": np.percentile(area_ratios, 25),
        "area_ratio_p75": np.percentile(area_ratios, 75),
        "pred_larger_ratio": np.mean(np.array(area_ratios) > 1.5),
        "pred_smaller_ratio": np.mean(np.array(area_ratios) < 0.67),
        "center_err_mean": np.mean(center_errs),
        "center_err_median": np.median(center_errs),
        "w_err_mean": np.mean(w_errs),
        "h_err_mean": np.mean(h_errs),
        "iou_by_size": {k: round(np.mean(v), 4) for k, v in size_bins.items()},
        "iou_by_ar": {k: round(np.mean(v), 4) for k, v in ar_bins.items()},
        "n_by_size": {k: len(v) for k, v in size_bins.items()},
        "n_by_ar": {k: len(v) for k, v in ar_bins.items()},
    }

# ── error type summary ────────────────────────────────────────────────

def error_type_summary(records):
    counts = defaultdict(int)
    for r in records:
        counts[r["err_type"]] += 1
        counts[f"size_{r['size_cat']}"] += 1
    total = len(records)
    return {k: {"count": v, "pct": round(100 * v / total, 1)} for k, v in sorted(counts.items())}

# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("E-41: Validation Error Audit")
    print("=" * 60)

    # ── Load val dataset ──────────────────────────────────────────
    print("\n[1/7] Loading val dataset ...")
    val_ds = RealVideoBBoxDataset(
        [VAL_DATASET], T=T, target_size=INPUT_SIZE,
        augment=False, temporal_stride=1, cache_dir=None,
        resized_root=RESIZED_ROOT, return_mask=False,
    )
    _, val_idx = split_by_video(val_ds, val_ratio=VAL_RATIO, seed=SPLIT_SEED)
    print(f"  Val samples: {len(val_idx)} in {len(set(_video_name_from_sample(val_ds.samples[i]) for i in val_idx))} videos")

    val_subset = Subset(val_ds, val_idx)
    val_loader = DataLoader(val_subset, batch_size=BS, shuffle=False,
                            collate_fn=collate_video_clips, num_workers=0)

    # ── Also load train dataset for distribution comparison ───────
    print("  Loading train dataset for distribution comparison ...")
    train_ds = RealVideoBBoxDataset(
        [VAL_DATASET], T=T, target_size=INPUT_SIZE,
        augment=False, temporal_stride=1, cache_dir=None,
        resized_root=RESIZED_ROOT, return_mask=False,
    )

    # ── Load models ───────────────────────────────────────────────
    print("\n[2/7] Loading checkpoints ...")
    ckpt_e31 = r"D:\dualvcod\local_runs\autoresearch\expl_31_diou_30ep\checkpoint_best.pth"
    ckpt_e40 = r"D:\dualvcod\local_runs\autoresearch\expl_40_densefg_30ep\checkpoint_best.pth"

    models = {}
    for name, ckpt_path, head in [
        ("E-31", ckpt_e31, "current_direct_bbox"),
        ("E-40", ckpt_e40, "dense_fg_aux"),
    ]:
        if os.path.exists(ckpt_path):
            print(f"  Loading {name} from {ckpt_path} ...")
            models[name] = load_model(ckpt_path, head)
        else:
            print(f"  [WARN] {name} checkpoint not found: {ckpt_path}")

    # ── Run inference ─────────────────────────────────────────────
    print("\n[3/7] Running inference on val set ...")
    all_records = {}
    for name, model in models.items():
        t0 = time.time()
        preds, gts, indices = run_val_inference(model, val_loader)
        # Map loader indices back to dataset indices
        ds_indices = [val_idx[i] for i in indices]
        records = analyze_samples(preds, gts, val_ds, ds_indices)
        all_records[name] = records
        miou = np.mean([r["iou"] for r in records])
        print(f"  {name}: {len(records)} frames, mIoU={miou:.4f}, time={time.time()-t0:.1f}s")

    # ── Per-video analysis ────────────────────────────────────────
    print("\n[4/7] Per-video analysis ...")
    for name, records in all_records.items():
        vmetrics = per_video_metrics(records)

        # Save CSV
        csv_path = os.path.join(OUT_DIR, f"per_video_metrics_{name.replace('-','_').lower()}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=vmetrics[0].keys())
            w.writeheader()
            w.writerows(vmetrics)
        print(f"  {name} per-video CSV: {csv_path}")

        # Worst 20
        worst20 = vmetrics[:20]
        print(f"\n  {name} Worst 20 videos:")
        print(f"  {'Video':<30s} {'mIoU':>7s} {'R@0.5':>7s} {'n_frames':>9s} {'top_error':>16s}")
        print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*9} {'-'*16}")
        for v in worst20:
            print(f"  {v['video']:<30s} {v['mIoU']:7.4f} {v['r05']:7.4f} {v['n_frames']:9d} {v['top_error']:>16s}")

        # Bottom N% contribution
        all_miou = np.array([r["iou"] for r in records])
        total_err = np.sum(1.0 - all_miou)
        sorted_errs = np.sort(1.0 - all_miou)[::-1]
        for pct in [10, 20]:
            n_cut = int(len(sorted_errs) * pct / 100)
            contrib = np.sum(sorted_errs[:n_cut]) / max(total_err, 1e-8)
            print(f"\n  Bottom {pct}% frames contribute {contrib*100:.1f}% of total error")

        # Save per-sample errors CSV
        sample_csv = os.path.join(OUT_DIR, f"per_sample_errors_{name.replace('-','_').lower()}.csv")
        with open(sample_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)

    # ── Overlays for worst 100 ─────────────────────────────────────
    print("\n[5/7] Generating worst-100 overlays ...")
    primary_records = all_records.get("E-31", list(all_records.values())[0])
    sorted_records = sorted(primary_records, key=lambda r: r["iou"])
    worst100 = sorted_records[:100]

    for rank, r in enumerate(worst100):
        sample = val_ds.samples[r["ds_idx"]]
        t = r["frame_t"]
        pred = np.array([r["pred_x1"], r["pred_y1"], r["pred_x2"], r["pred_y2"]])
        gt = np.array([r["gt_x1"], r["gt_y1"], r["gt_x2"], r["gt_y2"]])
        out_path = os.path.join(OVERLAY_DIR, f"rank_{rank:03d}_iou{r['iou']:.4f}_{r['video']}_t{t}.jpg")
        draw_overlay(sample, t, pred, gt, r["iou"], out_path)

    print(f"  Saved {len(worst100)} overlays to {OVERLAY_DIR}")

    # ── Error type summary ─────────────────────────────────────────
    print("\n[6/7] Error type summary ...")
    for name, records in all_records.items():
        es = error_type_summary(records)
        csv_path = os.path.join(OUT_DIR, f"error_type_summary_{name.replace('-','_').lower()}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["error_type", "count", "pct"])
            for k, v in es.items():
                w.writerow([k, v["count"], v["pct"]])
        print(f"\n  {name} error distribution:")
        for k, v in sorted(es.items(), key=lambda x: -x[1]["count"]):
            if not k.startswith("size_"):
                print(f"    {k}: {v['count']} ({v['pct']}%)")

    # ── Train/val distribution ─────────────────────────────────────
    print("\n[7/7] Train/val distribution comparison ...")
    train_stats, val_stats = dist_analysis(train_ds, val_ds, val_idx)

    for label, stats in [("Train", train_stats), ("Val", val_stats)]:
        areas = np.array(stats["areas"])
        ars = np.array(stats["ars"])
        print(f"\n  {label} ({stats['n_samples']} samples):")
        print(f"    Sources: {stats['sources']}")
        print(f"    BBox area: mean={areas.mean():.4f} median={np.median(areas):.4f} "
              f"p10={np.percentile(areas, 10):.4f} p90={np.percentile(areas, 90):.4f}")
        print(f"    Aspect ratio: mean={ars.mean():.3f} median={np.median(ars):.3f} "
              f"p10={np.percentile(ars, 10):.3f} p90={np.percentile(ars, 90):.3f}")

    # Check for significant differences
    t_area, v_area = np.array(train_stats["areas"]), np.array(val_stats["areas"])
    t_ar, v_ar = np.array(train_stats["ars"]), np.array(val_stats["ars"])
    print(f"\n  Distribution differences:")
    print(f"    Area mean: train={t_area.mean():.4f} val={v_area.mean():.4f} "
          f"(diff={v_area.mean()-t_area.mean():.4f})")
    print(f"    AR mean: train={t_ar.mean():.3f} val={v_ar.mean():.3f} "
          f"(diff={v_ar.mean()-t_ar.mean():.3f})")

    # ── Systematic bias ────────────────────────────────────────────
    for name, records in all_records.items():
        bias = systematic_bias(records)
        print(f"\n  {name} systematic biases:")
        print(f"    Area ratio: mean={bias['area_ratio_mean']:.3f} median={bias['area_ratio_median']:.3f}")
        print(f"    Pred >1.5x GT: {bias['pred_larger_ratio']:.1%}")
        print(f"    Pred <0.67x GT: {bias['pred_smaller_ratio']:.1%}")
        print(f"    Center error: mean={bias['center_err_mean']:.4f} median={bias['center_err_median']:.4f}")
        print(f"    Width error mean: {bias['w_err_mean']:.4f}  Height error mean: {bias['h_err_mean']:.4f}")
        print(f"    IoU by size: {bias['iou_by_size']}")
        print(f"    IoU by AR: {bias['iou_by_ar']}")

    # ── E-31 vs E-40 comparison ────────────────────────────────────
    if "E-31" in all_records and "E-40" in all_records:
        print(f"\n{'='*60}")
        print("E-31 vs E-40 Comparison")
        print(f"{'='*60}")
        r31 = all_records["E-31"]
        r40 = all_records["E-40"]
        assert len(r31) == len(r40), "Mismatched record counts"

        # Per-frame IoU diff
        diffs = [r40[i]["iou"] - r31[i]["iou"] for i in range(len(r31))]
        print(f"  mIoU E-31: {np.mean([r['iou'] for r in r31]):.4f}")
        print(f"  mIoU E-40: {np.mean([r['iou'] for r in r40]):.4f}")
        print(f"  E-40 - E-31 mean diff: {np.mean(diffs):.4f}")
        print(f"  E-40 better frames: {sum(1 for d in diffs if d > 0.01)} ({100*sum(1 for d in diffs if d > 0.01)/len(diffs):.1f}%)")
        print(f"  E-40 worse frames:  {sum(1 for d in diffs if d < -0.01)} ({100*sum(1 for d in diffs if d < -0.01)/len(diffs):.1f}%)")
        print(f"  Within ±0.01:       {sum(1 for d in diffs if -0.01 <= d <= 0.01)} ({100*sum(1 for d in diffs if -0.01 <= d <= 0.01)/len(diffs):.1f}%)")

        # Worst video overlap
        v31 = {v["video"]: v["mIoU"] for v in per_video_metrics(r31)}
        v40 = {v["video"]: v["mIoU"] for v in per_video_metrics(r40)}
        worst31 = set(list(dict(sorted(v31.items(), key=lambda x: x[1])).keys())[:20])
        worst40 = set(list(dict(sorted(v40.items(), key=lambda x: x[1])).keys())[:20])
        overlap = worst31 & worst40
        print(f"\n  Worst-20 video overlap: {len(overlap)}/20")
        if overlap:
            print(f"  Shared worst videos: {overlap}")

        # R@0.5 comparison
        r05_31 = np.mean([r["iou"] >= 0.5 for r in r31])
        r05_40 = np.mean([r["iou"] >= 0.5 for r in r40])
        print(f"  R@0.5 E-31: {r05_31:.4f}  E-40: {r05_40:.4f}  diff: {r05_40-r05_31:+.4f}")

        # Error type shifts
        e31_types = defaultdict(int)
        e40_types = defaultdict(int)
        for r in r31: e31_types[r["err_type"]] += 1
        for r in r40: e40_types[r["err_type"]] += 1
        for et in sorted(set(list(e31_types.keys()) + list(e40_types.keys()))):
            c31 = e31_types.get(et, 0)
            c40 = e40_types.get(et, 0)
            print(f"  {et}: E-31={c31} E-40={c40} diff={c40-c31:+d}")

        # Per-size IoU comparison
        for sz in ["tiny", "small", "medium", "large"]:
            i31 = [r["iou"] for r in r31 if r["size_cat"] == sz]
            i40 = [r["iou"] for r in r40 if r["size_cat"] == sz]
            if i31 and i40:
                print(f"  {sz}: E-31 mIoU={np.mean(i31):.4f} E-40={np.mean(i40):.4f} diff={np.mean(i40)-np.mean(i31):+.4f}")

        # R@0.5 gap: E-40 is higher at 8ep (0.248 vs 0.196) — check at 30ep
        print(f"\n  R@0.5 analysis:")
        for thresh in [0.3, 0.5, 0.7]:
            r31_r = np.mean([r["iou"] >= thresh for r in r31])
            r40_r = np.mean([r["iou"] >= thresh for r in r40])
            print(f"    R@{thresh}: E-31={r31_r:.4f} E-40={r40_r:.4f} diff={r40_r-r31_r:+.4f}")

    print(f"\n{'='*60}")
    print(f"E-41 audit complete. Output: {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
