"""E-41.5: Unified re-evaluation of E-31, E-39, E-40 with dual metrics.

Computes BOTH:
  (A) Per-sample mIoU   — original training metric: mean of per-clip mean IoU
  (B) Per-frame mIoU    — new metric: mean across all (N*T) individual frames
  (C) Full per-frame diagnostics

Uses the EXACT same val split as training (MoCA, seed=42, val_ratio=0.2).
"""
import sys, os, time, json
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import MicroVCOD
from src.dataset_real import RealVideoBBoxDataset, collate_video_clips
from eval.eval_video_bbox import compute_metrics, compute_per_frame_metrics, bbox_iou

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T = 5
INPUT_SIZE = 224
BS = 64
NUM_WORKERS = 2
VAL_DATASET = r"C:\datasets\MoCA"
VAL_RATIO = 0.2
SPLIT_SEED = 42
RESIZED_ROOT = r"C:\datasets_224"


def _video_name_from_sample(sample):
    d = sample.get("frame_dir", "")
    parts = d.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p in ("JPEGImages", "Imgs", "frames", "TrainDataset_per_sq"):
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
    val_idx = [i for v in val_videos for i in video_to_indices[v]]
    return val_idx


def load_model(ckpt_path, head_type):
    model = MicroVCOD(T=T, backbone_name="mobilenet_v3_small", head_type=head_type)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


def run_inference(model, loader):
    all_preds, all_gts = [], []
    for frames, gt_bboxes in loader:
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                pred = model(frames.to(DEVICE))
        all_preds.append(pred.float().cpu())
        all_gts.append(gt_bboxes)
    preds = torch.cat(all_preds, dim=0)  # (N, T, 4)
    gts = torch.cat(all_gts, dim=0)
    return preds, gts


def compute_per_sample_miou(preds, gts):
    """Original training metric: mean of per-sample (per-clip) mean IoU."""
    N, T, _ = preds.shape
    sample_mious = []
    for i in range(N):
        t_ious = []
        for t in range(T):
            t_ious.append(float(bbox_iou(preds[i, t], gts[i, t])))
        sample_mious.append(np.mean(t_ious))
    return float(np.mean(sample_mious))


def compute_per_frame_miou(preds, gts):
    """Per-frame mIoU: mean of all N*T individual IoUs."""
    N, T, _ = preds.shape
    all_ious = []
    for i in range(N):
        for t in range(T):
            all_ious.append(float(bbox_iou(preds[i, t], gts[i, t])))
    return float(np.mean(all_ious))


def main():
    print("=" * 60)
    print("E-41.5: Unified Re-evaluation — E-31 vs E-39 vs E-40")
    print("=" * 60)

    # Load val dataset (exact same as training)
    print("\n[1/4] Loading val dataset (MoCA, seed=42, val_ratio=0.2) ...")
    ds = RealVideoBBoxDataset(
        [VAL_DATASET], T=T, target_size=INPUT_SIZE,
        augment=False, temporal_stride=1, resized_root=RESIZED_ROOT,
        return_mask=False
    )
    val_idx = split_by_video(ds, VAL_RATIO, SPLIT_SEED)
    val_ds = Subset(ds, val_idx)
    val_loader = DataLoader(
        val_ds, batch_size=BS, shuffle=False,
        collate_fn=collate_video_clips, num_workers=NUM_WORKERS,
        pin_memory=True
    )
    print(f"  Val samples: {len(val_ds)}  Videos: {len(set(_video_name_from_sample(ds.samples[i]) for i in val_idx))}")
    print(f"  Val frames: {len(val_ds) * T}")

    # Define trials to evaluate
    trials = [
        {
            "name": "E-31", "trial_id": "expl_31_diou_30ep",
            "ckpt": r"D:\dualvcod\local_runs\autoresearch\expl_31_diou_30ep\checkpoint_best.pth",
            "head": "direct_bbox",
        },
        {
            "name": "E-39", "trial_id": "expl_39_densefg_8ep",
            "ckpt": r"D:\dualvcod\local_runs\autoresearch\expl_39_densefg_8ep\checkpoint_best.pth",
            "head": "dense_fg_aux",
        },
        {
            "name": "E-40", "trial_id": "expl_40_densefg_30ep",
            "ckpt": r"D:\dualvcod\local_runs\autoresearch\expl_40_densefg_30ep\checkpoint_best.pth",
            "head": "dense_fg_aux",
        },
    ]

    results = {}
    for trial in trials:
        print(f"\n[2/4] Evaluating {trial['name']} ({trial['head']}) ...")
        model = load_model(trial["ckpt"], trial["head"])
        t0 = time.time()
        preds, gts = run_inference(model, val_loader)
        t_inf = time.time() - t0

        # (A) Per-sample mIoU (original training metric)
        ps_miou = compute_per_sample_miou(preds, gts)

        # (B) Per-frame mIoU
        pf_miou = compute_per_frame_miou(preds, gts)

        # (C) Full per-frame diagnostics
        pf = compute_per_frame_metrics(preds, gts)

        results[trial["name"]] = {
            "per_sample_mIoU": ps_miou,
            "per_frame_mIoU": pf_miou,
            **pf,
            "inference_time_s": t_inf,
        }
        print(f"  per-sample mIoU: {ps_miou:.4f}  per-frame mIoU: {pf_miou:.4f}  time: {t_inf:.1f}s")

    # ── Comparison table ──
    print(f"\n{'=' * 90}")
    print("  COMPARISON TABLE")
    print(f"{'=' * 90}")

    header = f"{'Metric':<30} {'E-31':>10} {'E-39':>10} {'E-40':>10} {'E-40 vs E-31':>15}"
    print(header)
    print("-" * 75)

    metrics_rows = [
        ("per-sample mIoU (old)", "per_sample_mIoU", ".4f", ""),
        ("per-frame mIoU (new)", "per_frame_mIoU", ".4f", ""),
        ("bad_frame_rate", "bad_frame_rate", ".3f", ""),
        ("R@0.5", "R@0.5", ".4f", ""),
        ("IoU tiny", "IoU_tiny", ".4f", "+"),
        ("IoU small", "IoU_small", ".4f", "+"),
        ("IoU medium", "IoU_medium", ".4f", "+"),
        ("IoU large", "IoU_large", ".4f", "+"),
        ("area_ratio_mean", "area_ratio_mean", ".4f", ""),
        ("center_error_mean", "center_error_mean", ".4f", ""),
        ("width_error_mean", "width_error_mean", ".4f", ""),
        ("height_error_mean", "height_error_mean", ".4f", ""),
        ("n_good", "n_good", "", ""),
        ("n_pred_too_large", "n_pred_too_large", "", ""),
        ("n_pred_too_small", "n_pred_too_small", "", ""),
        ("n_center_shift", "n_center_shift", "", ""),
        ("n_scale_mismatch", "n_scale_mismatch", "", ""),
        ("total_frames", "total_frames", "", ""),
    ]

    for label, key, fmt, direction in metrics_rows:
        v31 = results["E-31"][key]
        v39 = results["E-39"][key]
        v40 = results["E-40"][key]

        if direction == "+":
            diff = v40 - v31
            diff_str = f"+{diff:{fmt}}" if diff >= 0 else f"{diff:{fmt}}"
        else:
            diff = v40 - v31
            diff_str = f"{diff:+{fmt}}" if fmt else f"{v40 - v31:+d}"

        if isinstance(v31, float):
            print(f"{label:<30} {v31:10{fmt}} {v39:10{fmt}} {v40:10{fmt}} {diff_str:>15}")
        else:
            print(f"{label:<30} {v31:10} {v39:10} {v40:10} {diff_str:>15}")

    # ── Decide: is dense_fg_aux the new default? ──
    print(f"\n{'=' * 60}")
    print("  DECISION: dense_fg_aux as default training head?")
    print(f"{'=' * 60}")
    e31_pf = results["E-31"]["per_frame_mIoU"]
    e40_pf = results["E-40"]["per_frame_mIoU"]
    e39_pf = results["E-39"]["per_frame_mIoU"]
    e31_bad = results["E-31"]["bad_frame_rate"]
    e40_bad = results["E-40"]["bad_frame_rate"]
    e40_tiny = results["E-40"]["IoU_tiny"]
    e31_tiny = results["E-31"]["IoU_tiny"]

    print(f"  E-31 per-frame mIoU: {e31_pf:.4f}  bad={e31_bad:.3f}  tiny={e31_tiny:.4f}")
    print(f"  E-39 per-frame mIoU: {e39_pf:.4f} (8ep)")
    print(f"  E-40 per-frame mIoU: {e40_pf:.4f}  bad={e40_bad:.3f}  tiny={e40_tiny:.4f}")

    if e40_pf > e31_pf and e40_bad < e31_bad and e40_tiny > e31_tiny:
        print(f"\n  VERDICT: dense_fg_aux SHOULD be the default training head.")
        print(f"  - Per-frame mIoU: +{e40_pf - e31_pf:.4f}")
        print(f"  - Bad frame rate: {e40_bad - e31_bad:+.3f}")
        print(f"  - Tiny IoU: +{e40_tiny - e31_tiny:.4f}")
        print(f"  - Inference params: identical (dense head not executed)")
    else:
        print(f"\n  VERDICT: INCONCLUSIVE — check individual metrics.")

    # Also check E-39 at 8ep
    if e39_pf > 0.7:
        print(f"\n  E-39 at 8ep achieves {e39_pf:.4f} per-frame mIoU — {e39_pf / e40_pf * 100:.1f}% of E-40 30ep in 26.7% epochs.")
        print(f"  For fast prototyping, 8ep dense_fg_aux is the most sample-efficient option.")

    print(f"\nE-41.5 complete.")


if __name__ == "__main__":
    main()
