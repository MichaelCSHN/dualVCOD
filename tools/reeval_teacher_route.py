"""Teacher route unified re-evaluation: E-39, E-40, E-45, E-46, E-47.

Computes per-frame mIoU, bad_frame_rate, R@0.5, size-stratified IoU,
area_ratio, center_error, error counts, and per-video mIoU for key videos.

Uses the SAME val split as training: np.random.RandomState(42), val_ratio=0.2.
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

# Videos to report per-video mIoU
KEY_VIDEOS = [
    "flatfish_2",
    "white_tailed_ptarmigan",
    "pygmy_seahorse_0",
]


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
    return val_idx, val_videos


def load_model(ckpt_path, head_type):
    model = MicroVCOD(T=T, backbone_name="mobilenet_v3_small", head_type=head_type)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    epoch = ckpt.get("epoch", "?")
    return model, epoch


def run_inference(model, loader):
    all_preds, all_gts = [], []
    for frames, gt_bboxes in loader:
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                pred = model(frames.to(DEVICE))
        all_preds.append(pred.float().cpu())
        all_gts.append(gt_bboxes)
    preds = torch.cat(all_preds, dim=0)
    gts = torch.cat(all_gts, dim=0)
    return preds, gts


def compute_per_video_miou(preds, gts, val_ds, val_idx, ds):
    """Compute per-video per-frame mIoU for specific videos."""
    N, Tp, _ = preds.shape
    # Map each prediction to its video
    video_ious = defaultdict(list)
    for i in range(N):
        sample = ds.samples[val_idx[i]]
        vn = _video_name_from_sample(sample)
        for t in range(Tp):
            iou_val = float(bbox_iou(preds[i, t], gts[i, t]))
            video_ious[vn].append(iou_val)

    result = {}
    for vn in KEY_VIDEOS:
        if vn in video_ious:
            result[vn] = float(np.mean(video_ious[vn]))
        else:
            result[vn] = None
    return result


def main():
    t0_total = time.time()
    print("=" * 70)
    print("Teacher Route Unified Re-evaluation")
    print("E-39, E-40 (ref) + E-45, E-46, E-47, E-48, E-49 (Teacher route)")
    print("=" * 70)

    # Load val dataset
    print("\n[1/3] Loading val dataset (MoCA, np.random.RandomState(42), val_ratio=0.2) ...")
    ds = RealVideoBBoxDataset(
        [VAL_DATASET], T=T, target_size=INPUT_SIZE,
        augment=False, temporal_stride=1, resized_root=RESIZED_ROOT,
        return_mask=False
    )
    val_idx, val_videos = split_by_video(ds, VAL_RATIO, SPLIT_SEED)
    val_ds = Subset(ds, val_idx)
    val_loader = DataLoader(
        val_ds, batch_size=BS, shuffle=False,
        collate_fn=collate_video_clips, num_workers=NUM_WORKERS,
        pin_memory=True
    )
    print(f"  Val samples: {len(val_ds)}  Videos: {len(val_videos)}")
    print(f"  Val frames: {len(val_ds) * T}")

    trials = [
        {"name": "E-39", "ckpt": r"D:\dualvcod\local_runs\autoresearch\expl_39_densefg_8ep\checkpoint_best.pth", "head": "dense_fg_aux"},
        {"name": "E-40", "ckpt": r"D:\dualvcod\local_runs\autoresearch\expl_40_densefg_30ep\checkpoint_best.pth", "head": "dense_fg_aux"},
        {"name": "E-45", "ckpt": r"D:\dualvcod\local_runs\autoresearch\expl_45_baseline_hard_8ep\checkpoint_best.pth", "head": "dense_fg_aux"},
        {"name": "E-46", "ckpt": r"D:\dualvcod\local_runs\autoresearch\expl_46_softmask_8ep\checkpoint_best.pth", "head": "dense_fg_aux"},
        {"name": "E-47", "ckpt": r"D:\dualvcod\local_runs\autoresearch\expl_47_softbbox_8ep\checkpoint_best.pth", "head": "dense_fg_aux"},
        {"name": "E-48", "ckpt": r"D:\dualvcod\local_runs\autoresearch\expl_48_size_adaptive_8ep\checkpoint_best.pth", "head": "dense_fg_aux"},
        {"name": "E-49", "ckpt": r"D:\dualvcod\local_runs\autoresearch\expl_49_center_extent_8ep\checkpoint_best.pth", "head": "dense_ce_aux"},
    ]

    results = {}
    for trial in trials:
        print(f"\n[2/3] Evaluating {trial['name']} ...")
        model, epoch = load_model(trial["ckpt"], trial["head"])
        t0 = time.time()
        preds, gts = run_inference(model, val_loader)
        t_inf = time.time() - t0
        N, Tp, _ = preds.shape

        pf = compute_per_frame_metrics(preds, gts)
        per_vid = compute_per_video_miou(preds, gts, val_ds, val_idx, ds)

        results[trial["name"]] = {
            "checkpoint_epoch": epoch,
            **pf,
            "per_video_mIoU": per_vid,
            "inference_time_s": t_inf,
        }
        print(f"  epoch={epoch}  pf_mIoU={pf['per_frame_mIoU']:.4f}  bad={pf['bad_frame_rate']:.4f}  R@0.5={pf['R@0.5']:.4f}  time={t_inf:.1f}s")

    # ── Print comprehensive comparison ──
    print(f"\n{'=' * 120}")
    print("  COMPREHENSIVE COMPARISON TABLE")
    print(f"{'=' * 120}")

    # Column set: E-39 E-40 E-45 E-46 E-48 E-47 E46-E45 E48-E45 E47-E45
    col_w = 10

    # (1) Core metrics
    print(f"\n  -- Core Metrics --")
    core_rows = [
        ("per_frame_mIoU", "per_frame_mIoU", ".4f"),
        ("bad_frame_rate", "bad_frame_rate", ".4f"),
        ("R@0.5", "R@0.5", ".4f"),
        ("checkpoint_epoch", "checkpoint_epoch", ""),
    ]
    header = (f"{'Metric':<28} {'E-39':>{col_w}} {'E-40':>{col_w}} {'E-45':>{col_w}} "
              f"{'E-46':>{col_w}} {'E-48':>{col_w}} {'E-47':>{col_w}} "
              f"{'E46-E45':>{col_w}} {'E48-E45':>{col_w}} {'E47-E45':>{col_w}}")
    print(header)
    print("-" * len(header))
    for label, key, fmt in core_rows:
        vals = {t["name"]: results[t["name"]][key] for t in trials}
        if isinstance(vals["E-39"], float):
            d46 = vals["E-46"] - vals["E-45"]
            d48 = vals["E-48"] - vals["E-45"]
            d47 = vals["E-47"] - vals["E-45"]
            print(f"{label:<28} {vals['E-39']:{col_w}{fmt}} {vals['E-40']:{col_w}{fmt}} "
                  f"{vals['E-45']:{col_w}{fmt}} {vals['E-46']:{col_w}{fmt}} "
                  f"{vals['E-48']:{col_w}{fmt}} {vals['E-47']:{col_w}{fmt}} "
                  f"{d46:+{col_w}{fmt}} {d48:+{col_w}{fmt}} {d47:+{col_w}{fmt}}")
        else:
            s39 = str(vals["E-39"]); s40 = str(vals["E-40"]); s45 = str(vals["E-45"])
            s46 = str(vals["E-46"]); s48 = str(vals["E-48"]); s47 = str(vals["E-47"])
            print(f"{label:<28} {s39:>{col_w}} {s40:>{col_w}} {s45:>{col_w}} "
                  f"{s46:>{col_w}} {s48:>{col_w}} {s47:>{col_w}}")

    # (2) Size-stratified IoU
    print(f"\n  -- IoU by Object Size --")
    size_rows = [("IoU_tiny", "IoU_tiny"), ("IoU_small", "IoU_small"),
                 ("IoU_medium", "IoU_medium"), ("IoU_large", "IoU_large")]
    print(header)
    print("-" * len(header))
    for label, key in size_rows:
        vals = {t["name"]: results[t["name"]][key] for t in trials}
        d46 = vals["E-46"] - vals["E-45"]
        d48 = vals["E-48"] - vals["E-45"]
        d47 = vals["E-47"] - vals["E-45"]
        print(f"{label:<28} {vals['E-39']:{col_w}.4f} {vals['E-40']:{col_w}.4f} "
              f"{vals['E-45']:{col_w}.4f} {vals['E-46']:{col_w}.4f} "
              f"{vals['E-48']:{col_w}.4f} {vals['E-47']:{col_w}.4f} "
              f"{d46:+{col_w}.4f} {d48:+{col_w}.4f} {d47:+{col_w}.4f}")

    # (3) Area ratio & center error
    print(f"\n  -- Area Ratio & Center Error --")
    ar_rows = [
        ("area_ratio_mean", "area_ratio_mean", ".4f"),
        ("area_ratio_median", "area_ratio_median", ".4f"),
        ("center_error_mean", "center_error_mean", ".4f"),
        ("center_error_median", "center_error_median", ".4f"),
    ]
    print(header)
    print("-" * len(header))
    for label, key, fmt in ar_rows:
        vals = {t["name"]: results[t["name"]][key] for t in trials}
        d46 = vals["E-46"] - vals["E-45"]
        d48 = vals["E-48"] - vals["E-45"]
        d47 = vals["E-47"] - vals["E-45"]
        print(f"{label:<28} {vals['E-39']:{col_w}{fmt}} {vals['E-40']:{col_w}{fmt}} "
              f"{vals['E-45']:{col_w}{fmt}} {vals['E-46']:{col_w}{fmt}} "
              f"{vals['E-48']:{col_w}{fmt}} {vals['E-47']:{col_w}{fmt}} "
              f"{d46:+{col_w}{fmt}} {d48:+{col_w}{fmt}} {d47:+{col_w}{fmt}}")

    # (4) Error counts
    print(f"\n  -- Error Classification --")
    err_rows = [
        ("n_good", "n_good"),
        ("n_pred_too_large", "n_pred_too_large"),
        ("n_pred_too_small", "n_pred_too_small"),
        ("n_center_shift", "n_center_shift"),
        ("n_scale_mismatch", "n_scale_mismatch"),
        ("total_frames", "total_frames"),
    ]
    print(header)
    print("-" * len(header))
    for label, key in err_rows:
        vals = {t["name"]: results[t["name"]][key] for t in trials}
        d46 = vals["E-46"] - vals["E-45"]
        d48 = vals["E-48"] - vals["E-45"]
        d47 = vals["E-47"] - vals["E-45"]
        print(f"{label:<28} {vals['E-39']:>{col_w}} {vals['E-40']:>{col_w}} "
              f"{vals['E-45']:>{col_w}} {vals['E-46']:>{col_w}} "
              f"{vals['E-48']:>{col_w}} {vals['E-47']:>{col_w}} "
              f"{d46:+{col_w}} {d48:+{col_w}} {d47:+{col_w}}")

    # (5) Per-video mIoU
    print(f"\n  -- Per-Video mIoU (Key Videos) --")
    header5 = (f"{'Video':<30} {'E-39':>{col_w}} {'E-40':>{col_w}} {'E-45':>{col_w}} "
               f"{'E-46':>{col_w}} {'E-48':>{col_w}} {'E-47':>{col_w}} "
               f"{'E46-E45':>{col_w}} {'E48-E45':>{col_w}} {'E47-E45':>{col_w}}")
    print(header5)
    print("-" * len(header5))
    for vn in KEY_VIDEOS:
        vals = {}
        for t in trials:
            pv = results[t["name"]]["per_video_mIoU"].get(vn)
            vals[t["name"]] = pv if pv is not None else float('nan')
        if all(not np.isnan(vals[t["name"]]) for t in trials):
            d46 = vals["E-46"] - vals["E-45"]
            d48 = vals["E-48"] - vals["E-45"]
            d47 = vals["E-47"] - vals["E-45"]
            print(f"{vn:<30} {vals['E-39']:{col_w}.4f} {vals['E-40']:{col_w}.4f} "
                  f"{vals['E-45']:{col_w}.4f} {vals['E-46']:{col_w}.4f} "
                  f"{vals['E-48']:{col_w}.4f} {vals['E-47']:{col_w}.4f} "
                  f"{d46:+{col_w}.4f} {d48:+{col_w}.4f} {d47:+{col_w}.4f}")
        else:
            s39 = f"{vals['E-39']:.4f}" if not np.isnan(vals['E-39']) else "N/A"
            s40 = f"{vals['E-40']:.4f}" if not np.isnan(vals['E-40']) else "N/A"
            s45 = f"{vals['E-45']:.4f}" if not np.isnan(vals['E-45']) else "N/A"
            s46 = f"{vals['E-46']:.4f}" if not np.isnan(vals['E-46']) else "N/A"
            s48 = f"{vals['E-48']:.4f}" if not np.isnan(vals['E-48']) else "N/A"
            s47 = f"{vals['E-47']:.4f}" if not np.isnan(vals['E-47']) else "N/A"
            print(f"{vn:<30} {s39:>{col_w}} {s40:>{col_w}} {s45:>{col_w}} "
                  f"{s46:>{col_w}} {s48:>{col_w}} {s47:>{col_w}} {'N/A':>{col_w}} {'N/A':>{col_w}} {'N/A':>{col_w}}")

    # (6) Consistency check: training log vs reeval
    print(f"\n  -- Training Log vs Reeval Consistency --")
    train_log = {
        "E-45": 0.3046,
        "E-46": 0.3054,
        "E-47": 0.2993,
        "E-48": 0.2978,
    }
    print(f"  {'Trial':<8} {'Train best_val_mIoU':>22} {'Reeval pf_mIoU':>18} {'Delta':>10}")
    print(f"  {'-'*58}")
    for name in ["E-45", "E-46", "E-48", "E-47"]:
        tlm = train_log[name]
        rpm = results[name]["per_frame_mIoU"]
        delta = rpm - tlm
        print(f"  {name:<8} {tlm:22.4f} {rpm:18.4f} {delta:+10.4f}")

    total_time = time.time() - t0_total
    print(f"\nReeval complete in {total_time:.1f}s.")
    print("=" * 70)

    # Save raw results for report generation
    out_path = r"D:\dualvcod\local_runs\reeval_teacher_route.json"
    # Convert numpy values
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)
    print(f"\nRaw results saved to: {out_path}")


if __name__ == "__main__":
    main()
