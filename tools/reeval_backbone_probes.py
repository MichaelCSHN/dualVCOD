"""Backbone capacity probe unified re-evaluation: E-45, E-50, E-51.

Compares MobileNetV3-Small baseline against MV3-Large and EfficientNet-B0
— all with hard dense_fg_aux supervision. Same np.random.RandomState(42)
val split as Teacher route reeval for cross-comparability.
"""
import sys, os, time, json
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import MicroVCOD
from src.dataset_real import RealVideoBBoxDataset, collate_video_clips
from eval.eval_video_bbox import compute_metrics, compute_per_frame_metrics, bbox_iou, count_parameters

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T = 5
INPUT_SIZE = 224
BS = 64
NUM_WORKERS = 2
VAL_DATASET = r"C:\datasets\MoCA"
VAL_RATIO = 0.2
SPLIT_SEED = 42
RESIZED_ROOT = r"C:\datasets_224"

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


def load_model(ckpt_path, head_type, backbone_name):
    model = MicroVCOD(T=T, backbone_name=backbone_name, head_type=head_type)
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
    N, Tp, _ = preds.shape
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


def compute_fps(model):
    dummy = torch.randn(1, T, 3, INPUT_SIZE, INPUT_SIZE, device=DEVICE)
    for _ in range(5):
        _ = model(dummy)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(30):
        _ = model(dummy)
    torch.cuda.synchronize()
    return round(30 / max(time.time() - t0, 0.001), 1)


def main():
    t0_total = time.time()
    print("=" * 70)
    print("Backbone Capacity Probe Unified Re-evaluation")
    print("E-45 (MV3-Small baseline) vs E-50 (MV3-Large) vs E-51 (EffB0)")
    print("=" * 70)

    # Load val dataset
    print("\n[1/3] Loading val dataset ...")
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
        {"name": "E-45", "ckpt": r"D:\dualvcod\local_runs\autoresearch\expl_45_baseline_hard_8ep\checkpoint_best.pth", "head": "dense_fg_aux", "backbone": "mobilenet_v3_small"},
        {"name": "E-50", "ckpt": r"D:\dualvcod\local_runs\autoresearch\expl_50_mv3large_densefg_8ep\checkpoint_best.pth", "head": "dense_fg_aux", "backbone": "mobilenet_v3_large"},
        {"name": "E-51", "ckpt": r"D:\dualvcod\local_runs\autoresearch\expl_51_effb0_densefg_8ep\checkpoint_best.pth", "head": "dense_fg_aux", "backbone": "efficientnet_b0"},
    ]

    results = {}
    for trial in trials:
        ckpt_path = trial["ckpt"]
        if not os.path.exists(ckpt_path):
            print(f"\n[2/3] {trial['name']}: SKIP — checkpoint not found: {ckpt_path}")
            continue
        print(f"\n[2/3] Evaluating {trial['name']} ({trial['backbone']}) ...")
        model, epoch = load_model(trial["ckpt"], trial["head"], trial["backbone"])
        n_params = count_parameters(model)
        t0 = time.time()
        preds, gts = run_inference(model, val_loader)
        t_inf = time.time() - t0
        N, Tp, _ = preds.shape

        pf = compute_per_frame_metrics(preds, gts)
        per_vid = compute_per_video_miou(preds, gts, val_ds, val_idx, ds)
        fps = compute_fps(model)

        results[trial["name"]] = {
            "checkpoint_epoch": epoch,
            "backbone": trial["backbone"],
            "n_params": n_params,
            "fps": fps,
            **pf,
            "per_video_mIoU": per_vid,
            "inference_time_s": t_inf,
        }
        print(f"  params={n_params:,}  epoch={epoch}  pf_mIoU={pf['per_frame_mIoU']:.4f}  bad={pf['bad_frame_rate']:.4f}  R@0.5={pf['R@0.5']:.4f}  fps={fps}  time={t_inf:.1f}s")

    # Only print tables if we have at least E-45 + one probe
    available = [t["name"] for t in trials if t["name"] in results]
    if len(available) < 2 or "E-45" not in results:
        print("\nInsufficient results for comparison (need E-45 + at least one probe).")
        return

    col_w = 10
    all_names = ["E-45", "E-50", "E-51"]
    active = [n for n in all_names if n in results]

    # Build header with deltas
    delta_cols = []
    for n in active:
        if n != "E-45":
            delta_cols.append(f"{n}-E45")
    header_names = active + delta_cols
    header = f"{'Metric':<28} " + " ".join(f"{n:>{col_w}}" for n in header_names)
    sep = "-" * len(header)

    def print_table(title, rows, fmt_fn=None):
        print(f"\n  -- {title} --")
        print(header)
        print(sep)
        for label, key in rows:
            vals = {n: results[n][key] for n in active}
            formatted = [f"{vals[n]:{col_w}.4f}" if isinstance(vals[n], float) else f"{str(vals[n]):>{col_w}}" for n in active]
            for n in active:
                if n != "E-45":
                    if isinstance(vals[n], float) and isinstance(vals["E-45"], float):
                        d = vals[n] - vals["E-45"]
                        formatted.append(f"{d:+{col_w}.4f}")
                    else:
                        formatted.append(f"{'N/A':>{col_w}}")
            print(f"{label:<28} " + " ".join(formatted))

    # 1) Core metrics
    print_table("Core Metrics", [
        ("per_frame_mIoU", "per_frame_mIoU"),
        ("bad_frame_rate", "bad_frame_rate"),
        ("R@0.5", "R@0.5"),
        ("checkpoint_epoch", "checkpoint_epoch"),
    ])

    # 1b) Model info
    print_table("Model Info", [
        ("n_params", "n_params"),
        ("inference_fps", "fps"),
    ])

    # 2) Size-stratified IoU
    print_table("IoU by Object Size", [
        ("IoU_tiny", "IoU_tiny"),
        ("IoU_small", "IoU_small"),
        ("IoU_medium", "IoU_medium"),
        ("IoU_large", "IoU_large"),
    ])

    # 3) Area ratio & center error
    print_table("Area Ratio & Center Error", [
        ("area_ratio_mean", "area_ratio_mean"),
        ("area_ratio_median", "area_ratio_median"),
        ("center_error_mean", "center_error_mean"),
        ("center_error_median", "center_error_median"),
    ])

    # 4) Error classification
    print_table("Error Classification", [
        ("n_good", "n_good"),
        ("n_pred_too_large", "n_pred_too_large"),
        ("n_pred_too_small", "n_pred_too_small"),
        ("n_center_shift", "n_center_shift"),
        ("n_scale_mismatch", "n_scale_mismatch"),
        ("total_frames", "total_frames"),
    ])

    # 5) Per-video mIoU
    print(f"\n  -- Per-Video mIoU --")
    print(header)
    print(sep)
    for vn in KEY_VIDEOS:
        vals = {}
        for n in active:
            pv = results[n]["per_video_mIoU"].get(vn)
            vals[n] = pv if pv is not None else float('nan')
        formatted = [f"{vals[n]:{col_w}.4f}" if not np.isnan(vals[n]) else f"{'N/A':>{col_w}}" for n in active]
        for n in active:
            if n != "E-45":
                if not np.isnan(vals[n]) and not np.isnan(vals["E-45"]):
                    d = vals[n] - vals["E-45"]
                    formatted.append(f"{d:+{col_w}.4f}")
                else:
                    formatted.append(f"{'N/A':>{col_w}}")
        print(f"{vn:<30} " + " ".join(formatted))

    # 6) Training log vs Reeval
    print(f"\n  -- Training Log vs Reeval --")
    print(f"  {'Trial':<8} {'Backbone':<22} {'Train best_val_mIoU':>22} {'Reeval pf_mIoU':>18} {'Ratio':>8}")
    print(f"  {'-'*78}")
    for n in active:
        meta_path = os.path.join(os.path.dirname(trials[all_names.index(n)]["ckpt"]), "metadata.json")
        train_miou = None
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            train_miou = meta.get("best_val_miou")
        rpm = results[n]["per_frame_mIoU"]
        if train_miou is not None:
            ratio = rpm / train_miou if train_miou > 0 else float('nan')
            print(f"  {n:<8} {results[n]['backbone']:<22} {train_miou:22.4f} {rpm:18.4f} {ratio:8.2f}x")
        else:
            print(f"  {n:<8} {results[n]['backbone']:<22} {'N/A':>22} {rpm:18.4f}")

    total_time = time.time() - t0_total
    print(f"\nReeval complete in {total_time:.1f}s.")
    print("=" * 70)

    # Save raw results
    out_path = r"D:\dualvcod\local_runs\reeval_backbone_probes.json"
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
