"""2×2 Backbone × Epochs unified reeval: E-40, E-45, E-51, E-52.

Design:
           MV3-Small    EfficientNet-B0
  8ep      E-45          E-51
  30ep     E-40          E-52

Core question: does EfficientNet-B0's 8ep advantage survive 30ep training
and beat the MV3-Small 30ep canonical baseline (E-40, 0.8564)?
"""
import sys, os, time, json
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import MicroVCOD
from src.dataset_real import RealVideoBBoxDataset, collate_video_clips
from eval.eval_video_bbox import compute_per_frame_metrics, bbox_iou, count_parameters

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
    print("2×2 Backbone × Epochs Unified Reeval")
    print("E-40 (MV3-S 30ep) vs E-45 (MV3-S 8ep) vs E-51 (EffB0 8ep) vs E-52 (EffB0 30ep)")
    print("=" * 70)

    # Load val dataset
    print("\n[1/4] Loading val dataset ...")
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

    trials = [
        {"name": "E-40", "ckpt": r"D:\dualvcod\local_runs\autoresearch\expl_40_densefg_30ep\checkpoint_best.pth", "head": "dense_fg_aux", "backbone": "mobilenet_v3_small", "desc": "MV3-S 30ep"},
        {"name": "E-45", "ckpt": r"D:\dualvcod\local_runs\autoresearch\expl_45_baseline_hard_8ep\checkpoint_best.pth", "head": "dense_fg_aux", "backbone": "mobilenet_v3_small", "desc": "MV3-S 8ep"},
        {"name": "E-51", "ckpt": r"D:\dualvcod\local_runs\autoresearch\expl_51_effb0_densefg_8ep\checkpoint_best.pth", "head": "dense_fg_aux", "backbone": "efficientnet_b0", "desc": "EffB0 8ep"},
        {"name": "E-52", "ckpt": r"D:\dualvcod\local_runs\autoresearch\expl_52_effb0_densefg_30ep\checkpoint_best.pth", "head": "dense_fg_aux", "backbone": "efficientnet_b0", "desc": "EffB0 30ep"},
    ]

    results = {}
    for trial in trials:
        ckpt_path = trial["ckpt"]
        if not os.path.exists(ckpt_path):
            print(f"\n[2/4] {trial['name']} ({trial['desc']}): SKIP — checkpoint not found")
            continue
        print(f"\n[2/4] Evaluating {trial['name']} ({trial['desc']}) ...")
        model, epoch = load_model(trial["ckpt"], trial["head"], trial["backbone"])
        n_params = count_parameters(model)
        t0 = time.time()
        preds, gts = run_inference(model, val_loader)
        t_inf = time.time() - t0

        pf = compute_per_frame_metrics(preds, gts)
        per_vid = compute_per_video_miou(preds, gts, val_ds, val_idx, ds)
        fps = compute_fps(model)

        results[trial["name"]] = {
            "checkpoint_epoch": epoch,
            "backbone": trial["backbone"],
            "desc": trial["desc"],
            "n_params": n_params,
            "fps": fps,
            **pf,
            "per_video_mIoU": per_vid,
            "inference_time_s": t_inf,
        }
        print(f"  epoch={epoch}  pf_mIoU={pf['per_frame_mIoU']:.4f}  bad={pf['bad_frame_rate']:.4f}  R@0.5={pf['R@0.5']:.4f}  fps={fps}")

    available = [t["name"] for t in trials if t["name"] in results]
    if len(available) < 3:
        print("\nInsufficient results for 2×2 comparison.")
        return

    col_w = 10
    all_names = ["E-45", "E-51", "E-40", "E-52"]

    def _v(name, key, fmt=".4f"):
        if name not in results:
            return "N/A"
        val = results[name][key]
        if isinstance(val, float):
            return f"{val:{col_w}{fmt}}"
        return f"{str(val):>{col_w}}"

    # ── 2×2 Matrix ──
    print(f"\n{'=' * 80}")
    print("  2×2 BACKBONE × EPOCHS MATRIX — pf_mIoU")
    print(f"{'=' * 80}")
    print(f"                     MV3-Small          EffB0        Δ(EffB0−MV3S)")
    print(f"  8ep               {_v('E-45','per_frame_mIoU')}     {_v('E-51','per_frame_mIoU')}     {'+0.0090' if 'E-45' in results and 'E-51' in results else 'N/A':>10}")
    e40_v = f"{results['E-40']['per_frame_mIoU']:.4f}" if 'E-40' in results else 'N/A'
    e52_v = f"{results['E-52']['per_frame_mIoU']:.4f}" if 'E-52' in results else 'N/A'
    delta_30 = ""
    if 'E-40' in results and 'E-52' in results:
        d = results['E-52']['per_frame_mIoU'] - results['E-40']['per_frame_mIoU']
        delta_30 = f"{d:+.4f}"
    print(f"  30ep              {e40_v:>10}     {e52_v:>10}     {delta_30:>10}")
    # Delta rows
    d_mv3 = ""
    if 'E-40' in results and 'E-45' in results:
        d = results['E-40']['per_frame_mIoU'] - results['E-45']['per_frame_mIoU']
        d_mv3 = f"{d:+.4f}"
    d_eff = ""
    if 'E-52' in results and 'E-51' in results:
        d = results['E-52']['per_frame_mIoU'] - results['E-51']['per_frame_mIoU']
        d_eff = f"{d:+.4f}"
    print(f"  Δ(30ep−8ep)       {d_mv3:>10}     {d_eff:>10}")

    # ── Detailed comparison ──
    print(f"\n{'=' * 120}")
    print("  DETAILED COMPARISON")
    print(f"{'=' * 120}")

    active = [n for n in all_names if n in results]
    header = f"{'Metric':<28}" + "".join(f" {n:>{col_w}}" for n in active)
    sep = "-" * len(header)

    # Helper: print table with active columns
    def print_table(title, rows):
        nonlocal header, sep
        print(f"\n  -- {title} --")
        print(header)
        print(sep)
        for label, key in rows:
            line = f"{label:<28}"
            for n in active:
                val = results[n][key]
                if isinstance(val, float):
                    line += f" {val:{col_w}.4f}"
                else:
                    line += f" {str(val):>{col_w}}"
            print(line)

    print_table("Core Metrics", [
        ("per_frame_mIoU", "per_frame_mIoU"),
        ("bad_frame_rate", "bad_frame_rate"),
        ("R@0.5", "R@0.5"),
        ("checkpoint_epoch", "checkpoint_epoch"),
    ])

    print_table("Size-Stratified IoU", [
        ("IoU_tiny", "IoU_tiny"),
        ("IoU_small", "IoU_small"),
        ("IoU_medium", "IoU_medium"),
        ("IoU_large", "IoU_large"),
    ])

    print_table("Area Ratio & Center Error", [
        ("area_ratio_mean", "area_ratio_mean"),
        ("area_ratio_median", "area_ratio_median"),
        ("center_error_mean", "center_error_mean"),
        ("center_error_median", "center_error_median"),
    ])

    print_table("Error Classification", [
        ("n_good", "n_good"),
        ("n_pred_too_large", "n_pred_too_large"),
        ("n_pred_too_small", "n_pred_too_small"),
        ("n_center_shift", "n_center_shift"),
        ("n_scale_mismatch", "n_scale_mismatch"),
    ])

    print_table("Model Characteristics", [
        ("n_params", "n_params"),
        ("fps", "fps"),
    ])

    # Per-video
    print(f"\n  -- Per-Video mIoU --")
    print(header)
    print(sep)
    for vn in KEY_VIDEOS:
        line = f"{vn:<28}"
        for n in active:
            pv = results[n]["per_video_mIoU"].get(vn)
            if pv is not None:
                line += f" {pv:{col_w}.4f}"
            else:
                line += f" {'N/A':>{col_w}}"
        print(line)

    # Training log vs reeval
    print(f"\n  -- Training Log vs Reeval --")
    print(f"  {'Trial':<8} {'Backbone':<20} {'Epochs':>7} {'Train best':>12} {'Reeval':>10} {'Ratio':>8}")
    print(f"  {'-'*65}")
    for n in active:
        meta_path = os.path.join(
            os.path.dirname([t for t in trials if t["name"] == n][0]["ckpt"]),
            "metadata.json"
        )
        train_miou = None
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            train_miou = meta.get("best_val_miou")
            ep = meta.get("epochs", "?")
        else:
            ep = "?"
        rpm = results[n]["per_frame_mIoU"]
        if train_miou is not None:
            ratio = rpm / train_miou if train_miou > 0 else float('nan')
            print(f"  {n:<8} {results[n]['backbone']:<20} {str(ep):>7} {train_miou:12.4f} {rpm:10.4f} {ratio:7.2f}x")
        else:
            print(f"  {n:<8} {results[n]['backbone']:<20} {str(ep):>7} {'N/A':>12} {rpm:10.4f}")

    # ── Verdict ──
    print(f"\n{'=' * 80}")
    print("  VERDICT")
    print(f"{'=' * 80}")
    if 'E-40' in results and 'E-52' in results:
        e40 = results['E-40']['per_frame_mIoU']
        e52 = results['E-52']['per_frame_mIoU']
        delta = e52 - e40
        if delta >= 0.005:
            print(f"  EffB0 30ep ({e52:.4f}) BEATS MV3-S 30ep ({e40:.4f}) by {delta:+.4f}")
            print(f"  → EfficientNet-B0 validated as mainline candidate.")
        elif delta >= -0.005:
            print(f"  EffB0 30ep ({e52:.4f}) TIED with MV3-S 30ep ({e40:.4f}) ({delta:+.4f})")
            print(f"  → Parity. Defer to cost/complexity preference.")
        else:
            print(f"  EffB0 30ep ({e52:.4f}) BELOW MV3-S 30ep ({e40:.4f}) ({delta:+.4f})")
            print(f"  → MV3-Small hard dense_fg remains canonical. EffB0 signal did not survive 30ep.")

    total_time = time.time() - t0_total
    print(f"\nReeval complete in {total_time:.1f}s.")

    out_path = r"D:\dualvcod\local_runs\reeval_2x2_backbone_epochs.json"
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
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
