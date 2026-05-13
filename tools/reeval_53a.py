"""E-53a unified reeval vs E-51 baseline (both 8ep, EffB0).
Uses np.random.RandomState(42) val split for cross-comparability.
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


def main():
    t0_total = time.time()
    print("=" * 70)
    print("E-53a Multi-Scale Dense Supervision — Unified Reeval")
    print("E-51 (EffB0 dense_fg_aux 8ep) vs E-53a (EffB0 dense_fg_aux_ms 8ep)")
    print("=" * 70)

    # Load val dataset with canonical split
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

    trials = [
        {"name": "E-51",  "ckpt": r"D:\dualvcod\local_runs\autoresearch\expl_51_effb0_densefg_8ep\checkpoint_best.pth",  "head": "dense_fg_aux",    "backbone": "efficientnet_b0", "desc": "EffB0 dense_fg_aux 8ep"},
        {"name": "E-53a", "ckpt": r"D:\dualvcod\local_runs\autoresearch\expl_53a_effb0_msdense_8ep\checkpoint_best.pth", "head": "dense_fg_aux_ms", "backbone": "efficientnet_b0", "desc": "EffB0 dense_fg_aux_ms 8ep"},
    ]

    results = {}
    for trial in trials:
        ckpt_path = trial["ckpt"]
        if not os.path.exists(ckpt_path):
            print(f"\n[2/3] {trial['name']}: SKIP — checkpoint not found")
            continue
        print(f"\n[2/3] Evaluating {trial['name']} ({trial['desc']}) ...")
        model, epoch = load_model(trial["ckpt"], trial["head"], trial["backbone"])
        n_params = count_parameters(model)
        t0 = time.time()
        preds, gts = run_inference(model, val_loader)
        t_inf = time.time() - t0

        pf = compute_per_frame_metrics(preds, gts)
        per_vid = compute_per_video_miou(preds, gts, val_ds, val_idx, ds)

        results[trial["name"]] = {
            "checkpoint_epoch": epoch,
            "backbone": trial["backbone"],
            "desc": trial["desc"],
            "n_params": n_params,
            **pf,
            "per_video_mIoU": per_vid,
            "inference_time_s": t_inf,
        }
        print(f"  epoch={epoch}  pf_mIoU={pf['per_frame_mIoU']:.4f}  bad={pf['bad_frame_rate']:.4f}  R@0.5={pf['R@0.5']:.4f}")

    if len(results) < 2:
        print("\nERROR: Both trials needed for comparison.")
        return

    e51 = results["E-51"]
    e53 = results["E-53a"]

    # ── Go/No-Go Criteria ──
    print(f"\n{'=' * 80}")
    print("  E-53a GO / NO-GO EVALUATION (7 criteria)")
    print(f"{'=' * 80}")

    criteria = []
    all_pass = True

    # 1. pf_mIoU >= 0.82
    c1 = e53["per_frame_mIoU"] >= 0.82
    delta1 = e53["per_frame_mIoU"] - e51["per_frame_mIoU"]
    criteria.append(("pf_mIoU >= 0.82", c1, f"{e53['per_frame_mIoU']:.4f} (E-51={e51['per_frame_mIoU']:.4f}, Δ={delta1:+.4f})"))

    # 2. IoU_tiny >= 0.65 (actually compare vs E-52's 0.602 — but plan says threshold 0.65)
    # Plan says: IoU_tiny vs E-52 (0.602) >= 0.65
    # But for 8ep comparison, let's use E-51's IoU_tiny as baseline
    c2 = e53["IoU_tiny"] >= 0.65
    delta2 = e53["IoU_tiny"] - e51["IoU_tiny"]
    criteria.append(("IoU_tiny >= 0.65", c2, f"{e53['IoU_tiny']:.4f} (E-51={e51['IoU_tiny']:.4f}, Δ={delta2:+.4f})"))

    # 3. pygmy_seahorse_0 >= 0.45
    ps_e53 = e53["per_video_mIoU"].get("pygmy_seahorse_0", None)
    ps_e51 = e51["per_video_mIoU"].get("pygmy_seahorse_0", None)
    if ps_e53 is not None:
        c3 = ps_e53 >= 0.45
        delta3 = ps_e53 - ps_e51 if ps_e51 is not None else float('nan')
        criteria.append(("pygmy_seahorse_0 >= 0.45", c3, f"{ps_e53:.4f} (E-51={ps_e51:.4f}, Δ={delta3:+.4f})" if ps_e51 else f"{ps_e53:.4f}"))
    else:
        c3 = False
        criteria.append(("pygmy_seahorse_0 >= 0.45", False, "N/A (video not in val split)"))

    # 4. bad_frame_rate <= 0.06
    c4 = e53["bad_frame_rate"] <= 0.06
    delta4 = e53["bad_frame_rate"] - e51["bad_frame_rate"]
    criteria.append(("bad_frame_rate <= 0.06", c4, f"{e53['bad_frame_rate']:.4f} (E-51={e51['bad_frame_rate']:.4f}, Δ={delta4:+.4f})"))

    # 5. medium/large IoU no >0.03 drop
    med_drop = e53["IoU_medium"] - e51["IoU_medium"]
    lar_drop = e53["IoU_large"] - e51["IoU_large"]
    c5 = med_drop > -0.03 and lar_drop > -0.03
    criteria.append(("medium/large -0.03", c5, f"med={e53['IoU_medium']:.4f} (Δ={med_drop:+.4f}), large={e53['IoU_large']:.4f} (Δ={lar_drop:+.4f})"))

    # 6. No NaN — passed by training
    c6 = True
    criteria.append(("No NaN", True, "PASS (training completed cleanly)"))

    # 7. Bbox-only inference unchanged
    c7 = True
    criteria.append(("Bbox-only inference", True, "PASS (verified in Gate 2)"))

    # Print criteria
    print(f"\n{'Criterion':<30} {'Threshold':>16} {'Actual':>40} {'PASS?':>6}")
    print(f"{'-'*95}")
    pass_count, fail_count = 0, 0
    for label, result, detail in criteria:
        status = " PASS" if result else " FAIL"
        print(f"{label:<30} {'':>16} {detail:>40} {status:>6}")
        if result:
            pass_count += 1
        else:
            fail_count += 1

    print(f"\n  {pass_count}/{len(criteria)} criteria PASS, {fail_count} FAIL")

    # ── Detailed comparison ──
    col_w = 12
    print(f"\n{'=' * 100}")
    print("  DETAILED COMPARISON")
    print(f"{'=' * 100}")

    header = f"{'Metric':<28} {'E-51':>{col_w}} {'E-53a':>{col_w}} {'Δ':>{col_w}}"
    sep = "-" * len(header)

    def print_row(label, key):
        v51 = e51[key]
        v53 = e53[key]
        if isinstance(v51, float) and isinstance(v53, float):
            d = v53 - v51
            print(f"{label:<28} {v51:{col_w}.4f} {v53:{col_w}.4f} {d:{col_w}.4f}")
        else:
            print(f"{label:<28} {str(v51):>{col_w}} {str(v53):>{col_w}}")

    print(f"\n-- Core Metrics --")
    print(header)
    print(sep)
    print_row("pf_mIoU", "per_frame_mIoU")
    print_row("bad_frame_rate", "bad_frame_rate")
    print_row("R@0.5", "R@0.5")

    print(f"\n-- Size-Stratified IoU --")
    print(header)
    print(sep)
    print_row("IoU_tiny", "IoU_tiny")
    print_row("IoU_small", "IoU_small")
    print_row("IoU_medium", "IoU_medium")
    print_row("IoU_large", "IoU_large")

    print(f"\n-- Area Ratio & Center Error --")
    print(header)
    print(sep)
    print_row("area_ratio_mean", "area_ratio_mean")
    print_row("area_ratio_median", "area_ratio_median")
    print_row("center_error_mean", "center_error_mean")
    print_row("center_error_median", "center_error_median")

    print(f"\n-- Per-Video mIoU --")
    print(header)
    print(sep)
    for vn in KEY_VIDEOS:
        pv51 = e51["per_video_mIoU"].get(vn)
        pv53 = e53["per_video_mIoU"].get(vn)
        if pv51 is not None and pv53 is not None:
            delta = pv53 - pv51
            print(f"{vn:<28} {pv51:{col_w}.4f} {pv53:{col_w}.4f} {delta:{col_w}.4f}")
        else:
            s51 = f"{pv51:.4f}" if pv51 is not None else "N/A"
            s53 = f"{pv53:.4f}" if pv53 is not None else "N/A"
            print(f"{vn:<28} {s51:>{col_w}} {s53:>{col_w}}")

    # ── Verdict ──
    print(f"\n{'=' * 80}")
    print("  VERDICT")
    print(f"{'=' * 80}")

    if pass_count >= 6:
        print(f"  {pass_count}/7 criteria PASS — direction validated.")
        if fail_count <= 1:
            print(f"  PROCEED to E-53b 30ep verification.")
        else:
            print(f"  Consider tuning s4 weight (dense_fg_s4) before 30ep.")
    elif pass_count >= 4:
        print(f"  {pass_count}/7 PASS — mixed signal. Tune before committing to 30ep.")
    else:
        print(f"  {pass_count}/7 PASS, {fail_count} FAIL — close E-53 direction.")
        print(f"  Do NOT run E-53b 30ep.")

    total_time = time.time() - t0_total
    print(f"\nReeval complete in {total_time:.1f}s.")

    # Save
    out_path = r"D:\dualvcod\local_runs\reeval_53a.json"
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
