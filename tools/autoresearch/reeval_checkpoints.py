"""Unified reeval of top-K checkpoints from a trial directory.

Uses np.random.RandomState(42) split matching reeval_2x2_backbone_epochs.py protocol.
Reports whether best training-val checkpoint equals best unified-reeval checkpoint.

Usage:
  python tools/autoresearch/reeval_checkpoints.py --trial-dir local_runs/autoresearch/expl_NN_xxx [--backbone efficientnet_b0] [--head-type dense_fg_aux] [--input-size 224] [--T 5]
"""
import sys, os, time, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.model import MicroVCOD
from src.dataset_real import RealVideoBBoxDataset, collate_video_clips
from eval.eval_video_bbox import compute_per_frame_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_VAL = r"C:\datasets\MoCA"


def _video_name_from_sample(sample):
    d = sample.get("frame_dir", "")
    parts = d.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p in ("JPEGImages", "Imgs", "frames", "TrainDataset_per_sq"):
            return parts[i + 1] if i + 1 < len(parts) else ""
    return ""


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
    return val_idx, val_videos


def run_reeval(trial_dir, backbone="efficientnet_b0", head_type="dense_fg_aux",
               input_size=224, T=5, resized_root=None, temporal_stride=1):
    print("=" * 70)
    print(f"Top-K Checkpoint Unified Reeval")
    print(f"Trial: {trial_dir}")
    print(f"Backbone: {backbone}  Head: {head_type}  Input: {input_size}  T: {T}")
    print("=" * 70)

    # Find checkpoint files
    ckpt_files = []
    for k in range(1, 11):  # max 10
        path = os.path.join(trial_dir, f"checkpoint_rank{k}.pth")
        if os.path.isfile(path):
            ckpt_files.append((k, path))
    if not ckpt_files:
        # Fallback: use checkpoint_best.pth
        best_path = os.path.join(trial_dir, "checkpoint_best.pth")
        if os.path.isfile(best_path):
            ckpt_files = [(1, best_path)]
            print(f"  No checkpoint_rank*.pth found. Using checkpoint_best.pth")
        else:
            print("  ERROR: No checkpoints found.")
            return None

    print(f"  Found {len(ckpt_files)} checkpoint(s)")

    # Build val dataset with np.random.RandomState split
    print("\n[1/3] Building val dataset (np.random.RandomState split) ...")
    ds = RealVideoBBoxDataset(
        [DEFAULT_VAL], T=T, target_size=input_size,
        augment=False, temporal_stride=temporal_stride,
        resized_root=resized_root, return_mask=False
    )
    val_idx, val_videos = split_by_video_np(ds, val_ratio=0.2, seed=42)
    print(f"  Videos: {len(ds.samples)} samples, {len(val_videos)} val videos, {len(val_idx)} val indices")

    val_loader = DataLoader(
        Subset(ds, val_idx), batch_size=64, shuffle=False,
        collate_fn=collate_video_clips, num_workers=2, pin_memory=True
    )

    results = []
    for rank, ckpt_path in sorted(ckpt_files):
        print(f"\n[2/3] Evaluating checkpoint rank={rank} ({os.path.basename(ckpt_path)}) ...")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        train_epoch = ckpt.get("epoch", -1)
        train_miou = ckpt.get("miou", 0.0)

        model = MicroVCOD(T=T, backbone_name=backbone, head_type=head_type)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(DEVICE)
        model.eval()

        all_preds, all_gts = [], []
        t0 = time.time()
        for frames, gt_bboxes in val_loader:
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                    pred = model(frames.to(DEVICE))
            all_preds.append(pred.float().cpu())
            all_gts.append(gt_bboxes)

        preds = torch.cat(all_preds, dim=0)
        gts = torch.cat(all_gts, dim=0)
        pf = compute_per_frame_metrics(preds, gts)
        elapsed = time.time() - t0

        entry = {
            "rank": rank,
            "ckpt_file": os.path.basename(ckpt_path),
            "train_epoch": train_epoch,
            "train_val_miou": round(train_miou, 4),
            "reeval_pf_mIoU": round(pf["per_frame_mIoU"], 4),
            "reeval_R05": round(pf["R@0.5"], 4),
            "reeval_bad_frame_rate": round(pf["bad_frame_rate"], 4),
            "reeval_IoU_tiny": round(pf["IoU_tiny"], 4),
            "reeval_IoU_small": round(pf["IoU_small"], 4),
            "reeval_IoU_medium": round(pf["IoU_medium"], 4),
            "reeval_IoU_large": round(pf["IoU_large"], 4),
            "reeval_area_ratio_mean": round(pf["area_ratio_mean"], 4),
            "reeval_center_error_mean": round(pf["center_error_mean"], 4),
            "reeval_n_pred_too_large": pf["n_pred_too_large"],
            "reeval_n_pred_too_small": pf["n_pred_too_small"],
            "reeval_n_center_shift": pf["n_center_shift"],
        }
        results.append(entry)
        print(f"  epoch={train_epoch} train_val_mIoU={train_miou:.4f} "
              f"reeval_pf_mIoU={pf['per_frame_mIoU']:.4f} "
              f"bad={pf['bad_frame_rate']:.4f} R@0.5={pf['R@0.5']:.4f} "
              f"({elapsed:.1f}s)")

    # Report
    print("\n[3/3] Top-K Comparison")
    print("-" * 70)
    print(f"{'Rank':>4} {'Epoch':>5} {'Train mIoU':>10} {'Reeval pf_mIoU':>14} {'bad_frame':>9} {'R@0.5':>6} {'IoU_tiny':>8} {'IoU_small':>9} {'IoU_large':>9}")
    print("-" * 70)
    for r in results:
        print(f"{r['rank']:>4} {r['train_epoch']:>5} {r['train_val_miou']:>10.4f} {r['reeval_pf_mIoU']:>14.4f} {r['reeval_bad_frame_rate']:>9.4f} {r['reeval_R05']:>6.4f} {r['reeval_IoU_tiny']:>8.4f} {r['reeval_IoU_small']:>9.4f} {r['reeval_IoU_large']:>9.4f}")

    if results:
        best_by_train = max(results, key=lambda r: r["train_val_miou"])
        best_by_reeval = max(results, key=lambda r: r["reeval_pf_mIoU"])
        print(f"\n  Best by training val:    Rank {best_by_train['rank']} (epoch {best_by_train['train_epoch']}, train_mIoU={best_by_train['train_val_miou']:.4f})")
        print(f"  Best by unified reeval:  Rank {best_by_reeval['rank']} (epoch {best_by_reeval['train_epoch']}, reeval_pf_mIoU={best_by_reeval['reeval_pf_mIoU']:.4f})")
        if best_by_train["rank"] == best_by_reeval["rank"]:
            print("  => AGREEMENT: training best == reeval best")
        else:
            print(f"  => DISAGREEMENT: training best (rank {best_by_train['rank']}) != reeval best (rank {best_by_reeval['rank']})")

        out_path = os.path.join(trial_dir, "topk_reeval.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved: {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Unified reeval of top-K checkpoints")
    parser.add_argument("--trial-dir", required=True, help="Path to trial output directory")
    parser.add_argument("--backbone", default="efficientnet_b0")
    parser.add_argument("--head-type", default="dense_fg_aux")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--T", type=int, default=5, dest="T_val")
    parser.add_argument("--resized-root", default=r"C:\datasets_224")
    parser.add_argument("--temporal-stride", type=int, default=1)
    args = parser.parse_args()
    run_reeval(
        trial_dir=args.trial_dir,
        backbone=args.backbone,
        head_type=args.head_type,
        input_size=args.input_size,
        T=args.T_val,
        resized_root=args.resized_root,
        temporal_stride=args.temporal_stride,
    )


if __name__ == "__main__":
    main()
