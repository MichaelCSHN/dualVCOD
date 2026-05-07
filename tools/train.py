"""MicroVCOD trainer — full Train/Val loop with strict data isolation.

The ConcatDataset build order is critical: split_by_video() MUST be called
BEFORE the train_sets loop so that MoCA's val videos are excluded via Subset.

Usage:
    python tools/train.py                          # full training
    python tools/train.py --overfit                # 1-batch overfit (M3)
    python tools/train.py --epochs 30 --batch_size 16
"""

import sys
import os
import time
import random
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset

from src.dataset_real import RealVideoBBoxDataset, collate_video_clips
from src.model import MicroVCOD
from src.loss import BBoxLoss
from eval.eval_video_bbox import compute_metrics, count_parameters

# ── Device ──────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")


# ── Split helpers ────────────────────────────────────────────────────

def _video_name_from_sample(sample):
    """Extract video name from sample — uses video_dir if present, else frame_dir."""
    dir_path = sample.get("video_dir", sample["frame_dir"])
    return os.path.basename(dir_path.rstrip("/\\"))


def split_by_video(dataset, val_ratio=0.2, seed=42):
    """Split dataset indices by video so the same video never spans train/val."""
    video_to_indices = defaultdict(list)
    for i in range(len(dataset)):
        vname = _video_name_from_sample(dataset.samples[i])
        video_to_indices[vname].append(i)

    videos = sorted(video_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(videos)

    n_val = max(1, int(len(videos) * val_ratio))
    val_videos = set(videos[:n_val])
    train_videos = set(videos[n_val:])

    train_idx = [i for v in train_videos for i in video_to_indices[v]]
    val_idx = [i for v in val_videos for i in video_to_indices[v]]

    return train_idx, val_idx


# ── Logger ───────────────────────────────────────────────────────────

def log_header():
    print(f"{'Epoch':>5s} | {'Tr Loss':>8s} | {'Tr mIoU':>8s} | "
          f"{'Val mIoU':>8s} | {'Val R@.5':>8s} | {'LR':>9s} | {'Time':>6s}")
    print("-" * 75)


def log_step(epoch, tr_loss, tr_miou, val_miou, val_recall, lr, e_time):
    print(f"{epoch:5d} | {tr_loss:8.5f} | {tr_miou:7.4f}  | "
          f"{val_miou:7.4f}  | {val_recall:7.4f}  | {lr:8.6f} | {e_time:5.1f}s")


# ── Training loop ────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0.0
    total_miou = 0.0
    n_batches = 0

    for frames, gt_bboxes in loader:
        frames = frames.to(DEVICE)
        gt_bboxes = gt_bboxes.to(DEVICE)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            pred = model(frames)
            losses = criterion(pred, gt_bboxes)

        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += losses["loss"].item()
        total_miou += losses["mean_iou"].item()
        n_batches += 1

    return total_loss / n_batches, total_miou / n_batches


@torch.no_grad()
def validate(model, loader):
    model.eval()
    all_preds = []
    all_gts = []

    for frames, gt_bboxes in loader:
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            pred = model(frames.to(DEVICE))
        all_preds.append(pred.float().cpu())
        all_gts.append(gt_bboxes)

    preds = torch.cat(all_preds, dim=0)
    gts = torch.cat(all_gts, dim=0)
    metrics = compute_metrics(preds, gts)
    return metrics["mean_iou"], metrics["recall@0.5"]


def save_checkpoint(model, optimizer, epoch, miou, recall, tag, clean=False):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if clean:
        fname = f"clean_seed42_epoch{epoch:03d}.pth"
    else:
        fname = f"clean_seed42_best_{tag}.pth"
    path = os.path.join(CHECKPOINT_DIR, fname)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "miou": miou,
            "recall": recall,
        },
        path,
    )
    print(f"  [SAVED] {os.path.basename(path)}  (mIoU={miou:.4f}, R@0.5={recall:.4f})")


# ── Full training ────────────────────────────────────────────────────

def run_full_training(args):
    print("=" * 75)
    print("  Phase 2.1 Joint Training  |  MicroVCOD  |  MoCA+MoCA_Mask+CAD")
    print("=" * 75)
    print(f"  device     : {DEVICE}")
    print(f"  epochs     : {args.epochs}")
    print(f"  batch_size : {args.batch_size}")
    print(f"  num_workers: {args.num_workers}")
    print(f"  T          : {args.T}")
    print(f"  lr         : {args.lr}")
    print(f"  VRAM cap   : NONE (full power)")
    print(f"  AMP        : {DEVICE == 'cuda'}")
    print(f"  Augment    : HFlip(p=0.5) + ColorJitter(b=0.15) + StrideJitter(1-3)")
    print()

    # ── Datasets ──────────────────────────────────────────────────────
    DATASET_ROOTS = [
        r"D:\ML\COD_datasets\MoCA",
        r"D:\ML\COD_datasets\MoCA_Mask",
        r"D:\ML\COD_datasets\CamouflagedAnimalDataset",
    ]

    # STEP 0: Load MoCA WITHOUT augmentation to perform video-level split FIRST.
    #          train_idx is used to filter MoCA out of the joint train set.
    #          This MUST happen before the train_sets loop.
    moca_split_ds = RealVideoBBoxDataset([DATASET_ROOTS[0]], T=args.T, target_size=224, augment=False)
    train_idx, val_idx = split_by_video(moca_split_ds, val_ratio=0.2, seed=42)
    train_vids = len({_video_name_from_sample(moca_split_ds.samples[i]) for i in train_idx})
    val_vids = len({_video_name_from_sample(moca_split_ds.samples[i]) for i in val_idx})
    print(f"  MoCA video split  : {train_vids} train / {val_vids} val  ({len(train_idx)} / {len(val_idx)} windows)")

    # Extract MoCA val canonical_video_ids for filtering MoCA_Mask and CAD
    val_canonical_ids = set()
    for idx in val_idx:
        val_canonical_ids.add(_video_name_from_sample(moca_split_ds.samples[idx]))
    print(f"  MoCA val canonical_video_ids: {len(val_canonical_ids)}")
    print()

    # STEP 1: Build joint training set — MoCA portion is FILTERED by train_idx.
    #          MoCA_Mask and CAD are filtered by canonical_video_id against MoCA val.
    print("  Loading training datasets (augmentation ON) ...")
    train_sets = []
    all_excluded_videos = {}  # dataset_name -> set of excluded canonical_video_ids
    for root in DATASET_ROOTS:
        if not os.path.isdir(root):
            print(f"    {os.path.basename(root):12s} : [NOT FOUND — skipped]")
            continue
        ds = RealVideoBBoxDataset([root], T=args.T, target_size=224, augment=True)
        name = os.path.basename(root)
        # CRITICAL: filter MoCA to train split ONLY — prevents val leakage
        if "MoCA" in root and "MoCA_Mask" not in root:
            ds = Subset(ds, train_idx)
            print(f"    {name:12s} : {len(ds):5d} windows  [TRAIN-ONLY, filtered]")
        elif "MoCA_Mask" in root:
            # Filter out any video whose canonical_video_id matches a MoCA val video
            valid_indices = []
            excluded = set()
            for i, s in enumerate(ds.samples):
                cid = _video_name_from_sample(s)
                if cid in val_canonical_ids:
                    excluded.add(cid)
                else:
                    valid_indices.append(i)
            if excluded:
                all_excluded_videos["MoCA_Mask"] = excluded
            ds = Subset(ds, valid_indices)
            print(f"    {name:12s} : {len(ds):5d} windows  [filtered by canonical_video_id, excluded {len(excluded)} videos]")
            if excluded:
                for v in sorted(excluded):
                    print(f"           EXCLUDED: {v}")
        elif "CamouflagedAnimalDataset" in root:
            # Filter out any video whose canonical_video_id matches a MoCA val video
            valid_indices = []
            excluded = set()
            for i, s in enumerate(ds.samples):
                cid = _video_name_from_sample(s)
                if cid in val_canonical_ids:
                    excluded.add(cid)
                else:
                    valid_indices.append(i)
            if excluded:
                all_excluded_videos["CAD"] = excluded
            ds = Subset(ds, valid_indices)
            print(f"    {name:12s} : {len(ds):5d} windows  [filtered by canonical_video_id, excluded {len(excluded)} videos]")
            if excluded:
                for v in sorted(excluded):
                    print(f"           EXCLUDED: {v}")
        else:
            print(f"    {name:12s} : {len(ds):5d} windows")
        train_sets.append(ds)

    joint_train_ds = ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]
    print(f"    {'TOTAL':12s} : {len(joint_train_ds):5d} windows")
    print()

    # STEP 2: MoCA-only validation set (augmentation OFF, uses val_idx from split)
    print("  Loading MoCA validation set (augmentation OFF) ...")
    val_moca_ds = RealVideoBBoxDataset([DATASET_ROOTS[0]], T=args.T, target_size=224, augment=False)
    print(f"    MoCA val  : {len(val_idx)} windows  ({val_vids} videos)")
    print(f"    Joint train total: {len(joint_train_ds)} windows")

    # ── CANONICAL VIDEO ID OVERLAP GUARD ─────────────────────────────
    # Collect all canonical_video_ids from the joint training set
    joint_train_cids = set()
    for sub_ds in train_sets:
        if hasattr(sub_ds, 'indices') and hasattr(sub_ds, 'dataset'):
            for idx in sub_ds.indices:
                joint_train_cids.add(_video_name_from_sample(sub_ds.dataset.samples[idx]))
        else:
            for i in range(len(sub_ds)):
                joint_train_cids.add(_video_name_from_sample(sub_ds.samples[i]))

    overlap = joint_train_cids & val_canonical_ids
    print(f"  JointTrain canonical_video_ids : {len(joint_train_cids)}")
    print(f"  Val canonical_video_ids        : {len(val_canonical_ids)}")
    print(f"  Overlap                        : {len(overlap)}")

    if overlap:
        print()
        print(f"  " + "=" * 65)
        print(f"  *** TRAINING REFUSED — canonical_video_id overlap detected ***")
        print(f"  " + "=" * 65)
        print(f"  {len(overlap)} video(s) appear in BOTH JointTrain and Val:")
        for v in sorted(overlap):
            print(f"    LEAK: {v}")
        print()
        print(f"  Run: python tools/verify_leak_fix.py  for full diagnostic report.")
        print(f"  All overlap must be 0 before training is allowed.")
        sys.exit(1)

    print(f"  [OK] Zero canonical_video_id overlap — safe to proceed.")
    print()

    train_loader = DataLoader(
        joint_train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_video_clips, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(val_moca_ds, val_idx), batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_video_clips, num_workers=args.num_workers, pin_memory=True,
    )

    # ── Model ────────────────────────────────────────────────────────
    model = MicroVCOD(T=args.T, pretrained_backbone=True).to(DEVICE)
    n_params = count_parameters(model)
    print(f"  params      : {n_params:,}")
    print()

    # ── Loss, optimiser, scheduler ───────────────────────────────────
    criterion = BBoxLoss(smooth_l1_weight=1.0, giou_weight=1.0, use_diou=False)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    # ── Tracking ─────────────────────────────────────────────────────
    start_epoch = 1
    best_miou = 0.0
    best_recall = 0.0

    if args.resume:
        print(f"  Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_miou = ckpt.get("miou", 0.0)
        best_recall = ckpt.get("recall", 0.0)
        # Advance scheduler to match resumed epoch
        for _ in range(ckpt["epoch"]):
            scheduler.step()
        print(f"    Checkpoint epoch : {ckpt['epoch']}")
        print(f"    Checkpoint mIoU  : {ckpt['miou']:.4f}")
        print(f"    Checkpoint R@0.5 : {ckpt['recall']:.4f}")
        print(f"    Starting epoch   : {start_epoch}")
        print()
    t0 = time.time()

    log_header()

    for epoch in range(start_epoch, args.epochs + 1):
        e_start = time.time()

        tr_loss, tr_miou = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        val_miou, val_recall = validate(model, val_loader)
        e_time = time.time() - e_start

        log_step(epoch, tr_loss, tr_miou, val_miou, val_recall, current_lr, e_time)

        # Checkpoint best models
        is_smoke = (args.epochs == 1 and not args.resume)

        if is_smoke:
            # Smoke training: always save clean epoch checkpoint (once)
            # Only save on first epoch to avoid duplicates from best_miou/best_recall both firing
            save_checkpoint(model, optimizer, epoch, val_miou, val_recall, "smoke", clean=True)
        else:
            if val_miou > best_miou:
                best_miou = val_miou
                save_checkpoint(model, optimizer, epoch, val_miou, val_recall, "miou")

            if val_recall > best_recall:
                best_recall = val_recall
                save_checkpoint(model, optimizer, epoch, val_miou, val_recall, "recall")

    total_time = time.time() - t0

    # ── Final summary ────────────────────────────────────────────────
    print()
    print("=" * 75)
    print("  PHASE 2.1 TRAINING COMPLETE")
    print("=" * 75)
    print(f"  best val mIoU    : {best_miou:.4f}")
    print(f"  best val R@0.5   : {best_recall:.4f}")
    print(f"  total time       : {total_time:.0f}s  ({total_time/args.epochs:.0f}s/epoch)")
    print(f"  params           : {n_params:,}")
    print(f"  checkpoints      : {CHECKPOINT_DIR}")
    print(f"{'=' * 75}")
    print()
    print(f"  Next: python tools/benchmark.py")


# ── Overfit (M3 preserved) ───────────────────────────────────────────

def run_overfit(args):
    print("=" * 70)
    print("  M3 Overfit Sanity Check  |  MicroVCOD")
    print("=" * 70)
    print(f"  device    : {DEVICE}")
    print(f"  epochs    : {args.epochs}")
    print(f"  batch     : {args.batch_size} clips x T={args.T}")
    print()

    ds = RealVideoBBoxDataset([r"D:\ML\COD_datasets\MoCA"], T=args.T, target_size=224)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_video_clips)
    frames, gt_bboxes = next(iter(dl))
    B, T = frames.shape[0], frames.shape[1]
    print(f"  frozen batch: ({B}, {T}, 3, 224, 224)")
    print(f"  GT range   : [{gt_bboxes.min():.4f}, {gt_bboxes.max():.4f}]")
    print()

    model = MicroVCOD(T=T, pretrained_backbone=True).to(DEVICE)
    print(f"  params     : {count_parameters(model):,}")

    criterion = BBoxLoss(smooth_l1_weight=1.0, giou_weight=1.0, use_diou=False)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    best_iou = 0.0
    log_header()

    for epoch in range(1, args.epochs + 1):
        e_start = time.time()
        frame_batch = frames
        gt_batch = gt_bboxes
        model.train()
        frame_batch = frame_batch.to(DEVICE)
        gt_batch = gt_batch.to(DEVICE)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            pred = model(frame_batch)
            losses = criterion(pred, gt_batch)

        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        tr_miou = losses["mean_iou"].item()
        if tr_miou > best_iou:
            best_iou = tr_miou
        e_time = time.time() - e_start

        interval = 1 if args.epochs <= 50 else (5 if args.epochs <= 200 else 10)
        if epoch % interval == 0 or epoch == 1 or epoch == args.epochs:
            print(f"{epoch:5d} | {losses['loss'].item():8.5f} | {losses['smooth_l1'].item():8.5f} | "
                  f"{losses['giou_loss'].item():8.5f} | {tr_miou:7.4f}  | {'':>8s} | {scheduler.get_last_lr()[0]:8.6f} | {e_time:5.1f}s")

    print()
    print(f"  final mIoU : {tr_miou:.4f}  |  best mIoU : {best_iou:.4f}")
    print(f"  status     : {'PASSED' if best_iou >= 0.9 else 'CHECK LOGS'}")
    print(f"{'=' * 70}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VCOD Trainer")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    args = parser.parse_args()

    if args.overfit:
        run_overfit(args)
    else:
        run_full_training(args)


if __name__ == "__main__":
    main()
