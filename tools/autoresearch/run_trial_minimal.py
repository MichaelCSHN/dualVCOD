"""Minimal trial runner — no TeeLogger, direct stdout for debugging.

Anti-hang mechanisms:
  - cv2.setNumThreads(0) to prevent OpenCV thread interference
  - DataLoader timeout=120s + persistent_workers=False + prefetch_factor=2
  - Batch heartbeat every 10 batches
  - Per-epoch batch timing stats
  - Image path sanity scan before training
"""
import sys, os, time, json, random, argparse
from collections import defaultdict
from datetime import datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset

# Prevent OpenCV multi-threading from interfering with DataLoader workers
cv2.setNumThreads(0)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.dataset_real import RealVideoBBoxDataset, collate_video_clips
from src.model import MicroVCOD
from src.loss import BBoxLoss
from eval.eval_video_bbox import compute_metrics, count_parameters, bbox_iou
from tools.autoresearch.oom_recovery import (
    save_emergency_checkpoint, load_emergency_checkpoint,
    has_emergency_checkpoint, record_oom_event,
    is_oom_error, is_fatal_cuda_error,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPLIT_SEED = 42
VAL_RATIO = 0.2

DEFAULT_TRAIN_DATASETS = [
    r"D:\ML\COD_datasets\MoCA",
    r"D:\ML\COD_datasets\MoCA_Mask",
    r"D:\ML\COD_datasets\CamouflagedAnimalDataset",
]
DEFAULT_VAL_DATASET = r"D:\ML\COD_datasets\MoCA"


def _video_name_from_sample(sample):
    dir_path = sample.get("video_dir", sample["frame_dir"])
    return os.path.basename(dir_path.rstrip("/\\"))


def split_by_video(dataset, val_ratio=0.2, seed=42):
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


def image_sanity_scan(dataset, max_samples=200, label=""):
    """Quick scan of image paths — check existence and readability."""
    n = len(dataset)
    rng = random.Random(42)
    indices = rng.sample(range(n), min(max_samples, n))
    failed = []
    t0 = time.time()
    for idx in indices:
        sample = dataset.samples[idx] if hasattr(dataset, 'samples') else dataset[idx]
        frame_dir = sample.get("frame_dir", sample.get("video_dir", ""))
        frame_files = sample.get("frame_files", [])
        if not frame_dir or not frame_files:
            failed.append(f"idx={idx}: missing frame_dir or frame_files")
            continue
        # Check first frame
        fp = os.path.join(frame_dir, frame_files[0])
        if not os.path.exists(fp):
            failed.append(f"missing: {fp}")
            continue
        img = cv2.imread(fp)
        if img is None or img.size == 0:
            failed.append(f"unreadable: {fp}")
    elapsed = time.time() - t0
    if failed:
        for f in failed[:20]:
            log(f"  SANITY FAIL [{label}]: {f}")
        log(f"  SANITY [{label}]: {len(failed)}/{len(indices)} failed ({elapsed:.1f}s)")
    else:
        log(f"  SANITY [{label}]: {len(indices)}/{len(indices)} OK ({elapsed:.1f}s)")
    return len(failed) == 0


_LOG_FILE = None

def log(msg):
    print(msg, flush=True)
    if _LOG_FILE is not None:
        _LOG_FILE.write(msg + "\n")
        _LOG_FILE.flush()


def main():
    global _LOG_FILE
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_id", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume from emergency checkpoint if present")
    args = parser.parse_args()

    if args.output_dir:
        trial_dir = os.path.abspath(args.output_dir)
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        trial_dir = os.path.join(project_root, "local_runs", "autoresearch", args.trial_id)

    os.makedirs(trial_dir, exist_ok=True)

    # Open log file
    log_path = os.path.join(trial_dir, "trial.log")
    _LOG_FILE = open(log_path, "w", encoding="utf-8", buffering=1)

    with open(args.config, "r", encoding="utf-8") as f:
        trial_config = json.load(f)

    backbone = trial_config["backbone"]
    input_size = trial_config["input_size"]
    temporal_T_train = trial_config["temporal_T"]
    temporal_stride = trial_config.get("temporal_stride", 1)
    sampler = trial_config["sampler"]
    lr = trial_config["lr"]
    head_type = trial_config["head"]
    epochs = trial_config.get("epochs", 5)
    total_epochs = trial_config.get("total_epochs", epochs)
    batch_size = trial_config.get("batch_size", 24)
    train_batch_size = trial_config.get("train_batch_size", batch_size)
    eval_batch_size = trial_config.get("eval_batch_size", min(batch_size * 2, 64))
    num_workers = trial_config.get("num_workers", 0)
    train_seed = trial_config.get("train_seed", None)
    hypothesis = trial_config.get("hypothesis", "")

    if train_seed is not None:
        random.seed(train_seed)
        np.random.seed(train_seed)
        torch.manual_seed(train_seed)
        if DEVICE == "cuda":
            torch.cuda.manual_seed_all(train_seed)

    log("=" * 72)
    log(f"  TRIAL: {args.trial_id}")
    log("=" * 72)
    log(f"  Backbone: {backbone}  Size: {input_size}  T: {temporal_T_train}")
    log(f"  Head: {head_type}  LR: {lr}  Epochs: {epochs}  Batch: {batch_size}")
    log(f"  num_workers: {num_workers}")

    # ── Data ──
    log("\n[1/5] Building DataLoaders ...")
    t0 = time.time()

    moca_split_ds = RealVideoBBoxDataset(
        [DEFAULT_TRAIN_DATASETS[0]], T=temporal_T_train, target_size=input_size,
        augment=False, temporal_stride=temporal_stride
    )
    log(f"  MoCA split dataset: {len(moca_split_ds)} samples")

    # --- Image path sanity scan (quick sample) ---
    log("  Running image sanity scan (200 samples)...")
    image_sanity_scan(moca_split_ds, max_samples=200, label="MoCA")

    train_idx, val_idx = split_by_video(moca_split_ds, val_ratio=VAL_RATIO, seed=SPLIT_SEED)
    val_canonical_ids = {_video_name_from_sample(moca_split_ds.samples[i]) for i in val_idx}
    log(f"  Train idx: {len(train_idx)}, Val idx: {len(val_idx)}")

    train_sets = []
    for root in DEFAULT_TRAIN_DATASETS:
        if not os.path.isdir(root):
            continue
        ds = RealVideoBBoxDataset([root], T=temporal_T_train, target_size=input_size,
                                  augment=True, temporal_stride=temporal_stride)
        name = os.path.basename(root)
        if "MoCA" in root and "MoCA_Mask" not in root:
            ds = Subset(ds, train_idx)
        elif "MoCA_Mask" in root or "CamouflagedAnimalDataset" in root:
            valid_indices = [i for i, s in enumerate(ds.samples)
                             if _video_name_from_sample(s) not in val_canonical_ids]
            ds = Subset(ds, valid_indices)
        train_sets.append(ds)
        log(f"  {name}: {len(ds)} samples")

    joint_train_ds = ConcatDataset(train_sets)
    log(f"  Joint train: {len(joint_train_ds)} samples")

    # Check overlap
    joint_train_cids = set()
    for sub_ds in train_sets:
        if hasattr(sub_ds, 'indices') and hasattr(sub_ds, 'dataset'):
            for idx in sub_ds.indices:
                joint_train_cids.add(_video_name_from_sample(sub_ds.dataset.samples[idx]))
        else:
            for i in range(len(sub_ds)):
                joint_train_cids.add(_video_name_from_sample(sub_ds.samples[i]))
    overlap = joint_train_cids & val_canonical_ids
    if overlap:
        log(f"  FATAL: {len(overlap)} videos in both train and val!")
        return 1
    log(f"  Overlap check: OK (0)")

    train_batches_est = len(joint_train_ds) // train_batch_size
    train_loader = DataLoader(
        joint_train_ds, batch_size=train_batch_size, shuffle=True,
        collate_fn=collate_video_clips, num_workers=num_workers,
        pin_memory=(num_workers > 0),
        timeout=120 if num_workers > 0 else 0,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    val_moca_primary = RealVideoBBoxDataset(
        [DEFAULT_VAL_DATASET], T=temporal_T_train, target_size=input_size,
        augment=False, temporal_stride=temporal_stride,
    )
    val_loader = DataLoader(
        Subset(val_moca_primary, val_idx), batch_size=eval_batch_size, shuffle=False,
        collate_fn=collate_video_clips, num_workers=num_workers,
        pin_memory=(num_workers > 0),
        timeout=120 if num_workers > 0 else 0,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    log(f"  Data setup: {time.time() - t0:.1f}s")

    # ── Model ──
    log("\n[2/5] Building model ...")
    t0 = time.time()
    model = MicroVCOD(T=temporal_T_train, pretrained_backbone=True,
                      backbone_name=backbone, head_type=head_type).to(DEVICE)
    n_params = count_parameters(model)
    log(f"  Model built: {n_params:,} params ({time.time() - t0:.1f}s)")

    # ── Loss, optimizer, scheduler ──
    log("\n[3/5] Setting up optimizer ...")
    loss_weights = trial_config.get("loss_weights", {})
    criterion = BBoxLoss(
        smooth_l1_weight=loss_weights.get("smooth_l1", 1.0),
        giou_weight=loss_weights.get("giou", 1.0),
        use_diou=loss_weights.get("use_diou", False),
        center_weight=loss_weights.get("center", 0.0),
        log_wh_weight=loss_weights.get("log_wh", 0.0),
        objectness_weight=loss_weights.get("objectness", 0.1) if head_type == "objectness_aux_head" else 0.0,
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    warmup_epochs = trial_config.get("warmup_epochs", 0)
    if warmup_epochs > 0 and warmup_epochs < total_epochs:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs - warmup_epochs
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    log(f"  Optimizer ready")

    # ── Resume from emergency checkpoint ──
    start_epoch = 1
    best_miou = 0.0
    best_recall = 0.0
    metrics_log = []
    oom_retry_count = 0
    resume_batch_size = None

    if args.resume and has_emergency_checkpoint(trial_dir):
        log("\n[resume] Loading emergency checkpoint ...")
        ckpt = load_emergency_checkpoint(trial_dir, DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1  # resume from next epoch
        metrics_log = ckpt.get("train_metrics_so_far", [])
        oom_retry_count = ckpt.get("retry_count", 0)
        if metrics_log:
            best_miou = max(e["val_miou"] for e in metrics_log)
            best_recall = max(e["val_recall_at_0_5"] for e in metrics_log)
        if "batch_size" in ckpt.get("trial_config", {}):
            resume_batch_size = ckpt["trial_config"].get("train_batch_size")
        log(f"  Resuming from epoch {start_epoch}/{epochs}")
        log(f"  Restored best: mIoU={best_miou:.4f}  R@0.5={best_recall:.4f}")
        log(f"  Prior OOM retries: {oom_retry_count}")
        if resume_batch_size and resume_batch_size < train_batch_size:
            log(f"  Batch reduced: {train_batch_size} → {resume_batch_size} (post-OOM)")
            train_batch_size = resume_batch_size
    elif args.resume:
        log("\n[resume] No emergency checkpoint found — starting fresh")

    # ── Training ──
    log(f"\n[4/5] Training {epochs} epochs (heartbeat every 10 batches, timeout=120s)...")
    log(f"  train_batch={train_batch_size}  eval_batch={eval_batch_size}")
    log(f"{'Epoch':>5s} | {'Tr Loss':>8s} | {'Tr mIoU':>8s} | {'Val mIoU':>8s} | {'Val R@.5':>8s} | {'LR':>9s} | {'Time':>6s} | b_mean b_max")
    log("-" * 95)

    train_losses = []
    t0_train = time.time()
    train_batch_size_original = train_batch_size

    for epoch in range(start_epoch, epochs + 1):
        e_start = time.time()

        # Train
        model.train()
        total_loss = 0.0
        total_miou = 0.0
        n_batches = 0
        batch_times = []
        last_heartbeat = time.time()
        try:
            for frames, gt_bboxes in train_loader:
                batch_start = time.time()
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
                batch_elapsed = time.time() - batch_start
                batch_times.append(batch_elapsed)

                # Heartbeat every 10 batches
                if n_batches % 10 == 0:
                    heartbeat_elapsed = time.time() - last_heartbeat
                    last_heartbeat = time.time()
                    log(f"  [e{epoch}] batch {n_batches}/{train_batches_est}+ | "
                        f"last_10={heartbeat_elapsed:.1f}s | "
                        f"loss={losses['loss'].item():.4f} miou={losses['mean_iou'].item():.4f}")

                # Periodic cache clearing to prevent fragmentation OOM
                if n_batches % 100 == 0 and DEVICE == "cuda":
                    torch.cuda.empty_cache()

        except Exception as e:
            if is_oom_error(e):
                log(f"\n  [OOM] CUDA OutOfMemory at epoch {epoch} batch {n_batches}: {str(e)[:200]}")
                torch.cuda.empty_cache()
                gpu_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3) if DEVICE == "cuda" else 0.0
                gpu_resv = torch.cuda.max_memory_reserved() / (1024 ** 3) if DEVICE == "cuda" else 0.0
                save_emergency_checkpoint(
                    trial_dir=trial_dir, model=model, optimizer=optimizer,
                    scheduler=scheduler, scaler=scaler, epoch=epoch - 1,
                    batch_idx=n_batches, metrics_log=metrics_log,
                    trial_config=trial_config, retry_count=oom_retry_count,
                    failure_phase="train",
                    gpu_allocated_gib=gpu_alloc, gpu_reserved_gib=gpu_resv,
                )
                record_oom_event(trial_dir, epoch, "train", str(e))
                if oom_retry_count < 2:
                    log(f"  [OOM] Emergency checkpoint saved. Restart with --resume to continue with reduced batch.")
                else:
                    log(f"  [OOM] Max retries ({oom_retry_count}) reached. Marking resource_failed.")
                    metadata = {
                        "trial_id": args.trial_id, "status": "resource_failed",
                        "hypothesis": hypothesis, "backbone": backbone,
                        "input_size": input_size, "temporal_T_train": temporal_T_train,
                        "head": head_type, "lr": lr, "epochs": epochs,
                        "batch_size": batch_size, "train_seed": train_seed,
                        "total_params": n_params,
                        "best_val_miou": round(best_miou, 4),
                        "best_val_recall_at_0_5": round(best_recall, 4),
                        "oom_retry_count": oom_retry_count,
                        "finished_at": datetime.now().isoformat(),
                    }
                    with open(os.path.join(trial_dir, "metadata.json"), "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2)
                    # Remove emergency checkpoint since we're giving up
                    ckpt_path = os.path.join(trial_dir, "emergency_ckpt.pt")
                    if os.path.isfile(ckpt_path):
                        os.remove(ckpt_path)
                if _LOG_FILE is not None:
                    _LOG_FILE.close()
                return 0
            elif is_fatal_cuda_error(e):
                log(f"\n  [FATAL] Non-recoverable CUDA error at epoch {epoch} batch {n_batches}: {str(e)[:300]}")
                log(f"  [FATAL] This is NOT an OOM — likely illegal memory access or device crash.")
                log(f"  [FATAL] WILL NOT RETRY. Check for NaN poisoning, out-of-bounds indexing, or driver issues.")
                metadata = {
                    "trial_id": args.trial_id, "status": "hardware_error",
                    "hypothesis": hypothesis, "backbone": backbone,
                    "input_size": input_size, "temporal_T_train": temporal_T_train,
                    "head": head_type, "lr": lr, "epochs": epochs,
                    "batch_size": batch_size, "train_seed": train_seed,
                    "total_params": n_params,
                    "fatal_error": str(e)[:500],
                    "finished_at": datetime.now().isoformat(),
                }
                with open(os.path.join(trial_dir, "metadata.json"), "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)
                if _LOG_FILE is not None:
                    _LOG_FILE.close()
                return 0
            else:
                raise

        tr_loss = total_loss / n_batches
        tr_miou = total_miou / n_batches
        train_losses.append(tr_loss)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Clear CUDA cache between train and val
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # Validate
        model.eval()
        all_preds, all_gts = [], []
        for frames, gt_bboxes in val_loader:
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                    pred = model(frames.to(DEVICE))
            all_preds.append(pred.float().cpu())
            all_gts.append(gt_bboxes)
        preds = torch.cat(all_preds, dim=0)
        gts = torch.cat(all_gts, dim=0)
        val_metrics = compute_metrics(preds, gts)
        val_miou = val_metrics["mean_iou"]
        val_recall = val_metrics["recall@0.5"]
        e_time = time.time() - e_start

        mean_batch_time = np.mean(batch_times) if batch_times else 0.0
        max_batch_time = np.max(batch_times) if batch_times else 0.0
        log(f"{epoch:5d} | {tr_loss:8.5f} | {tr_miou:7.4f}  | {val_miou:7.4f}  | {val_recall:7.4f}  | {current_lr:8.6f} | {e_time:5.1f}s | b_mean={mean_batch_time:.2f}s b_max={max_batch_time:.2f}s")

        entry = {
            "epoch": epoch,
            "train_loss": round(tr_loss, 6),
            "train_miou": round(tr_miou, 4),
            "val_miou": round(val_miou, 4),
            "val_recall_at_0_5": round(val_recall, 4),
            "lr": round(current_lr, 8),
            "time_s": round(e_time, 1),
            "mean_batch_s": round(mean_batch_time, 3),
            "max_batch_s": round(max_batch_time, 3),
            "n_batches": n_batches,
        }
        metrics_log.append(entry)

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                "epoch": epoch,
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "miou": val_miou,
                "recall": val_recall,
            }, os.path.join(trial_dir, "checkpoint_best.pth"))

        if val_recall > best_recall:
            best_recall = val_recall

    total_train_time = time.time() - t0_train

    # Save per-epoch metrics
    with open(os.path.join(trial_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_log, f, indent=2)

    # ── Final eval ──
    log(f"\n[5/5] Final evaluation ...")
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    all_preds, all_gts = [], []
    for frames, gt_bboxes in val_loader:
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                pred = model(frames.to(DEVICE))
        all_preds.append(pred.float().cpu())
        all_gts.append(gt_bboxes)
    final_preds = torch.cat(all_preds, dim=0)
    final_gts = torch.cat(all_gts, dim=0)
    final_metrics = compute_metrics(final_preds, final_gts)

    # R@0.3
    all_ious = []
    for i in range(final_preds.shape[0]):
        for t in range(final_preds.shape[1]):
            all_ious.append(float(bbox_iou(final_preds[i, t], final_gts[i, t])))
    r03 = float(np.mean(np.array(all_ious) > 0.3))

    # Area metrics
    pred_areas = (final_preds[..., 2] - final_preds[..., 0]) * (final_preds[..., 3] - final_preds[..., 1])
    gt_areas = (final_gts[..., 2] - final_gts[..., 0]) * (final_gts[..., 3] - final_gts[..., 1])
    global_area_ratio = float(pred_areas.mean() / max(gt_areas.mean(), 1e-8))
    empty_rate = float((pred_areas < 1e-8).sum()) / max(pred_areas.numel(), 1)
    pred_sample_areas = pred_areas.mean(dim=1)
    gt_sample_areas = gt_areas.mean(dim=1)
    sample_ratios = pred_sample_areas / torch.clamp(gt_sample_areas, min=1e-8)
    mean_sample_area_ratio = float(sample_ratios.mean())

    # FPS
    dummy = torch.randn(1, temporal_T_train, 3, input_size, input_size, device=DEVICE)
    for _ in range(5):
        _ = model(dummy)
    torch.cuda.synchronize()
    t_fps = time.time()
    for _ in range(30):
        _ = model(dummy)
    torch.cuda.synchronize()
    fps = round(30 / max(time.time() - t_fps, 0.001), 1)

    gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # Stability
    last_3_mious = [e["val_miou"] for e in metrics_log[-3:]] if len(metrics_log) >= 3 else [0.0]
    miou_std_last_3 = round(float(np.std(last_3_mious)), 4)

    # ── Save metadata ──
    metadata = {
        "trial_id": args.trial_id,
        "status": "completed",
        "hypothesis": hypothesis,
        "backbone": backbone,
        "input_size": input_size,
        "temporal_T_train": temporal_T_train,
        "head": head_type,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_batch_size": train_batch_size_original,
        "eval_batch_size": eval_batch_size,
        "train_seed": train_seed,
        "total_params": n_params,
        "best_val_miou": round(best_miou, 4),
        "best_val_recall_at_0_5": round(best_recall, 4),
        "final_val_miou_fp32": round(final_metrics["mean_iou"], 4),
        "final_val_recall_at_0_5": round(final_metrics["recall@0.5"], 4),
        "final_val_recall_at_0_3": round(r03, 4),
        "global_area_ratio": round(global_area_ratio, 4),
        "mean_sample_area_ratio": round(mean_sample_area_ratio, 4),
        "empty_pred_rate": round(empty_rate, 4),
        "miou_std_last_3_epochs": miou_std_last_3,
        "gpu_mem_gib": round(gpu_mem, 2),
        "inference_fps": fps,
        "total_train_time_s": round(total_train_time, 0),
        "num_workers": num_workers,
        "no_commit": True,
        "no_push": True,
        "finished_at": datetime.now().isoformat(),
    }
    with open(os.path.join(trial_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    log(f"\n{'=' * 72}")
    log(f"  TRIAL COMPLETE")
    log(f"  Best mIoU: {best_miou:.4f}  Best R@0.5: {best_recall:.4f}")
    log(f"  Final mIoU: {final_metrics['mean_iou']:.4f}  R@0.3: {r03:.4f}")
    log(f"  Area ratio: {global_area_ratio:.4f}  Mean sample ratio: {mean_sample_area_ratio:.4f}  Empty: {empty_rate:.4f}")
    log(f"  FPS: {fps:.1f}  GPU: {gpu_mem:.2f} GiB  Train time: {total_train_time:.0f}s")
    log(f"{'=' * 72}")

    if _LOG_FILE is not None:
        _LOG_FILE.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
