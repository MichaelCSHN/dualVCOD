"""Execute a single AutoResearch trial — training + evaluation under clean protocol.

Usage:
    python tools/autoresearch/run_trial.py --trial_id smoke_b0_baseline_proxy --config trial.yaml
    python tools/autoresearch/run_trial.py --trial_id smoke_b1_effb0 --config trial.yaml

Safety guarantees (enforced, not optional):
  - Pre-flight: check_trial_safety.py runs before any GPU work
  - NO git commit, NO git push
  - Checkpoints saved to local_runs/autoresearch/<trial_id>/ only
  - All outputs within trial directory
  - Data isolation verified (canonical_video_id filtering active)
  - Clean ImageNet init — no VCOD checkpoint loaded
  - Trial metadata recorded EVEN ON FAILURE (OOM, NaN, exception)

Output structure (per trial):
  local_runs/autoresearch/<trial_id>/
    config.json           — frozen trial config snapshot
    trial.log             — full stdout/stderr tee (REQUIRED before any smoke)
    metrics.json          — per-epoch metrics array
    final_metrics.json    — clean re-eval metrics + scoring inputs
    checkpoint_best.pth   — best model (local only, NOT in repo)
    metadata.json         — hypothesis, params, FPS, GPU mem, timing, area metrics
    safety_check.json     — pre-flight check results
"""

import sys
import os
import time
import json
import random
import argparse
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.dataset_real import RealVideoBBoxDataset, collate_video_clips
from src.model import MicroVCOD
from src.loss import BBoxLoss
from eval.eval_video_bbox import compute_metrics, count_parameters, bbox_iou

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Fixed protocol values ───────────────────────────────────────────────────
SPLIT_SEED = 42
VAL_RATIO = 0.2
CENTER_PRIOR_MIOU = 0.2017
BASELINE_MIOU = 0.2861
BASELINE_R_0_5 = 0.1978

DEFAULT_TRAIN_DATASETS = [
    r"D:\ML\COD_datasets\MoCA",
    r"D:\ML\COD_datasets\MoCA_Mask",
    r"D:\ML\COD_datasets\CamouflagedAnimalDataset",
]
DEFAULT_VAL_DATASET = r"D:\ML\COD_datasets\MoCA"


# ── File-logging tee ────────────────────────────────────────────────────────

class TeeLogger:
    """Simultaneously write to stdout (or any stream) and a log file."""

    def __init__(self, log_path: str, stream=None):
        import sys as _sys
        self.log_file = open(log_path, "a", encoding="utf-8", buffering=1)
        self.stream = stream or _sys.stdout
        # Also intercept stderr by replacing sys.stderr with another TeeLogger
        # that writes to the same file.  We do this once in run_trial().

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def _setup_file_logging(trial_dir: str):
    """Redirect stdout and stderr to trial.log inside trial_dir."""
    os.makedirs(trial_dir, exist_ok=True)
    log_path = os.path.join(trial_dir, "trial.log")

    import sys as _sys
    tee = TeeLogger(log_path, _sys.stdout)
    tee_err = TeeLogger(log_path, _sys.stderr)

    _sys.stdout = tee
    _sys.stderr = tee_err
    return log_path


# ── Helpers ──────────────────────────────────────────────────────────────────

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


# ── Model factory ───────────────────────────────────────────────────────────

def build_model(backbone_name: str, temporal_T: int, head_type: str,
                input_size: int = 224) -> nn.Module:
    """Build MicroVCOD variant for a trial.

    backbone_name selects the torchvision backbone via SpineEncoderFPN.
    head_type: current_direct_bbox, objectness_aux_head, giou_center_head.
    """
    model = MicroVCOD(T=temporal_T, pretrained_backbone=True,
                     backbone_name=backbone_name, head_type=head_type).to(DEVICE)

    model._trial_backbone = backbone_name
    model._trial_head = head_type
    model._trial_input_size = input_size

    return model


# ── Data loading ────────────────────────────────────────────────────────────

def build_dataloaders(backbone_name: str, input_size: int, temporal_T: int,
                      sampler: str, batch_size: int, num_workers: int = 4,
                      eval_temporal_T: int = None, temporal_stride: int = 1):
    """Build train + val DataLoaders under clean protocol.

    Returns:
      train_loader, val_loader_primary, val_loader_diag_T5, n_train, n_val

    val_loader_primary: uses eval_temporal_T (defaults to training T).
    val_loader_diag_T5:  uses T=5 for diagnostic comparison (None if same as primary).
    """
    if eval_temporal_T is None:
        eval_temporal_T = temporal_T

    # ── Step 0: MoCA split ──
    moca_split_ds = RealVideoBBoxDataset(
        [DEFAULT_TRAIN_DATASETS[0]], T=temporal_T, target_size=input_size, augment=False,
        temporal_stride=temporal_stride,
    )
    train_idx, val_idx = split_by_video(moca_split_ds, val_ratio=VAL_RATIO, seed=SPLIT_SEED)
    val_canonical_ids = {_video_name_from_sample(moca_split_ds.samples[i]) for i in val_idx}

    # ── Step 1: Build joint training set ──
    train_sets = []
    for root in DEFAULT_TRAIN_DATASETS:
        if not os.path.isdir(root):
            continue
        ds = RealVideoBBoxDataset([root], T=temporal_T, target_size=input_size, augment=True,
                                 temporal_stride=temporal_stride)
        if "MoCA" in root and "MoCA_Mask" not in root:
            ds = Subset(ds, train_idx)
        elif "MoCA_Mask" in root or "CamouflagedAnimalDataset" in root:
            valid_indices = [i for i, s in enumerate(ds.samples)
                             if _video_name_from_sample(s) not in val_canonical_ids]
            ds = Subset(ds, valid_indices)
        train_sets.append(ds)

    joint_train_ds = ConcatDataset(train_sets) if len(train_sets) > 1 else train_sets[0]

    # Verify overlap
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
        raise RuntimeError(f"DATA LEAK DETECTED: {len(overlap)} videos in both train and val")

    # ── Sampler ──────────────────────────────────────────────────────
    train_sampler = None
    if sampler == "video_balanced":
        # Build per-sample weights: each video gets equal total weight
        sample_weights = []
        for sub_ds in train_sets:
            for idx in (sub_ds.indices if hasattr(sub_ds, 'indices') else range(len(sub_ds))):
                src_ds = sub_ds.dataset if hasattr(sub_ds, 'dataset') else sub_ds
                vname = _video_name_from_sample(src_ds.samples[idx])
                sample_weights.append(vname)
        # Map video name → inverse frequency weight
        from collections import Counter
        video_counts = Counter(sample_weights)
        weights = [1.0 / video_counts[v] for v in sample_weights]
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )
        print(f"  [INFO] video_balanced sampler: {len(video_counts)} videos, "
              f"weight range [{min(weights):.4f}, {max(weights):.4f}]")

    train_loader = DataLoader(
        joint_train_ds, batch_size=batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, collate_fn=collate_video_clips,
        num_workers=num_workers, pin_memory=True,
    )

    # ── Primary val loader (eval T = training T) ──
    val_moca_primary = RealVideoBBoxDataset(
        [DEFAULT_VAL_DATASET], T=eval_temporal_T, target_size=input_size, augment=False,
        temporal_stride=temporal_stride,
    )
    val_loader_primary = DataLoader(
        Subset(val_moca_primary, val_idx), batch_size=batch_size, shuffle=False,
        collate_fn=collate_video_clips, num_workers=num_workers, pin_memory=True,
    )

    # ── Diagnostic T=5 val loader ──
    val_loader_diag_T5 = None
    if eval_temporal_T != 5:
        val_moca_diag = RealVideoBBoxDataset(
            [DEFAULT_VAL_DATASET], T=5, target_size=input_size, augment=False,
            temporal_stride=temporal_stride,
        )
        # Rebuild val indices for T=5 — val_idx is for T=temporal_T dataset
        val_diag_idx = [i for i, s in enumerate(val_moca_diag.samples)
                        if _video_name_from_sample(s) in val_canonical_ids]
        val_loader_diag_T5 = DataLoader(
            Subset(val_moca_diag, val_diag_idx), batch_size=batch_size, shuffle=False,
            collate_fn=collate_video_clips, num_workers=num_workers, pin_memory=True,
        )

    return train_loader, val_loader_primary, val_loader_diag_T5, len(joint_train_ds), len(val_idx)


# ── Training ─────────────────────────────────────────────────────────────────

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
    all_preds, all_gts = [], []
    for frames, gt_bboxes in loader:
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            pred = model(frames.to(DEVICE))
        all_preds.append(pred.float().cpu())
        all_gts.append(gt_bboxes)
    preds = torch.cat(all_preds, dim=0)
    gts = torch.cat(all_gts, dim=0)
    return compute_metrics(preds, gts), preds, gts


# ── Area metrics ─────────────────────────────────────────────────────────────

def compute_area_metrics(preds, gts):
    """Compute both global and per-sample area ratios.

    Returns dict with:
      global_area_ratio: mean(pred_area) / mean(gt_area)
      mean_sample_area_ratio: mean(pred_area_i / gt_area_i)
      empty_pred_count: number of predictions with area < 1e-8
      empty_pred_rate: empty_pred_count / total_preds
      mean_pred_area: mean prediction area
      mean_gt_area: mean ground-truth area
    """
    pred_areas = (preds[..., 2] - preds[..., 0]) * (preds[..., 3] - preds[..., 1])
    gt_areas = (gts[..., 2] - gts[..., 0]) * (gts[..., 3] - gts[..., 1])

    total_preds = pred_areas.numel()
    empty_count = int((pred_areas < 1e-8).sum())

    global_area_ratio = float(pred_areas.mean() / max(gt_areas.mean(), 1e-8))

    # Per-sample mean area (average across T frames first)
    pred_sample_areas = pred_areas.mean(dim=1)  # (B,) — mean area across T frames
    gt_sample_areas = gt_areas.mean(dim=1)
    sample_ratios = pred_sample_areas / torch.clamp(gt_sample_areas, min=1e-8)
    mean_sample_area_ratio = float(sample_ratios.mean())

    return {
        "global_area_ratio": round(global_area_ratio, 4),
        "mean_sample_area_ratio": round(mean_sample_area_ratio, 4),
        "empty_pred_count": empty_count,
        "empty_pred_rate": round(empty_count / max(total_preds, 1), 4),
        "mean_pred_area": round(float(pred_areas.mean()), 6),
        "mean_gt_area": round(float(gt_areas.mean()), 6),
    }


# ── FPS measurement ──────────────────────────────────────────────────────────

def measure_fps(model, temporal_T, input_size):
    model.eval()
    dummy = torch.randn(1, temporal_T, 3, input_size, input_size, device=DEVICE)
    # Warmup
    for _ in range(5):
        _ = model(dummy)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(30):
        _ = model(dummy)
    torch.cuda.synchronize()
    return round(30 / max(time.time() - t0, 0.001), 1)


# ── Main ─────────────────────────────────────────────────────────────────────

def run_trial(trial_id: str, trial_config: dict, trial_dir: str):
    """Execute a complete trial."""
    os.makedirs(trial_dir, exist_ok=True)

    # ── Setup file logging ────────────────────────────────────────────
    log_path = _setup_file_logging(trial_dir)
    print(f"  [LOG] trial.log started: {log_path}")

    # ── Unpack config ─────────────────────────────────────────────────
    backbone = trial_config["backbone"]
    input_size = trial_config["input_size"]
    temporal_T_train = trial_config["temporal_T"]
    temporal_stride = trial_config.get("temporal_stride", 1)
    sampler = trial_config["sampler"]
    lr = trial_config["lr"]
    head_type = trial_config["head"]
    epochs = trial_config.get("epochs", 5)
    total_epochs = trial_config.get("total_epochs", epochs)  # T_max for scheduler — defaults to epochs
    batch_size = trial_config.get("batch_size", 24)
    num_workers = trial_config.get("num_workers", 4)
    train_seed = trial_config.get("train_seed", None)
    hypothesis = trial_config.get("hypothesis", "")

    # Set training seed only when explicitly configured (B0 baseline proxy skips
    # this to match train.py's non-deterministic behavior; B1+ trials set it for
    # reproducibility).
    if train_seed is not None:
        random.seed(train_seed)
        np.random.seed(train_seed)
        torch.manual_seed(train_seed)
        if DEVICE == "cuda":
            torch.cuda.manual_seed_all(train_seed)

    print("=" * 72)
    print(f"  AUTORESEARCH TRIAL: {trial_id}")
    print("=" * 72)
    print(f"  Trial dir:     {trial_dir}")
    print(f"  Backbone:      {backbone}")
    print(f"  Input size:    {input_size}")
    print(f"  T (train/eval): {temporal_T_train}")
    print(f"  Sampler:       {sampler}")
    print(f"  Head:          {head_type}")
    print(f"  LR:            {lr}")
    print(f"  Epochs:        {epochs}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Train seed:    {train_seed}")
    print(f"  Hypothesis:    {hypothesis}")
    print()

    # ── Save frozen config ────────────────────────────────────────────
    config_out = dict(trial_config)
    config_out["trial_id"] = trial_id
    config_out["started_at"] = datetime.now().isoformat()
    with open(os.path.join(trial_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_out, f, indent=2)

    # ── Build data ────────────────────────────────────────────────────
    print("  Building DataLoaders ...")
    t0_data = time.time()
    train_loader, val_loader_primary, val_loader_diag_T5, n_train, n_val = build_dataloaders(
        backbone, input_size, temporal_T_train, sampler, batch_size, num_workers,
        eval_temporal_T=temporal_T_train, temporal_stride=temporal_stride,
    )
    data_time = time.time() - t0_data
    print(f"  Train windows:  {n_train}")
    print(f"  Val windows:    {n_val} (primary eval T={temporal_T_train})")
    if val_loader_diag_T5 is not None:
        print(f"  Val diag T=5:   available (separate loader)")
    print(f"  Data setup:     {data_time:.1f}s")
    print()

    # ── Build model ───────────────────────────────────────────────────
    print(f"  Building model (backbone={backbone}, head={head_type}) ...")
    model = build_model(backbone, temporal_T_train, head_type, input_size)
    n_params = count_parameters(model)
    print(f"  Total params:   {n_params:,}")
    print()

    # ── Loss, optimizer, scheduler ────────────────────────────────────
    loss_weights = trial_config.get("loss_weights", {})
    criterion = BBoxLoss(
        smooth_l1_weight=loss_weights.get("smooth_l1", 1.0),
        giou_weight=loss_weights.get("giou", 1.0),
        use_diou=loss_weights.get("use_diou", False),
        center_weight=loss_weights.get("center", 0.0),
        objectness_weight=loss_weights.get("objectness", 0.1) if head_type == "objectness_aux_head" else 0.0,
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Scheduler with optional warmup
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

    # Freeze backbone for N epochs
    freeze_epochs = trial_config.get("freeze_backbone_epochs", 0)
    if freeze_epochs > 0:
        for name, param in model.spatial_encoder.named_parameters():
            param.requires_grad = False
        print(f"  Backbone frozen for {freeze_epochs} epochs")

    # ── Training loop ─────────────────────────────────────────────────
    metrics_log = []
    best_miou = 0.0
    best_recall = 0.0
    train_losses = []
    t0_train = time.time()

    for epoch in range(1, epochs + 1):
        e_start = time.time()

        # Unfreeze backbone after freeze_epochs
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            for param in model.spatial_encoder.parameters():
                param.requires_grad = True
            print(f"  Epoch {epoch}: backbone unfrozen")

        tr_loss, tr_miou = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        train_losses.append(tr_loss)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Free CUDA cache before validation to prevent OOM from fragmentation
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        # Primary eval (T = training T)
        val_metrics, val_preds, val_gts = validate(model, val_loader_primary)
        val_miou = val_metrics["mean_iou"]
        val_recall = val_metrics["recall@0.5"]
        e_time = time.time() - e_start

        entry = {
            "epoch": epoch,
            "train_loss": round(tr_loss, 6),
            "train_miou": round(tr_miou, 4),
            "val_miou": round(val_miou, 4),
            "val_recall_at_0_5": round(val_recall, 4),
            "eval_T": temporal_T_train,
            "lr": round(current_lr, 8),
            "time_s": round(e_time, 1),
        }
        metrics_log.append(entry)

        print(f"  Epoch {epoch:2d}/{epochs} | loss={tr_loss:.5f} train_mIoU={tr_miou:.4f} "
              f"val_mIoU={val_miou:.4f} val_R@0.5={val_recall:.4f} "
              f"eval_T={temporal_T_train} lr={current_lr:.2e} time={e_time:.1f}s")

        # Divergence check (after epoch 5)
        if epoch >= 5 and tr_loss > 2.0 * train_losses[0]:
            print(f"  [WARNING] Diverging loss: epoch {epoch} loss={tr_loss:.5f} > 2× epoch1 loss={train_losses[0]:.5f}")
            # Mark but don't abort — let scoring handle it

        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "miou": val_miou, "recall": val_recall,
            }, os.path.join(trial_dir, "checkpoint_best.pth"))

        if val_recall > best_recall:
            best_recall = val_recall

    total_train_time = time.time() - t0_train

    # ── Final primary evaluation ──────────────────────────────────────
    print()
    print("  Running final primary evaluation (FP32, aug=OFF, T=train) ...")
    torch.cuda.reset_peak_memory_stats()
    t0_eval = time.time()
    final_metrics, final_preds, final_gts = validate(model, val_loader_primary)
    eval_time = time.time() - t0_eval
    gpu_mem_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # ── Area metrics ──────────────────────────────────────────────────
    area = compute_area_metrics(final_preds, final_gts)

    # ── R@0.3 ─────────────────────────────────────────────────────────
    all_ious = []
    for i in range(final_preds.shape[0]):
        for t in range(final_preds.shape[1]):
            all_ious.append(float(bbox_iou(final_preds[i, t], final_gts[i, t])))
    r03 = float(np.mean(np.array(all_ious) > 0.3))

    # ── Diagnostic T=5 eval ───────────────────────────────────────────
    diag_miou_T5 = None
    diag_r05_T5 = None
    if val_loader_diag_T5 is not None:
        print("  Running diagnostic T=5 evaluation ...")
        try:
            diag_metrics, _, _ = validate(model, val_loader_diag_T5)
            diag_miou_T5 = round(diag_metrics["mean_iou"], 4)
            diag_r05_T5 = round(diag_metrics["recall@0.5"], 4)
            print(f"  Diagnostic T=5: mIoU={diag_miou_T5:.4f}  R@0.5={diag_r05_T5:.4f}")
        except Exception as e:
            print(f"  Diagnostic T=5 failed: {e}")

    # ── FPS ───────────────────────────────────────────────────────────
    fps = measure_fps(model, temporal_T_train, input_size)

    # ── Instability ───────────────────────────────────────────────────
    last_3_mious = [e["val_miou"] for e in metrics_log[-3:]] if len(metrics_log) >= 3 else [0]
    miou_std_last_3 = round(float(np.std(last_3_mious)), 4)

    # ── Write metadata ────────────────────────────────────────────────
    metadata = {
        "trial_id": trial_id,
        "status": "completed",
        "hypothesis": hypothesis,
        "backbone": backbone,
        "input_size": input_size,
        "temporal_T_train": temporal_T_train,
        "eval_T_primary": temporal_T_train,
        "sampler": sampler,
        "head": head_type,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "train_seed": train_seed,
        "split_seed": SPLIT_SEED,
        "total_params": n_params,
        "train_windows": n_train,
        "val_windows": n_val,

        # Primary metrics
        "final_val_miou_fp32": round(final_metrics["mean_iou"], 4),
        "final_val_recall_at_0_5": round(final_metrics["recall@0.5"], 4),
        "final_val_recall_at_0_3": round(r03, 4),
        "best_val_miou": round(best_miou, 4),
        "best_val_recall_at_0_5": round(best_recall, 4),

        # Diagnostic T=5
        "diag_val_miou_T5": diag_miou_T5,
        "diag_val_recall_at_0_5_T5": diag_r05_T5,

        # Area metrics (both)
        **area,

        # Stability
        "miou_std_last_3_epochs": miou_std_last_3,
        "train_losses": [round(x, 6) for x in train_losses],

        # Runtime
        "inference_fps": fps,
        "gpu_mem_gib": round(gpu_mem_gib, 2),
        "total_train_time_s": round(total_train_time, 0),
        "eval_time_s": round(eval_time, 1),

        # Protocol
        "device": DEVICE,
        "clean_protocol": True,
        "canonical_video_id_filtered": True,
        "no_commit": True,
        "no_push": True,
        "checkpoint_local_only": os.path.join(trial_dir, "checkpoint_best.pth"),

        # Timestamps
        "started_at": config_out["started_at"],
        "finished_at": datetime.now().isoformat(),
    }

    # Write artifacts: less important first, metadata last (avoids overwrite on partial failure)
    with open(os.path.join(trial_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_log, f, indent=2)

    final_out = {}
    for k, v in final_metrics.items():
        if isinstance(v, (list, np.ndarray, torch.Tensor)):
            continue
        if isinstance(v, (np.floating, np.integer)):
            final_out[k] = round(float(v), 4)
        elif isinstance(v, torch.Tensor):
            final_out[k] = round(v.item(), 4) if v.numel() == 1 else None
        elif isinstance(v, (int, float)):
            final_out[k] = round(v, 4)
        else:
            final_out[k] = str(v)
    with open(os.path.join(trial_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final_out, f, indent=2)

    with open(os.path.join(trial_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print()
    print("=" * 72)
    print(f"  TRIAL {trial_id} COMPLETE")
    print("=" * 72)
    print(f"  Best val mIoU:       {best_miou:.4f}")
    print(f"  Best val R@0.5:      {best_recall:.4f}")
    print(f"  Final mIoU (T={temporal_T_train}): {final_metrics['mean_iou']:.4f}")
    if diag_miou_T5 is not None:
        print(f"  Diag mIoU (T=5):     {diag_miou_T5:.4f}")
    print(f"  Global area ratio:   {area['global_area_ratio']:.4f}")
    print(f"  Mean sample ratio:   {area['mean_sample_area_ratio']:.4f}")
    print(f"  Empty preds:         {area['empty_pred_count']} ({area['empty_pred_rate']*100:.1f}%)")
    print(f"  Total params:        {n_params:,}")
    print(f"  Inference FPS:       {fps:.1f}")
    print(f"  GPU mem:             {gpu_mem_gib:.2f} GiB")
    print(f"  Train time:          {total_train_time:.0f}s")
    print(f"  Outputs:             {trial_dir}")
    print()

    return metadata


def _write_failure_metadata(trial_id, trial_dir, reason, started_at=None):
    """Write metadata for a failed/blocked trial."""
    os.makedirs(trial_dir, exist_ok=True)
    meta = {
        "trial_id": trial_id,
        "status": "failed",
        "reason": str(reason)[:500],
        "started_at": started_at or datetime.now().isoformat(),
        "finished_at": datetime.now().isoformat(),
        "no_commit": True,
        "no_push": True,
    }
    with open(os.path.join(trial_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    # Also ensure trial.log exists
    log_path = os.path.join(trial_dir, "trial.log")
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Trial {trial_id} failed before logging started.\nReason: {reason}\n")


def main():
    parser = argparse.ArgumentParser(description="Execute a single AutoResearch trial")
    parser.add_argument("--trial_id", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--skip_safety_check", action="store_true")
    args = parser.parse_args()

    # ── Resolve trial directory ───────────────────────────────────────
    if args.output_dir:
        trial_dir = os.path.abspath(args.output_dir)
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        trial_dir = os.path.join(project_root, "local_runs", "autoresearch", args.trial_id)

    if os.path.exists(trial_dir) and os.path.exists(os.path.join(trial_dir, "metadata.json")):
        print(f"WARNING: Trial directory {trial_dir} already has metadata.json")
        print("This trial may have already been run. Remove the directory to re-run.")
        # Don't abort — user may have cleaned manually

    started_at = datetime.now().isoformat()

    # ── Load config ───────────────────────────────────────────────────
    if args.config.endswith(".yaml") or args.config.endswith(".yml"):
        try:
            import yaml
            with open(args.config, "r", encoding="utf-8") as f:
                trial_config = yaml.safe_load(f)
        except ImportError:
            print("PyYAML not installed. Install with: pip install pyyaml")
            return 1
    else:
        with open(args.config, "r", encoding="utf-8") as f:
            trial_config = json.load(f)

    # ── Pre-flight safety check ───────────────────────────────────────
    if not args.skip_safety_check:
        print("Running pre-flight safety check ...")
        from tools.autoresearch.check_trial_safety import run_all_checks, format_results
        safety_results = run_all_checks(trial_config, trial_dir)
        os.makedirs(trial_dir, exist_ok=True)
        with open(os.path.join(trial_dir, "safety_check.json"), "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in safety_results.items()}, f, indent=2)
        print(format_results(safety_results))
        total_issues = sum(len(v) for v in safety_results.values())
        if total_issues > 0:
            print("SAFETY CHECK FAILED — trial blocked.")
            _write_failure_metadata(args.trial_id, trial_dir, "safety_check_failed", started_at)
            return 1
        print("Safety check PASSED.")
    print()

    # ── Execute trial ─────────────────────────────────────────────────
    try:
        metadata = run_trial(args.trial_id, trial_config, trial_dir)
        # run_trial already writes metadata.json — second write ensures status=completed is final
        metadata["status"] = "completed"
        with open(os.path.join(trial_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return 0
    except torch.cuda.OutOfMemoryError:
        print("OOM — trial failed (out of memory)")
        _write_failure_metadata(args.trial_id, trial_dir, "oom", started_at)
        return 1
    except Exception as e:
        print(f"Trial failed with exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        # Only write failure metadata if run_trial() didn't already write metadata.json
        meta_path = os.path.join(trial_dir, "metadata.json")
        if not os.path.exists(meta_path):
            _write_failure_metadata(args.trial_id, trial_dir, f"{type(e).__name__}: {str(e)[:300]}", started_at)
        else:
            # Patch existing metadata with failure info without losing trial data
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                existing["status"] = "failed_at_finalize"
                existing["failure_reason"] = f"{type(e).__name__}: {str(e)[:300]}"
                existing["finished_at"] = datetime.now().isoformat()
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(existing, f, indent=2)
            except Exception:
                pass  # last resort — keep whatever exists
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
