"""Batch-size ladder: find max safe train/eval batch before real training.

Runs a short probe with the actual model architecture to determine the
largest batch that fits in GPU memory, then applies a safety factor.

Cost: ~2-3 min. Recommended for: new backbones, T > 5, input_size > 224.
"""

import time, json, os, sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.dataset_real import RealVideoBBoxDataset, collate_video_clips
from src.model import MicroVCOD
from src.loss import BBoxLoss


def probe_batch_size(
    trial_config: dict,
    train_dataset,
    val_dataset,
    device: str = "cuda",
    safety_factor: float = 0.75,
) -> dict:
    """Run batch-size ladder and return safe train/eval batch sizes.

    Returns:
        {
            "max_safe_train_batch": int,
            "max_safe_eval_batch": int,
            "recommended_train_batch": int,
            "recommended_eval_batch": int,
            "ladder_results": [...],
            "ladder_time_s": float,
        }
    """
    t0 = time.time()

    backbone = trial_config["backbone"]
    input_size = trial_config["input_size"]
    T = trial_config["temporal_T"]
    lr = trial_config.get("lr", 0.001)
    head_type = trial_config.get("head", "current_direct_bbox")
    loss_weights = trial_config.get("loss_weights", {})

    train_candidates = [16, 24, 32, 48, 64]
    eval_candidates = [32, 48, 64, 96, 128]

    results = []

    # Build model once
    model = MicroVCOD(T=T, pretrained_backbone=True,
                      backbone_name=backbone, head_type=head_type).to(device)
    criterion = BBoxLoss(
        smooth_l1_weight=loss_weights.get("smooth_l1", 1.0),
        giou_weight=loss_weights.get("giou", 1.0),
        use_diou=loss_weights.get("use_diou", False),
        use_ciou=loss_weights.get("use_ciou", False),
        center_weight=loss_weights.get("center", 0.0),
        log_wh_weight=loss_weights.get("log_wh", 0.0),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # Warm-up allocation
    dummy = torch.randn(2, T, 3, input_size, input_size, device=device)
    with torch.amp.autocast("cuda", enabled=(device == "cuda")):
        _ = model(dummy)
    torch.cuda.empty_cache()

    # --- Train batch probe ---
    max_train_batch = 16
    for bs in train_candidates:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            loader = DataLoader(
                Subset(train_dataset, range(min(200, len(train_dataset)))),
                batch_size=bs, shuffle=True, collate_fn=collate_video_clips,
                num_workers=2, pin_memory=True,
            )
            model.train()
            for step, (frames, gt_bboxes) in enumerate(loader):
                if step >= 3:
                    break
                frames = frames.to(device)
                gt_bboxes = gt_bboxes.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    pred = model(frames)
                    losses = criterion(pred, gt_bboxes)
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
            results.append({"phase": "train", "batch_size": bs, "peak_mem_gib": round(peak_mem, 2), "status": "ok"})
            max_train_batch = bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results.append({"phase": "train", "batch_size": bs, "peak_mem_gib": None, "status": "oom"})
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                results.append({"phase": "train", "batch_size": bs, "peak_mem_gib": None, "status": "oom"})
                break
            raise

    # --- Eval batch probe ---
    max_eval_batch = 32
    for bs in eval_candidates:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            loader = DataLoader(
                Subset(val_dataset, range(min(200, len(val_dataset)))),
                batch_size=bs, shuffle=False, collate_fn=collate_video_clips,
                num_workers=2, pin_memory=True,
            )
            model.eval()
            with torch.no_grad():
                for step, (frames, gt_bboxes) in enumerate(loader):
                    if step >= 5:
                        break
                    with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                        _ = model(frames.to(device))
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
            results.append({"phase": "eval", "batch_size": bs, "peak_mem_gib": round(peak_mem, 2), "status": "ok"})
            max_eval_batch = bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results.append({"phase": "eval", "batch_size": bs, "peak_mem_gib": None, "status": "oom"})
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                results.append({"phase": "eval", "batch_size": bs, "peak_mem_gib": None, "status": "oom"})
                break
            raise

    torch.cuda.empty_cache()

    recommended_train = max(int(max_train_batch * safety_factor), 1)
    recommended_eval = max(int(max_eval_batch * safety_factor), 1)

    return {
        "max_safe_train_batch": max_train_batch,
        "max_safe_eval_batch": max_eval_batch,
        "recommended_train_batch": recommended_train,
        "recommended_eval_batch": recommended_eval,
        "ladder_results": results,
        "ladder_time_s": round(time.time() - t0, 1),
    }


def run_batch_ladder(trial_config: dict, output_dir: str, device: str = "cuda"):
    """Full batch-ladder workflow: probe → report → update config."""
    T = trial_config["temporal_T"]
    input_size = trial_config["input_size"]

    # Minimal dataset for probing
    moca_root = r"D:\ML\COD_datasets\MoCA"
    train_ds = RealVideoBBoxDataset([moca_root], T=T, target_size=input_size,
                                    augment=True, temporal_stride=trial_config.get("temporal_stride", 1))
    val_ds = RealVideoBBoxDataset([moca_root], T=T, target_size=input_size,
                                  augment=False, temporal_stride=trial_config.get("temporal_stride", 1))

    result = probe_batch_size(
        trial_config=trial_config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        device=device,
        safety_factor=trial_config.get("batch_safety_factor", 0.75),
    )

    path = os.path.join(output_dir, "batch_ladder.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result
