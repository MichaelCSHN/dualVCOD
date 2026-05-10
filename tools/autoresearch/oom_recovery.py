"""Emergency checkpoint save/load for OOM recovery and trial resume.

Full state checkpoint including optimizer, scheduler, scaler, and RNG state.
Separate from checkpoint_best.pth (which is best-model-only for eval).
"""

import os
import json
import time
import random
from datetime import datetime
from typing import Optional

import numpy as np
import torch


def save_emergency_checkpoint(
    trial_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    batch_idx: int,
    metrics_log: list,
    trial_config: dict,
    retry_count: int,
    failure_phase: str,
    gpu_allocated_gib: float,
    gpu_reserved_gib: float,
):
    """Save full training state for emergency recovery. All tensors moved to CPU first."""
    torch.cuda.empty_cache()

    # Capture RNG state
    rng_state = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    # Move model to CPU for safe serialization
    model_state = {k: v.cpu() for k, v in model.state_dict().items()}

    ckpt = {
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "batch_idx": batch_idx,
        "rng_state": rng_state,
        "trial_config": trial_config,
        "train_metrics_so_far": metrics_log,
        "retry_count": retry_count,
        "failure_type": "resource_oom",
        "oom_context": {
            "phase": failure_phase,
            "epoch": epoch,
            "batch_in_epoch": batch_idx,
            "gpu_allocated_gib": round(gpu_allocated_gib, 4),
            "gpu_reserved_gib": round(gpu_reserved_gib, 4),
        },
        "saved_at": datetime.now().isoformat(),
    }

    path = os.path.join(trial_dir, "emergency_ckpt.pt")
    torch.save(ckpt, path)

    # Also save augmented config for resume
    resume_config = dict(trial_config)
    resume_config["_resumed_from_emergency"] = True
    resume_config["_retry_count"] = retry_count
    resume_config["_failed_at_epoch"] = epoch
    resume_config_path = os.path.join(trial_dir, "config_resume.json")
    with open(resume_config_path, "w", encoding="utf-8") as f:
        json.dump(resume_config, f, indent=2, ensure_ascii=False)

    return path


def has_emergency_checkpoint(trial_dir: str) -> bool:
    return os.path.isfile(os.path.join(trial_dir, "emergency_ckpt.pt"))


def load_emergency_checkpoint(trial_dir: str, device: str = "cuda"):
    """Load full training state from emergency checkpoint. Returns dict with all state."""
    path = os.path.join(trial_dir, "emergency_ckpt.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No emergency checkpoint at {path}")

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Restore RNG state before any other operations
    rng = ckpt.get("rng_state", {})
    if rng.get("random") is not None:
        random.setstate(rng["random"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"])
    if rng.get("torch_cuda") is not None and device == "cuda":
        torch.cuda.set_rng_state_all(rng["torch_cuda"])

    return ckpt


def record_oom_event(trial_dir: str, epoch: int, phase: str, error_msg: str):
    """Append OOM event to oom_events.jsonl."""
    events_path = os.path.join(trial_dir, "oom_events.jsonl")
    event = {
        "timestamp": datetime.now().isoformat(),
        "epoch": epoch,
        "phase": phase,
        "error": error_msg[:500],
    }
    with open(events_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def is_oom_error(error: Exception) -> bool:
    """Check if error is a CUDA OutOfMemoryError (safe to retry)."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    msg = str(error).lower()
    if "out of memory" in msg and ("cuda" in msg or "gpu" in msg):
        return True
    return False


def is_fatal_cuda_error(error: Exception) -> bool:
    """Check if error is a fatal CUDA error (NOT safe to retry).

    Fatal errors include:
    - Illegal memory access (CUDA error: an illegal memory access was encountered)
    - CUBLAS_STATUS_EXECUTION_FAILED
    - GPU lost / device removed
    """
    msg = str(error).lower()
    fatal_patterns = [
        "illegal memory access",
        "cublas_status_execution_failed",
        "device-side assert",
        "unknown error",
        "driver error",
        "device lost",
        "cuda error: unknown",
    ]
    for pat in fatal_patterns:
        if pat in msg:
            return True
    return False
