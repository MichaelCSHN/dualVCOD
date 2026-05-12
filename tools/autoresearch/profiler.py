"""Per-epoch timing and memory profiler with near-zero overhead.

Writes per-epoch profiles to trial_dir/profiles.jsonl.
Overhead: ~2 time.perf_counter() calls per batch (negligible).
"""

import time, json, os
from dataclasses import dataclass, field
from typing import Optional
import torch

try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False


def _get_gpu_utilization() -> float:
    """Query GPU utilization via pynvml. Returns -1 if unavailable."""
    if not _NVML_AVAILABLE:
        return -1.0
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        pynvml.nvmlShutdown()
        return float(util.gpu)
    except Exception:
        return -1.0


@dataclass
class EpochProfile:
    epoch: int
    data_time_s: float = 0.0
    h2d_time_s: float = 0.0
    forward_time_s: float = 0.0
    loss_time_s: float = 0.0
    backward_time_s: float = 0.0
    optimizer_time_s: float = 0.0
    validation_time_s: float = 0.0
    total_time_s: float = 0.0
    gpu_allocated_gib: float = 0.0
    gpu_reserved_gib: float = 0.0
    gpu_util_pct: float = 0.0
    cpu_ram_gib: float = 0.0
    batch_size: int = 0
    eval_batch_size: int = 0
    num_workers: int = 0
    n_batches: int = 0
    bottleneck: str = "unknown"

    @property
    def compute_time_s(self) -> float:
        """Sum of forward + loss + backward + optimizer (GPU compute)."""
        return self.forward_time_s + self.loss_time_s + self.backward_time_s + self.optimizer_time_s

    @property
    def train_step_time_s(self) -> float:
        """Total per-batch train step (data + h2d + compute)."""
        return self.data_time_s + self.h2d_time_s + self.compute_time_s

    def to_dict(self):
        return {
            "epoch": self.epoch,
            "data_time_s": round(self.data_time_s, 2),
            "h2d_time_s": round(self.h2d_time_s, 2),
            "forward_time_s": round(self.forward_time_s, 2),
            "loss_time_s": round(self.loss_time_s, 2),
            "backward_time_s": round(self.backward_time_s, 2),
            "optimizer_time_s": round(self.optimizer_time_s, 2),
            "validation_time_s": round(self.validation_time_s, 2),
            "total_time_s": round(self.total_time_s, 1),
            "gpu_allocated_gib": round(self.gpu_allocated_gib, 3),
            "gpu_reserved_gib": round(self.gpu_reserved_gib, 3),
            "gpu_util_pct": round(self.gpu_util_pct, 1),
            "cpu_ram_gib": round(self.cpu_ram_gib, 2),
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "num_workers": self.num_workers,
            "n_batches": self.n_batches,
            "bottleneck": self.bottleneck,
        }


class EpochProfiler:
    """Per-epoch timing tracker with fine-grained phase breakdown.

    Usage:
        p = EpochProfiler(epoch, batch_size, eval_batch_size, num_workers, trial_dir)
        for batch in loader:
            p.tick_dataloader()       # batch fetched from loader
            # frames.to(device), gt.to(device)
            p.tick_h2d()              # data on GPU
            # pred = model(frames)
            p.tick_forward()          # forward pass done
            # losses = criterion(pred, gt)
            p.tick_loss()             # loss computed
            # backward + unscale + clip
            p.tick_backward()         # backward done
            # optimizer.step() + scaler.update()
            p.tick_optimizer()        # optimizer step done
        p.start_validation()
        # validation loop
        p.stop()
        p.save(trial_dir)
    """

    def __init__(self, epoch: int, batch_size: int, eval_batch_size: int,
                 num_workers: int, trial_dir: str):
        self.epoch = epoch
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.trial_dir = trial_dir
        self._epoch_start = time.perf_counter()
        self._phase = "train"
        self._timer = time.perf_counter()

        self.profile = EpochProfile(
            epoch=epoch,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
        )

    def tick_dataloader(self):
        """Call when a new batch arrives from dataloader."""
        now = time.perf_counter()
        self.profile.data_time_s += now - self._timer
        self._timer = now

    def tick_h2d(self):
        """Call after frames.to(device) and gt_bboxes.to(device)."""
        now = time.perf_counter()
        self.profile.h2d_time_s += now - self._timer
        self._timer = now

    def tick_forward(self):
        """Call after model(frames) — forward pass only."""
        now = time.perf_counter()
        self.profile.forward_time_s += now - self._timer
        self._timer = now

    def tick_loss(self):
        """Call after criterion(pred, gt_bboxes) — loss computation."""
        now = time.perf_counter()
        self.profile.loss_time_s += now - self._timer
        self._timer = now

    def tick_backward(self):
        """Call after loss.backward() + unscale + clip_grad_norm."""
        now = time.perf_counter()
        self.profile.backward_time_s += now - self._timer
        self._timer = now

    def tick_optimizer(self):
        """Call after optimizer.step() + scaler.update()."""
        now = time.perf_counter()
        self.profile.optimizer_time_s += now - self._timer
        self._timer = now

    def start_validation(self):
        """Call before validation loop."""
        self._timer = time.perf_counter()
        self._phase = "val"

    def stop(self):
        """Call after validation loop and all epoch work is done."""
        self.profile.validation_time_s = time.perf_counter() - self._timer
        self.profile.total_time_s = time.perf_counter() - self._epoch_start

        if torch.cuda.is_available():
            self.profile.gpu_allocated_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
            self.profile.gpu_reserved_gib = torch.cuda.max_memory_reserved() / (1024 ** 3)
            self.profile.gpu_util_pct = _get_gpu_utilization()

        # Bottleneck heuristic
        dt = self.profile.data_time_s
        compute = self.profile.compute_time_s
        vl = self.profile.validation_time_s
        if dt > 0.35 * (compute + dt) and dt > 1.0:
            self.profile.bottleneck = "dataloader"
        elif vl > 0.5 * self.profile.total_time_s:
            self.profile.bottleneck = "validation"
        elif compute > 0.6 * self.profile.total_time_s:
            self.profile.bottleneck = "compute"
        else:
            self.profile.bottleneck = "balanced"

    def save(self):
        """Append profile to trial_dir/profiles.jsonl."""
        path = os.path.join(self.trial_dir, "profiles.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.profile.to_dict(), ensure_ascii=False) + "\n")

    def summary_line(self) -> str:
        """Single-line summary for console output."""
        p = self.profile
        total_train = p.train_step_time_s
        return (f"  [profile e{p.epoch}] data={p.data_time_s:.0f}s "
                f"h2d={p.h2d_time_s:.1f}s fwd={p.forward_time_s:.0f}s "
                f"loss={p.loss_time_s:.0f}s bwd={p.backward_time_s:.0f}s "
                f"opt={p.optimizer_time_s:.0f}s val={p.validation_time_s:.0f}s "
                f"gpu={p.gpu_allocated_gib:.2f}GiB util={p.gpu_util_pct:.0f}% "
                f"bottleneck={p.bottleneck}")
