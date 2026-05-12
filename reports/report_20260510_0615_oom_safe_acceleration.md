# OOM-Safe Acceleration and Auto-Resume Design

**Date:** 2026-05-10 06:15
**Author:** AutoResearch (MichaelCSHN)
**Status:** Design — no implementation yet
**Dependency:** PyTorch 2.6.0+cu124, RTX 4090 (23 GiB)

---

## 1. Motivation

Current trial runner (`tools/autoresearch/run_trial_minimal.py`) uses a fixed batch_size=16, fixed eval_batch_size=train_batch_size, no gradient accumulation, and no OOM recovery. When OOM occurs, the trial is abandoned — wasting compute and obscuring whether the failure was architectural (bad model) or infrastructural (batch too big).

This design introduces:
- **Zero-overhead profiling**: per-epoch timing breakdown at negligible cost
- **OOM auto-recovery**: emergency checkpoint → process restart → reduced batch → resume
- **Batch ladder**: optional pre-training probe to find max safe batch size
- **Gradient accumulation**: compensates for reduced batch to preserve effective batch size
- **Safety guardrails**: no retry on illegal memory access / GPU lost; no concurrent training; no auto commit/push

---

## 2. Architecture

### 2.1 New Module Layout

```
tools/autoresearch/
├── run_trial_minimal.py          ← main entry (extended)
├── profiler.py                   ← NEW: timing + memory profiling
├── batch_ladder.py               ← NEW: safe batch-size probe
├── oom_recovery.py               ← NEW: emergency checkpoint + resume
└── config_safety.py              ← NEW: config validation + CUDA env setup
```

### 2.2 Trial Lifecycle with OOM Recovery

```
┌─────────────────┐
│  Load trial      │
│  config          │
└────────┬────────┘
         ▼
┌─────────────────┐    batch_too_large
│  Batch Ladder    │──────────────────┐
│  (optional)      │                  │
└────────┬────────┘                  │
         │ batch_ok                    │
         ▼                             ▼
┌─────────────────┐    OOM ┌──────────────────┐
│  Training Loop   │────────│  Emergency Save   │
│  w/ profiling    │        │  (ckpt + config)  │
└────────┬────────┘        └────────┬─────────┘
         │                           │
         │ complete                  ▼
         ▼                   ┌──────────────────┐
┌─────────────────┐         │  Restart Process   │
│  Final Eval      │         │  lower batch_size  │
│  Save metadata   │         │  load emergency ckpt│
└────────┬────────┘         └────────┬─────────┘
         │                           │
         ▼                           ▼
     [done]              ┌───────────────────┐
                         │  Resume Training    │
                         │  from saved epoch   │
                         └────────┬──────────┘
                                  │ retry_count > 2
                                  ▼
                          ┌───────────────────┐
                          │  Mark as            │
                          │  resource_failed    │
                          └───────────────────┘
```

### 2.3 Config Schema Extension

```json
{
  "batch_size": 16,                    // legacy: sets both train + eval
  "train_batch_size": null,            // NEW: override train batch
  "eval_batch_size": null,             // NEW: override eval batch
  "batch_ladder": false,               // NEW: run pre-train batch probe
  "batch_ladder_max_power": 2,         // NEW: max 2^x steps above base
  "batch_safety_factor": 0.75,         // NEW: use 75% of max safe batch
  "grad_accum_steps": 1,               // NEW: gradient accumulation
  "oom_max_retries": 2,                // NEW: max OOM recovery attempts
  "num_workers": 2,
  "eval_num_workers": null,            // NEW: separate eval workers
  "cuda_alloc_conf": "max_split_size_mb:256",  // NEW: CUDA allocator config
  "profile_every_epoch": true,         // NEW: enable per-epoch profiling
  "benchmark_epoch": 1                 // NEW: dedicate epoch 1 to profiling
}
```

---

## 3. Component Design

### 3.1 Profiler (`profiler.py`)

**Goal:** Per-epoch timing breakdown + memory tracking with near-zero overhead.

**Design:**
```python
@dataclass
class EpochProfile:
    epoch: int
    data_time_s: float           # time spent in DataLoader __iter__ + collate
    forward_time_s: float        # model(frames) + loss computation
    backward_time_s: float       # scaler.scale(loss).backward()
    optimizer_time_s: float      # scaler.step() + scaler.update()
    validation_time_s: float     # full val loop
    total_time_s: float
    gpu_allocated_gib: float     # torch.cuda.max_memory_allocated() / 1e9
    gpu_reserved_gib: float      # torch.cuda.max_memory_reserved() / 1e9
    cpu_ram_gib: float           # psutil.Process.memory_info().rss
    batch_size: int
    eval_batch_size: int
    num_workers: int
    bottleneck: str              # "compute" | "dataloader" | "validation" | "balanced"
```

**Implementation approach:**
- Wrap `train_loader` iteration with `time.perf_counter()` timers
- Use `torch.cuda.synchronize()` before each timer boundary for accurate GPU timing
- Capture `torch.cuda.max_memory_allocated()` at epoch boundaries (already partially done)
- Bottleneck heuristic: if `data_time > 0.3 * (forward + backward)` → dataloader bound; if `forward + backward > 0.6 * total` → compute bound
- Store profiles in `trial_dir/profiles.jsonl` (append per epoch)
- Print single-line profile summary after each epoch log line

**Overhead:** ~2 `time.perf_counter()` calls per batch (essentially zero). `cuda.synchronize()` already exists in the current loop.

### 3.2 Batch Ladder (`batch_ladder.py`)

**Goal:** Find maximum safe batch size before starting actual training.

**Algorithm:**
```
1. Build model + optimizer + criterion with base config
2. Load a small fixed dataset subset (200 samples, same seed)
3. For batch_size in [16, 24, 32, 48, 64]:
   a. Create DataLoader with candidate batch_size
   b. Run 3 training steps: forward + backward + optimizer.step()
   c. Run 3 eval steps: forward only
   d. If CUDA OOM → break, use previous batch_size
   e. Record peak memory
4. Report: max_safe_train_batch, max_safe_eval_batch
5. Set actual batch = floor(max_safe * safety_factor)
```

**Key design decisions:**
- Use same model architecture, input size, and T as the real trial
- Warm-up the GPU first (1 dummy pass) to avoid cold-start allocator artifacts
- Separately test train batch (forward+backward, higher memory) and eval batch (forward only, lower memory)
- Run both with `torch.no_grad()` for eval memory test, `torch.autocast` for train memory test
- The probe runs BEFORE the real DataLoader is built (isolated memory)
- On OOM: `torch.cuda.empty_cache()`, try next (not crash)

**Cost:** ~2-3 minutes per trial. Optional (config: `batch_ladder: true`). Recommended for: new backbones, T > 5, input_size > 224.

**Safety:** The ladder starts SMALL and scales UP. Never starts large and crashes. OOM in ladder is expected and handled gracefully.

### 3.3 Train/Eval Batch Separation

**Goal:** Use larger batch for eval (no gradients, less memory) to speed up validation.

**Current state:** `val_loader` uses same `batch_size` as train.

**Design:**
```python
train_batch_size = trial_config.get("train_batch_size", batch_size)
eval_batch_size = trial_config.get("eval_batch_size", batch_size * 2)
# eval can typically use 2x-4x train batch since no backward pass
```

**OOM logic:**
- If OOM during training → halve `train_batch_size`, keep `eval_batch_size` unchanged
- If OOM during validation → halve `eval_batch_size`, keep `train_batch_size` unchanged
- Minimum batch_size = 1 (single sample)

**Implementation:**
```python
train_loader = DataLoader(..., batch_size=train_batch_size, ...)
val_loader = DataLoader(..., batch_size=eval_batch_size, ...)
```

### 3.4 OOM Auto-Recovery (`oom_recovery.py`)

**Goal:** Catch OOM, save state, restart with reduced batch.

**Emergency Checkpoint Format:**
```python
emergency_ckpt = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "scaler_state_dict": scaler.state_dict(),
    "epoch": current_epoch,           # resume from NEXT epoch
    "batch_idx": batch_idx,           # for intra-epoch resume
    "rng_state": {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    },
    "trial_config": trial_config,     # augmented with reduced batch
    "train_metrics_so_far": metrics_log,  # preserve partial metrics
    "retry_count": retry_count + 1,
    "failure_type": "resource_oom",
    "oom_context": {
        "phase": "train" | "val" | "final_eval",
        "epoch": current_epoch,
        "batch_in_epoch": batch_idx,
        "gpu_allocated_gib": ...,
        "gpu_reserved_gib": ...,
    },
    "saved_at": datetime.now().isoformat(),
}
```

**Recovery Protocol:**
```
1. except torch.cuda.OutOfMemoryError as e:
2.   log("OOM at epoch {e} batch {b}: {msg}")
3.   torch.cuda.empty_cache()
4.   Save emergency checkpoint to trial_dir/emergency_ckpt.pt
5.   Update trial_config: reduce affected batch_size by factor of 2,
       increase grad_accum_steps to maintain effective batch
6.   Save augmented config to trial_dir/config_resume.json
7.   Record OOM in trial_dir/oom_events.jsonl
8.   sys.exit(0)  # clean exit, not crash
9. Launcher script detects emergency checkpoint (metadata.json absent,
   emergency_ckpt.pt present), restarts with:
   python run_trial_minimal.py --trial_id ... --resume emergency_ckpt.pt
10. run_trial_minimal.py loads emergency checkpoint, restores all state,
    continues training from saved epoch/batch
```

**Critical safety rules:**
- **Illegal memory access** (`CUDA illegal memory access`, `CUBLAS_STATUS_EXECUTION_FAILED`): do NOT retry. Log the error, mark trial `hardware_error`, stop. These signal real bugs (out-of-bounds indexing, NaN poisoning), not resource pressure.
- **GPU lost** (`CUDA error: unknown error`, driver timeout): do NOT retry. Mark `hardware_error`. GPU instability requires manual investigation.
- **Repeated OOM** after 2 retries with minimum batch: mark `resource_failed`. The model genuinely can't fit on this GPU.
- **Never skip** `torch.cuda.empty_cache()` after OOM — stale allocations persist.

### 3.5 CUDA Allocator Configuration

**Environment setup** (runs before `import torch` in the trial process):
```python
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
```

**For PyTorch >= 2.5** (our version: 2.6.0):
`expandable_segments:True` is available in PyTorch 2.4+. It pre-allocates large segments that expand on demand, reducing fragmentation OOM. However, it conflicts with `max_split_size_mb` in some CUDA driver versions. For RTX 4090 (23 GiB), `expandable_segments:True` is low-risk.

**Recommendation:**
```python
_alloc_conf = "max_split_size_mb:256,expandable_segments:True"
# Fall back to no-expandable if driver < 535
# CUDA 12.4 generally ships with driver >= 545
```

**Recording:** Log `os.environ["PYTORCH_CUDA_ALLOC_CONF"]` at trial start.

### 3.6 Gradient Accumulation

**Goal:** When OOM forces batch_size reduction, compensate via gradient accumulation to maintain effective batch size.

**Design:**
```python
effective_batch = train_batch_size * grad_accum_steps

# Training loop:
optimizer.zero_grad()
for micro_step in range(grad_accum_steps):
    with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
        pred = model(frames[micro_step::grad_accum_steps])
        losses = criterion(pred, gt_bboxes[micro_step::grad_accum_steps])
        loss = losses["loss"] / grad_accum_steps  # normalize
    scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
scaler.step(optimizer)
scaler.update()
```

**Simpler approach (preferred for stability):** Don't implement micro-batching within the DataLoader loop. Instead, just call `scaler.scale(loss).backward()` multiple times without calling `optimizer.step()`. The `.backward()` accumulates gradients. After `grad_accum_steps` forward/backward passes, call `optimizer.step()` once.

```python
optimizer.zero_grad()
for _ in range(grad_accum_steps):
    # get next batch from loader
    frames, gt_bboxes = next(train_iter)
    with torch.amp.autocast(...):
        pred = model(frames)
        losses = criterion(pred, gt_bboxes)
        loss = losses["loss"] / grad_accum_steps
    scaler.scale(loss).backward()
scaler.unscale_(optimizer)
clip_grad_norm_(...)
scaler.step(optimizer)
scaler.update()
```

**Tracking:** `effective_batch_size` recorded in metadata. `grad_accum_steps` logged per epoch.

---

## 4. Implementation Plan

### Phase A: Core safety (1-2 hours, no breaking changes)

| File | Change | Risk |
|------|--------|------|
| `run_trial_minimal.py` | Wrap main() in OOM try/except; add `--resume` flag | Low — additive |
| `oom_recovery.py` | Emergency checkpoint save/load functions | None — new file |
| `run_trial_minimal.py` | Separate `train_batch_size` / `eval_batch_size` config keys | Low — backward compatible |

### Phase B: Profiling (30 min)

| File | Change | Risk |
|------|--------|------|
| `profiler.py` | EpochProfile dataclass + timing hooks | None — new file |
| `run_trial_minimal.py` | Add perf_counter() timers around loop sections | Low — additive |

### Phase C: Batch ladder (1 hour)

| File | Change | Risk |
|------|--------|------|
| `batch_ladder.py` | Full ladder implementation | None — new file |
| `run_trial_minimal.py` | Optional `--run_batch_ladder` before training | Low — gated by config flag |

### Phase D: Gradient accumulation + CUDA config (30 min)

| File | Change | Risk |
|------|--------|------|
| `run_trial_minimal.py` | Add `grad_accum_steps` loop | Medium — changes training semantics |
| `config_safety.py` | CUDA allocator env setup + validation | None — new file |

### Phase E: Integration testing (1 hour)

- Test OOM recovery with artificially high batch_size → expect successful degrade
- Test batch ladder on MV3Large, EffB0 (new backbones)
- Test gradient accumulation produces identical gradients to normal batch (equivalence check)
- Test emergency checkpoint round-trip (save → load → resume → same loss)
- Test illegal memory access does NOT retry

---

## 5. Compatibility with Current Pipeline

### 5.1 Backward Compatibility
- All new config keys default to current behavior: `train_batch_size` defaults to `batch_size`, `eval_batch_size` defaults to `train_batch_size * 2`, `grad_accum_steps` defaults to 1, `batch_ladder` defaults to false.
- Existing trial configs (E-12 through E-21) run unchanged.
- `metadata.json` gains new optional fields (doesn't break existing parsers).

### 5.2 Running Trial (E-21)
- **No interruption.** E-21 continues with current code path.
- The new runner will be a separate entry point or gated by `--use_safe_runner` flag until validated.
- All changes are additive — the old code path remains as fallback.

### 5.3 Anti-Hang Mechanisms (Preserved)
- `cv2.setNumThreads(0)` — unchanged
- DataLoader `timeout=120` — unchanged, applied per-loader
- Heartbeat every 10 batches — unchanged
- `persistent_workers=False` — unchanged

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Emergency ckpt save itself triggers OOM | Low | High | `torch.cuda.empty_cache()` before save; save to CPU |
| Batch ladder OOM corrupts GPU state | Medium | Medium | `torch.cuda.empty_cache()` after each ladder step; fresh process for real training |
| Gradient accumulation changes training dynamics | Medium | High | Equivalence test: same effective batch = same loss after N batches; flag in metadata |
| expandable_segments causes driver issues | Low | Medium | Fallback to max_split_size_mb only; detect at trial start |
| Resume from mid-epoch breaks DataLoader state | Medium | Medium | Only resume at epoch boundaries (simpler, safer); discard partial epoch |
| Illegal memory access misidentified as OOM | Low | Critical | Explicit exception type checking; OOM catch is `torch.cuda.OutOfMemoryError` only |

---

## 7. Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Trial loss from OOM | 100% of trials marked failed | 0% (auto-degrade instead) |
| Eval throughput | ~85 FPS | ~120+ FPS (separate eval batch) |
| GPU utilization | ~60-70% (batch=16) | ~80-90% (ladder-fitted batch) |
| Profiling visibility | GPU mem only | Full per-epoch timing breakdown |
| Recovery time after OOM | Manual restart (hours) | Automatic resume (<2 min) |

---

## 8. What This Does NOT Do

- Does NOT change model architecture or loss functions
- Does NOT change dataset splitting or MoCA clean val protocol
- Does NOT enable concurrent training (explicitly prohibited)
- Does NOT auto commit/push (explicitly prohibited)
- Does NOT submit checkpoints/logs externally (explicitly prohibited)
- Does NOT retry on illegal memory access / GPU lost
- Does NOT change the LR schedule, optimizer, or AMP strategy
- Does NOT modify the existing checkpoint format (emergency ckpt is separate)

---

## 9. Next Steps

1. **Review this design** — confirm scope and priorities
2. **Phase A implementation** — core safety first (OOM recovery + batch separation)
3. **Shadow test** — run new runner alongside old on one 5ep probe, compare results
4. **Phase B+C+D** — profiling, ladder, gradient accumulation
5. **Full integration** — make new runner the default for E-22+ trials

---

*Generated by AutoResearch. No trials were interrupted or re-run in producing this report.*
