# Hung Trial Recovery and DataLoader Timeout Patch Report

**Generated**: 2026-05-09 ~14:50  
**Trial**: expl_12_30ep → expl_12_30ep_retry_timeout  
**Status**: infra/dataloader hang (NOT model failure)

---

## 1. Hung Process Forensic State

| Field | Value |
|-------|-------|
| **PID** | 10716 (killed before full forensics — prior session kill) |
| **Start time** | 2026-05-08 22:09:50 |
| **CPU time at hang** | ~7715 sec (2.14h) over ~14h wall clock → **85% idle** |
| **Memory** | ~1629–1746 MB (stable, no leak) |
| **Last log write** | 2026-05-08 22:19:24 |
| **Last successful epoch** | 1 |
| **GPU memory** | 1484 MiB / 24564 MiB (5% utilization, idle) |
| **GPU temp** | 30°C (idle) |
| **Artifacts** | config.json, checkpoint_best.pth (5.8 MB), trial.log (872 bytes) |
| **Missing artifacts** | metadata.json, metrics.json (trial never completed) |

### Epoch 1 Metrics (partial evidence, preserved)

| Metric | Value |
|--------|-------|
| Train Loss | 0.75400 |
| Train mIoU | 0.3801 |
| Val mIoU | 0.2425 |
| Val R@0.5 | 0.1352 |
| LR | 0.000400 (warmup step 1/3) |
| Time | 549.1s (~9.15 min) |

### Conclusion: NOT a model failure

- Model forward/backward works — epoch 1 completed 558 batches successfully
- OOM mitigation effective — no CUDA OOM with batch_size=16 + `torch.no_grad()` in val
- Process hung between epoch 1→2, with 85% idle CPU and stable memory
- Classic DataLoader/I/O blocking hang, likely Windows multiprocessing deadlock

---

## 2. Kill Process

| Action | Result |
|--------|--------|
| Kill PID 10716 | Killed (prior session) |
| Kill PID 28860 (premature restart) | Killed |
| Residual Python check | None — clean |
| GPU memory check | 0 MiB — clean |

---

## 3. Root Cause Assessment

The hang occurred after epoch 1 completed (validation ran, metrics logged) but before epoch 2 training started. The DataLoader iterator for the next epoch with `shuffle=True` would recreate workers. With the previous `num_workers=0`, this is a main-process operation — but `cv2.imread` on Windows can block indefinitely on certain filesystem states.

Hypothesis: `cv2.imread` hung on a specific image file during the first batch of epoch 2. With no timeout mechanism, the process blocked forever.

---

## 4. Anti-Hang Patches Applied to `run_trial_minimal.py`

| # | Patch | Detail |
|---|-------|--------|
| 1 | `cv2.setNumThreads(0)` | Prevent OpenCV internal threading from interfering with DataLoader |
| 2 | `DataLoader(timeout=120)` | PyTorch kills worker if batch not returned within 120s |
| 3 | `persistent_workers=False` | Workers recreated each epoch (safe on Windows) |
| 4 | `prefetch_factor=2` | Limit prefetch queue depth |
| 5 | `num_workers=2` | Two worker processes (timeout requires >0) |
| 6 | **Batch heartbeat** | Log every 10 batches: epoch, batch, last_10s, loss, miou |
| 7 | **Batch timing stats** | Track mean/max batch time per epoch, logged in epoch summary |
| 8 | **Image sanity scan** | Quick scan of 200 random MoCA images before training (cv2.imread + exists check) |

---

## 5. Retry Trial Configuration

| Field | Value |
|-------|-------|
| **New trial_id** | `expl_12_30ep_retry_timeout` |
| **Parent trial** | `expl_12_30ep` (hung) |
| **Backbone** | mobilenet_v3_small |
| **Input size** | 224 |
| **Temporal T** | 5 |
| **Head** | current_direct_bbox |
| **LR** | 0.001 |
| **Batch size** | 16 (safe, verified) |
| **Epochs** | 30 |
| **Warmup** | 3 epochs |
| **num_workers** | 2 |
| **Loss weights** | smooth_l1=1.0, giou=1.0, center=0.5 |
| **No commit** | True |
| **No push** | True |

---

## 6. Retry Result

*Pending — trial launching after report generation.*

---

## 7. Fallback Plan (if retry hangs again)

1. Do NOT blindly retry a third time
2. Run a 1-epoch debug trial with `num_workers=0` and verbose image-level logging
3. If `num_workers=0` succeeds: confirms multiprocessing/worker issue → stay at num_workers=0, add Python-level timeout watchdog thread
4. If `num_workers=0` also hangs: isolate bad file via per-sample try/except, log failing paths, skip bad samples
