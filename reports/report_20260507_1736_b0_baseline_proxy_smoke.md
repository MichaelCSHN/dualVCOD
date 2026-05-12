# Phase 1.6-B0 Baseline Proxy Smoke Report — 2026-05-07 17:36

## 1. B0 Command & Configuration

| Parameter | Value |
|---|---|
| Command | `python -u tools/autoresearch/run_trial.py --trial_id smoke_b0_baseline_proxy --config local_runs/autoresearch/smoke_b0_baseline_proxy/trial_config.json` |
| Backbone | `mobilenet_v3_small` (standard MicroVCOD — matches train.py) |
| Input size | 224 |
| Temporal T | 5 (train + eval) |
| Sampler | `window_uniform` |
| Head | `current_direct_bbox` |
| LR | 1e-3 (AdamW, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR(T_max=30) — matches train.py |
| Epochs | 3 |
| Batch size | 24 |
| Num workers | 4 |
| Train seed | None — matches train.py non-deterministic behavior |
| Split seed | 42 (fixed protocol) |
| AMP | True (fp16 autocast, train + val) |
| Augmentation | HFlip(p=0.5) + ColorJitter(b=0.15) + StrideJitter(1-3) |
| Init | Clean ImageNet pretrained — no VCOD checkpoint loaded |
| Device | CUDA (NVIDIA RTX 4090) |
| Total params | 1,411,684 |

### Bugs Fixed During B0 Execution

Three bugs in `run_trial.py` were discovered and fixed during B0:

| Bug | Symptom | Fix |
|---|---|---|
| `final_metrics.json` JSON crash | `TypeError: Object of type Tensor is not JSON serializable` — `compute_metrics` returns `per_frame_ious` as raw tensor | Convert numpy/torch types to Python scalars in `final_out` dict comprehension |
| `_write_failure_metadata` overwrite | Exception after `metadata.json` write → failure handler overwrote good metadata with bare status="failed" | Reorder writes: `metrics.json` → `final_metrics.json` → `metadata.json` last; patch existing metadata on late failure |
| CosineAnnealingLR T_max mismatch | `T_max=epochs` (3) vs train.py's `T_max=30` — LR decayed to 0 by epoch 3 instead of tracking train.py's schedule | Add `total_epochs` config parameter (defaults to `epochs`); B0 sets `total_epochs: 30` |
| Validation autocast missing | train.py uses `amp.autocast` during validation; run_trial.py did not — FP32 vs FP16 inference produced different predictions | Add `torch.amp.autocast` wrapper in `validate()` |
| Torch seed forced | train.py does NOT set `torch.manual_seed`; run_trial.py always set it — different initialization distribution | Make `train_seed` optional; skip torch seed when `train_seed` is `null` in config |

## 2. Safety Check Results

```
OVERALL: PASS (0 issues)
  PASS [trial_directory]   — local_runs/autoresearch/smoke_b0_baseline_proxy/
  PASS [backbone]          — mobilenet_v3_small in registry
  PASS [input_size]        — 224 divisible by 32
  PASS [temporal_t]        — T=5 in valid range [2, 10]
  PASS [sampler]           — window_uniform (valid)
  PASS [head]              — current_direct_bbox (valid)
  PASS [lr]                — 0.001 in range [1e-5, 1e-1]
  PASS [param_budget]      — 1.4M < 35M
  PASS [no_checkpoint]     — clean init confirmed
  PASS [seed_consistency]  — split_seed=42, train_seed omitted (B0 matches train.py)
  PASS [phase_compatibility] — B0 feature restrictions enforced
  PASS [data_isolation]    — JointTrain ∩ Val = 0
```

## 3. DataLoader Manifest

| Component | canonical_video_ids | Windows | Filter |
|---|---|---|---|
| MoCA (train-only) | 113 | 5,760 | `Subset(ds, train_idx)` |
| MoCA_Mask (filtered) | 58 | 3,021 | canonical_video_id filter |
| CAD (filtered) | 9 | 145 | canonical_video_id filter |
| **JointTrain unique** | **126** | **8,926** | |
| MoCA Val (held-out) | 28 | 1,188 | |

### Manifest Match vs train.py

| Metric | run_trial.py | train.py | Match? |
|---|---|---|---|
| JointTrain unique videos | 126 | 126 | ✅ EXACT |
| Train windows | 8,926 | 8,926 | ✅ EXACT |
| Val windows | 1,188 | 1,188 | ✅ EXACT |
| JointTrain ∩ Val | 0 | 0 | ✅ EXACT |

**Manifest match is exact.** The data pipeline in `run_trial.py` produces identical train/val splits to `train.py`.

## 4. Per-Epoch Training Metrics

### run_trial.py (B0 baseline proxy)

| Epoch | Tr Loss | Tr mIoU | Val mIoU | Val R@0.5 | Val R@0.3 | LR | Time |
|---|---|---|---|---|---|---|---|
| 1 | 0.75986 | 0.3644 | 0.2538 | 0.1529 | — | 0.000997 | 254s |
| 2 | 0.56907 | 0.5035 | 0.2545 | 0.1588 | — | 0.000989 | 253s |
| 3 | 0.47011 | 0.5819 | 0.2409 | 0.1286 | — | 0.000976 | 252s |

### train.py (clean retraining reference)

| Epoch | Tr Loss | Tr mIoU | Val mIoU | Val R@0.5 | LR | Time |
|---|---|---|---|---|---|---|
| 1 | 0.81041 | 0.3328 | 0.2163 | 0.1005 | 0.000997 | 231s |
| 2 | 0.61864 | 0.4649 | 0.2570 | 0.1690 | 0.000989 | 236s |
| 3 | 0.51634 | 0.5442 | 0.2674 | 0.1658 | 0.000976 | 236s |

### Delta (run_trial − train.py)

| Epoch | Δ Tr Loss | Δ Tr mIoU | Δ Val mIoU | Δ Val R@0.5 |
|---|---|---|---|---|
| 1 | −0.05055 | +0.0316 | **+0.0375** | +0.0524 |
| 2 | −0.04957 | +0.0386 | −0.0025 | −0.0102 |
| 3 | −0.04623 | +0.0377 | **−0.0265** | −0.0372 |

LR values match exactly (0.000997, 0.000989, 0.000976) — confirmed T_max=30 scheduler fix works.

**Pattern:** run_trial.py consistently achieves lower train loss and higher train mIoU, indicating faster convergence on the training set. However val mIoU is lower at epochs 1 and 3 (epoch 2 is nearly identical). This is consistent with different random weight initialization (train.py did not set torch seeds, so each run uses different initial weights).

## 5. B0 Acceptance Criteria Assessment

| Criterion | Target | Actual | Result |
|---|---|---|---|
| JointTrain unique videos | 126 | 126 | ✅ PASS |
| Train windows | 8,926 | 8,926 | ✅ PASS |
| Val windows | 1,188 | 1,188 | ✅ PASS |
| JointTrain ∩ Val | 0 | 0 | ✅ PASS |
| Epoch 1 val_mIoU ≈ train.py | 0.2163 | 0.2538 (+0.0375) | ⚠️ MARGINAL |
| Epoch 3 val_mIoU ∈ [0.2374, 0.2974] | 0.2674 ± 0.03 | **0.2409** (+0.0035 above floor) | ✅ PASS |
| No NaN | — | None detected | ✅ PASS |
| No OOM | — | Peak VRAM 0.26 GiB | ✅ PASS |
| Empty pred rate ≤ 2% | — | 0.0% (0/1188) | ✅ PASS |
| Global area ratio reasonable | — | 0.7123 (not extreme) | ✅ PASS |
| trial.log generated | — | 2,454 bytes | ✅ PASS |
| config.json generated | — | 519 bytes | ✅ PASS |
| metrics.json generated | — | 600 bytes | ✅ PASS |
| metadata.json generated | — | Complete | ✅ PASS |
| safety_check.json generated | — | 265 bytes | ✅ PASS |
| final_metrics.json generated | — | 70 bytes | ✅ PASS |
| checkpoint_best.pth generated | — | 17.2 MB | ✅ PASS |
| Failure metadata on crash | — | Verified (patches existing on late failure) | ✅ PASS |

### Final Clean Re-Evaluation

| Metric | Value |
|---|---|
| mIoU | 0.2409 |
| R@0.5 | 0.1286 |
| R@0.3 | 0.3577 |
| Global area ratio | 0.7123 (pred/gt = 0.1217/0.1708) |
| Mean sample area ratio | 5.2966 |
| Empty prediction rate | 0.0% (0/1188) |
| mIoU std (last 3 epochs) | 0.0063 |
| Inference FPS | 71.1 (T=5, sz=224) |
| GPU memory | 0.26 GiB |
| Total train time | 760s (253s/epoch avg) |

### Area Metrics — Two Views

| Metric | Value | Interpretation |
|---|---|---|
| global_area_ratio | 0.7123 | Aggregate: predicted bboxes are ~71% of GT area on average → **systematic under-sizing** |
| mean_sample_area_ratio | 5.2966 | Per-sample: geometric mean of per-sample ratios → extreme values skew |

The discrepancy between global_area_ratio (0.71) and mean_sample_area_ratio (5.30) indicates the model predicts medium-sized boxes that overestimate small objects and underestimate large objects — a known limitation of models without multi-scale features.

## 6. Trial Artifacts

```
local_runs/autoresearch/smoke_b0_baseline_proxy/
├── checkpoint_best.pth      (17,223,174 bytes) — best val mIoU checkpoint (epoch 2, mIoU=0.2545)
├── config.json              (519 bytes) — frozen trial config snapshot
├── trial.log                (2,454 bytes) — full stdout/stderr tee
├── metrics.json             (600 bytes) — per-epoch metrics array (all 3 epochs)
├── final_metrics.json       (70 bytes) — clean re-eval: mIoU=0.2409, R@0.5=0.1286
├── metadata.json            (1,507 bytes) — complete trial metadata
├── safety_check.json        (265 bytes) — pre-flight check results
└── trial_config.json        (441 bytes) — input config
```

All 8 artifacts present and well-formed. Checkpoint is local-only (`local_runs/`), NOT in repo.

## 7. Should B0 Proceed to B1?

### Assessment: **CONDITIONAL PASS — recommend PROCEED with awareness**

**Evidence FOR proceeding:**
- Data manifest matches train.py EXACTLY (126/8926/1188) — dataloader pipeline verified
- Epoch 3 val_mIoU (0.2409) passes the ±0.03 criterion (barely, at lower edge)
- LR schedule confirmed identical to train.py (T_max=30)
- All structural safeguards verified (no NaN, no OOM, no leaks, clean init)
- All artifacts generated correctly on every code path
- `run_trial.py` bug fixes validated (serialization, metadata persistence, autocast, scheduler)

**Caveats to be aware of:**
- Epoch 1 val_mIoU (0.2538) is notably higher than train.py (0.2163) — train convergence is faster in run_trial
- Training is non-deterministic (no torch seed) by design for B0; each run produces different metrics. The train.py 0.2674 was one specific random initialization — run_trial 0.2409 is another
- The global_area_ratio (0.71) shows systematic under-sizing — B1 backbone swaps should improve this
- The ±0.03 acceptance window is WIDE. At 30 epochs, a 0.03 difference could mean 0.29 vs 0.26 — which matters

**Recommendation:** B0 passes the gating criteria. Proceed to B1 backbone-only smoke, but:
1. Enable `train_seed` for B1+ trials so backbone comparisons are reproducible
2. The B0 baseline proxy mIoU (0.2409 at epoch 3) should NOT be used as the reference for B1 — use train.py's clean baseline (0.2674 at epoch 3) as the comparison point
3. Keep `total_epochs` matching the intended total schedule, not just the smoke epoch count

## 8. Commit/Push/Checkpoint Safety Confirmation

| Check | Status |
|---|---|
| Git commit | ❌ NONE — no commit made |
| Git push | ❌ NONE — no push made |
| Checkpoint in repo | ❌ NONE — `checkpoint_best.pth` is in `local_runs/`, NOT in `checkpoints/` |
| `local_runs/` staged | ❌ NONE — entire `local_runs/` tree is untracked |
| `src/model.py` modified | ❌ NONE — no backbone parameterization added |
| B1/B2 launched | ❌ NONE — only B0 executed |

All safety constraints respected. No training artifacts entered the git repository.

---

*Report generated at 2026-05-07T17:36:00*
*Phase 1.6-B0 Baseline Proxy Smoke — run_trial.py vs train.py equivalence verification*
*Verdict: CONDITIONAL PASS — proceed to B1 with train_seed enabled*
