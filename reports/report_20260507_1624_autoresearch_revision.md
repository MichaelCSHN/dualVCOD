# Phase 1.6 AutoResearch Plan Revision — 2026-05-07 16:24

## 0. Summary of Changes

This revision addresses 10 design issues identified in the v1 plan. All changes are design-only — no training has been run, no code has been committed, no checkpoints have been loaded or moved.

## 1. Design Issues Fixed

### 1.1 Eval T Rule (Issue #1)

**Old**: eval_T=5 hardcoded for all trials regardless of training T.
**New**: Primary eval T = training temporal_T. Diagnostic T=5 eval run in a separate pass when training T ≠ 5 and model architecture supports variable-T input (Conv1d + AdaptiveAvgPool make this possible in current MicroVCOD). Scoring/promotion use primary metric only. Diagnostic T=5 is recorded in metadata as `diag_val_miou_T5` for cross-T comparison.

**Rationale**: A model trained with T=3 should be evaluated on T=3 — evaluating with T=5 introduces a distribution shift that penalizes the model unfairly. The diagnostic T=5 pass provides additional information but does not affect ranking.

### 1.2 Baseline Reproduction Proxy (Issue #2)

**Old**: First smoke trial was efficientnet_b0 at 512px — no verification that run_trial.py produces identical results to train.py.
**New**: `smoke_b0_baseline_proxy` added as mandatory first trial. This replicates the current baseline (mobilenet_v3_small, sz=224, T=5, window_uniform, current_direct_bbox, lr=1e-3) via run_trial.py. Acceptance criterion: epoch-3 val_mIoU within 0.03 of train.py epoch-3 value. B0 must PASS before any B1 backbone swap trial can start.

**Rationale**: Without verifying run_trial.py reproduces train.py output, all subsequent trials could be comparing against a shifted baseline due to subtle differences in dataloader, loss accumulation, or metric computation. The proxy gate eliminates this risk.

### 1.3 Staged Engineering Decoupling (Issue #3)

**Old**: smoke_02 combined backbone swap + video_balanced sampler + objectness_aux_head in a single trial — 4 simultaneous changes.
**New**: Three-stage smoke plan:
- **B0** (1 trial): baseline proxy — 0 changes from train.py
- **B1** (3 trials): backbone swap only — sampler and head locked to window_uniform + current_direct_bbox
- **B2** (3 trials): sampler and head variants — on best B1 backbone, introduce video_balanced and objectness_aux_head one at a time

**Rationale**: Changing backbone, sampler, and head simultaneously makes it impossible to attribute performance changes. If smoke_02 failed, we wouldn't know whether the backbone, sampler, or head was at fault. Staged smoke isolates each variable.

### 1.4 Hard Reject Rules Enhanced (Issue #4)

**Old**: Single mIoU < 0.15 threshold applied uniformly regardless of epoch count.
**New**: Staged thresholds:

| Stage | Rule | Action |
|---|---|---|
| Always | OOM, NaN loss, data leak, params > 35M | Hard reject |
| ≥5 epochs | mIoU < center_prior + 0.03 (0.2317) | **Mark weak** (not rejected, but cannot be promoted) |
| ≥5 epochs | loss[last] > 2 × loss[1] | Hard reject |
| ≥12 epochs | mIoU < baseline - 0.03 (0.2561) | Hard reject |
| ≥12 epochs | R@0.5 < baseline - 0.05 (0.1478) | Hard reject |
| Universal | empty_pred_rate > 2% | Hard reject |
| Universal | global_area_ratio < 0.4 or > 2.5 | Hard reject |

**Rationale**: The 0.15 floor was too permissive — a model at 0.20 mIoU after 12 epochs is clearly worse than the 1.4M baseline at 0.2861 and should be rejected. The "weak" mark (not reject) at 0.2317 allows borderline trials to complete but prevents their promotion. Empty predictions > 2% indicate the model is not engaging with the data. Extreme area ratios indicate systematic bbox scale failure.

### 1.5 Scoring Function Enhanced (Issue #5)

**Old**: `composite = mIoU + 0.3×R@0.5 + 0.2×R@0.3`
**New**: `composite = base - area_penalty - empty_penalty - instability_penalty`

Where:
- `base = 1.0×mIoU + 0.3×R@0.5 + 0.2×R@0.3`
- `area_penalty = |ln(global_area_ratio)| × 0.05`
- `empty_penalty = empty_pred_rate × 0.50`
- `instability_penalty = 0.02 if std(mIoU_last_3) > 0.03 else 0.0`

Two area metrics recorded for every trial:
- **global_area_ratio** = mean(pred_area) / mean(gt_area) — aggregate bias
- **mean_sample_area_ratio** = mean(pred_area_i / gt_area_i) — per-sample calibration

**Rationale**: A model with mIoU=0.30 but global_area_ratio=3.0 (predicting bboxes 3× too large) is worse than a model with mIoU=0.29 and area_ratio=1.0. The penalty captures this. Empty predictions and training instability are also penalized. Both area metrics are recorded to avoid confusion in later analysis — they measure different things.

### 1.6 Objectness Aux Head Deferred (Issue #6)

**Old**: objectness_aux_head included in B1 smoke.
**New**: Marked as B2+ only in search space config. B0 and B1 trials are locked to current_direct_bbox. The head dimension remains defined in the YAML for exploration phases.

### 1.7 Video-Balanced Sampler Deferred (Issue #7)

**Old**: video_balanced included in B1 smoke.
**New**: Marked as B2+ only. B0 and B1 trials are locked to window_uniform. The sampler dimension remains defined for exploration phases.

### 1.8 Smoke Trial Sequence Revised (Issue #8)

**Old**: 4 diverse trials changing 4+ variables simultaneously.
**New**: 4 trials in staged order:

| ID | Phase | Backbone | Sz | T | Sampler | Head | LR | Epochs | Purpose |
|---|---|---|---|---|---|---|---|---|---|
| smoke_b0_baseline_proxy | B0 | mobilenet_v3_small | 224 | 5 | window_uniform | current_direct_bbox | 1e-3 | 1–3 | Reproduce train.py metrics |
| smoke_b1_effb0 | B1 | efficientnet_b0 | 512 | 5 | window_uniform | current_direct_bbox | 1e-3 | 3–5 | Minimal backbone upgrade |
| smoke_b1_mnv3large | B1 | mobilenet_v3_large | 512 | 5 | window_uniform | current_direct_bbox | 1e-3 | 3–5 | MV3 family scaling |
| smoke_b1_effb1 | B1 | efficientnet_b1 | 512 | 5 | window_uniform | current_direct_bbox | 3e-4 | 3–5 | Mid-capacity backbone, lower LR |

B2 trials (sampler/head smoke) follow after B1 confirms at least 1 backbone with mIoU > baseline - 0.02.

### 1.9 Script Readiness (Issue #9)

**Old**: trial.log was documented but not implemented.
**New**: `TeeLogger` class implemented in run_trial.py — captures both stdout and stderr to `trial.log`. Enabled before any GPU work starts. All trial artifacts (config.json, metrics.json, metadata.json, safety_check.json) are written even on failure. Failed trials get metadata with status="failed" and reason string.

### 1.10 Safety Enforcement (Issue #10)

**New**: All conditions confirmed:
- No training executed (verified: zero GPU time, no checkpoints touched)
- No commit (verified: git status shows all changes unstaged)
- No push (verified)
- No checkpoint deletion/movement (verified: all .pth files unchanged)
- No local_runs/checkpoints/outputs/logs/raw data staged

## 2. New Staged Smoke Plan

```
B0: smoke_b0_baseline_proxy  ── must PASS acceptance check ──▶
B1: smoke_b1_effb0, smoke_b1_mnv3large, smoke_b1_effb1  ──≥1 with mIoU>baseline-0.02──▶
B2: smoke_b2_vidbal, smoke_b2_objhead, smoke_b2_combined  ──▶
C:  24-trial Latin Hypercube exploration  ──▶
D:  Top-8 promotion to 12 epochs  ──▶
E:  Top-3 promotion to 30 epochs  ──▶
F:  Top-2 multi-seed confirmation
```

Each transition is gated by explicit acceptance criteria defined in the YAML config.

## 3. Updated Scoring Function

```
composite = base - area_penalty - empty_penalty - instability_penalty

base = 1.0 × val_miou + 0.3 × val_recall_at_0_5 + 0.2 × val_recall_at_0_3
area_penalty = |ln(global_area_ratio)| × 0.05
empty_penalty = empty_pred_rate × 0.50
instability_penalty = 0.02 if std(mIoU_last_3_epochs) > 0.03 else 0.0
```

Tiebreaker (in order): fewer params → higher FPS → higher mIoU.

Two area metrics recorded for every trial to prevent confusion:
- `global_area_ratio`: aggregate ratio — near 1.0 = unbiased on average
- `mean_sample_area_ratio`: per-sample ratio mean — captures per-sample calibration

## 4. Updated Hard Reject Rules

**Always-fire**: OOM, NaN loss, data leak, params > 35M
**After 5 epochs**: mark weak if mIoU < 0.2317; hard reject if loss diverging (>2× epoch1)
**After 12 epochs**: hard reject if mIoU < 0.2561 or R@0.5 < 0.1478
**Universal**: hard reject if empty_pred_rate > 2% or global_area_ratio < 0.4 or > 2.5

## 5. Eval T Primary/Diagnostic Rules

- **Primary eval T** = training temporal_T (always)
- **Diagnostic T=5 eval**: run only when training T ≠ 5 AND model architecture supports variable-T input
- **Scoring**: uses primary eval metrics only. Diagnostic T=5 recorded as `diag_val_miou_T5` in metadata.
- **Current architecture**: MicroVCOD TemporalNeighborhood uses Conv1d + AdaptiveAvgPool1d, which are T-agnostic — T=5 diagnostic eval is supported for all training T values
- **Future architectures**: if a backbone-specific temporal module is not variable-T, diagnostic eval is skipped with a log message

## 6. Readiness for B0 Baseline Proxy Smoke

### Is B0 ready to run?

**YES**, with the following conditions understood:

The `smoke_b0_baseline_proxy` trial uses:
- backbone=mobilenet_v3_small → standard MicroVCOD (matches train.py exactly)
- input_size=224 → matches train.py exactly
- T=5 → matches train.py exactly
- sampler=window_uniform → matches train.py exactly
- head=current_direct_bbox → matches train.py exactly
- lr=1e-3 → matches train.py exactly

**No model.py changes are needed for B0** — the current `MicroVCOD(T=5, pretrained_backbone=True)` is exactly the baseline architecture. B0 verifies that run_trial.py's dataloader, loss accumulation, metric computation, and checkpoint format are identical to train.py.

### What B0 will NOT test:
- Backbone swapping (that's B1)
- Sampler variants (that's B2)
- Head variants (that's B2)
- Variable eval T (training and eval both use T=5)

### B0 acceptance criteria:
- Epoch 1 val_mIoU ≈ train.py epoch 1 (0.2163 from report_20260507_1336.md)
- Epoch 3 val_mIoU within 0.03 of train.py epoch 3 (0.2674)
- Data manifest matches (126 unique canonical_video_ids, 8926 train windows, 1188 val windows)
- No empty predictions, no NaN, no data leak

## 7. Required src/model.py Changes Before B1 (Minimum List)

B0 requires zero changes to src/model.py. B1 backbone smoke requires:

1. **`SpatialEncoderFPN.__init__` parameterization** (~40 lines):
   - Add `backbone_name` parameter
   - Use `backbone_registry.py` to get factory and stage slices
   - Dynamically set lateral/smooth conv channel counts from probed backbone stage channels
   - Fallback to current mobilenet_v3_small if backbone_name is "mobilenet_v3_small"

2. **`MicroVCOD.__init__` passthrough** (~5 lines):
   - Add `backbone_name` parameter
   - Pass through to `SpatialEncoderFPN`

3. **Backbone stage probing** (~15 lines):
   - On first use of a backbone, run a dummy forward pass to verify channel counts
   - Cache results to avoid repeated probing

4. **ObjectnessAuxHead** (NOT needed for B1, needed for B2):
   - Parallel MLP head: `AdaptiveAvgPool2d(1) → Linear(128→32) → ReLU → Linear(32→1) → Sigmoid`
   - GT objectness = (GT bbox area > 0.001).float()
   - BCE loss with weight 0.1 added to BBoxLoss

Total changes to src/model.py for B1 readiness: ~60 lines.
Additional for B2 readiness: ~40 lines.

## 8. Files Modified in This Revision

| File | Status | Changes |
|---|---|---|
| `configs/autoresearch/search_space_phase1_6.yaml` | Rewritten | v1→v2: staged phases B0/B1/B2, feature matrix, updated scoring, staged hard reject, area metrics, smoke trial sequence |
| `tools/autoresearch/run_trial.py` | Rewritten | TeeLogger for trial.log, eval T=training_T + diagnostic T=5, area metrics (both), FPS with training T, failure metadata on all code paths |
| `tools/autoresearch/score_trials.py` | Rewritten | Enhanced composite with penalties, staged hard reject, weak marking, score breakdown recording, both area metrics in CSV |
| `tools/autoresearch/aggregate_trials.py` | Updated | Area metrics in tables, strong/weak/rejected breakdown, updated scoring formula footer |
| `tools/autoresearch/check_trial_safety.py` | Updated | Phase-feature compatibility check (B0/B1/B2 gating) |

All files compile clean. Phase compatibility check verified with 5 test cases (all pass).

## 9. Verification Checklist

- [x] All Python files compile clean (4/4)
- [x] Phase compatibility check functional (B0/B1/B2 gates tested)
- [x] No training executed
- [x] No git commit
- [x] No git push
- [x] No checkpoints moved/deleted
- [x] No local_runs/checkpoints/outputs/logs/raw data touched
- [x] Search space YAML updated to v2
- [x] Two area metrics documented and computed in run_trial.py
- [x] trial.log file logging implemented (TeeLogger)
- [x] Failure metadata written on all code paths
- [x] Staged hard reject with center_prior, baseline, empty, area thresholds
- [x] Weak marking separate from hard reject
- [x] objectness_aux_head deferred to B2+
- [x] video_balanced deferred to B2+
- [x] Baseline proxy smoke mandatory before any backbone swap

---
*Report generated at 2026-05-07T16:24:00*
*Phase 1.6 AutoResearch Plan Revision 1*
*Design only — no training, no commit, no push*
