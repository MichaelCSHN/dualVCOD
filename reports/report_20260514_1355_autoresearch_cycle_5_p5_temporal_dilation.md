# Autoresearch Cycle 5 — P5 Temporal Dilation
## 2026-05-14 13:55

---

## Hypothesis
Wider temporal context via temporal_stride (frame spacing) provides richer
motion signal. Stride=2 (~0.33s at 30fps) and stride=3 (~0.50s) probe whether
temporal context width matters and whether the effect is monotonic.

## Experiments

| Config | Stride | Epochs | Batches/Ep | Time/Ep |
|--------|--------|--------|-----------|---------|
| `expl_p5_dilation_stride2_8ep.json` | 2 | 8 | 511 | 214s |
| `expl_p5_dilation_stride3_8ep.json` | 3 | 8 | 461 | 193s |

Baseline: E-45 (MV3-Small, stride=1, 8ep, pf_mIoU=0.8281)
Single variable: temporal_stride. No code changes — config only.

---

## P5-S2: Stride=2 Training Trajectory

| Epoch | Train Loss | Train mIoU | Val mIoU | R@0.5 |
|-------|-----------|-----------|---------|-------|
| 1 | 1.007 | 0.379 | 0.237 | 0.103 |
| 2 | 0.742 | 0.518 | 0.269 | 0.158 |
| 3 | 0.640 | 0.584 | 0.260 | 0.149 |
| 4 | 0.570 | 0.631 | 0.295 | 0.181 |
| 5 | 0.518 | 0.668 | 0.288 | 0.203 |
| **6** | **0.468** | **0.707** | **0.300** | **0.209** |
| 7 | 0.428 | 0.737 | 0.296 | **0.214** |
| 8 | 0.400 | 0.760 | 0.296 | 0.202 |

Peak val: epoch 6 (0.300). Final train mIoU: 0.760.

## P5-S2: Unified Reeval

| Checkpoint | Epoch | Train Val | pf_mIoU | bad | R@0.5 |
|-----------|-------|-----------|---------|-----|-------|
| Rank 1 | 6 | 0.300 | 0.755 | 0.093 | 0.907 |
| Rank 2 | 7 | 0.296 | 0.788 | 0.073 | 0.927 |
| **Rank 3** | **8** | **0.296** | **0.798** | **0.062** | **0.938** |

Training-val vs unified: 4/4 experiments now show disagreement.
Epoch 6 (train-best) → worst unified (0.755). Epoch 8 (train-worst) → best unified (0.798).

### P5-S2 vs E-45

| Metric | E-45 (stride=1) | P5-S2 (stride=2) | Delta |
|--------|----------------|-----------------|-------|
| pf_mIoU | **0.8281** | 0.7984 | **-0.030** ❌ |
| bad_frame_rate | **0.050** | 0.062 | +0.012 |
| R@0.5 | **0.950** | 0.938 | -0.012 |
| IoU_tiny | 0.584 | **0.599** | +0.015 |
| IoU_small | **0.706** | 0.688 | -0.018 |
| IoU_medium | 0.838 | **0.854** | +0.016 |
| IoU_large | **0.860** | 0.824 | **-0.036** ❌ |
| n_pred_too_large | **21** | 81 | +60 ⚠️ |
| n_pred_too_small | **79** | 92 | +13 |
| n_center_shift | **7** | 28 | +21 ⚠️ |

---

## P5-S3: Stride=3 Training Trajectory

| Epoch | Train Loss | Train mIoU | Val mIoU | R@0.5 |
|-------|-----------|-----------|---------|-------|
| 1 | 1.049 | 0.362 | 0.269 | 0.062 |
| 2 | 0.800 | 0.484 | 0.327 | 0.176 |
| 3 | 0.673 | 0.565 | 0.299 | 0.162 |
| 4 | 0.596 | 0.617 | 0.314 | 0.182 |
| 5 | 0.531 | 0.664 | 0.340 | 0.249 |
| **6** | **0.487** | **0.696** | **0.357** | **0.254** |
| 7 | 0.446 | 0.729 | 0.347 | 0.264 |
| 8 | 0.419 | 0.750 | 0.354 | **0.290** |

**Training val was strikingly high** — peak 0.357 vs E-45 peak 0.305.
This created a strong (false) signal that stride=3 was the "sweet spot."

## P5-S3: Unified Reeval

| Checkpoint | Epoch | Train Val | pf_mIoU | bad | R@0.5 |
|-----------|-------|-----------|---------|-----|-------|
| Rank 1 | 6 | 0.357 | 0.559 | 0.348 | 0.652 |
| **Rank 2** | **8** | **0.354** | **0.578** | **0.330** | **0.670** |

Only 2 checkpoints saved (heap had 2 entries at end; 3rd epoch may have
been from epoch 1 which was evicted with no replacement cycle completing).

### P5-S3 vs E-45

| Metric | E-45 (stride=1) | P5-S3 (stride=3) | Delta |
|--------|----------------|-----------------|-------|
| pf_mIoU | **0.8281** | 0.5781 | **-0.250** ❌ |
| bad_frame_rate | **0.050** | 0.330 | +0.280 |
| R@0.5 | **0.950** | 0.670 | -0.280 |
| IoU_tiny | **0.584** | 0.485 | -0.099 |
| IoU_small | **0.706** | 0.236 | **-0.470** ❌ |
| IoU_medium | **0.838** | 0.557 | **-0.281** |
| IoU_large | **0.860** | 0.814 | -0.046 |
| n_pred_too_large | **21** | 1239 | +1218 ❌❌ |
| area_ratio_mean | 1.0 | 2.51 | +151% |

---

## Consolidated P5 Analysis

### Monotonic Degradation Confirmed

| Metric | Stride=1 | Stride=2 | Stride=3 | Pattern |
|--------|---------|---------|---------|---------|
| pf_mIoU | 0.8281 | 0.7984 | 0.5781 | ↘ monotonic |
| IoU_small | 0.706 | 0.688 | 0.236 | ↘ accelerating |
| IoU_medium | 0.838 | 0.854 | 0.557 | ↗ then ↘ collapse |
| n_pred_too_large | 21 | 81 | 1239 | ↗ accelerating |
| bad_frame_rate | 0.050 | 0.062 | 0.330 | ↗ accelerating |

Training val was completely misleading — it showed stride=3 as best (0.357)
but unified reeval showed it as catastrophic (0.578). The training val ratio
collapsed from 2.66× (stride=2) to 1.63× (stride=3), meaning stride=3's
training val was artificially inflated.

### Why temporal dilation fails

1. **Frame decorrelation**: At stride=3 (~0.50s gap), objects move enough
   between frames that temporal correspondence becomes ambiguous. The model
   can't establish which feature in frame t belongs to which in frame t+3.

2. **Window overlap reduction**: stride=3 produces fewer windows per video
   (461 vs 511 batches/epoch), reducing effective training data. The model
   sees fewer temporal contexts.

3. **Train/val split interaction**: With wider stride, the random video split
   produces more divergent temporal distributions between train and val sets,
   inflating training val while deflating unified reeval.

4. **Dense temporal context is critical**: The original stride=1 (~0.17s)
   provides the right temporal granularity for MV3-Small. The temporal
   neighborhood module was designed for adjacent-frame context.

### Training Val vs Unified Reeval: 4/4 Disagreement

| Experiment | Train-Best Epoch | Train Val | Unified pf | Unified-Best Epoch | Unified pf |
|-----------|-----------------|-----------|-----------|-------------------|-----------|
| P1 (zoom) | 5 | 0.305 | 0.740 | **8** | **0.831** |
| P3 (coverage) | 5 | 0.307 | 0.777 | **8** | **0.842** |
| P5-S2 | 6 | 0.300 | 0.755 | **8** | **0.798** |
| P5-S3 | 6 | 0.357 | 0.559 | **8** | **0.578** |

**Pattern**: The LAST epoch is consistently the best by unified reeval,
regardless of training val trajectory. Training val ranking is anti-correlated
with true performance after epoch ~5 for 8ep runs.

---

## Go/No-Go Criteria

### P5-S2
| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| pf_mIoU +0.01 (success) | >0.838 | 0.798 | ❌ NOT MET |
| pf_mIoU drop >0.01 (no-go) | >-0.01 | -0.030 | **TRIGGERED** |
| Any size bin drop >0.03 (no-go) | >-0.03 | Large -0.036 | **TRIGGERED** |

**Verdict: NO-GO** ❌

### P5-S3
| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| pf_mIoU drop >0.01 (no-go) | >-0.01 | -0.250 | **TRIGGERED** |
| Any size bin drop >0.03 (no-go) | >-0.03 | IoU_small -0.470 | **TRIGGERED** |

**Verdict: NO-GO** ❌ (Catastrophic)

---

## P5 Verdict: **NO-GO** ❌ (Both Strides)

Temporal dilation (stride > 1) degrades performance monotonically.
Dense temporal context (stride=1, ~0.17s window) is the correct default.
P5 probes confirm the existing design choice and close the temporal dilation
direction.

### Implications for architecture
- TemporalNeighborhood was designed for adjacent-frame context — wider spacing
  breaks the core assumption of fine-grained temporal correspondence
- The T=5, stride=1 configuration is validated as the correct default
- Future temporal experiments (if any) should focus on T (number of frames)
  rather than stride (spacing between frames)

---

## Phase Summary: P1-P5 Complete

| Phase | Experiment | pf_mIoU | vs E-45 | Verdict |
|-------|-----------|---------|---------|---------|
| P1 | Zoom (conservative) | 0.831 | +0.003 | NO-GO (tiny -22%) |
| P3 | Coverage (low weight) | 0.842 | +0.014 | NO-GO (borderline) |
| P5-S2 | Temporal stride=2 | 0.798 | -0.030 | NO-GO |
| P5-S3 | Temporal stride=3 | 0.578 | -0.250 | NO-GO |

All Phase 2 experiments rejected. E-45 (MV3-Small, 8ep, baseline) remains
the strongest 8ep configuration. E-40 (MV3-Small, 30ep, 0.8564) and E-52
(EffB0, 30ep, 0.8711) remain the overall performance ceiling.

The core architecture (dense_fg_aux, T=5, stride=1, DIoU, hard target) is
validated as the correct minimalist design. Attempted improvements all failed
or showed insufficient evidence.

## Decision: Proceed to P6 Dense Target Refinement

P6 is the final active phase (P2/P4 are EffB0 variants, closed by P1/P3
no-go). P6 modifies `dense_target_mode` which is already wired through the
trial runner as a config parameter.

Next: P6 config — `expl_p6_dense_target_soft_8ep.json` exploring soft vs
hard dense targets. This was the "cautious probe" deferred from earlier
phases.
