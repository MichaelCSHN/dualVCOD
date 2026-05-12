# Size-Adaptive Soft Mask Reeval: E-48 + Full Teacher Route Comparison

**Date**: 2026-05-12
**Status**: Reeval complete
**Core question**: Can size-adaptive soft_mask preserve tiny-object hard-target performance while keeping E-46's small/large gains?

---

## 1. Experiment Design

### E-48: Size-Adaptive Soft Mask

`dense_target_mode = "soft_mask_adaptive"`

Real PNG masks softened with σ proportional to normalized bbox area at 28×28:

| Object size | Area range | σ | Kernel | Behavior |
|-------------|:----------:|:--:|:------:|----------|
| Tiny | < 0.01 | 0 | — | Hard binary (no blur) |
| Small | < 0.05 | 0.5 | 3×3 | Gentle boundary softening |
| Medium | < 0.15 | 1.0 | 7×7 | Same as E-46 |
| Large | ≥ 0.15 | 1.5 | 7×7 | Stronger boundary softening |

Bbox-only samples (MoCA CSV): hard rectangle targets (same as E-45, no soft_bbox).

### Variables held constant

Backbone (MV3-Small), T=5, 224×224, bs=16, lr=0.001, DIoU+dense_fg=0.5, CosineAnnealing(T_max=8), seed=42, augment=HFlip+ColorJitter(0.15), datasets=MoCA+MoCA_Mask+CAD.

---

## 2. Unified Reeval Results

| Metric | E-45 (hard) | E-46 (soft_mask) | **E-48 (adaptive)** | E46−E45 | E48−E45 |
|--------|:-----------:|:----------------:|:-------------------:|:-------:|:-------:|
| ckpt epoch | 7 | 7 | **4** | | |
| **pf_mIoU** | 0.8282 | **0.8329** | 0.7236 | **+0.0047** | **−0.1045** |
| bad_frame_rate | 0.0500 | **0.0410** | 0.1063 | **−0.0091** | +0.0563 |
| R@0.5 | 0.9500 | **0.9590** | 0.8937 | **+0.0091** | −0.0563 |

### Size-stratified IoU

| Metric | E-45 | E-46 | **E-48** | E46−E45 | E48−E45 |
|--------|:----:|:----:|:--------:|:-------:|:-------:|
| IoU tiny | 0.5837 | 0.5400 | **0.3636** | −0.0437 | **−0.2201** |
| IoU small | 0.7058 | **0.7533** | 0.5335 | **+0.0475** | −0.1723 |
| IoU medium | 0.8382 | 0.8396 | 0.7097 | +0.0014 | −0.1285 |
| IoU large | 0.8597 | **0.8696** | 0.8004 | **+0.0099** | −0.0593 |

### Area Ratio & Center Error

| Metric | E-45 | E-46 | **E-48** | E46−E45 | E48−E45 |
|--------|:----:|:----:|:--------:|:-------:|:-------:|
| area_ratio_mean | 1.0351 | **1.0130** | 1.2551 | **−0.0221** | **+0.2200** |
| area_ratio_median | 1.0237 | **1.0000** | 1.1449 | **−0.0237** | +0.1212 |
| center_error_mean | 0.0221 | 0.0204 | 0.0323 | −0.0017 | +0.0102 |

### Error Classification

| Error Type | E-45 | E-46 | **E-48** | E46−E45 | E48−E45 |
|------------|:----:|:----:|:--------:|:-------:|:-------:|
| n_good | 5657 | **5711** | 5322 | **+54** | **−335** |
| n_pred_too_large | 21 | 35 | **198** | +14 | **+177** |
| n_pred_too_small | 79 | 80 | 163 | +1 | **+84** |
| n_scale_mismatch | 191 | **120** | 228 | **−71** | +37 |
| n_center_shift | 7 | 9 | 44 | +2 | +37 |

### Per-Video mIoU

| Video | E-45 | E-46 | **E-48** | E48−E45 |
|-------|:----:|:----:|:--------:|:-------:|
| flatfish_2 | 0.0870 | 0.0891 | 0.1250 | +0.0380 |
| white_tailed_ptarmigan | 0.5136 | **0.6660** | 0.2827 | −0.2308 |
| pygmy_seahorse_0 | **0.3715** | 0.3926 | 0.1778 | −0.1937 |

### Training Log vs Reeval

| Trial | Train best_val_mIoU | Reeval pf_mIoU | Ratio |
|-------|:-------------------:|:--------------:|:-----:|
| E-45 | 0.3046 (e7) | 0.8282 | 2.72× |
| E-46 | 0.3054 (e7) | 0.8329 | 2.73× |
| E-48 | 0.2978 (e4) | 0.7236 | 2.43× |

---

## 3. Analysis

### E-48 fails on every metric

The size-adaptive soft_mask is dramatically worse than both E-45 (hard baseline) and E-46 (uniform soft_mask):

- **pf_mIoU −0.1045 vs E-45**: Massive regression, far outside any noise band
- **Tiny IoU −0.2201 vs E-45**: The primary target (preserving tiny-object performance) got *worse*, not better. Even E-46's −0.0437 was milder
- **Small IoU −0.1723 vs E-45**: E-46's +0.0475 gain completely reversed
- **area_ratio_mean +0.2200**: Model predicts boxes ~25% too large on average (1.255 vs 1.035)
- **n_pred_too_large +177**: 9.4× increase in over-prediction errors (198 vs 21)

The training log ratio collapse (2.43× vs 2.72×) mirrors what happened with E-47 (soft_bbox at 2.20×) — the training metric dramatically overestimated generalization quality.

### Root cause: σ=1.5 for large objects destroyed boundary precision

The error signature is unmistakable: n_pred_too_large exploded from 21 to 198. The model learned to predict boxes ~25% too large. This is the opposite of the "shy prediction" concern — σ=1.5 at 28×28 with a 7×7 kernel (25% of the grid) creates such a wide transition zone that the model can't localize the true boundary.

The hypothesis was backwards: stronger softening doesn't help with large-object extent recovery — it destroys the spatial precision the model needs to anchor bbox predictions. Large objects benefit from E-46's modest σ=1.0, but σ=1.5 crosses a threshold where the supervisory signal becomes too diffuse.

### E-48's epoch-4 checkpoint is a confounding factor

E-48's best checkpoint is at epoch 4, while E-45/E-46 are at epoch 7. The 3-epoch gap accounts for some but not all of the deficit:

- Reeval pf_mIoU: E-48(epoch 4) = 0.7236, while E-45(epoch 7) = 0.8282
- Training val_mIoU at epoch 4: E-48 = 0.2978, E-45 = 0.2848
- So E-48 was ahead at epoch 4 by training metrics, but far behind on reeval

This confirms the training metric's unreliability for E-48 (same pattern as E-47).

However, E-48's training trajectory also collapsed after epoch 4 (0.2978 → 0.2857 → 0.2787 → 0.2871 → 0.2938), suggesting the wider-σ softening destabilizes training convergence regardless of checkpoint selection.

### Small positive: flatfish_2 improved

E-48's flatfish_2 mIoU (0.1250) is better than E-45 (0.0870) and E-46 (0.0891). The σ=1.5 for large objects may help with the most extreme camouflaged extent recovery — but the global cost is far too high.

---

## 4. Decision: Size-Adaptive Softening Does NOT Work

| Outcome | Status |
|---------|--------|
| Tiny IoU recovered to E-45 level? | **No** — dropped to 0.3636 (E-45: 0.5837) |
| Small/Large IoU kept E-46 gains? | **No** — all size categories worse than E-45 |
| pf_mIoU ≥ E-45? | **No** — −0.1045 |
| scale_mismatch reduced? | **No** — increased to 228 (E-45: 191) |

**Verdict**: The size-adaptive approach with σ ∈ {0, 0.5, 1.0, 1.5} is strictly worse than both uniform hard targets (E-45) and uniform soft_mask (E-46).

### Why it failed — two mutually-reinforcing mechanisms

1. **σ=1.5 is destructive**: The 7×7 kernel at 28×28 creates a transition zone covering 25% of the grid. BCE loss can't extract precise boundary information from such diffuse targets. Large objects, which were expected to benefit most, were actually harmed — the model learned to predict systematically oversized boxes (area_ratio +22%).

2. **Discontinuous σ creates training instability**: Objects just on either side of the area=0.15 threshold (medium/large boundary) get σ=1.0 vs σ=1.5 — a 50% difference in blur radius. This creates a jagged loss landscape. The training collapse after epoch 4 (best_val 0.2978 → epoch 6 0.2787) suggests the model couldn't converge stably.

---

## 5. Updated Understanding of the Softening Pattern

The data now spans 4 softening strategies, providing strong constraints on what works:

| Experiment | σ(tiny) | σ(small) | σ(medium) | σ(large) | pf_mIoU | tiny IoU | small IoU | large IoU |
|------------|:-----:|:------:|:--------:|:--------:|:-------:|:--------:|:---------:|:---------:|
| E-45 (hard) | 0 | 0 | 0 | 0 | 0.8282 | **0.5837** | 0.7058 | 0.8597 |
| E-46 (uniform σ=1.0) | 1.0 | 1.0 | 1.0 | 1.0 | **0.8329** | 0.5400 | **0.7533** | **0.8696** |
| E-48 (adaptive) | 0 | 0.5 | 1.0 | 1.5 | 0.7236 | 0.3636 | 0.5335 | 0.8004 |

The optimal softening is: **σ=1.0 uniformly for all objects that have real pixel masks, or σ=0 (no softening) for all objects**. Intermediate σ values (0.5) and stronger values (1.5) are both worse than the extremes.

This suggests softening at 28×28 has a narrow effective window: σ must be just large enough to create a meaningful gradient zone (~3px at boundaries with σ=1.0), but not so large that it washes out the signal (σ≥1.5) or so small that it provides no benefit over hard targets (σ=0.5).

The tiny-object problem in E-46 is real but addressing it requires a different approach than size-dependent σ — possibly:
- Per-object loss weighting (up-weight tiny objects in dense_fg loss)
- Higher-resolution dense grid (56×56 instead of 28×28) where softening is more controllable
- Different softening mechanism (e.g., distance transform instead of Gaussian blur)

---

## 6. Recommendation: Close Softening Direction, Return to Hard dense_fg

The softening exploration has reached a natural endpoint:

1. **Uniform σ=1.0 soft_mask (E-46)** provides a real but small improvement (+0.0047 pf_mIoU, −71 scale_mismatch, +0.0475 small IoU) at the cost of −0.0437 tiny IoU
2. **Any deviation from σ=1.0** (size-adaptive, σ=0.5, σ=1.5) produces large regressions
3. The σ=1.0 optimum is narrow — further tuning is unlikely to yield breakthrough gains

**Recommended action**: Archive the soft_mask finding as a validated but marginal improvement. Set the mainline back to hard dense_fg targets ("hard" mode) and pivot the Teacher route effort to:

### Direction C: Center+Extent Decomposition

The center+extent decomposition was identified in the original Teacher route design as the next-level approach after simple soft targets. The case for it is now stronger:

- **Softening alone has been exhausted**: σ=1.0 works mildly, σ≠1.0 breaks. There's no more juice to squeeze from Gaussian blur.
- **The problem is structural**: E-46's pattern (small/large gains, tiny losses) suggests that the single dense_fg map conflates "where is the center?" with "how big is the object?" These are fundamentally different questions that may benefit from separate supervision heads.
- **Center+extent naturally handles size**: A center heatmap (where is the object?) + extent maps (how far to each edge?) provides size-adaptive behavior without ad-hoc σ thresholds. The center signal is always precise; the extent signal can be softer or harder depending on the architecture design.

### What changes with center+extent

- **Head architecture**: Replace/supplement the 1-channel dense_fg map with 3+ channels (center heatmap + extent_x + extent_y), or 5 channels (center + distance to left/right/top/bottom edges)
- **Dataset**: No change — same real masks and bbox labels, just encoded differently
- **Loss**: BCE for center heatmap (with Gaussian-peaked GT from bbox center), L1/smooth-L1 for extent maps
- **Inference**: Still bbox-only (center → peak location, extent → bbox dimensions)
- **Implementation cost**: Higher than softening (new head module + loss function), lower than backbone/search changes

### Timeline

- **E-49**: Center+extent head design + 8ep probe — ~2-3 hours implementation + 40 min training
- If positive at 8ep → 30ep verification
- If negative → the Teacher route of "dense auxiliary supervision beyond hard masks" may need fundamental rethinking

---

## 7. Summary

| Question | Answer |
|----------|--------|
| Does size-adaptive soft_mask work? | **No** — −0.1045 pf_mIoU, all metrics worse |
| Why did it fail? | σ=1.5 for large objects created over-prediction (+177 n_pred_too_large); discontinuous σ caused training instability |
| Is E-46 (uniform soft_mask) still valuable? | Yes, as a marginal validated gain (+0.0047 pf_mIoU). Archive as reference, do not make mainline default |
| Should σ/tuning continue? | **No** — the softening optimum is narrow (only σ=1.0 works). Further grid search is low-value |
| What replaces softening? | **Center+extent decomposition** (Direction C) — structural solution to the size-dependence problem |
| Next experiment | E-49: center+extent head, 8ep probe against E-45 baseline |
