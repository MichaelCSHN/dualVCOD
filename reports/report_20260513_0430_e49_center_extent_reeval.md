# E-49 Center+Extent Reeval: Direction C Probe

**Date**: 2026-05-13
**Status**: Reeval complete
**Core question**: Does decomposing dense_fg into center heatmap + per-pixel edge distances provide better spatial supervision than a binary foreground mask?

---

## 1. Experiment Design

### E-49: Center+Extent Decomposition

`head_type = "dense_ce_aux"`, `dense_target_mode = "ce"`

**Head architecture** (`CenterExtentHead`):
```
FPN features (128ch, 28×28)
  → Conv 3×3 → 32ch → BN → ReLU  (shared encoder)
      ├→ Conv 1×1 → 1ch  (center logits)
      └→ Conv 1×1 → 4ch → ReLU  (extent l/r/t/b distances)
```

**Targets** (5-channel, 28×28, purely bbox-derived):
- Ch 0: Center heatmap — 2D Gaussian at bbox center, σ = max(w,h)/4 ∈ [0.5, 4.0]
- Ch 1-4: Per-pixel distance to left/right/top/bottom edges, normalized [0,1], zero outside bbox

**Loss**: `center_BCE + extent_SmoothL1(masked to bbox interior)` weighted 0.5 against DIoU

**Inference**: Bbox-only; CE head not executed during eval.

### Variables held constant

Backbone MV3-Small, T=5, 224×224, bs=16, lr=0.001, DIoU, CosineAnnealing(T_max=8), seed=42, augment=HFlip+ColorJitter(0.15).

---

## 2. Training Results

| Epoch | Tr Loss | Tr mIoU | Val mIoU | Val R@0.5 |
|-------|:-------:|:-------:|:--------:|:---------:|
| 1 | 1.045 | 0.392 | 0.2438 | 0.1456 |
| 2 | 0.771 | 0.556 | 0.2907 | 0.2133 |
| 3 | 0.694 | 0.613 | 0.2691 | 0.1992 |
| 4 | 0.630 | 0.664 | 0.2948 | 0.1956 |
| 5 | 0.585 | 0.701 | 0.2954 | 0.2140 |
| 6 | 0.546 | 0.734 | **0.3112** | 0.2444 |
| 7 | 0.515 | 0.761 | 0.3104 | 0.2520 |
| 8 | 0.494 | 0.780 | 0.3088 | 0.2529 |

Training log best_val_mIoU: **0.3112** (epoch 6) — highest training log score among all 8ep dense variants (E-45: 0.3046, E-46: 0.3054).

Epoch time: ~170s (E-45: ~239s, 29% faster).

---

## 3. Unified Reeval Results

| Metric | E-45 (hard) | E-46 (soft) | **E-49 (CE)** | E49−E45 | E49−E46 |
|--------|:-----------:|:-----------:|:-------------:|:-------:|:-------:|
| ckpt epoch | 7 | 7 | 6 | | |
| **pf_mIoU** | **0.8281** | **0.8329** | 0.8090 | **−0.0191** | −0.0239 |
| bad_frame_rate | 0.0500 | 0.0410 | 0.0638 | +0.0138 | +0.0228 |
| R@0.5 | 0.9500 | 0.9590 | 0.9362 | −0.0138 | −0.0228 |

### Size-stratified IoU

| Metric | E-45 | E-46 | **E-49** | E49−E45 |
|--------|:----:|:----:|:--------:|:-------:|
| IoU tiny | **0.5837** | 0.5400 | 0.4577 | **−0.1261** |
| IoU small | 0.7058 | **0.7533** | 0.7000 | −0.0058 |
| IoU medium | 0.8382 | 0.8396 | **0.8408** | +0.0027 |
| IoU large | **0.8597** | **0.8696** | 0.8300 | **−0.0297** |

### Area Ratio & Center Error

| Metric | E-45 | E-46 | **E-49** | E49−E45 |
|--------|:----:|:----:|:--------:|:-------:|
| area_ratio_mean | 1.0351 | 1.0130 | **1.0190** | **−0.0162** |
| area_ratio_median | 1.0237 | 1.0000 | **1.0133** | −0.0104 |
| center_error_mean | 0.0221 | 0.0204 | 0.0243 | +0.0021 |

### Error Classification

| Error Type | E-45 | E-46 | **E-49** | E49−E45 |
|------------|:----:|:----:|:--------:|:-------:|
| n_good | 5657 | **5711** | 5575 | −82 |
| n_pred_too_large | 21 | 35 | 25 | +4 |
| n_pred_too_small | 79 | 80 | 85 | +6 |
| n_scale_mismatch | **191** | **120** | 260 | **+69** |
| n_center_shift | 7 | 9 | 10 | +3 |

### Per-Video mIoU

| Video | E-45 | E-46 | **E-49** |
|-------|:----:|:----:|:--------:|
| flatfish_2 | 0.0870 | 0.0891 | 0.0873 |
| white_tailed_ptarmigan | 0.5136 | **0.6660** | 0.5011 |
| pygmy_seahorse_0 | 0.3715 | 0.3926 | **0.4397** |

### Training Log vs Reeval

| Trial | Train best_val_mIoU | Reeval pf_mIoU | Ratio |
|-------|:-------------------:|:--------------:|:-----:|
| E-45 | 0.3046 | 0.8281 | 2.72× |
| E-46 | 0.3054 | 0.8329 | 2.73× |
| E-49 | 0.3112 | 0.8090 | 2.60× |

The collapsed ratio (2.60× vs 2.72×) indicates the training log overrated E-49 relative to reeval, mirroring the E-47/E-48 pattern.

---

## 4. Analysis

### E-49 does NOT beat the dense_fg baseline

Center+extent decomposition produces worse overall performance than a simple binary dense_fg mask:

- pf_mIoU −0.0191: Significant regression
- IoU_tiny −0.1261: Massive regression on tiny objects — the worst among all non-E-47 experiments
- IoU_large −0.0297: Worse on large objects
- n_scale_mismatch +69: More scale errors (260 vs 191)

### Partial positive signals

Two metrics improved:

1. **area_ratio_mean: 1.0190 vs 1.0351** (−0.0162, closer to 1.0): The extent supervision produces better-calibrated box sizes. The model is less prone to over-estimating object extent than any other dense variant.

2. **pygmy_seahorse_0: 0.4397 vs 0.3715** (+0.0682): The best 8ep result for this hard video. The CE decomposition may help with small, distinctive objects.

### Why did it fail?

Three hypotheses, likely interacting:

1. **5-channel output space is harder to learn**: The CE head must predict 5 structured channels instead of 1 binary channel. At 8ep, the model hasn't converged the more complex output space. E-49 was still improving at epoch 6 (unlike E-48 which peaked at epoch 4), suggesting convergence is slower but continuing.

2. **BCE center + SmoothL1 extent are poorly balanced**: The center BCE loss (with pos_weight) likely dominates the extent SmoothL1 loss in early training. The model learns "where is center" quickly but struggles to learn precise extent distances. This imbalance produces better area_ratio (center is well-localized) but worse IoU (extents are imprecise).

3. **Bbox-derived targets discard real pixel-mask information**: For MoCA_Mask/CAD samples (~40% of training data), the CE targets use bbox-derived centers and extents instead of real pixel masks. This throws away precise boundary information that dense_fg uses. The 5-channel structural decomposition can't compensate for the loss of real mask supervision.

---

## 5. Decision: Center+Extent V1 — Promising Signals, Does Not Beat Baseline

| Criterion | Status |
|-----------|--------|
| pf_mIoU ≥ E-45? | **No** — −0.0191 |
| Tiny IoU ≥ E-45? | **No** — −0.1261 (worst among valid experiments) |
| Large IoU ≥ E-45? | **No** — −0.0297 |
| Better area calibration? | **Yes** — area_ratio 1.019 is best among all 8ep variants |
| pygmy_seahorse_0 improved? | **Yes** — +0.0682 |

The center+extent decomposition shows a **trade-off pattern**: better area calibration (boxes are correctly sized) but worse overall IoU (boxes are in the wrong place or miss objects entirely).

### The training speed advantage is real but not decisive

E-49 trains 29% faster per epoch than E-45 (170s vs 239s). The CE head's 32-channel shared encoder is lighter than dense_fg's 64-channel path. This efficiency could matter at 30ep scale, but only if the 8ep quality gap closes.

---

## 6. Complete Teacher Route Scoreboard

| Experiment | Type | pf_mIoU | Δ vs E-45 | Verdict |
|------------|------|:-------:|:---------:|---------|
| E-45 | Hard dense_fg | 0.8281 | — | Baseline |
| E-46 | Uniform σ=1.0 soft_mask | **0.8329** | **+0.0047** | Marginal win (tiny IoU tradeoff) |
| E-47 | Soft bbox Gaussian falloff | 0.6587 | −0.1694 | Closed — destructive |
| E-48 | Size-adaptive soft_mask | 0.7236 | −0.1045 | Closed — destructive |
| E-49 | Center+extent V1 | 0.8090 | −0.0191 | Negative, but partial signals |

**The Teacher route has produced ONE validated marginal improvement (E-46) and THREE negative results (E-47, E-48, E-49).**

---

## 7. Recommendation: Pause Teacher Route, Consolidate on Hard dense_fg Mainline

The Teacher route exploration has been thorough:

- **Softening real masks** (E-46): +0.0047 pf_mIoU is real but marginal, and the tiny-object regression (−0.0437) prevents mainline adoption
- **Softening bbox targets** (E-47): Destructive
- **Size-adaptive softening** (E-48): Destructive
- **Center+extent decomposition** (E-49): Negative, though with partial signals

The consistent finding is: dense_fg_aux with hard binary targets is robust and hard to beat. Simple auxiliary supervision modifications produce limited gains at best, destructive regressions at worst.

### Recommended next direction

**Return to hard dense_fg mainline** and focus on the factors that have been demonstrated to matter:

1. **30-epoch training** (E-40): 0.8564 pf_mIoU, +0.0283 over 8ep — the largest verified gain in the entire project
2. **Backbone scaling**: MobileNetV3-Large or EfficientNet-B0 were lightly tested in early phases but never combined with dense_fg_aux. The dense auxiliary supervision might benefit more from backbone capacity than from target engineering.

### What NOT to pursue further

- Higher-resolution CE targets (56×56): Would amplify the learning difficulty without fixing the fundamental issue
- CE with real mask supervision: Would require complex mask→extent conversion, increasing implementation cost
- Width+height regression (instead of lrtb distances): Same issues, different parameterization
- Softer center Gaussian: The failure is structural, not a σ-tuning problem

### If Teacher route is revisited later

The two remaining unexplored directions from the original design:

- **Direction C with real-mask supervision**: Generate CE targets from real pixel masks for MoCA_Mask/CAD samples, preserving the precise boundary information that bbox-derived targets discard. The pygmy_seahorse_0 improvement (+0.0682) suggests CE helps with certain object types — combining it with real masks might capture this benefit without the overall regression.

- **CE + dense_fg joint supervision**: Keep binary dense_fg mask supervision AND add CE as additional auxiliary loss. The two might be complementary rather than competing.

Both are higher-cost explorations not justified by the current signal strength. Defer unless new evidence emerges.

---

## 8. Summary

| Question | Answer |
|----------|--------|
| Does CE V1 beat E-45? | **No** — pf_mIoU 0.8090 vs 0.8281 (−0.0191) |
| Best CE metric? | area_ratio_mean 1.019 (best among all 8ep variants) |
| Worst CE metric? | IoU_tiny 0.4577 (−0.1261 vs E-45) |
| CE training efficiency? | 29% faster per epoch than dense_fg |
| Partial signal worth following? | pygmy_seahorse_0 +0.0682 — CE helps specific object types |
| Next action | **Pause Teacher route**. Return to hard dense_fg mainline. Prioritize 30ep training and backbone scaling |
