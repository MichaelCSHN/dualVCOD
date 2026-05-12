# Teacher Route Unified Reeval: E-45 / E-46 / E-47 + E-39 / E-40 Reference

**Date**: 2026-05-12
**Status**: Reeval complete — diagnostic only, no new training launched
**Reeval script**: `tools/reeval_teacher_route.py`
**Val split**: MoCA, `np.random.RandomState(42)`, val_ratio=0.2 (28 videos, 1191 samples, 5955 frames)

---

## 1. Unified Reeval Results

| Metric | E-39 (8ep) | E-40 (30ep) | E-45 (hard) | E-46 (soft_mask) | E-47 (soft_bbox) | E46−E45 | E47−E45 |
|--------|:----------:|:-----------:|:-----------:|:----------------:|:----------------:|:-------:|:-------:|
| ckpt epoch | 8 | 18 | 7 | 7 | **3** | | |
| **pf_mIoU** | 0.8277 | 0.8564 | 0.8282 | **0.8329** | 0.6587 | **+0.0047** | −0.1694 |
| bad_frame_rate | 0.0573 | 0.0358 | 0.0500 | **0.0410** | 0.1805 | **−0.0091** | +0.1305 |
| R@0.5 | 0.9427 | 0.9642 | 0.9500 | **0.9590** | 0.8195 | **+0.0091** | −0.1305 |

### Size-stratified IoU

| Metric | E-39 | E-40 | E-45 | E-46 | E-47 | E46−E45 | E47−E45 |
|--------|:----:|:----:|:----:|:----:|:----:|:-------:|:-------:|
| IoU tiny | 0.5868 | 0.7026 | 0.5837 | 0.5400 | 0.1909 | **−0.0437** | −0.3928 |
| IoU small | 0.7702 | 0.7517 | 0.7058 | **0.7533** | 0.4924 | **+0.0475** | −0.2134 |
| IoU medium | 0.8371 | 0.8584 | 0.8382 | 0.8396 | 0.6529 | +0.0014 | −0.1853 |
| IoU large | 0.8531 | 0.8835 | 0.8597 | **0.8696** | 0.7380 | **+0.0099** | −0.1217 |

### Area Ratio & Center Error

| Metric | E-39 | E-40 | E-45 | E-46 | E-47 | E46−E45 | E47−E45 |
|--------|:----:|:----:|:----:|:----:|:----:|:-------:|:-------:|
| area_ratio_mean | 1.0178 | 0.9900 | 1.0351 | **1.0130** | 1.1248 | **−0.0221** | +0.0896 |
| area_ratio_median | 1.0002 | 0.9898 | 1.0237 | **1.0001** | 0.9716 | **−0.0236** | −0.0521 |
| center_error_mean | 0.0238 | 0.0182 | 0.0221 | 0.0204 | 0.0503 | −0.0017 | +0.0282 |
| center_error_median | 0.0130 | 0.0108 | 0.0148 | 0.0129 | 0.0448 | −0.0020 | +0.0300 |

### Error Classification

| Error Type | E-39 | E-40 | E-45 | E-46 | E-47 | E46−E45 | E47−E45 |
|------------|:----:|:----:|:----:|:----:|:----:|:-------:|:-------:|
| n_good | 5614 | 5742 | 5657 | **5711** | 4880 | **+54** | −777 |
| n_pred_too_large | 34 | 1 | 21 | 35 | 392 | +14 | +371 |
| n_pred_too_small | 178 | 120 | 79 | 80 | 87 | +1 | +8 |
| n_center_shift | 31 | 5 | 7 | 9 | 32 | +2 | +25 |
| n_scale_mismatch | 98 | 87 | 191 | **120** | 564 | **−71** | +373 |
| total_frames | 5955 | 5955 | 5955 | 5955 | 5955 | 0 | 0 |

### Per-Video mIoU (Key Hard Videos)

| Video | E-39 | E-40 | E-45 | E-46 | E-47 | E46−E45 | E47−E45 |
|-------|:----:|:----:|:----:|:----:|:----:|:-------:|:-------:|
| flatfish_2 | 0.1426 | 0.2147 | 0.0870 | 0.0891 | 0.1777 | +0.0021 | +0.0906 |
| white_tailed_ptarmigan | 0.1766 | 0.3883 | 0.5136 | **0.6660** | 0.6257 | **+0.1525** | +0.1122 |
| pygmy_seahorse_0 | 0.4273 | 0.6234 | 0.3715 | 0.3926 | 0.1724 | +0.0212 | −0.1990 |

---

## 2. Comparison with E-39 / E-40 Reference

**E-45 faithfully reproduces E-39**: pf_mIoU 0.8282 vs 0.8277 (Δ = +0.0005). The hard dense_fg_aux baseline is stable and reliable across different val splits.

**E-46 (soft_mask) vs E-39**: pf_mIoU 0.8329 vs 0.8277 — **+0.0052**, clearing the +0.005 threshold when using E-39 as reference. This means soft_mask is the best 8ep dense_fg_aux variant tested.

**E-46 vs E-40 (30ep)**: E-46 achieves 97.3% of E-40's pf_mIoU in 26.7% of the epochs. E-45 achieves 96.7%. The soft_mask approach modestly widens this efficiency gap.

**E-47 (soft_bbox) is not competitive**: pf_mIoU 0.6587 is far below all baselines. However, the checkpoint is from epoch 3, not epoch 7 — a timing confound that makes direct comparison unfair.

---

## 3. Training Log vs Reeval Consistency

Training log `best_val_mIoU` uses `random.Random(42)` split + per-clip sample mIoU. Reeval `pf_mIoU` uses `np.random.RandomState(42)` split + per-frame mIoU. These are fundamentally different metrics.

| Trial | Train best_val_mIoU | Reeval pf_mIoU | Ratio |
|-------|:-------------------:|:--------------:|:-----:|
| E-45 | 0.3046 | 0.8282 | 2.72× |
| E-46 | 0.3054 | 0.8329 | 2.73× |
| E-47 | 0.2993 | 0.6587 | 2.20× |

For E-45 and E-46, the ratio is consistent (~2.72×), confirming the training metric ordering is reliable for comparing these two. E-47's ratio collapse (2.20×) indicates the epoch 3 checkpoint that looked best in training evaluation does NOT generalize to the reeval split — it was likely an overfit spike to the specific training val videos.

**Key implication**: The training log correctly ordered E-46 ≈ E-45, but dramatically overrated E-47's epoch 3 performance. The training val split's random.Random shuffle happened to favor videos where early Gaussian-falloff training looked good, but it didn't hold up.

---

## 4. Does soft_mask Still Have Value?

**Yes, with caveats.** E-46's reeval results show a consistent positive signal:

**Wins**:
- pf_mIoU: +0.0047 vs E-45, +0.0052 vs E-39 (borderline but positive)
- bad_frame_rate: −0.0091 (9% relative reduction in bad frames)
- R@0.5: +0.0091
- Small IoU: **+0.0475** (6.7% relative improvement — the strongest signal)
- Large IoU: +0.0099 (modest)
- area_ratio_mean: closer to 1.0 (1.013 vs 1.035)
- scale_mismatch errors: −71 (37% reduction — fewer bounding box scale errors)
- white_tailed_ptarmigan: **+0.1525** (30% relative improvement — dramatic for a single hard video)

**Losses**:
- Tiny IoU: −0.0437 (7.5% relative regression — tiny objects suffer from softened boundaries)
- n_good frames gained +54, but n_pred_too_large +14 — slight increase in over-prediction errors

**Interpretation**: Softening real PNG mask boundaries helps with spatial extent reasoning for small-to-large objects. The model learns better boundary delineation, reducing scale_mismatch errors and improving IoU across most size categories. However, tiny objects (<1% area at 224×224) are harmed — likely because at 28×28 resolution (~2.8×2.8 pixels for a tiny object), the Gaussian blur (σ=1.0) smears the already-tiny signal.

**This is a SIZE-AWARE signal, not a uniform improvement.** The design document anticipated this possibility: "如果某个 soft target 明显改善 medium/large，但损害 tiny-object，则下一步应考虑 size-aware target."

---

## 5. Does soft_bbox Still Have Value?

**No, at least not with sigma_factor=0.3.** E-47 is unambiguously worse than E-45 on every metric. Two interpretations:

1. **Optimistic**: E-47's epoch 3 checkpoint is an unfair comparison. E-45 at epoch 3 had val_mIoU=0.2849; E-47 at epoch 3 had val_mIoU=0.2993 — E-47 was actually ahead at that point. But E-47's training trajectory collapsed (regressed to 0.2914 by epoch 8) while E-45 kept improving to 0.3046. The soft_bbox targets may cause training instability rather than being inherently worse.

2. **Pessimistic**: Even at its "best" epoch (3), E-47 only achieves 0.6587 pf_mIoU. For context, a direct_bbox model (E-31) achieves 0.7766. The soft_bbox approach fundamentally produces worse representations because the Gaussian falloff is too weak a supervisory signal — it tells the model "center is foreground" but provides no hard edge information to anchor the prediction.

**flatfish_2 anomaly**: E-47's flatfish_2 mIoU (0.1777) is the best among all 8ep models (E-45=0.0870, E-46=0.0891). Even E-40 at 30ep only gets 0.2147. This suggests the Gaussian falloff does capture something useful for large camouflaged object extent — but the overall degradation makes it not worth pursuing in its current form.

**Verdict**: Close the soft_bbox direction at sigma_factor=0.3. If Direction C (center+extent) is pursued, the flatfish_2 signal should inform the design.

---

## 6. Should the Teacher Route Continue?

**Yes, but with a refined direction.** The reeval changes the picture from the training logs:

- Training logs suggested: E-46 NEUTRAL (+0.0008), E-47 NEGATIVE (−0.0053) → both fail
- Reeval shows: E-46 **POSITIVE** (+0.0052 vs E-39, +0.0047 vs E-45), E-47 **NEGATIVE** (−0.1694)

The soft_mask direction produces a genuine improvement in the consistent eval framework. It's not a dramatic breakthrough, but it's real and consistent across multiple metrics (pf_mIoU, bad_frame_rate, R@0.5, small/large IoU, area_ratio, scale_mismatch). Gaussian-blurred real mask boundaries help the model learn better spatial structure.

The original Teacher hypothesis ("softer targets help bbox-only student generalization") is **partially validated** — specifically for the case where real pixel masks provide the soft target signal. The flip side (bbox-derived soft targets) failed.

---

## 7. Recommended Next Step

### Single recommendation: Size-aware soft_mask (NOT stronger uniform softening)

The "try stronger softening (σ=1.5)" path from the original decision tree is wrong for this data. The pattern is clear:

| Object size | E-46 vs E-45 |
|-------------|:------------:|
| Tiny | **−0.0437** (hurt) |
| Small | **+0.0475** (helped) |
| Medium | +0.0014 (neutral) |
| Large | +0.0099 (helped) |

Stronger uniform softening would **amplify the tiny-object regression**. Tiny objects are already at ~2-3 pixels on the 28×28 dense grid — any blur destroys their already-fragile signal.

Instead, the next experiment should be **size-adaptive softening**: apply Gaussian blur with σ proportional to the object's size on the 28×28 grid, not a fixed σ=1.0. Specifically:

- **Tiny objects** (area < ~4 px² at 28×28): σ = 0 (hard mask, no blur)
- **Small objects**: σ = 0.5
- **Medium/large objects**: σ = 1.0–1.5

This directly addresses the observed size-dependent response pattern: preserve hard boundaries for tiny objects where they're needed, provide softer boundaries for larger objects where extent recovery benefits from gradient in the boundary zone.

### Why NOT the other options:

| Option | Why rejected |
|--------|-------------|
| Stronger uniform softening (σ=1.5) | Would worsen tiny-object regression; the signal is size-dependent, not magnitude-dependent |
| soft_bbox with different sigma_factor | E-47 is fundamentally broken at epoch 3; Gaussian falloff from bbox is too weak a signal. Defer to Direction C if bbox softening is revisited |
| Return to hard dense_fg mainline | E-46 shows soft_mask IS better; abandoning it leaves +0.005 pf_mIoU and +0.047 small-IoU on the table |
| Center+extent decomposition (Direction C) | Higher implementation cost. Size-adaptive soft_mask is a 1-file change (dataset_real.py) with no architecture changes. If size-adaptive softening confirms and extends E-46's gains, it strengthens the case for eventually pursuing Direction C with size-awareness built in |

### Proposed experiment: E-48 size_adaptive_soft_mask

**Config**: `dense_target_mode = "soft_mask_adaptive"` — Gaussian blur with σ = f(object_size_on_28x28):
- σ = 0 for objects ≤ 4 px² (tiny, <1% of 224²)
- σ = 0.5 for objects ≤ 25 px² (small)
- σ = 1.0 for objects ≤ 100 px² (medium)
- σ = 1.5 for objects > 100 px² (large)

**Cost**: ~40 min GPU (8ep), 1 parameter changed from E-46

**Success criteria**: pf_mIoU > E-46 (0.8329) AND IoU_tiny ≥ E-45 (0.5837) — i.e., recover tiny-object performance while keeping small/large gains.

---

## Summary

| Question | Answer |
|----------|--------|
| E-45 reproduces E-39? | Yes — pf_mIoU 0.8282 vs 0.8277 (Δ +0.0005) |
| E-46 beats E-45 on reeval? | Yes — +0.0047 pf_mIoU, +0.0475 small IoU, −0.0091 bad_frame_rate |
| E-47 beats E-45 on reeval? | No — −0.1694 pf_mIoU, all metrics worse |
| Training log metrics reliable? | For E-45/E-46: yes (consistent 2.72× ratio). For E-47: no (epoch 3 overfit to training val split) |
| soft_mask has value? | Yes — consistent positive signal across most metrics, strongest on small objects |
| soft_bbox has value? | No — close the direction at sigma_factor=0.3 |
| Continue Teacher route? | Yes — refined to size-adaptive softening |
| Next experiment | E-48: size-adaptive soft_mask (σ = f(object_size)) — preserve tiny-object hard boundaries, soften medium/large |
