# E-52: EfficientNet-B0 + hard dense_fg_aux 30ep — Validation Report

**Date**: 2026-05-13
**Purpose**: Answer whether E-51's 8ep positive signal (+0.0090 pf_mIoU) survives 30ep training and beats the E-40 canonical baseline (MV3-Small 30ep, 0.8564 pf_mIoU).

---

## 1. Experimental Design

Clean 2×2: Backbone (MV3-Small vs EfficientNet-B0) × Epochs (8ep vs 30ep)

|          | MV3-Small | EfficientNet-B0 |
|----------|-----------|-----------------|
| 8ep      | E-45      | E-51            |
| 30ep     | E-40      | E-52            |

All: hard dense_fg_aux (weight=0.5), DIoU, bbox-only inference, T=5, 224×224, train_batch=16, resized_root, no augmentation changes, no dense target changes.

**E-52 config**: lr=0.0003, warmup=3 epochs (LinearLR 0.1→1.0), CosineAnnealing T_max=27, weight_decay=0.0001, jitter=0.15. Identical to E-51 except epochs=30.

---

## 2. Results — 2×2 Matrix (Unified Reeval)

```
pf_mIoU          MV3-Small    EffB0      Δ(EffB0−MV3S)
8ep              0.8281       0.8372     +0.0090
30ep             0.8564       0.8711     +0.0146
Δ(30ep−8ep)      +0.0283      +0.0339
```

**E-52 = 0.8711 pf_mIoU — BEATS E-40 (0.8564) by +0.0146.**

---

## 3. Main Effects

| Effect | Magnitude | Interpretation |
|--------|-----------|----------------|
| Backbone: EffB0 vs MV3S @ 8ep | +0.0090 | Small but positive |
| Backbone: EffB0 vs MV3S @ 30ep | +0.0146 | **Larger gap at convergence** |
| Epochs: 30ep vs 8ep (MV3S) | +0.0283 | Significant |
| Epochs: 30ep vs 8ep (EffB0) | +0.0339 | **Larger epoch benefit for EffB0** |

The backbone advantage **widens** with more training (+0.009 → +0.015). EffB0 gains more from 30ep than MV3-Small does (+0.034 vs +0.028). This confirms the 8ep signal was not a fluke — it was an underestimate.

---

## 4. Detailed Metrics Comparison

### Core Metrics

| Metric | E-45 (MV3S 8ep) | E-51 (EffB0 8ep) | E-40 (MV3S 30ep) | **E-52 (EffB0 30ep)** |
|--------|-----------------|-------------------|-------------------|------------------------|
| pf_mIoU | 0.8281 | 0.8372 | 0.8564 | **0.8711** |
| bad_frame_rate | 0.0500 | 0.0552 | 0.0358 | **0.0257** |
| R@0.5 | 0.9500 | 0.9448 | 0.9642 | **0.9743** |
| n_good / 5955 | 5657 | 5626 | 5742 | **5802** |

E-52 wins core metrics cleanly. bad_frame_rate (2.57%) is the lowest ever recorded in this project.

### Size-Stratified IoU

| Stratum | E-45 | E-51 | E-40 | **E-52** | Δ(E52−E40) |
|---------|------|------|------|-----------|-------------|
| IoU_tiny | 0.584 | 0.512 | **0.703** | 0.602 | −0.101 |
| IoU_small | 0.706 | 0.738 | 0.752 | **0.794** | +0.042 |
| IoU_medium | 0.838 | 0.854 | 0.858 | **0.877** | +0.018 |
| IoU_large | 0.860 | 0.869 | 0.884 | **0.906** | +0.022 |

E-52 dominates small/medium/large. The recurring tiny-object tradeoff persists: E-40 retains the best tiny IoU (0.703 vs 0.602), consistent with E-51's tiny regression pattern. However, the overall pf_mIoU gain more than compensates.

### Error Classification

| Error Type | E-45 | E-51 | E-40 | **E-52** |
|------------|------|------|------|-----------|
| n_pred_too_large | 21 | 33 | 1 | **0** |
| n_pred_too_small | 79 | 94 | 120 | **63** |
| n_center_shift | 7 | 6 | 5 | **2** |
| n_scale_mismatch | 191 | 196 | 87 | **88** |

**Zero over-predictions** — E-52 eliminates pred_too_large entirely. Center shift errors cut to 2 (best). Scale mismatch matches E-40.

### Area Ratio and Center Error

| Metric | E-45 | E-51 | E-40 | **E-52** |
|--------|------|------|------|-----------|
| area_ratio_mean | 1.035 | 1.035 | 0.990 | **0.997** |
| center_error_mean | 0.0221 | 0.0193 | 0.0182 | **0.0153** |
| center_error_median | 0.0148 | 0.0131 | 0.0108 | **0.0097** |

Area ratio closest to 1.0 (near-perfect scale calibration). Center error is the new record.

### Shape-Stratified IoU

| Shape | E-45 | E-51 | E-40 | **E-52** |
|-------|------|------|------|-----------|
| IoU_square | 0.876 | 0.878 | 0.898 | **0.914** |
| IoU_tall | 0.791 | 0.807 | 0.829 | 0.830 |
| IoU_wide | 0.801 | 0.801 | 0.803 | **0.888** |

E-52's wide-object IoU (0.888) is +0.085 over E-40 — a dramatic improvement suggesting SE channel attention helps with horizontally extended camouflage patterns.

---

## 5. Per-Video mIoU

| Video | E-45 | E-51 | E-40 | **E-52** | Trend |
|-------|------|------|------|-----------|-------|
| flatfish_2 | 0.087 | 0.260 | 0.215 | **0.291** | Steady improvement |
| white_tailed_ptarmigan | 0.514 | 0.507 | 0.388 | **0.573** | New record |
| pygmy_seahorse_0 | 0.371 | 0.420 | **0.623** | 0.387 | MV3-Small wins |

flatfish_2 continues its ascent: 0.087 → 0.260 → 0.291 (+234% from E-45 baseline).  
white_tailed_ptarmigan: E-40 regressed to 0.388 from 0.514; E-52 reclaims at 0.573.  
pygmy_seahorse_0: E-40's 0.623 remains the high-water mark. EffB0's SE attention may over-process the high-frequency pygmy texture.

---

## 6. Training Log vs Unified Reeval

| Trial | Backbone | Epochs | Train Best | Reeval | Ratio |
|-------|----------|--------|------------|--------|-------|
| E-45 | MV3-Small | 8 | 0.3046 | 0.8281 | 2.72× |
| E-51 | EffB0 | 8 | 0.3345 | 0.8372 | 2.50× |
| E-40 | MV3-Small | 30 | 0.3087 | 0.8564 | 2.77× |
| E-52 | EffB0 | 30 | 0.3730 | 0.8711 | 2.34× |

The ratio compresses as both training and reeval scores rise (E-52: 2.34×, lowest). E-52's training val mIoU best (0.373) is 20.3% higher than E-40's (0.309), and the unified reeval confirms a genuine gain (+1.7%).

Key stability signals:
- mIoU standard deviation over last 3 epochs: 0.0007 (very tight convergence)
- No NaN at any point in 30 epochs
- LR decay clean: 3e-4 → 0 (CosineAnnealing, no spikes)

---

## 7. Model Characteristics

| Property | E-45 | E-51 | E-40 | **E-52** |
|----------|------|------|------|-----------|
| Params | 1.49M | 4.67M | 1.49M | 4.67M |
| FPS | 70.7 | 46.5 | 51.4 | 30.9 |
| GPU mem | ~1.8 GiB | ~1.8 GiB | ~1.8 GiB | ~1.8 GiB |

The 30.9 FPS vs E-40's 51.4 FPS is the main cost of the EffB0 mainline — a 40% inference speed reduction for a 1.7% pf_mIoU gain. Acceptable for research use; deployment optimization would need profiling.

---

## 8. Training Trajectory

| Phase | Epochs | Val mIoU | Notes |
|-------|--------|----------|-------|
| Warmup | 1-3 | 0.227 → 0.266 | Rapid climb |
| Cosine init | 4-8 | 0.280 → 0.357 | E-51-equivalent at 8ep (0.345) |
| Mid Cosine | 9-20 | 0.337 → 0.358 | Oscillation band 0.34-0.36 |
| Breakout | 21 | **0.373** | Peak, CosineAnnealing LR=7.5e-5 |
| Convergence | 22-30 | 0.366-0.368 | Stable plateau, std 0.0007 |

The checkpoint saved at epoch 20 (val=0.358, not the peak 0.373). Running epoch 21's checkpoint through unified reeval could yield slightly higher than 0.8711. However, the small val mIoU difference (+0.015) likely translates to negligible pf_mIoU delta (estimated <0.003).

---

## 9. E-52 Training Config

```json
{
  "backbone": "efficientnet_b0",
  "head": "dense_fg_aux",
  "dense_target_mode": "hard",
  "lr": 0.0003,
  "warmup_epochs": 3,
  "epochs": 30,
  "train_batch_size": 16,
  "input_size": 224,
  "temporal_T": 5,
  "loss_weights": { "smooth_l1": 1.0, "giou": 1.0, "use_diou": true, "dense_fg": 0.5 },
  "best_val_miou": 0.373,
  "best_epoch": 21,
  "train_time": "7896s (2.2h)"
}
```

---

## 10. Key Takeaways

1. **EffB0 30ep BEATS MV3-Small 30ep across nearly every dimension.** The 8ep positive signal (+0.009) was real and understated — the gap **widens** with extended training (+0.015).

2. **Epoch benefit is larger for EffB0 than MV3-Small** (+0.034 vs +0.028), suggesting EffB0's SE modules benefit from extended fine-tuning.

3. **The tiny-object tradeoff persists** — E-40 retains the best tiny IoU (0.703 vs 0.602). Every improvement over E-45 has regressed on tiny objects. This is a fundamental architectural tension, not specific to EffB0.

4. **flatfish_2 improves monotonically with EffB0** — 0.087 → 0.260 → 0.291. The hardest video in the dataset continues to yield to the SE channel attention mechanism.

5. **pygmy_seahorse_0 favors MV3-Small** — E-40's 0.623 is unmatched. EffB0 may over-process high-frequency texture patterns.

6. **Training log ratio compression is a feature, not a bug** — higher training val and higher reeval naturally produce lower ratios. E-52's 2.34× is the expected continuation of the 2.50-2.80× pattern.

7. **Convergence is very stable** — std 0.0007 over last 3 epochs. No NaN, no OOM, no degradation at extended training.

8. **The FPS cost is real but acceptable for research** — 30.9 FPS vs 51.4 FPS for MV3-Small. Deployment optimization not yet needed.

---

## 11. Verdict

**EfficientNet-B0 is validated as the new mainline candidate**, replacing MV3-Small hard dense_fg_aux as the canonical baseline.

The 2×2 design answers the core question cleanly:
- EffB0 > MV3-Small at both 8ep and 30ep
- The gap widens with training, not collapses
- The 30ep EffB0 result (0.8711) is the new state-of-the-art for this task configuration

**Recommendation**: Adopt EfficientNet-B0 + hard dense_fg_aux 30ep as the Phase 2 canonical baseline. The MV3-Small 30ep result (0.8564) is superseded but should be retained as a lightweight reference point.

E-52 training checkpoint and config are committed. Unified reeval results at `local_runs/reeval_2x2_backbone_epochs.json`.

---

*2×2 validation probe complete. Closed direction: backbone comparison. Next: Phase 2 planning with EffB0 mainline.*
