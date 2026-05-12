# E-50: MobileNetV3-Large + dense_fg_aux Backbone Capacity Probe

**Date**: 2026-05-13
**Status**: Complete (training + unified reeval done)

---

## 1. Experiment Design

### Core Question
Does a 2.5× larger backbone in the same MobileNetV3 architecture family convert dense_fg_aux spatial supervision into better bbox-only inference?

### Variables

| Variable | E-45 (Baseline) | E-50 (Probe) |
|----------|:---------------:|:------------:|
| Backbone | MV3-Small (1.5M) | **MV3-Large (3.6M)** |
| Head | dense_fg_aux | dense_fg_aux |
| dense_target_mode | hard | hard |
| Resolution | 224 | 224 |
| T | 5 | 5 |
| Loss | DIoU | DIoU |
| LR | 0.001 | 0.001 |
| Epochs | 8 | 8 |
| Batch | 16 | 16 |

All other variables held constant (seed=42, weight_decay=1e-4, jitter=0.15, CosineAnnealing, resized_root).

### Bug Fix During Launch
MV3-Large's FPN stage slices in `backbone_registry.py` were incorrect: `(0, 4)` produced stride-4 (56×56) features instead of stride-8 (28×28). Corrected to `(0, 5)` which includes the stride-8 bottleneck block. This bug was hidden in prior B1 backbone readiness tests because they only verified bbox output shape (AdaptiveAvgPool2d accepts any spatial size), not intermediate FPN feature dimensions.

---

## 2. Training Results

| Epoch | Tr Loss | Tr mIoU | Val mIoU | Val pf_mIoU | Val R@0.5 |
|-------|:-------:|:-------:|:--------:|:-----------:|:---------:|
| 1 | 0.979 | 0.391 | 0.2624 | 0.2624 | 0.2064 |
| 2 | 0.706 | 0.537 | 0.2752 | 0.2752 | 0.2113 |
| 3 | 0.606 | 0.599 | 0.2839 | 0.2839 | 0.1958 |
| 4 | 0.533 | 0.650 | 0.3137 | 0.3137 | 0.2887 |
| 5 | 0.476 | 0.690 | 0.3030 | 0.3030 | 0.2247 |
| 6 | 0.423 | 0.731 | **0.3268** | **0.3268** | 0.2705 |
| 7 | 0.381 | 0.764 | 0.3189 | 0.3189 | 0.2670 |
| 8 | 0.350 | 0.790 | 0.3215 | 0.3215 | 0.2648 |

Training log best_val_mIoU: **0.3268** (epoch 6) vs E-45's 0.3046 (epoch 7).

Training time: 2062s (34.4 min) vs E-45's 1888s (31.5 min) — 9.2% slower per epoch.

### Epoch-level pf_mIoU Trend (Training Val Split)

| Metric | E-45 Epoch 7 | E-50 Epoch 6 | Δ |
|--------|:-----------:|:-----------:|:--:|
| val_mIoU | 0.3046 | 0.3268 | +0.0222 |
| val_pf_mIoU | 0.3046 | 0.3268 | +0.0222 |
| val_IoU_tiny | 0.0937 | 0.0870 | −0.0067 |
| val_IoU_small | 0.2199 | 0.2542 | +0.0343 |
| val_IoU_medium | 0.1533 | 0.1813 | +0.0280 |
| val_IoU_large | 0.4735 | 0.4769 | +0.0034 |
| val_n_pred_too_large | 2045 | 1921 | −124 |

E-50 shows stronger small/medium IoU on the training val split, but slightly worse tiny IoU.

---

## 3. Model Characteristics

| Metric | E-45 (MV3-Small) | E-50 (MV3-Large) |
|--------|:----------------:|:----------------:|
| Total params | 1,485,669 | 3,579,765 |
| Params ratio | 1.0× | 2.41× |
| GPU memory | 0.63 GiB | 1.33 GiB |
| Inference FPS | 78.7 | 43.7 |
| FPS ratio | 1.0× | 0.56× |
| Train time (8ep) | 1888s | 2062s |

---

## 4. Unified Reeval Results

| Metric | E-45 (MV3-Small) | E-50 (MV3-Large) | Δ |
|--------|:----------------:|:----------------:|:--:|
| ckpt epoch | 7 | 6 | |
| **pf_mIoU** | **0.8282** | **0.8069** | **−0.0212** |
| bad_frame_rate | 0.0500 | 0.0598 | +0.0097 |
| R@0.5 | 0.9500 | 0.9402 | −0.0097 |
| **Inference FPS** | 61.3 | 59.8 | −1.5 |

### Size-Stratified IoU

| Metric | E-45 | E-50 | Δ |
|--------|:----:|:----:|:--:|
| IoU tiny | **0.5837** | 0.4776 | **−0.1061** |
| IoU small | 0.7058 | **0.7273** | +0.0215 |
| IoU medium | **0.8382** | 0.8314 | −0.0068 |
| IoU large | **0.8597** | 0.8298 | −0.0298 |

### Area Ratio & Center Error

| Metric | E-45 | E-50 | Δ |
|--------|:----:|:----:|:--:|
| area_ratio_mean | **1.0351** | 1.0691 | +0.0340 |
| area_ratio_median | **1.0237** | 1.0557 | +0.0320 |
| center_error_mean | **0.0221** | 0.0261 | +0.0039 |

### Error Classification

| Error Type | E-45 | E-50 | Δ |
|------------|:----:|:----:|:--:|
| n_good | **5657** | 5599 | −58 |
| n_pred_too_large | 21 | **99** | **+78** |
| n_pred_too_small | 79 | 90 | +11 |
| n_scale_mismatch | 191 | **140** | −51 |
| n_center_shift | 7 | **27** | **+20** |

### Per-Video mIoU

| Video | E-45 | E-50 |
|-------|:----:|:----:|
| flatfish_2 | 0.0870 | 0.0745 |
| white_tailed_ptarmigan | 0.5136 | 0.5026 |
| pygmy_seahorse_0 | 0.3715 | 0.3531 |

### Training Log vs Reeval

| Trial | Train best_val_mIoU | Reeval pf_mIoU | Ratio |
|-------|:-------------------:|:--------------:|:-----:|
| E-45 | 0.3046 | 0.8282 | 2.72× |
| E-50 | 0.3268 | 0.8069 | **2.47×** |

The collapsed ratio (2.47× vs 2.72×) mirrors the failing Teacher route experiments — the training log overrated E-50 relative to unified reeval.

---

## 5. Analysis

### E-50 does NOT beat the baseline — despite higher training val mIoU

MV3-Large with dense_fg_aux is **worse** than MV3-Small:

- pf_mIoU −0.0212: Clear regression
- IoU_tiny −0.1061: Massive regression — worst among all non-E-47 experiments
- IoU_large −0.0298: Worse on large objects
- n_pred_too_large +78 (+371%): MV3-Large systematically over-predicts box sizes
- n_center_shift +20 (+286%): Worse center localization despite more params
- area_ratio_mean 1.0691 vs 1.0351: Boxes are larger than they should be

### Why does a larger backbone perform worse?

Three hypotheses:

1. **Overfitting at 8ep**: MV3-Large has 2.4× more params and may need more regularization or longer training. The training log (0.3268 vs 0.3046) suggested improvement, but unified reeval revealed the opposite — classic overfitting signal.

2. **Optimizer-LR mismatch**: lr=0.001 was tuned for MV3-Small. MV3-Large's different architecture may need a different LR. The CosineAnnealing schedule may exacerbate this.

3. **dense supervision doesn't scale with backbone size in the same family**: MobileNetV3 uses the same building blocks (inverted residuals with SE) at both scales. The dense_fg_aux head receives stride-8 FPN features that are fundamentally similar in representational capacity (both fused to 128ch). The extra backbone capacity may not translate to better stride-8 representations.

### Marginal positive signal

- IoU_small +0.0215: Small objects benefit slightly
- n_scale_mismatch −51: Fewer scale errors (but more over-prediction and center shift)

---

## 6. Decision: MV3-Large — Negative, Does Not Beat Baseline

| Criterion | Status |
|-----------|--------|
| pf_mIoU ≥ E-45? | **No** — −0.0212 |
| Tiny IoU ≥ E-45? | **No** — −0.1061 |
| Large IoU ≥ E-45? | **No** — −0.0298 |
| Over-prediction pattern | Severe — +371% n_pred_too_large |

**MV3-Large + dense_fg_aux at lr=0.001 does not beat MV3-Small.** The larger backbone in the same architecture family does not benefit from dense auxiliary supervision at 8ep. The training log misleadingly suggested improvement (0.3268 > 0.3046), but unified reeval revealed regression — confirming the importance of the reeval protocol.

---

## 6. Registry Bug Documentation

### Bug
MV3-Large (and all EfficientNet variants) had incorrect FPN stage slices in `backbone_registry.py`. The stage2 slice `(0, 4)` included blocks up to index 3, which only reaches stride 4 (56×56). The correct slice `(0, 5)` includes block index 4 (stride-2 bottleneck), reaching stride 8 (28×28).

### Impact
- Hidden for `current_direct_bbox` head: BBoxHead uses AdaptiveAvgPool2d which accepts any spatial size
- Fatal for `dense_fg_aux` head: Dense head output (56×56) mismatches mask targets (28×28)
- Affected all non-MV3-Small backbones: MV3-Large, EfficientNet-B0/B1/B2

### Fix
Updated all three affected registries with correct slices verified by channel/spatial probing at 224×224 input.

### Lesson
FPN stage slice validation must include spatial dimension checks, not just output shape verification. The `_probe_channels()` method should also report spatial dims.

---

*Report to be finalized after unified reeval.*
