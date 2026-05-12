# E-51: EfficientNet-B0 + dense_fg_aux Backbone Capacity Probe

**Date**: 2026-05-13
**Status**: Complete

---

## 1. Experiment Design

### Core Question
Does a different CNN architecture family (EfficientNet with SE modules and compound scaling) benefit more from dense_fg_aux spatial supervision than MobileNetV3?

### Variables

| Variable | E-45 (Baseline) | E-51 (Probe) |
|----------|:---------------:|:------------:|
| Backbone | MV3-Small (1.5M) | **EfficientNet-B0 (4.7M)** |
| Head | dense_fg_aux | dense_fg_aux |
| dense_target_mode | hard | hard |
| Resolution | 224 | 224 |
| T | 5 | 5 |
| Loss | DIoU | DIoU |
| **LR** | **0.001** | **0.0003** |
| **Warmup** | **0** | **3 epochs** |
| Epochs | 8 | 8 |
| Batch | 16 | 16 |

LR reduced to 0.0003 with 3-epoch linear warmup to avoid the NaN instability observed with EfficientNet-B0 at lr≥0.001 (E-24, smoke_b1). All other variables held constant.

### Prior NaN History
- smoke_b1_effb0: EffB0, 512px, lr=1e-3 → NaN at epochs 3-5
- E-24: EffB0, 224px, lr=1e-3, direct_bbox → NaN at epochs 3-5, best only 0.1708 (ep1)
- E-03: EffB0, 320px, lr=1e-4, warmup=5 → Stable at 1ep (0.2477)

### Bug Fix During Launch
Same FPN stage slice bug as E-50 — EfficientNet-B0's `(0, 3)` slice stopped at stride 4. Corrected to `(0, 4)` for stride 8.

---

## 2. Training Results

| Epoch | Tr Loss | Tr mIoU | Val mIoU | Val pf_mIoU | Val R@0.5 |
|-------|:-------:|:-------:|:--------:|:-----------:|:---------:|
| 1 | 1.210 | 0.367 | 0.2230 | 0.2230 | 0.1172 |
| 2 | 0.950 | 0.450 | 0.2519 | 0.2519 | 0.1460 |
| 3 | 0.778 | 0.511 | 0.2673 | 0.2673 | 0.1609 |
| 4 | 0.675 | 0.559 | 0.3021 | 0.3021 | 0.1990 |
| 5 | 0.581 | 0.619 | **0.3297** | **0.3297** | 0.3072 |
| 6 | 0.505 | 0.671 | 0.3271 | 0.3271 | 0.2828 |
| 7 | 0.445 | 0.717 | 0.3274 | 0.3274 | 0.2724 |
| 8 | 0.403 | 0.751 | **0.3345** | **0.3345** | 0.2813 |

Training log best_val_mIoU: **0.3345** (epoch 8) — highest ever training log score at 8ep (E-45: 0.3046, E-50: 0.3268).

**No NaN. lr=0.0003 + warmup=3 confirmed stable for EfficientNet-B0 + dense_fg_aux.**

Training time: 2056s (34.3 min). Slow early epochs due to warmup LR, faster later.

---

## 3. Model Characteristics

| Metric | E-45 (MV3-Small) | E-51 (EffB0) |
|--------|:----------------:|:------------:|
| Total params | 1,485,669 | 4,667,585 |
| Params ratio | 1.0× | 3.14× |
| GPU memory | 0.63 GiB | 1.83 GiB |
| Inference FPS | 61.3 | 39.9 |
| FPS ratio | 1.0× | 0.65× |
| Train time (8ep) | 1888s | 2056s |

---

## 4. Unified Reeval Results

| Metric | E-45 (MV3-Small) | E-51 (EffB0) | Δ |
|--------|:----------------:|:------------:|:--:|
| ckpt epoch | 7 | 8 | |
| **pf_mIoU** | **0.8282** | **0.8372** | **+0.0090** |
| bad_frame_rate | 0.0500 | 0.0552 | +0.0052 |
| R@0.5 | 0.9500 | 0.9448 | −0.0052 |

### Size-Stratified IoU

| Metric | E-45 | E-51 | Δ |
|--------|:----:|:----:|:--:|
| IoU tiny | **0.5837** | 0.5119 | −0.0718 |
| IoU small | 0.7058 | **0.7378** | +0.0320 |
| IoU medium | 0.8382 | **0.8539** | +0.0158 |
| IoU large | 0.8597 | **0.8694** | +0.0098 |

### Area Ratio & Center Error

| Metric | E-45 | E-51 | Δ |
|--------|:----:|:----:|:--:|
| area_ratio_mean | 1.0351 | **1.0345** | −0.0006 |
| area_ratio_median | 1.0237 | 1.0330 | +0.0093 |
| center_error_mean | 0.0221 | **0.0193** | −0.0028 |

### Error Classification

| Error Type | E-45 | E-51 | Δ |
|------------|:----:|:----:|:--:|
| n_good | 5657 | **5626** | −31 |
| n_pred_too_large | 21 | 33 | +12 |
| n_pred_too_small | 79 | 94 | +15 |
| n_scale_mismatch | 191 | 196 | +5 |
| n_center_shift | 7 | **6** | −1 |

### Per-Video mIoU

| Video | E-45 | E-51 | Δ |
|-------|:----:|:----:|:--:|
| flatfish_2 | 0.0870 | **0.2603** | **+0.1733** |
| white_tailed_ptarmigan | 0.5136 | 0.5075 | −0.0061 |
| pygmy_seahorse_0 | 0.3715 | **0.4197** | +0.0482 |

### Training Log vs Reeval

| Trial | Train best_val_mIoU | Reeval pf_mIoU | Ratio |
|-------|:-------------------:|:--------------:|:-----:|
| E-45 | 0.3046 | 0.8282 | 2.72× |
| E-51 | 0.3345 | 0.8372 | **2.50×** |

The ratio (2.50× vs 2.72×) shows metric collapse — the training log overrated E-51's improvement. Despite the highest-ever training log score (0.3345), the unified reeval gain is only +0.0090. This underscores the critical importance of the unified reeval protocol.

---

## 5. Analysis

### E-51 BEATS the canonical baseline — with caveats

EfficientNet-B0 + dense_fg_aux achieves pf_mIoU 0.8372, +0.0090 above the MV3-Small baseline. This is the **first verified improvement over E-45** in the backbone/Teacher exploration series.

**Strengths:**

1. **flatfish_2 breakthrough: 0.2603 vs 0.0870 (+0.1733, 3× improvement)**. The hardest video in the dataset benefits dramatically from EfficientNet features. This suggests EfficientNet's SE (Squeeze-and-Excitation) modules help with the extreme texture-matching camouflage that characterizes flatfish_2.

2. **Universal IoU improvement on non-tiny objects**: Small (+0.0320), medium (+0.0158), large (+0.0098) all improve. Only tiny objects regress.

3. **Better center localization**: center_error 0.0193 vs 0.0221 (−0.0028). EfficientNet learns more precise object centers.

4. **pygmy_seahorse_0 +0.0482**: Combined with E-49's +0.0682, this video consistently benefits from richer features.

**Weaknesses:**

1. **Tiny IoU −0.0718**: The recurring tradeoff. Every improvement over E-45 (E-46 soft_mask, E-51 EffB0) regresses on tiny objects. This suggests a fundamental tension: richer features help most objects but may over-smooth or lose the finest spatial details needed for tiny camouflage targets.

2. **3× params, 65% FPS**: EfficientNet-B0 trades inference speed for accuracy. At 39.9 FPS on RTX 4090, it may still be viable for edge deployment depending on target framerate.

3. **Conservative LR may have limited convergence**: lr=0.0003 was chosen for stability, not optimal learning. The training log shows continued improvement through epoch 8 (no clear plateau), suggesting longer training or higher LR could yield further gains.

### Why EfficientNet-B0 outperforms MV3-Small while MV3-Large underperforms

| Factor | MV3-Large | EfficientNet-B0 |
|--------|:---------:|:---------------:|
| Architecture family | Same as Small (inverted residuals) | Different (MBConv + SE modules) |
| SE modules | Present | Present (stronger) |
| Channel scaling | 24→40→112→960 | 40→112→1280 (wider at all stages) |
| FPN lat channels | 40→128 (stage2) | 40→128 (stage2) |
| Result | −0.0212 | +0.0090 |

The SE modules in EfficientNet-B0 provide channel-wise attention that may be particularly valuable for camouflage — where object and background share texture statistics, making channel re-weighting more informative than spatial expansion within the same architecture family.

### The training-log-to-reeval collapse is diagnostic

E-51 showed the highest-ever training log val mIoU (0.3345), but only +0.0090 in unified reeval. The ratio (2.50×) is closer to E-49's 2.60× than E-45's 2.72×. This pattern suggests the training val split systematically overrates EfficientNet features — possibly because the training val videos share more texture statistics with training videos than the unified reeval videos do.

---

## 6. Decision: EfficientNet-B0 — Positive Signal, Architecture Matters

| Criterion | Status |
|-----------|--------|
| pf_mIoU ≥ E-45? | **Yes** — +0.0090 |
| Tiny IoU ≥ E-45? | **No** — −0.0718 |
| Small IoU ≥ E-45? | **Yes** — +0.0320 |
| flatfish_2 improved? | **Massively** — +0.1733 (3×) |
| Better center error? | **Yes** — −0.0028 |
| Stable training? | **Yes** — No NaN with lr=0.0003 + warmup=3 |

**EfficientNet-B0 + dense_fg_aux is the first backbone variant to beat the MV3-Small canonical baseline.** The improvement is modest but real (+0.0090 pf_mIoU), with dramatic gains on the hardest video (+0.1733 on flatfish_2). The architecture family matters more than raw parameter count — MV3-Large (3.6M) underperforms while EfficientNet-B0 (4.7M) outperforms.

### Should this become the new mainline?

**Not yet.** Three concerns:

1. **Tiny IoU regression (−0.0718)**: Same pattern as E-46. Any mainline must not regress on tiny objects, which are the core camouflage challenge.

2. **3× param cost for small gain**: +0.0090 pf_mIoU at 3× params may not justify the deployment cost. The 30ep baseline (E-40, +0.0283) is a cheaper way to get more gain.

3. **LR wasn't optimized**: lr=0.0003 was chosen for stability. The true potential may be higher.

### Recommended follow-up (deferred)

- EfficientNet-B0 + dense_fg_aux at 30ep with lr tuning
- EfficientNet-B1/B2 probes to explore EfficientNet scaling curve under dense supervision
- Investigate whether SE modules specifically drive the flatfish_2 improvement

---

## 7. Complete Backbone Probe Scoreboard

| Experiment | Backbone | pf_mIoU | Δ vs E-45 | Verdict |
|------------|----------|:-------:|:---------:|---------|
| E-45 | MV3-Small (1.5M) | 0.8282 | — | **Canonical baseline** |
| E-50 | MV3-Large (3.6M) | 0.8069 | −0.0212 | Negative — same family, worse |
| E-51 | EfficientNet-B0 (4.7M) | 0.8372 | +0.0090 | **Positive — different family, better** |

Key insight: **Architecture family > parameter count.** Same-family scaling (MV3-Small→Large) underperforms; cross-family scaling (MV3→EfficientNet) outperforms. The SE modules and channel attention in EfficientNet may be particularly suited to camouflage perception.
