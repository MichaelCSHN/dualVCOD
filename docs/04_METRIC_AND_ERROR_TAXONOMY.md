# Metric & Error Taxonomy

This document defines "more accurate" for dualVCOD. All experiments are
evaluated against this taxonomy. No metric may be cherry-picked — the full
panel is reported for every experiment.

## 1. Core Metrics

Computed per-frame, then averaged across all frames in the unified reeval
set (np.random.RandomState(42) MoCA val split).

| Metric | Definition | Target |
|--------|-----------|--------|
| **pf_mIoU** | Per-frame mean IoU: mean of IoU(pred_i, gt_i) over all frames | Higher is better. Primary ranking metric. |
| **bad_frame_rate** | Fraction of frames with IoU < 0.5 | Lower is better. Deployment-critical: a "bad" frame is a miss. |
| **R@0.5** | Fraction of frames with IoU ≥ 0.5 (same as 1 − bad_frame_rate but reported independently for clarity) | Higher is better. |

### Interpretation Guide

- pf_mIoU measures average spatial overlap quality.
- bad_frame_rate measures catastrophic failure frequency.
- A model can have decent pf_mIoU but high bad_frame_rate if it fails
  completely on a subset of videos. Both must be tracked.

## 2. Size-Bin IoU

Frames are bucketed by ground-truth bbox area (fraction of image):

| Bin | Area Range | Typical Object |
|-----|-----------|----------------|
| **Tiny** | area < 0.01 | pygmy_seahorse, small crabs |
| **Small** | 0.01 ≤ area < 0.05 | Most camouflaged animals at distance |
| **Medium** | 0.05 ≤ area < 0.15 | Medium-distance or medium-size animals |
| **Large** | area ≥ 0.15 | Close-up or large animals (flatfish) |

**Target**: All bins should improve or stay flat. A gain in one bin at
the cost of another is a **regression** unless explicitly justified.

**Current pain points**:
- Tiny: pygmy_seahorse_0 mIoU ≈ 0.4 (best models), 0.008 (E-53a).
- Large: flatfish_2 mIoU ≈ 0.09 (best models) — under-coverage dominant.

## 3. Spatial Accuracy Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| **area_ratio_mean** | Mean(pred_area / gt_area) over all frames | Ideal = 1.0. < 1.0 = systematic under-prediction. > 1.0 = over-prediction. |
| **area_ratio_median** | Median(pred_area / gt_area) | Robust to outliers. Ideal = 1.0. |
| **center_error_mean** | Mean Euclidean distance between pred center and GT center (normalized coords) | Lower is better. Ideal = 0. |
| **center_error_median** | Median center distance | Robust to outliers. |

### Interpretation

- area_ratio < 1.0 + center_error large → pred is too small AND off-center
  (severe under-coverage).
- area_ratio > 1.0 + center_error small → pred is too large but centered
  (over-prediction, wasteful but less harmful).
- area_ratio ≈ 1.0 + center_error large → pred is right size but shifted
  (center localization failure).

## 4. Error Type Taxonomy

Derived from `eval/eval_video_bbox.py:compute_per_frame_metrics`.
Counts are aggregated over the full reeval set.

| Error Type | Criterion | What It Means |
|-----------|-----------|---------------|
| **pred_too_large** | pred_area / gt_area > 2.0 | Pred box covers > 2× the GT area. Over-prediction — model is "playing it safe" with large boxes. |
| **pred_too_small** | pred_area / gt_area < 0.5 | Pred box covers < 0.5× the GT area. Under-coverage — model fails to capture the full object extent. |
| **center_shift** | center_error > 0.1 (in normalized coords) | Pred center is more than 10% of image dimension away from GT center. Localization failure. |
| **scale_mismatch** | area_ratio < 0.5 OR area_ratio > 2.0 | Overlap with too_large/too_small but reported separately for cross-check. |

### Dominant Error Patterns

From Phase 2 error audits (E-41, E-43):

- **Tiny objects**: center_shift dominant — model roughly knows where the
  object is but can't localize precisely at 224×224 resolution.
- **Large objects**: pred_too_small dominant — model predicts conservative
  boxes that don't cover the full extent (flatfish_2).
- **Medium objects**: Mixed — scale_mismatch and center_shift roughly
  balanced.

## 5. Hard-Video Metrics

Per-video mIoU for three canonical hard cases. These serve as diagnostic
probes — a model that improves pf_mIoU but degrades on these videos is
suspicious.

| Video | Why Hard | Best Known mIoU |
|-------|----------|-----------------|
| **flatfish_2** | Genuine hard camouflage — object blends perfectly with sandy background. Large object under-coverage. | ~0.09 (E-45) |
| **pygmy_seahorse_0** | Tiny object (area ≈ 0.003). Resolution-limited. | ~0.42 (E-51) |
| **white_tailed_ptarmigan** | Seasonal camouflage — white bird on snow. | ~0.63 (E-53a, but that model regressed elsewhere) |

## 6. Efficiency Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| **FPS** | Frames per second on RTX 4090, including full forward pass (spatial + temporal + bbox head) | Higher is better. Reported at batch_size=1. |
| **Params** | Total trainable parameters | Lower is better. Deployment constraint. |

## 7. Composite Score (DRAFT — Not Yet Operational)

A weighted composite for go/no-go decisions:

```
composite = 0.30 × pf_mIoU
          + 0.15 × (1 − bad_frame_rate)
          + 0.10 × IoU_tiny
          + 0.10 × IoU_small
          + 0.10 × IoU_medium
          + 0.10 × IoU_large
          − 0.05 × |log(area_ratio_mean)|
          − 0.05 × center_error_mean
          − 0.05 × (hard_video_penalty)
```

This is a **draft**. It has NOT been calibrated against human judgment.
Do NOT use it as the sole decision criterion. The individual metrics
always take precedence in detailed analysis.

### Hard-Video Penalty (draft)

```
hard_video_penalty = max(0, 0.5 − pygmy_seahorse_0_mIoU)
                   + max(0, 0.3 − flatfish_2_mIoU)
```

Penalizes catastrophic failure on canonical hard cases.

### Calibration TODO

- Collect human pairwise preferences on 10-20 model outputs.
- Fit composite weights to match human rankings.
- Validate on held-out comparisons.

Until calibrated, use the full metric panel for decisions.

## 8. Reporting Checklist

Every experiment report MUST include all of:

- [ ] pf_mIoU
- [ ] bad_frame_rate
- [ ] R@0.5
- [ ] IoU_tiny, IoU_small, IoU_medium, IoU_large
- [ ] area_ratio_mean, area_ratio_median
- [ ] center_error_mean, center_error_median
- [ ] n_pred_too_large, n_pred_too_small, n_center_shift, n_scale_mismatch
- [ ] flatfish_2 mIoU, pygmy_seahorse_0 mIoU, white_tailed_ptarmigan mIoU
- [ ] FPS, Params (if model changed)
- [ ] Comparison against canonical baseline under identical reeval protocol
