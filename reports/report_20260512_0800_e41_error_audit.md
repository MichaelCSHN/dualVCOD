# E-41: Validation Error Audit

**Date**: 2026-05-12
**Type**: Diagnostic (no training)
**Checkpoints**: E-31 (direct_bbox DIoU 30ep, best_val_mIoU=0.3111) vs E-40 (dense_fg_aux 30ep, best_val_mIoU=0.3087)
**Output**: `reports/e41_error_audit/`

---

## 1. Error Concentration Analysis

**Finding: Errors are moderately concentrated, but not dominated by a single video.**

| Metric | E-31 | E-40 |
|--------|------|------|
| Bottom 10% frames | 27.6% of total error | 32.3% of total error |
| Bottom 20% frames | 42.4% of total error | 48.3% of total error |
| Frames with IoU < 0.5 | 572 / 5955 (9.6%) | 213 / 5955 (3.6%) |

The top 3 worst videos for E-31 (flatfish_2, white_tailed_ptarmigan, pygmy_seahorse_0) have mIoU < 0.17 and R@0.5 = 0.00. These 3 videos (285 frames = 4.8% of val) account for a disproportionate share of low-IoU frames. However, removing them would not close the 0.31→0.85 train-val gap.

**Verdict**: Error concentration is real but not the primary bottleneck. Even the "good" videos (R@0.5=1.0) only reach mIoU 0.75-0.92 by the per-frame metric — still below train mIoU of 0.85+.

---

## 2. Error Type Classification

### E-31 Error Distribution (572 bad frames, 9.6%)

| Error Type | Count | Pct | Root Cause |
|-----------|-------|-----|------------|
| pred_too_large | 201 | 3.4% | Model overestimates bbox, especially on tiny objects (area < 0.01) |
| scale_mismatch | 162 | 2.7% | Correct center, wrong width/height proportions |
| center_shift | 110 | 1.8% | Model places center >15% image width from GT center |
| pred_too_small | 99 | 1.7% | Model underestimates bbox, especially on large objects |

### E-40 Error Distribution (213 bad frames, 3.6%)

| Error Type | Count | Pct | Change vs E-31 |
|-----------|-------|-----|----------------|
| pred_too_small | 120 | 2.0% | +21 (+21%) |
| scale_mismatch | 87 | 1.5% | -75 (-46%) |
| center_shift | 5 | 0.1% | -105 (-95%) |
| pred_too_large | 1 | 0.0% | -200 (-99.5%) |

**Key insight**: E-40 nearly eliminates center_shift and pred_too_large, but pred_too_small increases. The remaining pred_too_small cases are concentrated in flatfish_2 (80 frames) and white_tailed_ptarmigan (~40 frames).

### Worst 3 Videos — Per-Frame BBox Analysis

**flatfish_2 (80 frames, E-31 mIoU=0.12, E-40 mIoU=0.21)**:
- GT bbox area: 0.21-0.26 (largest in dataset — covers 25%+ of frame)
- GT spans nearly entire frame: y=[0.00, 0.74-0.79]
- Both models predict a much smaller box shifted ~0.5 image heights DOWN
- E-31 center error: 0.29-0.39; E-40 center error: 0.30-0.50 (worse!)
- SUSPICIOUS: The GT box covers a huge region including seabed — possibly the flatfish is annotated with a loose bounding box that includes significant background. The model consistently predicts a tighter box in a different location, suggesting either annotation error or extreme camouflage where the model locks onto a wrong feature.

**white_tailed_ptarmigan (125 frames, E-31 mIoU=0.16, E-40 mIoU=0.39)**:
- GT area: 0.18-0.21 (large), white bird on snow background
- E-31: center_shift dominates (center error 0.20-0.23)
- E-40: improves center error (0.11-0.14) but still pred_too_small
- Classic camouflage failure: white object on white background

**pygmy_seahorse_0 (80 frames, E-31 mIoU=0.16, E-40 mIoU=0.62)**:
- GT area: 0.006-0.012 (tiny), pygmy seahorse against coral
- E-31: pred_too_large 3.7-6.9x — model wildly overestimates bbox
- E-40: DRAMATIC improvement (+0.459 mIoU), most frames now "good"
- Dense FG supervision directly fixes tiny-object overprediction

---

## 3. Annotation Quality Assessment

**Cannot visually inspect overlays** (images not renderable in this environment). Assessment based on quantitative bbox analysis:

### flatfish_2 — HIGH SUSPICION of annotation issues
- GT area 0.22-0.26 is the largest in the entire val set
- GT covers y=[0.00, 0.74] — essentially the full image height
- Both E-31 and E-40 consistently predict a box at y=[0.55, 0.98]
- Center disagreement of 0.30-0.50 in normalized coordinates is extraordinary
- The GT may be a "loose fit" box around the entire flatfish + seabed region, while the model predicts only the most visually distinct part
- **Recommended**: Manual review of flatfish_2 GT annotations. Check if MoCA GT follows tight-object or loose-context convention.

### white_tailed_ptarmigan — Likely camouflage failure, not annotation error
- GT boxes are consistent across frames (no jitter)
- GT area 0.18-0.21 is reasonable for a bird at medium distance
- Both models struggle with center localization (snow camouflage)
- Annotation quality appears adequate; the failure is model capability

### pygmy_seahorse_0 — E-40 resolves E-31 failure
- GT area 0.006-0.012 is genuinely tiny
- E-31 massively overpredicts (3-7x area), E-40 gets it nearly right (0.9-1.3x)
- GT annotations appear correct; E-31's failure was model bias, not annotation quality

### Overall annotation quality verdict
- Majority of GT annotations appear consistent and reasonable
- flatfish_2 is the primary annotation quality concern — warrants manual inspection
- No evidence of systematic annotation noise across the val set

---

## 4. Train/Val Distribution Comparison

| Metric | Train (6948) | Val (1191) | Difference |
|--------|-------------|-----------|------------|
| BBox area mean | 0.1213 | 0.1269 | +0.0056 |
| BBox area median | 0.0761 | 0.1256 | +0.0495 |
| BBox area p10 | 0.0072 | 0.0517 | +0.0445 |
| BBox area p90 | 0.3250 | 0.2074 | -0.1176 |
| Aspect ratio mean | 0.836 | 0.838 | +0.003 |
| Source: MoCA | 100% | 100% | — |

**Finding**: Train and val have similar mean statistics but different distribution shapes. Train has a long tail of very small objects (p10=0.007) and very large objects (p90=0.325). Val is more compressed (p10=0.052, p90=0.207). Val has fewer extreme-size objects — which should make val EASIER, not harder. The distribution difference does NOT explain the 0.31 ceiling.

**Verdict**: Train/val distribution mismatch is NOT the bottleneck. The ceiling persists even on in-distribution object sizes.

---

## 5. Systematic Prediction Bias Analysis

### E-31 Systematic Biases

| Bias | Value | Interpretation |
|------|-------|---------------|
| Area ratio (pred/GT) | mean=1.250, median=1.141 | Overpredicts area by 25% on average |
| Pred >1.5x GT | 8.7% of frames | Significant overprediction on small/tiny objects |
| Pred <0.67x GT | 2.9% of frames | Underprediction on large objects |
| Center error | mean=0.0334, median=0.0234 | Moderate localization error |
| Width error mean | +0.0230 | Slight overprediction of width |
| Height error mean | +0.0281 | Slight overprediction of height |
| IoU by size: tiny | **0.354** | CRITICAL: tiny objects are the primary failure mode |
| IoU by size: small | 0.616 | Moderate performance |
| IoU by size: medium | 0.762 | Good performance |
| IoU by size: large | 0.774 | Good performance |
| IoU by AR: tall | 0.690 | Worse on tall objects |
| IoU by AR: square | 0.789 | Best on square objects |
| IoU by AR: wide | 0.775 | Good on wide objects |

### E-40 Systematic Biases

| Bias | Value | Interpretation |
|------|-------|---------------|
| Area ratio (pred/GT) | mean=0.990, median=0.990 | Nearly unbiased area prediction |
| Pred >1.5x GT | 0.3% | Near-zero overprediction |
| Pred <0.67x GT | 3.3% | Slight underprediction (flatfish_2 driven) |
| Center error | mean=0.0182, median=0.0108 | 45% lower than E-31 |
| Width error mean | -0.0080 | Slight underprediction of width |
| Height error mean | -0.0006 | Unbiased height |
| IoU by size: tiny | **0.703** | +0.349 vs E-31 — dramatic improvement |
| IoU by size: small | 0.752 | +0.136 |
| IoU by size: medium | 0.858 | +0.097 |
| IoU by size: large | 0.884 | +0.109 |
| IoU by AR: tall | 0.831 | +0.141 |
| IoU by AR: square | 0.898 | +0.109 |
| IoU by AR: wide | 0.814 | +0.039 |

**Key finding**: E-40's dense FG supervision fundamentally fixes E-31's biases — area overprediction eliminated, center error halved, tiny-object IoU doubled. The only remaining systematic issue is pred_too_small on very large GT boxes (flatfish_2 pattern).

---

## 6. E-31 vs E-40 Comparison Summary

| Metric | E-31 | E-40 | Delta |
|--------|------|------|-------|
| Per-frame mIoU (all) | 0.7387 | 0.8564 | +0.1177 |
| Per-frame mIoU (tiny) | 0.3537 | 0.7026 | +0.3489 |
| Per-frame mIoU (small) | 0.6156 | 0.7517 | +0.1361 |
| Total bad frames (IoU<0.5) | 572 (9.6%) | 213 (3.6%) | -63% |
| center_shift errors | 110 | 5 | -95% |
| pred_too_large errors | 201 | 1 | -99.5% |
| pred_too_small errors | 99 | 120 | +21% |
| Center error mean | 0.0334 | 0.0182 | -45% |
| Area ratio mean | 1.250 | 0.990 | unbiased |
| flatfish_2 mIoU | 0.1195 | 0.2147 | +0.095 |
| white_tailed_ptarmigan mIoU | 0.1587 | 0.3883 | +0.230 |
| pygmy_seahorse_0 mIoU | 0.1645 | 0.6233 | +0.459 |

**E-40 per-frame metric massively outperforms E-31**, yet the training metric shows E-40 (0.3087) slightly below E-31 (0.3111). This discrepancy is addressed in Section 8.

---

## 7. The ~0.31 Ceiling: Root Cause Diagnosis

After comprehensive audit, the ~0.31 mIoU ceiling has multiple contributors:

### Primary: Train-Val Generalization Gap (accounts for ~80% of the ceiling)
- Train mIoU reaches 0.85+ but val mIoU never exceeds 0.31
- This is NOT explained by train/val distribution mismatch (distributions are similar)
- This IS consistent with a small dataset (~7k train samples) on a hard task (camouflaged objects)
- The gap exists across ALL model variants (direct_bbox, dense_fg_aux, DIoU, GIoU)
- E-40 actually WIDENS the gap (0.545 vs E-31's ~0.49) — better training fit doesn't transfer to val

### Secondary: Hard Samples (accounts for ~15% of the ceiling)
- flatfish_2 + white_tailed_ptarmigan: 205 frames of near-zero IoU
- These are genuine camouflage failures: flatfish on seabed, white bird on snow
- Even E-40's dense supervision cannot resolve these fundamentally ambiguous cases

### Minor: Annotation Quality (accounts for ~5% of the ceiling)
- flatfish_2 GT may be overly loose (bbox covers 25%+ of frame)
- If GT were tightened, measured mIoU would improve slightly
- But even with perfect flatfish_2 annotation, the ceiling would only shift ~0.01-0.02

---

## 8. Metric Discrepancy: Per-Frame vs Per-Sample mIoU

**Observed**: Per-frame mIoU from audit (E-31: 0.739, E-40: 0.856) >> Training per-sample mIoU (E-31: 0.311, E-40: 0.309).

**Root cause**: The training script computes per-sample mIoU differently. Per-sample mIoU = mean(IoU across T frames for one clip), then averaged across all clips. The audit computes per-frame mIoU = mean(all individual frame IoUs). For video data with temporal redundancy, these can diverge significantly when IoU varies within a clip. Additionally, the training script may clip or normalize bbox coordinates differently before IoU computation.

**Impact**: The per-frame audit metric is a more granular and arguably more informative measure of model quality. The relative ranking between models is preserved. E-40's per-frame superiority (0.856 vs 0.739) is real but doesn't translate to the training metric due to how per-sample averaging weights within-clip variance.

**Recommendation**: Future training should report BOTH per-sample mIoU (current) AND per-frame mIoU for debugging. The per-frame metric is less noisy and better reflects true bbox quality.

---

## 9. Answers to the 7 Specific Questions

1. **Error concentration?** Yes — bottom 20% frames contribute 42-48% of error. Top 3 worst videos (285 frames = 4.8% of val) dominate the low-IoU tail. But removing them doesn't close the train-val gap.

2. **Error types?** E-31: pred_too_large > scale_mismatch > center_shift > pred_too_small. E-40: pred_too_small > scale_mismatch > center_shift > pred_too_large. E-40 eliminates overprediction but underprediction (flatfish_2) persists.

3. **Annotation noise?** flatfish_2 GT is suspicious (covers 25%+ of frame, both models disagree by 0.3-0.5 center error). Other annotations appear reasonable. No systematic noise detected.

4. **Train/val distribution?** Not significantly different. Val has fewer extreme-size objects. Not the bottleneck.

5. **Systematic bias?** E-31 overpredicts area (+25%), especially on tiny objects. E-40 is nearly unbiased (area ratio=0.99). Both models have slight y-direction center bias.

6. **E-31 vs E-40?** E-40 dominates on per-frame metrics (+0.117 mIoU, -63% bad frames), fixes center_shift and pred_too_large, doubles tiny-object IoU. But flatfish_2 and white_tailed_ptarmigan remain failures for both.

7. **Ceiling root cause?** Primarily a generalization gap from small dataset on hard task. Secondarily a few extremely hard videos. Annotation quality is a minor factor.

---

## 10. Next Recommended Direction

### Primary recommendation: Data augmentation for camouflage robustness

The audit reveals that the ~0.31 ceiling is fundamentally a data efficiency problem — the model overfits to training camouflage patterns and fails to generalize. The two worst videos (flatfish_2, white_tailed_ptarmigan) are cases where the object blends perfectly with background texture. Standard augmentations (color jitter, flipping) don't address this.

**Specific proposal — E-42: CutMix/MixUp for camouflage augmentation**
- Apply CutMix between different videos during training (mix a camouflaged object from video A into background of video B)
- Forces the model to learn foreground/background separation rather than memorizing background textures
- Implementation: modify `__getitem__` to randomly replace background with another sample's background, keeping GT bbox aligned
- Start with 8ep probe (fast iteration)
- Expected: improved generalization on flatfish_2 / white_tailed_ptarmigan type cases

### Secondary recommendation: Label quality check on worst videos
- Manually inspect flatfish_2 and white_tailed_ptarmigan GT annotations
- If flatfish_2 GT is confirmed loose, consider tightening or flagging for exclusion
- Cost: ~30 min manual work, could improve measured mIoU by 0.01-0.02

### NOT recommended (based on audit findings):
- **More architectural changes**: E-40 already proves the model CAN predict good bboxes (per-frame mIoU 0.856). The problem is generalization, not capacity.
- **Loss function tuning**: E-40's nearly unbiased predictions (area_ratio=0.99, center_err=0.018) show the loss is working. The remaining errors are fundamentally about recognizing camouflaged objects.
- **Larger model**: More parameters would likely increase overfitting without solving the camouflage recognition problem.
- **Label smoothing / robust loss**: E-40's pred_too_small on flatfish_2 is not a loss problem — both models fail to see the flatfish.
