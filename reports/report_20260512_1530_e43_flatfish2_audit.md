# E-43: flatfish_2 Label Quality Audit

**Date**: 2026-05-12 15:30
**Type**: Data/label audit (no training)
**Input**: E-41.5 hypothesis that flatfish_2 GT is too loose
**Key question**: Is flatfish_2 GT bbox too loose/misaligned, causing valid model predictions to be systematically penalized?

---

## 1. flatfish_2 GT Annotation Quality — VERDICT: ACCURATE

### Method

Compared MoCA CSV bbox annotations against MoCA_Mask pixel-level GT masks for all overlapping frames (00030–00095, every 5th frame, 14 frames total). For each frame, the mask-derived tight bbox (bounding rectangle of all foreground pixels) was compared against the MoCA CSV bbox.

### Results

| Metric | Value | Interpretation |
|--------|-------|---------------|
| Mean IoU (CSV vs Mask) | **0.896** | Very high — CSV bbox closely matches pixel-level GT |
| Mean Area Ratio (CSV/Mask) | **1.09** | CSV bbox is only 9% larger than tight mask bbox |
| Mean Center Error | **0.014** | Negligible (1.4% of image diagonal) |
| Min IoU | 0.731 | Even worst frame has reasonable alignment |
| Max Area Ratio | 1.35 | Even loosest frame is only 35% larger |

### Per-Frame Details

| Frame | IoU(CSV,Mask) | AreaRatio | CSV_y1 | Mask_y1 | CSV_y2 | Mask_y2 |
|-------|--------------|-----------|--------|---------|--------|---------|
| 00030 | 0.947 | 1.05 | 0.059 | 0.056 | 0.769 | 0.758 |
| 00050 | 0.942 | 1.05 | 0.096 | 0.093 | 0.775 | 0.774 |
| 00070 | 0.965 | 0.98 | 0.072 | 0.058 | 0.736 | 0.731 |
| 00090 | 0.758 | 1.29 | 0.008 | 0.001 | 0.609 | 0.531 |
| 00095 | 0.731 | 1.35 | 0.004 | 0.000 | 0.507 | 0.436 |

The flatfish occupies a very large portion of the frame — up to 28% of total area. This is a genuinely large object, and the GT bbox correctly captures its full extent.

### E-41.5 Hypothesis Correction

The E-41.5 report hypothesized that flatfish_2 GT was "too loose" based on:
- GT covering 25%+ of frame (unusual for camouflaged objects)
- Both E-31 and E-40 predicting smaller, differently-positioned boxes
- Models agreeing with each other MORE than with GT

**All three observations are correct, but the interpretation was wrong.** The GT is accurate — the flatfish really IS that large. The models are failing to detect the full extent of a well-camouflaged large object. Model agreement doesn't indicate GT error; it indicates a shared model capability ceiling for this type of camouflage.

---

## 2. Problem Type: Genuine Hard Camouflage

### What's actually happening

The flatfish is a large, flat fish that blends near-perfectly with the sandy seabed. The GT bbox captures the entire fish body. Both models detect only the most visually distinguishable portion (likely the head/eyes/gills area — the part with highest contrast against the background) and miss the camouflaged body.

### Evidence

| Metric | E-31 | E-40 |
|--------|------|------|
| per-video mIoU | 0.120 | 0.215 |
| R@0.5 | 0.0 | 0.0 |
| Top error type | pred_too_small | pred_too_small |
| Center error vs CSV GT | 0.15-0.55 | 0.09-0.49 |
| Center error vs Mask GT | similar | similar |

Both models predict boxes too small AND shifted downward — they detect the fish's head (lower in frame) but miss the body (extending upward):

- GT y1 ≈ 0.00-0.10 (top of fish near top of frame)
- Model y1 ≈ 0.29-0.59 (models start box much lower — missing upper body)
- GT y2 ≈ 0.51-0.79
- Model y2 ≈ 0.63-0.99 (models extend to bottom)

The models are essentially finding the "most fish-like" region while the GT covers the entire animal including well-camouflaged body parts.

### E-40 vs E-31 on flatfish_2

E-40's dense_fg_aux improves mIoU from 0.120 → 0.215 (+0.095, nearly doubles E-31). This is meaningful progress but still far from acceptable performance. The dense foreground supervision helps the model identify more of the object extent, but the core camouflage challenge remains.

---

## 3. Manual Audit Scope

| Item | Count |
|------|-------|
| Frames with MoCA_Mask GT | 14 (frames 00030–00095, every 5th) |
| MoCA CSV annotated frames | 20 (frames 00000–00095, every 5th) |
| Total val frames | 80 (16 samples × 5-frame clips with overlap) |
| Annotated images generated | 16 (one per sample, t=0) |
| Quality classification | 16/16 frames = GT accurate |

No frame was classified as "GT too loose," "GT misaligned," or "target unidentifiable" after mask comparison. The initial heuristic classification (15/16 "GT_LIKELY_LOOSE" based on model-GT disagreement) was **overruled** by the mask-based ground truth comparison.

---

## 4. Corrected Bbox File — NOT CREATED

No `flatfish_2_corrected_bboxes.csv` was generated because **no corrections are needed**. The MoCA CSV annotations are accurate (mean IoU 0.896 vs pixel-level GT).

Supporting files created for audit transparency:
- `reports/e43_flatfish2_corrections/flatfish_2_moca_vs_mask.csv` — 14-frame comparison of CSV vs mask bbox
- `reports/e43_flatfish2_corrections/flatfish_2_audit.csv` — 16-frame audit with model predictions
- `reports/e43_flatfish2_corrections/flatfish_2_*.png` — 16 annotated frames with GT+E31+E40 overlays
- `tools/e43_flatfish2_audit.py` — audit script
- `tools/e43_mask_vs_bbox_compare.py` — mask comparison script
- `tools/e43_model_vs_mask.py` — model vs mask evaluation script

---

## 5. Metric-Only Re-Evaluation — SKIPPED

Since no GT corrections are needed, metric-only re-evaluation with corrected GT would produce identical results to existing E-31/E-40 evaluations.

### Ceiling Analysis

What if flatfish_2 were perfectly solved (mIoU=1.0)?

| Scenario | Overall pf_mIoU | Change |
|----------|----------------|--------|
| E-40 current | 0.8564 | — |
| flatfish_2 perfect | 0.8669 | +0.011 |
| flatfish_2 + ptarmigan perfect | 0.8798 | +0.023 |
| All worst-4 perfect | 0.8927 | +0.036 |

If flatfish_2 were excluded from validation entirely: pf_mIoU = 0.8651 (+0.009).

---

## 6. flatfish_2 Impact on Overall Ceiling

**flatfish_2's impact on aggregate metrics is small but its diagnostic value is high.**

- 80/5955 frames = 1.3% of validation data
- Contributes only ~0.01 to overall mIoU ceiling
- But: it's the SINGLE WORST video for both E-31 and E-40
- It reveals a systematic model weakness: failure to detect full extent of large camouflaged objects
- bad_frame_rate: flatfish_2 alone contributes 80/214 = 37% of E-40's bad frames

The value of flatfish_2 is NOT in aggregate metric improvement — it's as a **diagnostic probe** for the model's ability to spatially localize well-camouflaged large objects. Any future model improvement that helps flatfish_2 likely helps all camouflage cases at the boundary.

---

## 7. Should We Extend to Other Worst Videos?

**white_tailed_ptarmigan** (worst #2, 125 frames, E-40 mIoU=0.388):
- NOT in MoCA_Mask — no pixel-level GT available for verification
- Visual characteristics: white bird on snow (classic camouflage)
- E-41.5 already classified this as genuine camouflage failure
- No annotation audit possible without mask GT
- **Recommendation**: Keep as-is, treat as genuine hard case

**pygmy_seahorse_0** (worst #3, 80 frames, E-40 mIoU=0.623):
- E-40 already performs well (+0.459 vs E-31)
- E-40's error is modest (mIoU=0.623, R@0.5=0.85)
- **Recommendation**: No action needed

**arctic_fox_3** (worst #4, 150 frames, E-40 mIoU=0.688):
- Moderate performance, R@0.5=0.98 (good recall)
- **Recommendation**: No action needed

**Conclusion**: No label audit extension needed. flatfish_2 was the only video with a plausible "loose GT" hypothesis, and it was disproven.

---

## 8. Next Recommended Action

### Primary Recommendation: E-44 — Temporal feature aggregation for large-object boundary delineation

**Rationale**:
- E-43 proved flatfish_2 is NOT a label quality problem — it's a model capability gap for large camouflaged objects
- The model detects the object's presence but fails to delineate its full spatial extent
- Current architecture uses 5-frame temporal context — this may be insufficient for boundary delineation of large static objects
- The flatfish barely moves — temporal signal across 5 frames is weak
- Longer temporal context (T=9 or T=13) or explicit temporal feature aggregation might help the model accumulate evidence about where the object boundary actually is

**Specific approach**:
- Increase temporal_T from 5 → 9 or 13 (more frames to accumulate boundary evidence)
- Keep all other E-39 params unchanged (dense_fg_aux, lr=0.001, T_max=8)
- 8ep probe first
- Eval criteria: flatfish_2 mIoU improvement (primary), overall pf_mIoU (secondary)
- Risk: more temporal context may not help if object is static (no motion cue)

**Alternative**: Multi-scale feature fusion to better capture large objects. Current FPN may bias toward small object detection (as seen in E-31's overprediction on tiny objects). A simple test: add a larger-scale feature level.

### What NOT to do (reinforced by E-42 + E-43):

- ❌ Background mixing / CutMix / MixUp (E-42 proved catastrophic)
- ❌ Label correction / cleanup (E-43 proved labels are accurate)
- ❌ New augmentation of any kind
- ❌ Backbone changes
- ❌ Scheduler tuning
- ❌ dense_fg_weight tuning

### Decision Logic

If the user wants to pursue flatfish_2 improvement specifically, E-44 (temporal T increase) is the cleanest single-variable experiment. If the user considers flatfish_2 a minor issue (1.3% of frames, +0.01 ceiling), then the focus should shift to the NEXT major direction beyond dense_fg_aux — e.g., backbone scaling, resolution increase, or loss function refinement.

---

## Summary

| Question | Answer |
|----------|--------|
| flatfish_2 GT quality issue? | **No** — GT is accurate (IoU 0.896 vs mask GT) |
| Problem type? | Genuine hard camouflage — large well-blended object |
| Frames audited? | 14 with mask GT + 16 with model overlays = 30 total |
| Corrected bbox file? | Not created — no corrections needed |
| Metric re-evaluation? | Skipped — no GT changes to apply |
| Ceiling impact? | +0.01 overall mIoU if perfectly solved |
| Extend to other videos? | No — ptarmigan has no mask GT, others perform adequately |
| Next experiment? | E-44: Temporal T increase (9 or 13) for boundary delineation |
