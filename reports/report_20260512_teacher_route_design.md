# Teacher Route Experiment Group: Phase 16R

**Date**: 2026-05-12
**Status**: Design phase — awaiting user approval before launch
**Predecessor**: E-43 (flatfish_2 label audit), E-44 (T=9 negative)

---

## 1. Core Research Question

**Can softer, more structured teacher-style dense targets improve bbox-only student generalization compared to current hard binary dense_fg_aux targets, especially for large camouflaged object extent recovery?**

Current dense_fg_aux uses hard binary masks (0/1) as BCE targets. These provide no gradient about how "close" a pixel is to being foreground. The Teacher hypothesis: softer targets that encode spatial uncertainty (gradual boundaries, center-to-edge falloff) provide more informative supervision, helping the model recover complete object extent — particularly for large well-camouflaged objects like flatfish_2 where the current model detects presence but fails to delineate full extent.

---

## 2. Experiment Design: 3-Experiment Minimal Group

| Experiment | `dense_target_mode` | Real PNG masks (MoCA_Mask/CAD) | Bbox-only samples (MoCA CSV) |
|------------|---------------------|-------------------------------|------------------------------|
| **E-45** (baseline) | `hard` | Hard binary | Hard rectangle |
| **E-46** (soft mask) | `soft_mask` | Gaussian blurred (σ=1.0) | Hard rectangle |
| **E-47** (soft bbox) | `soft_bbox` | Hard binary | Gaussian falloff (σ_factor=0.3) |

### Why these 3 experiments are the minimum necessary set

- **E-45 vs E-46**: Isolates the effect of softening **real mask** boundaries. Only MoCA_Mask/CAD samples (~40% of training data) are affected. Answers: "Does softening boundaries on pixel-level GT masks help the model learn gradual foreground transitions?"

- **E-45 vs E-47**: Isolates the effect of replacing **hard bbox rectangles** with Gaussian spatial falloff. Only MoCA CSV samples (~60% of training data) are affected. Answers: "Can a structured spatial prior (center-peaked Gaussian) provide better supervision than a hard rectangle for bbox-only samples?"

- No E-48 (soft_all) included: If both E-46 and E-47 are positive, the combined effect is predictable and can be tested as a follow-up. If only one is positive, the isolated signal is cleaner without confounding. If neither is positive, combined would be negative too.

### Alternative directions considered but deferred

- **Direction C (center+extent decomposition)**: Requires head architecture change (predicting separate center heatmap + extent maps). Higher implementation cost. Only worth pursuing if E-46 or E-47 shows that softer targets help.
- **Direction D (mask-only teacher)**: Subsumed by E-46 — softening real masks IS the mask-derived teacher direction.

---

## 3. Variable Control: What Changes, What Stays Fixed

### Changed variables (per experiment)

| Variable | E-45 | E-46 | E-47 |
|----------|------|------|------|
| `dense_target_mode` | `hard` | `soft_mask` | `soft_bbox` |

### Held constant (all experiments)

| Variable | Value | Justification |
|----------|-------|---------------|
| backbone | MobileNetV3-Small | Mainline default |
| input_size | 224 | Mainline default |
| temporal_T | 5 | Mainline default |
| head | dense_fg_aux | Mainline default |
| lr | 0.001 | Mainline default |
| batch_size | 16/64 | Mainline default |
| epochs | 8 | Standard probe length |
| seed | 42 | Reproducibility |
| loss_weights | DIoU + dense_fg=0.5 | Mainline default |
| weight_decay | 1e-4 | Mainline default |
| scheduler | CosineAnnealingLR(T_max=8) | Mainline default |
| augmentation | HFlip + ColorJitter(0.15) | Mainline default |
| datasets | MoCA + MoCA_Mask + CAD | Mainline default |

---

## 4. Per-Experiment Hypotheses and Decision Criteria

### E-45 — Hard Target Baseline

**Sub-question**: Does the current hard-target dense_fg_aux setup reproduce E-39 results?

**Expected**: pf_mIoU ≈ 0.82–0.83, bad_frame_rate ≈ 0.04–0.05, area_ratio ≈ 0.95–1.05. Within ±0.01 of E-39 pf_mIoU (0.8277).

**Decision threshold**: Must complete successfully to serve as valid control. If pf_mIoU < 0.80 (significant regression vs E-39), investigate before proceeding.

### E-46 — Soft Mask Targets on True-Mask Samples

**Sub-question**: Does Gaussian-blurring real PNG mask boundaries improve the model's ability to learn object extent, particularly for large objects?

**Mechanism**: Hard mask boundaries at 28×28 create a sharp 1→0 transition. Gaussian blur (σ=1.0, ~3px at 28×28) creates a gradual transition zone where BCE targets encode "uncertain" boundary pixels. This softer supervision may:
- Reduce overconfidence at object boundaries
- Provide gradient signal for pixels near the boundary
- Help the model learn that boundaries are inherently uncertain for camouflaged objects

**Expected**: If softening real masks helps, pf_mIoU should increase (better boundary delineation) and large-object IoU should improve. Area_ratio may shift slightly (model learns to predict more complete extent). flatfish_2 per-video mIoU may improve as the model learns to handle gradual boundaries.

**Decision**: 
- **Positive**: pf_mIoU > E-45 + 0.005, OR large-object IoU > E-45 + 0.02, OR flatfish_2 per-video mIoU > E-39 (0.215)
- **Negative**: pf_mIoU < E-45 − 0.005 AND no improvement in large-object metrics
- **Neutral**: Within ±0.005 of E-45 on all metrics → soft real masks don't help but don't hurt

### E-47 — Soft Bbox Extent Targets for Bbox-Only Samples

**Sub-question**: Can a 2D Gaussian falloff from bbox center provide better supervision than a hard rectangle for samples without real pixel masks?

**Mechanism**: Hard rectangles say "all pixels inside are equally foreground, all outside are equally background." A 2D Gaussian (σ_x ∝ bbox_width, σ_y ∝ bbox_height) says "the center is definitely foreground, confidence falls off toward edges." This encodes:
- Spatial structure: center is more reliably foreground than edges
- Uncertainty: boundaries are uncertain (soft falloff extends slightly beyond bbox)
- Relative scale: larger objects get broader Gaussians (more uncertainty about exact extent)

**Expected**: The Gaussian target should provide more informative gradients. The model may learn to predict more conservative (centered) boxes initially, then expand to full extent as confidence grows. Area_ratio should move closer to 1.0 (less over/under-prediction). Large-object IoU should improve since the Gaussian naturally handles large extent uncertainty.

**Decision**:
- **Positive**: pf_mIoU > E-45 + 0.005, OR area_ratio closer to 1.0 than E-45, OR large-object IoU > E-45 + 0.02
- **Negative**: pf_mIoU < E-45 − 0.005 AND worse area_ratio
- **Neutral**: Within ±0.005 of E-45 → Gaussian falloff doesn't add value over hard rectangles

---

## 5. Experiment Group Success/Failure Criteria

### Good (Teacher route validated for 30ep)
- Either E-46 or E-47 pf_mIoU > E-45 by ≥ 0.005, OR
- Large-object IoU improves by ≥ 0.02 in either experiment, OR
- flatfish_2 per-video mIoU improves by ≥ 0.05 in either experiment

### Mixed (partial signal, refine and re-probe)
- One experiment positive, one negative → pursue the positive direction
- Both experiments show small improvements (< 0.005) but consistent across metrics → consider increasing sigma/blur parameters

### Bad (Teacher route negative at this level)
- Both E-46 and E-47 pf_mIoU ≤ E-45 (within noise)
- No improvement in large-object IoU or flatfish_2
- → Soft targets at this level don't help; reconsider Teacher route approach (e.g., Direction C center+extent decomposition, or different soft target formulation)

---

## 6. Data Scale and Estimated Cost

| Item | Estimate |
|------|----------|
| Training samples | ~4000–5000 (MoCA + MoCA_Mask + CAD combined) |
| Validation samples | ~700–900 (MoCA val split) |
| Real-mask samples (train) | ~40% (MoCA_Mask + CAD) |
| Bbox-only samples (train) | ~60% (MoCA CSV) |
| Epochs per experiment | 8 |
| Est. time per experiment | ~40–60 min (single GPU, bs=16) |
| **Total GPU hours** | **~2–3 hours** (3 experiments) |
| Disk per experiment | ~50 MB (checkpoint + logs) |

---

## 7. Risks and Controversial Points

### Risk 1: pos_weight behavior with soft targets
BCE pos_weight is computed dynamically: `pos_weight = n_neg / n_pos`. With soft targets, `n_pos` is the sum of soft values (not integer count). For Gaussian-blurred masks, the total "positive mass" decreases slightly (boundary pixels contribute 0.5 instead of 1.0). pos_weight may increase marginally but stays within the [1, 50] clamp. **Mitigation**: Monitor dense_fg_loss values in early batches; if pos_weight spikes, adjust clamping.

### Risk 2: Gaussian blur may be too subtle at 28×28
At 28×28 resolution, σ=1.0 affects only ~3 pixels at boundaries. For small objects (4×4 pixels), the blur zone is proportionally large (75% of object width). For large objects (15×15), it's modest (20%). This asymmetry may have unintended effects. **Mitigation**: Monitor IoU by size (tiny/small/medium/large) to detect size-specific regressions.

### Risk 3: Gaussian falloff may encourage "shy" predictions
If the Gaussian target is too soft, the model may learn to predict small, high-confidence boxes near the center rather than full-extent boxes. This would manifest as area_ratio < 0.8 (systematic under-prediction). **Mitigation**: area_ratio is a primary metric; if it drops significantly, the Gaussian is too soft.

### Risk 4: E-45 may not exactly reproduce E-39
Seed=42 provides reproducibility for model init and data splits, but CUDA nondeterminism and DataLoader shuffle order may introduce small variations. **Mitigation**: The decision thresholds (±0.005) account for run-to-run noise. If E-45 deviates > 0.01 from E-39, re-run before comparing.

### Risk 5: Only 3 experiments limits decomposition
With 3 experiments, we cannot distinguish between "Gaussian falloff works" and "any non-rectangle shape works" for bbox-derived targets. A 4th experiment with a different soft shape (e.g., soft rectangle with sigmoid edges instead of Gaussian) would decompose this. **Judgment**: Not worth the extra GPU hour for an 8ep probe — if E-47 is positive, the follow-up 30ep verification can include shape ablation.

---

## 8. Control Variable Checklist

- [x] backbone = mobilenet_v3_small (no change)
- [x] T = 5 (no change)
- [x] input_size = 224 (no change)
- [x] head = dense_fg_aux (no change)
- [x] lr = 0.001, scheduler = CosineAnnealingLR(T_max=8) (no change)
- [x] batch_size = 16, seed = 42 (no change)
- [x] loss_weights: DIoU + dense_fg=0.5 (no change)
- [x] augmentation: HFlip + ColorJitter(0.15) (no change)
- [x] datasets: MoCA + MoCA_Mask + CAD (no change)
- [x] bg_mix_prob = 0.0 (no background mixing — E-42 closed)
- [x] weight_decay = 1e-4 (no change)
- [x] Only variable changed: `dense_target_mode` (hard / soft_mask / soft_bbox)

---

## 9. Decision Tree

```
E-45, E-46, E-47 complete (8ep each)
│
├── BOTH E-46 AND E-47 POSITIVE (pf_mIoU > E-45 + 0.005)
│   └── Teacher route VALIDATED
│       ├── Run E-48 (soft_all) 8ep to confirm synergy
│       ├── Select best single-soft mode for 30ep verification
│       └── Consider Direction C (center+extent) as next-level Teacher
│
├── E-46 POSITIVE, E-47 NEGATIVE
│   └── Soft real masks help, soft bbox doesn't
│       ├── Run E-46 30ep verification (dense_target_mode=soft_mask)
│       ├── Tune Gaussian blur sigma (try σ=0.5, σ=1.5)
│       └── Consider: acquire more real-mask data to expand soft_mask coverage
│
├── E-47 POSITIVE, E-46 NEGATIVE
│   └── Gaussian falloff helps, soft real masks don't
│       ├── Run E-47 30ep verification (dense_target_mode=soft_bbox)
│       ├── Tune sigma_factor (try 0.2, 0.4)
│       └── Consider: test other falloff shapes (sigmoid edges, Laplacian)
│
├── BOTH NEUTRAL (within ±0.005, no large-object improvement)
│   └── Soft targets don't help but don't hurt
│       ├── Try stronger softening (σ=1.5, sigma_factor=0.4) in 2-experiment follow-up
│       └── If still neutral: abandon simple soft targets, pivot to Direction C
│
└── BOTH NEGATIVE (worse than E-45)
    └── Soft targets are harmful for this architecture
        ├── Record as negative finding, add to closed directions
        ├── Pivot to Direction C (center+extent decomposition) as alternative Teacher route
        └── OR: Return to bbox-only improvements (loss function, backbone scaling)
```

---

## Summary

| Item | Value |
|------|-------|
| Core question | Can softer teacher-style dense targets improve bbox-only student generalization? |
| Experiments | 3 (E-45 baseline, E-46 soft_mask, E-47 soft_bbox) |
| Variable changed | `dense_target_mode` only |
| Estimated GPU time | ~2–3 hours total |
| Primary metrics | pf_mIoU, large-object IoU, area_ratio, flatfish_2 per-video mIoU |
| Success threshold | pf_mIoU > baseline + 0.005 OR large-object IoU > baseline + 0.02 |
| Code changes | `dataset_real.py` (+3 methods, +1 param), `run_trial_minimal.py` (+1 config read, +3 passes) |
