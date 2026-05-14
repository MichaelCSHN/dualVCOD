# Campaign A5: E-52 EffB0 30ep Top-K Retrain — BREAKTHROUGH
## 2026-05-14

---

## Result

**E30 pf_mIoU = 0.9142 vs original E-52 E20 = 0.8711 → Δ = +0.043**

Checkpoint selection cost 0.043 pf_mIoU for EffB0. Combined with A4, the average cost across both backbones is 0.047 pf_mIoU.

---

## Training Trajectory

| Phase | Epochs | Train Val Range | Notes |
|-------|--------|-----------------|-------|
| Early | E1-E5 | 0.29-0.36 | Rapid climb |
| Mid | E6-E15 | 0.35-0.38 | Peak at E12 (0.379), E13 (0.362) |
| Late | E16-E30 | 0.34-0.36 | Gradual decline by training val |

Training val best: E12 (0.379). E30 only 0.346 — ranked 8th+ by training val, forced into top-5 only by final-epoch save policy.

## Unified Reeval (Top-5)

| Rank | Epoch | Train Val | pf_mIoU | IoU_tiny | IoU_small | IoU_large | R@0.5 | bad_frame |
|------|-------|-----------|---------|----------|----------|----------|-------|----------|
| 5 | **30** | 0.346 | **0.9142** | **0.836** | **0.895** | **0.912** | **0.965** | **0.035** |
| 4 | 22 | 0.359 | 0.8716 | 0.676 | 0.799 | 0.877 | 0.960 | 0.040 |
| 3 | 13 | 0.362 | 0.8052 | 0.483 | 0.614 | 0.834 | 0.921 | 0.079 |
| 1 | 12 | 0.379 | 0.7890 | 0.314 | 0.582 | 0.845 | 0.908 | 0.092 |
| 2 | 11 | 0.362 | 0.7460 | 0.478 | 0.612 | 0.776 | 0.920 | 0.080 |

**Training-val-best (E12) ≠ unified-best (E30). Disagreement: 8/8 across all campaigns.**

**Gap between training-best and unified-best: 0.125 pf_mIoU** — the largest yet observed.

**Final-epoch force-save VINDICATED AGAIN**: E30 was NOT in the training-val top-5. Without the policy, this result would have been lost.

---

## A4 vs A5: MV3-Small vs EffB0 at True Ceiling (E30)

| Metric | A4 (MV3-Small) | A5 (EffB0) | Delta |
|--------|---------------|-----------|-------|
| pf_mIoU | 0.9077 | **0.9142** | EffB0 +0.007 |
| IoU_tiny | **0.849** | 0.836 | MV3 +0.013 |
| IoU_small | 0.865 | **0.895** | EffB0 +0.030 |
| IoU_medium | 0.915 | **0.928** | EffB0 +0.013 |
| IoU_large | 0.911 | **0.912** | Tie |
| R@0.5 | **0.967** | 0.965 | Tie |
| bad_frame | **0.033** | 0.035 | Tie |

The "EffB0 > MV3-Small" narrative was a checkpoint selection artifact. At true ceiling, the backbones are within 0.007 pf_mIoU — effectively tied. MV3-Small is **stronger** on tiny objects (+0.013). EffB0 is stronger on small (+0.030) and medium (+0.013) objects. Large objects and detection rates are tied.

MV3-Small retains a genuine specialization for the hardest category (tiny objects), making it the stronger backbone for the problem's primary failure mode.

---

## Updated Baseline Table (Final)

| Baseline | Backbone | Ep | Checkpoint | pf_mIoU | IoU_tiny | Status |
|----------|---------|-----|-----------|---------|----------|--------|
| E-45 | MV3-Small | 8 | E7 (val=0.305) | 0.8281 | 0.584 | Under-reported? |
| A2 retrain | MV3-Small | 8 | E8 (val=0.294) | 0.8314 | 0.511 | Confirmed |
| **A4** | **MV3-Small** | **30** | **E30** | **0.9077** | **0.849** | **TRUE CEILING** |
| E-40 original | MV3-Small | 30 | E18 | 0.8564 | 0.703 | UNDER-REPORTED by 0.051 |
| E-51 | EffB0 | 8 | E8 | 0.8372 | 0.512 | Confirmed |
| E-52 original | EffB0 | 30 | E20 | 0.8711 | 0.602 | UNDER-REPORTED by 0.043 |
| **A5** | **EffB0** | **30** | **E30** | **0.9142** | **0.836** | **TRUE CEILING** |

---

## Implications

### 1. Checkpoint selection is the #1 problem — worse than any architecture change

| Problem | pf_mIoU cost |
|---------|-------------|
| Wrong checkpoint (A4 MV3-Small) | -0.051 |
| Wrong checkpoint (A5 EffB0) | -0.043 |
| Resolution 224→256 | -0.079 |
| P3 coverage | +0.014 (failed) |
| P1 zoom | +0.003 (failed) |

The 0.047 average checkpoint-selection cost exceeds every architectural intervention attempted in Phase 2.

### 2. Backbone comparison was fundamentally wrong

Original: EffB0 0.871 vs MV3-Small 0.856 = EffB0 "wins by 0.015"
Truth: EffB0 0.914 vs MV3-Small 0.908 = within 0.007 — no clear winner

MV3-Small is actually **better** for the hardest category (tiny objects, +0.013). EffB0 is better for small/medium objects. The choice depends on which failure mode is prioritized.

### 3. All Phase 2 results need reinterpretation

Against the true baseline (0.908-0.914), some Phase 2 "failures" may have shown positive signals. The 0.828 baseline was ~0.08 below the true ceiling — interventions were graded against a phantom.

### 4. Training val is actively harmful for checkpoint selection

8/8 disagreements across all campaigns. Training val consistently selects early-mid checkpoints (E11-E13 for EffB0, E14-E17 for MV3-Small) that are 0.05-0.12 below the true ceiling. The metric is not just noisy — it's systematically biased toward under-trained checkpoints.

### 5. Final-epoch force-save policy must be permanent

Both A4 and A5 breakthroughs depended on E30 being force-saved outside the top-K heap. Without this policy, both results would have been lost. The policy must be codified as a permanent feature of the trial runner.

---

## Next Actions

1. **Update `run_trial_minimal.py`**: Codify final-epoch force-save + top-K policy permanently
2. **Reinterpret Phase 2**: With true baselines, re-examine P1-P6 go/no-go decisions
3. **Campaign D (coverage)**: A5 E30 still shows n_pred_too_small=175 — coverage is the remaining error mode
4. **E-55 (large coverage penalty)**: With large-object IoU at 0.912, is there meaningful headroom?
5. **E-54 (zoom)**: Tiny-object IoU at 0.836-0.849 — zoom augmentation could push this higher
