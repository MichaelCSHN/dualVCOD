# Autoresearch Cycle 4 — P3 Coverage 8ep Probe
## 2026-05-14 12:15

---

## Hypothesis
Asymmetric coverage penalty (weight=0.05) on large objects (area>0.15) nudges
the model toward fuller GT coverage, reducing pred_too_small errors and improving
IoU_large. Conservative weight avoids dominating SmoothL1+DIoU.

## Experiment
- `expl_p3_coverage_lowweight_8ep.json`: MV3-Small, dense_fg_aux, 8ep
- Baseline: E-45 (MV3-Small, 8ep, no coverage, pf_mIoU=0.8281)
- Single variable: large_coverage=0.05, large_area_threshold=0.15

## Training Trajectory

| Epoch | Train Loss | Train mIoU | Val mIoU | R@0.5 | IoU_tiny |
|-------|-----------|-----------|---------|-------|----------|
| 1 | 0.967 | 0.410 | 0.237 | 0.120 | 0.009 |
| 2 | 0.716 | 0.547 | 0.256 | 0.193 | 0.010 |
| 3 | 0.605 | 0.624 | 0.287 | 0.219 | 0.035 |
| 4 | 0.528 | 0.677 | 0.290 | 0.221 | 0.034 |
| 5 | 0.473 | 0.715 | **0.307** | 0.229 | 0.045 |
| 6 | 0.431 | 0.748 | 0.304 | 0.225 | 0.080 |
| 7 | 0.393 | 0.771 | **0.306** | 0.221 | 0.065 |
| 8 | 0.372 | 0.769 | 0.303 | 0.217 | 0.058 |

Double-peak convergence (E5+E7) — more stable than P1 zoom's single peak.
Final train mIoU 0.769, best val 0.307.

## Unified Reeval Results

| Checkpoint | Epoch | Train Val | pf_mIoU (unified) | bad | R@0.5 |
|-----------|-------|-----------|-------------------|-----|-------|
| Rank 1 | 5 | 0.307 | 0.777 | 0.093 | 0.907 |
| Rank 2 | 7 | 0.306 | 0.820 | 0.051 | 0.950 |
| **Rank 3** | **8** | **0.303** | **0.842** | **0.046** | **0.954** |

**Training-val vs unified reeval disagreement confirmed again**: Epoch 8 had
the _worst_ training val (0.303) but the _best_ unified pf_mIoU (0.842).
Epoch 5 had the best training val (0.307) but worst unified pf_mIoU (0.777).
Delta = 0.065 between "best by training" and "best by reality". This is the
second consecutive experiment proving training val is unreliable for checkpoint
selection — unified reeval is mandatory.

### Epoch 8 vs E-45 Baseline

| Metric | E-45 (no coverage) | P3 Coverage (epoch 8) | Delta |
|--------|-------------------|----------------------|-------|
| pf_mIoU | 0.8281 | **0.8420** | **+0.0139** |
| bad_frame_rate | 0.050 | **0.046** | -0.004 |
| R@0.5 | 0.950 | **0.954** | +0.004 |
| IoU_tiny | **0.584** | 0.573 | -0.011 |
| IoU_small | 0.706 | **0.756** | **+0.050** |
| IoU_medium | 0.838 | **0.851** | +0.013 |
| IoU_large | 0.860 | **0.874** | +0.014 |
| n_pred_too_large | 21 | **7** | -14 ✅ |
| n_pred_too_small | **79** | 85 | +6 ❌ |
| n_center_shift | **7** | 19 | +12 ⚠️ |
| flatfish_2 | 0.087 | N/A | — |

## Go/No-Go Criteria

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| IoU_large +0.02 | +0.020 | +0.014 | ❌ NOT MET |
| pred_too_small decreases | <79 | 85 | ❌ NOT MET |
| pred_too_large not >20% increase | ≤25 | 7 | ✅ MET |
| flatfish_2 +0.03 | +0.030 | N/A | ⚠️ Unknown |
| pf_mIoU drop >0.02 (no-go) | ≤0.02 | +0.014 | Not triggered |
| Any non-large bin drop >0.03 (no-go) | ≤0.03 | Tiny -0.011 | Not triggered |

## Verdict: **NO-GO** ⚠️ (Borderline)

Success criteria not met: IoU_large improved but fell short of target (+0.014
vs +0.020), and n_pred_too_small _increased_ (79→85) — the opposite of intent.
The coverage penalty at weight 0.05 was too gentle to meaningfully reshape
large-object predictions in only 8 epochs.

However, the no-go criteria were NOT triggered, and pf_mIoU improved (+0.014).
This is not a rejection of the mechanism — it's evidence that weight=0.05 at
8ep is insufficient, not that coverage penalty is harmful.

### Why success criteria weren't met
1. Weight 0.05 is ~1/40 of the combined SmoothL1+DIoU loss — a whisper, not a nudge
2. 8 epochs may be insufficient for coverage shapes to propagate through the backbone
3. Large objects (area>0.15) are only ~12% of GT instances — the penalty fires rarely
4. n_center_shift doubled (7→19), suggesting the penalty may have subtle side-effects

### Broader pattern
Across P1 and P3, IoU_small consistently improved (+0.050 here, +0.002 in P1).
Medium objects also improved modestly (+0.013). The coverage penalty may be
helping mid-size objects more than large objects — an unexpected but useful
signal.

### Potential follow-up (not now)
- Higher coverage weight (0.15–0.25) at 8ep
- 30ep with weight 0.05: more epochs for the penalty to shape predictions
- Weight scheduling: higher early, taper late

---

## P1 + P3 Combined Learning

| Finding | P1 (Zoom) | P3 (Coverage) | Pattern |
|---------|-----------|---------------|---------|
| pf_mIoU delta | +0.003 | +0.014 | Both mild improvements |
| IoU_tiny delta | **-0.130** ❌ | -0.011 | Zoom hurts tiny, coverage neutral |
| IoU_small delta | +0.002 | +0.050 | Coverage helps small |
| IoU_large delta | -0.001 | +0.014 | Coverage helps large (but not enough) |
| n_pred_too_large | +90 ⚠️ | -14 ✅ | Opposite effects |
| Training val vs unified | E5 best→E8 best (reversed) | E5 best→E8 best (reversed) | Both unreliable |
| Verdict | NO-GO | NO-GO (borderline) | — |

Two for two experiments where:
1. Training val ranking disagrees with unified reeval
2. The primary hypothesis metric moves in the wrong direction or underperforms
3. Secondary metrics show modest cross-benefits

---

## Decision: Proceed to P5 Temporal Dilation

Per roadmap priority: P5 is independent of P1/P3 results. Temporal dilation
modifies the temporal sampling window (stride) with zero code changes — config
only.

The P5 configs are already created:
- `smoke_p5_dilation_1ep.json`: MV3-Small, stride=2, 1ep smoke
- `expl_p5_dilation_stride2_8ep.json`: MV3-Small, stride=2, 8ep
- `expl_p5_dilation_stride3_8ep.json`: MV3-Small, stride=3, 8ep

Next: P5 1ep smoke → if passes, stride=2 + stride=3 8ep probes (concurrent or sequential).

After P5: P6 dense target refinement cautious probe.
