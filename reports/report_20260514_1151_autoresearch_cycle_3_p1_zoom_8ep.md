# Autoresearch Cycle 3 — P1 Zoom 8ep Probe
## 2026-05-14 11:51

---

## Hypothesis
Probabilistic per-clip zoom on small/tiny objects improves feature resolution
for the primary failure mode (tiny-object mIoU near zero). Conservative probs
(50/30/10/0), context_factor=2.0, clip-consistent, training-only.

## Experiment
- `expl_p1_zoom_conservative_8ep.json`: MV3-Small, dense_fg_aux, 8ep
- Baseline: E-45 (MV3-Small, 8ep, no zoom, pf_mIoU=0.8281)
- Single variable: zoom_enabled=true, conservative probs

## Training Trajectory

| Epoch | Train Loss | Train mIoU | Val mIoU | R@0.5 | IoU_tiny |
|-------|-----------|-----------|---------|-------|----------|
| 1 | 0.970 | 0.410 | 0.239 | 0.116 | 0.022 |
| 2 | 0.721 | 0.544 | 0.253 | 0.186 | 0.011 |
| 3 | 0.611 | 0.617 | 0.290 | 0.211 | 0.018 |
| 4 | 0.537 | 0.669 | 0.276 | 0.198 | 0.029 |
| **5** | **0.482** | **0.707** | **0.305** | **0.234** | **0.098** |
| 6 | 0.439 | 0.739 | 0.288 | 0.206 | 0.090 |
| 7 | 0.401 | 0.768 | 0.295 | 0.213 | 0.063 |
| 8 | 0.380 | 0.785 | 0.300 | 0.217 | 0.089 |

Peak training val: epoch 5 (0.305). Train saturated at 0.785.

## Unified Reeval Results

| Checkpoint | Epoch | Train Val | pf_mIoU (unified) | bad | R@0.5 |
|-----------|-------|-----------|-------------------|-----|-------|
| Rank 1 | 1 | 0.239 | 0.495 | 0.480 | 0.520 |
| Rank 2 | 2 | 0.253 | 0.616 | 0.217 | 0.783 |
| Rank 3 | 8 | 0.300 | 0.831 | 0.068 | 0.932 |
| Best (epoch 5) | 5 | **0.305** | 0.740 | 0.119 | 0.881 |

**Critical finding**: Training val ranking disagrees with unified reeval.
Training-best epoch 5 had unified pf_mIoU=0.740, while "worse" epoch 8 had
pf_mIoU=0.831. Unified reeval is essential for correct checkpoint selection.

### Epoch 8 vs E-45 Baseline

| Metric | E-45 (no zoom) | P1 Zoom (epoch 8) | Delta |
|--------|---------------|-------------------|-------|
| pf_mIoU | 0.8281 | 0.8309 | +0.003 |
| bad_frame_rate | **0.050** | 0.068 | +0.018 |
| R@0.5 | **0.950** | 0.932 | -0.018 |
| IoU_tiny | **0.584** | 0.454 | **-0.130** ❌ |
| IoU_small | 0.706 | 0.708 | +0.002 |
| IoU_medium | 0.838 | **0.861** | +0.023 |
| IoU_large | 0.860 | 0.859 | -0.001 |
| n_pred_too_large | **21** | 111 | +90 ⚠️ |

## Go/No-Go Criteria

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| IoU_tiny ≥ 0.55 | 0.55 | 0.454 | ❌ NOT MET |
| pf_mIoU drop ≤ 0.01 | 0.828→0.819 | 0.831 | ✅ MET |
| Any size bin drop > 0.05 | None > 0.05 | IoU_tiny -0.130 | ❌ TRIGGERED |
| pf_mIoU drop > 0.02 (no-go) | >0.02 | +0.003 | Not triggered |

## Verdict: **NO-GO** ❌

Zoom redistributes performance — medium objects improve (+0.023) at the cost
of tiny objects (-0.130). The primary hypothesis (zoom helps tiny objects) is
NOT supported. pf_mIoU is flat.

### Why did this happen?
1. Conservative probs mean only 50% of tiny objects get zoom — 50% don't
2. Model has to handle two different data distributions in 8 epochs
3. Zoom changes the bbox context — what looks "normal" in a zoomed view
   may not generalize to the unzoomed validation views
4. 8 epochs is insufficient for the model to reconcile zoomed and unzoomed
   distributions for the hardest (tiny) objects

### Potential follow-up (not now)
- 30ep zoom: more epochs might allow generalization
- More aggressive probs (80/50/20): more zoom exposure
- Smaller context_factor (1.5): less aggressive zoom

## Bugs Found

**Top-K heap**: Checkpoint_rank1 and rank2 are from epochs 1 and 2 (val 0.239,
0.253), which should have been evicted by epochs 3-7. The min-heap replacement
logic appears to not properly evict old entries. Needs investigation before
next 8ep probes that use topk_checkpoints.

---

## Decision: Proceed to P3 Coverage 8ep Probe

Per roadmap: P1 no-go → P2 (EffB0 zoom) closed. P3 is independent (coverage
loss), proceed immediately.

Next: `expl_p3_coverage_lowweight_8ep.json` — MV3-Small, coverage weight=0.05,
8ep, vs E-45 baseline.
