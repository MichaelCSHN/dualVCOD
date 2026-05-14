# GreenVCOD Strategic Re-Analysis & Revised Exploration Plan — 2026-05-14 09:59

## Context

Post-Phase-2, pre-experiment governance review. The docs overhaul (reports/report_20260514_1230) established baselines, closed directions, eval
protocol, and a P0-P7 priority queue. This report re-examines that queue
through the lens of the GreenVCOD paper (arXiv:2501.10914v1), separates
transferable principles from paper-specific machinery, and proposes a
revised exploration order.

No training was run. No new experimental claims are made.

---

## 1. GreenVCOD Re-Interpretation: What the Paper Actually Teaches

GreenVCOD is a **video camouflage object detection** system that achieves
SOTA on MoCA-Mini. Its architecture has four stages:

1. **COD-pretrained backbone** (EfficientNet-B4) extracts frame features
2. **Short-term + Long-term temporal neighborhoods** aggregate multi-scale
   motion cues via order-invariant pooling
3. **Dense prediction teacher** generates per-frame segmentation maps as
   intermediate supervision targets
4. **XGBoost prediction refinement** post-processes bbox predictions using
   hand-crafted features

The paper reports +2.3% mIoU over the prior SOTA. But the headline number
conflates four simultaneous interventions, making it impossible to
attribute gain to any single component.

### 1.1 What dualVCOD Already Does Better (or Equivalently)

| GreenVCOD Component | dualVCOD Equivalent | Verdict |
|---------------------|---------------------|---------|
| Short-term temporal neighborhood (T=5) | TemporalNeighborhood with order-invariant mean pooling, T=5 | **Equivalent.** dualVCOD's version is cleaner — no learned fusion weights to overfit. |
| Dense prediction maps (teacher) | `dense_fg_aux` — BCE dense foreground head on stride-8 features, weight=0.5, training-only | **Equivalent in spirit, simpler in execution.** GreenVCOD uses a pretrained teacher; we use GT bbox → binary mask. Our version has zero teacher-training cost and can't drift from GT. |
| COD10K/CAMO pretraining | Not implemented (P5 in roadmap) | **GreenVCOD's gain from COD pretraining is unquantified** — it's bundled with the other three components. |

### 1.2 What GreenVCOD Does That dualVCOD Should NOT Copy

| Component | Why Reject |
|-----------|-----------|
| EfficientNet-B4 backbone | 19M params, ~2× slower than EffB0. Violates "lighter, faster" charter. EffB0 (5.3M) already provides +0.0147 pf_mIoU over MV3-Small (2.5M). |
| XGBoost prediction refinement | Adds 80+ hand-crafted features, non-differentiable post-processing, separate training pipeline. Breaks end-to-end differentiability. Inference cost increase. |
| Segmentation-based inference | dualVCOD charter: bbox-only at inference. Non-negotiable. |
| Teacher-network dense supervision | Requires training a separate teacher model. Our `dense_fg_aux` (GT-derived masks) achieves the same principle with zero extra training. |
| Long-term temporal neighborhood (T=15+) | GreenVCOD's long-term branch uses stride-3 sampling over 15+ frames. Memory scaling is O(T) for feature concatenation. Our T=5 with temporal_stride exploration (P6) is a more parameter-efficient way to probe longer windows. |

### 1.3 What GreenVCOD Validates (Principles dualVCOD Should Absorb)

**Principle 1: Dense spatial supervision is the single most impactful
architectural choice.**

GreenVCOD's ablation (Table 3) shows that removing the dense prediction
teacher causes the largest single-component drop. This validates
dualVCOD's `dense_fg_aux` as a load-bearing design decision, not an
optional add-on. It also suggests that **making dense supervision stronger**
— not adding new heads, but improving the quality of the dense target —
could be the highest-leverage unexplored direction.

**Principle 2: Temporal context matters, but the form is flexible.**

GreenVCOD uses two temporal branches (short + long). dualVCOD uses one.
Both use order-invariant pooling. The key insight isn't the specific
architecture — it's that **camouflage is defined by motion**, and motion
requires multiple frames. Any change that improves temporal reasoning
(stride, dilation, longer T, attention-based pooling) is worth exploring,
but the paper doesn't prescribe which form is best.

**Principle 3: COD pretraining helps, but the marginal gain above strong
spatial + temporal baselines is unknown.**

GreenVCOD's paper doesn't isolate COD pretraining from the other three
components. The COD community broadly reports +1-3% from COD pretraining,
but these gains are measured on image-based COD benchmarks (COD10K, NC4K,
CAMO), not on video camouflage. Video camouflage is qualitatively different
— motion is the primary cue, and motion doesn't exist in static images.
COD pretraining on static images may not transfer cleanly.

**Principle 4: Frame-level spatial quality determines overall performance.**

GreenVCOD computes per-frame metrics and reports frame-level mIoU. This
validates dualVCOD's `pf_mIoU` as the primary metric and our focus on
per-frame error types. The paper's error patterns (small-object misses,
large-object under-coverage) match our Phase 2 audits.

**Principle 5: Simple interventions dominate complex ones.**

GreenVCOD's biggest gain comes from dense supervision (a training-only
loss change). The most complex component (XGBoost) is also the least
justified — no ablation isolates its contribution. This reinforces
dualVCOD's design philosophy: **zero-inference-cost, single-variable
interventions first.**

### 1.4 The Real Lesson

GreenVCOD's contribution isn't its specific architecture. It's the
**validation of a design philosophy** that dualVCOD already follows:
dense spatial supervision + temporal aggregation + bbox-only inference.
The paper's unexplored potential lies in what it DIDN'T ablate:
- Is COD pretraining worth the engineering cost when you already have
  strong dense supervision?
- Does long-term temporal context help beyond short-term for video
  camouflage specifically?
- Could a simpler refinement step (e.g., temporal smoothing of bbox
  predictions) achieve most of what XGBoost does?

These are the questions dualVCOD should answer, not by copying GreenVCOD's
architecture, but by testing the principles it implies.

---

## 2. E-54 / E-55 / E-56 Repositioning

These three implementations were written during the docs overhaul. They
have NOT been validated on real data. They are:

| ID | File | What It Does | Config Gate |
|----|------|-------------|-------------|
| E-54 | `src/dataset_real.py` | Scale-aware natural zoom augmentation | `zoom_enabled=false` (default) |
| E-55 | `src/loss.py` | Large-object asymmetric coverage penalty | `loss_weights.large_coverage=0.0` (default) |
| E-56 | `run_trial_minimal.py` + `reeval_checkpoints.py` | Top-K checkpoint saving + unified reeval | `topk_checkpoints=1` (default) |

### 2.1 Current Status

**Implementation complete, NOT experimentally validated.**

- Code compiles and imports successfully.
- Synthetic unit tests pass (E-54: zoom transform math verified; E-55:
  asymmetric penalty verified on toy tensors).
- All defaults are zero/off — no impact on existing training pipelines.
- No real-data training has been run.

### 2.2 What They Are

1. **Infrastructure**, not results. They are code ready for P1-P4 probes.
2. **Subject to revision.** Smoke tests may reveal issues requiring
   parameter or code changes before 8ep probes.
3. **Gated behind config flags.** Removing them from any config restores
   exact baseline behavior.
4. **Not to be cited as methods that "work."** No performance claim is made.

### 2.3 Validation Sequence (After Governance)

1. **1-epoch smoke test E-54** — verify zoom triggers on tiny clips, no
   NaN, val loss unchanged (zoom is train-only), bbox transforms correct
   on real images.
2. **1-epoch smoke test E-55** — verify coverage penalty > 0 only for
   large GT, loss stable, no gradient explosion.
3. **Validate E-56 on existing checkpoints** — run `reeval_checkpoints.py`
   on E-51 or E-52 saved checkpoints, verify ranking comparison output,
   confirm protocol matches manual unified reeval results.
4. Only after all three pass: proceed to 8ep probes.

---

## 3. Priority Re-Evaluation

### 3.1 Current Priority (from docs/03)

```
P0: Governance (no training)
P1: MV3-Small zoom probe (1ep→8ep)
P2: EffB0 zoom probe (1ep→8ep)
P3: MV3-Small coverage probe (1ep→8ep)
P4: EffB0 coverage probe (1ep→8ep)
P5: COD dense pretraining
P6: Temporal dilation
P7: EffB0 → MV3-Small distillation
```

### 3.2 Assessment Criteria

Each priority is scored on four dimensions (1=worst, 5=best):

| Dimension | Description |
|-----------|-------------|
| **Impact** | Expected pf_mIoU gain if successful |
| **Confidence** | Likelihood of success given theory + Phase 2 evidence |
| **Cost** | Engineering + compute cost (lower is better = higher score) |
| **Inference** | Zero inference cost = 5, adds params/FPS = lower |

### 3.3 Per-Direction Assessment

#### P1/P2 — Zoom Probe (MV3-Small / EffB0)

- **Impact: 4/5.** Tiny objects are the #1 failure mode. pygmy_seahorse_0
  mIoU ≈ 0.4 even in best models. Zoom directly addresses the resolution
  bottleneck at zero inference cost.
- **Confidence: 3.5/5.** Natural zoom is a well-established augmentation in
  general object detection. GreenVCOD validates multi-scale processing
  helps. Risk: 224×224 input may not contain enough pixel information for
  objects at area < 0.01 even with zoom.
- **Cost: 4/5.** Training-only augmentation. No architecture changes.
  Slight per-iteration overhead from F.interpolate.
- **Inference: 5/5.** Zero inference cost.
- **Overall: 4.1/5.** Strong. Should stay at P1/P2.

#### P3/P4 — Coverage-Aware Loss (MV3-Small / EffB0)

- **Impact: 3.5/5.** Large-object under-coverage is the #2 failure mode.
  flatfish_2 mIoU ≈ 0.09. But large objects are fewer frames → smaller
  impact on aggregate pf_mIoU.
- **Confidence: 3/5.** Asymmetric coverage penalty is theoretically sound
  (standard IoU loss is symmetric and can be satisfied by a small pred
  inside large GT). Risk: penalty weight 0.1 may be too small to matter
  or too large and cause pred_too_large explosion.
- **Cost: 4/5.** Loss-only change. No architecture changes. Minimal
  compute overhead (intersection computation already done for IoU).
- **Inference: 5/5.** Zero inference cost.
- **Overall: 3.6/5.** Good. Should stay at P3/P4.

#### P5 — COD Dense Pretraining

- **Impact: 3/5.** GreenVCOD uses COD pretraining but doesn't ablate it.
  Community evidence suggests +1-3% on image COD benchmarks. Transfer to
  video camouflage is uncertain — static camouflage (texture matching) vs
  video camouflage (motion detection) are different tasks.
- **Confidence: 2/5.** This is the riskiest direction. COD datasets
  (COD10K: 10K images, CAMO: 2.5K, NC4K: 4K) are image-based. The
  temporal dimension is lost in pretraining. Negative transfer is a real
  possibility — the model could learn to rely on texture patterns that
  don't generalize to motion-based camouflage.
- **Cost: 1.5/5.** Highest engineering cost. Requires: downloading and
  preprocessing 3 COD datasets, adapting dataset adapter for image-only
  data, implementing two-phase training loop, disk space for 16K+ images.
  Compute: 10-20 epochs pretraining + 30ep fine-tuning ≈ 40-50 epochs
  total per run.
- **Inference: 5/5.** Zero inference cost (same model).
- **Overall: 2.6/5.** Weakest link in current P1-P7. Should be demoted.

**Verdict: DEMOTE from P5 to P7 (reserve).** COD pretraining is worth
keeping on the roadmap as a reserve direction, but it should not block
simpler, cheaper, higher-confidence directions. Revisit after P1-P6
results are in — if zero-cost interventions saturate, the engineering
investment becomes more justified.

#### P6 — Temporal Dilation

- **Impact: 3.5/5.** Fast-motion degradation is a real failure mode
  (Phase 2 audits show some videos have rapid, erratic motion). But the
  fraction of affected frames is unknown — we need to quantify before
  prioritizing. GreenVCOD validates temporal context matters.
- **Confidence: 3.5/5.** Temporal stride > 1 is a well-established
  technique in video understanding. It's conceptually simple: sample
  every Nth frame to capture longer motion patterns without increasing
  memory. Risk: stride > 1 reduces effective frame rate and may miss
  brief motion cues.
- **Cost: 4.5/5.** Single config parameter change. No code changes needed
  (temporal_stride already exists in dataset). Zero architecture impact.
- **Inference: 5/5.** Zero inference cost (same T, same model).
- **Overall: 4.0/5.** Comparable to zoom and coverage, but simpler to
  test. Should be PROMOTED.

**Verdict: PROMOTE from P6 to P5.** This is the cheapest experiment on
the roadmap. One config parameter. 8ep probe is ~2 GPU-hours on RTX 4090.
There's no reason to defer it behind COD pretraining (40-50 epochs of
engineering-heavy work).

#### P7 — EffB0 → MV3-Small Distillation

- **Impact: 4/5.** If successful, gets EffB0-quality predictions at
  MV3-Small cost. The teacher-student gap is +0.0147 pf_mIoU — modest but
  meaningful. Directly serves the "lighter, faster, more accurate" charter.
- **Confidence: 2/5.** Distillation for dense prediction tasks (bbox
  regression) is less established than for classification. Standard KL
  distillation on bbox coordinates may not capture what makes EffB0 better
  (likely better feature representations, not just better coordinate
  outputs). Feature-level distillation (FPN feature alignment) is more
  promising but more complex.
- **Cost: 1.5/5.** Requires: loading frozen teacher model during training
  (2× memory), designing distillation loss, tuning distillation weight,
  30ep training. Engineering: moderate. Compute: ~1.5× per-iteration cost
  (teacher forward pass).
- **Inference: 5/5.** Zero inference cost (student-only at inference).
- **Overall: 2.9/5.** High-impact if it works, low confidence it will
  work with naive bbox-level distillation. Should stay as a reserve
  direction.

**Verdict: DEMOTE from P7 to P8 (reserve, post-P1-P6).** Distillation
should not be attempted until: (a) the EffB0 teacher is fully
characterized (what specifically makes it better?), and (b) simpler
zero-cost interventions are exhausted. If P1-P4 close most of the
MV3-Small/EffB0 gap, distillation becomes unnecessary.

### 3.4 New Directions from GreenVCOD Re-Analysis

Two directions emerge from the paper re-interpretation that are NOT in
the current roadmap:

#### NEW-A: Stronger Dense Supervision Target

**Rationale:** GreenVCOD's biggest gain comes from dense prediction maps.
dualVCOD's `dense_fg_aux` uses a hard binary mask (1 inside bbox, 0
outside). A hard mask is a crude approximation — it treats all pixels
inside the bbox as equally "foreground" when in reality the object
occupies only a fraction of the bbox. A **soft mask** (e.g., Gaussian
centered on bbox center, or bbox-shaped but with edge softening) could
provide a more informative training signal.

**Why not in current roadmap:** The current roadmap focuses on bbox-level
interventions (zoom, coverage). Improving the dense supervision target
is a natural extension of the dense_fg_aux principle that GreenVCOD
validates.

**Cost:** Medium. Requires: designing soft mask generation, modifying
`DenseForegroundLoss` target construction, tuning softness parameters.
Training-only. Zero inference cost.

**Priority:** Should slot between P4 (coverage) and P5 (temporal dilation).
It's simpler than COD pretraining, builds on existing infrastructure, and
is directly validated by GreenVCOD's strongest result.

#### NEW-B: Temporal Bbox Smoothing (Lightweight Post-Processing)

**Rationale:** GreenVCOD uses XGBoost for prediction refinement. Most of
XGBoost's features are temporal (bbox size change, center shift between
frames). A minimal version — exponential moving average of bbox
coordinates across consecutive frames — could capture the same temporal
smoothing benefit with zero learned parameters and negligible inference
cost.

**Why not in current roadmap:** The roadmap has no post-processing
directions (by design — bbox-only, end-to-end). But temporal smoothing
is not "adding a model component" — it's a 3-line EMA on bbox outputs.
It doesn't break differentiability (applied at inference only).

**Cost:** Minimal. EMA smoothing on bbox coordinates. No training changes.
Zero parameter cost. Negligible FPS impact.

**Priority:** Reserve direction. Test only if P6 (temporal dilation)
shows temporal context helps but doesn't fully close the fast-motion gap.

### 3.5 Revised Priority Ladder

```
P0: Governance & Infrastructure (CURRENT — no training)
P1: MV3-Small zoom probe (1ep→8ep)
P2: EffB0 zoom probe (8ep)
P3: MV3-Small coverage probe (1ep→8ep)
P4: EffB0 coverage probe (8ep)
P5: Temporal dilation (8ep)                          ← PROMOTED from P6
P6: Stronger dense supervision target (8ep)          ← NEW from GreenVCOD
P7: COD dense pretraining (reserve)                  ← DEMOTED from P5
P8: EffB0 → MV3-Small distillation (reserve)         ← DEMOTED from P7
Reserve: Temporal bbox smoothing                     ← NEW, test only if P5/P6 show temporal gains
```

### 3.6 Justification for Re-Ordering

1. **P1-P4 unchanged.** Zoom and coverage are the two highest-confidence,
   zero-inference-cost directions targeting the two largest failure modes.
   MV3-Small first (cheaper compute) gates EffB0 follow-up. No reason to
   change.

2. **P5 (temporal dilation) promoted above COD.** Temporal dilation is
   cheaper (8ep, ~2 GPU-hours vs 40-50 epochs), simpler (one config
   parameter vs multi-dataset engineering), and directly validated by
   GreenVCOD's finding that temporal context matters. It should not be
   blocked behind a complex, uncertain direction.

3. **P6 (stronger dense supervision) added.** GreenVCOD's strongest single
   result is the value of dense prediction maps. Improving our existing
   `dense_fg_aux` target quality is a natural, high-leverage extension.
   It costs less than COD pretraining and has stronger theoretical
   grounding.

4. **P7 (COD pretraining) demoted to reserve.** The engineering cost is
   high, the marginal gain above strong spatial + temporal baselines is
   unknown (GreenVCOD doesn't ablate it), and negative transfer is a real
   risk. Revisit when P1-P6 results are in.

5. **P8 (distillation) demoted to reserve.** Should only be attempted
   after: (a) EffB0 advantage is fully characterized (what specific error
   types does it fix?), (b) P1-P6 have closed as much of the gap as
   possible, (c) we have a specific hypothesis about what knowledge to
   transfer (bbox coordinates? features? attention maps?).

---

## 4. Revised Next-Phase Exploration Plan

### Phase A: Governance (Est. 1-2 sessions, no GPU)

**Complete P0 tasks:**
1. Archive E-52 (checkpoint, config, unified reeval, final report).
2. Validate `reeval_checkpoints.py` on E-51 or E-52 existing checkpoints.
3. Patch docs/01 and docs/03 to reflect revised priority (see §5).
4. Smoke-test E-54 (zoom) — 1 epoch, verify no NaN, zoom triggers.
5. Smoke-test E-55 (coverage) — 1 epoch, verify penalty behavior.

**Go/No-Go Gate:** All 5 governance tasks complete → enter Phase B.

### Phase B: Tiny-Object Precision (Est. 2-4 GPU-days)

**P1: MV3-Small Zoom Probe**
- 1ep smoke → if stable, 8ep probe vs E-45 (MV3-Small 8ep baseline).
- Success: IoU_tiny ≥ 0.55, pygmy_seahorse_0 +0.05, pf_mIoU drop ≤ 0.01.
- If 8ep passes: 30ep full run vs E-40.
- If 8ep fails: CLOSE zoom for MV3-Small. Write no-go report. Proceed to
  P3 (skip P2 — if zoom doesn't help MV3-Small, unlikely to help EffB0).

**P2: EffB0 Zoom Probe (only if P1 passes 8ep)**
- 8ep probe vs E-51 (EffB0 8ep baseline).
- Same success/no-go criteria referenced to E-51.
- If passes: decide on 30ep follow-up based on magnitude of gain.

### Phase C: Large-Object Coverage (Est. 2-4 GPU-days)

**P3: MV3-Small Coverage Probe**
- 1ep smoke → if stable, 8ep probe vs E-45.
- Success: IoU_large +0.02, flatfish_2 +0.03, pred_too_small decreases,
  pred_too_large does NOT increase > 20%.
- If 8ep passes: 30ep full run vs E-40.
- If 8ep fails: CLOSE coverage for MV3-Small. Proceed to P5 (skip P4).

**P4: EffB0 Coverage Probe (only if P3 passes 8ep)**
- 8ep probe vs E-51.
- Same success/no-go criteria.

### Phase D: Temporal & Dense Supervision (Est. 2-3 GPU-days)

**P5: Temporal Dilation**
- 8ep probe. Test stride ∈ {2, 3}. T=5 fixed.
- Baseline: E-45 (MV3-Small, stride=1, 8ep).
- Success: improvement on fast-motion videos, no degradation on static
  camouflage (pygmy_seahorse_0, flatfish_2).
- If passes: 30ep follow-up with best stride.

**P6: Stronger Dense Supervision Target**
- 8ep probe. Test: Gaussian soft mask vs hard binary mask in dense_fg_aux.
- Baseline: E-45 (hard binary mask, 8ep).
- Success: pf_mIoU improvement OR size-bin IoU improvement without
  regression elsewhere.
- If fails: CLOSE. Hard binary mask is sufficient.

### Phase E: Reserve Directions (Only if P1-P6 Saturate)

**P7: COD Pretraining** — revisit only if:
- P1-P6 combined gains < 0.01 pf_mIoU, AND
- A specific failure mode remains that COD pretraining could plausibly fix
  (generalization to unseen camouflage patterns), AND
- COD datasets are available on disk.

**P8: Distillation** — revisit only if:
- EffB0 advantage over MV3-Small is fully characterized, AND
- Feature-level distillation (not bbox-level KL) is the approach, AND
- P1-P6 gains on MV3-Small have been maximized first.

**Temporal Bbox Smoothing** — test only if:
- P5 (temporal dilation) shows temporal context helps but doesn't fully
  close the fast-motion gap.

---

## 5. Docs Patch Assessment

Two docs need minimal patches to reflect the revised priorities:

### 5.1 docs/01_CURRENT_TASK.md — Patch Required

**Changes:**
- Priority queue table (§2): reorder P5→P6→P7→P8. Add NEW P6 (stronger
  dense supervision). Renumber P7 (COD, reserve), P8 (distill, reserve).
- Add one-line entries for the two new reserve directions.
- No structural changes. No changes to P0-P4, experiment design rules,
  negative space, or E-54/E-55/E-56 positioning.

**Patch size:** ~8 lines changed in priority table + ~4 lines added for
new directions.

### 5.2 docs/03_EXPERIMENT_ROADMAP.md — Patch Required

**Changes:**
- Priority ladder diagram: reorder to match revised P0-P8.
- P5 entry: replace COD pretraining with temporal dilation (move temporal
  dilation content from old P6, update priority label).
- P6 entry: NEW — stronger dense supervision target (write from scratch:
  why, target error, baseline, single variable, probe plan, success/no-go,
  inference cost, rollback).
- P7 entry: COD pretraining, relabeled as "Reserve" with explicit
  conditions for activation.
- P8 entry: distillation, relabeled as "Reserve" with explicit conditions.
- Reserve entry: temporal bbox smoothing (2-3 lines).
- No changes to P0-P4 entries. No changes to design principles or gating
  rules.

**Patch size:** ~40 lines changed/added. One new full entry (P6), two
relabeled entries (P7, P8), one new short entry (reserve).

### 5.3 Docs NOT Requiring Changes

- **docs/00_PROJECT_CHARTER.md** — Baselines, closed directions, charter
  unchanged. Priority ordering is a tactical detail, not charter-level.
- **docs/02_DEVELOPMENT_PROTOCOL.md** — Protocol, eval mandate, top-K
  standard unchanged.
- **docs/04_METRIC_AND_ERROR_TAXONOMY.md** — Metrics unchanged. If P6
  (stronger dense supervision) introduces a new loss term, the loss would
  be training-only and not affect the metric taxonomy.

---

## 6. What This Report Does NOT Do

- Does NOT launch any training runs.
- Does NOT run E-54/E-55/E-56 on real data.
- Does NOT make performance claims about unimplemented methods.
- Does NOT revive E-53a, E-53b, bg_mix, soft_bbox, Center+Extent,
  MV3-Large, or any closed direction.
- Does NOT propose copying GreenVCOD's EfficientNet-B4, XGBoost, or
  segmentation inference.
- Does NOT change any code or config files.
- Does NOT claim the revised priority is final — it is a recommendation
  for review.

---

## 7. Summary

| Question | Answer |
|----------|--------|
| Absorb from GreenVCOD? | Dense spatial supervision primacy, temporal context value, frame-level evaluation, COD pretraining as reserve (not priority). |
| Reject from GreenVCOD? | EffB4 backbone, XGBoost refinement, segmentation inference, teacher-network supervision. |
| Priority changes? | Temporal dilation P5←P6. COD pretraining P5→P7 (reserve). Distillation P7→P8 (reserve). NEW: stronger dense supervision at P6. NEW: temporal bbox smoothing as reserve. |
| P1-P4 unchanged? | Yes. Zoom and coverage remain highest-confidence, zero-cost directions for the two largest failure modes. |
| Docs patches? | docs/01: priority table reorder (~8 lines). docs/03: P5/P6/P7/P8 rewrite + new entries (~40 lines). docs/00/02/04: no changes. |
| E-54/E-55/E-56 status? | Implementation complete, NOT validated. Smoke tests required before 8ep probes. |
| Next action? | Complete P0 governance (E-52 archive, top-K validation, docs patches). Then smoke tests. Then P1 8ep probe. |
