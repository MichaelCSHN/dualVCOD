# GreenVCOD Strategic Re-Analysis & Revised Exploration Plan (Amended) — 2026-05-14 10:07

**Amends report_20260514_0959.** Corrections to GreenVCOD technical description,
ablation claims, and SOTA characterization. Revised P6 (dense target refinement)
with explicit distinction from closed soft_bbox/adaptive softening and no-go
conditions. Docs patch recommendations updated.

---

## 1. GreenVCOD Re-Interpretation: What the Paper Actually Contains

GreenVCOD (arXiv:2501.10914v1) is a video camouflage object detection
system. Its pipeline has four stages:

1. **EfficientNet-B4 backbone** extracts per-frame features
2. **Cascaded multi-resolution XGBoost** produces **pixel-wise prediction
   maps** — a dense spatial representation that localizes camouflage at
   the pixel level before bbox regression
3. **Temporal neighborhood prediction cube** aggregates per-frame
   prediction maps over short-term and long-term windows via
   order-invariant pooling, producing a spatiotemporal representation
4. **Temporal XGBoost refinement** post-processes bbox predictions using
   hand-crafted temporal features (bbox size change, center drift, etc.)

The core technical idea is **prediction-map based dense spatial
representation** — converting backbone features into per-pixel camouflage
scores before temporal aggregation and bbox regression. This is NOT a
teacher-student distillation setup; GreenVCOD trains the pixel-wise
XGBoost directly as part of its pipeline. There is no separate "teacher"
model.

### 1.1 Performance Context

GreenVCOD Table 1 reports results on MoCA-Mini against prior methods
(SLT-Net, ZoomNet, etc.):

- **Best on**: MAE (Mean Absolute Error) and enhanced-alignment measure
- **mIoU**: Competitive but **below SLT-Net** on mIoU
- **Efficiency**: Strong — achieves competitive results with lower
  computational cost than SLT-Net

GreenVCOD is best characterized as **competitive and efficient**, not as
comprehensive SOTA. It does NOT claim +2.3% mIoU over all methods. The
headline gains are on MAE and alignment metrics, not on mIoU.

### 1.2 Ablation Context

The paper's ablation (Table 2 or equivalent) compares:
- Short-term temporal neighborhood only
- Long-term temporal neighborhood only
- Ensemble (short-term + long-term)

The ablation isolates temporal modeling choices, NOT the contribution of
individual components (backbone, prediction maps, XGBoost refinement).
Table 3 is a complexity/efficiency comparison, not a component ablation.

**There is no published ablation that isolates the contribution of COD
pretraining, pixel-wise prediction maps, or XGBoost refinement
individually.** The paper does not provide the data to attribute gain to
any single component beyond temporal neighborhood design.

### 1.3 What dualVCOD Already Does

| GreenVCOD Component | dualVCOD Status | Assessment |
|---------------------|-----------------|------------|
| Short-term temporal neighborhood | `TemporalNeighborhood` with order-invariant mean pooling, T=5 | Equivalent in function, simpler in design |
| Pixel-wise dense prediction maps | `dense_fg_aux` — BCE dense foreground head on stride-8 features, training-only | Similar principle (dense spatial representation as intermediate signal) but different mechanism (binary mask target vs XGBoost pixel scores) |
| COD pretraining | Not implemented | Exists in GreenVCOD but contribution is not isolated |

### 1.4 What dualVCOD Should NOT Copy

| Component | Rationale for Rejection |
|-----------|------------------------|
| EfficientNet-B4 backbone | 19M params. dualVCOD's EffB0 (5.3M) already provides a meaningful upgrade over MV3-Small (2.5M). B4 violates "lighter, faster" charter. |
| Cascaded XGBoost pixel-wise prediction | Non-differentiable, requires separate XGBoost training per resolution level. Adds inference cost and complexity. `dense_fg_aux` achieves the dense spatial supervision principle with a trainable, end-to-end BCE head. |
| Temporal XGBoost refinement | 80+ hand-crafted features, separate training pipeline. Inference cost increase. Breaks end-to-end differentiability. |
| Segmentation-based evaluation | dualVCOD charter: bbox-only at inference. No segmentation output. |

### 1.5 Principles dualVCOD Should Absorb

**Principle 1: Dense spatial representation is structurally central to
GreenVCOD's pipeline.**

The pixel-wise prediction maps convert abstract backbone features into
explicit spatial localization signals before temporal aggregation. This
validates dualVCOD's `dense_fg_aux` as a load-bearing design choice —
dense spatial supervision provides a richer training signal than bbox
regression alone. Whether a stronger form of this signal (beyond hard
binary masks) would help is an open question, not a settled conclusion.

**Principle 2: Long-term temporal context matters, and the form is
flexible.**

GreenVCOD's ablation shows that combining short-term and long-term
temporal neighborhoods outperforms either alone. This validates temporal
context as important without prescribing a specific architecture.
dualVCOD can probe longer temporal windows via `temporal_stride > 1`
(keeping T=5, sampling every Nth frame) — a zero-parameter,
zero-inference-cost experiment.

**Principle 3: Frame-level spatial quality drives overall performance.**

GreenVCOD computes and reports per-frame metrics. This validates
dualVCOD's `pf_mIoU` as the primary metric and the frame-level error
type taxonomy (pred_too_large, pred_too_small, center_shift,
scale_mismatch).

**Principle 4: Simple interventions dominate complex ones.**

The most impactful design choice in GreenVCOD (dense spatial
representation) is a training-time mechanism, not an inference-time
addition. The most complex component (temporal XGBoost refinement) has
no isolated ablation. This reinforces dualVCOD's design philosophy:
zero-inference-cost, single-variable interventions first.

**Principle 5: COD pretraining is present but unquantified.**

COD pretraining (COD10K, CAMO, NC4K) is part of GreenVCOD's pipeline but
its marginal contribution is not isolated in any ablation. The
engineering cost of integrating three COD datasets is non-trivial. It
belongs on the roadmap as a reserve direction, not a near-term priority.

### 1.6 The Real Lesson

GreenVCOD's value to dualVCOD is NOT its specific architecture. It is
**independent validation of design principles** dualVCOD already follows:
dense spatial representation as intermediate signal, temporal aggregation
for motion-based camouflage, and frame-level evaluation. The paper's
unexplored potential lies in what it did NOT isolate: how much does COD
pretraining help above a strong spatial+temporal baseline? How much does
long-term temporal context add beyond short-term alone? Can a simpler
refinement (lightweight temporal smoothing) achieve most of what XGBoost
does?

These are the questions dualVCOD should answer — not by copying
GreenVCOD's components, but by testing the principles they imply.

---

## 2. E-54 / E-55 / E-56 Repositioning

(Unchanged from original report. Reproduced for completeness.)

| ID | File | What It Does | Config Gate |
|----|------|-------------|-------------|
| E-54 | `src/dataset_real.py` | Scale-aware natural zoom augmentation | `zoom_enabled=false` (default) |
| E-55 | `src/loss.py` | Large-object asymmetric coverage penalty | `loss_weights.large_coverage=0.0` (default) |
| E-56 | `run_trial_minimal.py` + `reeval_checkpoints.py` | Top-K checkpoint saving + unified reeval | `topk_checkpoints=1` (default) |

**Status**: Implementation complete, NOT experimentally validated.
- Code compiles. Synthetic unit tests pass.
- All defaults are zero/off — no impact on existing pipelines.
- Subject to revision based on smoke test results.
- Not to be cited as validated methods.

---

## 3. Priority Re-Evaluation (Corrected)

### 3.1 Assessment Criteria

| Dimension | Description |
|-----------|-------------|
| **Impact** | Expected pf_mIoU gain if successful |
| **Confidence** | Likelihood of success given theory + Phase 2 evidence + GreenVCOD context |
| **Cost** | Engineering + compute cost (higher score = lower cost) |
| **Inference** | Zero inference cost = 5, adds params/FPS = lower |

### 3.2 Per-Direction Assessment

#### P1/P2 — Zoom Probe (MV3-Small / EffB0)

**Assessment unchanged.** Tiny objects are the #1 failure mode.
Zero-inference-cost augmentation. GreenVCOD doesn't use zoom but
validates multi-scale processing value.

**Verdict: Keep at P1/P2.**

#### P3/P4 — Coverage-Aware Loss (MV3-Small / EffB0)

**Assessment unchanged.** Large-object under-coverage is #2 failure mode.
Theoretically sound asymmetric penalty. Zero inference cost.

**Verdict: Keep at P3/P4.**

#### P5 (old) → COD Dense Pretraining

- **Impact: 3/5.** GreenVCOD uses it but contribution is not isolated.
  Community evidence: +1-3% on image COD benchmarks. Transfer to video
  camouflage uncertain (static texture vs motion cues).
- **Confidence: 2/5.** Highest uncertainty. No ablation to quantify
  marginal gain. Negative transfer risk (static COD patterns may not
  generalize to motion-based camouflage).
- **Cost: 1.5/5.** Highest engineering cost: download 3 datasets, adapt
  data loader for image-only mode, implement two-phase training, 40-50
  epochs total.
- **Inference: 5/5.** Zero inference cost (same model).
- **Overall: 2.6/5.**

**Verdict: DEMOTE to reserve (P7).**

#### P6 (old) → Temporal Dilation

- **Impact: 3.5/5.** GreenVCOD's ablation validates long-term temporal
  context improves results. Temporal stride is the cheapest possible
  probe — one config parameter.
- **Confidence: 3.5/5.** Well-established technique in video
  understanding. GreenVCOD provides independent validation that longer
  temporal windows help for camouflage specifically.
- **Cost: 4.5/5.** Zero code changes. Single config parameter.
  8ep probe = ~2 GPU-hours.
- **Inference: 5/5.** Same T, same model.
- **Overall: 4.0/5.**

**Verdict: PROMOTE to P5.**

#### P7 (old) → Distillation

- **Impact: 4/5.** Gets EffB0-quality at MV3-Small cost — directly serves
  charter.
- **Confidence: 2/5.** Distillation for bbox regression is less
  established than for classification. Naive KL on bbox coordinates may
  not capture what makes EffB0 better. Feature-level distillation is more
  promising but more complex.
- **Cost: 1.5/5.** 2× memory during training (teacher + student), ~1.5×
  per-iteration compute, 30ep required.
- **Inference: 5/5.** Student-only at inference.
- **Overall: 2.9/5.**

**Verdict: DEMOTE to reserve (P8).** Should only be attempted after: (a)
EffB0 advantage is fully characterized (what specific error types does it
fix?), (b) P1-P6 zero-cost interventions are exhausted.

### 3.3 New Direction: Dense Target Refinement (Cautious Probe)

**Origin**: GreenVCOD's pixel-wise prediction maps suggest that **how**
dense spatial information is represented matters. dualVCOD's
`dense_fg_aux` currently uses a hard binary mask: 1.0 inside the GT bbox,
0.0 outside. This is a crude spatial target — it treats all pixels inside
the bbox as equally "foreground" when the object typically occupies only
a fraction of the bbox area.

**Hypothesis**: A refined dense supervision target (e.g., a 2D Gaussian
centered on the bbox, or a mask that decays from the bbox center) could
provide a more informative spatial training signal than a hard binary
mask, improving the backbone's spatial feature quality without changing
what the bbox head learns to output.

**Critical distinction from closed directions:**

| Closed Direction | What It Modified | Why It Failed |
|-----------------|-----------------|---------------|
| **soft_bbox** (Phase 2) | **BBox regression target** — replaced hard bbox coordinates with Gaussian-softened targets for the primary bbox head | Made the primary regression task harder: the model had to predict a distribution rather than a point estimate. Uncertainty was injected into the task the model is evaluated on. |
| **adaptive Gaussian softening** (Phase 2) | **BBox regression target** — adaptive σ for Gaussian bbox targets based on object size | Same failure mode as soft_bbox, with added complexity of size-dependent σ. |
| **E-53a multi-scale dense** (Phase 2, CLOSED) | **Multi-scale dense supervision** — added dense_fg heads at multiple FPN levels (stride-4 in addition to stride-8) | Catastrophically degraded tiny-object detection (pygmy_seahorse_0 mIoU dropped from ~0.4 to 0.008). Dense supervision at finer scales caused the model to overfit to high-resolution texture patterns. |

| **NEW: Dense target refinement** | **Auxiliary dense_fg_aux target only** — softens the BCE target mask for the training-only auxiliary head. BBox regression targets are UNCHANGED (hard DIoU/GIoU). Primary task unaffected. | **Hypothesized to be safer because**: (1) the auxiliary head is stripped at inference — softening only affects the training signal, not the evaluation target; (2) the bbox head still receives hard, precise regression targets; (3) the auxiliary head's job is to improve backbone features, not to produce precise outputs itself. |

**Why this might help where E-53a hurt**: E-53a added dense supervision
at finer FPN resolutions (stride-4, 56×56), which pulled the backbone
toward high-frequency texture features that don't generalize (especially
harmful for tiny, texture-sparse objects). Dense target refinement at
stride-8 (28×28) changes only the *target quality*, not the resolution
or architecture. It operates at the same spatial scale as the current
working design.

**Risk**: The closed directions above share a common thread — softening
spatial targets in any form may be fundamentally unhelpful for camouflage
detection, where precise boundaries are inherently ambiguous. If the hard
binary mask is already optimal (it provides a strong "object somewhere
in this box" signal without making fine-grained boundary claims), any
softening could degrade performance by making the auxiliary task either
too easy (less informative gradient) or too specific (encouraging the
model to predict boundaries it cannot learn).

**Probe design (conservative)**:

- **Single variable**: `dense_target_mode: "gaussian"` — replace hard
  binary mask with 2D Gaussian centered on bbox center, σ proportional
  to bbox size, normalized to peak at 1.0. Config-gated, default
  `"binary"`.
- **Baseline**: E-45 (MV3-Small, hard binary mask, 8ep).
- **1-epoch smoke**: Verify target generation, no NaN, loss magnitude
  comparable to binary baseline.
- **8-epoch probe**: Compare against E-45 under unified reeval.

**Success criteria (8ep)**:
- pf_mIoU does not drop > 0.01 vs baseline
- At least one of: IoU_tiny improvement ≥ 0.02, OR IoU_large improvement
  ≥ 0.02, OR center_error_mean decrease ≥ 0.005
- No size bin drops > 0.02

**No-Go criteria (any one triggers immediate CLOSE)**:
- pf_mIoU drops > 0.01
- **Tiny-object IoU degrades by > 0.02** — this is the E-53a signature;
  if it appears, the direction is CLOSED regardless of other metrics
- **pygmy_seahorse_0 mIoU drops by > 0.03** — canonical tiny-object
  probe; degradation here indicates softening harms small-object
  localization
- Any size bin drops > 0.03
- Training instability (NaN, loss spikes)
- **pred_too_large increases > 20%** — indicates the softer target is
  encouraging the bbox head to over-predict (the auxiliary signal is
  leaking into bbox behavior despite separate heads)

**Inference cost**: Zero. `dense_fg_aux` is training-only. No change to
model forward pass.

**Rollback**: Set `dense_target_mode: "binary"` in config. No code
removal needed — both modes coexist in the same code path.

**Verdict: Cautious probe at P6.** This is NOT a high-confidence
direction. It earns its place because: (a) it's cheap (8ep probe,
training-only, zero inference cost), (b) GreenVCOD validates the general
principle that dense spatial representation quality matters, and (c) the
no-go conditions are designed to catch E-53a-like degradation at the
8ep checkpoint before any 30ep investment. If it fails at 8ep, it is
CLOSED — one 8ep run is an acceptable cost for a principled probe.

### 3.4 Revised Priority Ladder

```
P0: Governance & Infrastructure (CURRENT — no training)
P1: MV3-Small zoom probe (1ep→8ep)
P2: EffB0 zoom probe (8ep, only if P1 passes)
P3: MV3-Small coverage probe (1ep→8ep)
P4: EffB0 coverage probe (8ep, only if P3 passes)
P5: Temporal dilation (8ep)                          ← PROMOTED from old P6
P6: Dense target refinement (CAUTIOUS, 1ep→8ep)      ← NEW, with strict no-go
P7: COD dense pretraining (RESERVE)                  ← DEMOTED from old P5
P8: EffB0 → MV3-Small distillation (RESERVE)         ← DEMOTED from old P7
```

### 3.5 Justification for Re-Ordering

1. **P1-P4 unchanged.** Zoom and coverage are the two highest-confidence,
   zero-inference-cost directions targeting the two largest failure modes.

2. **P5 (temporal dilation) promoted.** GreenVCOD's long-term temporal
   context ablation is the most actionable finding for dualVCOD. Temporal
   stride is the cheapest experiment on the entire roadmap — one config
   parameter, zero code changes, ~2 GPU-hours for 8ep. It should not be
   blocked behind complex, uncertain directions.

3. **P6 (dense target refinement) added as cautious probe.** GreenVCOD
   validates that dense spatial representation matters. Our current hard
   binary mask is crude. A soft target probe costs one 8ep run and has
   explicit no-go conditions modeled on E-53a's failure signature. If it
   fails → CLOSED. If it passes → a potentially high-impact,
   zero-inference-cost improvement.

4. **P7 (COD pretraining) demoted to reserve.** Highest engineering cost,
   most uncertain payoff, contribution not isolated in GreenVCOD.
   Revisit only if P1-P6 saturate.

5. **P8 (distillation) demoted to reserve.** Highest complexity.
   Requires EffB0 characterization and exhausted zero-cost alternatives
   before attempting.

---

## 4. Revised Next-Phase Exploration Plan

### Phase A: Governance (Est. 1-2 sessions, no GPU)

**Complete P0 tasks:**
1. Archive E-52 (checkpoint, config, unified reeval, final report).
2. Validate `reeval_checkpoints.py` on E-51 or E-52 existing checkpoints.
3. Patch docs/01 and docs/03 to reflect revised priority (see §5).
4. Smoke-test E-54 (zoom) — 1 epoch, verify no NaN, zoom triggers.
5. Smoke-test E-55 (coverage) — 1 epoch, verify penalty behavior.

**Gate**: All 5 tasks complete → Phase B.

### Phase B: Tiny-Object Precision (Est. 2-4 GPU-days)

**P1: MV3-Small Zoom Probe**
- 1ep smoke → 8ep probe vs E-45.
- Success: IoU_tiny ≥ 0.55, pygmy_seahorse_0 +0.05, pf_mIoU drop ≤ 0.01.
- If 8ep passes: 30ep vs E-40. If fails: CLOSE zoom for MV3-Small, skip P2.

**P2: EffB0 Zoom Probe** (only if P1 passes 8ep)
- 8ep probe vs E-51. Same criteria referenced to E-51.

### Phase C: Large-Object Coverage (Est. 2-4 GPU-days)

**P3: MV3-Small Coverage Probe**
- 1ep smoke → 8ep probe vs E-45.
- Success: IoU_large +0.02, flatfish_2 +0.03, pred_too_small decreases,
  pred_too_large does NOT increase > 20%.
- If 8ep passes: 30ep vs E-40. If fails: CLOSE coverage for MV3-Small, skip P4.

**P4: EffB0 Coverage Probe** (only if P3 passes 8ep)
- 8ep probe vs E-51. Same criteria.

### Phase D: Temporal & Dense Supervision (Est. 2-3 GPU-days)

**P5: Temporal Dilation**
- 8ep probe. Stride ∈ {2, 3}. T=5 fixed. Baseline: E-45.
- Success: improvement on fast-motion videos, no degradation on static
  camouflage (pygmy_seahorse_0, flatfish_2).
- If passes: 30ep follow-up with best stride.

**P6: Dense Target Refinement (CAUTIOUS)**
- 1ep smoke → 8ep probe. Gaussian soft mask vs hard binary mask.
  Baseline: E-45.
- Success: pf_mIoU stable, at least one spatial metric improves, no
  E-53a-like tiny-object regression.
- No-go (immediate CLOSE): tiny-object IoU drops > 0.02, pygmy_seahorse_0
  drops > 0.03, pred_too_large increases > 20%, any size bin drops > 0.03,
  pf_mIoU drops > 0.01.
- If 8ep fails: CLOSE dense target refinement. Hard binary mask is
  sufficient.

### Phase E: Reserve Directions (Only if P1-P6 Saturate)

**P7: COD Pretraining** — revisit only if P1-P6 combined gains < 0.01
pf_mIoU AND a specific failure mode remains that COD pretraining could
plausibly address.

**P8: Distillation** — revisit only if EffB0 advantage is fully
characterized AND feature-level distillation (not bbox-level KL) is the
approach AND P1-P6 MV3-Small gains are maximized.

---

## 5. Docs Patch Assessment (Revised)

### 5.1 docs/01_CURRENT_TASK.md — Patch Required

**Changes:**
- §2 Priority queue table: reorder to P0→P1→P2→P3→P4→P5→P6→P7→P8.
  - P5: Temporal dilation (promoted from old P6)
  - P6: Dense target refinement (NEW, marked as "cautious probe, strict no-go")
  - P7: COD pretraining (demoted, marked "RESERVE")
  - P8: Distillation (demoted, marked "RESERVE")
- Add P6 entry distinguishing it from closed soft_bbox / adaptive
  Gaussian softening / E-53a.
- Add reserve conditions for P7/P8.
- No changes to P0-P4, experiment design rules, negative space, or
  E-54/E-55/E-56 positioning.

**Patch size:** ~12 lines changed + ~6 lines added.

### 5.2 docs/03_EXPERIMENT_ROADMAP.md — Patch Required

**Changes:**
- Priority ladder diagram: reorder to P0→P1→P2→P3→P4→P5→P6→P7→P8.
- P5: Replace COD pretraining entry with temporal dilation (move temporal
  dilation content from old P6 section, update priority label).
- P6: NEW entry — dense target refinement. Must include:
  - Why: GreenVCOD validates dense spatial representation quality matters;
    current hard binary mask is crude
  - Distinction from closed directions: table showing soft_bbox,
    adaptive Gaussian softening, and E-53a all modified different things
    with different failure modes
  - Target error type: spatial accuracy (center_error, area_ratio)
  - Baseline: E-45
  - Single variable: `dense_target_mode` (binary → gaussian)
  - Probe plan: 1ep smoke → 8ep probe
  - Success criteria (as specified in §3.3 above)
  - No-go criteria (as specified in §3.3 above, with E-53a signature
    explicitly flagged)
  - Inference cost: Zero
  - Rollback: config flag
  - **Marked as "CAUTIOUS PROBE"** — not high-confidence, one 8ep run
    to test, immediate close on no-go
- P7: COD pretraining, relabeled "RESERVE" with activation conditions.
- P8: Distillation, relabeled "RESERVE" with activation conditions.
- No changes to P0-P4 entries, design principles, or gating rules.

**Patch size:** ~50 lines changed/added. One rewritten entry (P5), one
new cautious entry (P6), two relabeled entries (P7, P8).

### 5.3 Docs NOT Requiring Changes

- **docs/00_PROJECT_CHARTER.md** — Baselines, closed directions, charter,
  dense training-only docs unchanged. Priority ordering is tactical.
- **docs/02_DEVELOPMENT_PROTOCOL.md** — Protocol, eval mandate, top-K
  standard unchanged.
- **docs/04_METRIC_AND_ERROR_TAXONOMY.md** — Metric definitions unchanged.
  Dense target refinement is a training-only loss change; it does not
  add new metrics.

### 5.4 Important: What the Docs Patches Should NOT Do

- Do NOT claim GreenVCOD proves dense target refinement works
- Do NOT claim GreenVCOD achieves comprehensive SOTA on mIoU
- Do NOT reference a "dense prediction teacher" or "teacher ablation"
- Do NOT position P6 as high-confidence — it is explicitly a cautious
  probe with strict no-go
- Do NOT conflate P6 with closed soft_bbox / adaptive softening — the
  distinction must be explicit
- Do NOT remove or weaken the E-53a closed direction documentation

---

## 6. Summary

| Question | Amended Answer |
|----------|---------------|
| GreenVCOD's method? | EffB4 → cascaded XGBoost pixel-wise prediction maps → temporal prediction cube → temporal XGBoost refinement. NOT teacher-student. |
| GreenVCOD performance? | Competitive and efficient. Best on MAE and alignment; mIoU below SLT-Net. NOT comprehensive SOTA. |
| GreenVCOD ablation? | Short-term vs long-term vs ensemble temporal comparison. No component-level ablation isolating COD pretraining or XGBoost. |
| Absorb from GreenVCOD? | Dense spatial representation as structural principle; long-term temporal context value; frame-level evaluation; COD pretraining as reserve only. |
| Reject from GreenVCOD? | EffB4, XGBoost (pixel-wise + temporal), segmentation inference. |
| Priority changes? | P5: temporal dilation (promoted). P6: dense target refinement (NEW, cautious, strict no-go). P7: COD pretraining (reserve). P8: distillation (reserve). |
| P6 vs closed directions? | Dense target refinement modifies auxiliary BCE mask only, not bbox regression targets (soft_bbox) and not multi-scale architecture (E-53a). No-go explicitly checks for E-53a-like tiny-object degradation. |
| Docs patches? | docs/01: ~18 lines. docs/03: ~50 lines. docs/00/02/04: unchanged. |
| E-54/E-55/E-56? | Implementation complete, NOT validated. Smoke tests before 8ep probes. |
