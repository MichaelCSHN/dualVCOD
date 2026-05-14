# Docs Patch Report — Post-GreenVCOD Strategic Review — 2026-05-14 10:27

## Context

Following the amended GreenVCOD strategic review (report_20260514_1007),
docs/01 and docs/03 were patched to reflect the revised P0-P8 priority
ladder. The original 0959 report was marked as superseded. No training
was run. No experimental code was modified.

---

## 1. Original 0959 Report — Superseded

**File**: `reports/report_20260514_0959_greenvcod_strategic_review.md`

A prominent superseded banner was added at the top of the file (after the
title, before the body). The banner:

- Declares the report SUPERSEDED in both English and Chinese
- Links to the amended report `report_20260514_1007_greenvcod_strategy_amended.md`
- Lists the three specific errors in the original: (1) describing
  pixel-wise prediction maps as a "dense prediction teacher",
  (2) implying component-level ablation for dense prediction,
  (3) overstating GreenVCOD as comprehensive SOTA / mIoU winner
- Instructs readers to use the amended report as the authoritative reference

The original report body is preserved intact for history. The banner
ensures any future agent or reader sees the superseded status before
reading the inaccurate content.

---

## 2. docs/01_CURRENT_TASK.md — Changes

**Section modified**: §2 Priority Queue (P1-P7 → P1-P8)

| Change | Detail |
|--------|--------|
| Section heading | `P1-P7` → `P1-P8` |
| P5 (new) | Temporal dilation — promoted from old P6. T=5 fixed, stride ∈ {2,3}, zero params, zero architecture change. Motivated by GreenVCOD long-term temporal context validation, not copying GreenVCOD ensemble. |
| P6 (NEW) | Dense target refinement — marked CAUTIOUS PROBE. Modifies only training-only `dense_fg_aux` BCE target at existing stride-8. Bbox regression unchanged. Explicitly distinguished from soft_bbox, adaptive Gaussian softening, and E-53a. Strict no-go: pf_mIoU drop >0.01, tiny IoU drop >0.02, pygmy_seahorse_0 drop >0.03, any size bin drop >0.03, pred_too_large increase >20%, NaN/loss instability. |
| P7 (relabeled) | COD dense pretraining — demoted from old P5 to RESERVE. Activation condition: only if P1-P6 saturate and a failure mode plausibly requires image-COD spatial priors. Acknowledges GreenVCOD uses COD pretraining but does not isolate its contribution. |
| P8 (relabeled) | EffB0 → MV3-Small distillation — demoted from old P7 to RESERVE. Activation condition: only after EffB0/E-52 advantage fully characterized and MV3-Small zero-cost interventions exhausted. Prefer feature-level/dense-logit guidance over naive bbox KL if attempted. |

**Unchanged**: P0-P4 entries, E-54/E-55/E-56 positioning, experiment design
rules (§3), negative space (§4), and all other sections.

---

## 3. docs/03_EXPERIMENT_ROADMAP.md — Changes

**Sections modified**: Priority ladder, P5, P6, P7 (all replaced).

| Change | Detail |
|--------|--------|
| Priority ladder | P0-P7 → P0-P8. P5-P8 reordered: temporal dilation → dense target refinement → COD reserve → distillation reserve. P2/P4 annotated "only if P1/P3 passes." |
| P5 (rewritten) | Old COD pretraining replaced with temporal dilation (content adapted from old P6, expanded). Explicitly states this is NOT a GreenVCOD short/long ensemble copy. Baseline narrowed to E-45 (MV3-Small, 8ep). Success/no-go criteria expanded: bad_frame_rate, center_error, size-bin degradation checks. Rollback plan added. |
| P6 (NEW) | Dense target refinement — full entry (~70 lines). Includes: motivation referencing GreenVCOD but not overclaiming; distinction table vs soft_bbox / adaptive Gaussian softening / E-53a; explicit "What this is NOT" list (no multi-scale, no stride-4, no segmentation, no new heads, no forward pass change); 1ep smoke → 8ep probe plan; quantified success criteria; 7-item no-go list with E-53a signature explicitly flagged; zero inference cost; config-gated rollback. Marked "CAUTIOUS PROBE" in header. |
| P7 (relabeled) | Old P5 COD pretraining content preserved and relabeled as RESERVE. Added: status declaration, activation condition (P1-P6 saturate + specific failure mode), acknowledgment that GreenVCOD does not isolate COD pretraining contribution. |
| P8 (relabeled) | Old P7 distillation content preserved and relabeled as RESERVE. Added: status declaration, activation condition (EffB0 characterized + MV3-Small zero-cost saturated), guidance to prefer feature-level / dense-logit over naive bbox KL. |

**Unchanged**: P0-P4 entries (full content preserved), design principles,
1ep→8ep→30ep gating rules, backbone parallel-track policy.

---

## 4. Why Temporal Dilation Was Promoted to P5

Three reasons:

1. **Cost**: Temporal dilation is the cheapest experiment on the roadmap.
   One config parameter (`temporal_stride`). Zero code changes. 8ep probe
   is ~2 GPU-hours on RTX 4090. No architecture, params, or inference
   changes.

2. **GreenVCOD validation**: GreenVCOD's ablation (short-term vs long-term
   vs ensemble) is the most actionable finding for dualVCOD — it
   independently validates that longer temporal context improves video
   camouflage detection. This doesn't mean copying GreenVCOD's dual-branch
   architecture; it means testing the principle with the lightest possible
   intervention.

3. **Dependency ordering**: There's no reason to block a 2-GPU-hour
   experiment behind COD pretraining (40-50 epochs of engineering-heavy
   work with uncertain payoff). Cheap, high-confidence probes should
   precede expensive, low-confidence ones.

---

## 5. Why Dense Target Refinement Is Only a Cautious Probe (P6)

Four reasons:

1. **GreenVCOD does NOT prove soft targets work.** GreenVCOD uses
   pixel-wise prediction maps as a structural component, but never
   ablates hard vs soft targets. The paper validates that dense spatial
   representation matters — not that Gaussian targets beat binary masks.

2. **Softening has failed before in dualVCOD.** soft_bbox (Gaussian bbox
   regression targets) and adaptive Gaussian softening (size-dependent σ)
   were both tried and closed in Phase 2. While P6 modifies a different
   target (auxiliary BCE mask, not bbox regression), the common thread —
   softening spatial targets for camouflage detection — has a negative
   track record.

3. **The distinction from closed directions is real but narrow.** P6
   modifies only the training-only auxiliary head at existing resolution.
   The bbox head still receives hard targets. The failure mode of E-53a
   (multi-scale → catastrophic tiny-object degradation) is explicitly
   checked in P6's no-go criteria. But the risk of similar degradation
   is non-zero.

4. **One 8ep run is the right cost for a principled probe.** If it works:
   a potentially valuable zero-inference-cost improvement. If it fails
   (which is more likely given the track record): the direction is CLOSED
   with clear evidence, and hard binary masks are validated as sufficient.
   Either outcome is worth 2 GPU-hours.

---

## 6. Why COD Pretraining and Distillation Were Demoted to Reserve

### COD Pretraining (P7 RESERVE)

- GreenVCOD uses COD pretraining but does NOT isolate its contribution
  in any ablation. The marginal gain above strong spatial + temporal
  baselines is unknown.
- Engineering cost is the highest on the roadmap: download and preprocess
  3 datasets (COD10K, CAMO, NC4K), adapt data loader for image-only mode,
  implement two-phase training loop.
- Compute cost is high: 10-20 epochs pretraining + 30ep fine-tuning =
  40-50 epochs per run.
- Risk of negative transfer: static image camouflage (texture matching)
  may not transfer to video camouflage (motion detection).
- **Activation condition**: P1-P6 zero-cost interventions must be
  exhausted first. If they close the gap, COD pretraining is unnecessary.
  If they don't, the remaining failure modes will indicate whether COD
  pretraining is the right next step.

### Distillation (P8 RESERVE)

- Distillation for bbox regression is less established than for
  classification. Naive KL on bbox coordinates is unlikely to transfer
  what makes EffB0 better (likely feature quality, not coordinate outputs).
- Requires 2× GPU memory during training (teacher + student).
- The teacher-student gap is modest (+0.0147 pf_mIoU). If P1-P6 close
  even half of this gap on MV3-Small, the residual gain from distillation
  may not justify the engineering cost.
- **Activation condition**: EffB0 advantage must be fully characterized
  first — what specific error types does it fix? Only then can a targeted
  distillation approach (feature-level, dense-logit matching) be designed.
  And P1-P6 MV3-Small gains must be maximized first.

---

## 7. Why No Training Experiments Were Run

1. **Governance first**: The P0 governance tasks are not yet complete.
   E-52 archive, top-K validation, and composite score calibration are
   prerequisites for any new experiment.

2. **E-54/E-55/E-56 not smoke-tested**: The zoom and coverage code is
   implementation-complete but has not been validated on real data.
   Running 8ep probes without 1ep smoke tests risks wasting GPU-hours
   on code bugs.

3. **Top-K infrastructure not validated**: `reeval_checkpoints.py` has
   not been run on existing checkpoints. Launching new experiments
   without validated eval infrastructure would risk unreliable checkpoint
   selection.

4. **Docs consistency not yet verified**: The patches in this commit
   need review before new experiments reference them.

---

## 8. Files Changed in This Commit

| File | Action | Lines Changed |
|------|--------|--------------|
| `reports/report_20260514_0959_greenvcod_strategic_review.md` | Superseded banner added | +10 |
| `docs/01_CURRENT_TASK.md` | Priority table P1-P7 → P1-P8, new P5-P8 descriptions | ~30 |
| `docs/03_EXPERIMENT_ROADMAP.md` | Priority ladder + P5/P6/P7/P8 rewritten | ~120 |
| `reports/report_20260514_1027_docs_patch_after_greenvcod_review.md` | This report (NEW) | ~200 |

**Not modified**: `docs/00_PROJECT_CHARTER.md`, `docs/02_DEVELOPMENT_PROTOCOL.md`,
`docs/04_METRIC_AND_ERROR_TAXONOMY.md`, any source code or config files.

---

## 9. Next Steps (Recommendation)

1. **Review the docs patches** — confirm P5-P8 ordering and P6 no-go
   criteria match the strategic intent.
2. **Complete P0 governance**:
   - Archive E-52 with final report
   - Validate `topk_checkpoints` + `reeval_checkpoints.py` on E-51 or
     E-52 existing checkpoints
3. **Smoke-test E-54 (zoom)** — 1 epoch, verify no NaN, zoom triggers
4. **Smoke-test E-55 (coverage)** — 1 epoch, verify penalty behavior
5. **Launch P1 8ep probe** (MV3-Small zoom) — only after all above complete
