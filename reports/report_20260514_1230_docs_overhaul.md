# Documentation Overhaul Report — 2026-05-14 12:30

## Context

Post-Phase-2 governance pass. All five project docs rewritten or created
to reflect current state: Phase 2 exploration complete, entering Next-Phase
Precision Engineering. No training was run. No experimental claims were made.

## Files Changed

| File | Action | Lines |
|------|--------|-------|
| `docs/00_PROJECT_CHARTER.md` | Revised | 140 (+ expanded from ~54) |
| `docs/01_CURRENT_TASK.md` | Rewritten | 109 (+ expanded from ~44) |
| `docs/02_DEVELOPMENT_PROTOCOL.md` | Revised | 92 (+ expanded from ~37) |
| `docs/03_EXPERIMENT_ROADMAP.md` | **NEW** | 256 |
| `docs/04_METRIC_AND_ERROR_TAXONOMY.md` | **NEW** | 153 |

## Core Changes per Document

### 00_PROJECT_CHARTER.md

- **Status**: Phase 1 → "Phase 2 Exploration Complete → Next-Phase
  Precision Engineering".
- **Canonical baseline**: E-40 (MV3-Small, hard dense_fg_aux, 30ep,
  pf_mIoU 0.8564) with full config table.
- **Stronger variant**: E-52 (EffB0, 30ep, pf_mIoU 0.8711) positioned as
  reference variant, NOT default replacement.
- **Dense training-only**: Explicitly documented — `dense_fg_aux` is
  training-only, stripped at inference.
- **Closed directions**: Full table of 14 closed directions with key
  results and verdicts (E-53a, bg_mix, soft_bbox, adaptive softening,
  Center+Extent, MV3-Large, weight_decay=1e-3, strong jitter, .npy cache,
  50ep, objectness, CosineWarmRestart, log-WH, center aux).
- **No GreenVCOD reproduction**: Explicitly stated — we absorb temporal
  window concept only.
- **Active/pending**: References roadmap doc. Notes E-54/E-55/E-56 as
  "implementation complete, not validated."

### 01_CURRENT_TASK.md

- **Deleted**: All M1-M4 stub tasks (conda env, dummy dataloader,
  MicroVCOD_Lite, synthetic smoke test).
- **Replaced with**: Next-Phase Precision Engineering, priority queue
  P0-P7.
- **P0 (current)**: E-52 archive, top-K reeval infrastructure, composite
  score design, docs/README consistency.
- **E-54/E-55/E-56 repositioning**: Explicitly marked as "implementation
  complete, NOT experimentally validated." Positioned as infrastructure
  ready for P1-P4, subject to revision based on smoke tests.
- **Experiment design rules**: 8-item mandatory checklist for all future
  experiments (error type, success/no-go criteria, unified reeval, etc.).
- **Negative space**: What we do NOT do — grid search, new aux heads
  without hypothesis, bg_mix/E-53b, training before P0 complete.

### 02_DEVELOPMENT_PROTOCOL.md

- **Deleted**: M1-M4 milestone framework.
- **Replaced with**: Experiment governance protocol.
- **Experiment proposal template**: 8-field template (ID, target error,
  hypothesis, baseline, single variable, success criteria, no-go criteria,
  inference impact, rollback).
- **Unified reeval mandate**: Training-log best_val_mIoU alone is
  insufficient. All conclusions require np.random.RandomState(42) reeval.
- **Top-K standard**: `topk_checkpoints: 3` + `reeval_checkpoints.py` as
  standard procedure for any cited result.
- **Required metrics**: Full checklist (core, size, spatial, error counts,
  hard videos, efficiency).
- **Report naming**: `reports/report_yyyymmddhhmm.md` in UTF-8.
- **CTO self-check**: 5 questions before every action.

### 03_EXPERIMENT_ROADMAP.md (NEW)

Priority-ordered queue with full specification for each:

| Priority | Direction | Target Error | Inference Cost |
|----------|-----------|-------------|----------------|
| P0 | Governance | N/A | N/A |
| P1 | MV3-Small zoom probe | Tiny-object inaccuracy | Zero |
| P2 | EffB0 zoom probe | Tiny-object inaccuracy | Zero |
| P3 | MV3-Small coverage probe | Large-object under-coverage | Zero |
| P4 | EffB0 coverage probe | Large-object under-coverage | Zero |
| P5 | COD dense pretraining | Generalization gap | Zero |
| P6 | Temporal dilation | Fast-motion degradation | Zero |
| P7 | EffB0 → MV3-Small distillation | Backbone cost gap | Zero |

Each entry includes: why, target error type, baseline, single variable,
probe plan (1ep→8ep→30ep gating), success criteria with quantified
thresholds, no-go criteria, inference cost, and rollback plan.

Key design principle: 1-epoch smoke before 8ep probe, 8ep go/no-go before
30ep full run. No experiment gets 30ep without passing the 8ep gate.

### 04_METRIC_AND_ERROR_TAXONOMY.md (NEW)

- **Core metrics**: pf_mIoU, bad_frame_rate, R@0.5 with interpretation
  guide.
- **Size-bin IoU**: Tiny/Small/Medium/Large definitions with area
  thresholds and current pain points.
- **Spatial accuracy**: area_ratio (mean/median), center_error (mean/median)
  with diagnostic interpretation patterns.
- **Error type taxonomy**: pred_too_large, pred_too_small, center_shift,
  scale_mismatch — criteria, meaning, dominant patterns from Phase 2
  audits.
- **Hard-video metrics**: flatfish_2, pygmy_seahorse_0,
  white_tailed_ptarmigan with best-known mIoU values.
- **Efficiency**: FPS, Params.
- **Composite score (DRAFT)**: Weighted formula with explicit "not yet
  calibrated" warning. Includes calibration TODO.
- **Reporting checklist**: 13-item mandatory checklist.

## E-54/E-55/E-56 Repositioning

These three code implementations are now documented as:

1. **Infrastructure**, not results.
2. **Implementation complete** — code compiles, passes synthetic unit tests.
3. **Not experimentally validated** — no real-data training has been run.
4. **Subject to revision** — smoke tests may reveal issues requiring
   parameter or code changes before 8ep probes.
5. **Gated behind config flags** — all defaults are zero/off, no impact
   on existing training pipelines.

They are referenced in the roadmap as "code ready" for P1-P4 but are NOT
claimed as validated methods.

## Why Experiments Have Not Been Run

1. **Governance first**: The project needed consistent documentation
   before launching new training. Previous docs were at Phase 1 stub level.
2. **Baseline anchoring**: P1-P4 success/no-go criteria reference E-40/E-45/
   E-51/E-52 baselines. These needed to be clearly documented before
   comparisons could be meaningful.
3. **Top-K infrastructure**: `reeval_checkpoints.py` and `topk_checkpoints`
   are implemented but not yet validated on a real experiment. Running
   P1-P4 without validated eval infrastructure would risk unreliable
   checkpoint selection.
4. **Composite score**: The draft composite score needs calibration before
   it can gate go/no-go decisions. Individual metric thresholds serve as
   interim criteria.

## Next Steps

1. Validate `topk_checkpoints` + `reeval_checkpoints.py` on E-51 or E-52
   checkpoint files (no new training needed — use existing checkpoints).
2. Run 1-epoch smoke tests for E-54 and E-55 (separately, not combined).
3. If smokes pass: launch P1 8ep probe.
4. Archive E-52 with final report.
5. Calibrate composite score on existing experiment pairs.

## Verification

- All 5 docs: valid UTF-8, readable.
- No training code modified in this pass.
- No experimental claims made about E-54/E-55/E-56.
- Docs are internally consistent: same baselines, same closed directions,
  same eval protocol referenced across all files.
