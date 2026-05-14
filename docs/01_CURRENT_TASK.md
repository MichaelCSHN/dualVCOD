# Current Task: Next-Phase Precision Engineering

Phase 2 exploration is complete. We are no longer probing hyperparameters or
trying arbitrary auxiliary heads. The current phase targets **specific,
named failure modes** with minimal, justified interventions.

## 1. State at Entry

- Canonical baseline: E-40 (MV3-Small, hard dense_fg_aux, 30ep) —
  pf_mIoU 0.8564.
- Stronger variant: E-52 (EffB0, hard dense_fg_aux, 30ep) —
  pf_mIoU 0.8711.
- Evaluation protocol: Unified reeval on np.random.RandomState(42)
  MoCA val split.
- E-53a closed: multi-scale dense supervision catastrophically degraded
  tiny-object detection.
- E-54/E-55/E-56 code: Implemented but **not experimentally validated**.

## 2. Priority Queue

### P0 — Governance & Infrastructure (CURRENT)

These must complete before any new training experiments.

1. **E-52 archive cleanup**
   - Archive E-52 checkpoint, config, and unified reeval output.
   - Write final E-52 report with full metric breakdown.
   - Update all docs to reference E-52 as stronger variant.
   - Status: E-52 trained and reevaluated; archive pending.

2. **Top-K checkpoint unified reeval infrastructure**
   - `tools/autoresearch/run_trial_minimal.py` now supports `topk_checkpoints`
     config parameter. `tools/autoresearch/reeval_checkpoints.py` is the
     standalone reeval script.
   - Must validate on at least one completed experiment before declaring
     operational.
   - Status: Implementation complete, not validated.

3. **Composite score design**
   - Draft a composite metric that weights pf_mIoU, bad_frame_rate,
     size-bin IoU, and hard-video metrics into a single go/no-go number.
   - See `docs/04_METRIC_AND_ERROR_TAXONOMY.md` for the draft.
   - Status: Draft written, not calibrated.

4. **Docs/README consistency**
   - All five docs (`00`–`04`) + README must agree on baselines,
     closed directions, eval protocol, and next steps.
   - Status: This document set is the reconciliation.

### P1-P7 — Precision Engineering Experiments

See `docs/03_EXPERIMENT_ROADMAP.md` for full details. Summary:

| Priority | Direction | Target Error Type | Status |
|----------|-----------|-------------------|--------|
| P1 | MV3-Small conservative zoom probe | Tiny-object inaccuracy | Not started |
| P2 | EffB0 zoom probe | Tiny-object inaccuracy | Not started |
| P3 | MV3-Small coverage-aware loss | Large-object under-coverage | Not started |
| P4 | EffB0 coverage-aware loss | Large-object under-coverage | Not started |
| P5 | COD dense pretraining | Generalization gap | Not started |
| P6 | Temporal dilation | Fast-motion degradation | Not started |
| P7 | EffB0 → MV3-Small distillation | Backbone cost gap | Not started |

### E-54/E-55/E-56 — Implementation Complete, Not Validated

Code for three precision-engineering experiments has been written:

- **E-54** (`src/dataset_real.py`): Scale-aware natural zoom augmentation.
  Expands bbox by context_factor, crops, resizes back to 224. Triggers with
  higher probability for smaller objects. Clip-consistent across T=5.
- **E-55** (`src/loss.py`): Large-object under-coverage penalty.
  `coverage = |pred∩gt| / |gt_area|`. Asymmetric — only penalizes
  under-coverage. Weight 0.1, threshold area > 0.15.
- **E-56** (`tools/autoresearch/run_trial_minimal.py` +
  `tools/autoresearch/reeval_checkpoints.py`): Top-K checkpoint saving
  and post-training unified reeval.

**These are NOT experimentally validated.** They compile and pass synthetic
unit tests but have not been run on real data. No performance claims should
be made. They are positioned as:

1. Infrastructure ready for P1-P4 experiments.
2. Code to be validated with 1-epoch smoke tests before any 8ep/30ep run.
3. Subject to revision based on smoke test results.

## 3. Experiment Design Rules (All Future Work)

Every new experiment must:

1. Name the specific error type it targets (see taxonomy in `04`).
2. State expected improvement in which metrics.
3. Define success criteria AND no-go criteria BEFORE training.
4. Use a single-variable change vs the canonical baseline.
5. Run unified reeval (not training-log best_val_mIoU) as the final
   arbiter.
6. Report: pf_mIoU, bad_frame_rate, R@0.5, size-bin IoU, area_ratio,
   center_error, error type counts, hard-video metrics.
7. State whether inference cost changes (params, FPS).
8. Provide a rollback plan if the experiment fails.

## 4. What We Do NOT Do

- No more grid-search hyperparameter tuning.
- No new auxiliary head architectures without a specific error-type
  hypothesis.
- No background mixing, soft_bbox, adaptive softening, Center+Extent,
  MV3-Large, or E-53b.
- No training runs before P0 governance tasks complete.
- No claims about E-54/E-55/E-56 efficacy before experimental validation.
