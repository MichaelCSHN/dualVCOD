# Experiment Roadmap: Next-Phase Precision Engineering

Priority-ordered queue. Each entry must pass its go/no-go gate before
the next priority in that track is started. Backbone tracks (MV3-Small
and EffB0) can proceed in parallel if GPU capacity permits.

## Priority Ladder

```
P0: Governance (no training)         ← CURRENT
P1: MV3-Small zoom probe (1ep→8ep)
P2: EffB0 zoom probe (1ep→8ep)
P3: MV3-Small coverage probe (1ep→8ep)
P4: EffB0 coverage probe (1ep→8ep)
P5: COD dense pretraining
P6: Temporal dilation
P7: EffB0 → MV3-Small distillation
```

---

## P0 — Governance & Infrastructure

**Why**: Docs, baselines, and eval infrastructure must be solid before any
new training. Prevents confusion about which results are canonical.

**Tasks**:
1. Archive E-52 (checkpoint, config, unified reeval).
2. Validate `topk_checkpoints` + `reeval_checkpoints.py` on one existing
   experiment.
3. Finalize composite score draft (docs/04).
4. Achieve docs/README consistency.

**Success**: All five docs + README agree on baselines, closed directions,
eval protocol, and P1-P7 priorities.

**No-Go**: N/A — governance only.

---

## P1 — MV3-Small Conservative Zoom Probe

**Why**: Tiny objects (area < 0.01) are the single largest remaining error
source. pygmy_seahorse_0 mIoU ≈ 0.4 at best. At 224×224, a 5×5 pixel
object is essentially invisible. Natural zoom during training gives the
model higher-resolution views of small objects without breaking the
camouflage context.

**Target error type**: Tiny-object inaccuracy, bad_frame_rate on
small-object videos.

**Baseline**: E-40 (MV3-Small, hard dense_fg_aux, 30ep, pf_mIoU 0.8564).

**Single variable**: `zoom_enabled=true` in dataset (code: E-54
implementation in `src/dataset_real.py`).

**Probe plan**:
1. 1-epoch smoke: verify zoom triggers, no NaN, loss stable.
2. 8-epoch probe: compare against E-45 (MV3-Small 8ep, pf_mIoU 0.8281).
3. If 8ep passes go/no-go: 30ep full run vs E-40.

**Success criteria (8ep)**:
- IoU_tiny ≥ 0.55 (E-45: 0.5119 baseline at 8ep for EffB0; MV3-Small
  tiny baseline TBD from E-45 reeval)
- pygmy_seahorse_0 mIoU improves by ≥ 0.05
- pf_mIoU does not drop > 0.01 vs baseline
- No other size bin drops > 0.03

**No-Go criteria**:
- pf_mIoU drops > 0.02
- Any size bin drops > 0.05
- pygmy_seahorse_0 degrades (like E-53a)
- Training becomes unstable (NaN, loss spikes)

**Inference cost**: Zero. Zoom is training-only augmentation. No change
to model, params, or FPS.

**Rollback**: Remove `zoom_enabled=true` from config. Code is gated behind
config flag — no code rollback needed.

---

## P2 — EffB0 Zoom Probe

**Why**: If P1 shows zoom helps MV3-Small, verify the benefit transfers
to the stronger backbone. EffB0 may benefit less (it already extracts
better features from small regions) or more (zoom provides higher-quality
input to a better encoder).

**Target error type**: Same as P1.

**Baseline**: E-51 (EffB0, hard dense_fg_aux, 8ep, pf_mIoU 0.8372).

**Single variable**: `zoom_enabled=true`, backbone=efficientnet_b0.

**Probe plan**: 8ep only. Compare against E-51. If 8ep passes, decide
whether 30ep P2 follow-up is warranted.

**Success/no-go**: Same thresholds as P1, referenced to E-51 baseline.

**Inference cost**: Zero.

**Rollback**: Same as P1.

---

## P3 — MV3-Small Coverage-Aware Loss Probe

**Why**: Large objects (area > 0.15) suffer from pred_too_small —
predictions that fail to cover the full GT extent. flatfish_2 mIoU ≈ 0.09
is dominated by under-coverage. Standard IoU/GIoU losses are symmetric and
can be satisfied by a small prediction inside a large GT. An asymmetric
coverage penalty specifically targets this.

**Target error type**: Large-object under-coverage, pred_too_small.

**Baseline**: E-40 (MV3-Small, hard dense_fg_aux, 30ep, pf_mIoU 0.8564).

**Single variable**: `loss_weights.large_coverage=0.1` (code: E-55
implementation in `src/loss.py`).

**Probe plan**:
1. 1-epoch smoke: verify coverage loss > 0 only for large GT, no NaN.
2. 8-epoch probe vs E-45.
3. If 8ep passes: 30ep full run vs E-40.

**Success criteria (8ep)**:
- IoU_large improves by ≥ 0.02
- flatfish_2 mIoU improves by ≥ 0.03
- pred_too_small count decreases
- pf_mIoU does not drop > 0.01
- pred_too_large does NOT increase significantly (confirms asymmetry works)

**No-Go criteria**:
- pf_mIoU drops > 0.02
- pred_too_large increases > 20% (penalty encouraging blind enlargement)
- Any non-large size bin drops > 0.03
- flatfish_2 degrades

**Inference cost**: Zero. Coverage penalty is loss-only — no change to
model forward pass.

**Rollback**: Set `loss_weights.large_coverage=0.0` in config. Code is
gated behind zero-default weight.

---

## P4 — EffB0 Coverage-Aware Loss Probe

**Why**: Same logic as P2 → P3. Verify the coverage penalty transfers to
the stronger backbone.

**Target error type**: Same as P3.

**Baseline**: E-51 (EffB0, 8ep, pf_mIoU 0.8372).

**Single variable**: `loss_weights.large_coverage=0.1`,
backbone=efficientnet_b0.

**Probe plan**: 8ep only. Compare against E-51.

**Success/no-go**: Same thresholds as P3, referenced to E-51 baseline.

**Inference cost**: Zero.

---

## P5 — COD Dense Pretraining

**Why**: Current training starts from ImageNet-pretrained backbones.
ImageNet features are optimized for semantic classification, not
camouflage detection. Pretraining the dense foreground head on a larger
COD dataset (COD10K, CAMO, NC4K) before fine-tuning on MoCA could improve
the quality of spatial features, especially for boundary-ambiguous cases.

**Target error type**: Generalization gap, boundary ambiguity.

**Baseline**: E-40 (MV3-Small) or E-52 (EffB0).

**Single variable**: Pretraining phase on COD datasets before MoCA
fine-tuning.

**Probe plan**:
1. Feasibility study: can we load and train on COD10K + CAMO + NC4K
   with our existing dataset adapter?
2. If feasible: pretrain 10-20 epochs on COD, fine-tune 30ep on MoCA.
3. Compare against no-pretraining baseline.

**Success criteria**: pf_mIoU improves by ≥ 0.01 over baseline.
No catastrophic forgetting on MoCA-specific hard cases.

**No-Go criteria**: Pretraining degrades MoCA performance (negative
transfer). COD dataset integration requires > 1 week of engineering.

**Inference cost**: Zero (same model architecture).

**Dependencies**: COD dataset availability and disk space.

---

## P6 — Temporal Dilation

**Why**: Current T=5 with temporal_stride=1 covers ~0.17 seconds at 30fps.
Fast-moving or erratically-moving camouflaged objects may benefit from
longer temporal context. Temporal dilation (sampling every Nth frame over
a wider window) could capture motion patterns without increasing T.

**Target error type**: Fast-motion degradation, temporal ambiguity.

**Baseline**: E-40 or E-52.

**Single variable**: `temporal_stride > 1` or variable-stride sampling.

**Probe plan**: 8ep probe with stride ∈ {2, 3, "variable"}. T=5 fixed.

**Success criteria**: Improvement on videos with known fast motion.
No degradation on static camouflage.

**No-Go criteria**: pf_mIoU drops > 0.01. Degradation on pygmy_seahorse_0
or flatfish_2 (slow, static camouflage).

**Inference cost**: Zero (same T, same model).

---

## P7 — EffB0 → MV3-Small Distillation

**Why**: E-52 (EffB0) beats E-40 (MV3-Small) by +0.0146 pf_mIoU, but
EffB0 has higher parameter count and FPS cost. If we can distill the
EffB0 teacher's knowledge into an MV3-Small student, we get the accuracy
of EffB0 at the cost of MV3-Small. This directly serves the "lighter,
faster, more accurate" charter.

**Target error type**: Backbone cost/quality gap.

**Baseline**: E-40 (MV3-Small student, 0.8564) and E-52 (EffB0 teacher,
0.8711).

**Single variable**: Knowledge distillation loss (e.g., KL on bbox
predictions, feature alignment on FPN outputs).

**Probe plan**:
1. Load frozen E-52 as teacher.
2. Train MV3-Small student with distillation loss + standard loss.
3. 30ep run. Compare student against E-40 (no distillation) and E-52
   (teacher upper bound).

**Success criteria**: Student pf_mIoU achieves ≥ 50% of the teacher-student
gap closure (≥ 0.8637). Inference cost unchanged from MV3-Small.

**No-Go criteria**: Student performs worse than E-40 (distillation hurts).
Implementation requires > 3 days of engineering.

**Inference cost**: Zero (MV3-Small at inference).

**Dependencies**: E-52 frozen checkpoint available.
