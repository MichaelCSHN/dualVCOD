# dualVCOD: Project Charter & Current State

> **Core principle**: Model-centric, fast iteration, no process busywork.
> Single RTX 4090 must run the full pipeline. Deployment output is BBox only —
> no instance-level segmentation at inference.
>
> **Primary reference**: [arXiv:2501.10914v1] Section 3 (GreenVCOD) —
> temporal window concept only. This project is an independent PyTorch
> implementation named **MicroVCOD**. Zero lines of code reused.
> Pure gradient-descent training. We do NOT reproduce GreenVCOD.

## 1. Project End-State Goal

A lightweight Video Camouflaged Object Detection (VCOD) model for
single-modality (RGB) video streams.

- **Hardware constraint**: Train on single RTX 4090 (24 GB).
  Deploy on edge devices (e.g., drone-side).
- **Output**: One bounding box per frame. No pixel-level mask at inference.
- **Training-only**: Dense auxiliary supervision heads are allowed during
  training but stripped at inference — output is always BBox only.

## 2. Current Phase: Phase 2 Exploration Complete → Next-Phase Precision Engineering

Phase 2 (controlled single-variable trials, E-01 through E-53a) is complete.
50+ experiments explored backbone, loss, temporal, auxiliary supervision,
augmentation, and hyperparameter axes.

**Phase 2 confirmed**: dense_fg_aux training + bbox-only inference is the
winning architecture class. DIoU + hard dense targets + 30ep training is the
recipe. EfficientNet-B0 is a stronger backbone candidate but is not yet the
default replacement for MobileNetV3-Small.

We are now entering **Next-Phase Precision Engineering**: targeted fixes for
specific failure modes, not broad exploration. Every new experiment must name
the error type it targets and define success/no-go criteria in advance.

### 2.1 Canonical Baseline (Edge Deploy Target)

| Setting | Value |
|---------|-------|
| Architecture | MicroVCOD |
| Head (train) | hard `dense_fg_aux` (weight 0.5) |
| Head (inference) | bbox only |
| Backbone | MobileNetV3-Small |
| Input | 224×224, T=5 |
| Loss | SmoothL1 + DIoU |
| Optimizer | AdamW, lr=0.001, weight_decay=1e-4 |
| Scheduler | CosineAnnealing (T_max=30) |
| Augmentation | ColorJitter=0.15, random flip |
| Epochs | 30 |
| Eval protocol | Unified reeval, np.random.RandomState(42) split |

**E-40** (30ep): per-frame mIoU **0.8564**, bad_frame_rate 0.0358, R@0.5 0.9642.

### 2.2 Stronger Variant (Reference, Not Default)

| Setting | Value |
|---------|-------|
| Backbone | EfficientNet-B0 |
| LR | 0.0003 |
| Warmup | 3-epoch LinearLR |

**E-52** (30ep): per-frame mIoU **0.8711**, bad_frame_rate 0.0257, R@0.5 0.9743.

E-52 beats E-40 by +0.0146 pf_mIoU across all metrics. However,
EfficientNet-B0 has higher parameter count and per-step cost. It is a
**promising stronger variant** — not yet the default mainline replacement.
Backbone selection must be justified per-experiment with cost/quality
trade-off analysis.

## 3. CTO "Red Lines" (Non-Goals)

Without explicit instruction, the following are FORBIDDEN:

1. **No explicit optical flow** — lightweight efficiency is paramount.
2. **No 3D convolutions or large Video Transformers**.
3. **No mask output at inference** — all evaluation metrics are BBox-based.
4. **No GreenVCOD reproduction** — we absorb high-level inspiration only
   (temporal window, spatial prediction quality). All code is original.
5. **No grid search** — hypothesis-driven, minimum-necessary experiment
   groups only.

## 4. Architecture Baseline

- **Input**: T=5 consecutive frames at 224×224.
- **Spatial encoder**: Lightweight CNN backbone (MobileNetV3-Small or
  EfficientNet-B0) → FPN with stride-8 output (28×28 feature map).
- **Temporal neck**: TemporalNeighborhood — multi-frame pooling.
  Known limitation: order-invariant (forward == reversed).
  GlobalAvgPool removes temporal position information.
- **Output head**: BBoxHead → (B, T, 4) normalized coordinates.
- **Training-only aux heads**: DenseForegroundHead (28×28 BCE) for
  auxiliary supervision. Stripped at inference.

## 5. Closed Directions

The following were explored in Phase 2 and **closed** — none matched or
beat the canonical baseline. Do not re-explore without new evidence.

| Direction | Key Result | Verdict |
|-----------|-----------|---------|
| Background mixing (CutMix/MixUp) | E-42 pf_mIoU 0.6087 (−0.22 vs baseline) | Destructive: breaks camouflage task |
| `soft_bbox` (Gaussian-softened targets) | No improvement over hard targets | Closed |
| Size-adaptive Gaussian softening | No improvement over hard targets | Closed |
| Center+Extent auxiliary head (V1) | E-49 pf_mIoU 0.7953 (−0.06 vs baseline) | Regresses core metrics |
| MV3-Large backbone | No benefit over MV3-Small (E-50/E-23/E-34) | Closed — larger isn't better |
| `weight_decay=1e-3` | E-37 over-regularizes | Closed |
| Strong color jitter (0.3) | Degrades performance (E-38) | Closed |
| Frame-level `.npy` cache | Large disk, no speed benefit | Closed |
| 50-epoch training | Diminishing returns vs 30ep (E-33) | Not worth cost |
| Multi-scale dense supervision (s4+s8) | E-53a pf_mIoU 0.7540 (−0.08) | **Catastrophic** — pygmy_seahorse_0 0.008 |
| Objectness auxiliary head | Adds params, negligible gain | Closed |
| CosineWarmRestart scheduler | No benefit | Closed |
| Log-WH size loss | Negligible impact | Closed |
| Center auxiliary loss | Marginal, not adopted | Closed |

## 6. Active and Pending Directions

See `docs/03_EXPERIMENT_ROADMAP.md` for the full priority queue.

Current P0 (governance):
- E-52 archive cleanup
- Top-K checkpoint unified reeval infrastructure
- Docs/README consistency

Pending experiments (implementation complete, not validated):
- E-54: Scale-aware natural zoom (targets tiny-object inaccuracy)
- E-55: Large-object under-coverage penalty (targets pred_too_small)
- E-56: Top-K checkpoint unified reeval (targets checkpoint selection)

## 7. Repository Hygiene

- Datasets are **NOT** committed (`data/`, `MoCA/`, `COD10K/`, etc.)
- Checkpoints are **NOT** committed (`*.pth`, `*.pt`)
- `local_runs/` experiment outputs: JSON metadata tracked; logs and
  checkpoints git-ignored
- `resized_root` (e.g., `C:\datasets_224`) is a local I/O accelerator —
  not part of the repository
- All evaluation conclusions use unified reeval protocol
