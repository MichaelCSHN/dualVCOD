# Phase 1.3 Report — M3 Training Pipeline & Overfit Sanity Check

**Date**: 2026-05-06
**Status**: COMPLETE · OVERFIT TEST PASSED

---

## 1. New Files

| File | Purpose |
|------|---------|
| `src/loss.py` | `BBoxLoss` — composite SmoothL1 + GIoU loss with DIoU option; `box_giou()`, `box_diou()` standalone functions |
| `tools/train.py` | Memory-constrained trainer: AMP mixed precision, AdamW + CosineAnnealingLR, `--overfit` mode, simple terminal logging |

### Modified Files

| File | Change |
|------|--------|
| `src/dataset_real.py` | Added `np.clip(bbox, 0.0, 1.0)` to MoCA BBox normalization (defense against out-of-bounds annotations) |
| `eval/eval_video_bbox.py` | Added `.clamp(min=0)` to area calculations in `bbox_iou` (robustness against invalid box coordinates) |

---

## 2. Loss Design

### BBoxLoss
```
L = λ_smooth * SmoothL1(pred, gt, β=0.1) + λ_giou * (1 - GIoU(pred, gt))
```

| Component | Purpose |
|-----------|---------|
| SmoothL1 (β=0.1) | Per-coordinate regression; β=0.1 switches from L1 to L2 near zero, giving fine-grained gradient for tiny camouflaged targets |
| GIoU | Scale-invariant box-level overlap; handles non-overlapping predictions (penalises via enclosing box area) |

GIoU ranges from -1 (disjoint boxes) to 1 (perfect overlap). The loss term `1 - GIoU` ranges from 0 to 2.

DIoU is also implemented as an alternative — adds center-distance penalty for faster convergence on distant targets.

---

## 3. Training Infrastructure

### Memory Safety
```python
torch.cuda.set_per_process_memory_fraction(0.45, 0)
```
Locks GreenVCOD_Box to ≤45% of RTX 4090 VRAM (~10.8 GB of 24 GB), leaving headroom for concurrent workloads.

### Training Recipe
| Parameter | Value |
|-----------|-------|
| Precision | AMP (float16 forward, float32 weights) |
| Optimiser | AdamW (lr=1e-3, weight_decay=1e-4) |
| LR Schedule | CosineAnnealingLR (T_max=epochs) |
| Gradient clipping | max_norm=2.0 |
| Batch norm | train-mode for overfit eval |

### Why no WandB/TensorBoard
Terminal logger only — zero-friction for single-GPU 4090 iteration. No heavy external dependencies.

---

## 4. Overfit Sanity Check Results

**Configuration**: 8 clips x T=5 from MoCA, 224x224, 200 epochs on frozen batch.

### Loss & mIoU Trajectory

| Epoch | Total Loss | SmoothL1 | GIoU Loss | mIoU | Δ |
|-------|-----------|----------|-----------|------|---|
| 1 | 1.4224 | 0.1182 | 1.3042 | 0.0031 | — |
| 10 | 1.0421 | 0.1359 | 0.9062 | 0.1876 | +0.18 |
| 25 | 0.7516 | 0.1144 | 0.6372 | 0.3733 | +0.19 |
| 50 | 0.6161 | 0.0533 | 0.5628 | 0.4535 | +0.08 |
| 75 | 0.3769 | 0.0148 | 0.3621 | 0.6395 | +0.19 |
| 100 | 0.2759 | 0.0065 | 0.2694 | 0.7311 | +0.09 |
| 125 | 0.1401 | 0.0006 | 0.1395 | 0.8638 | +0.13 |
| **135** | **0.1016** | **0.0004** | **0.1013** | **0.9003** | **+0.04** |
| 150 | 0.0642 | 0.0001 | 0.0641 | 0.9366 | +0.04 |
| 175 | 0.0274 | 0.0001 | 0.0273 | 0.9727 | +0.04 |
| 200 | 0.0168 | 0.0001 | 0.0167 | **0.9833** | +0.01 |

### Convergence Analysis

```
mIoU
1.0 ┤                                    ·········
0.9 ┤                              ······
0.8 ┤                         ·····
0.6 ┤                    ····
0.4 ┤              ······
0.2 ┤        ······
0.0 ┼──┬────┬────┬────┬────┬────┬────┬────┬────┬───
    0   20   40   60   80  100  120  140  160  180  200 epoch
```

- **Epoch 1–60**: Slow progress (mIoU 0.0 → 0.5). Backbone adapts pretrained features to the target bbox regression task.
- **Epoch 60–135**: Acceleration phase (mIoU 0.5 → 0.90). SmoothL1 dominates, rapidly reducing coordinate error.
- **Epoch 135–200**: Fine-tuning (mIoU 0.90 → 0.983). GIoU loss refines box overlaps.

### Final Status

| Metric | Value | Threshold | Result |
|--------|-------|-----------|--------|
| Final mIoU | **0.9833** | ≥ 0.90 | **PASSED** |
| Best mIoU | **0.9835** | — | — |
| Final Loss | 0.0168 | — | Converged |
| Gradient flow | No NaN/Inf | — | Clean |
| Training time | 18.0s (200 epochs) | — | 0.09s/epoch |

---

## 5. Design Verification

| Constraint | Status |
|------------|--------|
| BBox-only output (no masks) | [x] |
| No optical flow | [x] |
| No 3D convolution | [x] |
| No heavy Video Transformer | [x] |
| VRAM ≤ 45% (RTX 4090) | [x] |
| AMP mixed precision | [x] |
| No WandB/TensorBoard bloat | [x] |
| SmoothL1 + GIoU composite loss | [x] |

---

## 6. Bug Fixes Applied

1. **GT BBox out-of-bounds**: MoCA annotations occasionally extend beyond 1280x720 → added `np.clip(0,1)` in `RealVideoBBoxDataset._index_moca`.
2. **IoU area underflow**: `eval_video_bbox.bbox_iou` lacked `.clamp(min=0)` on box areas; predictions with flipped coords (x1 > x2) produced negative areas → matched behaviour to `loss._box_iou`.

---

## 7. Next Steps (M4)

1. **Full training on MoCA**: Extend `tools/train.py` with train/val split, checkpointing, multi-epoch training.
2. **Benchmark**: Measure final mIoU, Recall@0.5, FPS, and parameter count on held-out MoCA test set.
3. **Hyperparameter sweep**: Tune λ_smooth/λ_giou weights, learning rate, and T for optimal convergence.
4. **Model ablation**: Compare DummyVCOD vs GreenVCOD_Box on real data.
