# Phase 1.1 Report — M1 Pipeline Validation

**Date**: 2026-05-06
**Status**: COMPLETE

## Environment

| Item | Value |
|------|-------|
| Conda environment | `dualvcod` (cloned from `dualcod-cu`) |
| Python | 3.10.20 |
| PyTorch | 2.6.0+cu124 |
| CUDA available | Yes |
| GPU | NVIDIA RTX 4090 (detected as `cuda`) |

## New Files

| File | Purpose |
|------|---------|
| `src/__init__.py` | Package init |
| `src/dataloader.py` | `SyntheticVideoDataset` — generates (T, C, H, W) clips + GT BBoxes; `collate_video_clips` — stacks to (B, T, C, H, W) |
| `src/model.py` | `DummyVCOD` — lightweight GreenVCOD S3-inspired model: `SpatialEncoder` (3-layer stride-2 CNN) → `TemporalNeighborhood` (1D Conv on T-axis, short+long term fusion) → `BBoxHead` (FC → 4 coords) |
| `eval/__init__.py` | Package init |
| `eval/eval_video_bbox.py` | `bbox_iou`, `compute_metrics` (mean IoU, Recall@0.5), `benchmark_fps`, `count_parameters` |
| `scripts/smoke_test.py` | End-to-end smoke test using synthetic data |

## Model Architecture (DummyVCOD)

```
Input: (B, T, 3, 224, 224)
  │
  ▼ SpatialEncoder (shared across T)
  │   Conv2d(3→32, k3, s2) → BN → ReLU
  │   Conv2d(32→64, k3, s2) → BN → ReLU
  │   Conv2d(64→128, k3, s2) → BN → ReLU
  │   → (B*T, 128, 28, 28)
  │
  ▼ TemporalNeighborhood
  │   Short-term: Conv1d(k=3) along T axis
  │   Long-term:  AdaptiveAvgPool1d → Conv1d(k=1) → broadcast
  │   Gate: residual with sigmoid modulation
  │   → (B, T, 128, 28, 28)
  │
  ▼ BBoxHead
      AdaptiveAvgPool2d(1) → FC(128→64)→ReLU → FC(64→4)→Sigmoid
      → (B, T, 4)
```

**Design constraints respected:**
- No optical flow
- No 3D convolutions or large Video Transformers
- BBox-only output (x1, y1, x2, y2 normalized to [0,1])
- 200,900 trainable parameters

## Smoke Test Results

| Metric | Value |
|--------|-------|
| Input shape | (4, 5, 3, 224, 224) |
| Output shape | (4, 5, 4) |
| Output value range | [0.4807, 0.5264] |
| Shape check | PASSED |
| NaN check | PASSED |
| Inf check | PASSED |
| BBox [0,1] range | PASSED |
| Mean BBox IoU | 0.0023 (random dummy, expected) |
| Recall@0.5 | 0.0000 (random dummy, expected) |
| FPS (CUDA, RTX 4090) | 644.6 frames/sec |
| Parameters | 200,900 |

## Verification Summary

- **Dataloader**: Produces correct (B, T, C, H, W) tensor with matching GT BBoxes (B, T, 4).
- **Model forward**: Accepts (B, T, C, H, W), outputs (B, T, 4) with values in [0, 1].
- **Evaluation**: `compute_metrics` correctly calculates per-frame IoU and aggregate statistics.
- **FPS benchmark**: Measures throughput including the full forward pass.

Pipeline is validated end-to-end. Ready for M2: GreenVCOD TN module formal implementation.
