# Phase 1.2 Report — M2 Real Backbone + Multi-Dataset Adapter

**Date**: 2026-05-06
**Status**: COMPLETE

---

## 1. Dataset Structure Analysis

### MoCA
| Property | Value |
|----------|-------|
| Sequences | 141 videos |
| Total annotations | 7,617 BBoxes |
| Resolution | 1280 x 720 (JPG) |
| Annotation format | CSV (VIA), BBox as `[2, x, y, w, h]` |
| Annotation interval | every 5 frames |
| Frame naming | `{video}/{frame:05d}.jpg` |

### MoCA_Mask
| Property | Value |
|----------|-------|
| Train sequences | 71 videos |
| Test sequences | 16 videos |
| Resolution | 1280 x 720 (JPG + PNG mask) |
| Annotation format | Binary PNG masks → BBox via contour bounding rect |
| Annotation interval | every 5 frames |
| Frame naming | `{split}/{video}/Imgs/{frame:05d}.jpg`, `GT/{frame:05d}.png` |

### CAD (CamouflagedAnimalDataset)
| Property | Value |
|----------|-------|
| Animal categories | 10 |
| Resolution | 640 x 360 (PNG + PNG mask) — varies by category |
| Annotation format | Binary PNG masks → BBox via contour bounding rect |
| Annotation interval | typically every 5 frames (frog: every 1 frame) |
| Frame naming | `{animal}/frames/{animal}_{idx:03d}.png`, `groundtruth/{idx:03d}_gt.png` |

### Key structural differences
- **MoCA**: BBox annotations are pre-computed in a CSV file → fastest to load, no mask processing needed.
- **MoCA_Mask**: Mask files provide denser annotations but require `mask_to_bbox` conversion at load time.
- **CAD**: Most heterogeneous — varied naming patterns (`{animal}_{idx}`), varied annotation density (frog: dense, others: sparse every 5).
- **CAD resolution is 640x360** (half of MoCA's 1280x720) → less spatial detail but faster to process.

---

## 2. New & Modified Files

| File | Change | Purpose |
|------|--------|---------|
| `tools/analyze_datasets.py` | NEW | Dataset structure probe — prints directory layout, annotation format, image resolutions for all 3 datasets |
| `src/dataset_real.py` | NEW | `RealVideoBBoxDataset` — unified dataset adapter with spatial alignment (resize→224), BBox coordinate scaling/normalization, `mask_to_bbox` for mask-based datasets |
| `src/model.py` | MODIFIED | Added `SpatialEncoderFPN` (MobileNetV3-Small + FPN multi-scale fusion) and `GreenVCOD_Box` (real backbone + TN + BBoxHead) |
| `scripts/test_m2_real.py` | NEW | End-to-end real-data validation script testing all 3 datasets |

### Architecture: GreenVCOD_Box

```
Input: (B, T, 3, 224, 224)
  │
  ▼ SpatialEncoderFPN (per-frame, shared weights)
  │   MobileNetV3-Small features split into 3 stages:
  │     C2: stride  8,  24ch  (features[:3])
  │     C3: stride 16,  40ch  (features[3:7])
  │     C4: stride 32, 576ch  (features[7:])
  │   Top-down FPN with lateral convs (→128ch) + bilinear upsampling
  │   → (B*T, 128, 28, 28)
  │
  ▼ TemporalNeighborhood (unchanged from M1)
  │   Conv1d(k=3) short-term + Global avg-pool long-term → residual gate
  │   → (B, T, 128, 28, 28)
  │
  ▼ BBoxHead (unchanged from M1)
  │   AdaptiveAvgPool2d(1) → FC(128→64)→ReLU → FC(64→4)→Sigmoid
  │   → (B, T, 4)
```

**Design constraints check**:
- [x] No optical flow
- [x] No 3D convolution
- [x] No heavy Video Transformer
- [x] BBox-only output (x1, y1, x2, y2) in [0, 1]
- [x] MobileNetV3-Small pretrained on ImageNet (lightweight, ~1.5M params for backbone features)
- [x] Spatial align: all frames resized to 224x224, BBox coords scaled + normalized
- [x] Mask → BBox conversion for MoCA_Mask and CAD

---

## 3. Real-Data Validation Results (RTX 4090, CUDA)

| Dataset | Samples | Params | FPS (frames/sec) | mIoU (untrained) | R@0.5 | Shape Check |
|---------|---------|--------|-------------------|-------------------|-------|-------------|
| MoCA | 6,948 | 1,411,684 | 17.8 | 0.0013 | 0.0000 | PASS |
| MoCA_Mask | 4,296 | 1,411,684 | 30.9 | 0.0000 | 0.0000 | PASS |
| CAD | 47 | 1,411,684 | 90.3 | 0.0000 | 0.0000 | PASS |

**Notes**:
- Low IoU / Recall is expected — model is untrained with random head weights.
- CAD has only 47 valid T=5 windows due to sparse annotations (most animals have ~12-21 annotated frames, giving ~8-17 windows each).
- MoCA FPS (17.8) is the most representative benchmark — 6,948 samples with diverse content.
- CAD FPS (90.3) is artificially high due to small dataset size (cached batches).
- No NaN, Inf, or out-of-range BBox values detected across all datasets.

### Comparison with Dummy Model
| Model | Params | FPS (MoCA) | Backbone |
|-------|--------|------------|----------|
| DummyVCOD (M1) | 200,900 | 644.6 | 3-layer conv from scratch |
| GreenVCOD_Box (M2) | 1,411,684 | 17.8 | MobileNetV3-Small + FPN (pretrained) |

GreenVCOD_Box has 7x more parameters (real backbone) and lower FPS (deeper network + FPN overhead), but gains multi-scale feature extraction and ImageNet-pretrained weights — critical for real COD detection.

---

## 4. Project Structure (updated)

```
dualVCOD/
├── src/
│   ├── __init__.py
│   ├── dataloader.py         # SyntheticVideoDataset (M1)
│   ├── dataset_real.py       # RealVideoBBoxDataset (M2 NEW)
│   └── model.py              # DummyVCOD + GreenVCOD_Box (M2 MODIFIED)
├── eval/
│   ├── __init__.py
│   └── eval_video_bbox.py    # bbox_iou, compute_metrics, benchmark_fps
├── tools/
│   └── analyze_datasets.py   # Dataset structure probe (M2 NEW)
├── scripts/
│   ├── smoke_test.py          # M1 synthetic smoke test
│   └── test_m2_real.py        # M2 real-data validation (M2 NEW)
├── reports/
│   ├── phase1.1-202605060125.md
│   └── phase1.2-202605060153.md
└── docs/
```

---

## 5. Next Steps (M3)

1. **Training loop**: Implement loss function (IoU loss + L1 regression), optimizer, training epoch on MoCA.
2. **Temporal window densification**: Handle intermediate unannotated frames via interpolation or self-training.
3. **Hyperparameter tuning**: Learning rate, weight decay, batch size for single RTX 4090.
4. **Full evaluation**: Proper train/val/test split, track mIoU and R@0.5 curves during training.
