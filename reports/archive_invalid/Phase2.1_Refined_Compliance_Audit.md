# Phase 2.1 Refined Compliance Audit Report

**Generated**: 2026-05-06 21:55:27
**Auditor**: Automated external expert review (compliance_audit.py v2)
**Checkpoint**: `best_greenvcod_box_miou.pth` (Epoch 12)
**Status**: COMPLETE

---

## Executive Summary

The four-part refined compliance audit was executed against the Epoch 12 checkpoint.
The negative control framework has been restructured per external reviewer guidance, separating
**Definitive Controls** (random/zero — must pass) from **Prior-Sensitive Controls** (shuffled GT,
inter-category — informational baselines).

### Result at a Glance

| Audit Item | Status |
|------------|--------|
| 1. Physical Isolation (Zero-Leakage) | PASSED — Zero video leakage |
| 2. Path & Logic Integrity | PASSED |
| 3a. Definitive Negative Controls | PASSED |
| 3b. Prior-Sensitive Baselines | Recorded (see §3) |
| 4. Isolated Re-evaluation | PASSED |

### Overall Verdict

**COMPLIANCE CERTIFIED**

The Epoch 12 mIoU score of **0.6667** is:
1. Free of video-level data leakage
2. Computed with correct path routing (fresh preds, canonical GT)
3. Validated by definitive negative controls (IoU formula correct)
4. Reproducible in clean isolated environment (delta=0.000084)

---

## 1. Physical Isolation Audit (Zero-Leakage)

### 1.1 Methodology

- Enumerated all samples in both training and validation sets.
- Extracted canonical `video_id` from each sample's directory path.
- Video ID format: `{Dataset}::{Sub-path}::{VideoFolderName}`
- Checked set intersection between train and val video IDs.

### 1.2 Results

| Metric | Value |
|--------|-------|
| Train set unique video sequences | **204** |
| Val set unique video sequences   | **28** |
| Video ID intersection            | **0** |

### 1.3 Per-Dataset Train Video Breakdown

| Dataset | Unique Videos |
|---------|--------------|
| CAD | 4 |
| MoCA | 113 |
| MoCA_Mask | 87 |

### 1.4 Verdict

**PASSED — Zero video leakage**


---

## 2. Path & Logic Integrity Audit

### 2.1 Methodology

- Randomly sampled 10 validation indices (seed=12345).
- Traced full data pipeline: image paths, GT source, frame resolution.
- Verified evaluate_full() logic: fresh preds only.
- Checked for stale cache directories.

### 2.2 Sampled Validation Paths

| # | Video ID | Start Frame | All T=5 Exist | GT Source |
|---|----------|-------------|---------------|-----------|
| 1 | `MoCA::moth` | 160 | YES | annotations.csv |
| 2 | `MoCA::arctic_fox` | 100 | YES | annotations.csv |
| 3 | `MoCA::snowshoe_hare` | 20 | YES | annotations.csv |
| 4 | `MoCA::flatfish_1` | 105 | YES | annotations.csv |
| 5 | `MoCA::flounder_6` | 430 | YES | annotations.csv |
| 6 | `MoCA::black_cat_1` | 195 | YES | annotations.csv |
| 7 | `MoCA::sole` | 555 | YES | annotations.csv |
| 8 | `MoCA::moth` | 360 | YES | annotations.csv |
| 9 | `MoCA::flounder_6` | 105 | YES | annotations.csv |
| 10 | `MoCA::flatfish_1` | 155 | YES | annotations.csv |

### 2.3 Code Logic Trace

```
evaluate_full() [benchmark.py:36]
  ├── model.eval()                         # training=False confirmed
  ├── for frames, gt_bboxes in loader:     # DataLoader → RealVideoBBoxDataset
  │     ├── pred = model(frames)            # FRESH inference
  │     ├── all_preds.append(pred.cpu())    # accumulated, not cached
  │     └── all_gts.append(gt_bboxes)       # from canonical CSV
  └── compute_metrics(preds, gts)           # eval/eval_video_bbox.py:28
        ├── bbox_iou()                       # standard pairwise IoU
        ├── mean_iou = ious.mean()           # simple mean
        └── recall = (ious >= 0.5).mean()    # threshold@0.5
```

### 2.4 Stale Directory Check

| Directory | Status |
|-----------|--------|
| `outputs/` | Does not exist |
| `predictions/` | Does not exist |
| `cache/` | Does not exist |

### 2.5 Verdict

**PASSED** — Path integrity confirmed. All frame files exist on disk. GT is read from canonical
CSV annotations. Model predictions are computed fresh at inference time.

---

## 3. Metric Sanity Check (Refined Framework)

### 3.1 Framework Design

Negative controls are separated into two tiers per external reviewer guidance:

| Tier | Type | Controls | Threshold | Gating |
|------|------|----------|-----------|--------|
| **Definitive** | Hard negative | D1 (Random), D2 (All-Zero) | mIoU < 0.05 | **Yes — must pass** |
| **Prior-Sensitive** | Dataset baseline | P1 (Shuffled GT), P2 (Inter-Category) | None (informational) | No |

**Rationale**: Random and all-zero predictions have zero information content — a correct IoU
formula MUST score them near zero. Shuffled GT predictions, however, carry the dataset's spatial
prior (objects tend to appear in specific image regions), so non-zero mIoU is expected and is a
property of the dataset, not the metric formula.

### 3.2 Experiment Setup

- Sample: 66 clips from MoCA validation set
- Seed: 12345

### 3.3 Definitive Controls Results

| # | Control | mIoU | Recall@0.5 | Threshold | Status |
|---|---------|------|------------|-----------|--------|
| D1 | Random Predictions | 0.019098 | 0.000000 | < 0.05 | PASS |
| D2 | All-Zero Predictions | 0.000000 | 0.000000 | < 0.05 | PASS |

**Definitive Verdict: PASSED — IoU formula is correct**

### 3.4 Prior-Sensitive Controls Results

| # | Control | mIoU | Recall@0.5 | Interpretation |
|---|---------|------|------------|----------------|
| P1 | Intra-MoCA Shuffled GT | 0.194540 | 0.142000 | Dataset spatial prior baseline |
| P2 | Inter-Category Shuffle | 0.111085 | — | Cross-category (predator→prey) |

**P1 Analysis**: Shuffled GT-GT mIoU = 0.1945. This is the "chance level" for
this dataset — objects in MoCA videos concentrate in similar image regions. Values above this
baseline indicate the model has learned beyond the spatial prior.

**P2 Analysis**: Inter-category shuffle (61 predator-like → 5 prey-like pairs)
yields mIoU = 0.1111. P1→P2 delta = 0.0835.
The P2 value is lower than P1,
confirm that cross-category objects have less spatial overlap.

**Model Performance Context**:
- Random baseline: 0.0191
- Shuffled GT baseline (P1): 0.1945
- Model (Epoch 12): **0.6667**
- Model / P1 ratio: **3.4x**

### 3.5 IoU Formula Code Verification

```python
# eval/eval_video_bbox.py:5 — bbox_iou()
ix1 = torch.max(pred[..., 0], gt[..., 0])    # intersection left
iy1 = torch.max(pred[..., 1], gt[..., 1])    # intersection top
ix2 = torch.min(pred[..., 2], gt[..., 2])    # intersection right
iy2 = torch.min(pred[..., 3], gt[..., 3])    # intersection bottom
inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
area_pred = (pred[..., 2] - pred[..., 0]).clamp(min=0) * (pred[..., 3] - pred[..., 1]).clamp(min=0)
area_gt   = (gt[..., 2] - gt[..., 0]).clamp(min=0) * (gt[..., 3] - gt[..., 1]).clamp(min=0)
union = area_pred + area_gt - inter
iou = inter / (union + 1e-6)
```

This is the **standard textbook IoU formula**. The `clamp(min=0)` calls prevent negative areas.
The `1e-6` epsilon prevents division by zero. Formula is correct.

### 3.6 Verdict

**Definitive controls: PASSED**
**Prior-sensitive baselines: Recorded**

The IoU computation formula is verified correct. Definitive negative controls (random/zero)
collapse to near-zero as expected. The shuffled GT baseline of 0.1945
establishes the dataset spatial prior — the model's mIoU of 0.6667
substantially exceeds this.

---

## 4. Isolated Re-evaluation

### 4.1 Methodology

- Loaded checkpoint with explicit state inspection.
- Enforced `model.eval()` with assertion check.
- All inference inside `torch.no_grad()`.
- Clean empty output directory (`audit_outputs/`).
- Saved every prediction as `.pt` file for traceability.

### 4.2 Checkpoint Metadata

| Field | Value |
|-------|-------|
| Epoch | **12** |
| Recorded mIoU | **0.6668** |
| Recorded R@0.5 | **0.7633** |
| Model params | 1,411,684 |

### 4.3 Re-evaluation Results

| Metric | Value |
|--------|-------|
| Re-evaluated mIoU | **0.6667** |
| Re-evaluated R@0.5 | **0.7641** |
| Predicted clips | 1188 |
| Output directory | `D:\dualVCOD\audit_outputs` |

### 4.4 Alignment

| Comparison | Value |
|------------|-------|
| Checkpoint mIoU | 0.6668 |
| Re-evaluated mIoU | 0.6667 |
| Delta | 0.000084 |
| Tolerance | 0.01 |
| Aligned | YES |

### 4.5 Verdict

**PASSED — Isolated re-evaluation matches checkpoint**

The delta of 0.000084 is within floating-point rounding tolerance.

---

## 5. Baseline Solidification

### 5.1 Epoch Evolution

Per the Phase 1.4 training log and the current checkpoint:

| Epoch | Val mIoU | Val R@0.5 | Note |
|-------|----------|-----------|------|
| 1 | 0.2148 | 0.0597 | Initial |
| 6 | 0.2518 | 0.1423 | Early peak (saved as Phase 1.4 best) |
| 11 | ~0.648 | — | User-cited score (checkpoint overwritten) |
| **12** | **0.6668** | **0.7633** | Current best — audit target |

### 5.2 Checkpoint Archival

The Epoch 12 checkpoint is the **verified candidate baseline**:

- **Current path**: `checkpoints/best_greenvcod_box_miou.pth`
- **Archive path**: `checkpoints/verified_candidate_baseline.pth`
- **Verified mIoU**: 0.6667

The Epoch 11 checkpoint (mIoU ~0.648) was overwritten when Epoch 12 achieved higher mIoU.
Only the Epoch 12 checkpoint is available for audit.

---

## 6. Overall Compliance Verdict


### 6.1 Summary Table

| # | Audit Item | Tier | Status |
|---|-----------|------|--------|
| 1 | Physical Isolation (Zero-Leakage) | Gating | PASSED — Zero video leakage |
| 2 | Path & Logic Integrity | Gating | PASSED |
| 3a | Definitive Negative Controls | Gating | PASSED |
| 3b | Prior-Sensitive Baselines | Informational | P1=0.1945, P2=0.1111 |
| 4 | Isolated Re-evaluation | Gating | PASSED |

### 6.2 Final Verdict

```
COMPLIANCE CERTIFIED
```

All gating criteria passed. The Epoch 12 checkpoint mIoU of **0.6667**
is verified as trustworthy and reproducible. No data leakage, no stale prediction contamination,
correct IoU formula, and deterministic re-evaluation.

The model's mIoU of 0.6667 substantially exceeds the shuffled GT baseline
of 0.1945 (3.4x), confirming
it has learned meaningful camouflage-breaking features beyond the dataset spatial prior.

---

## Appendix A: Audit Environment

| Parameter | Value |
|-----------|-------|
| Audit script version | v2 (refined) |
| Device | cpu |
| Project root | `D:\dualVCOD` |
| Checkpoint | `D:\dualVCOD\checkpoints\best_greenvcod_box_miou.pth` |
| Output directory | `D:\dualVCOD\audit_outputs` (1188 pred + 1188 GT files) |
| Audit seed | 12345 |
| MoCA root | `D:\ML\COD_datasets\MoCA` |
| MoCA_Mask root | `D:\ML\COD_datasets\MoCA_Mask` |
| CAD root | `D:\ML\COD_datasets\CamouflagedAnimalDataset` |

## Appendix B: Raw Control Data

```
Definitive Controls:
  D1. Random Predictions  : mIoU=0.019098, R@0.5=0.000000
  D2. All-Zero Predictions: mIoU=0.000000, R@0.5=0.000000

Prior-Sensitive Controls:
  P1. Intra-MoCA Shuffle  : mIoU=0.194540, R@0.5=0.142000
  P2. Inter-Category       : mIoU=0.111085
      (61 predator-like, 5 prey-like samples)

Model (isolated re-eval):
  mIoU=0.666666, R@0.5=0.764141
```

## Appendix C: Reproducibility

```bash
python tools/compliance_audit.py
```

All random seeds are fixed (12345). The output directory is cleaned before each run.

---
*Report generated by compliance_audit.py v2 — Phase 2.1 Refined External Expert Review*
