# dualVCOD

Lightweight bounding-box-level Video Camouflaged Object Detection.

## Status

**Project under clean retraining.**

The previous Phase 2.1 checkpoint and its reported mIoU=0.8705 have been
**invalidated** due to MoCA train/validation data leakage. The leaked
checkpoint is not included in this repository and must not be used as a
generalization result.

See `reports/report_20260507_redteam.md` for the full Red-Team Audit findings.

## Project Scope

- Lightweight bbox-only VCOD — no pixel-level segmentation
- Multi-frame visual aggregation (5-frame temporal windows)
- Clean train/val split with manifest-driven auditing
- Reproducible red-team evaluation framework
- Single-GPU training (RTX 4090, <45% VRAM headroom)

## Method

**MicroVCOD** — an independent PyTorch implementation sharing only the
high-level "temporal window" concept with arXiv:2501.10914 (GreenVCOD).
Zero lines of code reused. Gradient-based training with MobileNetV3-Small
backbone, not Green Learning.

## Repository Structure

```
src/              # Model architecture, dataset adapters, loss functions
tools/            # Training, benchmarking, auditing, verification scripts
eval/             # BBox evaluation metrics (IoU, Recall@0.5, FPS)
reports/          # Audit reports and documentation
  archive_invalid/  # Invalidated Phase 2.1 results (leaked checkpoint)
docs/             # Project charter and development protocols
scripts/          # Smoke tests and validation scripts
```

## Setup

```bash
pip install torch torchvision numpy opencv-python matplotlib pillow
```

Or from requirements:

```bash
pip install -r requirements.txt
```

## Audit

```bash
# Red-team audit (all 4 tasks)
python tools/red_team_audit.py

# Data leak verification (no GPU needed)
python tools/verify_leak_fix.py
```

## Training

Clean training must use the fixed `tools/train.py` which filters MoCA
to train-only videos before constructing the ConcatDataset:

```bash
python tools/train.py --epochs 30 --batch_size 24
```

**Critical**: Validation videos must not be included in the training
ConcatDataset. The fix is verified by `tools/verify_leak_fix.py`.

## Disclaimer

- Datasets (MoCA, MoCA_Mask, CAD) are not included in this repository
- Model checkpoints (.pth) are not included
- The previous mIoU=0.8705 is invalid and must not be cited

## License

MIT
