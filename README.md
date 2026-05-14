# dualVCOD

Lightweight bounding-box-only Video Camouflaged Object Detection.

Goal: lighter backbone, faster throughput, stronger per-frame detection —
bbox-only at inference, with optional dense auxiliary supervision during
training.

## Status

**Phase 2 exploration complete.** 50+ controlled single-variable trials across
backbone, loss, temporal, and auxiliary-supervision axes.

### Canonical baseline

| Setting | Value |
|---------|-------|
| Architecture | MicroVCOD |
| Head | hard `dense_fg_aux` |
| Backbone | MobileNetV3-Small |
| Input | 224×224, T=5 |
| Inference | bbox only |
| Eval | unified reeval (RandomState(42) split) |

**E-40** (30ep): per-frame mIoU **0.8564**.

### Stronger variant

**E-51** — EfficientNet-B0 + hard `dense_fg_aux`, 8ep: per-frame mIoU **0.8372**.

Reaches near-baseline quality in 8 epochs vs 30, but with higher parameter
count and per-step cost. A promising variant that needs careful cost/quality
comparison at matched epoch budget before being declared a replacement.

### E-53a — multi-scale dense supervision (CLOSED)

E-53a added a stride-4 (56×56) DenseForegroundHead alongside the existing
stride-8 (28×28) head. Evaluated under the unified reeval protocol:

| Metric | E-51 (8ep) | E-53a (8ep) |
|--------|-----------|-------------|
| pf_mIoU | 0.8372 | 0.7540 |
| IoU_tiny | 0.5119 | 0.2768 |
| bad_frame_rate | 0.0552 | 0.0900 |

**Direction closed** — degraded all metrics. Catastrophic collapse on
`pygmy_seahorse_0` (0.008 IoU). E-53b 30ep will not run. Code is retained
behind `head_type="dense_fg_aux_ms"` as an isolated extension point;
default code paths are unaffected.

### Closed directions

These were explored and closed: none matched or beat E-40 / E-51.

- background mixing / CutMix / MixUp
- `soft_bbox` (Gaussian-softened bbox targets)
- size-adaptive Gaussian softening
- Center+Extent auxiliary head (V1)
- `weight_decay=1e-3`
- strong color jitter
- MV3-Large backbone
- frame-level `.npy` cache
- naive longer training (no benefit beyond 30ep on current data scale)
- multi-scale dense supervision (E-53a, see above)

## Project Scope

- Lightweight bbox-only VCOD — no pixel-level segmentation at inference
- Multi-frame temporal aggregation (T=5 sliding windows)
- Clean train/val split with manifest-driven auditing
- Reproducible evaluation framework
- Single-GPU training (RTX 4090)

## Method

**MicroVCOD** — an independent PyTorch implementation sharing only the
high-level "temporal window" concept with arXiv:2501.10914 (GreenVCOD).
Gradient-based training with lightweight CNN backbones, not Green Learning.

## Repository Structure

```
src/              # Model architecture, dataset adapters, loss functions
tools/            # Training, reeval, auditing, verification scripts
  autoresearch/   # Trial runner, safety checker, scoring
eval/             # BBox evaluation metrics (IoU, Recall@0.5, FPS)
reports/          # Experiment reports and error analysis
  archive_invalid/  # Invalidated Phase 2.1 results (leaked checkpoint)
configs/          # Trial configuration files
  autoresearch/
scripts/          # Smoke tests
docs/             # Project charter and development protocols
```

## Setup

```bash
pip install -r requirements.txt
```

Core dependencies: `torch`, `torchvision`, `numpy`, `opencv-python`.

## Quick Smoke Test

```bash
# Synthetic data pipeline smoke (no dataset needed)
python scripts/smoke_test.py

# Red-team audit (all 4 tasks)
python tools/red_team_audit.py
```

## Training

```bash
# Single trial from a JSON config
python tools/autoresearch/run_trial_minimal.py \
  --config configs/autoresearch/expl_NN_xxx.json

# Legacy training (MV3-Small, current_direct_bbox head)
python tools/train.py --epochs 30 --batch_size 24
```

Trial outputs land in `local_runs/autoresearch/<trial_id>/` and are covered
by `.gitignore` (JSON metadata is tracked; checkpoints and logs are not).

The `resized_root` path (e.g. `C:\datasets_224`) in config files is a local
I/O accelerator. It is not part of the repository and must be set per-machine.

## Evaluation Protocol

All key conclusions use **unified reeval** — a fixed `RandomState(42)` split
applied consistently across trials. Training-log `best_val_mIoU` alone is
insufficient for comparison.

Required metrics per trial:

- per-frame mIoU (`pf_mIoU`)
- bad-frame rate
- Recall@0.5
- IoU by size bin (tiny / small / medium / large)
- area ratio (mean, median)
- center error (mean, median)
- error type counts (too large / too small / center shift / scale mismatch)
- per-video breakdown for hard cases

```bash
# 2×2 backbone × epochs comparison
python tools/reeval_2x2_backbone_epochs.py
```

## Data Leakage Notice

The previous Phase 2.1 checkpoint and its reported mIoU=0.8705 have been
**invalidated** due to MoCA train/validation data leakage. The leaked
checkpoint is not included in this repository.

**Do not cite mIoU=0.8705** — it is a leaked result, not a generalization
result. All Phase 2 metrics in this README come from the fixed split with
manifest-audited train/val separation.

See `reports/report_20260507_redteam.md` for the full Red-Team Audit findings.

## Repository Hygiene

- Datasets (`data/`, `MoCA/`, `COD10K/`, etc.) are **not committed**
- Checkpoints (`*.pth`, `*.pt`) are **not committed**
- `local_runs/` experiment outputs are partially tracked (JSON metadata only)
- Temporary logs and cache files are git-ignored

## License

MIT
