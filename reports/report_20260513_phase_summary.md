# dualVCOD Phase Exploration Summary

**Date**: 2026-05-13
**Purpose**: Comprehensive archive of Phase 1 experiment exploration for strategic project review

---

## 1. Project Goals

- **Hardware constraint**: Single RTX 4090 (24 GB VRAM) trainable
- **Deployment target**: Edge/mobile inference (bbox-only output, no mask head at runtime)
- **Priority**: Practical, stable, deployable — not maximal accuracy at any cost
- **Core model**: MicroVCOD (MobileNetV3-Small FPN + TemporalNeighborhood + BBoxHead)
- **Key evaluation metric**: Per-frame mIoU (pf_mIoU) via unified reeval on `np.random.RandomState(42)` MoCA val split

---

## 2. Explored Directions and Conclusions

### 2.1 Loss Functions

| Direction | Key Result | Verdict |
|-----------|-----------|---------|
| GIoU (baseline) | Standard composite loss | Baseline |
| DIoU | 0.8145 pf_mIoU at 8ep (E-29) | **Adopted** — slightly better center localization |
| CIoU | Slightly worse than DIoU | Not adopted |
| Center auxiliary loss | Center=0.2 borderline beneficial | Marginal, not adopted |
| Log-WH size loss | Negligible impact | Not adopted |
| Objectness auxiliary head | Adds params, negligible gain | Not adopted |

### 2.2 Data Pipeline

| Direction | Key Result | Verdict |
|-----------|-----------|---------|
| resized_root (pre-resize to 224) | 2.2× DataLoader speedup, identical accuracy | **Adopted** |
| Frame-level .npy cache | Large disk usage, no speed benefit | Closed |
| NVMe SSD | Helped but resized_root already sufficient | Informational |
| num_workers=2 | Best balance for RTX 4090 | **Adopted** |
| bs=16 | Stable, fits 24 GB comfortably | **Adopted** |

### 2.3 Training Hyperparameters

| Direction | Key Result | Verdict |
|-----------|-----------|---------|
| CosineAnnealing (T_max=epochs) | Standard | **Adopted** |
| CosineWarmRestart | No benefit | Not adopted |
| weight_decay=1e-4 | Stable training | **Adopted** |
| weight_decay=1e-3 | Over-regularizes (E-37) | Closed |
| ColorJitter=0.15 | Mild augmentation | **Adopted** |
| ColorJitter=0.3 | Degrades performance | Closed |
| lr=0.001 | Standard for MV3-Small | **Adopted** |
| 30-epoch training | pf_mIoU 0.8564 (E-40), +0.0283 over 8ep | **Largest verified gain** |
| 50-epoch training | Diminishing returns vs 30ep | Not worth cost |

### 2.4 Auxiliary Supervision — dense_fg_aux

| Direction | Key Result | Verdict |
|-----------|-----------|---------|
| dense_fg_aux with hard targets | 0.8281 pf_mIoU at 8ep (E-45) | **Canonical baseline** |
| dense_fg_aux 30ep | 0.8564 pf_mIoU (E-40) | **Best known result** |
| dense_fg_weight=0.5 | Good balance | **Adopted** |
| dense_fg_aux vs direct_bbox | +0.03-0.04 pf_mIoU, changes error structure | **Only direction that reliably changes error types** |

### 2.5 Background Mixing

| Direction | Key Result | Verdict |
|-----------|-----------|---------|
| bg_mix_prob=0.3 | pf_mIoU 0.6087 (E-42), −0.2194 vs E-45 | **Closed — destructive** |
| CutMix / MixUp variants | Not tested | Deferred — likely same failure mode |

Root cause: background replacement destroys the camouflage task essence. Camouflaged objects are defined by blending with their specific backgrounds.

### 2.6 Label Audit

| Direction | Key Result | Verdict |
|-----------|-----------|---------|
| E-41 error audit | Identified error taxonomy (4 types) | Diagnostic tool validated |
| E-43 flatfish_2 audit | flatfish_2 is genuine hard camouflage, not label error | **Confirmed: dataset quality is adequate** |
| flatfish_2 mIoU | 0.087 (E-45) — hardest video in dataset | Serves as hardness benchmark |

### 2.7 Teacher Route — Soft Dense Targets

| Experiment | Type | pf_mIoU | Δ vs E-45 | Verdict |
|------------|------|:-------:|:---------:|---------|
| E-45 | Hard dense_fg | 0.8281 | baseline | Mainline |
| E-46 | Uniform σ=1.0 soft_mask | 0.8329 | +0.0047 | Marginal win (tiny IoU tradeoff −0.0437) |
| E-47 | Soft bbox Gaussian falloff | 0.6587 | −0.1694 | **Closed — destructive** |
| E-48 | Size-adaptive soft_mask | 0.7236 | −0.1045 | **Closed — destructive** |
| E-49 | Center+Extent V1 | 0.8090 | −0.0191 | Negative, partial signals |

Key takeaway: Simple auxiliary supervision modifications produce limited gains at best, destructive regressions at worst. Hard binary dense_fg_aux is robust and hard to beat.

### 2.8 Backbone Capacity

| Direction | pf_mIoU | Δ vs E-45 | Verdict |
|-----------|:-------:|:---------:|---------|
| MV3-Small (1.5M) E-45 | 0.8282 | baseline | **Canonical baseline** |
| MV3-Large (3.6M) E-50 | 0.8069 | −0.0212 | **Negative** — same-family scaling underperforms |
| EfficientNet-B0 (4.7M) E-51 | 0.8372 | +0.0090 | **Positive** — cross-family scaling matters |

Key finding: architecture family matters more than parameter count. EfficientNet's MBConv+SE blocks outperform MobileNetV3's inverted residuals at similar resolutions. flatfish_2 shows dramatic 3× improvement (0.087 → 0.260) with EfficientNet-B0.

| Discovery | Detail |
|-----------|--------|
| Backbone registry FPN slice bug | **Discovered 2026-05-13**: MV3-Large and all EfficientNet variants had wrong FPN stage slices, producing 56×56 instead of 28×28 features. Hidden because prior probes used direct_bbox head. | **Fixed for all backbones** |
| NaN resolved for EffB0 | lr=0.0003 + warmup=3 stable; NaN at lr≥0.001 confirmed | Stable training established |

---

## 3. Current Best Mainline

```
Head:       dense_fg_aux (hard targets, weight=0.5)
Backbone:   MobileNetV3-Small (1.5M params)
Resolution: 224×224
Temporal:   T=5, stride=1
Loss:       SmoothL1 (β=0.1) + DIoU
Optimizer:  AdamW (lr=0.001, weight_decay=1e-4)
Scheduler:  CosineAnnealing (T_max=epochs)
Batch:      16 train / 64 eval
Data:       resized_root=C:\datasets_224
Augment:    HFlip + ColorJitter(0.15)
Seed:       42
Inference:  Bbox-only (dense head skipped at eval)
Evaluation: Per-frame mIoU on np.random.RandomState(42) MoCA val split
```

**Best verified results**:
- 8ep: pf_mIoU = 0.8281 (E-45)
- 30ep: pf_mIoU = 0.8564 (E-40) — +0.0283 gain

---

## 4. Closed Directions

These directions have been tested and **should not be revisited** without new evidence:

1. **Background mixing / CutMix / MixUp** — destroys camouflage task semantics
2. **Soft bbox Gaussian targets** — catastrophic regression (−0.1694)
3. **Size-adaptive Gaussian softening** — large-object over-prediction (−0.1045)
4. **Center+Extent V1 as direct replacement** — below baseline (−0.0191)
5. **weight_decay=1e-3** — over-regularizes
6. **Strong color jitter (≥0.3)** — degrades performance
7. **Naively extending epochs beyond 30** — diminishing returns
8. **Frame-level .npy cache** — no speed benefit, large disk

---

## 5. Valuable but Deferred Directions

These directions showed partial signals or are not yet fully explored:

1. **Uniform soft_mask σ=1.0 (E-46)**: +0.0047 pf_mIoU is real but marginal; tiny IoU regression prevents mainline adoption. Worth revisiting if tiny-object performance can be addressed.

2. **Center+Extent with real mask supervision**: pygmy_seahorse_0 improvement (+0.0682) suggests CE helps specific object types. Combining with real PNG masks might capture this benefit without overall regression.

3. **CE + dense_fg joint supervision**: Two auxiliary heads might be complementary. Higher implementation cost.

4. **Backbone scaling under dense_fg_aux**: E-50/E-51 results pending. If stronger backbones benefit more from dense supervision, this could be the next growth axis.

5. **Temporal T > 5**: E-44 (T=9) showed slightly different behavior. Worth revisiting as diagnostic.

6. **Higher-resolution CE targets (56×56)**: Deferred until CE fundamentals are validated.

---

## 6. Key Reflections

### 6.1 Metric Consistency
Training log mIoU (computed on `random.Random(42)` split) differs systematically from unified reeval mIoU (computed on `np.random.RandomState(42)` split). The ratio (reeval/training) is a diagnostic: stable experiments show ~2.72×, failing experiments collapse to 2.20-2.60×. **All future experiments must report per-frame metrics from unified reeval.**

### 6.2 dense_fg_aux is the Only Structural Change
Among all explored directions, only dense_fg_aux (hard targets) reliably changes the error structure — reducing scale_mismatch errors and improving small/medium IoU. Everything else (loss functions, schedulers, augmentations) produces marginal shifts.

### 6.3 flatfish_2 is Real Hard Camouflage
The E-43 audit confirmed: flatfish_2 annotations are correct. The video's extreme camouflage (flatfish matching seabed texture) makes it genuinely hard. It serves as the most informative benchmark sample.

### 6.4 Background Replacement Breaks the Task
E-42's failure is instructive: camouflaged object detection depends on the relationship between object and background. Random background replacement destroys this relationship.

### 6.5 Soft Target Engineering Has Limited Returns
Three variants (soft_mask, soft_bbox, size-adaptive) plus one structural alternative (center+extent) produced one marginal win and three negative results. The hard binary target is a strong local optimum.

### 6.6 FPN Stage Slice Registry Bug
Discovered 2026-05-13: MV3-Large and all EfficientNet backbones in the registry had incorrect FPN stage slices that stopped at stride 4 instead of stride 8. This produced 56×56 features instead of 28×28 — only caught when dense_fg_aux was used (which validates feature map dimensions against mask targets). Prior direct_bbox probes were unaffected because the BBoxHead uses AdaptiveAvgPool2d which accepts any spatial size. **Fixed for all affected backbones.**

### 6.7 The Project Needs Strategic Reassessment
After exhaustive exploration, the marginal gains from "small tweaks" are diminishing. The next phase should either:
- Consolidate and package current results as a publication-ready system
- Explore genuinely new structural directions (not parameter tuning)
- Or accept the current performance ceiling and document it honestly

### 6.8 Architecture Family Matters More Than Parameter Count
E-50 (MV3-Large, 3.6M) underperforms E-45 (MV3-Small, 1.5M) while E-51 (EfficientNet-B0, 4.7M) outperforms both. Same-family scaling is not the right growth axis — cross-family architectural differences (SE modules, channel attention) matter more for camouflage perception than raw capacity within the inverted residual design space.

---

## 7. Next-Step Candidates

These are candidate directions for strategic discussion — **not a decision to implement**:

### A. Canonical Baseline + Ablation
Consolidate hard dense_fg_aux as the canonical system. Run systematic ablations (remove components one at a time) to produce a clean publication narrative.

### B. Backbone Capacity Under dense_fg_aux
E-50 (MV3-Large) failed (−0.0212), E-51 (EfficientNet-B0) succeeded (+0.0090). Architecture family > parameter count. EfficientNet's SE modules may be key — flatfish_2 3× improvement is the strongest single-video gain in the project. EfficientNet-B0 at 30ep with lr tuning is the highest-ROI follow-up.

### C. Multi-Scale Dense Supervision
Apply dense_fg_aux at multiple FPN levels (not just stride 8), or add extent-aware refinement that preserves the spatial structure without the CE decomposition's complexity.

### D. Teacher Route Higher-Order Variants
The core insight (richer spatial supervision) may be correct but the current implementations too simple. Higher-order variants (real-mask CE, joint CE+dense_fg) could capture the benefit without the regression.

### E. Hard-Case Analysis and Narrative
Instead of chasing metrics, conduct detailed hard-case analysis (flatfish_2, pygmy_seahorse, etc.) and build a publication narrative around the system's behavior on genuinely difficult camouflage.

---

## 8. Experiment Index

### Phase 1 Early (E-01 to E-30): Baselines and Diagnostics

| ID | Description | Key Metric | Status |
|----|-------------|-----------|--------|
| E-01 | Low LR + warmup stability | Stable | Baseline |
| E-03 | EffB0 stability diagnostic | 0.2477 (1ep) | NaN solved at lr=1e-4 |
| E-07 | Objectness auxiliary head | Marginal | Closed |
| E-12 | GIoU + center + warmup 30ep | 0.8564 (reeval) | Reference |
| E-24 | EffB0 8ep direct_bbox | NaN at epoch 3 | Closed (lr too high) |
| E-27 | DIoU + center=0.2 8ep | Baseline | Informational |
| E-29 | DIoU T=7 8ep | 0.8145 | Informational |

### Phase 1.5 Pipeline (E-31 to E-38): Optimization

| ID | Description | Key Metric | Status |
|----|-------------|-----------|--------|
| E-31 | DIoU 30ep | Baseline | Reference |
| E-32 | CIoU 8ep | Below DIoU | Closed |
| E-35 | bs=16 50ep | Diminishing returns | Closed |
| E-36 | CosineWarmRestart | No benefit | Closed |
| E-37 | weight_decay=1e-3 | Over-regularized | Closed |
| E-38 | Jitter=0.3 | Degraded | Closed |

### Phase 2 dense_fg_aux (E-39 to E-44): Core Auxiliary Supervision

| ID | Description | Key Metric | Status |
|----|-------------|-----------|--------|
| E-39 | dense_fg_aux 8ep (first) | 0.8277 | Reference |
| E-40 | dense_fg_aux 30ep | **0.8564** | Best known |
| E-41 | Error audit | Taxonomy validated | Diagnostic |
| E-41.5 | Metric integration | per-frame metrics | Adopted |
| E-42 | bg_mix_prob=0.3 | 0.6087 (−0.2194) | Closed |
| E-43 | flatfish_2 label audit | Real camouflage confirmed | Diagnostic |
| E-44 | T=9 8ep | Informational | Deferred |

### Teacher Route (E-45 to E-49): Soft Target Engineering

| ID | Description | pf_mIoU | Δ vs E-45 | Status |
|----|-------------|:-------:|:---------:|--------|
| E-45 | Hard baseline (reproduce E-39) | 0.8281 | baseline | **Mainline** |
| E-46 | Uniform σ=1.0 soft_mask | 0.8329 | +0.0047 | Marginal (tiny IoU tradeoff) |
| E-47 | Soft bbox Gaussian falloff | 0.6587 | −0.1694 | Closed |
| E-48 | Size-adaptive soft_mask | 0.7236 | −0.1045 | Closed |
| E-49 | Center+Extent V1 | 0.8090 | −0.0191 | Negative |

### Backbone Capacity (E-50 to E-51): Scaling Under Dense Supervision

| ID | Description | pf_mIoU | Δ vs E-45 | Status |
|----|-------------|:-------:|:---------:|--------|
| E-50 | MV3-Large + dense_fg_aux 8ep | 0.8069 | −0.0212 | Negative |
| E-51 | EfficientNet-B0 + dense_fg_aux 8ep (lr=3e-4, warmup=3) | 0.8372 | +0.0090 | Positive |

---

## 9. Repository Asset Map

### Key Source Files
- `src/model.py` — MicroVCOD architecture (SpatialEncoderFPN, TemporalNeighborhood, BBoxHead, DenseForegroundHead, CenterExtentHead)
- `src/dataset_real.py` — RealVideoBBoxDataset with dense target generation (hard/soft_mask/soft_bbox/ce modes)
- `src/loss.py` — BBoxLoss with SmoothL1, DIoU, GIoU, CIoU, dense_fg, dense_ce components
- `eval/eval_video_bbox.py` — Per-frame metrics, size-stratified IoU, error classification

### Training Infrastructure
- `tools/autoresearch/run_trial_minimal.py` — Main training script with OOM recovery
- `tools/autoresearch/backbone_registry.py` — Backbone FPN configuration registry
- `tools/autoresearch/oom_recovery.py` — Emergency checkpoint and OOM retry logic
- `tools/autoresearch/config_safety.py` — GPU preflight checks
- `tools/autoresearch/profiler.py` — Per-epoch batch timing profiler

### Evaluation and Analysis
- `tools/reeval_trials.py` — General trial reeval script
- `tools/reeval_teacher_route.py` — Teacher route unified reeval (E-39 through E-49)
- `tools/reeval_backbone_probes.py` — Backbone probe unified reeval (E-45 + E-50 + E-51)
- `tools/run_error_audit.py` — Per-sample error classification audit
- `tools/e43_flatfish2_audit.py` — E-43 flatfish_2 label audit
- `tools/e43_mask_vs_bbox_compare.py` — Mask vs bbox comparison for E-43
- `tools/e43_model_vs_mask.py` — Model prediction vs ground truth mask comparison

### Data Pipeline Tools
- `tools/generate_resized_dataset.py` — Pre-resize dataset to 224×224
- `tools/micro_benchmark_decode.py` — JPEG decode micro-benchmark
- `tools/test_dataloader_speed.py` — DataLoader speed profiling
- `tools/verify_color_path.py` — Color augmentation path verification

### Configuration Files
All experiment configs in `configs/autoresearch/`:
- `expl_39_densefg_8ep.json` through `expl_51_effb0_densefg_8ep.json` — Phase 2+ experiment configs
- Pipeline profiling configs: `_prof_*.json`, `_speedtest_*.json`, `_pipeprof_*.json`, `_qual_*.json`

### Reports
All exploration reports in `reports/`:
- Phase 1: `report_20260507_*` through `report_20260510_*`
- Phase 1.5/2: `report_20260511_*` through `report_20260513_*`
- Error audit CSVs: `reports/e41_error_audit/`, `reports/e43_flatfish2_corrections/`

### Experiment Run Data
Metadata and metrics in `local_runs/autoresearch/`:
- Each trial has `metadata.json` (hyperparams + final metrics) and `metrics.json` (per-epoch metrics)
- `checkpoint_best.pth` files excluded from git via `.gitignore`
- `reeval_teacher_route.json` — cached unified reeval results

---

## 10. How to Regenerate Key Assets

Assets NOT committed to git (reproducible or too large):

| Asset | Regeneration Command |
|-------|---------------------|
| resized_root (C:\datasets_224) | `python tools/generate_resized_dataset.py` |
| Training checkpoints | `python tools/autoresearch/run_trial_minimal.py --trial_id <trial> --config configs/autoresearch/<config>.json` |
| Teacher route reeval | `python tools/reeval_teacher_route.py` |
| Backbone probe reeval | `python tools/reeval_backbone_probes.py` |
| Error audit CSVs | `python tools/run_error_audit.py` |
| flatfish_2 audit | `python tools/e43_flatfish2_audit.py` |
| Visual overlays | `python tools/generate_paper_assets.py` |
| flatfish_2 annotated frames | `python tools/e43_model_vs_mask.py` |

---

*This document is the authoritative summary of dualVCOD Phase 1 exploration as of 2026-05-13. All conclusions are supported by unified reeval metrics and documented in individual experiment reports.*
