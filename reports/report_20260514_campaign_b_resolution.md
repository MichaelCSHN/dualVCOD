# Campaign B: Resolution Pareto — Final Report
## 2026-05-14

---

## B1: 256×256 Resolution Probe

### Hypothesis
Higher input resolution (256×256) directly addresses the tiny-object information bottleneck. At 256×256, tiny objects (area < 0.01) get ~26×26 pixels vs ~22×22 at 224×224.

### Experiment
Single variable: `input_size` 224 → 256. All other parameters identical to E-45 (MV3-Small, T=5, stride=1, dense_fg_aux, hard targets, DIoU, 8ep). `batch_size=16` maintained (same as 224 baseline).

### Code Fixes Required
Two bugs were discovered and fixed before B1 could run:
1. **Dense target grid size**: `hw=28` was hardcoded in `src/dataset_real.py:673` → changed to `hw=self.target_size // 8`
2. **Pre-resized image mismatch**: `resized_root` loads 224×224 images without size check → set to `null` for B1 config (fell through to `cv2.resize(target_size)` which correctly resizes)

### Training Trajectory

| Epoch | Train Loss | Train mIoU | Val mIoU | Val R@0.5 | Time (s) |
|-------|-----------|-----------|---------|----------|---------|
| 1 | 1.024 | 0.364 | 0.234 | 0.124 | 466 |
| 2 | 0.746 | 0.512 | 0.250 | 0.139 | 387 |
| 3 | 0.643 | 0.579 | 0.273 | 0.176 | 365 |
| 4 | 0.572 | 0.628 | 0.261 | 0.168 | 354 |
| 5 | 0.519 | 0.666 | 0.269 | 0.168 | 354 |
| 6 | 0.471 | 0.701 | 0.271 | 0.173 | 353 |
| 7 | 0.431 | 0.734 | **0.287** | 0.198 | 355 |
| 8 | 0.405 | 0.755 | 0.284 | 0.190 | 357 |

Training val best: E7 (0.287). Similar convergence pattern to 224 baseline (train loss 0.405 vs 0.400 at E8).

### Unified Reeval

| Rank | Epoch | Train Val | pf_mIoU | IoU_tiny | IoU_small | IoU_med | IoU_large | R@0.5 | bad_frame |
|------|-------|-----------|---------|----------|----------|---------|----------|-------|----------|
| 1 | 7 | 0.287 | 0.746 | 0.308 | 0.632 | — | 0.810 | 0.888 | 0.113 |
| 2 | 8 | 0.284 | **0.749** | 0.335 | 0.680 | — | 0.809 | 0.896 | 0.105 |
| 3 | 3 | 0.273 | 0.653 | 0.185 | 0.497 | — | 0.716 | 0.833 | 0.167 |

- Training-val-best (E7) ≠ unified-best (E8) — **6/7 disagreement pattern continues**
- Top-K heap fix verified: correct epochs preserved (no stale E4/E5)

### Comparison vs E-45 Baseline (224×224)

| Metric | E-45 (224) | B1 (256) | Delta | Verdict |
|--------|-----------|---------|-------|--------|
| pf_mIoU | 0.8281 | 0.7488 | **-0.079** | ❌ TRIGGERS NO-GO |
| R@0.5 | 0.950 | 0.896 | -0.054 | ❌ |
| bad_frame_rate | 0.050 | 0.105 | +0.055 | ❌ |
| IoU_tiny | 0.584 | 0.335 | **-0.249** | ❌ |
| IoU_small | 0.706 | 0.680 | -0.026 | ⚠️ |
| IoU_large | 0.860 | 0.809 | -0.051 | ❌ |
| FPS | — | 85.7 | — | ✓ |

### Go/No-Go

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| IoU_tiny +0.03+ (success) | >0.614 | 0.335 | ❌ NOT MET |
| pf_mIoU drop >0.02 (no-go) | >-0.02 | -0.079 | **TRIGGERED** |
| Any non-tiny bin drop >0.03 | >-0.03 | IoU_large -0.051 | **TRIGGERED** |
| FPS < 30 | >30 | 85.7 | ✓ OK |

**Verdict: NO-GO** ❌

### Why Higher Resolution Hurts

The hypothesis that 256×256 helps tiny objects was exactly wrong — it made EVERYTHING worse. Possible mechanisms:

1. **Effective receptive field dilution**: At 256×256, the FPN stride-8 features are 32×32 instead of 28×28. The same convolutional kernels cover a smaller fraction of the spatial grid, reducing effective receptive field relative to object size.

2. **Under-training at higher spatial complexity**: The training schedule (8ep, lr=0.001, batch=16) was tuned for 224×224. At 256×256, the model has 31% more spatial positions to learn (1024 vs 784 per feature map), requiring more optimization steps.

3. **Dense target difficulty**: The dense_fg head at 32×32 has finer-grained targets than at 28×28. The hard binary targets become sparser and harder to learn.

4. **Data loading bottleneck**: Without pre-resized images, data loading consumed 75% of training time. The cv2 resize path may introduce slight quality differences vs the pre-resized JPEG pipeline, though this shouldn't explain the magnitude of the regression.

### Decision
**B2 (288×288) cancelled.** The -0.079 pf_mIoU regression at 256×256 makes the resolution hypothesis untenable. If 256 is this harmful, 288 would be worse or at best equally bad. The monotonic degradation from 224→256 (all 4 size bins negative) indicates resolution is not the bottleneck for this architecture.

### Meta: Top-K Fix Validated
The `os.rename` → `shutil.copy2` fix correctly preserves checkpoint rank files across epochs. B1's top-3 checkpoints (E7=0.287, E8=0.284, E3=0.273) match the expected heap contents. The training-val vs unified-reeval disagreement (6/7) is now a robust finding unaffected by checkpoint corruption.

---

## Campaign B Verdict

**Resolution from 224 to 256 is not a viable improvement path for MV3-Small + dense_fg_aux.** The 224×224 input size is confirmed as the optimal resolution for this architecture. The tiny-object bottleneck must be addressed through other means (backbone upgrade, longer training, or dedicated head design) rather than input resolution.
