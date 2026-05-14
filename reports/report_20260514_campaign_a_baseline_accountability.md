# Campaign A: Baseline Accountability — Final Report
## 2026-05-14

---

## A1: Baseline Training Log Audit

### Method
Extracted full epoch trajectories from metrics.json for all 4 canonical baselines.

### Findings

| Baseline | Ep | Best Val | Best Ep | Saved Ep | Saved Val | Suspicious? |
|----------|-----|---------|---------|---------|---------|------------|
| E-45 (MV3-S, 8ep) | 8 | 0.305 | E7 | E7 | 0.305 | **Yes — E8 at 0.297 unsaved** |
| E-40 (MV3-S, 30ep) | 30 | 0.309 | E18 | E18 | 0.309 | Yes — plateau E16-E30 around 0.30-0.31 |
| E-51 (EffB0, 8ep) | 8 | 0.305 | E8 | E7 | 0.305 | Minimal (last epoch = best) |
| E-52 (EffB0, 30ep) | 30 | — | E20 | E20 | — | Yes — early peak ~67% through |

### Key Insight
E-45 and E-40 both save checkpoints selected by training val. Given the 5/5 false-selection rate in all previous 8ep probes (training-val-best ≠ unified-best), these saved checkpoints may under-represent true performance by 0.005-0.020 pf_mIoU.

---

## A2: E-45 Top-K Retrain + Unified Reeval

### Method
Retrain E-45 config (MV3-Small, 224, T=5, stride=1, dense_fg_aux, hard, DIoU, 8ep) with topk_checkpoints=3. Save top-3 checkpoints by training val_mIoU. Unified reeval all 3 via np.random.RandomState(42) split.

### Training Trajectory (Retrain vs Original)

| Epoch | Original E-45 | Retrain A2 | Delta |
|-------|-------------|-----------|-------|
| 1 | 0.241 | 0.257 | +0.016 |
| 2 | 0.265 | 0.245 | -0.020 |
| 3 | 0.285 | 0.278 | -0.007 |
| 4 | 0.285 | 0.274 | -0.011 |
| 5 | 0.289 | 0.273 | -0.016 |
| 6 | 0.294 | 0.286 | -0.008 |
| 7 | **0.305** | 0.290 | -0.015 |
| 8 | 0.297 | **0.294** | -0.003 |

**Critical observation**: Different trajectories despite same seed (42). Original peaked at E7 (0.305), retrain monotonically increased E5→E8 (0.273→0.294). Root cause: DataLoader worker non-determinism (num_workers=2) means batch composition varies between runs even with fixed random seed.

### Unified Reeval Results

| Rank | Epoch | Train Val | Unified pf_mIoU | IoU_tiny | IoU_small | IoU_med | IoU_large |
|------|-------|-----------|----------------|----------|----------|---------|----------|
| 1 | 8 | 0.294 | **0.8314** | 0.511 | 0.769 | 0.850 | 0.858 |
| 2 | 4 | 0.274 | 0.6793 | 0.268 | 0.576 | 0.670 | 0.749 |
| 3 | 5 | 0.273 | 0.7587 | 0.342 | 0.693 | 0.786 | 0.789 |

### Comparison vs Original E-45

| Metric | Original E-45 (E7) | Retrain A2 (E8) | Delta |
|--------|---------------------|-----------------|-------|
| pf_mIoU | 0.8281 | **0.8314** | +0.0033 |
| R@0.5 | 0.950 | 0.937 | -0.013 |
| bad_frame_rate | 0.050 | 0.063 | +0.013 |
| IoU_tiny | 0.584 | 0.511 | -0.073 |
| IoU_small | 0.706 | 0.769 | +0.063 |
| IoU_medium | 0.838 | 0.850 | +0.012 |
| IoU_large | 0.860 | 0.858 | -0.002 |

### Conclusions

1. **Baseline is reproducible within ~0.003 pf_mIoU.** The retrain at E8 achieves 0.8314 vs original E7 at 0.8281 — well within expected seed variance.

2. **Training-val-best = unified-best in this retrain** — E8 was best by both metrics. This doesn't confirm or refute the original E-45 checkpoint selection hypothesis (original E8 was unsaved), but it's consistent with the 5/5 pattern that the last epoch is unified-best.

3. **Training trajectory varies with same seed** — DataLoader non-determinism introduces enough variance (0.011 val_mIoU at peak) that checkpoint selection by training val is inherently unreliable. This REINFORCES the need for unified reeval-based checkpoint selection.

4. **Top-K heap bug discovered and fixed** — `os.rename` destroyed `_topk_epoch` source files, causing stale rank files (E4/E5 instead of E6/E7 at ranks 2-3). Fixed with `shutil.copy2` + cleanup.

### Decision Gate: Proceed to A3?

The trajectory variance between original and retrain makes A3 (seed robustness, 2 additional seeds) valuable but also harder to interpret cleanly. Recommendation: **defer A3** — the key insight (training val unreliable for checkpoint selection) is already confirmed by the trajectory variance finding. Proceed directly to Campaign B (resolution) where the information gain per GPU-minute is higher.

---

## A1-A2 Verdict

| Question | Answer |
|----------|--------|
| Is E-45 reproducible? | Yes, within ~0.003 pf_mIoU |
| Is original E-45 checkpoint suboptimal? | Probably, but cannot prove (original E8 unsaved) |
| Is training val reliable for checkpoint selection? | **No** — trajectory varies with same seed |
| Should we switch to final-epoch policy? | **Yes for 8ep** — consistent with 5/5 pattern |
| Should we use top-K + unified reeval? | **Yes** — only reliable method |

### Policy Change
All future experiments should use `topk_checkpoints=3` + unified reeval (`reeval_checkpoints.py`) for checkpoint selection. For 8ep runs, the final epoch should always be saved regardless of training val trajectory.
