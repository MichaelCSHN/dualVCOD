# Phase 2 Direction Report

**Date**: 2026-05-11
**Scope**: E-33 (50ep main-line), E-34 (MV3-Large probe), NVMe migration, batch size analysis

---

## 1. NVMe Migration Summary

Dataset paths migrated from D:\ (SATA HDD) to C:\ (NVMe SSD). No training speed improvement observed — bottleneck is CPU JPEG decode (~90% of data_time), not disk I/O (~10%). C:\datasets retained as default for path uniformity.

| Metric | HDD (D:\) | NVMe (C:\) | Δ |
|--------|-----------|-----------|----|
| data_time/epoch | 144.2s | 152.8s | +6.0% |
| total epoch | 211.0s | 223.6s | +6.0% |

---

## 2. E-34: MV3-Large 8ep Probe — FAIL

| Metric | E-34 (MV3-Large) | E-28 (MV3-Small) | Δ |
|--------|------------------|-------------------|-----|
| best_val_mIoU | 0.2593 | 0.2893 | **-0.0300** |
| total_params | 3,505,780 | 1,411,684 | +148% |
| GPU memory | 6.35 GiB | 2.30 GiB | +176% |
| train_time (8ep) | 1653s | 1616s | +2.3% |

**Conclusion**: MV3-Large is strictly worse on this dataset (~9k samples). Larger backbone overfits more severely. Do NOT promote to 30ep. Stay with MV3-Small.

---

## 3. E-33: 50ep Main-Line Ceiling Test — CRITICAL

### 3.1 Raw Results

```
Config: DIoU + center=0.0 + MV3-Small + T=5 + 224 + bs=48 + nw=4 + 50ep
Best epoch:  14, val mIoU = 0.2874, val R@0.5 = 0.1785
Final epoch: 50, val mIoU = 0.2668, val R@0.5 = 0.1788
Total time:  10,003s (2.78h), avg 199.7s/epoch
```

### 3.2 Epoch-by-Epoch Val mIoU Trajectory

```
Epoch  1: 0.2461    Epoch 14: 0.2874 ← PEAK
Epoch  3: 0.2589    Epoch 20: 0.2794
Epoch  5: 0.2146    Epoch 25: 0.2846
Epoch  7: 0.2695    Epoch 30: 0.2689
Epoch 10: 0.2614    Epoch 40: 0.2708
Epoch 12: 0.2615    Epoch 50: 0.2668
```

Peak at epoch 14 (0.2874), then persistent degradation. Train mIoU reached 0.86 at epoch 50 — severe overfitting. The val mIoU never exceeded the 8ep baseline from E-28 (0.2893 with bs=16).

### 3.3 Comparison with bs=16 Baselines

| Trial | Epochs | bs | nw | Best mIoU | Train Time |
|-------|--------|----|----|-----------|------------|
| E-28 | 8 | 16 | 2 | **0.2893** | 1,616s |
| E-31 | 30 | 16 | 2 | **0.3111** | 9,403s |
| E-33 | 50 | 48 | 4 | **0.2874** | 10,003s |

E-33 at 50ep is **worse than E-28 at 8ep**. The only changed variables: bs (48 vs 16) and nw (4 vs 2).

### 3.4 Root Cause: Batch Size Effect

With bs=48, the model sees only 186 batches/epoch across 8,926 samples. At 50 epochs, that's only 9,300 total gradient updates. Compare:

| Metric | bs=16 | bs=48 |
|--------|-------|-------|
| Batches/epoch | 558 | 186 |
| Updates at 30ep | 16,740 | 5,580 |
| Updates at 50ep | 27,900 | 9,300 |
| Gradient noise | Higher (better exploration) | Lower (worse exploration) |
| Peak mIoU | 0.3111 | 0.2874 |

The smaller bs=16 provides 3x more gradient updates at the same epoch count. On a small dataset with a rough loss landscape, more frequent noisier updates are essential for escaping poor local minima. Larger batches converge to smoother but worse minima.

The LR was not scaled with batch size (both used lr=0.001). Linear scaling rule suggests lr=0.003 for bs=48, which might partially compensate but is unlikely to fully close the 0.024 gap.

### 3.5 Implications for Prior Experiments

E-29 (T=7, 8ep, bs=48) and E-32 (CIoU, 8ep, bs=48) were both run with bs=48. Their negative results may be partially attributable to batch size penalty, not purely to the hypothesis variable. However:

- E-29 (0.2877) was compared against E-28 (0.2893, bs=16) — the 0.0016 gap is small enough that bs=48 penalty could explain it. T=7 might be neutral, not negative.
- E-32 (0.2564) was dramatically worse. Even accounting for ~0.01-0.02 bs penalty, CIoU would still be clearly inferior to DIoU.

---

## 4. 30ep → 50ep: No Meaningful Gains

The original question was whether extending training from 30ep to 50ep yields meaningful gains. E-33 cannot directly answer this because the batch size confound invalidates the comparison. However:

- With bs=16: E-31 reached 0.3111 at 30ep. The `miou_std_last_3_epochs` was 0.0009, suggesting stability near the peak.
- E-33 with bs=48 peaked at epoch 14 and then degraded, suggesting that with proper bs, even 30ep may already be near the ceiling for this architecture.

**We need a clean 50ep run with bs=16 to properly answer the 30ep→50ep question.** However, the low `miou_std_last_3_epochs` (0.0009) from E-31 suggests the model had largely converged by 30ep and additional epochs would yield diminishing returns (likely ≤ 0.005 mIoU gain).

---

## 5. Scheduler Recommendation

**Not yet.** Before testing scheduler improvements, we must:

1. **Resolve the batch size question.** All future quality-critical experiments must use bs=16 (or scale LR with bs). The speed-quality tradeoff must be quantified.
2. **Confirm the true DIoU baseline at 8ep and 30ep with bs=16.** E-28 (0.2893) and E-31 (0.3111) are the reference points.

Once bs=16 is confirmed as the canonical config, scheduler testing (cosine warmup, OneCycleLR) becomes the natural next step to squeeze remaining gains from the current architecture. But with `miou_std_last_3_epochs = 0.0009` at 30ep, the ceiling may only be ~0.315-0.318.

---

## 6. Next Recommended Experiment

### Priority 1: Re-run E-33 with bs=16 (clean 50ep ceiling test)

```
Config: DIoU + center=0.0 + MV3-Small + T=5 + 224 + bs=16 + nw=2 + 50ep
Trial:  expl_35_bs16_50ep
Parent: expl_31_diou_30ep
Cost:   ~3.5h (254s/epoch × 50)
```

This is the only way to properly answer: does 30ep→50ep help? Without resolving the bs confound, all other experiments are built on shaky ground.

### Priority 2: If 50ep ceiling confirmed (~0.315), test CosineWarmup

Only if Priority 1 shows that 50ep > 30ep by a meaningful margin (>0.005), test cosine warmup to extend effective training duration.

### Priority 3: bs=32 middle ground

If Priority 1 confirms bs=16 is required for quality, test bs=32 as a speed compromise:
```
Config: DIoU + MV3-Small + T=5 + 224 + bs=32 + nw=4 + 8ep (cheap probe)
Expected: mIoU between bs=16 and bs=48 values
```

---

## 7. Decision Log

| Decision | Verdict | Rationale |
|----------|---------|-----------|
| MV3-Large | **Abandon** | -0.030 mIoU, +148% params, +176% GPU memory |
| bs=48 for quality runs | **Reject** | -0.024 mIoU vs bs=16, invalidates comparisons |
| bs=48 for cheap probes | **Use with caution** | Acceptable for rough ranking, but expect ~0.01-0.02 negative bias |
| Default dataset path | **C:\datasets** | No speed gain but path uniformity |
| 50ep with current arch | **Inconclusive** | Need bs=16 re-run |

---

*Report generated: 2026-05-11T11:20*
*Data: E-33 completed 2026-05-11T10:54*
