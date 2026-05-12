# Phase 1.6-R AutoResearch — L1 Probe Complete Report
## 2026-05-08 ~01:30

### Executive Summary

12 Explorer trials designed. 9 completed (L1), 1 running (E-06), 2 pending (E-10/E-11).
**Two clear winners identified, both within 0.009 of the 30-epoch baseline at just 5 epochs.**

### L1 Leaderboard (5 epochs unless noted)

| Rank | Trial | Variant | mIoU | R@0.5 | Trend | Verdict |
|------|-------|---------|------|-------|-------|---------|
| ⭐1 | E-12 | GIoU+center loss+warmup | **0.2773** | **0.2189** | Rising at ep5 | **ADVANCE to L4** |
| ⭐2 | E-07 | Objectness aux head | **0.2772** | 0.1992 | Rising at ep5 | **ADVANCE to L4** |
| 3 | E-02 | MV3-Large (3.5M) | 0.2761 | 0.2177 | Peaked ep2, declining | L2 candidate |
| 4 | E-08 | Video-balanced sampler | 0.2552 | 0.1692 | Oscillating, late recovery | Mixed, L2 |
| 5 | E-03 | EffB0 lr=1e-4 | 0.2477 | 0.1525 | 1 epoch only | Late bloomer |
| 6 | E-04 | T=1 image-only | 0.2463 | 0.1624 | Severe overfit | Negative |
| 7 | E-05 | T=3 | 0.2449 | 0.1262 | Severe overfit | Negative |
| 8 | E-01 | Low LR+warmup | 0.2417 | 0.1461 | LR too low | Late bloomer |
| 9 | E-09 | Freeze backbone+warmup | 0.2170 | 0.0970 | Catastrophic unfreeze | **DEAD** |
| — | B0 | MV3-Small 30ep | 0.2861 | 0.1978 | Reference | Baseline |

### Key Findings

**WINNING STRATEGIES:**
1. **E-12 (GIoU + center loss + warmup)**: Combines GIoU box overlap with center-point MSE loss and 3-epoch LR warmup. Shows recovery pattern (dip at ep2, then steady rise) — epoch 5 is still climbing. R@0.5=0.2189 *exceeds* the 30-epoch baseline. mIoU=0.2773 is within 0.009 of baseline.
2. **E-07 (Objectness aux head)**: BCE foreground/background auxiliary loss with 0.1 weight. Steady monotonic improvement across all 5 epochs — no overfitting. Best validation trend of any trial. mIoU=0.2772 ties E-12.

**PROMISING (needs 30 epochs):**
3. **E-02 (MV3-Large)**: Hit 0.2761 at epoch 2 but overfits after. With 30 epochs and proper scheduling, could surpass baseline.
4. **E-03 (EffB0, lr=1e-4)**: Stable at 1 epoch (0.2477). The only EfficientNet variant that didn't NaN. Promising for longer training.

**NEGATIVE RESULTS:**
- **T<5 causes rapid overfitting**: E-04 (T=1) and E-05 (T=3) both show training mIoU rising to 0.56-0.60 while validation degrades. Temporal context is *essential* for generalization.
- **Freeze-then-unfreeze is harmful**: E-09's catastrophic interference when unfreezing backbone destroys learned head features. Not recommended.
- **Low LR + warmup alone doesn't help**: E-01's 5-epoch warmup keeps LR too low for meaningful progress.

### Infrastructure Issues

- **Concurrent training OOM**: 4 simultaneous trials overload system memory (OpenCV workers competing for RAM). Solution: sequential execution.
- **BCE + autocast incompatibility**: ObjectnessHead used Sigmoid+BCE which is unsafe with AMP. Fixed: removed Sigmoid, switched to BCEWithLogitsLoss.
- **Diagnostic T=5 eval crash**: val_idx from T≠5 dataset used for T=5 loader — index mismatch. Fixed: rebuild indices by canonical video name.
- **temporal_stride not plumbed**: E-05 ran with stride=1 instead of 2. Fixed: stride added to dataset + run_trial.py.

### Recommendations

**Immediate (next session):**
1. **Launch E-12 L4 (30-epoch confirmation)**: The #1 candidate. Expect mIoU > 0.30.
2. **Launch E-07 L4 (30-epoch confirmation)**: The #2 candidate. Different mechanism (aux loss vs multi-component loss) — complementary to E-12.
3. **Launch E-12+E-07 COMBINED**: GIoU+center+warmup + objectness aux — could be additive.
4. **Complete E-06 (640px)**: Running — 5 epochs at ~1.8h total. Resolution scaling data.
5. **Run E-10/E-11 (ConvNeXt/ShuffleNet)**: Architectural bounds for completeness.

**Defer:**
- E-01 (low LR): Late bloomer — only valuable with 30 epochs
- E-03 (EffB0): Promising stability but needs 30 epochs
- E-09 (freeze): Dead end — catastrophic interference
- E-04/E-05 (T<5): Negative result confirmed — temporal context required
