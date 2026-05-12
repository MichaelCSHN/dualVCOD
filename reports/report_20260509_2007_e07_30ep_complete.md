# E-07 30-Epoch Confirmation Trial — Final Report

**Trial ID**: `expl_07_30ep_retry_timeout`
**Parent trial**: `expl_07_30ep` (original hung)
**Status**: **COMPLETED — Hypothesis Partially Confirmed**
**Completed**: 2026-05-09 20:06

---

## Summary

| Metric | 5ep Baseline | E-07 30ep Best | E-07 30ep Final | E-12 30ep Best | Δ E-07 vs E-12 |
|--------|-------------|----------------|-----------------|----------------|----------------|
| **Val mIoU** | 0.2773 | **0.3012** (ep11) | 0.2895 | 0.3056 (ep22) | -0.0044 |
| **Val R@0.5** | 0.2189 | **0.2559** (ep12) | 0.1976 | 0.2554 (ep14) | +0.0005 |
| **Val R@0.3** | — | — | 0.4736 | 0.465 | +0.009 |
| **Empty pred rate** | — | — | 0.0% | 0.0% | tie |
| **Stability (mIoU std last 3)** | — | — | 0.0002 | 0.001 | better |

## Configuration

- Backbone: mobilenet_v3_small, Input: 224px, T=5
- Head: **objectness_aux_head** (BCE foreground/background supervision)
- Loss: SmoothL1=1.0, GIoU=1.0, Objectness=0.1
- No warmup, CosineAnnealingLR (30 epochs)
- Batch: 16, num_workers: 2, LR: 0.001
- Anti-hang patches: DataLoader timeout=120s, heartbeat every 10 batches, cv2.setNumThreads(0)

## Full Epoch Log

| Ep | Tr Loss | Tr mIoU | Val mIoU | Val R@0.5 | LR |
|----|---------|---------|----------|-----------|---|
| 1 | 0.8156 | 0.3287 | 0.2614 | 0.2089 | 0.00100 |
| 2 | 0.6361 | 0.4531 | 0.2844 | 0.2249 | 0.00099 |
| 3 | 0.5240 | 0.5393 | 0.2567 | 0.1382 | 0.00098 |
| 4 | 0.4730 | 0.5792 | 0.2785 | 0.1749 | 0.00096 |
| 5 | 0.4366 | 0.6098 | 0.2375 | 0.1448 | 0.00093 |
| 6 | 0.4090 | 0.6325 | 0.2815 | 0.1852 | 0.00090 |
| 7 | 0.3917 | 0.6477 | 0.2559 | 0.1512 | 0.00087 |
| 8 | 0.3776 | 0.6601 | 0.2845 | 0.2042 | 0.00084 |
| 9 | 0.3572 | 0.6781 | 0.2722 | 0.1525 | 0.00079 |
| 10 | 0.3445 | 0.6888 | 0.2871 | 0.1803 | 0.00075 |
| 11 | 0.3339 | 0.6984 | **0.3012** | 0.2269 | 0.00070 |
| 12 | 0.3203 | 0.7094 | 0.2952 | **0.2559** | 0.00066 |
| 13 | 0.3110 | 0.7180 | 0.2701 | 0.1556 | 0.00060 |
| 14 | 0.3031 | 0.7250 | 0.2855 | 0.1966 | 0.00055 |
| 15 | 0.2888 | 0.7377 | 0.2841 | 0.1830 | 0.00050 |
| 16 | 0.2791 | 0.7466 | 0.2754 | 0.1904 | 0.00045 |
| 17 | 0.2714 | 0.7534 | 0.2858 | 0.1816 | 0.00040 |
| 18 | 0.2596 | 0.7643 | 0.2769 | 0.1864 | 0.00034 |
| 19 | 0.2511 | 0.7720 | 0.2812 | 0.1998 | 0.00030 |
| 20 | 0.2413 | 0.7805 | 0.2806 | 0.1854 | 0.00025 |
| 21 | 0.2365 | 0.7849 | 0.2889 | 0.2086 | 0.00021 |
| 22 | 0.2293 | 0.7915 | 0.2861 | 0.1852 | 0.00017 |
| 23 | 0.2225 | 0.7982 | 0.2882 | 0.1970 | 0.00013 |
| 24 | 0.2168 | 0.8033 | 0.2896 | 0.1993 | 0.00010 |
| 25 | 0.2106 | 0.8095 | 0.2917 | 0.2027 | 0.00007 |
| 26 | 0.2064 | 0.8134 | 0.2872 | 0.1919 | 0.00004 |
| 27 | 0.2029 | 0.8169 | 0.2878 | 0.1955 | 0.00002 |
| 28 | 0.2005 | 0.8188 | 0.2889 | 0.1997 | 0.00001 |
| 29 | 0.1980 | 0.8212 | 0.2892 | 0.1973 | 0.00000 |
| 30 | 0.1965 | 0.8226 | 0.2895 | 0.1976 | 0.00000 |

**Best mIoU**: 0.3012 (epoch 11) — earliest 0.30 break of any trial
**Best R@0.5**: 0.2559 (epoch 12)

## Engineering Metrics

| Metric | E-07 | E-12 |
|--------|------|------|
| Total params | 1,415,845 | 1,411,684 |
| GPU memory | 0.19 GiB | 0.19 GiB |
| Inference FPS | 83.8 | 80.8 |
| Total train time | 9,575s (2.66h) | 9,482s (2.63h) |
| Mean epoch time | ~319s | ~316s |
| Empty pred rate | 0.0% | 0.0% |
| Area ratio (global) | 0.8331 | 0.8418 |
| Area ratio (per-sample) | 7.0615 | 5.0508 |
| Stability (mIoU std, last 3) | 0.0002 | 0.001 |

## E-07 vs E-12 Head-to-Head

| Dimension | E-07 (Objectness) | E-12 (Center Loss) | Winner |
|-----------|-------------------|---------------------|--------|
| Convergence speed | 0.30 at ep11 | 0.30 at ep14 | E-07 |
| Peak mIoU | 0.3012 | 0.3056 | E-12 |
| Peak R@0.5 | 0.2559 | 0.2554 | tie |
| Final mIoU | 0.2895 | 0.3043 | E-12 |
| Late consolidation | No | Yes (ep22 peak) | E-12 |
| Oscillation | High (0.24-0.30) | Moderate (0.25-0.31) | E-12 |
| R@0.3 | 0.4736 | 0.465 | E-07 |
| Area coverage | 7.06x | 5.05x | E-07 (wider) |

## Key Findings

1. **Hypothesis partially confirmed**: Objectness head breaks 0.30 earlier than center loss (ep11 vs ep14), but cannot sustain it — unlike E-12 which consolidated to 0.304 final
2. **Two-phase behavior**: Rapid rise epochs 1-12 (peaking at ep11-12), then abrupt drop to 0.27 plateau with slow drift back to 0.29
3. **No late consolidation**: Unlike E-12 (best at ep22), E-07's best was at ep11 with no later recovery — the objectness signal appears less stable than center loss for long runs
4. **0.30 barrier broken**: Confirms E-07 can reach 0.30+ territory, just can't hold it — the mechanism works but needs stabilization
5. **Zero empty rate maintained**: Objectness head at 0.1 weight is sufficient to prevent prediction collapse
6. **Wider bbox coverage**: Mean per-sample area ratio 7.06 vs 5.05 for E-12 — objectness head produces more spread-out bounding boxes
7. **Anti-hang patches 100% successful**: No infra issues across all 30 epochs

## Promotion Decision

**NO PROMOTE** — Objectness aux head alone is insufficient. Key failure mode: early peak (ep11) followed by collapse to plateau 0.27-0.29, no late consolidation. Center loss (E-12) is strictly superior for final solution quality and stability.

**Recommendation**: The combined E-12+E-07 probe (center loss + objectness head together) is the logical next step — E-07 provides fast early convergence, E-12 provides late-stage stability. The combination may be complementary.

## Artifacts

- `local_runs/autoresearch/expl_07_30ep_retry_timeout/checkpoint_best.pth`
- `local_runs/autoresearch/expl_07_30ep_retry_timeout/metrics.json`
- `local_runs/autoresearch/expl_07_30ep_retry_timeout/metadata.json`
- `local_runs/autoresearch/expl_07_30ep_retry_timeout/trial.log`

No commit, no push.
