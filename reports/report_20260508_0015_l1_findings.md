# Phase 1.6-R — L1 Probe Results & Findings
## 2026-05-08 00:15

### Key Finding: GIoU + Center Loss + Warmup is Breakthrough

**E-12 (GIoU+center+warmup, 5 epochs): mIoU=0.2773, R@0.5=0.2189**
- Within 0.009 of the 30-epoch baseline (0.2861)
- R@0.5 already *exceeds* baseline (0.2189 > 0.1978)
- Convergence pattern: dip at epoch 2, recovery epoch 3-4, strong epoch 5
- This is the ONLY 5-epoch trial to approach baseline performance

### Completed L1 Results

| Trial | Variant | Epochs | Best mIoU | Best R@0.5 | vs Baseline |
|-------|---------|--------|-----------|------------|-------------|
| B0 | MV3-Small T=5 30ep | 30 | 0.2861 | 0.1978 | — |
| **E-12** | **GIoU+center+warmup** | **5** | **0.2773** | **0.2189** | **-0.009 / +0.021** |
| E-03 | EffB0 lr=1e-4 | 1 | 0.2477 | 0.1525 | -0.038 |
| E-01 | Low LR warmup | 1* | 0.2488 | 0.1539 | -0.037 |
| E-04 | T=1 image-only | 5 | 0.2463 | 0.1624 | -0.040 |
| E-02 | MV3-Large | 1* | 0.2456 | 0.1758 | -0.041 |
| E-05 | T=3 | 5 | 0.2449 | 0.1262 | -0.041 |

*E-01, E-02 partial — system OOM during concurrent runs, re-running sequentially

### Failed (Infrastructure)

- E-01, E-02, E-07: OpenCV OOM from 4 concurrent trainings (fixed — re-running sequentially)
- E-07 initial: BCE + autocast incompatibility (fixed — Sigmoid removed, BCEWithLogits)

### New Implementations Verified

- temporal_stride plumbing (dataset + run_trial.py)
- Diagnostic T=5 val index fix (rebuild from canonical_ids, not reuse val_idx)
- ObjectnessHead: Sigmoid→logits for autocast compatibility

### Next: Re-run E-01, E-02, E-07 sequentially, then E-06/E-08/E-09
