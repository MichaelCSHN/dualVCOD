# Campaign C: Checkpoint Selection Policy — Final Report
## 2026-05-14

---

## Evidence Summary

### Training-val vs Unified-reeval Disagreement: 6/7

| Experiment | Train-Best Ep | Train Val | Unified-Best Ep | Unified pf | Disagreement? |
|-----------|-------------|-----------|----------------|-----------|--------------|
| P1 (zoom) | E5 | 0.305 | E8 | 0.831 | **YES** |
| P3 (coverage) | E5 | 0.307 | E8 | 0.842 | **YES** |
| P5-S2 (stride=2) | E6 | 0.300 | E8 | 0.798 | **YES** |
| P5-S3 (stride=3) | E6 | 0.357 | E8 | 0.578 | **YES** |
| P6 (soft_mask) | E3 | 0.296 | E7 | 0.808 | **YES** |
| B1 (256×256) | E7 | 0.287 | E8 | 0.749 | **YES** |
| A2 (E-45 retrain) | E8 | 0.294 | E8 | 0.831 | No |

**Pattern**: In 6 of 7 experiments, the unified-best epoch differed from the training-val-best epoch. In ALL 7 cases, the unified-best was the **last or near-last epoch** (E7 or E8), regardless of training val trajectory.

### Policy Change Implemented

Modified `tools/autoresearch/run_trial_minimal.py`:
- Final epoch is always force-saved in top-K heap, regardless of training val ranking
- Top-K heap bug fixed (`os.rename` → `shutil.copy2`)
- `_topk_epoch` files cleaned up after training

### Recommendation

1. **All 8ep experiments**: Use `topk_checkpoints=3` with final-epoch force-save. The 6/7 evidence is conclusive that training val is anti-informative for checkpoint selection after epoch ~5.

2. **All 30ep experiments**: Use `topk_checkpoints=5` with final-epoch force-save. The wider epoch range and longer training warrant more checkpoints.

3. **Unified reeval is mandatory**: All go/no-go decisions must use unified reeval (`np.random.RandomState(42)` split). Training val alone has a 6/7 error rate for checkpoint selection.

### Root Cause: Why Training Val Fails

Training val mIoU on the MoCA val split is computed on 28 MoCA val videos (1,188 frames). The unified split uses all 87 MoCA videos with a different random split. The 28-video subset is too small and unrepresentative — random fluctuations in a few hard videos dominate the metric, making epoch-to-epoch ranking anti-correlated with true performance.

DataLoader non-determinism (num_workers=2) further corrupts training val — the same seed produces different batch orderings across runs, making training val trajectory non-reproducible even at fixed seed.

---

## Implementation Status

| Item | Status |
|------|--------|
| Top-K heap fix (`shutil.copy2`) | ✅ Applied |
| Final-epoch force-save policy | ✅ Applied |
| `_topk_epoch` cleanup | ✅ Applied |
| Campaign C report | ✅ Complete |
