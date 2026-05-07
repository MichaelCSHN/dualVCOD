# Archive of Invalidated Phase 2.1 Results

**Date archived**: 2026-05-07

**Reason**: All reports in this directory were generated using the **leaked** Phase 2.1 checkpoint (`best_greenvcod_box_miou.pth`, Epoch 29, claimed mIoU=0.8705).

## Why These Results Are Invalid

On 2026-05-07, an external Red-Team Audit (`../report_20260507_redteam.md`) discovered a **critical data leakage bug** in `tools/train.py`:

- The `ConcatDataset` used for training included the **FULL MoCA dataset** (141 videos)
- 28 MoCA validation videos (1,188 temporal windows) were **not excluded** from training
- The `split_by_video()` function produced `train_idx` but it was **never used** to filter the training data
- Therefore, the claimed mIoU=0.8705 reflects **overfitting to seen data**, not out-of-sample generalization

## The Fix

The bug was fixed on 2026-05-07:
- `split_by_video()` now runs **before** `ConcatDataset` construction
- MoCA is wrapped with `Subset(ds, train_idx)` to exclude all 28 val videos
- Verified by `tools/verify_leak_fix.py` — zero MoCA intra-dataset overlap

## What These Reports Are Good For

- **Historical record** of the audit process
- **Methodological reference** — the audit framework and visualization code remain valid
- **Cautionary tale** — demonstrates the importance of intra-dataset leakage checks

## What These Reports Must NOT Be Used For

- **Publication** — the mIoU=0.8705 figure must never be cited as a valid result
- **Model comparison** — do not compare against these numbers
- **Performance claims** — all quantitative claims are invalidated

## Next Steps

The model must be **retrained from scratch** using the fixed `tools/train.py`.
The red-team audit framework (`tools/red_team_audit.py`) should be re-run on the
cleanly trained checkpoint to establish honest baseline metrics.

---

See `../report_20260507_redteam.md` for the full audit findings and fix details.
