# README Update — 2026-05-14 11:15

## Change Summary

Replaced stale README.md (dated ~2026-05-07) with current dualVCOD project
status reflecting all Phase 2 findings through E-53a closure.

## What Changed

### Replaced
- Vague status line ("Project under clean retraining")
- Training section pointing only to legacy `tools/train.py`
- Standalone "Audit" section with duplicated commands
- "Disclaimer" section (merged into Data Leakage Notice + Repository Hygiene)

### Added
1. **Canonical baseline** — E-40 table (MicroVCOD, hard dense_fg_aux, MV3-Small, T=5, 224, pf_mIoU 0.8564)
2. **Stronger variant** — E-51 (EffB0 + dense_fg_aux, 8ep pf_mIoU 0.8372) with cost caveat
3. **E-53a closure** — comparison table vs E-51, direction closed, code retained behind gate
4. **Closed directions** — full enumerated list (10 items)
5. **Evaluation Protocol** — unified reeval (RandomState(42)), required metrics checklist, `reeval_2x2_backbone_epochs.py` usage
6. **Repository Hygiene** — explicit list of what is/isn't committed
7. **Quick Smoke Test** — `scripts/smoke_test.py` + red-team audit
8. **Goal statement** — lighter, faster, bbox-only inference, dense aux in training allowed

### Preserved
- Data leakage notice (mIoU=0.8705 invalidated)
- Reference to `reports/report_20260507_redteam.md`
- MIT License
- Method description (MicroVCOD, independent implementation)
- Repository structure (updated with current directories)

## Verification

- `git diff README.md` — clean, no unexpected changes
- `git status` — only README.md modified; all untracked files are local_runs/, configs/, reports/, tools/ new additions — none staged accidentally
- No training code touched
