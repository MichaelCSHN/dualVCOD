# Development Protocol: Experiment Governance

Phase 1–2 used milestone-driven development (M1–M4). We are now in an
experiment-governance phase. The protocol below replaces the old milestone
framework.

## 1. Experiment Proposal Template

Before writing code or launching training, every experiment must answer:

```
Experiment ID:    E-NN
Target Error:     Which error type(s) from the taxonomy (see docs/04)?
Hypothesis:       Why should this change improve the target error?
Baseline:         Which canonical experiment (E-40 / E-52) is the reference?
Single Variable:  What exactly changes? (One thing only.)
Success Criteria: Quantified metric thresholds (e.g., IoU_tiny >= 0.55).
No-Go Criteria:   What would cause this direction to be CLOSED?
                  (e.g., pf_mIoU drops >0.02, any size bin drops >0.05.)
Inference Impact: Does this change params, FPS, or inference code path?
Rollback Plan:    How to revert if the experiment fails.
```

## 2. Unified Reeval — Mandatory

**Training-log `best_val_mIoU` is NOT sufficient for conclusions.**

All key conclusions MUST use the unified reeval protocol:
- Split: `np.random.RandomState(42)` on MoCA videos, 20% val.
- Metrics: `compute_per_frame_metrics` from `eval/eval_video_bbox.py`.
- Comparison: always against canonical baselines evaluated under the
  identical protocol.

Training-log metrics are for monitoring only.

## 3. Top-K Checkpoint Unified Reeval — Standard Procedure

For any experiment that will be cited as a result:

1. Set `topk_checkpoints: 3` in the trial config.
2. Train normally. The runner saves `checkpoint_rank1.pth` through
   `checkpoint_rank3.pth` (top 3 by training val_mIoU).
3. After training, run:
   ```bash
   python tools/autoresearch/reeval_checkpoints.py \
     --trial-dir local_runs/autoresearch/expl_NN_xxx \
     --backbone efficientnet_b0 --head-type dense_fg_aux
   ```
4. The script outputs a ranking comparison and saves `topk_reeval.json`.
5. Report whether the best training checkpoint equals the best unified
   reeval checkpoint. If they differ, the unified reeval best is the
   canonical result for that experiment.

## 4. Required Metrics per Experiment

Every experiment report must include:

| Category | Metrics |
|----------|---------|
| Core | per-frame mIoU (pf_mIoU), bad_frame_rate, R@0.5 |
| Size breakdown | IoU_tiny, IoU_small, IoU_medium, IoU_large |
| Spatial accuracy | area_ratio (mean, median), center_error (mean, median) |
| Error counts | n_pred_too_large, n_pred_too_small, n_center_shift, n_scale_mismatch |
| Hard videos | Per-video mIoU for flatfish_2, pygmy_seahorse_0, white_tailed_ptarmigan |
| Efficiency | FPS (on RTX 4090), parameter count |

## 5. Report Naming and Format

- Report files: `reports/report_yyyymmddhhmm.md` (UTC+8 timestamp).
- Go/no-go decisions: `reports/error_analysis/E{NN}_go_nogo_yyyymmdd.md`.
- All reports in UTF-8 encoding, GitHub-flavored Markdown.
- Every report must link to its corresponding trial config and metrics.

## 6. Commit Discipline

- Commit when a logical unit is complete: a verified experiment, a
  documentation overhaul, a bug fix with test.
- Do NOT commit partial implementations.
- Do NOT commit local_runs outputs except JSON metadata.
- Do NOT commit checkpoints.
- Config files in `configs/autoresearch/` should be committed alongside
  their corresponding code changes.

## 7. CTO Self-Check (Before Every Action)

1. Does this change keep inference BBox-only?
2. Does this change fit on a single RTX 4090?
3. Is this a single-variable change with a specific error-type hypothesis?
4. Will the result be evaluated under unified reeval?
5. Is there a rollback plan if it fails?

If any answer is No, reconsider the approach.
