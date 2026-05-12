# Phase 1.6 Controlled AutoResearch Plan — 2026-05-07 15:57

## 1. Objective

Systematically search for the best architecture and training configuration to improve clean GreenVCOD performance above the current lower-bound baseline (mIoU=0.2861, 1.4M MobileNetV3-Small), while staying within deployable parameter budgets for drone-edge platforms:

- **Light drone mainline**: 5–15M params
- **Compact edge performance line**: 15–35M params

All trials must maintain strict clean data isolation (canonical_video_id filtering, same MoCA val split seed=42). No legacy checkpoint may be used for initialization, distillation, or teacher supervision.

## 2. Fixed Protocol

Every trial shares these invariants — **zero deviation allowed**:

| Parameter | Fixed Value | Rationale |
|---|---|---|
| Train datasets | MoCA (113) + MoCA_Mask filtered (58) + CAD filtered (9) = 126 unique canonical_video_ids | Clean protocol from Phase 1.5 |
| Val dataset | MoCA held-out 28 videos, 1,188 windows | Same split for all trials |
| Split seed | 42 (video-level, ratio 0.2) | Deterministic, auditable |
| canonical_video_id filter | ON (MoCA_Mask + CAD filtered against MoCA val) | Prevents cross-dataset leakage |
| Eval aug | OFF | Clean re-eval protocol |
| Eval T | 5 (regardless of training T) | Consistent temporal context at eval |
| Eval precision | FP32 (authoritative), AMP evaluated for comparison | Identical to Phase 1.5 |
| Optimizer | AdamW, weight_decay=1e-4 | Standard for all trials |
| Scheduler | CosineAnnealingLR, T_max=epochs | Standard for all trials |
| AMP | fp16 autocast (training only) | Standard for all trials |
| Grad clip | max_norm=2.0 | Standard for all trials |
| Backbone init | ImageNet pretrained (torchvision weights) | Clean init — no VCOD checkpoint loaded |
| Seed for split | 42 (data) | Identical data split across all trials |
| Train seed | varies per trial (recorded in metadata) | Allows multi-seed confirmation |
| No commit | True | Enforced |
| No push | True | Enforced |
| Checkpoints | local_runs/autoresearch/<trial_id>/ only | Never in git |

## 3. Allowed Variables (Search Dimensions)

| Dimension | Type | Values | Description |
|---|---|---|---|
| `backbone` | categorical | `mobilenet_v3_large`, `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2` | torchvision backbone for FPN spatial encoder |
| `input_size` | categorical | `512`, `640` | Frame resize target (H=W, divisible by 32) |
| `temporal_T` | categorical | `3`, `5` | Number of frames per training clip window |
| `sampler` | categorical | `window_uniform`, `video_balanced` | Window drawing strategy |
| `lr` | categorical | `3e-4`, `1e-3` | AdamW initial learning rate |
| `head` | categorical | `current_direct_bbox`, `objectness_aux_head` | Detection head architecture |

Total combinations: 4 × 2 × 2 × 2 × 2 × 2 = **128** — too large for full grid search. We use Latin Hypercube sampling and phased promotion.

## 4. Forbidden Variables

These must **never** change across any trial:

- Val dataset composition (28 MoCA videos, 1,188 windows)
- Train/val split (seed=42 locked)
- canonical_video_id filter (always ON)
- Eval protocol (FP32, aug=OFF, T=5)
- Backbone init source (always ImageNet, never VCOD checkpoint)
- Test-time augmentation (always OFF)
- `verified_candidate_baseline.pth` — must NOT be loaded for any purpose (init, teacher, distillation, comparison baseline)
- Any pre-leak-fix checkpoint

## 5. Scoring Function

**Composite score** (higher = better):

```
composite = val_miou_fp32 + 0.3 × val_recall_at_0_5 + 0.2 × val_recall_at_0_3
```

Tiebreaker order:
1. Fewer total params (prefer smaller)
2. Higher inference FPS (prefer faster)

Rationale: mIoU dominates the score, but R@0.5 and R@0.3 provide gradient for recall-oriented improvements. The 0.3/0.2 weights ensure a recall improvement of +0.05 adds ~0.015-0.025 to the composite — meaningful but not dominant.

## 6. Hard Reject Rules

A trial is permanently rejected if **any** of these fire:

| # | Rule | Condition | Rationale |
|---|---|---|---|
| H1 | mIoU floor | val_mIoU < 0.15 after ≥5 epochs | Insufficient signal — backbone/setup fundamentally broken |
| H2 | OOM | CUDA OutOfMemoryError | Config too large for RTX 4090 (24 GiB) |
| H3 | NaN loss | NaN in training loss at any step | Numerical instability |
| H4 | Diverging loss | loss[epoch_N] > 2 × loss[epoch_1] for N≥5 | Training unstable |
| H5 | Data leak | check_trial_safety.py FAIL | Data isolation violation |
| H6 | Param budget | total_params > 35M | Exceeds edge deployment budget |

Hard-rejected trials:
- Are NOT ranked (rank = null)
- Are NOT eligible for promotion
- Their metadata is preserved for audit
- Their checkpoint is NOT deleted (may be useful for debugging)

## 7. Promotion Rules

| Transition | Condition |
|---|---|
| B → C pool | Pass hard-reject (all smoke trials that survive) |
| C → D (top 8) | Rank ≤ 8 in Phase 1.6-C AND pass hard-reject |
| D → E (top 3) | Rank ≤ 3 in Phase 1.6-D AND pass hard-reject AND val_mIoU > baseline_mIoU + 0.01 |
| E → F (top 2) | Rank ≤ 2 in Phase 1.6-E AND pass hard-reject AND val_mIoU_std_last_5_epochs < 0.02 |

The `val_mIoU > baseline + 0.01` gate at D→E ensures we don't promote models that are merely "least bad." The std stability gate at E→F ensures promoted candidates have converged.

## 8. Trial Lifecycle

Each trial follows this exact sequence:

```
1. check_trial_safety.py  ── pre-flight validation (data isolation, params, seeds)
   │  FAIL → trial blocked, metadata written with reason
   ▼
2. run_trial.py           ── training + eval + metadata recording
   │  OOM/NaN/exception → metadata written with failure reason
   ▼
3. score_trials.py         ── composite scoring + hard-reject check
   │
   ▼
4. aggregate_trials.py     ── cross-trial analysis + promotion recommendations
```

### Per-trial output structure

```
local_runs/autoresearch/<trial_id>/
  config.json           — frozen trial config snapshot
  trial.log             — redirected stdout/stderr (TODO: implement file logging)
  metrics.json          — per-epoch metrics array
  final_metrics.json    — clean re-eval metrics
  checkpoint_best.pth   — best model weights (LOCAL ONLY, not in git)
  metadata.json         — hypothesis, params, FPS, GPU mem, timing, status
  safety_check.json     — pre-flight check results
```

## 9. Trial Schedule & GPU Budget

### Phase 1.6-B: Smoke Trials (4 trials, 3 epochs each)

Maximally-diverse hand-picked configs to probe search space boundaries.

| ID | Backbone | Sz | T | Sampler | Head | LR | Hypothesis |
|---|---|---|---|---|---|---|---|
| smoke_01 | efficientnet_b0 | 512 | 3 | window_uniform | current_direct_bbox | 1e-3 | EN-B0 at 512 with T=3 tests backbone upgrade at reduced temporal context |
| smoke_02 | mobilenet_v3_large | 640 | 5 | video_balanced | objectness_aux_head | 3e-4 | MV3-Large at 640 with balanced sampling + aux head tests high-res + balanced data |
| smoke_03 | efficientnet_b1 | 512 | 5 | window_uniform | current_direct_bbox | 3e-4 | EN-B1 with current head tests stronger backbone at moderate res |
| smoke_04 | efficientnet_b2 | 640 | 3 | video_balanced | current_direct_bbox | 1e-3 | EN-B2 at 640 with T=3 tests max-capacity backbone at reduced temporal context |

**Estimated GPU time**: 4 × 3 epochs × ~8min/epoch ≈ 1.6 hours

The smoke trials test:
- All 4 backbone variants (covers backbone dimension)
- Both input sizes (512, 640)
- Both T values (3, 5)
- Both samplers
- Both head types
- Both LR values

This is a full-coverage probe of the 6-dimensional space in 4 trials.

### Phase 1.6-C: Exploration (24 trials, 5 epochs each)

Latin Hypercube sampling across 6 dimensions. The LHS design ensures each dimension value appears roughly equally often, while covering the space more evenly than random sampling.

**Sampling method**: `scipy.stats.qmc.LatinHypercube` (or manual stratified random if scipy unavailable).

**Estimated GPU time**: 24 × 5 epochs × ~8min/epoch ≈ 16 hours

### Phase 1.6-D: Top-8 Promotion (8 trials, 12 epochs)

Top 8 by composite score from Phase 1.6-C. Extended training reveals whether early convergence advantages persist.

**Estimated GPU time**: 8 × 12 epochs × ~8min/epoch ≈ 12.8 hours

### Phase 1.6-E: Top-3 Promotion (3 trials, 30 epochs)

Top 3 from Phase 1.6-D. Full 30-epoch training to match Phase 1.5 baseline training budget.

**Estimated GPU time**: 3 × 30 epochs × ~8min/epoch ≈ 12 hours

### Phase 1.6-F: Multi-Seed Confirmation (4 trials, 30 epochs)

Top 2 configs from Phase 1.6-E, retrained with seeds 123 and 456. Estimates variance and confirms results are not seed artifacts.

**Estimated GPU time**: 4 × 30 epochs × ~8min/epoch ≈ 16 hours

### Total GPU Budget

| Phase | Trials | Epochs/trial | Est. time |
|---|---|---|---|
| 1.6-B (Smoke) | 4 | 3 | ~1.6 h |
| 1.6-C (Exploration) | 24 | 5 | ~16 h |
| 1.6-D (Top-8) | 8 | 12 | ~12.8 h |
| 1.6-E (Top-3) | 3 | 30 | ~12 h |
| 1.6-F (Multi-Seed) | 4 | 30 | ~16 h |
| **Total** | **43 trials** | | **~58 hours on RTX 4090** |

Note: per-epoch time is estimated at ~8 min for a ~5-9M param model at 512-640px. Actual times may vary 2× depending on backbone and resolution. With faster models or smaller batch sizes, total could be 30-80 hours.

## 10. Safety Rules (Enforced by Scripts)

1. **No auto commit** — scripts never call `git commit`
2. **No auto push** — scripts never call `git push`
3. **No checkpoints in repo** — all checkpoints go to `local_runs/autoresearch/<trial_id>/`
4. **Pre-flight safety check** — `check_trial_safety.py` runs before every trial
5. **Failed trials preserved** — metadata and partial outputs kept for debugging
6. **Data isolation verified** — canonical_video_id overlap checked at dataloader build time
7. **No legacy checkpoint loading** — every trial starts from ImageNet pretrained weights
8. **Reports are summaries only** — no per-sample CSVs committed from auto search

## 11. Script Skeleton Status

All scripts are designed and compile-clean. Status notes:

| File | Status | Notes |
|---|---|---|
| `configs/autoresearch/search_space_phase1_6.yaml` | Ready | Complete search space + schedule definition |
| `tools/autoresearch/__init__.py` | Ready | Package init |
| `tools/autoresearch/backbone_registry.py` | Ready | 5 backbones registered, channel probe verified (mobilenet_v3_small: [24,40,576]) |
| `tools/autoresearch/check_trial_safety.py` | Ready | 10 checks, returns PASS/FAIL with detailed reasons |
| `tools/autoresearch/run_trial.py` | Skeleton | Core training loop complete. **Needs before first use**: (1) extend `src/model.py` `SpatialEncoderFPN` to accept backbone_name, (2) implement `objectness_aux_head` variant, (3) implement `video_balanced` sampler with WeightedRandomSampler, (4) add file-based logging of stdout/stderr |
| `tools/autoresearch/score_trials.py` | Ready | Composite scoring, hard reject, ranking, CSV+JSON output |
| `tools/autoresearch/aggregate_trials.py` | Ready | Dimension trends, Pareto frontier, promotion recommendations, Markdown report |

### Required model.py changes before first run_trial.py execution

The `SpatialEncoderFPN` in `src/model.py` (line 132) currently hardcodes `mobilenet_v3_small`. To support the search space, it needs:

1. Accept a `backbone_name` parameter
2. Use `backbone_registry.py` to get the factory and stage slices
3. Dynamically set lateral/smooth convolution channel counts based on probed backbone stage channels

Estimated implementation effort: ~50-80 lines of changes to `src/model.py`.

The `objectness_aux_head` requires:
1. A parallel `ObjectnessHead` (AdaptiveAvgPool2d → Linear(128→32) → ReLU → Linear(32→1) → Sigmoid)
2. Modified `BBoxLoss` to accept aux predictions and add BCE loss with weight 0.1
3. GT objectness derived from GT bbox area > 0.001

Estimated implementation effort: ~30-50 lines of changes.

## 12. Readiness Assessment

### Can we enter smoke trial phase now?

**YES, with one prerequisite.** The scripts are structurally complete and compile-clean. Before running the first `smoke_01`, the model backbone parameterization must be implemented in `src/model.py` (backbone_name → SpatialEncoderFPN routing). The existing `MicroVCOD` hardcodes MobileNetV3-Small and will ignore any `backbone` config value.

Recommended order:
1. Implement `SpatialEncoderFPN(backbone_name)` parameterization in `src/model.py`
2. Run smoke_01 (efficientnet_b0, 512px, T=3) — simplest upgrade path
3. If smoke_01 passes, run smoke_02-smoke_04 in parallel or sequence
4. Review smoke results, fix any issues
5. Generate LHS design and launch Phase 1.6-C

### What's ready right now

- Search space fully defined (6 dimensions, 128 total combos)
- Backbone registry functional (5 backbones, verified channel probe)
- Safety checker functional (10 checks)
- Scoring and aggregation functional
- Trial lifecycle fully specified
- Phase schedule with promotion gates defined
- All scripts compile-clean (verified)

### What needs implementation before execution

| Item | Location | Effort |
|---|---|---|
| Backbone-swappable SpatialEncoderFPN | `src/model.py` | Medium |
| objectness_aux_head | `src/model.py` + `src/loss.py` | Small |
| video_balanced sampler | `tools/autoresearch/run_trial.py` | Small |
| File-based logging (trial.log) | `tools/autoresearch/run_trial.py` | Trivial |
| Latin Hypercube sampling | `tools/autoresearch/run_trial.py` or separate script | Small |

## 13. First 4 Smoke Trials Recommendation

```
smoke_01: efficientnet_b0, sz=512, T=3, window_uniform, current_direct_bbox, lr=1e-3
smoke_02: mobilenet_v3_large, sz=640, T=5, video_balanced, objectness_aux_head, lr=3e-4
smoke_03: efficientnet_b1, sz=512, T=5, window_uniform, current_direct_bbox, lr=3e-4
smoke_04: efficientnet_b2, sz=640, T=3, video_balanced, current_direct_bbox, lr=1e-3
```

This covers all 6 dimensions at their extremes, providing boundary information before committing to the 24-trial exploration phase. If any smoke trial OOMs or produces NaN, that corner of the space can be excluded from the LHS design.

Expected smoke outcomes:
- All 4 should pass hard-reject (mIoU > 0.15 after 3 epochs)
- Higher-resolution (640px) models should show better spatial localization
- Larger backbones (efficientnet_b1/b2) should show higher capacity
- objectness_aux_head may help with the "no_response" error type (currently 18.9%)
- If efficientnet_b2 at 640 OOMs, cap search space at efficientnet_b1 × 640 or efficientnet_b2 × 512

---
*Report generated at 2026-05-07T15:57:00*
*Phase 1.6 AutoResearch Plan — design only, no training executed*
*Scripts: tools/autoresearch/*, config: configs/autoresearch/search_space_phase1_6.yaml*
